"""Compact, checksummed epoch orders and constant-time CPT prefix accounting.

Each epoch gets a fresh permutation from a dedicated CPU torch.Generator. Its
first permutation is identical to the historical single-epoch trainer. Orders
live in int32/int64 .npy files; checkpoints carry copies and small metadata,
never a millions-element Python list. LR steps never cross an epoch boundary.
"""
import hashlib
import math
from pathlib import Path
import shutil

import numpy as np
import torch


ORDINARY_GENERATOR = "torch.randperm(cpu), sequential seeded generator"
EXTENSION_GENERATOR = "torch.randperm(cpu), preserved completed prefix then uniform complement"


def indices_sha256(values):
    """Canonical shuffle-prefix digest, independent of the stored integer width."""
    return hashlib.sha256(memoryview(np.ascontiguousarray(values, dtype=np.int64)).cast("B")).hexdigest()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_permutation(order, blocks):
    values = np.asarray(order)
    if values.ndim != 1 or len(values) != blocks or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Saved shuffle order has invalid shape, length or dtype")
    # O(n log n) once per load, O(n) bounded NumPy memory, no Python integers.
    if not np.array_equal(np.sort(values), np.arange(blocks, dtype=values.dtype)):
        raise ValueError("Saved shuffle order is not an exact dataset permutation")


class EpochOrders:
    def __init__(self, directory, metadata, validate=True):
        self.directory = Path(directory)
        self.metadata = metadata
        if metadata.get("schema_version") != 1 or metadata.get("generator") not in (ORDINARY_GENERATOR, EXTENSION_GENERATOR):
            raise ValueError("Unsupported compact shuffle metadata")
        for key in ("epochs", "dataset_blocks", "effective_batch"):
            value = metadata.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Invalid shuffle {key}")
        self.blocks = metadata["dataset_blocks"]
        self.batch = metadata["effective_batch"]
        self.epochs = metadata["epochs"]
        self.epoch_steps = math.ceil(self.blocks / self.batch)
        files = metadata.get("files")
        if not isinstance(files, list) or len(files) != self.epochs:
            raise ValueError("Shuffle metadata does not cover all epochs")
        self.orders = []
        for epoch, item in enumerate(files):
            if item.get("name") != f"order_epoch_{epoch:05d}.npy":
                raise ValueError("Unexpected epoch order filename")
            path = self.directory / item["name"]
            if sha256(path) != item.get("sha256"):
                raise ValueError(f"Shuffle file checksum mismatch: {path.name}")
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if str(values.dtype) != item.get("dtype") or item.get("blocks") != self.blocks:
                raise ValueError("Shuffle metadata dtype/block count mismatch")
            if validate:
                validate_permutation(values, self.blocks)
            tail_position = item.get("tail_position")
            if (isinstance(tail_position, bool) or not isinstance(tail_position, int)
                    or not 0 <= tail_position < self.blocks or int(values[tail_position]) != self.blocks - 1):
                raise ValueError("Shuffle tail position mismatch")
            self.orders.append(values)
        if metadata["generator"] == EXTENSION_GENERATOR:
            self.validate_extension(validate)

    def validate_extension(self, reconstruct=True):
        extension = self.metadata.get("prefix_extension")
        if not isinstance(extension, dict) or extension.get("schema_version") != 1:
            raise ValueError("Missing prefix-preserving shuffle audit")
        count = extension.get("prefix_blocks")
        completed = extension.get("completed_steps")
        old_blocks = extension.get("old_dataset_blocks")
        if (any(isinstance(x, bool) or not isinstance(x, int) for x in (count, completed, old_blocks))
                or not 0 < count <= old_blocks < self.blocks or count != completed * self.batch
                or self.epochs != 2):
            raise ValueError("Invalid completed shuffle-prefix boundary")
        prefix = self.orders[0][:count]
        if (np.any(prefix < 0) or np.any(prefix >= old_blocks)
                or indices_sha256(prefix) != extension.get("prefix_indices_sha256")):
            raise ValueError("Preserved shuffle prefix differs from its audit")
        if reconstruct:
            # Prove that every unconsumed block was sampled uniformly without
            # replacement, and that epoch two was independently reshuffled.
            remaining = np.ones(self.blocks, dtype=np.bool_)
            remaining[prefix] = False
            complement = np.flatnonzero(remaining)
            generator = torch.Generator().manual_seed(self.metadata["seed"])
            expected = complement[torch.randperm(len(complement), generator=generator).numpy()]
            if not np.array_equal(self.orders[0][count:], expected):
                raise ValueError("Extension remainder is not the signed uniform complement permutation")
            expected_second = torch.randperm(self.blocks, generator=generator).numpy()
            if not np.array_equal(self.orders[1], expected_second):
                raise ValueError("Extension second epoch differs from the signed full permutation")

    @classmethod
    def create(cls, directory, blocks, batch, epochs, seed):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        generator = torch.Generator().manual_seed(seed)
        metadata = {"schema_version": 1, "dataset_blocks": blocks,
                    "effective_batch": batch, "epochs": epochs, "seed": seed,
                    "generator": ORDINARY_GENERATOR, "files": []}
        dtype = np.int32 if blocks <= np.iinfo(np.int32).max else np.int64
        for epoch in range(epochs):
            values = torch.randperm(blocks, generator=generator).numpy().astype(dtype, copy=False)
            path = directory / f"order_epoch_{epoch:05d}.npy"
            np.save(path, values, allow_pickle=False)
            metadata["files"].append({"name": path.name, "sha256": sha256(path),
                                      "dtype": str(values.dtype), "blocks": blocks,
                                      "tail_position": int(np.flatnonzero(values == blocks - 1)[0])})
        return cls(directory, metadata, validate=False)

    @classmethod
    def extend_completed_prefix(cls, directory, previous, completed_steps, blocks, batch, epochs, seed):
        if (not isinstance(previous, cls) or previous.epochs != 1 or previous.batch != batch
                or previous.metadata["seed"] != seed or epochs != 2 or blocks <= previous.blocks
                or not 0 < completed_steps <= previous.epoch_steps):
            raise ValueError("Unsupported completed single-epoch order extension")
        prefix_blocks = completed_steps * batch
        if prefix_blocks > previous.blocks:
            raise ValueError("A partial last batch cannot be embedded in an extended epoch")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        dtype = np.int32 if blocks <= np.iinfo(np.int32).max else np.int64
        prefix = np.asarray(previous.orders[0][:prefix_blocks], dtype=dtype)
        remaining = np.ones(blocks, dtype=np.bool_)
        remaining[prefix] = False
        complement = np.flatnonzero(remaining)
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(len(complement), generator=generator).numpy()
        values = np.concatenate((prefix, complement[permutation])).astype(dtype, copy=False)
        metadata = {"schema_version": 1, "dataset_blocks": blocks, "effective_batch": batch,
                    "epochs": epochs, "seed": seed, "generator": EXTENSION_GENERATOR,
                    "prefix_extension": {"schema_version": 1, "completed_steps": completed_steps,
                                         "prefix_blocks": prefix_blocks, "old_dataset_blocks": previous.blocks,
                                         "source_order_sha256": previous.metadata["files"][0]["sha256"],
                                         "prefix_indices_sha256": indices_sha256(prefix)}, "files": []}
        for epoch in range(epochs):
            if epoch:
                values = torch.randperm(blocks, generator=generator).numpy().astype(dtype, copy=False)
            path = directory / f"order_epoch_{epoch:05d}.npy"
            np.save(path, values, allow_pickle=False)
            metadata["files"].append({"name": path.name, "sha256": sha256(path),
                                      "dtype": str(values.dtype), "blocks": blocks,
                                      "tail_position": int(np.flatnonzero(values == blocks - 1)[0])})
        return cls(directory, metadata, validate=False)

    def copy_to(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        for item in self.metadata["files"]:
            shutil.copyfile(self.directory / item["name"], directory / item["name"])

    def indices(self, zero_based_step):
        if not 0 <= zero_based_step < self.epochs * self.epoch_steps:
            raise ValueError("Step lies outside the saved epoch orders")
        epoch, local_step = divmod(zero_based_step, self.epoch_steps)
        start = local_step * self.batch
        return self.orders[epoch][start:start + self.batch]

    def prefix_counts(self, completed_steps, sequence_length, token_count):
        if not 0 <= completed_steps <= self.epochs * self.epoch_steps:
            raise ValueError("Completed steps exceed the saved epoch orders")
        remainder = token_count % sequence_length
        inputs = targets = applied = 0
        remaining = completed_steps
        for item in self.metadata["files"]:
            steps = min(remaining, self.epoch_steps)
            selected = min(self.blocks, steps * self.batch)
            epoch_inputs = selected * sequence_length
            if remainder and item["tail_position"] < selected:
                epoch_inputs -= sequence_length - remainder
            inputs += epoch_inputs
            targets += epoch_inputs - selected
            applied += steps
            # With sequence_length>=2, only a singleton one-token tail can
            # form an update without any supervised target.
            tail_step = item["tail_position"] // self.batch
            singleton_tail = self.batch == 1 or (self.blocks % self.batch == 1
                                                and item["tail_position"] == self.blocks - 1)
            if remainder == 1 and singleton_tail and tail_step < steps:
                applied -= 1
            remaining -= steps
            if remaining == 0:
                break
        return inputs, targets, applied


def load_saved_orders(state, checkpoint, validate=True):
    """Load modern compact orders or preserve the historical list format."""
    if "order_metadata" in state:
        return EpochOrders(Path(checkpoint) / "orders", state["order_metadata"], validate=validate)
    order = state.get("order")
    if not isinstance(order, list) or not order or any(isinstance(x, bool) or not isinstance(x, int) for x in order):
        raise ValueError("Missing valid legacy shuffle order")
    if validate:
        validate_permutation(np.asarray(order, dtype=np.int64), len(order))
    return order


def order_indices(orders, zero_based_step, effective_batch):
    if isinstance(orders, EpochOrders):
        if effective_batch != orders.batch:
            raise ValueError("Effective batch differs from saved shuffle metadata")
        return orders.indices(zero_based_step)
    start = zero_based_step * effective_batch
    return orders[start:start + effective_batch]


def prefix_counts(orders, completed_steps, effective_batch, sequence_length, token_count):
    if isinstance(orders, EpochOrders):
        if orders.batch != effective_batch or orders.blocks != math.ceil(token_count / sequence_length):
            raise ValueError("Dataset/batch differs from saved shuffle metadata")
        return orders.prefix_counts(completed_steps, sequence_length, token_count)
    selection = orders[:min(len(orders), completed_steps * effective_batch)]
    remainder = token_count % sequence_length
    inputs = len(selection) * sequence_length
    if remainder and len(orders) - 1 in selection:
        inputs -= sequence_length - remainder
    applied = min(completed_steps, math.ceil(len(orders) / effective_batch))
    if (remainder == 1 and len(orders) - 1 in selection and (effective_batch == 1
            or (len(orders) % effective_batch == 1 and len(selection) == len(orders)
                and orders[-1] == len(orders) - 1))):
        applied -= 1
    return inputs, inputs - len(selection), applied
