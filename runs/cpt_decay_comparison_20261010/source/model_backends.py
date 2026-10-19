"""Instance-local Qwen2 training backends, preserving FP32 master parameters.

``configure_model(model, options)`` returns ``(forward_callable, metadata)``.
The callable accepts the same arguments as ``model``. Save, inspect, and optimize
the underlying ``model``; compiled wrappers are only used to execute forwards.
For a reference evaluation use ``eager_native(model)`` and call ``model`` inside
the context, or use ``eager_native_forward(model, **batch)`` directly.

The caller owns autocast: forward under BF16 autocast, backward outside it.
Compiled calls should receive an already-resolved causal attention mask, e.g.
``attention_mask={"full_attention": None}``, for this Transformers version.
"""

from contextlib import contextmanager
import functools
import importlib.metadata
from types import FunctionType, MethodType

import torch


KERNELS = ("native", "rms", "rms_swiglu", "rms_swiglu_rope")
LOSSES = ("native", "native_chunked", "liger_ce_fp32", "liger_ce_bf16", "liger_lce")
COMPILE_SCOPES = ("none", "layers", "model", "mlp")


def _originals(model):
    state = getattr(model, "_cpt_backend_originals", None)
    if state is None:
        state = {
            "model_forward": model.forward,
            "loss_function": getattr(model, "_cpt_native_loss_function", model.loss_function),
            "modules": [(module, module.forward) for layer in model.model.layers
                        for module in (layer.input_layernorm, layer.post_attention_layernorm,
                                       layer.mlp, layer.self_attn)] + [(model.model.norm, model.model.norm.forward)],
            "activations": [(layer.mlp, layer.mlp.act_fn) for layer in model.model.layers],
        }
        model._cpt_backend_originals = state
        model._cpt_native_forward = state["model_forward"]
        model._cpt_native_loss_function = state["loss_function"]
    return state


def _restore_native(model):
    state = _originals(model)
    model.forward = state["model_forward"]
    model.loss_function = state["loss_function"]
    for module, forward in state["modules"]:
        module.forward = forward
    for module, activation in state["activations"]:
        module.act_fn = activation
    # nn.Module.compile modifies the call implementation, not the parameters or
    # state_dict names. Disabling it also allows sequential calibration cases.
    for module in model.modules():
        if getattr(module, "_compiled_call_impl", None) is not None:
            module._compiled_call_impl = None


@contextmanager
def eager_native(model):
    """Temporarily bypass every local patch and compiled module call."""
    state = _originals(model)
    current_forward, current_loss = model.forward, model.loss_function
    current_modules = [(module, module.forward) for module, _ in state["modules"]]
    current_activations = [(module, module.act_fn) for module, _ in state["activations"]]
    compiled = [(module, module._compiled_call_impl) for module in model.modules()
                if getattr(module, "_compiled_call_impl", None) is not None]
    _restore_native(model)
    try:
        yield model
    finally:
        model.forward, model.loss_function = current_forward, current_loss
        for module, forward in current_modules:
            module.forward = forward
        for module, activation in current_activations:
            module.act_fn = activation
        for module, call in compiled:
            module._compiled_call_impl = call


def eager_native_forward(model, *args, **kwargs):
    with eager_native(model):
        return model(*args, **kwargs)


@contextmanager
def _eager_calls(model):
    compiled = [(module, module._compiled_call_impl) for module in model.modules()
                if getattr(module, "_compiled_call_impl", None) is not None]
    for module, _ in compiled:
        module._compiled_call_impl = None
    try:
        yield model
    finally:
        for module, call in compiled:
            module._compiled_call_impl = call


def _patch_norm(module, in_place=False):
    from liger_kernel.ops import LigerRMSNormFunction

    def forward(self, hidden_states):
        # HF promotes its final weight multiplication if these dtypes differ;
        # Liger stores the result in the input dtype. Do not silently change it.
        if hidden_states.dtype != self.weight.dtype:
            raise RuntimeError("Liger RMSNorm requires matching hidden/weight dtypes for HF-equivalent semantics")
        return LigerRMSNormFunction.apply(hidden_states, self.weight, self.variance_epsilon,
                                         0.0, "llama", in_place, None)

    module.forward = MethodType(forward, module)


def _patch_rope(module):
    from liger_kernel.transformers.rope import liger_rotary_pos_emb

    original = module.forward.__func__
    if "apply_rotary_pos_emb" not in original.__code__.co_names:
        raise RuntimeError("Installed Qwen2 attention forward cannot be patched locally for RoPE")
    # A private globals dictionary substitutes only this attention instance's
    # RoPE call. Other model instances and the Transformers module stay intact.
    namespace = dict(original.__globals__)
    namespace["apply_rotary_pos_emb"] = liger_rotary_pos_emb
    local = FunctionType(original.__code__, namespace, original.__name__,
                         original.__defaults__, original.__closure__)
    local.__kwdefaults__ = original.__kwdefaults__
    functools.update_wrapper(local, original)
    # An unwrap-aware compiler must trace the substituted globals, rather than
    # follow __wrapped__ back to the native function's globals dictionary.
    del local.__wrapped__
    module.forward = MethodType(local, module)


def _patch_ce(model, dtype):
    from liger_kernel.transformers import LigerCrossEntropyLoss

    def loss_function(logits, labels, vocab_size, num_items_in_batch=None,
                      ignore_index=-100, shift_labels=None, **kwargs):
        if shift_labels is None:
            shift_labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = LigerCrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)(
            logits.to(dtype=dtype).reshape(-1, vocab_size), shift_labels.reshape(-1).to(logits.device))
        return loss / num_items_in_batch if num_items_in_batch is not None else loss

    model.loss_function = loss_function


def configure_model(model, options):
    """Configure one model; no global class/function monkey patches are used.

    Options: kernels, loss, chunk_mem_const, compile_scope, compile_mode,
    compile_fullgraph, compile_dynamic, compile_shape, compile_label_alias,
    rms_in_place, emulate_precision_casts, preserve_silu, and ce_chunk_rows. Omitted options use the native
    eager path. compile_label_alias='same_as_input' sends masked CPT tails to
    eager execution, avoiding a second graph for distinct label storage. The
    default 'any' supports independently masked SFT labels. BF16-only CE is an
    explicitly different rounding path and still requires numerical checks.
    """
    options = dict(options)
    accepted = {"kernels", "loss", "chunk_mem_const", "compile_scope", "compile_mode",
                "compile_fullgraph", "compile_dynamic", "compile_shape", "compile_label_alias",
                "rms_in_place", "emulate_precision_casts", "preserve_silu", "ce_chunk_rows"}
    unexpected = set(options) - accepted
    if unexpected:
        raise ValueError(f"Unknown backend options: {sorted(unexpected)}")
    kernels = options.get("kernels", "native")
    loss = options.get("loss", "native")
    scope = options.get("compile_scope", "none")
    mode = options.get("compile_mode", "default")
    chunks = options.get("chunk_mem_const", 1)
    compile_shape = options.get("compile_shape", [16, 2048])
    compile_label_alias = options.get("compile_label_alias", "any")
    preserve_silu = options.get("preserve_silu", False)
    ce_chunk_rows = options.get("ce_chunk_rows", 4096)
    if kernels not in KERNELS or loss not in LOSSES or scope not in COMPILE_SCOPES:
        raise ValueError(f"Unsupported backend: kernels={kernels}, loss={loss}, compile_scope={scope}")
    if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 1:
        raise ValueError("chunk_mem_const must be a positive integer")
    if compile_label_alias not in ("any", "same_as_input"):
        raise ValueError("compile_label_alias must be 'any' or 'same_as_input'")
    if not isinstance(preserve_silu, bool):
        raise ValueError("preserve_silu must be a boolean")
    if isinstance(ce_chunk_rows, bool) or not isinstance(ce_chunk_rows, int) or ce_chunk_rows < 1:
        raise ValueError("ce_chunk_rows must be a positive integer")
    if loss == "native_chunked" and scope == "model":
        raise ValueError("Native chunked CE recomputation must remain outside a compiled model graph")
    if preserve_silu and kernels not in ("native", "rms"):
        raise ValueError("preserve_silu requires native/rms kernels; fused SwiGLU ignores the activation module")
    if len(compile_shape) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                                    for value in compile_shape):
        raise ValueError("compile_shape must contain two positive integers: batch and sequence length")
    if model.config.model_type != "qwen2":
        raise ValueError("This backend currently supports only Qwen2 models")
    if not all(parameter.dtype == torch.float32 for parameter in model.parameters()):
        raise ValueError("This training backend requires FP32 master parameters")
    _restore_native(model)
    # AOTAutograd must model the actual training loop: the caller exits forward
    # autocast before loss.backward(). The default assumes the opposite.
    import torch._functorch.config as functorch_config
    functorch_config.backward_pass_autocast = "off"
    import torch._inductor.config as inductor_config
    inductor_config.emulate_precision_casts = options.get("emulate_precision_casts", False)

    if kernels != "native":
        _patch_norm(model.model.norm, options.get("rms_in_place", False))
        for layer in model.model.layers:
            _patch_norm(layer.input_layernorm, options.get("rms_in_place", False))
            _patch_norm(layer.post_attention_layernorm, options.get("rms_in_place", False))
    if kernels in ("rms_swiglu", "rms_swiglu_rope"):
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
        for layer in model.model.layers:
            layer.mlp.forward = MethodType(LigerSwiGLUMLP.forward, layer.mlp)
    if kernels == "rms_swiglu_rope":
        for layer in model.model.layers:
            _patch_rope(layer.self_attn)
    if preserve_silu:
        from native_ops import NativeSiLU
        if model.config.hidden_act not in ("silu", "swish"):
            raise ValueError("Opaque native SiLU requires a SiLU model activation")
        for layer in model.model.layers:
            layer.mlp.act_fn = NativeSiLU()
    if loss == "native_chunked":
        from native_chunked_ce import causal_lm_loss
        model.loss_function = functools.partial(causal_lm_loss, chunk_rows=ce_chunk_rows)
    elif loss in ("liger_ce_fp32", "liger_ce_bf16"):
        _patch_ce(model, torch.float32 if loss == "liger_ce_fp32" else torch.bfloat16)
    elif loss == "liger_lce":
        from liger_kernel.transformers.model.qwen2 import lce_forward

        def forward(self, *args, **kwargs):
            kwargs.setdefault("chunk_mem_const", chunks)
            return lce_forward(self, *args, **kwargs)

        model.forward = MethodType(forward, model)
    model._cpt_eager_forward = model.forward
    compile_kwargs = {"mode": mode, "fullgraph": options.get("compile_fullgraph", False),
                      "dynamic": options.get("compile_dynamic", False)}
    if scope in ("layers", "mlp"):
        for layer in model.model.layers:
            (layer if scope == "layers" else layer.mlp).compile(**compile_kwargs)
        callable_model = model
    elif scope == "model":
        callable_model = torch.compile(model, **compile_kwargs)
    else:
        callable_model = model

    metadata = {
        "kernels": kernels, "loss": loss, "chunk_mem_const": chunks,
        "compile_scope": scope, "compile_mode": mode, "compile_shape": list(compile_shape),
        "compile_label_alias": compile_label_alias,
        "compile_fullgraph": compile_kwargs["fullgraph"], "compile_dynamic": compile_kwargs["dynamic"],
        "rms_in_place": options.get("rms_in_place", False),
        "emulate_precision_casts": options.get("emulate_precision_casts", False),
        "preserve_silu": preserve_silu,
        "ce_chunk_rows": ce_chunk_rows,
        "master_parameter_dtype": "float32", "forward_autocast_dtype": "bfloat16",
        "backward_autocast": "off", "instance_local": True,
        "torch_version": torch.__version__, "transformers_version": importlib.metadata.version("transformers"),
        "liger_version": importlib.metadata.version("liger-kernel"),
        "native_validation": "with eager_native(model): model(**batch)",
        "notes": ["Training-loop code must still verify FP32 gradients and optimizer states.",
                  "RoPE and BF16 CE change intermediate rounding; verify before selecting them.",
                  "Pass an already-resolved attention mask dictionary to compiled forwards."],
    }
    model._cpt_backend_metadata = metadata
    if scope != "none":
        compiled_model = callable_model

        def execute(*args, **kwargs):
            force = kwargs.pop("_cpt_force_compile", False)
            inputs = kwargs.get("input_ids")
            if inputs is None:
                inputs = kwargs.get("inputs_embeds")
            if inputs is None and args:
                inputs = args[0]
            matches = inputs is not None and tuple(inputs.shape[:2]) == tuple(execute.compile_shape)
            # Both native Qwen2 and its local LCE forward place labels at the
            # sixth positional argument. Normal CPT calls use keyword labels.
            labels = kwargs.get("labels", args[5] if len(args) > 5 else None)
            labels_match = compile_label_alias == "any" or labels is inputs
            if force or (model.training and matches and labels_match):
                return compiled_model(*args, **kwargs)
            # Shape tails, masked-label tails and validation use patched eager
            # execution. The explicit native-validation context is separate.
            with _eager_calls(model):
                return model(*args, **kwargs)

        def set_compile_shape(shape):
            if len(shape) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                                      for value in shape):
                raise ValueError("compile shape must contain two positive integers")
            execute.compile_shape = tuple(shape)
            metadata["compile_shape"] = list(shape)

        execute.compile_shape = tuple(compile_shape)
        execute.set_compile_shape = set_compile_shape
        execute.compiled_model = compiled_model
        execute.eager_model = model
        callable_model = execute
    return callable_model, metadata


attach_backend = configure_model
