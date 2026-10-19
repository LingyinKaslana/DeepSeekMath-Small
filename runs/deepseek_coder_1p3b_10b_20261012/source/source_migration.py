"""Strict resume signatures with auditable source-only or native-backend changes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from training_extension import (authorize_extension_signatures, validate_extension_entry,
                                verify_extension_snapshots)


def without_source(signature):
    return {key: value for key, value in signature.items() if key != 'source_sha256'}


def without_execution(signature):
    return {key: value for key, value in signature.items()
            if key not in ('source_sha256', 'backend_config_sha256')}


def authorize_signature(saved, current, allow_source_migration=False):
    if saved == current:
        return False
    if (not allow_source_migration or without_source(saved) != without_source(current)
            or not isinstance(saved.get('source_sha256'), dict)
            or not isinstance(current.get('source_sha256'), dict)):
        raise ValueError('Resume configuration/data/LR/seed/source differs from the saved training state; '
                         '--allow-source-migration permits source_sha256 changes only')
    return True


def validate_native_backend_change(before, after):
    """Only approved execution and explicitly identified description fields vary."""
    allowed = {'execution_mode', 'mask_policy', 'compiled_optimizer', 'options', 'purpose', 'validation_evidence'}
    changes = []
    for key in sorted(set(before) | set(after)):
        if key in before and key in after and before[key] == after[key]:
            continue
        if key not in allowed:
            raise ValueError(f'Native backend migration cannot change {key}')
        if key == 'options':
            expected = dict(before[key], compile_scope='none')
            if after[key] != expected:
                raise ValueError('Native backend migration may change options.compile_scope only')
            changes.append({'field': 'options.compile_scope', 'before': before[key].get('compile_scope'), 'after': 'none', 'kind': 'execution'})
        else:
            changes.append({'field': key, 'before': before.get(key), 'after': after.get(key),
                            'kind': 'description_only' if key in ('purpose', 'validation_evidence') else 'execution'})
    if (before.get('execution_mode', 'configured') == 'standard_native'
            or after.get('execution_mode') != 'standard_native' or after.get('mask_policy') != 'ones'
            or after.get('compiled_optimizer', False) or after.get('fused_optimizer', False)
            or after['options'].get('compile_scope') != 'none'
            or after['options'].get('kernels', 'native') != 'native'
            or after['options'].get('loss', 'native') != 'native'
            or after['options'].get('preserve_silu', False)):
        raise ValueError('Only a configured backend to untouched standard_native transition is permitted')
    return changes


def authorize_native_signature(saved, current, before_backend, after_backend):
    if saved == current:
        return False
    if (without_execution(saved) != without_execution(current)
            or not isinstance(saved.get('source_sha256'), dict)
            or not isinstance(current.get('source_sha256'), dict)
            or saved.get('backend_config_sha256') == current.get('backend_config_sha256')):
        raise ValueError('Native migration permits backend/source implementation changes only; data/LR/seed/context/batch/epochs remain fixed')
    validate_native_backend_change(before_backend, after_backend)
    return True


def native_configs(checkpoint, current_path, saved, current, sha):
    before_path, after_path = checkpoint.parent / 'backend_config.json', current_path
    if sha(before_path) != saved['backend_config_sha256'] or sha(after_path) != current['backend_config_sha256']:
        raise ValueError('Native migration backend snapshots do not match signatures')
    return json.loads(before_path.read_text()), json.loads(after_path.read_text())


def validate_lineage(state):
    """Return original signature and a continuous narrowly authorized change chain."""
    signature = state['signature']
    original = state.get('original_signature', signature)
    migrations = state.get('source_migrations', [])
    if not isinstance(migrations, list) or not isinstance(original, dict):
        raise ValueError('Invalid source-migration audit')
    previous = original
    for entry in migrations:
        if (entry.get('from_signature') != previous or not isinstance(entry.get('completed_steps'), int)
                or isinstance(entry.get('completed_steps'), bool) or entry['completed_steps'] < 0):
            raise ValueError('Invalid or discontinuous source-migration audit')
        following = entry.get('to_signature')
        if not isinstance(following, dict):
            raise ValueError('Missing migration target signature')
        actual_fields = sorted(key for key in set(previous) | set(following) if previous.get(key) != following.get(key))
        if entry.get('changed_fields') != actual_fields:
            raise ValueError('Migration changed_fields differs from actual signatures')
        if entry.get('authorized_by') == '--allow-source-migration':
            if not authorize_signature(previous, following, True):
                raise ValueError('Source migration must record an actual source-only change')
        elif entry.get('authorized_by') == '--allow-native-backend-migration':
            if not authorize_native_signature(previous, following, entry.get('from_backend_config'), entry.get('to_backend_config')):
                raise ValueError('Native migration must record an actual backend transition')
            expected_changes = validate_native_backend_change(entry['from_backend_config'], entry['to_backend_config'])
            if entry.get('backend_changes') != expected_changes:
                raise ValueError('Native backend change audit differs from its configuration snapshots')
            for name in ('from_backend_config_path', 'to_backend_config_path'):
                if not isinstance(entry.get(name), str) or not entry[name]:
                    raise ValueError(f'Missing native-backend snapshot {name}')
        elif entry.get('authorized_by') == '--training-extension-file':
            authorize_extension_signatures(previous, following)
            validate_extension_entry(entry)
        else:
            raise ValueError('Unknown migration authorization')
        for name in ('from_source_directory', 'to_source_directory', 'resumed_checkpoint', 'saved_utc'):
            if not isinstance(entry.get(name), str) or not entry[name]:
                raise ValueError(f'Missing source-migration audit {name}')
        previous = following
    if previous != signature:
        raise ValueError('Original signature / migration chain differs from current signature')
    if migrations and any(a['completed_steps'] > b['completed_steps'] for a, b in zip(migrations, migrations[1:])):
        raise ValueError('Source-migration steps are not monotonic')
    if migrations and migrations[-1]['completed_steps'] > state.get('completed_steps', -1):
        raise ValueError('Source migration lies beyond saved training prefix')
    return original, migrations


def _entry(state, current, checkpoint, output, authorization):
    return {'authorized_by': authorization,
            'changed_fields': sorted(key for key in set(state['signature']) | set(current) if state['signature'].get(key) != current.get(key)),
            'completed_steps': state['completed_steps'], 'from_signature': state['signature'],
            'to_signature': current, 'resumed_checkpoint': str(checkpoint.resolve()),
            'from_source_directory': str((checkpoint.parent / 'source').resolve()),
            'to_source_directory': str((output / 'source').resolve()),
            'saved_utc': datetime.now(timezone.utc).isoformat()}


def apply_source_migration(state, current, checkpoint, output):
    original, migrations = validate_lineage(state)
    changed = authorize_signature(state['signature'], current, True)
    state['original_signature'] = original
    if changed:
        migrations = [*migrations, _entry(state, current, checkpoint, output, '--allow-source-migration')]
    state['signature'], state['source_migrations'] = current, migrations
    validate_lineage(state)
    return migrations


def apply_native_migration(state, current, checkpoint, output, sha):
    original, migrations = validate_lineage(state)
    before, after = native_configs(checkpoint, output / 'backend_config.json', state['signature'], current, sha)
    changed = authorize_native_signature(state['signature'], current, before, after)
    state['original_signature'] = original
    if changed:
        entry = _entry(state, current, checkpoint, output, '--allow-native-backend-migration')
        entry.update(from_backend_config=before, to_backend_config=after,
                     from_backend_config_path=str((checkpoint.parent / 'backend_config.json').resolve()),
                     to_backend_config_path=str((output / 'backend_config.json').resolve()),
                     backend_changes=validate_native_backend_change(before, after))
        migrations = [*migrations, entry]
    state['signature'], state['source_migrations'] = current, migrations
    validate_lineage(state)
    return migrations


def verify_lineage_snapshots(state, sha):
    original, migrations = validate_lineage(state)
    for entry in migrations:
        for side in ('from', 'to'):
            directory = Path(entry[f'{side}_source_directory'])
            for name, digest in entry[f'{side}_signature']['source_sha256'].items():
                if (Path(name).name != name or not isinstance(digest, str)
                        or re.fullmatch(r'[0-9a-f]{64}', digest) is None):
                    raise ValueError('Invalid source-migration snapshot hash')
                path = directory / name
                if not path.is_file() or sha(path) != digest:
                    raise ValueError(f'Source-migration historical snapshot mismatch: {path}')
            if entry['authorized_by'] == '--allow-native-backend-migration':
                path = Path(entry[f'{side}_backend_config_path'])
                if (not path.is_file() or sha(path) != entry[f'{side}_signature']['backend_config_sha256']
                        or json.loads(path.read_text()) != entry[f'{side}_backend_config']):
                    raise ValueError(f'Native backend historical snapshot mismatch: {path}')
        if entry['authorized_by'] == '--training-extension-file':
            verify_extension_snapshots(entry, sha)
    return {'original_signature': original, 'source_migrations': migrations,
            'source_only_migration_count': sum(entry['authorized_by'] == '--allow-source-migration' for entry in migrations),
            'native_backend_migration_count': sum(entry['authorized_by'] == '--allow-native-backend-migration' for entry in migrations),
            'training_extension_count': sum(entry['authorized_by'] == '--training-extension-file' for entry in migrations)}
