"""CPU-only checks shared by the generated-dense training entry and tests."""
import hashlib

import numpy as np


def file_hash(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def validate_training_inputs(config, args, charts, chart_manifest, dense_manifest):
    if not isinstance(config.get('chart_every'), int) or config['chart_every'] < 2:
        raise ValueError('chart_every must be an integer >= 2 so both stages receive updates')
    if args.iterations < 1 or config['generated_rgb_weight'] <= 0:
        raise ValueError('iterations and generated_rgb_weight must be positive')
    if not all(isinstance(i, int) and i > 0 for k in ('save_iterations', 'checkpoint_iterations') for i in config[k]):
        raise ValueError('Save/checkpoint iterations must be positive integers')
    if args.densification_interval < 1 or args.opacity_reset_interval < 1:
        raise ValueError('Densification and opacity reset intervals must be positive')
    required = {'pts', 'depths', 'confs', 'prior_depths', 'scale_factor'}
    if set(charts) != required:
        raise ValueError(f'Expected upstream chart fields {sorted(required)}, got {sorted(charts)}')
    scale = charts['scale_factor']
    if np.size(scale) != 1 or not np.isfinite(scale).all() or float(scale) <= 0:
        raise ValueError('Chart scale_factor must be a positive finite scalar')
    names = [e['name'] for e in chart_manifest['frames']]
    dense = [e['name'] for e in dense_manifest['frames']]
    if not names or names != sorted(set(names)) or dense != sorted(set(dense)):
        raise ValueError('Manifests must use unique, sorted image names')
    if not set(names) <= set(dense):
        raise ValueError('Chart views must be included in dense supervision')
    if any(e['source_type'] != 'generated' for e in dense_manifest['frames']):
        raise ValueError('This entry uses a uniform generated-image loss policy')
    shape = charts['depths'].shape
    if len(shape) != 3 or shape[0] != len(names) or min(shape[1:]) < 2:
        raise ValueError('Chart depth shape must be (number_of_keyframes, H>=2, W>=2)')
    if any(charts[k].shape != shape for k in ('confs', 'prior_depths')) or charts['pts'].shape != shape+(3,):
        raise ValueError('Chart geometry/priors/confidence shapes disagree')
    if not np.isfinite(charts['confs']).all():
        raise ValueError('Nonfinite chart confidence')
    for key in ('depths', 'prior_depths'):
        depth = charts[key]/float(scale)
        valid = np.isfinite(depth) & (depth > .01) & (depth < 100.)
        if np.any(~valid.reshape(len(names), -1).any(1)):
            raise ValueError(f'A chart has no valid {key} in the camera range 0.01–100')
