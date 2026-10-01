"""
Call public Steps 1, 2, and 4. For accepted bursts only, replace the climbing image by the selected candidate and immediately redistribute every image to equal cumulative arc-length targets using coordinate-wise linear interpolation on the current polyline. The interpolation is a disclosed deterministic surrogate for the paper's geometry-only path redistribution. Leave every nonaccepted path unchanged. For each mechanism record std(segment lengths)/(mean(segment lengths)+1e-15). Define path_checksum=dot(updated_paths.ravel(),0.9+(k mod 37)/41)+dot(spacing,11+m/29). Return updated paths, spacing, the full policy evidence, four upstream checksums, and path_checksum.

The source decides when redistribution occurs and that optimizer history is reset. Linear interpolation is used only to make this audit deterministic and is not attributed to the paper.

Returns
-------
tuple : updated paths, spacing, actions, thresholds, branch evidence, and checksums.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reparameterize_successful_paths(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9, step_size: float = 0.19) -> tuple:
    """Redistribute successful paths and return spacing and policy diagnostics.

    Raises:
        ValueError: If step_size or an upstream panel argument is inadmissible.
    """
    return None, None, None, None, None, None, None, None, None, None, None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _linear_reparameterize(path):
    delta = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(delta)))
    target = np.linspace(0.0, arc[-1], len(path))
    out = np.empty_like(path)
    for d in range(path.shape[1]):
        out[:, d] = np.interp(target, arc, path[:, d])
    return out

def _oracle_reparameterize_successful_paths(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9, step_size: float = 0.19) -> tuple:
    paths, _, _, _, _ = _oracle_generate_oci_neb_panel(seed, n_mechanisms, n_images)
    indices, _, _, _, _, _, _, _ = _oracle_compute_climbing_controls(seed, n_mechanisms, n_images)
    action, thresholds, accepted, positions, alpha, curvature, fnew, fci, panel_checksum, control_checksum, dimer_checksum, policy_checksum = _oracle_apply_oci_neb_policy(seed, n_mechanisms, n_images, step_size)
    updated = paths.copy()
    spacing = np.empty(n_mechanisms, dtype=float)
    for m, k in enumerate(indices):
        if accepted[m]:
            updated[m, k] = positions[m]
            updated[m] = _linear_reparameterize(updated[m])
        gaps = np.linalg.norm(np.diff(updated[m], axis=0), axis=1)
        spacing[m] = np.std(gaps) / (np.mean(gaps) + 1e-15)
    checksum = float(np.dot(updated.ravel(), 0.9 + (np.arange(updated.size) % 37) / 41.0)
                     + np.dot(spacing, 11.0 + np.arange(n_mechanisms) / 29.0))
    return updated, spacing, action, thresholds, alpha, curvature, fnew, fci, panel_checksum, control_checksum, dimer_checksum, policy_checksum, checksum

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    flat_setup = 'np=__import__("numpy")\ndef _tree(value):\n    if isinstance(value, np.ndarray):\n        return [float(value.ndim), *[float(x) for x in value.shape], *value.astype(float).ravel().tolist()]\n    if isinstance(value, (tuple, list)):\n        out = [float(len(value))]\n        for item in value:\n            out.extend(_tree(item))\n        return out\n    return [float(value)]'
    error_setup = 'def _value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1.0\n    return 0.0'
    return [
      {'tol':1e-9,'setup':flat_setup,'call':'_tree(reparameterize_successful_paths())','gold_call':'_tree(_oracle_reparameterize_successful_paths())'},
      {'tol':1e-9,'setup':flat_setup,'call':'_tree(reparameterize_successful_paths(260601,8,7,.11))','gold_call':'_tree(_oracle_reparameterize_successful_paths(260601,8,7,.11))'},
      {'tol':1e-9,'setup':flat_setup,'call':'_tree(reparameterize_successful_paths(260777,24,11,.31))','gold_call':'_tree(_oracle_reparameterize_successful_paths(260777,24,11,.31))'},
      {'tol':1e-9,'setup':error_setup+'\na=(260529,18,6,.19)','call':'_value_error(reparameterize_successful_paths,*a)','gold_call':'_value_error(_oracle_reparameterize_successful_paths,*a)'},
    ]
