"""
This is the final orchestrator. Require an integer seed. Call public Step 6 on its complete default axes; Step 6 executes public Steps 1 through 5 internally. Reuse public Step 5 once for the default branch evidence. From the tensor compute median_success, population spread of success, worst_failure, and the leading singular value of centered channels 0 through 5. Define alignment_margin=mean(alpha-1/sqrt(2)), force_gain=mean(1-Fnew/(FCI+1e-15)), path_uniformity=1/(1+mean(spacing)), stability=(median_success+max(force_gain,0)+path_uniformity)/(1+spread+worst_failure+leading_singular), and J=(1+max(alignment_margin,0))*stability/(1+mean(thresholds)+tensor_checksum/10000). Return unrounded J followed by all five upstream checksums, tensor_checksum, four branch counts, and all seven tensor and stability diagnostics.

The final scalar comes from the same full source-controlled path exercised by integration. Step 7 calls Step 6, and Step 6 runs Steps 1 through 5 internally.

Returns
-------
tuple[float,...] : J, six checksums, four branch counts, and eight robustness diagnostics.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_oci_neb_audit(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9) -> tuple:
    """Return J and every named diagnostic from the complete default audit.

    Raises:
        ValueError: If seed or an upstream panel argument is inadmissible.
    """
    return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_oci_neb_audit(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9) -> tuple:
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    tensor, tensor_checksum, panel_checksum, control_checksum, dimer_checksum, policy_checksum, path_checksum = _oracle_evaluate_oci_neb_robustness_tensor(seed, n_mechanisms, n_images)
    default = _oracle_reparameterize_successful_paths(seed, n_mechanisms, n_images)
    spacing, action, thresholds, alpha, curvature, fnew, fci = default[1:8]
    success_count = float(np.sum(action == 3))
    restore_count = float(np.sum(action == 1))
    failure_count = float(np.sum(action == 2))
    inactive_count = float(np.sum(action == 0))
    median_success = float(np.median(tensor[:, 0]))
    spread = float(np.std(tensor[:, 0], ddof=0))
    worst_failure = float(np.max(tensor[:, 2]))
    centered = tensor[:, :6] - np.mean(tensor[:, :6], axis=0)
    leading_singular = float(np.linalg.svd(centered, compute_uv=False)[0])
    alignment_margin = float(np.mean(alpha - 2 ** -0.5))
    force_gain = float(np.mean(1.0 - fnew / (fci + 1e-15)))
    path_uniformity = float(1.0 / (1.0 + np.mean(spacing)))
    stability = float((median_success + max(force_gain, 0.0) + path_uniformity) /
                      (1.0 + spread + worst_failure + leading_singular))
    j = float((1.0 + max(alignment_margin, 0.0)) * stability /
              (1.0 + np.mean(thresholds) + tensor_checksum / 10000.0))
    return (j, panel_checksum, control_checksum, dimer_checksum, policy_checksum, path_checksum,
            tensor_checksum, success_count, restore_count, failure_count, inactive_count,
            median_success, spread, worst_failure, leading_singular, alignment_margin,
            force_gain, path_uniformity, stability)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    flat_setup = 'np=__import__("numpy")\ndef _tree(value):\n    if isinstance(value, np.ndarray):\n        return [float(value.ndim), *[float(x) for x in value.shape], *value.astype(float).ravel().tolist()]\n    if isinstance(value, (tuple, list)):\n        out = [float(len(value))]\n        for item in value:\n            out.extend(_tree(item))\n        return out\n    return [float(value)]'
    error_setup = 'def _value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1.0\n    return 0.0'
    return [
      {'tol':1e-9,'setup':flat_setup,'call':'_tree(compute_oci_neb_audit())','gold_call':'_tree(_oracle_compute_oci_neb_audit())'},
      {'tol':1e-9,'setup':flat_setup,'call':'_tree(compute_oci_neb_audit(260601,8,7))','gold_call':'_tree(_oracle_compute_oci_neb_audit(260601,8,7))'},
      {'tol':1e-9,'setup':flat_setup,'call':'_tree(compute_oci_neb_audit(260777,24,11))','gold_call':'_tree(_oracle_compute_oci_neb_audit(260777,24,11))'},
      {'tol':1e-9,'setup':error_setup+'\na=("bad",18,9)','call':'_value_error(compute_oci_neb_audit,*a)','gold_call':'_value_error(_oracle_compute_oci_neb_audit,*a)'},
    ]
