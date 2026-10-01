"""
Call public Step 5, which executes public Steps 1 through 4 internally. Also call public Steps 1 and 2 to recover force baselines and latch counts. Require each axis to contain at least two values, every relative trigger in (0,1), every alignment threshold in (1/sqrt(2),1], and every force scale positive. In C order over lambda_axis, alignment_axis, and force_axis, reapply the explicit Step 4 policy to force_scale*Fnew. The tensor has len(lambda_axis)*len(alignment_axis)*len(force_axis) rows and 8 columns; only the default axes produce 80 by 8. Store success, restoration, failure, and inactive fractions; mean force reduction over all mechanisms, with rejected and inactive mechanisms contributing zero; mean threshold/F0; dot(action,1+m/23); and mean(spacing*(1+accepted)). Defaults are lambda_axis=(0.24,0.31,0.38,0.46), alignment_axis=(0.72,0.85,0.91,0.96), and force_axis=(0.82,0.94,1.0,1.08,1.22). Define tensor_checksum=dot(tensor.ravel(),0.6+(k mod 53)/59). Return tensor, tensor_checksum, panel_checksum, control_checksum, dimer_checksum, policy_checksum, and path_checksum.

The robustness tensor perturbs policy margins around the source configuration while preserving the source branch semantics in every cell.

Returns
-------
tuple : (L*A*F) by 8 tensor, tensor checksum, and five upstream checksums.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_oci_neb_robustness_tensor(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9, lambda_axis: tuple = (0.24, 0.31, 0.38, 0.46), alignment_axis: tuple = (0.72, 0.85, 0.91, 0.96), force_axis: tuple = (0.82, 0.94, 1.0, 1.08, 1.22)) -> tuple:
    """Return the complete adaptive-policy robustness tensor and checksums.

    Raises:
        ValueError: If an axis or an upstream panel argument is inadmissible.
    """
    return None, None, None, None, None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _apply_policy(alpha, curvature, fnew, fci, stable, baselines, lambda_rel, alpha_tol, kappa):
    n = len(alpha)
    action = np.zeros(n, dtype=int)
    thresholds = lambda_rel * baselines
    accepted = np.zeros(n, dtype=bool)
    for i in range(n):
        active = stable[i] >= kappa and fci[i] < thresholds[i]
        if not active: action[i] = 0
        elif curvature[i] > 0.0: action[i] = 1
        elif alpha[i] < alpha_tol or fnew[i] >= fci[i]:
            action[i] = 2
            thresholds[i] = baselines[i] * lambda_rel * (0.5 + 0.5 * alpha[i])
        else:
            action[i] = 3
            accepted[i] = True
            thresholds[i] = fnew[i] * (0.5 + 0.4 * fnew[i] / fci[i])
    return action, thresholds, accepted

def _oracle_evaluate_oci_neb_robustness_tensor(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9, lambda_axis: tuple = (0.24, 0.31, 0.38, 0.46), alignment_axis: tuple = (0.72, 0.85, 0.91, 0.96), force_axis: tuple = (0.82, 0.94, 1.0, 1.08, 1.22)) -> tuple:
    if min(len(lambda_axis), len(alignment_axis), len(force_axis)) < 2:
        raise ValueError("every robustness axis must contain at least two values")
    if any(x <= 0.0 or x >= 1.0 for x in lambda_axis) or any(x <= 2 ** -0.5 or x > 1.0 for x in alignment_axis) or any(x <= 0.0 for x in force_axis):
        raise ValueError("robustness axes contain inadmissible values")
    updated, spacing, action0, thresholds0, alpha, curvature, fnew, fci, panel_checksum, control_checksum, dimer_checksum, policy_checksum, path_checksum = _oracle_reparameterize_successful_paths(seed, n_mechanisms, n_images)
    _, _, _, baselines, _ = _oracle_generate_oci_neb_panel(seed, n_mechanisms, n_images)
    _, _, _, _, stable, _, _, _ = _oracle_compute_climbing_controls(seed, n_mechanisms, n_images)
    rows = []
    for lam in lambda_axis:
        for atol in alignment_axis:
            for scale in force_axis:
                action, thresholds, accepted = _apply_policy(alpha, curvature, fnew * scale, fci, stable, baselines, float(lam), float(atol), 5)
                success = np.mean(action == 3)
                restored = np.mean(action == 1)
                failed = np.mean(action == 2)
                inactive = np.mean(action == 0)
                reduction = np.mean(np.where(accepted, 1.0 - fnew * scale / (fci + 1e-15), 0.0))
                threshold_mean = np.mean(thresholds / (baselines + 1e-15))
                action_checksum = np.dot(action, 1.0 + np.arange(len(action)) / 23.0)
                spacing_score = np.mean(spacing * (1.0 + accepted.astype(float)))
                rows.append([success, restored, failed, inactive, reduction, threshold_mean, action_checksum, spacing_score])
    tensor = np.asarray(rows, dtype=float)
    weights = 0.6 + (np.arange(tensor.size) % 53) / 59.0
    checksum = float(np.dot(tensor.ravel(), weights))
    return tensor, checksum, panel_checksum, control_checksum, dimer_checksum, policy_checksum, path_checksum

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    flat_setup = 'np=__import__("numpy")\ndef _tree(value):\n    if isinstance(value, np.ndarray):\n        return [float(value.ndim), *[float(x) for x in value.shape], *value.astype(float).ravel().tolist()]\n    if isinstance(value, (tuple, list)):\n        out = [float(len(value))]\n        for item in value:\n            out.extend(_tree(item))\n        return out\n    return [float(value)]\ndef _first_inactive(value):\n    tensor=np.asarray(value[0])\n    assert tensor.shape == (8,8)\n    assert np.all(tensor[0, :3] == 0) and tensor[0, 3] == 1\n    return _tree(value)'
    error_setup = 'def _value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1.0\n    return 0.0'
    return [
        {
            'tol': 1e-9,
            'setup': flat_setup,
            'call': '_tree(evaluate_oci_neb_robustness_tensor())',
            'gold_call': '_tree(_oracle_evaluate_oci_neb_robustness_tensor())',
        },
        {
            'tol': 1e-9,
            'setup': flat_setup,
            'call': '_tree(evaluate_oci_neb_robustness_tensor(260777,24,11,(.2,.3,.45),(.71,.88),(.8,1.,1.3)))',
            'gold_call': '_tree(_oracle_evaluate_oci_neb_robustness_tensor(260777,24,11,(.2,.3,.45),(.71,.88),(.8,1.,1.3)))',
        },
        {
            'tol': 1e-9,
            'setup': flat_setup,
            'call': '_first_inactive(evaluate_oci_neb_robustness_tensor(260529,18,9,(.12,.16),(.72,.85),(.9,1.1)))',
            'gold_call': '_first_inactive(_oracle_evaluate_oci_neb_robustness_tensor(260529,18,9,(.12,.16),(.72,.85),(.9,1.1)))',
        },
        {
            'tol': 1e-9,
            'setup': error_setup + '\nbad=(.31,)',
            'call': '_value_error(evaluate_oci_neb_robustness_tensor,260529,18,9,bad,(.72,.85),(.9,1.1))',
            'gold_call': '_value_error(_oracle_evaluate_oci_neb_robustness_tensor,260529,18,9,bad,(.72,.85),(.9,1.1))',
        },
    ]
