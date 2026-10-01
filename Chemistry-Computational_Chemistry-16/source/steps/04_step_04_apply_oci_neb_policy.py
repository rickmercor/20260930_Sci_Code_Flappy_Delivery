"""
Call public Steps 1 through 3. Recover the source values for the default relative trigger, alignment tolerance, stability count, linear alignment-failure penalty, and successful-burst threshold update. For this audit, resolve overlapping conditions with this explicit benchmark branch order: active means stable>=kappa and FCI<lambda_rel*F0; otherwise action 0. If active and curvature>0, action 1, restore the original climbing position, and clear the stored mode before NEB resumes. Otherwise, if alpha<alpha_tol OR Fnew>=FCI, action 2, reject the burst, and retain the original climbing position even when force improved. Only the remaining branch is accepted as action 3 and uses the candidate position. Use defaults lambda_rel=0.31, alpha_tol=0.85, and kappa=5. Define policy_checksum=dot(action,7+m/13)+dot(thresholds,2+m/17)+dot(positions.ravel(),0.8+(k mod 19)/23). Return action, thresholds, acceptance mask, selected positions, alignment, curvature, Fnew, FCI, panel_checksum, control_checksum, dimer_checksum, and policy_checksum.

The source supplies the adaptive thresholds and update formulas. When its conditions overlap, this benchmark deliberately gives alignment and force failure priority over force improvement, and keeps the original climbing point for every nonaccepted branch. Positive curvature additionally clears the stored mode before ordinary NEB resumes. These audit conventions make the deterministic branch order unambiguous without attributing the overlap resolution to the paper.

Returns
-------
tuple : actions, thresholds, accepted mask, positions, branch evidence, and checksums.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apply_oci_neb_policy(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9, step_size: float = 0.19, lambda_rel: float = 0.31, alpha_tol: float = 0.85, kappa: int = 5) -> tuple:
    """Apply the explicit audit branch order and return branch diagnostics.

    Raises:
        ValueError: If a policy parameter or upstream panel argument is inadmissible.
    """
    return None, None, None, None, None, None, None, None, None, None, None, None

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
        if not active:
            action[i] = 0
        elif curvature[i] > 0.0:
            action[i] = 1
        elif alpha[i] < alpha_tol or fnew[i] >= fci[i]:
            action[i] = 2
            thresholds[i] = baselines[i] * lambda_rel * (0.5 + 0.5 * alpha[i])
        else:
            action[i] = 3
            accepted[i] = True
            thresholds[i] = fnew[i] * (0.5 + 0.4 * fnew[i] / fci[i])
    return action, thresholds, accepted

def _oracle_apply_oci_neb_policy(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9, step_size: float = 0.19, lambda_rel: float = 0.31, alpha_tol: float = 0.85, kappa: int = 5) -> tuple:
    if not (0.0 < lambda_rel < 1.0) or not (2 ** -0.5 < alpha_tol <= 1.0) or not isinstance(kappa, int) or kappa < 1:
        raise ValueError("inadmissible policy parameter")
    paths, _, _, _, _ = _oracle_generate_oci_neb_panel(seed, n_mechanisms, n_images)
    indices, _, _, _, _, _, _, _ = _oracle_compute_climbing_controls(seed, n_mechanisms, n_images)
    axes, alpha, curvature, candidates, fnew, fci, stable, baselines, panel_checksum, control_checksum, dimer_checksum = _oracle_run_aligned_dimer_bursts(seed, n_mechanisms, n_images, step_size)
    action, thresholds, accepted = _apply_policy(alpha, curvature, fnew, fci, stable, baselines, lambda_rel, alpha_tol, kappa)
    positions = np.array([paths[m, k] for m, k in enumerate(indices)])
    positions[accepted] = candidates[accepted]
    checksum = float(np.dot(action, 7.0 + np.arange(n_mechanisms) / 13.0)
                     + np.dot(thresholds, 2.0 + np.arange(n_mechanisms) / 17.0)
                     + np.dot(positions.ravel(), 0.8 + (np.arange(positions.size) % 19) / 23.0))
    return action, thresholds, accepted, positions, alpha, curvature, fnew, fci, panel_checksum, control_checksum, dimer_checksum, checksum

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    flat_setup = 'np=__import__("numpy")\ndef _tree(value):\n    if isinstance(value, np.ndarray):\n        return [float(value.ndim), *[float(x) for x in value.shape], *value.astype(float).ravel().tolist()]\n    if isinstance(value, (tuple, list)):\n        out = [float(len(value))]\n        for item in value:\n            out.extend(_tree(item))\n        return out\n    return [float(value)]\ndef _inactive(value):\n    assert np.all(np.asarray(value[0]) == 0)\n    return _tree(value)'
    error_setup = 'def _value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1.0\n    return 0.0'
    return [
        {
            'tol': 1e-9,
            'setup': flat_setup,
            'call': '_tree(apply_oci_neb_policy())',
            'gold_call': '_tree(_oracle_apply_oci_neb_policy())',
        },
        {
            'tol': 1e-9,
            'setup': flat_setup,
            'call': '_inactive(apply_oci_neb_policy(260529,18,9,.19,.12,.72,5))',
            'gold_call': '_inactive(_oracle_apply_oci_neb_policy(260529,18,9,.19,.12,.72,5))',
        },
        {
            'tol': 1e-9,
            'setup': flat_setup,
            'call': '_tree(apply_oci_neb_policy(260777,24,11,.31))',
            'gold_call': '_tree(_oracle_apply_oci_neb_policy(260777,24,11,.31))',
        },
        {
            'tol': 1e-9,
            'setup': error_setup + '\na=(260529,18,9,.7)',
            'call': '_value_error(apply_oci_neb_policy,*a)',
            'gold_call': '_value_error(_oracle_apply_oci_neb_policy,*a)',
        },
    ]
