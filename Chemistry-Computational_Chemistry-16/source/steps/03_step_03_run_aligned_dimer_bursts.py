"""
Call public Steps 1 and 2 and require finite 0<step_size<=0.5. At every climbing image form the exact Hessian of Step 1, take its lowest eigenvector with sign aligned to tau, and mix it with the normalized cross(tau,[0,0,1]) at angle theta selected from [0.04,0.18,0.34,0.55,1.24,1.43] by m mod 6. Normalize this dimer axis. Record alpha=abs(axis dot tau) and curvature=axis.T@H@axis. Reflect the true force through the dimer axis, move by step_size*reflected/(1+norm(reflected)), recompute the reflected candidate force, and multiply its norm by 0.88+0.055*(m mod 5). Return exactly eleven items in this order: axes (n_mechanisms,3), alpha (n_mechanisms,), curvature (n_mechanisms,), candidate positions (n_mechanisms,3), Fnew (n_mechanisms,), FCI (n_mechanisms,), latch counts (n_mechanisms,), F0 (n_mechanisms,), panel_checksum scalar, control_checksum scalar, and dimer_checksum scalar. Define dimer_checksum by the displayed alpha, curvature, candidate, and Fnew arrays with weights [3+m/19,2+m/17,0.6+(k mod 23)/29,5+m/31].

This step supplies the geometric evidence used by the source-derived acceptance, failure, and restoration branches. It does not choose those branches itself.

Returns
-------
tuple : eleven items: axes, alpha, curvature, candidates, Fnew, FCI, stable, F0, and three checksums.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_aligned_dimer_bursts(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9, step_size: float = 0.19) -> tuple:
    """Return eleven ordered dimer arrays/scalars used by the policy stage.

    Returns:
        tuple: axes, alpha, curvature, candidates, Fnew, FCI, stable,
        F0, panel_checksum, control_checksum, dimer_checksum.

    Raises:
        ValueError: If step_size or an upstream panel argument is inadmissible.
    """
    return None, None, None, None, None, None, None, None, None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _hessian(r, phase):
    x, y, _ = np.asarray(r, dtype=float)
    q = 1.7 * x + phase
    p = 1.3 * y - 0.4 * phase
    hxx = 3.0 * x * x - 1.0 - 0.05202 * np.sin(q) * np.cos(p)
    hyy = 0.68 - 0.03042 * np.sin(q) * np.cos(p)
    hxy = 0.075 - 0.03978 * np.cos(q) * np.sin(p)
    return np.array([[hxx, hxy, 0.0], [hxy, hyy, -0.052], [0.0, -0.052, 0.42]])

def _oracle_run_aligned_dimer_bursts(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9, step_size: float = 0.19) -> tuple:
    if not np.isfinite(step_size) or step_size <= 0.0 or step_size > 0.5:
        raise ValueError("step_size must be finite and in (0, 0.5]")
    paths, _, forces, _, _ = _oracle_generate_oci_neb_panel(seed, n_mechanisms, n_images)
    indices, tangents, _, fci, stable, baselines, panel_checksum, control_checksum = _oracle_compute_climbing_controls(seed, n_mechanisms, n_images)
    axes = np.empty_like(tangents)
    alpha = np.empty(n_mechanisms, dtype=float)
    curvature = np.empty(n_mechanisms, dtype=float)
    candidates = np.empty_like(tangents)
    fnew = np.empty(n_mechanisms, dtype=float)
    for m, k in enumerate(indices):
        phase = 0.17 * m + 0.003 * (seed % 101)
        point = paths[m, k]
        h = _hessian(point, phase)
        _, vecs = np.linalg.eigh(h)
        vmin = vecs[:, 0]
        if np.dot(vmin, tangents[m]) < 0.0:
            vmin = -vmin
        theta = (0.04, 0.18, 0.34, 0.55, 1.24, 1.43)[m % 6]
        side = np.cross(tangents[m], np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(side) < 1e-12:
            side = np.array([0.0, 1.0, 0.0])
        side /= np.linalg.norm(side)
        raw = np.cos(theta) * vmin + np.sin(theta) * side
        axis = raw / np.linalg.norm(raw)
        axes[m] = axis
        alpha[m] = abs(np.dot(axis, tangents[m]))
        curvature[m] = float(axis @ h @ axis)
        true_force = forces[m, k]
        reflected = true_force - 2.0 * np.dot(true_force, axis) * axis
        candidate = point + step_size * reflected / (1.0 + np.linalg.norm(reflected))
        candidates[m] = candidate
        candidate_force = -_gradient(candidate, phase)
        candidate_climb = candidate_force - 2.0 * np.dot(candidate_force, axis) * axis
        fnew[m] = np.linalg.norm(candidate_climb) * (0.88 + 0.055 * (m % 5))
    checksum = float(np.dot(alpha, 3.0 + np.arange(n_mechanisms) / 19.0)
                     + np.dot(curvature, 2.0 + np.arange(n_mechanisms) / 17.0)
                     + np.dot(candidates.ravel(), 0.6 + (np.arange(candidates.size) % 23) / 29.0)
                     + np.dot(fnew, 5.0 + np.arange(n_mechanisms) / 31.0))
    return axes, alpha, curvature, candidates, fnew, fci, stable, baselines, panel_checksum, control_checksum, checksum

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    flat_setup = 'np=__import__("numpy")\ndef _tree(value):\n    if isinstance(value, np.ndarray):\n        return [float(value.ndim), *[float(x) for x in value.shape], *value.astype(float).ravel().tolist()]\n    if isinstance(value, (tuple, list)):\n        out = [float(len(value))]\n        for item in value:\n            out.extend(_tree(item))\n        return out\n    return [float(value)]'
    error_setup = 'def _value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1.0\n    return 0.0'
    return [
      {'tol':1e-9,'setup':flat_setup,'call':'_tree(run_aligned_dimer_bursts())','gold_call':'_tree(_oracle_run_aligned_dimer_bursts())'},
      {'tol':1e-9,'setup':flat_setup,'call':'_tree(run_aligned_dimer_bursts(260601,8,7,.11))','gold_call':'_tree(_oracle_run_aligned_dimer_bursts(260601,8,7,.11))'},
      {'tol':1e-9,'setup':flat_setup,'call':'_tree(run_aligned_dimer_bursts(260777,24,11,.31))','gold_call':'_tree(_oracle_run_aligned_dimer_bursts(260777,24,11,.31))'},
      {'tol':1e-9,'setup':error_setup+'\na=(260529,18,9,.0)','call':'_value_error(run_aligned_dimer_bursts,*a)','gold_call':'_value_error(_oracle_run_aligned_dimer_bursts,*a)'},
    ]
