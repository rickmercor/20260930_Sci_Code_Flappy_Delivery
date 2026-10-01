"""
Calibrate the PETRA-EDR forecast.

Treat every row of calibration_targets as a separate target state and average its MPD with the other rows for the same candidate. Rank all flattened reference states by Bray-Curtis dissimilarity, breaking ties by trajectory and then time; keep at most k states with distance strictly below epsilon. Give every retained state the weight obtained from the minimum distance of its trajectory, with q equal to that trajectory distance divided by max(epsilon, the largest retained trajectory distance), and use the named PETRA kernel. Sweep only within the retained state's own trajectory for at most max_steps. In each secondary prediction used for MPD, sweep in the direction back toward the original target for abs(offset) steps, take the smallest Bray-Curtis distance from the target to that reconstructed path, and average the usable offsets. Return the earliest minimum candidate index followed by every candidate's mean MPD.

Returns
-------
return a float64 array of shape (6,): the selected zero-based candidate followed by the five MPD values.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def petra_calibration(reference: "np.ndarray", calibration_targets: "np.ndarray", candidates: list[tuple[int, float, int, str, float]], max_steps: int) -> "np.ndarray":
    """Calibrate the PETRA forecast.
 
    Returns
    -------
    A float64 vector containing the selected zero-based candidate followed by every MPD score.
 
    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
 
def _bc_rows(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    num = np.abs(x[:, None, :] - y[None, :, :]).sum(axis=2)
    den = (x[:, None, :] + y[None, :, :]).sum(axis=2)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0.0)
 
 
def _rank(reference, target, k, eps):
    r = np.asarray(reference, dtype=np.float64)
    flat = r.reshape(-1, r.shape[2])
    d = _bc_rows(flat, np.asarray(target, dtype=np.float64)[None, :])[:, 0]
    ids = np.column_stack(np.unravel_index(np.arange(flat.shape[0]), r.shape[:2]))
    order = np.lexsort((ids[:, 1], ids[:, 0], d))
    order = order[d[order] < eps][:k]
    return ids[order], d[order]
 
 
def _kernel_weights(reference, target, idx, kernel, alpha, eps):
    r = np.asarray(reference, dtype=np.float64)
    d = _bc_rows(r.reshape(-1, r.shape[2]), np.asarray(target)[None, :])[:, 0].reshape(r.shape[:2])
    td = np.min(d, axis=1)[idx[:, 0]]
    q = td / max(float(eps), float(td.max()))
    if kernel == "linear":
        return 1.0 - q
    if kernel == "power":
        return 1.0 - q ** alpha
    if kernel == "exponential":
        return np.exp(-alpha * q)
    if kernel == "Gaussian":
        return np.exp(-(q ** 2.0))
    if kernel == "hyperbolic":
        return ((1.0 + q) ** (-alpha) - 2.0 ** (-alpha)) / (1.0 - 2.0 ** (-alpha))
    if kernel == "spherical":
        return 1.0 - 1.5 * q + 0.5 * q ** 3.0
    raise ValueError("unknown kernel")
 
 
def _sweep(reference, idx, weights, direction, min_pts, max_steps):
    r = np.asarray(reference, dtype=np.float64)
    out = []
    for shift in range(1, max_steps + 1):
        moved = idx[:, 1] + direction * shift
        valid = (moved >= 0) & (moved < r.shape[1]) & (weights > 0.0)
        if np.count_nonzero(valid) < min_pts:
            break
        w = weights[valid]
        out.append((r[idx[valid, 0], moved[valid]] * w[:, None]).sum(axis=0) / w.sum())
    return np.asarray(out, dtype=np.float64).reshape(-1, r.shape[2])
 
 
def _forecast(reference, target, candidate, max_steps):
    k, eps, min_pts, kernel, alpha = candidate
    idx, _ = _rank(reference, target, int(k), float(eps))
    if len(idx) < min_pts:
        raise ValueError("insufficient neighbours")
    w = _kernel_weights(reference, target, idx, kernel, float(alpha), float(eps))
    back = _sweep(reference, idx, w, -1, int(min_pts), int(max_steps))
    fwd = _sweep(reference, idx, w, 1, int(min_pts), int(max_steps))
    offsets = np.arange(-len(back), len(fwd) + 1, dtype=np.float64)
    states = np.vstack([back[::-1], np.asarray(target, dtype=np.float64)[None, :], fwd])
    return offsets, states, idx, w
 
 
def _mpd(reference, target, candidate, max_steps):
    offsets, states, _, _ = _forecast(reference, target, candidate, max_steps)
    values = []
    for off, state in zip(offsets, states):
        if off == 0:
            continue
        k, eps, min_pts, kernel, alpha = candidate
        idx, _ = _rank(reference, state, int(k), float(eps))
        if len(idx) < min_pts:
            continue
        w = _kernel_weights(reference, state, idx, kernel, float(alpha), float(eps))
        path = _sweep(reference, idx, w, -1 if off > 0 else 1, int(min_pts), abs(int(off)))
        if len(path):
            repath = np.vstack([state[None, :], path])
            values.append(float(_bc_rows(np.asarray(target)[None, :], repath).min()))
    if not values:
        raise ValueError("candidate has no usable re-prediction")
    return float(np.mean(values))
 
 
def _oracle_petra_calibration(reference: "np.ndarray", calibration_targets: "np.ndarray", candidates: list[tuple[int, float, int, str, float]], max_steps: int) -> "np.ndarray":
    r = np.asarray(reference, dtype=np.float64)
    x = np.asarray(calibration_targets, dtype=np.float64)
    if r.ndim != 3 or x.ndim != 2 or x.shape[1] != r.shape[2] or len(candidates) < 2:
        raise ValueError("unaligned PETRA inputs")
    scores = np.array([np.mean([_mpd(r, row, c, int(max_steps)) for row in x]) for c in candidates], dtype=np.float64)
    return np.concatenate([[float(np.argmin(scores))], scores]).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': "import numpy as np\nreference=np.array([[[62.0, 26.0, 8.0, 3.0, 1.0], [56.0, 28.0, 10.0, 4.0, 2.0], [48.0, 31.0, 12.0, 6.0, 3.0], [38.0, 32.0, 17.0, 8.0, 5.0], [28.0, 30.0, 22.0, 12.0, 8.0], [21.0, 25.0, 25.0, 19.0, 10.0], [17.0, 20.0, 26.0, 23.0, 14.0], [13.0, 18.0, 23.0, 26.0, 20.0]], [[65.0, 24.0, 7.0, 3.0, 1.0], [57.0, 28.0, 9.0, 4.0, 2.0], [49.0, 31.0, 12.0, 5.0, 3.0], [39.0, 31.0, 17.0, 8.0, 5.0], [29.0, 30.0, 21.0, 12.0, 8.0], [21.0, 26.0, 25.0, 17.0, 11.0], [16.0, 21.0, 26.0, 22.0, 15.0], [12.0, 18.0, 24.0, 26.0, 20.0]], [[60.0, 28.0, 8.0, 3.0, 1.0], [53.0, 30.0, 11.0, 4.0, 2.0], [46.0, 32.0, 13.0, 6.0, 3.0], [37.0, 31.0, 18.0, 9.0, 5.0], [29.0, 28.0, 22.0, 13.0, 8.0], [22.0, 24.0, 24.0, 19.0, 11.0], [18.0, 20.0, 25.0, 23.0, 15.0], [14.0, 18.0, 23.0, 25.0, 20.0]], [[66.0, 23.0, 7.0, 3.0, 1.0], [58.0, 27.0, 9.0, 4.0, 2.0], [49.0, 30.0, 12.0, 6.0, 3.0], [39.0, 30.0, 17.0, 9.0, 5.0], [29.0, 29.0, 20.0, 14.0, 8.0], [21.0, 25.0, 24.0, 19.0, 11.0], [16.0, 21.0, 25.0, 22.0, 16.0], [11.0, 19.0, 24.0, 25.0, 21.0]], [[59.0, 27.0, 10.0, 3.0, 1.0], [53.0, 29.0, 12.0, 4.0, 2.0], [45.0, 32.0, 14.0, 6.0, 3.0], [37.0, 32.0, 17.0, 9.0, 5.0], [29.0, 29.0, 22.0, 12.0, 8.0], [23.0, 24.0, 24.0, 18.0, 11.0], [18.0, 20.0, 26.0, 21.0, 15.0], [15.0, 16.0, 24.0, 25.0, 20.0]], [[63.0, 27.0, 6.0, 3.0, 1.0], [56.0, 29.0, 9.0, 4.0, 2.0], [47.0, 32.0, 12.0, 6.0, 3.0], [38.0, 32.0, 17.0, 8.0, 5.0], [28.0, 30.0, 21.0, 13.0, 8.0], [21.0, 25.0, 25.0, 18.0, 11.0], [17.0, 20.0, 26.0, 22.0, 15.0], [13.0, 17.0, 24.0, 26.0, 20.0]]], dtype=float)\ncalibration_targets=np.array([[53.0, 30.0, 10.0, 5.0, 2.0], [36.0, 31.0, 18.0, 10.0, 5.0], [20.0, 24.0, 25.0, 19.0, 12.0]], dtype=float)\ndisturbed=np.array([[56.0, 28.0, 10.0, 4.0, 2.0], [46.0, 31.0, 14.0, 6.0, 3.0], [18.0, 17.0, 18.0, 25.0, 22.0], [20.0, 20.0, 20.0, 23.0, 17.0], [22.0, 22.0, 21.0, 20.0, 15.0], [20.0, 23.0, 23.0, 19.0, 15.0]], dtype=float)\ncandidates=[(4, 0.42, 3, 'linear', 1.0), (5, 0.42, 3, 'exponential', 2.0), (6, 0.42, 4, 'hyperbolic', 3.0), (7, 0.42, 4, 'spherical', 1.0), (8, 0.42, 5, 'power', 2.0)]\nmax_steps=4\nindices=np.array([1, 2, 4], dtype=int)\nguild_map=np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.25, -0.15, 0.35, -0.2, 0.4]], dtype=float)", 'call': 'petra_calibration(reference,calibration_targets,candidates,max_steps)', 'gold_call': '_oracle_petra_calibration(reference,calibration_targets,candidates,max_steps)'}, {'setup': "import numpy as np\nreference=np.array([[[62.0, 26.0, 8.0, 3.0, 1.0], [56.0, 28.0, 10.0, 4.0, 2.0], [48.0, 31.0, 12.0, 6.0, 3.0], [38.0, 32.0, 17.0, 8.0, 5.0], [28.0, 30.0, 22.0, 12.0, 8.0], [21.0, 25.0, 25.0, 19.0, 10.0], [17.0, 20.0, 26.0, 23.0, 14.0], [13.0, 18.0, 23.0, 26.0, 20.0]], [[65.0, 24.0, 7.0, 3.0, 1.0], [57.0, 28.0, 9.0, 4.0, 2.0], [49.0, 31.0, 12.0, 5.0, 3.0], [39.0, 31.0, 17.0, 8.0, 5.0], [29.0, 30.0, 21.0, 12.0, 8.0], [21.0, 26.0, 25.0, 17.0, 11.0], [16.0, 21.0, 26.0, 22.0, 15.0], [12.0, 18.0, 24.0, 26.0, 20.0]], [[60.0, 28.0, 8.0, 3.0, 1.0], [53.0, 30.0, 11.0, 4.0, 2.0], [46.0, 32.0, 13.0, 6.0, 3.0], [37.0, 31.0, 18.0, 9.0, 5.0], [29.0, 28.0, 22.0, 13.0, 8.0], [22.0, 24.0, 24.0, 19.0, 11.0], [18.0, 20.0, 25.0, 23.0, 15.0], [14.0, 18.0, 23.0, 25.0, 20.0]], [[66.0, 23.0, 7.0, 3.0, 1.0], [58.0, 27.0, 9.0, 4.0, 2.0], [49.0, 30.0, 12.0, 6.0, 3.0], [39.0, 30.0, 17.0, 9.0, 5.0], [29.0, 29.0, 20.0, 14.0, 8.0], [21.0, 25.0, 24.0, 19.0, 11.0], [16.0, 21.0, 25.0, 22.0, 16.0], [11.0, 19.0, 24.0, 25.0, 21.0]], [[59.0, 27.0, 10.0, 3.0, 1.0], [53.0, 29.0, 12.0, 4.0, 2.0], [45.0, 32.0, 14.0, 6.0, 3.0], [37.0, 32.0, 17.0, 9.0, 5.0], [29.0, 29.0, 22.0, 12.0, 8.0], [23.0, 24.0, 24.0, 18.0, 11.0], [18.0, 20.0, 26.0, 21.0, 15.0], [15.0, 16.0, 24.0, 25.0, 20.0]], [[63.0, 27.0, 6.0, 3.0, 1.0], [56.0, 29.0, 9.0, 4.0, 2.0], [47.0, 32.0, 12.0, 6.0, 3.0], [38.0, 32.0, 17.0, 8.0, 5.0], [28.0, 30.0, 21.0, 13.0, 8.0], [21.0, 25.0, 25.0, 18.0, 11.0], [17.0, 20.0, 26.0, 22.0, 15.0], [13.0, 17.0, 24.0, 26.0, 20.0]]], dtype=float)\ncalibration_targets=np.array([[53.0, 30.0, 10.0, 5.0, 2.0], [36.0, 31.0, 18.0, 10.0, 5.0], [20.0, 24.0, 25.0, 19.0, 12.0]], dtype=float)\ndisturbed=np.array([[56.0, 28.0, 10.0, 4.0, 2.0], [46.0, 31.0, 14.0, 6.0, 3.0], [18.0, 17.0, 18.0, 25.0, 22.0], [20.0, 20.0, 20.0, 23.0, 17.0], [22.0, 22.0, 21.0, 20.0, 15.0], [20.0, 23.0, 23.0, 19.0, 15.0]], dtype=float)\ncandidates=[(4, 0.42, 3, 'linear', 1.0), (5, 0.42, 3, 'exponential', 2.0), (6, 0.42, 4, 'hyperbolic', 3.0), (7, 0.42, 4, 'spherical', 1.0), (8, 0.42, 5, 'power', 2.0)]\nmax_steps=4\nindices=np.array([1, 2, 4], dtype=int)\nguild_map=np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.25, -0.15, 0.35, -0.2, 0.4]], dtype=float)\ncalibration_targets=calibration_targets[:1]\ncandidates=candidates[:2]", 'call': 'petra_calibration(reference,calibration_targets,candidates,max_steps)', 'gold_call': '_oracle_petra_calibration(reference,calibration_targets,candidates,max_steps)'}, {'setup': "import numpy as np\nreference=np.ones((2,3,2))\ncalibration_targets=np.ones((1,3))\ncandidates=[(2,.5,1,'linear',1.),(2,.5,1,'power',2.)]\nmax_steps=2\ndef candidate_code():\n    try:\n        petra_calibration(reference,calibration_targets,candidates,max_steps)\n    except ValueError:\n        return 1.0\n    return 0.0\ndef oracle_code():\n    try:\n        _oracle_petra_calibration(reference,calibration_targets,candidates,max_steps)\n    except ValueError:\n        return 1.0\n    return 0.0", 'call': 'candidate_code()', 'gold_call': 'oracle_code()', 'tol': 0.0}]
