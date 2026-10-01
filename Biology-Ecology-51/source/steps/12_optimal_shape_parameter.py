"""
Orchestrator: the source's selection of the neighbourhood parameters by minimising the mean predicted deviation. For the same regime, target, k and min_pts as step 10, (i) at the fixed radius return alpha_star, the shape parameter in [alpha_min, alpha_max] at which the step-10 deviation attains its minimum over that range, and the deviation there, alpha_star exact to 1e-9 and the deviation to 1e-9 (a sampled or tolerance-limited search does not meet the contract); the deviation is not smooth in alpha: it jumps where a neighbourhood or a set of averaged states changes and has kinks where a nearest forecast state or the sign of a species difference changes, so the minimum may lie at such a point, and the caller guarantees that the minimum over the range is unique, lies strictly inside the range and is attained at a point where the deviation is continuous (raise ValueError otherwise); (ii) at the fixed shape parameter alpha_ref return the radius selection of step 11 over [radius_min, radius_max]. Return the five values [alpha_star, deviation at alpha_star, minimal deviation over the radius range, infimum of the optimal radii, supremum of the optimal radii]. Call the earlier step functions, including steps 10 and 11, rather than reimplementing them.

The shape parameter sets how sharply the weighting favours the trajectories most similar to the target; the source selects it, together with the radius, by minimising the deviation, and reports optima near the upper end of its search range for real forest data.

Returns
-------
numpy.ndarray of float64 with shape (5,): [optimal shape parameter, deviation at the optimum, minimal deviation over the radius range, infimum and supremum of the optimal radii].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def optimal_shape_parameter(n_traj: int, n_states: int, base: "numpy.ndarray", amplitude: float,
                            growth: "numpy.ndarray", capacity: "numpy.ndarray", interaction: "numpy.ndarray",
                            source_a: int, source_b: int, first_state: int, last_state: int, k: int,
                            radius: float, min_pts: int, alpha_min: float, alpha_max: float,
                            alpha_ref: float, radius_min: float, radius_max: float) -> "numpy.ndarray":
    """Orchestrator: the source's selection of the neighbourhood parameters by minimising the mean predicted deviation. For the same regime, target, k and min_pts as step 10, (i) at the fixed radius return alpha_star, the shape parameter in [alpha_min, alpha_max] at which the step-10 deviation attains its minimum over that range, and the deviation there, alpha_star exact to 1e-9 and the deviation to 1e-9 (a sampled or tolerance-limited search does not meet the contract); the deviation is not smooth in alpha: it jumps where a neighbourhood or a set of averaged states changes and has kinks where a nearest forecast state or the sign of a species difference changes, so the minimum may lie at such a point, and the caller guarantees that the minimum over the range is unique, lies strictly inside the range and is attained at a point where the deviation is continuous (raise ValueError otherwise); (ii) at the fixed shape parameter alpha_ref return the radius selection of step 11 over [radius_min, radius_max]. Return the five values [alpha_star, deviation at alpha_star, minimal deviation over the radius range, infimum of the optimal radii, supremum of the optimal radii]. Call the earlier step functions, including steps 10 and 11, rather than reimplementing them.

    Parameters
    ----------
    n_traj : int
        Number of regime trajectories, at least 2.
    n_states : int
        Number of states per trajectory, at least 2.
    base : numpy.ndarray
        One-dimensional array of the S positive base abundances.
    amplitude : float
        Relative amplitude of the hashed perturbation, in [0, 1).
    growth : numpy.ndarray
        One-dimensional array of the S intrinsic growth rates.
    capacity : numpy.ndarray
        One-dimensional array of the S positive carrying capacities.
    interaction : numpy.ndarray
        Array of shape (S, S) of interaction coefficients.
    source_a : int
        Index (1-based) of the first regime trajectory whose initial community is averaged.
    source_b : int
        Index (1-based) of the second regime trajectory whose initial community is averaged.
    first_state : int
        First observed state (1-based) of the target's Ricker run.
    last_state : int
        Last observed state (1-based) of the target's Ricker run, greater than first_state.
    k : int
        Number of nearest states of the neighbourhood.
    radius : float
        Positive dissimilarity threshold of the neighbourhood, fixed during the search.
    min_pts : int
        Minimum number of states required to form a predicted state.
    alpha_min : float
        Lower end of the shape-parameter range, positive.
    alpha_max : float
        Upper end of the shape-parameter range, greater than alpha_min.
    alpha_ref : float
        Positive shape parameter at which the radius selection of step 11 is run.
    radius_min : float
        Lower end of the radius domain of the radius selection, positive.
    radius_max : float
        Upper end of the radius domain, greater than radius_min.

    Returns
    -------
    result : numpy.ndarray
        Array of five float64 values: the optimal shape parameter, the deviation at it, the minimal deviation over the radius range, and the infimum and supremum of the optimal radii.

    Raises
    ------
    ValueError
        If the regime, target or neighbourhood arguments are invalid (as in step 10), alpha_min is not positive or not below alpha_max, alpha_ref is not positive, no weighted trajectory or no predicted state exists, the minimum over the range is not attained strictly inside it at a point of continuity, or the radius selection's inputs are invalid for step 11.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _shifted_sets(neighbours: "numpy.ndarray", trajectory_ids: "numpy.ndarray", state_ids: "numpy.ndarray",
                  direction: int, min_pts: int, n_states: int) -> list:
    """Index sets of the i-th shifted states of the neighbours (the states step 07 averages), one per sweep step, in
    sweep order; the sweep stops when fewer than min_pts states remain."""
    tr = np.asarray(trajectory_ids); st = np.asarray(state_ids); sets = []
    lookup = {(tr[i], st[i]): i for i in range(tr.size)}
    for i in range(1, n_states + 1):
        idx = [lookup[(tr[j], st[j] + direction * i)] for j in neighbours if (tr[j], st[j] + direction * i) in lookup]
        if len(idx) < min_pts:
            break
        sets.append(np.array(idx, dtype=int))
    return sets


def _oracle_optimal_shape_parameter(n_traj: int, n_states: int, base: "numpy.ndarray", amplitude: float,
                                    growth: "numpy.ndarray", capacity: "numpy.ndarray", interaction: "numpy.ndarray",
                                    source_a: int, source_b: int, first_state: int, last_state: int, k: int,
                                    radius: float, min_pts: int, alpha_min: float, alpha_max: float,
                                    alpha_ref: float, radius_min: float, radius_max: float) -> "numpy.ndarray":
    """ORCHESTRATOR: the source's selection of the shape parameter by minimising the mean predicted deviation at a fixed
    radius. The deviation is a piecewise-smooth function of alpha: inside a piece every neighbourhood, every set of
    averaged states and every nearest-forecast-state choice of Eq 3 is fixed, and the deviation is an explicit function
    of the exponential trajectory weights; the derivative on a piece is d MPD / d alpha = -(1 / (2 n_X m_X)) sum over
    (i, j) of sum_s sign(x_is - z*_s) dz*_s / d alpha with dz / d alpha = -Cov_w(d_T / d_max, states) over the states
    averaged into z* (the weighted covariance between the scaled trajectory dissimilarity and the state variables); the
    sign pattern is part of the piece structure, so kinks of the deviation are piece boundaries. The
    pieces are located by scanning alpha and bisecting the structural changes (steps 05-09 rebuilt at every probe), the
    stationary points of every piece are bracketed and bisected on the exact derivative, the one-sided limits at the
    piece ends are recorded; the caller guarantees that the minimum over [alpha_min, alpha_max] is unique and attained
    strictly inside the range at a point of continuity, a stationary point or a kink (ValueError otherwise). The radius
    selection of step 11 is run at alpha_ref on [radius_min, radius_max] and the result is [alpha*, deviation at alpha*,
    minimal deviation over the radius range, infimum and supremum of the optimal radii]."""
    if int(n_traj) != n_traj or n_traj < 2 or int(n_states) != n_states or n_states < 2:
        raise ValueError("n_traj must be at least 2 and n_states at least 2")
    if int(source_a) != source_a or int(source_b) != source_b or not (1 <= source_a <= n_traj and 1 <= source_b <= n_traj):
        raise ValueError("source_a and source_b must index regime trajectories (1-based)")
    if int(first_state) != first_state or int(last_state) != last_state or not (1 <= first_state < last_state <= n_states):
        raise ValueError("first_state and last_state must satisfy 1 <= first_state < last_state <= n_states")
    lo, hi = float(alpha_min), float(alpha_max)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0.0 or hi <= lo:
        raise ValueError("alpha_min and alpha_max must satisfy 0 < alpha_min < alpha_max")
    if int(k) != k or k < 1 or int(min_pts) != min_pts or min_pts < 1 or not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("k and min_pts must be positive integers and radius positive")
    b = np.asarray(base, dtype=np.float64)
    S = b.size
    X = _oracle_regime_states(n_traj, n_states, b, amplitude, growth, capacity, interaction)
    tr = np.repeat(np.arange(1, int(n_traj) + 1), int(n_states))
    st = np.tile(np.arange(1, int(n_states) + 1), int(n_traj))
    s_idx = np.arange(1, S + 1)
    inits = []
    for j in (int(source_a), int(source_b)):
        h = (1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001
        inits.append(b * (1.0 + float(amplitude) * (2.0 * h / 1000.0 - 1.0)))
    N = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), growth, capacity, interaction, int(n_states))
    labels = np.arange(int(first_state), int(last_state) + 1)
    target = N[labels - 1] / N[labels - 1].sum(axis=1, keepdims=True)
    Xa = np.vstack([X, target])
    tra = np.concatenate([tr, np.zeros(labels.size, dtype=int)])
    sta = np.concatenate([st, labels])
    D = _oracle_bray_curtis_matrix(Xa)
    t1 = X.shape[0]; t2 = Xa.shape[0] - 1
    tid = np.arange(t1, t2 + 1)
    n_reg = X.shape[0]
    trr = np.concatenate([tr, [0]])
    own = set(labels.astype(float).tolist())
    # alpha-independent part, built once through the step functions: weighted set, scaled dissimilarities, the target's
    # neighbourhoods (the weight filter only asks which trajectories carry a weight) and the sweep index sets
    w_ref = _oracle_trajectory_weights(D, tra, sta, 0, k, radius, 0.5 * (lo + hi))
    weighted = list(dict.fromkeys(tra[w_ref > 0].tolist()))
    if not weighted:
        raise ValueError("no weighted trajectory")
    d_T = {lab: _oracle_trajectory_dissimilarity(D, tid, np.where(tra == lab)[0]) for lab in weighted}
    d_max = max(float(radius), max(d_T.values()))
    delta = np.zeros(tra.size)
    for lab in weighted:
        delta[tra == lab] = d_T[lab] / d_max
    if not np.allclose(w_ref[delta > 0], np.exp(-0.5 * (lo + hi) * delta[delta > 0]), rtol=0, atol=1e-12) or np.any(w_ref[delta == 0] != 0.0):
        raise ValueError("the weights do not follow the exponential of the scaled trajectory dissimilarities")
    nb_back = _oracle_neighbour_states(D, tra, t1, k, radius, w_ref)
    nb_forw = _oracle_neighbour_states(D, tra, t2, k, radius, w_ref)
    sets_back = _shifted_sets(nb_back, tra, sta, -1, min_pts, int(n_states))
    sets_forw = _shifted_sets(nb_forw, tra, sta, 1, min_pts, int(n_states))
    pred = [(float(labels[0] - i), idx) for i, idx in enumerate(sets_back, 1)][::-1] + [(float(labels[-1] + i), idx) for i, idx in enumerate(sets_forw, 1)]
    if not pred:
        raise ValueError("no predicted state")
    back = _oracle_sweep_states(Xa, tra, sta, nb_back, w_ref, min_pts, -1)
    forw = _oracle_sweep_states(Xa, tra, sta, nb_forw, w_ref, min_pts, 1)
    P_ref = _oracle_predicted_trajectory(target, labels, back, forw)
    rows = {float(r[0]): r[1:] for r in P_ref if float(r[0]) not in own}
    for lab, idx in pred:
        ww = w_ref[idx]; y = (ww[:, None] * Xa[idx]).sum(axis=0) / ww.sum()
        if lab not in rows or not np.allclose(rows[lab], y, rtol=0, atol=1e-12):
            raise ValueError("the sweep index sets do not reproduce the predicted states")
    wr_mask = np.concatenate([w_ref[:n_reg] > 0, [False]])
    wr_vec = np.where(wr_mask, 1.0, 0.0)
    D_re = np.zeros((n_reg + 1, n_reg + 1)); D_re[:n_reg, :n_reg] = D[:n_reg, :n_reg]
    # step 03 on one stacked probe fixes the convention the fast path reproduces
    y0 = rows[pred[0][0]]
    if not np.allclose(_oracle_bray_curtis_matrix(np.vstack([X, y0[None, :]]))[n_reg, :n_reg], np.abs(X - y0).sum(axis=1) / (X + y0).sum(axis=1), rtol=0, atol=1e-15):
        raise ValueError("the Bray-Curtis rule does not reproduce")

    def _structure(alpha):
        """The alpha-dependent structure: the re-forecast neighbourhood and sweep index sets of every predicted state."""
        w = np.where(delta > 0, np.exp(-alpha * delta), 0.0)
        refc = []
        for lab, idx in pred:
            ww = w[idx]; y = (ww[:, None] * Xa[idx]).sum(axis=0) / ww.sum()
            dvec = np.abs(X - y).sum(axis=1) / (X + y).sum(axis=1)     # Bray-Curtis of y to the regime states (step 03's rule)
            Dr = D_re.copy(); Dr[n_reg, :n_reg] = dvec; Dr[:n_reg, n_reg] = dvec
            nb = _oracle_neighbour_states(Dr, trr, n_reg, k, radius, wr_vec)
            zsets = _shifted_sets(nb, tr, st, -1, min_pts, int(n_states)) + _shifted_sets(nb, tr, st, 1, min_pts, int(n_states))
            refc.append((idx, tuple(int(v) for v in nb), zsets))
        return w, delta, pred, refc

    def _evaluate(alpha, struct, fixed=None):
        """Deviation and its alpha-derivative at alpha on the piece described by struct (weights recomputed for this
        alpha); with fixed = the (nearest-state choice, sign pattern) list of a probe inside the piece, the piece's own
        formula is continued analytically, so that its derivative is continuous on the piece."""
        w0, delta, pred, refc = struct
        w = np.zeros(tra.size)
        m = delta > 0
        w[m] = np.exp(-alpha * delta[m])                                # exp(-alpha d_T / d_max), same weighted set
        def _wmean(idx, Xs):
            ww = w[idx]; W = ww.sum(); z = (ww[:, None] * Xs[idx]).sum(axis=0) / W
            dz = -((ww * delta[idx])[:, None] * Xs[idx]).sum(axis=0) / W + z * (ww * delta[idx]).sum() / W
            return z, dz
        devs = []; ders = []; choices = []; q = 0
        for (idx_y, nb, zsets) in refc:
            y, dy = _wmean(idx_y, Xa)
            cand = [(y, dy)] + [_wmean(zi, X) for zi in zsets]
            for xi in target:
                if fixed is None:
                    bcs = [np.abs(xi - z).sum() / (xi + z).sum() for z, _ in cand]
                    a = int(np.argmin(bcs)); z, dz = cand[a]; sg = np.sign(xi - z)
                else:
                    a, sgt = fixed[q]; z, dz = cand[a]; sg = np.array(sgt, dtype=float)
                q += 1
                num = (sg * (xi - z)).sum(); den = (xi + z).sum()
                devs.append(num / den)
                ders.append((-(sg * dz).sum() * den - num * dz.sum()) / (den * den))
                choices.append((a, tuple(int(v) for v in sg)))
        return float(np.mean(devs)), float(np.mean(ders)), tuple(choices)

    def _signature(alpha):
        struct = _structure(alpha); v, g, choices = _evaluate(alpha, struct)
        sig = (tuple((nb, tuple(tuple(int(u) for u in zi) for zi in zs)) for _, nb, zs in struct[3]), choices)
        return sig, (struct, choices), v, g

    v0 = _evaluate(0.5 * (lo + hi), _structure(0.5 * (lo + hi)))[0]
    if abs(v0 - _oracle_mean_predicted_deviation(Xa, tra, sta, 0, k, radius, min_pts, w_ref, P_ref)) > 1e-12:
        raise ValueError("the piecewise evaluation does not reproduce the deviation of step 09")
    # 1. pieces: scan alpha, then split every scan interval whose ends differ in structure, recursively, until each
    #    sub-interval holds a single structural change (located to 1e-8) - kinks (sign patterns), nearest-state choices
    #    and re-forecast memberships all count
    step = 0.05                                                         # scan step; every change is then bisected
    grid = [lo + i * step for i in range(int(np.floor((hi - lo) / step)) + 1)]
    if grid[-1] < hi - 1e-12:
        grid.append(hi)
    probes = {a: _signature(a) for a in grid}
    def _split(a, c, sig_a, sig_c, out):
        if sig_a == sig_c:
            return
        if c - a <= 1e-6:
            out.append(0.5 * (a + c)); return
        m = 0.5 * (a + c); sm = _signature(m); probes[m] = sm
        _split(a, m, sig_a, sm[0], out); _split(m, c, sm[0], sig_c, out)
    edges = [lo]
    for i in range(len(grid) - 1):
        found = []
        _split(grid[i], grid[i + 1], probes[grid[i]][0], probes[grid[i + 1]][0], found)
        edges.extend(found)
    edges.append(hi)
    # the structure of each piece: a probe strictly inside it
    structs = []
    for p in range(len(edges) - 1):
        m = 0.5 * (edges[p] + edges[p + 1])
        if m not in probes:
            probes[m] = _signature(m)
        structs.append(probes[m][1])
    # 2. candidates: the stationary points inside each piece (bisection on the exact derivative of the piece's formula)
    #    and the one-sided limits at the piece ends
    cands = []
    for p in range(len(edges) - 1):
        a, c = edges[p], edges[p + 1]; struct, fixed = structs[p]
        eps_in = 1e-9 * max(1.0, abs(c - a))
        xs = np.linspace(a + eps_in, c - eps_in, 17)
        vals = [_evaluate(x, struct, fixed) for x in xs]
        cands.append(("left-end", p, a, vals[0][0])); cands.append(("right-end", p, c, vals[-1][0]))
        for i in range(len(xs) - 1):
            if vals[i][1] < 0.0 <= vals[i + 1][1]:
                x1, x2 = xs[i], xs[i + 1]
                for _ in range(80):
                    mid = 0.5 * (x1 + x2); gm = _evaluate(mid, struct, fixed)[1]
                    if gm < 0.0:
                        x1 = mid
                    else:
                        x2 = mid
                    if x2 - x1 < 1e-14:
                        break
                xm = 0.5 * (x1 + x2); cands.append(("stationary", p, xm, _evaluate(xm, struct, fixed)[0]))
    best = min(cands, key=lambda t: t[3])
    if best[0] == "stationary":
        alpha_star, value = float(best[2]), best[3]
    else:
        # a piece end: the minimum is attained there only if the deviation is continuous across the boundary (a kink)
        p = best[1]; q = p + 1 if best[0] == "right-end" else p - 1
        if q < 0 or q >= len(structs):
            raise ValueError("the minimum over [alpha_min, alpha_max] lies at an end of the range")
        other = [t for t in cands if t[1] == q and t[0] == ("left-end" if best[0] == "right-end" else "right-end")][0]
        if abs(other[3] - best[3]) > 1e-9:
            raise ValueError("the minimum over [alpha_min, alpha_max] lies at a discontinuity of the deviation")
        # refine the kink location: bisection on the structure between the two pieces
        left, right = (p, q) if best[0] == "right-end" else (q, p)
        a, c = edges[left] - 0.0, edges[left + 1] + 0.0
        a, c = max(edges[left], a - 1e-6), min(edges[left + 2] if left + 2 < len(edges) else hi, c + 1e-6)
        sig_left = _signature(0.5 * (edges[left] + edges[left + 1]))[0]
        lo_b, hi_b = edges[left + 1] - 1e-6, edges[left + 1] + 1e-6
        while hi_b - lo_b > 4e-16 * max(1.0, abs(hi_b)):
            m = 0.5 * (lo_b + hi_b)
            if _signature(m)[0] == sig_left:
                lo_b = m
            else:
                hi_b = m
        alpha_star = 0.5 * (lo_b + hi_b)
        value = min(_evaluate(alpha_star, structs[left][0], structs[left][1])[0], _evaluate(alpha_star, structs[left + 1][0], structs[left + 1][1])[0])
    check = _oracle_forecast_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a,
                                       source_b, first_state, last_state, k, radius, min_pts, alpha_star)
    if abs(check - value) > 1e-9:
        raise ValueError("the optimum does not reproduce through the single-parameter forecast")
    if not np.isfinite(alpha_ref) or alpha_ref <= 0.0:
        raise ValueError("alpha_ref must be positive")
    rad = _oracle_optimal_radius_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a,
                                           source_b, first_state, last_state, k, radius_min, radius_max, min_pts, alpha_ref)
    return np.array([alpha_star, check, rad[0], rad[1], rad[2]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\ngrowth, capacity, interaction = r, K, A\nn_traj, n_states, amplitude, source_a, source_b, first_state, last_state = 36, 8, 0.35, 13, 31, 4, 6\nk, radius, min_pts, alpha_min, alpha_max = 25, 0.065, 3, 1.0, 5.0\nalpha_ref, radius_min, radius_max = 3.0, 0.05, 0.07\n",
            "call": "optimal_shape_parameter(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius, min_pts, alpha_min, alpha_max, alpha_ref, radius_min, radius_max)",
            "gold_call": "_oracle_optimal_shape_parameter(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius, min_pts, alpha_min, alpha_max, alpha_ref, radius_min, radius_max)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\ngrowth, capacity, interaction = r, K, A\nn_traj, n_states, amplitude, source_a, source_b, first_state, last_state = 36, 8, 0.35, 9, 24, 4, 6\nk, radius, min_pts, alpha_min, alpha_max = 25, 0.065, 3, 1.0, 5.0\nalpha_ref, radius_min, radius_max = 3.0, 0.05, 0.068\n",
            "call": "optimal_shape_parameter(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius, min_pts, alpha_min, alpha_max, alpha_ref, radius_min, radius_max)",
            "gold_call": "_oracle_optimal_shape_parameter(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius, min_pts, alpha_min, alpha_max, alpha_ref, radius_min, radius_max)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\ngrowth, capacity, interaction = r, K, A\nn_traj, n_states, amplitude, source_a, source_b, first_state, last_state = 36, 8, 0.35, 5, 14, 2, 5\nk, radius, min_pts, alpha_min, alpha_max = 25, 0.065, 3, 1.0, 5.0\nalpha_ref, radius_min, radius_max = 3.0, 0.05, 0.068\n",
            "call": "optimal_shape_parameter(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius, min_pts, alpha_min, alpha_max, alpha_ref, radius_min, radius_max)",
            "gold_call": "_oracle_optimal_shape_parameter(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius, min_pts, alpha_min, alpha_max, alpha_ref, radius_min, radius_max)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\ngrowth, capacity, interaction = r, K, A\nn_traj, n_states, amplitude, source_a, source_b, first_state, last_state = 36, 8, 0.35, 13, 31, 4, 6\nk, radius, min_pts, alpha_min, alpha_max = 25, 0.065, 3, 5.0, 1.0\nalpha_ref, radius_min, radius_max = 3.0, 0.05, 0.07\ndef run_model():\n    try:\n        optimal_shape_parameter(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius, min_pts, alpha_min, alpha_max, alpha_ref, radius_min, radius_max)\n        return 0\n    except ValueError:\n        return 1\ndef _fx_safe(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "_fx_safe(lambda: _oracle_optimal_shape_parameter(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius, min_pts, alpha_min, alpha_max, alpha_ref, radius_min, radius_max))",
        },
    ]
