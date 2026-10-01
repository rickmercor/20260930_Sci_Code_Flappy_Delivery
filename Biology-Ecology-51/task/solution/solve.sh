#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def ricker_trajectory(initial: "numpy.ndarray", growth: "numpy.ndarray", capacity: "numpy.ndarray",
                              interaction: "numpy.ndarray", n_states: int) -> "numpy.ndarray":
    """Discrete-time Ricker-Lotka-Volterra map N_{t+1,s} = N_{t,s} exp(r_s (1 - (A N_t)_s / K_s)); row t = state t+1."""
    N0 = np.asarray(initial, dtype=np.float64); r = np.asarray(growth, dtype=np.float64)
    K = np.asarray(capacity, dtype=np.float64); A = np.asarray(interaction, dtype=np.float64)
    if N0.ndim != 1 or N0.size < 1:
        raise ValueError("initial must be a non-empty 1-D array")
    S = N0.size
    if r.shape != (S,) or K.shape != (S,) or A.shape != (S, S):
        raise ValueError("growth and capacity must have the length of initial and interaction must be square of that size")
    if not (np.all(np.isfinite(N0)) and np.all(np.isfinite(r)) and np.all(np.isfinite(K)) and np.all(np.isfinite(A))):
        raise ValueError("all inputs must be finite")
    if np.any(N0 < 0.0) or np.any(K <= 0.0):
        raise ValueError("initial abundances must be nonnegative and capacities positive")
    if int(n_states) != n_states or n_states < 1:
        raise ValueError("n_states must be a positive integer")
    out = np.empty((int(n_states), S))
    N = N0.copy()
    out[0] = N
    for t in range(1, int(n_states)):
        N = N * np.exp(r * (1.0 - (A @ N) / K))
        out[t] = N
    return out

import numpy as np


def regime_states(n_traj: int, n_states: int, base: "numpy.ndarray", amplitude: float,
                          growth: "numpy.ndarray", capacity: "numpy.ndarray",
                          interaction: "numpy.ndarray") -> "numpy.ndarray":
    """Relative-abundance states of the synthetic regime: trajectory j starts from base_s (1 + amplitude (2 h/1000 - 1)),
    h = (1103 j + 617 s + 251 j s) mod 1001, runs the Ricker map for n_states states; rows ordered by trajectory then state."""
    b = np.asarray(base, dtype=np.float64)
    if b.ndim != 1 or b.size < 1 or not np.all(np.isfinite(b)) or np.any(b <= 0.0):
        raise ValueError("base must be a non-empty 1-D array of positive finite values")
    if int(n_traj) != n_traj or n_traj < 1 or int(n_states) != n_states or n_states < 1:
        raise ValueError("n_traj and n_states must be positive integers")
    if not np.isfinite(amplitude) or not (0.0 <= amplitude < 1.0):
        raise ValueError("amplitude must lie in [0, 1)")
    S = b.size
    s_idx = np.arange(1, S + 1)
    rows = []
    for j in range(1, int(n_traj) + 1):
        h = (1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001                 # integer hash, exact
        N0 = b * (1.0 + float(amplitude) * (2.0 * h / 1000.0 - 1.0))
        N = ricker_trajectory(N0, growth, capacity, interaction, int(n_states))
        rows.append(N / N.sum(axis=1, keepdims=True))                        # relative abundances
    return np.vstack(rows)

import numpy as np


def bray_curtis_matrix(states: "numpy.ndarray") -> "numpy.ndarray":
    """Pairwise Bray-Curtis dissimilarity sum|x - y| / sum(x + y) between the rows of states."""
    X = np.asarray(states, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError("states must be a 2-D array with at least one row and one column")
    if not np.all(np.isfinite(X)) or np.any(X < 0.0):
        raise ValueError("states must be finite and nonnegative")
    num = np.abs(X[:, None, :] - X[None, :, :]).sum(axis=2)
    den = (X[:, None, :] + X[None, :, :]).sum(axis=2)
    if np.any(den <= 0.0):
        raise ValueError("every pair of rows must have a positive total abundance")
    D = num / den
    np.fill_diagonal(D, 0.0)
    return D

import numpy as np


def _segment_projection(d_ref: float, d1: float, d2: float) -> "numpy.ndarray":
    """Companion-source projection of a point onto a directed segment from the three dissimilarities (segment length
    d_ref, point to start d1, point to end d2): [start-to-projection, projection-to-end, point-to-segment], with the
    triangle-inequality correction (the smallest constant added to the three values that closes the triangle) and the
    nearest-endpoint rule when the projection falls outside the segment."""
    dref, a, b = float(d_ref), float(d1), float(d2)
    k = max(0.0, dref - (a + b), b - (dref + a), a - (b + dref))       # ecotraj: correct the triangle inequality
    a, b, dref = a + k, b + k, dref + k
    a1 = (a * a + dref * dref - b * b) / (2.0 * dref)
    a2 = dref - a1
    sq = a * a - a1 * a1
    h = np.sqrt(sq) if sq >= 0.0 else np.nan
    if np.isnan(h) or a1 < 0.0 or a2 < 0.0:                          # outside the segment: nearest endpoint
        if a < b:
            return np.array([0.0, dref, a])
        return np.array([dref, 0.0, b])
    return np.array([a1, a2, h])


def _segment_pair_distance(D: "numpy.ndarray", s1: int, e1: int, s2: int, e2: int) -> float:
    """Companion-source directed distance between two segments (s1->e1, s2->e2) from the dissimilarity matrix."""
    ps1 = _segment_projection(D[s2, e2], D[s1, s2], D[s1, e2])
    pe1 = _segment_projection(D[s2, e2], D[e1, s2], D[e1, e2])
    ps2 = _segment_projection(D[s1, e1], D[s1, s2], D[e1, s2])
    pe2 = _segment_projection(D[s1, e1], D[s1, e2], D[e1, e2])
    ds1, de1, ds2, de2 = ps1[2], pe1[2], ps2[2], pe2[2]
    if ps1[0] > pe1[0]:                                               # segment 1 runs against segment 2
        de1 = min(D[s1, e1] + ps1[2], D[s1, e1] + pe1[2])
    if ps2[0] > pe2[0]:
        de2 = min(D[s2, e2] + ps2[2], D[s2, e2] + pe2[2])
    return float(max(ds1, de1, ds2, de2))


def trajectory_dissimilarity(dissimilarity: "numpy.ndarray", target_states: "numpy.ndarray",
                                     reference_states: "numpy.ndarray") -> float:
    """Directed segment path dissimilarity from the target trajectory to a reference trajectory (companion source):
    the mean, over the target's segments, of the smallest directed segment distance to any reference segment."""
    D = np.asarray(dissimilarity, dtype=np.float64)
    ta = np.asarray(target_states).astype(int); tb = np.asarray(reference_states).astype(int)
    n = D.shape[0] if D.ndim == 2 else 0
    if D.ndim != 2 or D.shape[1] != n or n < 2 or not np.all(np.isfinite(D)) or np.any(D < 0.0):
        raise ValueError("dissimilarity must be a finite nonnegative square matrix")
    if ta.ndim != 1 or tb.ndim != 1 or ta.size < 2 or tb.size < 2:
        raise ValueError("each trajectory needs at least two states")
    if np.any(ta < 0) or np.any(ta >= n) or np.any(tb < 0) or np.any(tb >= n):
        raise ValueError("trajectory indices must be valid state indices")
    seg_a = [(int(ta[i]), int(ta[i + 1])) for i in range(ta.size - 1)]
    seg_b = [(int(tb[i]), int(tb[i + 1])) for i in range(tb.size - 1)]
    for s_, e_ in seg_a + seg_b:
        if D[s_, e_] <= 0.0:
            raise ValueError("segments must have positive length")
    M = np.array([[_segment_pair_distance(D, s1, e1, s2, e2) for (s2, e2) in seg_b] for (s1, e1) in seg_a])
    return float(np.mean(M.min(axis=1)))

import numpy as np


def trajectory_weights(dissimilarity: "numpy.ndarray", trajectory_ids: "numpy.ndarray",
                               state_ids: "numpy.ndarray", target_label: int, k: int, radius: float,
                               alpha: float) -> "numpy.ndarray":
    """Per-state vector of the source's exponential distance weights around a multi-state target: w_T = exp(-alpha d_T /
    d_max) for every reference trajectory T that contains one of the k nearest states of either extreme of the target and
    has a state within the radius of some target state, with d_T the directed segment path dissimilarity from the target
    to T (step 04) and d_max = max(radius, max d_T over those trajectories); all other states, including the target's
    own, get 0."""
    D = np.asarray(dissimilarity, dtype=np.float64); tr = np.asarray(trajectory_ids); st = np.asarray(state_ids)
    n = tr.size
    if D.shape != (n, n) or n < 1 or not np.all(np.isfinite(D)) or st.shape != (n,):
        raise ValueError("dissimilarity must be a finite square matrix matching trajectory_ids and state_ids")
    tid = np.where(tr == target_label)[0]
    if tid.size < 2:
        raise ValueError("the target must hold at least two states")
    if int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    if not np.isfinite(radius) or radius <= 0.0 or not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("radius and alpha must be positive and finite")
    tid = tid[np.argsort(st[tid], kind="stable")]
    t1, t2 = int(tid[0]), int(tid[-1])
    others = np.array([i for i in range(n) if tr[i] != target_label], dtype=int)
    if others.size == 0:
        raise ValueError("the target must not be the only trajectory")
    k_traj = set()
    for ext in (t1, t2):
        order = others[np.argsort(D[others, ext], kind="stable")][: int(k)]
        k_traj |= set(tr[order].tolist())
    within = others[(D[np.ix_(others, tid)] <= float(radius)).any(axis=1)]
    eps_traj = set(tr[within].tolist())
    d_traj = {}
    for lab in dict.fromkeys(tr[others].tolist()):
        idx = others[tr[others] == lab]
        if lab in k_traj and lab in eps_traj and idx.size >= 2:
            idx = idx[np.argsort(st[idx], kind="stable")]
            d_traj[lab] = trajectory_dissimilarity(D, tid, idx)
    d_max = max(float(radius), max(d_traj.values())) if d_traj else float(radius)
    w = np.zeros(n)
    for lab, d in d_traj.items():
        w[tr == lab] = np.exp(-float(alpha) * d / d_max)
    return w

import numpy as np


def neighbour_states(dissimilarity: "numpy.ndarray", trajectory_ids: "numpy.ndarray",
                             target_index: int, k: int, radius: float,
                             weights: "numpy.ndarray" = None) -> "numpy.ndarray":
    """Indices of the reference states that PETRA-EDR keeps around one target state: the k nearest states that do not
    belong to the target's trajectory, restricted afterwards (no replacement) to dissimilarities below radius and, when
    weights are given, to states of positively weighted trajectories; ascending dissimilarity, ties by index."""
    D = np.asarray(dissimilarity, dtype=np.float64); tr = np.asarray(trajectory_ids)
    n = tr.size
    if D.shape != (n, n) or n < 1:
        raise ValueError("dissimilarity must be a square matrix matching trajectory_ids")
    if not np.all(np.isfinite(D)):
        raise ValueError("dissimilarity must be finite")
    if int(target_index) != target_index or not (0 <= target_index < n):
        raise ValueError("target_index must index a row of the matrix")
    if int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("weights must be a finite nonnegative vector matching trajectory_ids")
    t = int(target_index)
    candidates = np.array([i for i in np.argsort(D[:, t], kind="stable") if tr[i] != tr[t]], dtype=int)
    kept = candidates[: int(k)]                                              # k nearest ...
    kept = kept[D[kept, t] < float(radius)]                                  # ... then the radius, strict, no replacement
    if weights is not None:
        kept = kept[w[kept] > 0.0]                                           # ... then positive trajectory weight
    return kept.astype(int)

import numpy as np


def sweep_states(states: "numpy.ndarray", trajectory_ids: "numpy.ndarray", state_ids: "numpy.ndarray",
                         neighbours: "numpy.ndarray", weights: "numpy.ndarray", min_pts: int,
                         direction: int) -> "numpy.ndarray":
    """PETRA-EDR sweep in one direction: the i-th predicted state (i = 1, 2, ...) is the weighted mean of the states lying
    i positions before (direction -1) or after (direction +1) each kept neighbour within its own trajectory, formed while at
    least min_pts such states exist; rows ordered by increasing i."""
    X = np.asarray(states, dtype=np.float64); tr = np.asarray(trajectory_ids); st = np.asarray(state_ids)
    nb = np.asarray(neighbours, dtype=int); w = np.asarray(weights, dtype=np.float64)
    n = tr.size
    if X.ndim != 2 or X.shape[0] != n or st.shape != (n,) or w.shape != (n,):
        raise ValueError("states, trajectory_ids, state_ids and weights must describe the same n states")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(w))) or np.any(w < 0.0):
        raise ValueError("states must be finite and weights finite and nonnegative")
    if nb.ndim != 1 or (nb.size and (nb.min() < 0 or nb.max() >= n)):
        raise ValueError("neighbours must be a 1-D array of valid state indices")
    if int(min_pts) != min_pts or min_pts < 1:
        raise ValueError("min_pts must be a positive integer")
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 (backward) or +1 (forward)")
    # position of every state within its own trajectory (states sorted by state_ids)
    pos_of = {}
    for lab in dict.fromkeys(tr.tolist()):
        idx = np.where(tr == lab)[0]
        idx = idx[np.argsort(st[idx], kind="stable")]
        pos_of[lab] = idx
    out = []
    i = 1
    while True:
        shifted = []
        for j in nb:
            idx = pos_of[tr[j]]
            p = int(np.where(idx == j)[0][0]) + int(direction) * i
            if 0 <= p < idx.size:
                shifted.append(int(idx[p]))
        if len(shifted) < int(min_pts):
            break
        ww = w[shifted]
        if ww.sum() <= 0.0:
            raise ValueError("the contributing states carry no weight")
        out.append((X[shifted] * ww[:, None]).sum(axis=0) / ww.sum())
        i += 1
    return np.array(out).reshape(-1, X.shape[1])

import numpy as np


def predicted_trajectory(target_states: "numpy.ndarray", target_labels: "numpy.ndarray", backward: "numpy.ndarray",
                                 forward: "numpy.ndarray") -> "numpy.ndarray":
    """Assemble the PETRA-EDR predicted trajectory of a target from its observed states and the backward sweep of its
    first state and forward sweep of its last state (step 07): column 0 holds the label (the observed labels, then
    decreasing by one per backward step from the first label and increasing by one per forward step from the last) and
    the remaining columns the state variables; rows sorted by label."""
    X = np.asarray(target_states, dtype=np.float64); lab = np.asarray(target_labels, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[0] < 1 or lab.shape != (X.shape[0],) or not np.all(np.isfinite(X)) or not np.all(np.isfinite(lab)):
        raise ValueError("target_states must be a finite (n_X, S) array with one finite label per state")
    if np.any(lab != np.round(lab)) or np.unique(lab).size != lab.size:
        raise ValueError("target_labels must be distinct integers")
    S = X.shape[1]
    B = np.asarray(backward, dtype=np.float64).reshape(-1, S) if np.asarray(backward).size else np.empty((0, S))
    F = np.asarray(forward, dtype=np.float64).reshape(-1, S) if np.asarray(forward).size else np.empty((0, S))
    if not (np.all(np.isfinite(B)) and np.all(np.isfinite(F))):
        raise ValueError("the sweeps must be finite")
    rows = [np.concatenate([[float(l)], x]) for l, x in zip(lab, X)]
    lo, hi = float(lab.min()), float(lab.max())
    for i, row in enumerate(B, 1):
        rows.append(np.concatenate([[lo - i], row]))
    for i, row in enumerate(F, 1):
        rows.append(np.concatenate([[hi + i], row]))
    out = np.array(rows)
    return out[np.argsort(out[:, 0], kind="stable")]

import numpy as np


def mean_predicted_deviation(states: "numpy.ndarray", trajectory_ids: "numpy.ndarray",
                                     state_ids: "numpy.ndarray", target_label: int, k: int, radius: float,
                                     min_pts: int, weights: "numpy.ndarray",
                                     predicted: "numpy.ndarray") -> float:
    """Mean predicted deviation (source Eq 3): every predicted state becomes a single-state target in the reference set
    formed by the regime trajectories without the original target, is forecast in both directions with the same k, radius,
    min_pts and the original target's trajectory weights (steps 03, 06, 07, 08), and MPD is the mean over the observed
    target states and the predicted states of the minimum Bray-Curtis dissimilarity between the observed state and the
    states of that forecast."""
    X = np.asarray(states, dtype=np.float64); tr = np.asarray(trajectory_ids); st = np.asarray(state_ids)
    w = np.asarray(weights, dtype=np.float64); P = np.asarray(predicted, dtype=np.float64)
    n = tr.size
    if X.ndim != 2 or X.shape[0] != n or st.shape != (n,) or w.shape != (n,):
        raise ValueError("states, trajectory_ids, state_ids and weights must describe the same n states")
    tid = np.where(tr == target_label)[0]
    if tid.size < 1:
        raise ValueError("target_label must name at least one state")
    if P.ndim != 2 or P.shape[1] != X.shape[1] + 1 or P.shape[0] < 1:
        raise ValueError("predicted must have one label column plus the state variables")
    own = set(st[tid].astype(float).tolist())
    pred_rows = P[[float(l) not in own for l in P[:, 0]]]
    if pred_rows.shape[0] == 0:
        raise ValueError("the predicted trajectory holds no predicted state")
    keep = np.where(tr != target_label)[0]                                # regime without the original target
    devs = []
    for row in pred_rows:
        label, y = float(row[0]), row[1:]
        Xr = np.vstack([X[keep], y[None, :]])
        trr = np.concatenate([tr[keep].astype(object), np.array(["__predicted__"], dtype=object)])
        str_ = np.concatenate([st[keep], [label]])
        wr = np.concatenate([w[keep], [0.0]])
        ti = Xr.shape[0] - 1
        Dr = bray_curtis_matrix(Xr)
        nb = neighbour_states(Dr, trr, ti, k, radius, wr)
        back = sweep_states(Xr, trr, str_, nb, wr, min_pts, -1)
        forw = sweep_states(Xr, trr, str_, nb, wr, min_pts, 1)
        Z = predicted_trajectory(y, np.array([label]), back, forw)[:, 1:]
        for i in tid:
            num = np.abs(Z - X[i][None, :]).sum(axis=1); den = (Z + X[i][None, :]).sum(axis=1)
            devs.append(float(np.min(num / den)))
    return float(np.mean(devs))

import numpy as np


def forecast_deviation(n_traj: int, n_states: int, base: "numpy.ndarray", amplitude: float,
                               growth: "numpy.ndarray", capacity: "numpy.ndarray", interaction: "numpy.ndarray",
                               source_a: int, source_b: int, first_state: int, last_state: int, k: int,
                               radius: float, min_pts: int, alpha: float) -> float:
    """Whole forecast at one radius: build the regime (step 02), the target (states first_state..last_state of the run
    started from the mean of the initial communities of trajectories source_a and source_b), the dissimilarity matrix
    (03), the weights (05), the trajectory dissimilarities of the weighted trajectories (04, as a consistency gate on the
    weights), the two neighbourhoods (06), the sweeps (07), the predicted trajectory (08) and its mean predicted deviation
    (09)."""
    if int(n_traj) != n_traj or n_traj < 2 or int(n_states) != n_states or n_states < 2:
        raise ValueError("n_traj must be at least 2 and n_states at least 2")
    if int(source_a) != source_a or int(source_b) != source_b or not (1 <= source_a <= n_traj and 1 <= source_b <= n_traj):
        raise ValueError("source_a and source_b must index regime trajectories (1-based)")
    if int(first_state) != first_state or int(last_state) != last_state or not (1 <= first_state < last_state <= n_states):
        raise ValueError("first_state and last_state must satisfy 1 <= first_state < last_state <= n_states")
    b = np.asarray(base, dtype=np.float64)
    S = b.size
    X = regime_states(n_traj, n_states, b, amplitude, growth, capacity, interaction)
    tr = np.repeat(np.arange(1, int(n_traj) + 1), int(n_states))
    st = np.tile(np.arange(1, int(n_states) + 1), int(n_traj))
    s_idx = np.arange(1, S + 1)
    inits = []
    for j in (int(source_a), int(source_b)):
        h = (1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001
        inits.append(b * (1.0 + float(amplitude) * (2.0 * h / 1000.0 - 1.0)))
    N = ricker_trajectory(0.5 * (inits[0] + inits[1]), growth, capacity, interaction, int(n_states))
    labels = np.arange(int(first_state), int(last_state) + 1)
    target = N[labels - 1] / N[labels - 1].sum(axis=1, keepdims=True)
    Xa = np.vstack([X, target])
    tra = np.concatenate([tr, np.zeros(labels.size, dtype=int)])          # trajectory 0 = the target
    sta = np.concatenate([st, labels])
    D = bray_curtis_matrix(Xa)
    w = trajectory_weights(D, tra, sta, 0, k, radius, alpha)
    t1 = X.shape[0]; t2 = Xa.shape[0] - 1
    tid = np.arange(t1, t2 + 1)
    weighted = list(dict.fromkeys(tra[w > 0].tolist()))
    if weighted and not np.all(w[tra != 0] == 1.0):                     # consistency gate on the weights (step 04)
        d_T = {lab: trajectory_dissimilarity(D, tid, np.where(tra == lab)[0]) for lab in weighted}
        d_max = max(float(radius), max(d_T.values()))
        for lab, d in d_T.items():
            if abs(float(w[tra == lab][0]) - np.exp(-float(alpha) * d / d_max)) > 1e-9:
                raise ValueError("trajectory weights are inconsistent with the trajectory dissimilarities")
    nb_back = neighbour_states(D, tra, t1, k, radius, w)
    nb_forw = neighbour_states(D, tra, t2, k, radius, w)
    back = sweep_states(Xa, tra, sta, nb_back, w, min_pts, -1)
    forw = sweep_states(Xa, tra, sta, nb_forw, w, min_pts, 1)
    P = predicted_trajectory(target, labels, back, forw)
    return mean_predicted_deviation(Xa, tra, sta, 0, k, radius, min_pts, w, P)

import numpy as np


def optimal_radius_deviation(n_traj: int, n_states: int, base: "numpy.ndarray", amplitude: float,
                                     growth: "numpy.ndarray", capacity: "numpy.ndarray", interaction: "numpy.ndarray",
                                     source_a: int, source_b: int, first_state: int, last_state: int, k: int,
                                     radius_min: float, radius_max: float, min_pts: int,
                                     alpha: float) -> "numpy.ndarray":
    """The source's radius selection by minimising the mean predicted deviation: the infimum of the step-10
    deviation over [radius_min, radius_max] and the infimum/supremum of the radii attaining it. Because the forecast depends
    on the radius only through threshold comparisons (and the scale factor exceeds every radius of the domain), the
    deviation is a step function of the radius; this function locates its pieces exactly - the radii at which a state
    enters a neighbourhood of a target extreme (06), a trajectory becomes eligible for a weight (05), or a state enters
    the neighbourhood of a predicted state in the re-forecasts of step 09 - evaluates the deviation on every open piece
    (no isolated breakpoint value lies below the piece minimum for the shipped cases; the cross-check through step 10 at
    the optimum guards the enumeration) and returns [infimum, infimum of the attaining radii, supremum of the attaining
    radii]."""
    if int(n_traj) != n_traj or n_traj < 2 or int(n_states) != n_states or n_states < 2:
        raise ValueError("n_traj must be at least 2 and n_states at least 2")
    if int(source_a) != source_a or int(source_b) != source_b or not (1 <= source_a <= n_traj and 1 <= source_b <= n_traj):
        raise ValueError("source_a and source_b must index regime trajectories (1-based)")
    if int(first_state) != first_state or int(last_state) != last_state or not (1 <= first_state < last_state <= n_states):
        raise ValueError("first_state and last_state must satisfy 1 <= first_state < last_state <= n_states")
    lo, hi = float(radius_min), float(radius_max)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0.0 or hi <= lo:
        raise ValueError("radius_min and radius_max must satisfy 0 < radius_min < radius_max")
    if int(k) != k or k < 1 or int(min_pts) != min_pts or min_pts < 1 or not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("k and min_pts must be positive integers and alpha positive")
    b = np.asarray(base, dtype=np.float64)
    S = b.size
    X = regime_states(n_traj, n_states, b, amplitude, growth, capacity, interaction)
    tr = np.repeat(np.arange(1, int(n_traj) + 1), int(n_states))
    st = np.tile(np.arange(1, int(n_states) + 1), int(n_traj))
    s_idx = np.arange(1, S + 1)
    inits = []
    for j in (int(source_a), int(source_b)):
        h = (1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001
        inits.append(b * (1.0 + float(amplitude) * (2.0 * h / 1000.0 - 1.0)))
    N = ricker_trajectory(0.5 * (inits[0] + inits[1]), growth, capacity, interaction, int(n_states))
    labels = np.arange(int(first_state), int(last_state) + 1)
    target = N[labels - 1] / N[labels - 1].sum(axis=1, keepdims=True)
    Xa = np.vstack([X, target])
    tra = np.concatenate([tr, np.zeros(labels.size, dtype=int)])
    sta = np.concatenate([st, labels])
    D = bray_curtis_matrix(Xa)
    t1 = X.shape[0]; t2 = Xa.shape[0] - 1
    tid = np.arange(t1, t2 + 1)
    # radii at which the target's neighbourhoods or the weighted set can change (the k nearest states are fixed)
    breaks = set()
    for ext in (t1, t2):
        for i in neighbour_states(D, tra, ext, k, 2.0, None):
            breaks.add(float(D[i, ext]))
    for lab in range(1, int(n_traj) + 1):
        breaks.add(float(D[np.ix_(np.where(tra == lab)[0], tid)].min()))
    edges = [lo] + sorted(x for x in breaks if lo < x < hi) + [hi]
    pieces = []
    for a, c in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (a + c)
        w = trajectory_weights(D, tra, sta, 0, k, mid, alpha)
        weighted = list(dict.fromkeys(tra[w > 0].tolist()))
        if not weighted or np.all(w[tra != 0] == 1.0):
            raise ValueError("no weighted trajectory for a radius inside the domain")
        d_T = max(trajectory_dissimilarity(D, tid, np.where(tra == lab)[0]) for lab in weighted)
        if d_T < c - 1e-12:                                            # the deviation would vary continuously with the radius
            raise ValueError("the scale factor falls below a radius inside the domain")
        nb_back = neighbour_states(D, tra, t1, k, mid, w)
        nb_forw = neighbour_states(D, tra, t2, k, mid, w)
        back = sweep_states(Xa, tra, sta, nb_back, w, min_pts, -1)
        forw = sweep_states(Xa, tra, sta, nb_forw, w, min_pts, 1)
        P = predicted_trajectory(target, labels, back, forw)
        if P.shape[0] == labels.size:
            raise ValueError("no predicted state for a radius inside the domain")
        # radii at which a re-forecast neighbourhood can change: the k nearest regime states of every predicted state
        sub = set()
        own = set(labels.astype(float).tolist())
        for row in P:
            if float(row[0]) in own:
                continue
            Dr = bray_curtis_matrix(np.vstack([X, row[1:][None, :]]))
            trr = np.concatenate([tr, [0]])
            for i in neighbour_states(Dr, trr, X.shape[0], k, 2.0, None):
                d = float(Dr[i, X.shape[0]])
                if a < d < c:
                    sub.add(d)
        sedges = [a] + sorted(sub) + [c]
        for a2, c2 in zip(sedges[:-1], sedges[1:]):
            val = mean_predicted_deviation(Xa, tra, sta, 0, k, 0.5 * (a2 + c2), min_pts, w, P)
            pieces.append((a2, c2, val))
    best = min(p[2] for p in pieces)
    opt = [p for p in pieces if abs(p[2] - best) <= 1e-12]
    eps_lo, eps_hi = min(p[0] for p in opt), max(p[1] for p in opt)
    check = forecast_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a,
                                       source_b, first_state, last_state, k, 0.5 * (opt[0][0] + opt[0][1]), min_pts, alpha)
    if abs(check - best) > 1e-12:
        raise ValueError("the optimal piece does not reproduce through the single-radius forecast")
    return np.array([best, eps_lo, eps_hi])

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


def optimal_shape_parameter(n_traj: int, n_states: int, base: "numpy.ndarray", amplitude: float,
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
    X = regime_states(n_traj, n_states, b, amplitude, growth, capacity, interaction)
    tr = np.repeat(np.arange(1, int(n_traj) + 1), int(n_states))
    st = np.tile(np.arange(1, int(n_states) + 1), int(n_traj))
    s_idx = np.arange(1, S + 1)
    inits = []
    for j in (int(source_a), int(source_b)):
        h = (1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001
        inits.append(b * (1.0 + float(amplitude) * (2.0 * h / 1000.0 - 1.0)))
    N = ricker_trajectory(0.5 * (inits[0] + inits[1]), growth, capacity, interaction, int(n_states))
    labels = np.arange(int(first_state), int(last_state) + 1)
    target = N[labels - 1] / N[labels - 1].sum(axis=1, keepdims=True)
    Xa = np.vstack([X, target])
    tra = np.concatenate([tr, np.zeros(labels.size, dtype=int)])
    sta = np.concatenate([st, labels])
    D = bray_curtis_matrix(Xa)
    t1 = X.shape[0]; t2 = Xa.shape[0] - 1
    tid = np.arange(t1, t2 + 1)
    n_reg = X.shape[0]
    trr = np.concatenate([tr, [0]])
    own = set(labels.astype(float).tolist())
    # alpha-independent part, built once through the step functions: weighted set, scaled dissimilarities, the target's
    # neighbourhoods (the weight filter only asks which trajectories carry a weight) and the sweep index sets
    w_ref = trajectory_weights(D, tra, sta, 0, k, radius, 0.5 * (lo + hi))
    weighted = list(dict.fromkeys(tra[w_ref > 0].tolist()))
    if not weighted:
        raise ValueError("no weighted trajectory")
    d_T = {lab: trajectory_dissimilarity(D, tid, np.where(tra == lab)[0]) for lab in weighted}
    d_max = max(float(radius), max(d_T.values()))
    delta = np.zeros(tra.size)
    for lab in weighted:
        delta[tra == lab] = d_T[lab] / d_max
    if not np.allclose(w_ref[delta > 0], np.exp(-0.5 * (lo + hi) * delta[delta > 0]), rtol=0, atol=1e-12) or np.any(w_ref[delta == 0] != 0.0):
        raise ValueError("the weights do not follow the exponential of the scaled trajectory dissimilarities")
    nb_back = neighbour_states(D, tra, t1, k, radius, w_ref)
    nb_forw = neighbour_states(D, tra, t2, k, radius, w_ref)
    sets_back = _shifted_sets(nb_back, tra, sta, -1, min_pts, int(n_states))
    sets_forw = _shifted_sets(nb_forw, tra, sta, 1, min_pts, int(n_states))
    pred = [(float(labels[0] - i), idx) for i, idx in enumerate(sets_back, 1)][::-1] + [(float(labels[-1] + i), idx) for i, idx in enumerate(sets_forw, 1)]
    if not pred:
        raise ValueError("no predicted state")
    back = sweep_states(Xa, tra, sta, nb_back, w_ref, min_pts, -1)
    forw = sweep_states(Xa, tra, sta, nb_forw, w_ref, min_pts, 1)
    P_ref = predicted_trajectory(target, labels, back, forw)
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
    if not np.allclose(bray_curtis_matrix(np.vstack([X, y0[None, :]]))[n_reg, :n_reg], np.abs(X - y0).sum(axis=1) / (X + y0).sum(axis=1), rtol=0, atol=1e-15):
        raise ValueError("the Bray-Curtis rule does not reproduce")

    def _structure(alpha):
        """The alpha-dependent structure: the re-forecast neighbourhood and sweep index sets of every predicted state."""
        w = np.where(delta > 0, np.exp(-alpha * delta), 0.0)
        refc = []
        for lab, idx in pred:
            ww = w[idx]; y = (ww[:, None] * Xa[idx]).sum(axis=0) / ww.sum()
            dvec = np.abs(X - y).sum(axis=1) / (X + y).sum(axis=1)     # Bray-Curtis of y to the regime states (step 03's rule)
            Dr = D_re.copy(); Dr[n_reg, :n_reg] = dvec; Dr[:n_reg, n_reg] = dvec
            nb = neighbour_states(Dr, trr, n_reg, k, radius, wr_vec)
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
    if abs(v0 - mean_predicted_deviation(Xa, tra, sta, 0, k, radius, min_pts, w_ref, P_ref)) > 1e-12:
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
    check = forecast_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a,
                                       source_b, first_state, last_state, k, radius, min_pts, alpha_star)
    if abs(check - value) > 1e-9:
        raise ValueError("the optimum does not reproduce through the single-parameter forecast")
    if not np.isfinite(alpha_ref) or alpha_ref <= 0.0:
        raise ValueError("alpha_ref must be positive")
    rad = optimal_radius_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a,
                                           source_b, first_state, last_state, k, radius_min, radius_max, min_pts, alpha_ref)
    return np.array([alpha_star, check, rad[0], rad[1], rad[2]])
SCICODE_GOLD_EOF
