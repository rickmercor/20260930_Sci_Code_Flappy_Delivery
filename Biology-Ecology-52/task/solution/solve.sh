#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import math
import numpy as np


def trophic_positions(diet: "numpy.ndarray") -> "numpy.ndarray":
    """Trophic positions y = (I - Q)^-1 1 of a diet-coefficient matrix Q (rows predators, columns prey)."""
    Q = np.asarray(diet, dtype=np.float64)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or Q.shape[0] < 1:
        raise ValueError("diet must be a square matrix")
    if not np.all(np.isfinite(Q)) or np.any(Q < 0.0):
        raise ValueError("diet coefficients must be finite and nonnegative")
    if np.any(np.abs(np.diag(Q)) > 0.0):
        raise ValueError("cannibalism is excluded: the diagonal must be zero")
    sums = Q.sum(axis=1)
    if not np.all((np.abs(sums) < 1e-12) | (np.abs(sums - 1.0) < 1e-12)):
        raise ValueError("every row must sum to 0 (producer) or 1 (consumer)")
    n = Q.shape[0]
    M = np.eye(n) - Q
    if abs(np.linalg.det(M)) < 1e-12:
        raise ValueError("I - Q is singular: the web has no producer or contains a closed feeding loop")
    return np.linalg.solve(M, np.ones(n))

import math
import numpy as np


def pair_diet_coefficients(y_below: float, y_above: float, centre: float) -> "numpy.ndarray":
    """Two-prey solution of Eq 7: the diet coefficients (below, above) of a predator whose prey mean position is centre."""
    yb, ya, c = float(y_below), float(y_above), float(centre)
    if not all(np.isfinite([yb, ya, c])):
        raise ValueError("all inputs must be finite")
    if not (yb <= c <= ya) or ya <= yb:
        raise ValueError("the centre must lie between the two prey positions, y_below < y_above")
    # the predator's prey mean c is a weighted average of the two prey positions (Eq 6a) with weights summing
    # to one (Eq 6b): linear interpolation of c between y_below and y_above
    return np.array([(ya - c) / (ya - yb), (c - yb) / (ya - yb)])

import math
import numpy as np


def candidate_pairs(positions: "numpy.ndarray", predator: int, sigma: float) -> "numpy.ndarray":
    """Source decomposition at trophic specialization sigma: all (below, above) pairs of admissible prey of the predator."""
    y = np.asarray(positions, dtype=np.float64)
    if y.ndim != 1 or y.size < 3 or not np.all(np.isfinite(y)):
        raise ValueError("positions must be a finite 1-D array of at least three species")
    if int(predator) != predator or not (0 <= predator < y.size):
        raise ValueError("predator must be a valid species index")
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be finite and nonnegative")
    i = int(predator)
    c = y[i] - 1.0
    idx = np.arange(y.size)
    # strict 3-sigma window on each side of the centre; the predator itself is never a prey (no cannibalism)
    below = idx[(idx != i) & (y < c) & (y > c - 3.0 * float(sigma))]
    above = idx[(idx != i) & (y > c) & (y < c + 3.0 * float(sigma))]
    # every prey below the centre paired with every prey above it (the two-prey solution needs one of each)
    pairs = np.array([(int(d), int(u)) for d in below for u in above], dtype=np.int64).reshape(-1, 2)
    return pairs

import math
import numpy as np


def pair_thresholds(positions: "numpy.ndarray", predator: int, pairs: "numpy.ndarray") -> "numpy.ndarray":
    """Admissibility threshold of every given pair: the pair is a candidate for sigma above max(d_b, d_a)/3."""
    y = np.asarray(positions, dtype=np.float64)
    P = np.asarray(pairs)
    if y.ndim != 1 or y.size < 3 or not np.all(np.isfinite(y)):
        raise ValueError("positions must be a finite 1-D array of at least three species")
    if int(predator) != predator or not (0 <= predator < y.size):
        raise ValueError("predator must be a valid species index")
    if P.ndim != 2 or P.shape[1] != 2 or P.size and (P.min() < 0 or P.max() >= y.size):
        raise ValueError("pairs must be an integer array of shape (n_pairs, 2) with valid species indices")
    c = y[int(predator)] - 1.0
    if P.shape[0] and (np.any(y[P[:, 0]] >= c) or np.any(y[P[:, 1]] <= c)):
        raise ValueError("every pair must hold one prey below and one prey above the predator's centre")
    # the strict 3-sigma window admits the lower prey for sigma > (c - y_b)/3 and the upper prey for sigma > (y_a - c)/3;
    # an empty pair set gives an empty threshold array
    return np.maximum(c - y[P[:, 0]], y[P[:, 1]] - c).astype(np.float64) / 3.0

import math
import numpy as np


def marginal_pair_weights(positions: "numpy.ndarray", predator: int, pairs: "numpy.ndarray",
                                  thresholds: "numpy.ndarray", sigma_max: float) -> "numpy.ndarray":
    """Eq 29 evaluated exactly: the sigma-averaged Bayes numerator of every candidate pair on [0, sigma_max]."""
    y = np.asarray(positions, dtype=np.float64)
    P = np.asarray(pairs)
    thr = np.asarray(thresholds, dtype=np.float64)
    if not np.isfinite(sigma_max) or sigma_max <= 0.0:
        raise ValueError("sigma_max must be finite and positive")
    if y.ndim != 1 or y.size < 3 or not np.all(np.isfinite(y)):
        raise ValueError("positions must be a finite 1-D array of at least three species")
    if int(predator) != predator or not (0 <= predator < y.size):
        raise ValueError("predator must be a valid species index")
    if P.ndim != 2 or P.shape[1] != 2 or P.shape[0] == 0:
        raise ValueError("the predator has no candidate prey pair")
    if thr.shape != (P.shape[0],) or not np.all(np.isfinite(thr)) or np.any(thr < 0.0) or np.any(thr >= sigma_max):
        raise ValueError("thresholds must be one finite value per pair, nonnegative and below sigma_max")
    c = y[int(predator)] - 1.0
    smax = float(sigma_max)
    # the admissible set, and with it the prior 1/n_pairs(sigma), is piecewise constant between the thresholds
    knots = np.unique(np.concatenate([thr, [smax]]))
    weights = np.zeros(P.shape[0])
    for k, (d, u) in enumerate(P):
        a = 0.5 * ((c - y[d]) ** 2 + (y[u] - c) ** 2)            # numerator of the Gaussian exponent
        total = 0.0
        for lo, hi in zip(knots[:-1], knots[1:]):
            if hi <= thr[k]:
                continue                                         # the pair does not exist yet on this interval
            n_pairs = int(np.sum(thr <= lo + 1e-15))             # pairs admissible on (lo, hi)
            # int_lo^hi exp(-a / s^2) / (2 pi s^2) ds = [erfc(sqrt(a)/hi) - erfc(sqrt(a)/lo)] * sqrt(pi / a) / (4 pi)
            ra = math.sqrt(a)
            total += (math.erfc(ra / hi) - math.erfc(ra / lo)) * math.sqrt(math.pi / a) / (4.0 * math.pi) / n_pairs
        weights[k] = total / smax                                # uniform average over sigma in [0, sigma_max]
    return weights

import math
import numpy as np


def superpose_row(n_species: int, pairs: "numpy.ndarray", pair_diets: "numpy.ndarray",
                          weights: "numpy.ndarray") -> "numpy.ndarray":
    """Superposition (Eqs 23-26): the diet row from the pair diets weighted by the normalised pair weights."""
    P = np.asarray(pairs)
    D = np.asarray(pair_diets, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if int(n_species) != n_species or n_species < 3:
        raise ValueError("n_species must be an integer of at least 3")
    n = int(n_species)
    if P.ndim != 2 or P.shape[1] != 2 or P.shape[0] == 0 or P.min() < 0 or P.max() >= n:
        raise ValueError("pairs must be a non-empty integer array of shape (n_pairs, 2) with valid species indices")
    if D.shape != (P.shape[0], 2) or not np.all(np.isfinite(D)) or np.any(D < 0.0):
        raise ValueError("pair_diets must be a finite nonnegative array of shape (n_pairs, 2)")
    if w.shape != (P.shape[0],) or not np.all(np.isfinite(w)) or np.any(w < 0.0) or w.sum() <= 0.0:
        raise ValueError("weights must be one finite nonnegative value per pair with a positive sum")
    probs = w / w.sum()                                           # law of total probability over the pairs
    row = np.zeros(n)
    for p, (d, u), t in zip(probs, P, D):
        row[d] += p * t[0]
        row[u] += p * t[1]
    return row

import math
import numpy as np


def reconstruction_metrics(true_diet: "numpy.ndarray", diet: "numpy.ndarray") -> "numpy.ndarray":
    """Metrics of Eqs 30-39: [TP, FN, TN, FP, TPR, FPR, BA, mean error, similarity] of the reconstruction against the true web."""
    Qt = np.asarray(true_diet, dtype=np.float64)
    Q = np.asarray(diet, dtype=np.float64)
    if Qt.ndim != 2 or Qt.shape[0] != Qt.shape[1] or Q.shape != Qt.shape or Qt.shape[0] < 2:
        raise ValueError("both diet matrices must be square, of the same size, with at least two species")
    if not (np.all(np.isfinite(Qt)) and np.all(np.isfinite(Q))) or np.any(Qt < 0.0) or np.any(Q < 0.0):
        raise ValueError("diet coefficients must be finite and nonnegative")
    n = Qt.shape[0]
    At = (Qt > 0.0).astype(np.float64)                            # adjacency (Eq 30)
    A = (Q > 0.0).astype(np.float64)
    off = np.ones((n, n)) - np.eye(n)                            # I_ij = 1 - delta_ij
    TP = float(np.sum(At * A))
    FN = float(np.sum(At * (off - A)))
    TN = float(np.sum((off - At) * (off - A)))
    FP = float(np.sum((off - At) * A))
    L_true, L = float(At.sum()), float(A.sum())
    if L_true == 0.0 or n * (n - 1) - L_true == 0.0:
        raise ValueError("the true web must have at least one link and at least one absent link")
    nt, nr = np.linalg.norm(Qt), np.linalg.norm(Q)
    if nr == 0.0:
        raise ValueError("the reconstructed matrix must have at least one nonzero entry")
    TPR = TP / L_true                                             # recall (Eq 35)
    FPR = (L - TP) / (n * (n - 1) - L_true)                       # fallout (Eq 36)
    BA = 0.5 * (TPR + 1.0 - FPR)                                  # balanced accuracy (Eq 37)
    mean_error = float(np.sum(np.abs(Qt - Q))) / (n * n)         # Eq 38
    similarity = float(np.sum(Qt * Q)) / (nt * nr)               # Eq 39 (entry-wise product over Frobenius norms)
    return np.array([TP, FN, TN, FP, TPR, FPR, BA, mean_error, similarity])

import math
import numpy as np


def _primitive_dA(a: float, s: float) -> float:
    """d/dA of the step-05 primitive int_0^s exp(-A/x^2)/(2 pi x^2) dx, i.e. -int_0^s exp(-A/x^2)/(2 pi x^4) dx (u = 1/x)."""
    ra = math.sqrt(a)
    return -(math.sqrt(math.pi) * math.erfc(ra / s) / (4.0 * a * ra) + math.exp(-a / s ** 2) / (2.0 * a * s)) / (2.0 * math.pi)


def _primitive_dAA(a: float, s: float) -> float:
    """d^2/dA^2 of the step-05 primitive: int_0^s exp(-A/x^2)/(2 pi x^6) dx = (1/2pi) int_{1/s}^inf u^4 exp(-A u^2) du."""
    ra = math.sqrt(a)
    return (3.0 * math.sqrt(math.pi) * math.erfc(ra / s) / (8.0 * a ** 2 * ra)
            + math.exp(-a / s ** 2) * (1.0 / (2.0 * a * s ** 3) + 3.0 / (4.0 * a ** 2 * s))) / (2.0 * math.pi)


def _pair_density(a: float, s: float) -> float:
    """The product of the two Gaussian densities of a pair at specialization s: exp(-A/s^2) / (2 pi s^2)."""
    return math.exp(-a / s ** 2) / (2.0 * math.pi * s ** 2)


def marginal_weight_derivatives(positions: "numpy.ndarray", predator: int, pairs: "numpy.ndarray",
                                        sigma_max: float, species: int) -> "numpy.ndarray":
    """Exact first and second derivatives of the Eq 29 averaged weights (step 05) with respect to one trophic position."""
    y = np.asarray(positions, dtype=np.float64)
    P = np.asarray(pairs)
    if not np.isfinite(sigma_max) or sigma_max <= 0.0:
        raise ValueError("sigma_max must be finite and positive")
    if y.ndim != 1 or y.size < 3 or not np.all(np.isfinite(y)):
        raise ValueError("positions must be a finite 1-D array of at least three species")
    if int(predator) != predator or not (0 <= predator < y.size):
        raise ValueError("predator must be a valid species index")
    if int(species) != species or not (0 <= species < y.size):
        raise ValueError("species must be a valid species index")
    if P.ndim != 2 or P.shape[1] != 2 or P.shape[0] == 0 or P.min() < 0 or P.max() >= y.size:
        raise ValueError("pairs must be a non-empty integer array of shape (n_pairs, 2) with valid species indices")
    i, m = int(predator), int(species)
    c = y[i] - 1.0
    smax = float(sigma_max)
    d_b = c - y[P[:, 0]]
    d_a = y[P[:, 1]] - c
    if np.any(d_b <= 0.0) or np.any(d_a <= 0.0):
        raise ValueError("every pair must hold one prey below and one prey above the predator's centre")
    if np.any(np.abs(d_b - d_a) <= 1e-12 * np.maximum(d_b, d_a)):
        raise ValueError("a pair with equal prey offsets has no derivative: its threshold switches between the two prey")
    thr = np.maximum(d_b, d_a) / 3.0
    if np.any(thr >= smax):
        raise ValueError("every pair must be a candidate at sigma_max (threshold below sigma_max)")
    # how the centre and the two offsets of every pair move with the position of the species (linearly)
    dc = 1.0 if m == i else 0.0
    dd_b = dc - (P[:, 0] == m).astype(np.float64)
    dd_a = (P[:, 1] == m).astype(np.float64) - dc
    A = 0.5 * (d_b ** 2 + d_a ** 2)
    dA = d_b * dd_b + d_a * dd_a                                  # first derivative of A
    d2A = dd_b ** 2 + dd_a ** 2                                   # second derivative of A (the offsets are linear)
    dthr = np.where(d_b > d_a, dd_b, dd_a) / 3.0                  # the threshold follows the farther prey, linearly
    # knots of the piecewise-constant candidate count: equal thresholds form one knot that must move as one
    order = np.argsort(thr, kind="stable")
    knots, members = [], []
    for k in order:
        if knots and abs(thr[k] - knots[-1]) <= 1e-12 * max(1.0, knots[-1]):
            members[-1].append(k)
        else:
            knots.append(thr[k]); members.append([k])
    dknots = np.zeros(len(knots))
    for g, mem in enumerate(members):
        moves = dthr[mem]
        if np.max(moves) - np.min(moves) > 1e-12:
            raise ValueError("pairs with equal thresholds that move apart under the perturbation have no derivative")
        dknots[g] = moves[0]
    group_of = np.zeros(P.shape[0], dtype=np.int64)
    for g, mem in enumerate(members):
        group_of[mem] = g
    counts = np.cumsum([len(mem) for mem in members])           # candidate pairs above each knot
    edges = np.array(knots + [smax])
    out = np.zeros((P.shape[0], 2))
    for k in range(P.shape[0]):
        g0 = group_of[k]
        a = A[k]
        sum_FA = sum_FAA = 0.0                                    # sums over the intervals above the pair's threshold
        bnd_g = bnd_gA = bnd_gs = 0.0                              # boundary terms: density, its A-derivative, its sigma-derivative
        for g in range(g0, len(knots)):
            lo, hi = edges[g], edges[g + 1]
            sum_FA += (_primitive_dA(a, hi) - _primitive_dA(a, lo)) / counts[g]
            sum_FAA += (_primitive_dAA(a, hi) - _primitive_dAA(a, lo)) / counts[g]
            if g > g0:
                # another group's knot moves: the prior 1/n jumps there from 1/n_before to 1/n_after
                coef = (1.0 / counts[g - 1] - 1.0 / counts[g]) * dknots[g]
                dens = _pair_density(a, lo)
                bnd_g += dens * coef
                bnd_gA += -dens / lo ** 2 * coef                  # d/dA of the density
                bnd_gs += dens * (2.0 * a / lo ** 3 - 2.0 / lo) * coef * dknots[g]   # d/dsigma of the density, times the knot speed
        # the pair's own threshold moves: the lower limit of its contribution
        lo0 = edges[g0]
        coef0 = -dknots[g0] / counts[g0]
        dens0 = _pair_density(a, lo0)
        bnd_g += dens0 * coef0
        bnd_gA += -dens0 / lo0 ** 2 * coef0
        bnd_gs += dens0 * (2.0 * a / lo0 ** 3 - 2.0 / lo0) * coef0 * dknots[g0]
        first = dA[k] * sum_FA + bnd_g
        # second order: the likelihood term twice differentiated, the moving knots inside it (once more each), and the
        # boundary terms differentiated through the exponent and through the knot position
        second = d2A[k] * sum_FA + dA[k] ** 2 * sum_FAA + 2.0 * dA[k] * bnd_gA + bnd_gs
        out[k, 0] = first / smax
        out[k, 1] = second / smax
    return out

import math
import numpy as np


def diet_row_derivatives(positions: "numpy.ndarray", predator: int, sigma_max: float,
                                 species: int) -> "numpy.ndarray":
    """Exact first and second derivatives of the reconstructed diet row of one predator (steps 02-06) w.r.t. one position."""
    y = np.asarray(positions, dtype=np.float64)
    if not np.isfinite(sigma_max) or sigma_max <= 0.0:
        raise ValueError("sigma_max must be finite and positive")
    if y.ndim != 1 or y.size < 3 or not np.all(np.isfinite(y)):
        raise ValueError("positions must be a finite 1-D array of at least three species")
    if int(predator) != predator or not (0 <= predator < y.size):
        raise ValueError("predator must be a valid species index")
    if int(species) != species or not (0 <= species < y.size):
        raise ValueError("species must be a valid species index")
    i, m, n = int(predator), int(species), y.size
    c = y[i] - 1.0
    if np.any((np.arange(n) != i) & (np.abs(y - c) < 1e-12)):
        raise ValueError("a prey sits exactly at the predator's centre; the two-prey decomposition does not apply")
    if np.any((np.arange(n) != i) & (np.abs(np.abs(y - c) - 3.0 * float(sigma_max)) < 1e-12)):
        raise ValueError("a species sits exactly on the admissibility window at sigma_max; the row has no derivative there")
    pairs = candidate_pairs(y, i, sigma_max)
    if pairs.shape[0] == 0:
        raise ValueError("the predator has no candidate prey pair at sigma_max")
    thresholds = pair_thresholds(y, i, pairs)
    w = marginal_pair_weights(y, i, pairs, thresholds, sigma_max)
    dw = marginal_weight_derivatives(y, i, pairs, sigma_max, m)
    W, W1, W2 = w.sum(), dw[:, 0].sum(), dw[:, 1].sum()
    probs = w / W
    p1 = dw[:, 0] / W - w * W1 / W ** 2                           # normalisation over the pairs, first order
    p2 = dw[:, 1] / W - 2.0 * dw[:, 0] * W1 / W ** 2 - w * W2 / W ** 2 + 2.0 * w * W1 ** 2 / W ** 3
    dc = 1.0 if m == i else 0.0
    out = np.zeros((n, 2))
    for k, (b, a) in enumerate(pairs):
        diet = pair_diet_coefficients(y[b], y[a], c)
        span = y[a] - y[b]
        dy_a, dy_b = float(a == m), float(b == m)
        # Eq 7: Q_b = (y_a - c) / (y_a - y_b), Q_a = 1 - Q_b; numerator and denominator are linear in the position
        dnum, dspan = dy_a - dc, dy_a - dy_b
        q1 = (dnum * span - (y[a] - c) * dspan) / span ** 2
        q2 = -2.0 * dspan * q1 / span
        out[b, 0] += p1[k] * diet[0] + probs[k] * q1
        out[a, 0] += p1[k] * diet[1] - probs[k] * q1
        out[b, 1] += p2[k] * diet[0] + 2.0 * p1[k] * q1 + probs[k] * q2
        out[a, 1] += p2[k] * diet[1] - 2.0 * p1[k] * q1 - probs[k] * q2
    return out

import math
import numpy as np


def similarity_laplacian(true_diet: "numpy.ndarray", sigma_max: float) -> float:
    """ORCHESTRATOR: positions of the true web, reconstruction from the positions alone, and the Laplacian of the
    similarity (Eq 39) with respect to the positions of the consumers reconstructed by pairs (the sum of the second
    partial derivatives), which fixes the leading bias of the expected similarity under small position errors."""
    Qt = np.asarray(true_diet, dtype=np.float64)
    if not np.isfinite(sigma_max) or sigma_max <= 0.0:
        raise ValueError("sigma_max must be finite and positive")
    y = trophic_positions(Qt)
    n = y.size
    producers = np.where(np.abs(y - 1.0) < 1e-12)[0]
    if producers.size == 0:
        raise ValueError("the web has no primary producer")
    estimated = []                                                # consumers whose rows come from the pair decomposition
    Q = np.zeros((n, n))
    for i in range(n):
        if abs(y[i] - 1.0) < 1e-12:
            continue                                              # producer: feeds on nothing
        if abs(y[i] - 2.0) < 1e-12:
            Q[i, producers] = 1.0 / producers.size               # primary consumer: producers only (source convention)
            continue
        c = y[i] - 1.0
        if np.any((np.arange(n) != i) & (np.abs(y - c) < 1e-12)):
            raise ValueError("a prey sits exactly at the centre of predator %d; the two-prey decomposition does not apply" % i)
        pairs = candidate_pairs(y, i, sigma_max)
        if pairs.shape[0] == 0:
            raise ValueError("predator %d has no candidate prey pair at sigma_max" % i)
        thresholds = pair_thresholds(y, i, pairs)
        weights = marginal_pair_weights(y, i, pairs, thresholds, sigma_max)
        diets = np.array([pair_diet_coefficients(y[d], y[u], c) for d, u in pairs])
        Q[i] = superpose_row(n, pairs, diets, weights)
        estimated.append(i)
    metrics = reconstruction_metrics(Qt, Q)
    theta = metrics[8]
    norm_true, norm_rec = np.linalg.norm(Qt), np.linalg.norm(Q)
    laplacian = 0.0
    for m in estimated:
        dQ, d2Q = np.zeros((n, n)), np.zeros((n, n))
        for i in estimated:
            der = diet_row_derivatives(y, i, sigma_max, m)   # rows of producers and primary consumers are fixed
            dQ[i], d2Q[i] = der[:, 0], der[:, 1]
        # Eq 39: theta = N / (|Qt| |Q|) with N = <Q, Qt>; only the reconstructed matrix moves with the positions
        num1, num2 = np.sum(dQ * Qt), np.sum(d2Q * Qt)
        r1 = np.sum(Q * dQ) / norm_rec
        r2 = (np.sum(dQ * dQ) + np.sum(Q * d2Q)) / norm_rec - np.sum(Q * dQ) ** 2 / norm_rec ** 3
        num0 = theta * norm_true * norm_rec
        laplacian += (num2 / norm_rec - 2.0 * num1 * r1 / norm_rec ** 2 - num0 * r2 / norm_rec ** 2
                      + 2.0 * num0 * r1 ** 2 / norm_rec ** 3) / norm_true
    return float(laplacian)
SCICODE_GOLD_EOF
