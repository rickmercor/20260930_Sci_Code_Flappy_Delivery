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


def use_probabilities(resource_matrix: "numpy.ndarray") -> "numpy.ndarray":
    """Conditional resource-use probabilities p_ij = N_ij / Y_i (Eq 6); a species with Y_i = 0 gets a zero row."""
    N = np.asarray(resource_matrix, dtype=np.float64)
    if N.ndim != 2 or N.shape[0] < 1 or N.shape[1] < 1:
        raise ValueError("resource_matrix must be a 2-D array with at least one species and one state")
    if not np.all(np.isfinite(N)) or np.any(N < 0.0):
        raise ValueError("resource_matrix entries must be finite and nonnegative")
    if N.sum() <= 0.0:
        raise ValueError("resource_matrix must contain at least one positive entry")
    Y = N.sum(axis=1)
    return np.where(Y[:, None] > 0.0, N / np.where(Y[:, None] > 0.0, Y[:, None], 1.0), 0.0)

import numpy as np


def _xlogx(a):
    """x ln x with the convention 0 ln 0 = 0 (vectorised)."""
    a = np.asarray(a, dtype=np.float64)
    return np.where(a > 0.0, a * np.log(np.where(a > 0.0, a, 1.0)), 0.0)


def resource_entropies(resource_matrix: "numpy.ndarray") -> "numpy.ndarray":
    """[H(X), H(Y), H(XY), H_Y(X)] of Eqs 10-13 in nats, from the pooled probabilities of Eqs 5, 7 and 9."""
    N = np.asarray(resource_matrix, dtype=np.float64)
    if N.ndim != 2 or N.shape[0] < 1 or N.shape[1] < 1:
        raise ValueError("resource_matrix must be a 2-D array with at least one species and one state")
    if not np.all(np.isfinite(N)) or np.any(N < 0.0):
        raise ValueError("resource_matrix entries must be finite and nonnegative")
    if N.sum() <= 0.0:
        raise ValueError("resource_matrix must contain at least one positive entry")
    Z = N.sum()
    P = N.sum(axis=0) / Z                                   # marginal of the resource states
    Q = N.sum(axis=1) / Z                                   # marginal of the species
    pi = N / Z
    HX = -_xlogx(P).sum()
    HY = -_xlogx(Q).sum()
    HXY = -_xlogx(pi).sum()
    return np.array([HX, HY, HXY, HXY - HY])

import numpy as np


def _xlogx(a):
    """x ln x with the convention 0 ln 0 = 0 (vectorised)."""
    a = np.asarray(a, dtype=np.float64)
    return np.where(a > 0.0, a * np.log(np.where(a > 0.0, a, 1.0)), 0.0)


def _check_matrix(resource_matrix):
    N = np.asarray(resource_matrix, dtype=np.float64)
    if N.ndim != 2 or N.shape[0] < 1 or N.shape[1] < 1:
        raise ValueError("resource_matrix must be a 2-D array with at least one species and one state")
    if not np.all(np.isfinite(N)) or np.any(N < 0.0):
        raise ValueError("resource_matrix entries must be finite and nonnegative")
    if N.sum() <= 0.0:
        raise ValueError("resource_matrix must contain at least one positive entry")
    return N


def state_contributions(resource_matrix: "numpy.ndarray", use_probs: "numpy.ndarray",
                                entropies: "numpy.ndarray") -> "numpy.ndarray":
    """delta_j of Eq 16: the contribution of resource state j to the standardized resource heterogeneity M(X)."""
    N = _check_matrix(resource_matrix)
    p = np.asarray(use_probs, dtype=np.float64)
    ent = np.asarray(entropies, dtype=np.float64)
    if p.shape != N.shape:
        raise ValueError("use_probs must have the shape of resource_matrix")
    if ent.shape != (4,) or not np.all(np.isfinite(ent)):
        raise ValueError("entropies must be the finite array [H(X), H(Y), H(XY), H_Y(X)]")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("use_probs must be finite and nonnegative")
    HX = float(ent[0])
    if HX <= 0.0:
        raise ValueError("H(X) must be positive: the matrix needs at least two occupied resource states")
    Z = N.sum()
    pi = N / Z
    P = N.sum(axis=0) / Z
    # sum_i pi_ij ln p_ij - P_j ln P_j = sum_i pi_ij ln(p_ij / P_j): the per-state share of the shared
    # information m(X) = H(X) - H_Y(X); the states sum to M(X) = m(X)/H(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(p > 0.0, pi * np.log(np.where(p > 0.0, p, 1.0)), 0.0).sum(axis=0)
    return (term - _xlogx(P)) / HX

def _state_contributions_error_code(resource_matrix: "numpy.ndarray", use_probs: "numpy.ndarray",
                                           entropies: "numpy.ndarray") -> int:
    """0 if state_contributions accepts these inputs, 1 if it raises ValueError."""
    try:
        state_contributions(resource_matrix, use_probs, entropies)
        return 0
    except ValueError:
        return 1

import numpy as np


def modified_weights(contributions: "numpy.ndarray", n_occupied: int) -> "numpy.ndarray":
    """ed_j of Eq 33: exp(delta_j r'/r) normalised to unit sum, r' the occupied states of the complete matrix."""
    delta = np.asarray(contributions, dtype=np.float64)
    if delta.ndim != 1 or delta.size < 1 or not np.all(np.isfinite(delta)):
        raise ValueError("contributions must be a non-empty finite 1-D array")
    r = delta.size
    if int(n_occupied) != n_occupied or not (1 <= n_occupied <= r):
        raise ValueError("n_occupied must be an integer between 1 and the number of resource states")
    # the exponent is scaled by the occupancy fraction r'/r of the COMPLETE matrix (not of a reduced one)
    w = np.exp(delta * (float(int(n_occupied)) / r))
    return w / w.sum()

def _modified_weights_error_code(contributions: "numpy.ndarray", n_occupied: int) -> int:
    """0 if modified_weights accepts these inputs, 1 if it raises ValueError."""
    try:
        modified_weights(contributions, n_occupied)
        return 0
    except ValueError:
        return 1

import numpy as np


def _xlogx_geometry(a):
    a = np.asarray(a, dtype=np.float64)
    return np.where(a > 0.0, a * np.log(np.where(a > 0.0, a, 1.0)), 0.0)


def _adjusted_geometry(N, weights, k):
    totals = (weights[None, :] * float(k) * N).sum(axis=1)
    if np.any((N.sum(axis=1) > 0.0) & (totals <= 0.0)):
        raise ValueError("positive species abundance must have positive weighted abundance")
    return np.where(totals[:, None] > 0.0,
                    N / np.where(totals[:, None] > 0.0, totals[:, None], 1.0), 0.0)


def _breadth_geometry(row, weights, k):
    if row.sum() <= 0.0:
        raise ValueError("every focal species must use at least one resource state")
    return float(-(float(k) / np.log(float(k))) * (weights * _xlogx_geometry(row)).sum())


def _overlap_geometry(a, b, weights, k):
    if a.sum() <= 0.0 or b.sum() <= 0.0:
        raise ValueError("both focal species must use at least one resource state")
    bracket = _xlogx_geometry(a) + _xlogx_geometry(b) - _xlogx_geometry(a + b)
    return float(-(weights * float(k) * bracket).sum() / (2.0 * np.log(2.0)))


def noncircular_niche_geometry(resource_matrix: "numpy.ndarray", k: float) -> "numpy.ndarray":
    N = np.asarray(resource_matrix, dtype=np.float64)
    if (N.ndim != 2 or N.shape[0] < 3 or N.shape[1] < 2
            or not np.all(np.isfinite(N)) or np.any(N < 0.0)
            or np.any(N.sum(axis=1) <= 0.0) or not np.isfinite(k) or k <= 1.0):
        raise ValueError("a finite nonnegative matrix with three positive species and k > 1 is required")
    occupied = int((N.sum(axis=0) > 0.0).sum())
    if occupied < 2:
        raise ValueError("at least two resource states must be occupied")
    n = N.shape[0]
    result = np.eye(n, dtype=np.float64)

    def _factors(reduced):
        p = use_probabilities(reduced)
        ent = resource_entropies(reduced)
        delta = state_contributions(reduced, p, ent)
        return modified_weights(delta, occupied)

    for i in range(n):
        weights = _factors(np.delete(N, i, axis=0))
        row = _adjusted_geometry(N[i:i + 1], weights, k)[0]
        result[i, i] = _breadth_geometry(row, weights, k)
    for i in range(n):
        for h in range(i):
            weights = _factors(np.delete(N, [h, i], axis=0))
            rows = _adjusted_geometry(N[[h, i]], weights, k)
            result[h, i] = result[i, h] = _overlap_geometry(rows[0], rows[1], weights, k)
    return result

import numpy as np


def effective_competition(base_competition: "numpy.ndarray", geometry: "numpy.ndarray",
                                  coupling: "numpy.ndarray") -> "numpy.ndarray":
    B = np.asarray(base_competition, dtype=np.float64)
    G = np.asarray(geometry, dtype=np.float64)
    C = np.asarray(coupling, dtype=np.float64)
    if (B.ndim != 3 or B.shape[0] < 1 or B.shape[1] != B.shape[2]
            or B.shape[1] < 2 or G.shape != B.shape[1:] or C.shape != (B.shape[0], 2)
            or any(not np.all(np.isfinite(x)) for x in (B, G, C))
            or np.any(B < 0.0) or np.max(np.abs(np.diagonal(B, axis1=1, axis2=2))) > 1e-12
            or np.max(np.abs(G-G.T)) > 1e-10):
        raise ValueError("aligned finite competition, symmetric geometry and coupling are required")
    n = len(G)
    beta = np.diag(G)
    beta0 = beta-beta.mean()
    mask = ~np.eye(n, dtype=bool)
    mean_overlap = G[mask].mean()
    centered_overlap = G-mean_overlap
    result = B.copy()
    for s in range(len(B)):
        exponent = (C[s, 0]*centered_overlap
                    + C[s, 1]*(beta0[None, :]-beta0[:, None]))
        result[s] *= np.exp(exponent)
        np.fill_diagonal(result[s], 0.0)
    return result

import numpy as np


def effective_regulation(base_designs: "numpy.ndarray", geometry: "numpy.ndarray",
                                 breadth_response: "numpy.ndarray") -> "numpy.ndarray":
    D = np.asarray(base_designs, dtype=np.float64)
    G = np.asarray(geometry, dtype=np.float64)
    q = np.asarray(breadth_response, dtype=np.float64)
    if (D.ndim != 2 or D.shape[0] < 1 or D.shape[1] < 2
            or G.shape != (D.shape[1], D.shape[1]) or q.shape != (D.shape[0],)
            or any(not np.all(np.isfinite(x)) for x in (D, G, q))
            or np.any(D <= 0.0) or np.max(np.abs(G-G.T)) > 1e-10):
        raise ValueError("positive plans, aligned symmetric geometry and one response per plan are required")
    beta = np.diag(G)
    return D*np.exp(q[:, None]*(beta-beta.mean())[None, :])

import numpy as np


def _largest_real_certificate(matrix):
    values = np.linalg.eigvals(matrix)
    real = values.real[np.abs(values.imag) <= 1e-8*np.maximum(1.0, np.abs(values.real))]
    return float(max(0.0, np.max(real, initial=0.0)))


def _certificate_parts(interactions):
    K = np.asarray(interactions, dtype=np.float64)
    n = len(K)
    mean = K.mean(axis=0)
    centered = K-mean[None, :]
    basis = np.ones((n, 1), dtype=np.float64)/np.sqrt(n)
    for j in range(n-1):
        vector = centered@basis[:, j]
        for _ in range(2):
            vector -= basis@(basis.T@vector)
        size = np.linalg.norm(vector)
        if size <= 1e-11*max(1.0, np.linalg.norm(centered)):
            break
        basis = np.column_stack([basis, vector/size])
    baseline = _largest_real_certificate(-basis.T@centered@basis)
    parts = [baseline]
    for i in range(n):
        keep = np.arange(n) != i
        reduced = (np.ones((n-1, 1))*K[i, keep][None, :]
                   - K[np.ix_(keep, keep)])
        value = _largest_real_certificate(reduced) if n > 1 else 0.0
        parts.append(value if value > baseline+1e-10 else 0.0)
    return np.asarray(parts, dtype=np.float64)


def coexistence_certificates(competition: "numpy.ndarray", regulation: "numpy.ndarray") -> "numpy.ndarray":
    B = np.asarray(competition, dtype=np.float64)
    d = np.asarray(regulation, dtype=np.float64)
    if (B.ndim != 3 or B.shape[0] < 1 or B.shape[1] != B.shape[2]
            or B.shape[1] < 2 or d.shape != (B.shape[1],)
            or not np.all(np.isfinite(B)) or not np.all(np.isfinite(d))
            or np.any(B < 0.0) or np.any(d <= 0.0)
            or np.max(np.abs(np.diagonal(B, axis1=1, axis2=2))) > 1e-12):
        raise ValueError("a finite nonnegative condition panel and positive regulation are required")
    rows = []
    for base in B:
        interactions = base/d[None, :]
        parts = _certificate_parts(interactions)
        feasibility = float(parts.max())
        stability = float(max(0.0, -np.linalg.eigvalsh((interactions+interactions.T)/2.0)[0]))
        rows.append(np.r_[feasibility, stability, parts])
    return np.asarray(rows, dtype=np.float64)

import numpy as np
from numpy.polynomial import Chebyshev as _Chebyshev


def _canonical_refuge(interactions):
    mean = interactions.mean(axis=0)
    return np.vstack([interactions-mean[None, :], mean])


def _response_refuge(packed, regulation, loadings, effort):
    centered, mean = packed[:-1], packed[-1]
    try:
        solved = np.linalg.solve(effort*np.eye(len(regulation))+centered,
                                 np.column_stack([np.ones(len(regulation)), loadings]))
    except np.linalg.LinAlgError as exc:
        raise ValueError("singular canonical response system") from exc
    denominator = 1.0+mean@solved[:, 0]
    if abs(denominator) <= 1e-13:
        raise ValueError("singular full response system")
    corrected = solved-np.outer(solved[:, 0], mean@solved)/denominator
    return corrected/regulation[:, None]


def _radii_refuge(response, reserves, cholesky):
    lengths = np.linalg.norm(response[:, 1:]@cholesky, axis=1)
    if np.any(lengths == 0.0):
        raise ValueError("every population needs a nonzero climate response")
    return (response[:, 0]-reserves)/lengths


def _optimize_refuge_exact(packed_panel, regulation, loadings, reserves,
                           cholesky, lower, upper):
    matrices = packed_panel[:, :-1]+packed_panel[:, -1, None, :]
    transformed_loadings = loadings@cholesky

    def _values(effort):
        return np.concatenate([
            _radii_refuge(_response_refuge(packed, regulation, forcing, effort),
                          reserves, cholesky)
            for packed, forcing in zip(packed_panel, loadings)
        ])

    if lower == upper:
        radius = float(_values(lower).min())
        return np.array([lower, radius, radius/lower], dtype=np.float64)
    n = len(regulation)
    nodes = np.cos((np.arange(n+1)+0.5)*np.pi/(n+1))
    efforts = (upper+lower)/2.0+(upper-lower)/2.0*nodes
    numerators, variances = [], []
    for packed, interactions, forcing in zip(packed_panel, matrices, transformed_loadings):
        samples = []
        for effort in efforts:
            response = _response_refuge(packed, regulation, forcing, effort)
            response[:, 0] -= reserves
            samples.append(response*np.linalg.det(effort*np.eye(n)+interactions))
        samples = np.asarray(samples)
        samples /= max(1e-100, np.max(np.abs(samples)))
        for i in range(n):
            numerators.append(_Chebyshev.fit(nodes, samples[:, i, 0], n, domain=[-1, 1]))
            terms = [_Chebyshev.fit(nodes, samples[:, i, j], n-1, domain=[-1, 1])
                     for j in range(1, forcing.shape[1]+1)]
            variances.append(sum((term*term for term in terms), _Chebyshev([0.0]))
                             *_Chebyshev([(lower+upper)/2.0, (upper-lower)/2.0])**2)
    candidates = [-1.0, 1.0]

    def _add_roots(polynomial):
        coefficients = polynomial.coef.copy()
        while (len(coefficients) > 1
               and abs(coefficients[-1]) < 1e-12*max(1e-100, np.max(np.abs(coefficients)))):
            coefficients = coefficients[:-1]
        roots = _Chebyshev(coefficients).roots()
        candidates.extend(roots.real[(np.abs(roots.imag) < 1e-7)
                                     & (np.abs(roots.real) < 1.0)])

    for numerator, variance in zip(numerators, variances):
        _add_roots(2.0*numerator.deriv()*variance-numerator*variance.deriv())
    for i in range(len(numerators)):
        for j in range(i):
            _add_roots(numerators[i]**2*variances[j]-numerators[j]**2*variances[i])
    candidates = np.asarray(candidates, dtype=np.float64)
    candidate_efforts = np.sort((upper+lower)/2.0+(upper-lower)/2.0*candidates)
    scores = np.asarray([_values(effort).min()/effort for effort in candidate_efforts])
    best = scores.max()
    chosen = np.flatnonzero(scores >= best-1e-12)[0]
    effort = float(candidate_efforts[chosen])
    radius = float(_values(effort).min())
    return np.array([effort, radius, radius/effort], dtype=np.float64)


def select_niche_refuge(resource_matrix: "numpy.ndarray", k: float,
                                base_competition: "numpy.ndarray", base_designs: "numpy.ndarray",
                                loadings: "numpy.ndarray", reserves: "numpy.ndarray",
                                covariance: "numpy.ndarray", upper_efforts: "numpy.ndarray",
                                coupling: "numpy.ndarray", breadth_response: "numpy.ndarray",
                                buffer: float, minimum_effort: float) -> float:
    N = np.asarray(resource_matrix, dtype=np.float64)
    base_B = np.asarray(base_competition, dtype=np.float64)
    base_D = np.asarray(base_designs, dtype=np.float64)
    U = np.asarray(loadings, dtype=np.float64)
    ell = np.asarray(reserves, dtype=np.float64)
    Sigma = np.asarray(covariance, dtype=np.float64)
    upper = np.asarray(upper_efforts, dtype=np.float64)
    coupling = np.asarray(coupling, dtype=np.float64)
    response = np.asarray(breadth_response, dtype=np.float64)
    if (base_B.ndim != 3 or base_B.shape[0] < 1 or base_B.shape[1] != base_B.shape[2]
            or base_D.ndim != 2 or base_D.shape[1] != base_B.shape[1] or base_D.shape[0] < 1
            or U.ndim != 3 or U.shape[:2] != base_B.shape[:2] or U.shape[2] < 1
            or ell.shape != (base_B.shape[1],) or Sigma.shape != (U.shape[2], U.shape[2])
            or upper.shape != (base_D.shape[0],) or coupling.shape != (base_B.shape[0], 2)
            or response.shape != (base_D.shape[0],)
            or any(not np.all(np.isfinite(x)) for x in
                   (N, base_B, base_D, U, ell, Sigma, upper, coupling, response))
            or np.any(base_D <= 0.0) or np.any(ell < 0.0) or np.any(upper <= 0.0)
            or not np.isfinite(buffer) or buffer <= 1.0
            or not np.isfinite(minimum_effort) or minimum_effort <= 0.0):
        raise ValueError("finite aligned ecological inputs and valid policy constants are required")
    try:
        cholesky = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be symmetric positive definite") from exc
    if np.max(np.abs(Sigma-Sigma.T)) > 1e-10:
        raise ValueError("covariance must be symmetric positive definite")

    geometry = noncircular_niche_geometry(N, k)
    competition = effective_competition(base_B, geometry, coupling)
    regulation = effective_regulation(base_D, geometry, response)
    plan_scores = []
    for plan, (slopes, high) in enumerate(zip(regulation, upper)):
        certificates = coexistence_certificates(competition, slopes)
        low = max(float(minimum_effort), float(buffer)*float(certificates[:, :2].max()))
        if low > high:
            plan_scores.append((plan, -np.inf))
            continue
        packed = np.asarray([_canonical_refuge(condition/slopes[None, :])
                             for condition in competition])
        optimum = _optimize_refuge_exact(packed, slopes, U, ell, cholesky, low, float(high))
        plan_scores.append((plan, float(optimum[2]) if optimum[1] >= 0.0 else -np.inf))
    finite = [item for item in plan_scores if np.isfinite(item[1])]
    if not finite:
        return 0.0
    best = max(score for _, score in finite)
    return float(next(score for _, score in finite if score >= best-1e-10))

import numpy as np


def _stress_plan_panel(resource_matrix, k, base_competition, base_designs,
                       loadings, reserves, covariance, upper_efforts,
                       coupling, breadth_response, buffer, minimum_effort):
    geometry = noncircular_niche_geometry(resource_matrix, k)
    competition = effective_competition(base_competition, geometry, coupling)
    regulation = effective_regulation(base_designs, geometry, breadth_response)
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be symmetric positive definite") from exc
    rows = []
    for slopes, high in zip(regulation, upper_efforts):
        certificates = coexistence_certificates(competition, slopes)
        low = max(float(minimum_effort), float(buffer)*float(certificates[:, :2].max()))
        if low > high:
            rows.append([np.nan, np.nan, -np.inf])
            continue
        packed = np.asarray([_canonical_refuge(condition/slopes[None, :])
                             for condition in competition])
        optimum = _optimize_refuge_exact(packed, slopes, loadings, reserves,
                                          cholesky, low, float(high))
        if optimum[1] < 0.0:
            optimum[2] = -np.inf
        rows.append(optimum)
    return np.asarray(rows, dtype=np.float64)


def select_stress_tested_refuge(resource_matrix: "numpy.ndarray", k: float,
                                        base_competition: "numpy.ndarray", base_designs: "numpy.ndarray",
                                        loadings: "numpy.ndarray", reserves: "numpy.ndarray",
                                        covariance: "numpy.ndarray", upper_efforts: "numpy.ndarray",
                                        coupling: "numpy.ndarray", breadth_response: "numpy.ndarray",
                                        buffer: float, minimum_effort: float,
                                        attenuation: float, minimum_wins: int,
                                        regret_weight: float) -> float:
    N = np.asarray(resource_matrix, dtype=np.float64)
    if (N.ndim != 2 or N.shape[0] < 3 or N.shape[1] < 2
            or not np.all(np.isfinite(N)) or np.any(N < 0.0)):
        raise ValueError("resource_matrix must be a finite nonnegative matrix")
    occupied = np.flatnonzero(N.sum(axis=0) > 0.0)
    if (not np.isfinite(attenuation) or not 0.0 < attenuation < 1.0
            or not isinstance(minimum_wins, (int, np.integer))
            or minimum_wins < 1 or minimum_wins > len(occupied)
            or not np.isfinite(regret_weight) or regret_weight < 0.0):
        raise ValueError("invalid stress-audit controls")

    common = (k, np.asarray(base_competition, dtype=np.float64),
              np.asarray(base_designs, dtype=np.float64),
              np.asarray(loadings, dtype=np.float64),
              np.asarray(reserves, dtype=np.float64),
              np.asarray(covariance, dtype=np.float64),
              np.asarray(upper_efforts, dtype=np.float64),
              np.asarray(coupling, dtype=np.float64),
              np.asarray(breadth_response, dtype=np.float64),
              buffer, minimum_effort)
    nominal = _stress_plan_panel(N, *common)
    stress = []
    for state in occupied:
        changed = N.copy()
        changed[:, state] *= attenuation
        stress.append(_stress_plan_panel(changed, *common))
    stress = np.asarray(stress, dtype=np.float64)
    scores = stress[:, :, 2]
    if np.any(~np.any(np.isfinite(scores), axis=1)):
        return 0.0
    best = np.max(np.where(np.isfinite(scores), scores, -np.inf), axis=1)
    decisions = np.full(scores.shape[1], -np.inf, dtype=np.float64)
    for plan in range(scores.shape[1]):
        if not np.isfinite(nominal[plan, 2]) or np.any(~np.isfinite(scores[:, plan])):
            continue
        regrets = best-scores[:, plan]
        wins = int(np.count_nonzero(regrets <= 1e-10))
        if wins < minimum_wins:
            continue
        tail = float(np.sort(regrets)[-2:].mean()) if len(regrets) > 1 else float(regrets[0])
        decisions[plan] = nominal[plan, 2]-regret_weight*tail
    if not np.any(np.isfinite(decisions)):
        return 0.0
    best_decision = float(np.max(decisions))
    chosen = int(np.flatnonzero(decisions >= best_decision-1e-10)[0])
    return float(decisions[chosen])
SCICODE_GOLD_EOF
