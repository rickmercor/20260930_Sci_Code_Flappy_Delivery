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

def estimate_derivatives(times: np.ndarray, concentrations: np.ndarray) -> np.ndarray:
    """Reference nonuniform finite-difference implementation."""
    t = np.asarray(times, dtype=float)
    c = np.asarray(concentrations, dtype=float)

    if t.ndim != 1 or t.size < 3:
        raise ValueError("times must be one-dimensional with at least three entries")
    if c.ndim != 2 or c.shape[0] != t.size or c.shape[1] < 1:
        raise ValueError("concentrations must have shape (n_times, n_species)")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(c)):
        raise ValueError("inputs must be finite")

    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("times must be strictly increasing")

    derivative = np.empty_like(c, dtype=float)
    derivative[0] = (c[1] - c[0]) / dt[0]
    derivative[-1] = (c[-1] - c[-2]) / dt[-1]

    h_minus = t[1:-1] - t[:-2]
    h_plus = t[2:] - t[1:-1]

    left = -h_plus / (h_minus * (h_minus + h_plus))
    center = (h_plus - h_minus) / (h_minus * h_plus)
    right = h_minus / (h_plus * (h_minus + h_plus))

    derivative[1:-1] = (
        left[:, None] * c[:-2]
        + center[:, None] * c[1:-1]
        + right[:, None] * c[2:]
    )
    return derivative

def mass_action_features(concentrations: np.ndarray, reactions: np.ndarray) -> np.ndarray:
    """Reference mass-action feature construction."""
    c = np.asarray(concentrations, dtype=float)
    r = np.asarray(reactions)

    if c.ndim != 2 or c.shape[0] < 1 or c.shape[1] < 1:
        raise ValueError("concentrations must be a nonempty 2D array")
    if r.ndim != 2 or r.shape[0] < 1 or r.shape[1] != 2 * c.shape[1]:
        raise ValueError("reactions must have shape (n_reactions, 2*n_species)")
    if not np.all(np.isfinite(c)) or np.any(c < 0.0):
        raise ValueError("concentrations must be finite and nonnegative")
    if not np.all(np.isfinite(r)) or np.any(r < 0) or not np.all(r == np.floor(r)):
        raise ValueError("reaction coefficients must be nonnegative integers")

    reactants = r[:, : c.shape[1]].astype(int)
    return np.prod(c[:, None, :] ** reactants[None, :, :], axis=2).astype(float)

def stoichiometric_design(
    features: np.ndarray,
    reactions: np.ndarray,
    derivative_scales: np.ndarray,
) -> np.ndarray:
    """Reference normalized stoichiometric design construction."""
    phi = np.asarray(features, dtype=float)
    r = np.asarray(reactions, dtype=float)
    scales = np.asarray(derivative_scales, dtype=float)

    if phi.ndim != 2 or phi.shape[0] < 1 or phi.shape[1] < 1:
        raise ValueError("features must be a nonempty 2D array")
    if r.ndim != 2 or r.shape[0] != phi.shape[1] or r.shape[1] % 2 != 0:
        raise ValueError("reaction dimensions do not match features")

    n_species = r.shape[1] // 2
    if scales.shape != (n_species,) or np.any(scales <= 0.0):
        raise ValueError("derivative_scales must contain one positive value per species")
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(r)) or not np.all(np.isfinite(scales)):
        raise ValueError("inputs must be finite")

    stoichiometry = r[:, n_species:] - r[:, :n_species]
    tensor = np.einsum("tr,rs->tsr", phi, stoichiometry)
    tensor = tensor / scales[None, :, None]
    return tensor.reshape(phi.shape[0] * n_species, phi.shape[1])

def fit_rate_constants(
    design: np.ndarray,
    derivatives: np.ndarray,
    derivative_scales: np.ndarray,
) -> np.ndarray:
    """Reference simultaneous nonnegative least-squares fit."""
    import numpy as np
    from scipy.optimize import lsq_linear

    x = np.asarray(design, dtype=float)
    d = np.asarray(derivatives, dtype=float)
    scales = np.asarray(derivative_scales, dtype=float)

    if d.ndim != 2 or d.shape[0] < 1 or d.shape[1] < 1:
        raise ValueError("derivatives must be a nonempty 2D array")
    if scales.shape != (d.shape[1],) or np.any(scales <= 0.0):
        raise ValueError("derivative_scales must be positive and match species")
    if x.ndim != 2 or x.shape[0] != d.size or x.shape[1] < 1:
        raise ValueError("design must have n_times*n_species rows")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(d)) or not np.all(np.isfinite(scales)):
        raise ValueError("inputs must be finite")

    target = (d / scales[None, :]).reshape(-1)
    fit = lsq_linear(
        x,
        target,
        bounds=(0.0, np.inf),
        tol=1e-12,
        lsmr_tol="auto",
        max_iter=500,
    )
    if not fit.success:
        raise RuntimeError("nonnegative rate fit did not converge")

    residual = (target - x @ fit.x).reshape(d.shape)
    loss = float(np.mean(np.sum(residual * residual, axis=1)))
    return np.concatenate((fit.x.astype(float), np.array([loss], dtype=float)))

def concentration_loss(
    times: np.ndarray,
    initial_concentrations: np.ndarray,
    observed_concentrations: np.ndarray,
    reactions: np.ndarray,
    rates: np.ndarray,
    concentration_scales: np.ndarray,
) -> float:
    """Reference ODE trajectory loss."""
    import numpy as np
    from scipy.integrate import solve_ivp

    t = np.asarray(times, dtype=float)
    y0 = np.asarray(initial_concentrations, dtype=float)
    yobs = np.asarray(observed_concentrations, dtype=float)
    r = np.asarray(reactions, dtype=float)
    k = np.asarray(rates, dtype=float)
    scales = np.asarray(concentration_scales, dtype=float)

    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if yobs.ndim != 2 or yobs.shape[0] != t.size:
        raise ValueError("observed_concentrations must align with times")

    n_species = yobs.shape[1]
    if y0.shape != (n_species,) or scales.shape != (n_species,):
        raise ValueError("initial concentrations and scales must match species")
    if r.ndim != 2 or r.shape[1] != 2 * n_species or k.shape != (r.shape[0],):
        raise ValueError("reaction and rate dimensions do not match")
    if np.any(y0 < 0.0) or np.any(yobs < 0.0) or np.any(k < 0.0) or np.any(scales <= 0.0):
        raise ValueError("concentrations, rates, and scales must be valid")

    reactants = r[:, :n_species].astype(int)
    stoich = r[:, n_species:] - r[:, :n_species]

    def rhs(_time, y):
        y = np.maximum(y, 0.0)
        phi = np.prod(y[None, :] ** reactants, axis=1)
        return (k * phi) @ stoich

    sol = solve_ivp(
        rhs,
        (float(t[0]), float(t[-1])),
        y0,
        t_eval=t,
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError("ODE integration failed")

    residual = (sol.y.T - yobs) / scales[None, :]
    return float(np.mean(np.sum(residual * residual, axis=1)))

def expression_complexity(reactions: np.ndarray) -> float:
    """Reference expression-complexity score."""
    r = np.asarray(reactions, dtype=float)

    if r.ndim != 2 or r.shape[0] < 1 or r.shape[1] % 2 != 0:
        raise ValueError("reactions must have shape (n_reactions, 2*n_species)")
    if not np.all(np.isfinite(r)) or np.any(r < 0) or not np.all(r == np.floor(r)):
        raise ValueError("reaction coefficients must be nonnegative integers")

    n_species = r.shape[1] // 2
    reactants = r[:, :n_species]
    products = r[:, n_species:]

    order_penalty = np.maximum(np.sum(reactants, axis=1) - 1.0, 0.0)
    stoich_weight = np.sum(reactants + products, axis=1)
    return float(np.sum(1.0 + order_penalty + 0.25 * stoich_weight))

def pareto_front(losses: np.ndarray, complexities: np.ndarray) -> np.ndarray:
    """Reference nondominated-front calculation."""
    loss = np.asarray(losses, dtype=float)
    comp = np.asarray(complexities, dtype=float)

    if loss.ndim != 1 or comp.ndim != 1 or loss.shape != comp.shape or loss.size < 1:
        raise ValueError("losses and complexities must be matching one-dimensional arrays")
    if not np.all(np.isfinite(loss)) or not np.all(np.isfinite(comp)):
        raise ValueError("inputs must be finite")

    keep = np.ones(loss.size, dtype=bool)
    for i in range(loss.size):
        dominated = (
            (loss <= loss[i])
            & (comp <= comp[i])
            & ((loss < loss[i]) | (comp < comp[i]))
        )
        if np.any(dominated):
            keep[i] = False
    return keep

def discover_sisr_mechanism(
    times,
    concentrations,
    candidate_reactions,
    tolerance=0.10,
):
    import numpy as np

    t = np.asarray(times, dtype=float)
    c = np.asarray(concentrations, dtype=float)

    if t.ndim != 1 or t.size < 3 or c.ndim != 2 or c.shape[0] != t.size:
        raise ValueError("times and concentrations have incompatible shapes")
    if np.any(np.diff(t) <= 0.0) or np.any(c < 0.0):
        raise ValueError("times must increase and concentrations must be nonnegative")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")

    dc = estimate_derivatives(t, c)
    dscale = np.maximum(np.max(np.abs(dc), axis=0), 1e-12)
    cscale = np.maximum(np.max(np.abs(c), axis=0), 1e-12)

    derivative_losses = []
    concentration_losses = []
    complexities = []

    for mechanism in candidate_reactions:
        r = np.asarray(mechanism, dtype=float)
        n_species = c.shape[1]
        if r.ndim != 2 or r.shape[1] != 2 * n_species:
            raise ValueError("each mechanism must use [reactants | products] columns")
        try:
            features = mass_action_features(c, r)
            design = stoichiometric_design(features, r, dscale)
            fit = fit_rate_constants(design, dc, dscale)
            rates = np.asarray(fit[:-1], dtype=float)
            derivative_losses.append(float(fit[-1]))
            concentration_losses.append(
                float(concentration_loss(t, c[0], c, r, rates, cscale))
            )
            complexities.append(float(expression_complexity(r)))
        except Exception:
            derivative_losses.append(np.inf)
            concentration_losses.append(np.inf)
            complexities.append(np.inf)

    dloss = np.asarray(derivative_losses)
    closs = np.asarray(concentration_losses)
    comp = np.asarray(complexities)

    best = float(np.min(closs))
    eligible = closs <= best * (1.0 + tolerance) + 1e-12
    pareto = np.asarray(pareto_front(closs, comp), dtype=bool)
    candidates = np.flatnonzero(eligible & pareto)
    if candidates.size == 0:
        candidates = np.array([int(np.argmin(closs))])

    chosen = sorted(
        candidates.tolist(),
        key=lambda i: (comp[i], closs[i], dloss[i], i),
    )[0]
    return int(chosen + 1)
SCICODE_GOLD_EOF
