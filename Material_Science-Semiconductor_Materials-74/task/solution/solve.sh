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

def compute_diamond_metrics(
    primal_edge_length: float,
    dual_edge_length: float,
    angle_radians: float,
    primal_normal: np.ndarray,
    dual_normal: np.ndarray,
) -> np.ndarray:
    if not np.isfinite(primal_edge_length) or primal_edge_length <= 0.0:
        raise ValueError("primal_edge_length must be finite and > 0")
    if not np.isfinite(dual_edge_length) or dual_edge_length <= 0.0:
        raise ValueError("dual_edge_length must be finite and > 0")
    if not np.isfinite(angle_radians):
        raise ValueError("angle_radians must be finite")

    primal = np.asarray(primal_normal, dtype=float)
    dual = np.asarray(dual_normal, dtype=float)
    if primal.shape != (2,) or dual.shape != (2,):
        raise ValueError("Both normals must be arrays of shape (2,)")
    if not np.all(np.isfinite(primal)) or not np.all(np.isfinite(dual)):
        raise ValueError("Both normals must be finite")

    area = 0.5 * float(primal_edge_length) * float(dual_edge_length) * np.sin(angle_radians)
    if area <= 0.0:
        raise ValueError("angle_radians must define a strictly positive diamond area")

    coupling = float(np.dot(primal, dual))
    return np.array([float(area), coupling], dtype=float)

import numpy as np

def evaluate_bernoulli(argument: float) -> float:
    t = float(argument)
    if not np.isfinite(t):
        raise ValueError("argument must be finite")
    if abs(t) < 1.0e-6:
        return float(1.0 - t / 2.0 + t**2 / 12.0 - t**4 / 720.0)
    return float(t / np.expm1(t))

import numpy as np

def compute_primal_electron_flux(
    area: float,
    coupling: float,
    primal_length: float,
    dual_length: float,
    diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    n_k: float,
    n_l: float,
    n_kstar: float,
    n_lstar: float,
) -> float:
    for label, positive in (
        ("area", area),
        ("primal_length", primal_length),
        ("dual_length", dual_length),
        ("diffusion", diffusion),
    ):
        if not np.isfinite(positive) or positive <= 0.0:
            raise ValueError(f"{label} must be finite and > 0")
    for finite in (coupling, psi_k, psi_l, psi_kstar, psi_lstar,
                   n_k, n_l, n_kstar, n_lstar):
        if not np.isfinite(finite):
            raise ValueError("coupling, potentials and densities must be finite")

    delta_primal = float(psi_k) - float(psi_l)
    delta_dual = float(psi_kstar) - float(psi_lstar)
    primal_bracket = evaluate_bernoulli(delta_primal) * n_k - evaluate_bernoulli(-delta_primal) * n_l
    dual_bracket = evaluate_bernoulli(delta_dual) * n_kstar - evaluate_bernoulli(-delta_dual) * n_lstar
    return float(
        primal_length**2 * diffusion * primal_bracket / (2.0 * area)
        + primal_length * dual_length * diffusion * coupling * dual_bracket / (2.0 * area)
    )

import numpy as np

def compute_dual_electron_flux(
    area: float,
    coupling: float,
    primal_length: float,
    dual_length: float,
    diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    n_k: float,
    n_l: float,
    n_kstar: float,
    n_lstar: float,
) -> float:
    for label, positive in (
        ("area", area),
        ("primal_length", primal_length),
        ("dual_length", dual_length),
        ("diffusion", diffusion),
    ):
        if not np.isfinite(positive) or positive <= 0.0:
            raise ValueError(f"{label} must be finite and > 0")
    for finite in (coupling, psi_k, psi_l, psi_kstar, psi_lstar,
                   n_k, n_l, n_kstar, n_lstar):
        if not np.isfinite(finite):
            raise ValueError("coupling, potentials and densities must be finite")

    delta_primal = float(psi_k) - float(psi_l)
    delta_dual = float(psi_kstar) - float(psi_lstar)
    primal_bracket = evaluate_bernoulli(delta_primal) * n_k - evaluate_bernoulli(-delta_primal) * n_l
    dual_bracket = evaluate_bernoulli(delta_dual) * n_kstar - evaluate_bernoulli(-delta_dual) * n_lstar
    return float(
        primal_length * dual_length * diffusion * coupling * primal_bracket / (2.0 * area)
        + dual_length**2 * diffusion * dual_bracket / (2.0 * area)
    )

import numpy as np

def compute_primal_hole_flux(
    area: float,
    coupling: float,
    primal_length: float,
    dual_length: float,
    diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    p_k: float,
    p_l: float,
    p_kstar: float,
    p_lstar: float,
) -> float:
    for label, positive in (
        ("area", area),
        ("primal_length", primal_length),
        ("dual_length", dual_length),
        ("diffusion", diffusion),
    ):
        if not np.isfinite(positive) or positive <= 0.0:
            raise ValueError(f"{label} must be finite and > 0")
    for finite in (coupling, psi_k, psi_l, psi_kstar, psi_lstar,
                   p_k, p_l, p_kstar, p_lstar):
        if not np.isfinite(finite):
            raise ValueError("coupling, potentials and densities must be finite")

    delta_primal = float(psi_k) - float(psi_l)
    delta_dual = float(psi_kstar) - float(psi_lstar)
    primal_bracket = evaluate_bernoulli(-delta_primal) * p_k - evaluate_bernoulli(delta_primal) * p_l
    dual_bracket = evaluate_bernoulli(-delta_dual) * p_kstar - evaluate_bernoulli(delta_dual) * p_lstar
    return float(
        primal_length**2 * diffusion * primal_bracket / (2.0 * area)
        + primal_length * dual_length * diffusion * coupling * dual_bracket / (2.0 * area)
    )

import numpy as np

def compute_dual_hole_flux(
    area: float,
    coupling: float,
    primal_length: float,
    dual_length: float,
    diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    p_k: float,
    p_l: float,
    p_kstar: float,
    p_lstar: float,
) -> float:
    for label, positive in (
        ("area", area),
        ("primal_length", primal_length),
        ("dual_length", dual_length),
        ("diffusion", diffusion),
    ):
        if not np.isfinite(positive) or positive <= 0.0:
            raise ValueError(f"{label} must be finite and > 0")
    for finite in (coupling, psi_k, psi_l, psi_kstar, psi_lstar,
                   p_k, p_l, p_kstar, p_lstar):
        if not np.isfinite(finite):
            raise ValueError("coupling, potentials and densities must be finite")

    delta_primal = float(psi_k) - float(psi_l)
    delta_dual = float(psi_kstar) - float(psi_lstar)
    primal_bracket = evaluate_bernoulli(-delta_primal) * p_k - evaluate_bernoulli(delta_primal) * p_l
    dual_bracket = evaluate_bernoulli(-delta_dual) * p_kstar - evaluate_bernoulli(delta_dual) * p_lstar
    return float(
        primal_length * dual_length * diffusion * coupling * primal_bracket / (2.0 * area)
        + dual_length**2 * diffusion * dual_bracket / (2.0 * area)
    )

import numpy as np

def _diamond_flux(
    area, coupling, primal_length, dual_length, diffusion,
    psi_a, psi_b, psi_astar, psi_bstar,
    dens_a, dens_b, dens_astar, dens_bstar,
    charge_sign, primal_side,
):
    """One harmonic-average diamond flux for either carrier and either direction."""
    delta_primal = charge_sign * (float(psi_a) - float(psi_b))
    delta_dual = charge_sign * (float(psi_astar) - float(psi_bstar))
    primal_bracket = evaluate_bernoulli(delta_primal) * dens_a - evaluate_bernoulli(-delta_primal) * dens_b
    dual_bracket = evaluate_bernoulli(delta_dual) * dens_astar - evaluate_bernoulli(-delta_dual) * dens_bstar
    if primal_side:
        return float(
            primal_length**2 * diffusion * primal_bracket / (2.0 * area)
            + primal_length * dual_length * diffusion * coupling * dual_bracket / (2.0 * area)
        )
    return float(
        primal_length * dual_length * diffusion * coupling * primal_bracket / (2.0 * area)
        + dual_length**2 * diffusion * dual_bracket / (2.0 * area)
    )


def compute_reversed_interface_fluxes(
    primal_length: float,
    dual_length: float,
    angle_radians: float,
    primal_normal: np.ndarray,
    dual_normal: np.ndarray,
    electron_diffusion: float,
    hole_diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    n_k: float,
    n_l: float,
    n_kstar: float,
    n_lstar: float,
    p_k: float,
    p_l: float,
    p_kstar: float,
    p_lstar: float,
) -> np.ndarray:
    for label, positive in (
        ("primal_length", primal_length),
        ("dual_length", dual_length),
        ("electron_diffusion", electron_diffusion),
        ("hole_diffusion", hole_diffusion),
    ):
        if not np.isfinite(positive) or positive <= 0.0:
            raise ValueError(f"{label} must be finite and > 0")
    if not np.isfinite(angle_radians):
        raise ValueError("angle_radians must be finite")

    forward_primal = np.asarray(primal_normal, dtype=float)
    forward_dual = np.asarray(dual_normal, dtype=float)
    if forward_primal.shape != (2,) or forward_dual.shape != (2,):
        raise ValueError("Both normals must be arrays of shape (2,)")
    if not np.all(np.isfinite(forward_primal)) or not np.all(np.isfinite(forward_dual)):
        raise ValueError("Both normals must be finite")
    for finite in (psi_k, psi_l, psi_kstar, psi_lstar,
                   n_k, n_l, n_kstar, n_lstar, p_k, p_l, p_kstar, p_lstar):
        if not np.isfinite(finite):
            raise ValueError("All potentials and densities must be finite")

    area = 0.5 * float(primal_length) * float(dual_length) * np.sin(angle_radians)
    if area <= 0.0:
        raise ValueError("angle_radians must define a strictly positive diamond area")

    # Reverse both outward normals; their projection on each other is unchanged.
    reversed_primal = -forward_primal
    reversed_dual = -forward_dual
    coupling = float(np.dot(reversed_primal, reversed_dual))

    # Exchange K with L and K* with L*, then re-evaluate the same discretization.
    electron_primal = _diamond_flux(
        area, coupling, primal_length, dual_length, electron_diffusion,
        psi_l, psi_k, psi_lstar, psi_kstar, n_l, n_k, n_lstar, n_kstar, 1.0, True,
    )
    electron_dual = _diamond_flux(
        area, coupling, primal_length, dual_length, electron_diffusion,
        psi_l, psi_k, psi_lstar, psi_kstar, n_l, n_k, n_lstar, n_kstar, 1.0, False,
    )
    hole_primal = _diamond_flux(
        area, coupling, primal_length, dual_length, hole_diffusion,
        psi_l, psi_k, psi_lstar, psi_kstar, p_l, p_k, p_lstar, p_kstar, -1.0, True,
    )
    hole_dual = _diamond_flux(
        area, coupling, primal_length, dual_length, hole_diffusion,
        psi_l, psi_k, psi_lstar, psi_kstar, p_l, p_k, p_lstar, p_kstar, -1.0, False,
    )
    return np.array(
        [electron_primal, electron_dual, hole_primal, hole_dual], dtype=float
    )

import numpy as np

def _edge_flux(edge_length, area, diffusion, delta_psi, dens_a, dens_b, charge_sign):
    """One decoupled exponentially fitted edge flux."""
    t = charge_sign * float(delta_psi)
    bracket = evaluate_bernoulli(t) * dens_a - evaluate_bernoulli(-t) * dens_b
    return float(edge_length**2 * diffusion * bracket / (2.0 * area))


def compute_orthogonal_sg_fluxes(
    area: float,
    primal_length: float,
    dual_length: float,
    electron_diffusion: float,
    hole_diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    n_k: float,
    n_l: float,
    n_kstar: float,
    n_lstar: float,
    p_k: float,
    p_l: float,
    p_kstar: float,
    p_lstar: float,
) -> np.ndarray:
    for label, positive in (
        ("area", area),
        ("primal_length", primal_length),
        ("dual_length", dual_length),
        ("electron_diffusion", electron_diffusion),
        ("hole_diffusion", hole_diffusion),
    ):
        if not np.isfinite(positive) or positive <= 0.0:
            raise ValueError(f"{label} must be finite and > 0")
    for finite in (psi_k, psi_l, psi_kstar, psi_lstar,
                   n_k, n_l, n_kstar, n_lstar, p_k, p_l, p_kstar, p_lstar):
        if not np.isfinite(finite):
            raise ValueError("All potentials and densities must be finite")

    delta_primal = float(psi_k) - float(psi_l)
    delta_dual = float(psi_kstar) - float(psi_lstar)
    return np.array(
        [
            _edge_flux(primal_length, area, electron_diffusion, delta_primal, n_k, n_l, 1.0),
            _edge_flux(dual_length, area, electron_diffusion, delta_dual, n_kstar, n_lstar, 1.0),
            _edge_flux(primal_length, area, hole_diffusion, delta_primal, p_k, p_l, -1.0),
            _edge_flux(dual_length, area, hole_diffusion, delta_dual, p_kstar, p_lstar, -1.0),
        ],
        dtype=float,
    )

import numpy as np

def _frozen_coefficient(potential_a: float, potential_b: float, sign: float) -> float:
    """Reciprocal mean of exp(-z u) along an edge, u linear between a and b."""
    return float(np.exp(sign * potential_a) * evaluate_bernoulli(sign * (potential_a - potential_b)))


def compute_consistency_defect(
    area: float,
    primal_length: float,
    diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    dens_k: float,
    dens_l: float,
    charge_sign: float,
) -> np.ndarray:
    for label, positive in (
        ("area", area),
        ("primal_length", primal_length),
        ("diffusion", diffusion),
    ):
        if not np.isfinite(positive) or positive <= 0.0:
            raise ValueError(f"{label} must be finite and > 0")
    if charge_sign not in (1.0, -1.0):
        raise ValueError("charge_sign must be +1.0 for electrons or -1.0 for holes")
    for finite in (psi_k, psi_l, psi_kstar, psi_lstar, dens_k, dens_l):
        if not np.isfinite(finite):
            raise ValueError("All potentials and densities must be finite")

    sign = float(charge_sign)
    coefficient_primal = _frozen_coefficient(psi_kstar, psi_lstar, sign)
    coefficient_dual = _frozen_coefficient(psi_k, psi_l, sign)
    phi_k = float(dens_k) * np.exp(-sign * float(psi_k))
    phi_l = float(dens_l) * np.exp(-sign * float(psi_l))
    defect = (
        primal_length**2 * diffusion
        * (coefficient_primal - coefficient_dual) * (phi_k - phi_l)
        / (2.0 * area)
    )
    return np.array([coefficient_primal, coefficient_dual, float(defect)], dtype=float)

import numpy as np

def compute_signed_current_defect(
    primal_length: float,
    dual_length: float,
    angle_radians: float,
    primal_normal: np.ndarray,
    dual_normal: np.ndarray,
    angle_radians_perp: float,
    primal_normal_perp: np.ndarray,
    dual_normal_perp: np.ndarray,
    electron_diffusion: float,
    hole_diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    n_k: float,
    n_l: float,
    n_kstar: float,
    n_lstar: float,
    p_k: float,
    p_l: float,
    p_kstar: float,
    p_lstar: float,
) -> float:
    for label, positive in (
        ("primal_length", primal_length),
        ("dual_length", dual_length),
        ("electron_diffusion", electron_diffusion),
        ("hole_diffusion", hole_diffusion),
    ):
        if not np.isfinite(positive) or positive <= 0.0:
            raise ValueError(f"{label} must be finite and > 0")
    for finite in (psi_k, psi_l, psi_kstar, psi_lstar,
                   n_k, n_l, n_kstar, n_lstar, p_k, p_l, p_kstar, p_lstar):
        if not np.isfinite(finite):
            raise ValueError("All potentials and densities must be finite")

    psi = (psi_k, psi_l, psi_kstar, psi_lstar)
    n = (n_k, n_l, n_kstar, n_lstar)
    p = (p_k, p_l, p_kstar, p_lstar)

    # Step 2: the exponential fitting weights of both directions must be usable.
    weights = [
        evaluate_bernoulli(psi_k - psi_l),
        evaluate_bernoulli(psi_l - psi_k),
        evaluate_bernoulli(psi_kstar - psi_lstar),
        evaluate_bernoulli(psi_lstar - psi_kstar),
    ]
    if not all(np.isfinite(w) and w > 0.0 for w in weights):
        raise ValueError("Exponential fitting weights must be finite and strictly positive")

    # Step 1: geometry of the non-orthogonal diamond.
    area, coupling = compute_diamond_metrics(
        primal_length, dual_length, angle_radians, primal_normal, dual_normal
    )
    area, coupling = float(area), float(coupling)

    # Steps 3-6: the four carrier fluxes in the forward orientation.
    forward = np.array(
        [
            compute_primal_electron_flux(
                area, coupling, primal_length, dual_length, electron_diffusion, *psi, *n),
            compute_dual_electron_flux(
                area, coupling, primal_length, dual_length, electron_diffusion, *psi, *n),
            compute_primal_hole_flux(
                area, coupling, primal_length, dual_length, hole_diffusion, *psi, *p),
            compute_dual_hole_flux(
                area, coupling, primal_length, dual_length, hole_diffusion, *psi, *p),
        ],
        dtype=float,
    )

    # Step 7: the same interface seen by the neighbouring cells must cancel pairwise.
    reverse = np.asarray(
        compute_reversed_interface_fluxes(
            primal_length, dual_length, angle_radians, primal_normal, dual_normal,
            electron_diffusion, hole_diffusion, *psi, *n, *p),
        dtype=float,
    )
    if not np.all(np.abs(forward + reverse) <= 1.0e-12 * (1.0 + np.abs(forward))):
        raise ValueError("Interface fluxes do not cancel pairwise under orientation reversal")

    # Step 8: the auxiliary orthogonal diamond must give finite decoupled fluxes.
    area_perp, _ = compute_diamond_metrics(
        primal_length, dual_length, angle_radians_perp,
        primal_normal_perp, dual_normal_perp,
    )
    orthogonal = np.asarray(
        compute_orthogonal_sg_fluxes(
            float(area_perp), primal_length, dual_length,
            electron_diffusion, hole_diffusion, *psi, *n, *p),
        dtype=float,
    )
    if not np.all(np.isfinite(orthogonal)):
        raise ValueError("Orthogonal-limit fluxes must be finite")

    # Step 9: primal-flux defect of each carrier.
    electron_defect = np.asarray(
        compute_consistency_defect(
            area, primal_length, electron_diffusion, *psi, n_k, n_l, 1.0),
        dtype=float,
    )[2]
    hole_defect = np.asarray(
        compute_consistency_defect(
            area, primal_length, hole_diffusion, *psi, p_k, p_l, -1.0),
        dtype=float,
    )[2]

    # Step 10: defect of the signed electrical current. The electron carrier flux
    # represents -J_n / q and the hole carrier flux +J_p / q, so the carriers enter
    # the current with opposite signs.
    return float(-electron_defect + hole_defect)
SCICODE_GOLD_EOF
