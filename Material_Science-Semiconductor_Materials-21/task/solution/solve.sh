#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_system_coefficients(kn_ep: float, kn_pe: float, kn_pp: float,
                                        C_e: float = 1.0, C_p: float = 1.0,
                                        v_e: float = 1.0, v_p: float = 1.0) -> np.ndarray:
    import numpy as np

    vals = [float(v) for v in (kn_ep, kn_pe, kn_pp, C_e, C_p, v_e, v_p)]
    if not all(np.isfinite(v) and v > 0.0 for v in vals):
        raise ValueError("Knudsen numbers, capacities and speeds must be finite and > 0")
    kn_ep, kn_pe, kn_pp, C_e, C_p, v_e, v_p = vals

    kn_p = 1.0 / (1.0 / kn_pe + 1.0 / kn_pp)
    den = kn_pe * C_e + kn_ep * C_p
    w_e, w_p = kn_pe / den, kn_ep / den
    kappa_e = v_e * v_e * C_e * kn_ep / 3.0
    kappa_p = v_p * v_p * C_p * kn_p / 3.0
    four_pi = 4.0 * np.pi

    return np.array([
        kn_p, C_p / den, C_e / den,
        kappa_e * w_e, kappa_e * w_p,
        kappa_p * w_e * kn_p / kn_pe,
        kappa_p * w_p * kn_p / kn_pe + (kappa_p / C_p) * kn_p / kn_pp,
        w_e * C_e / four_pi, w_p * C_e / four_pi,
        w_e * C_p * kn_p / (four_pi * kn_pe),
        w_p * C_p * kn_p / (four_pi * kn_pe) + kn_p / (four_pi * kn_pp),
    ], dtype=float)

def compute_angular_moments(u: float) -> np.ndarray:
    import numpy as np

    u = float(u)
    if not (np.isfinite(u) and u >= 0.0):
        raise ValueError("u must be finite and >= 0")
    four_pi = 4.0 * np.pi

    if u <= 1.0e-1:
        u2 = u * u
        s0 = (1.0 - u2 / 3.0 + u2**2 / 5.0 - u2**3 / 7.0
              + u2**4 / 9.0 - u2**5 / 11.0 + u2**6 / 13.0 - u2**7 / 15.0)
        s2 = (1.0 / 3.0 - u2 / 5.0 + u2**2 / 7.0 - u2**3 / 9.0
              + u2**4 / 11.0 - u2**5 / 13.0 + u2**6 / 15.0 - u2**7 / 17.0)
        return np.array([four_pi * s0, four_pi * s2], dtype=float)

    ratio = np.arctan(u) / u
    return np.array([four_pi * ratio, four_pi * (1.0 - ratio) / (u * u)], dtype=float)

def build_unaccelerated_matrix(coefficients: np.ndarray, moments_e: np.ndarray,
                                       moments_p: np.ndarray) -> np.ndarray:
    import numpy as np

    coefficients = np.asarray(coefficients, dtype=float).ravel()
    moments_e = np.asarray(moments_e, dtype=float).ravel()
    moments_p = np.asarray(moments_p, dtype=float).ravel()
    if coefficients.shape != (11,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients must be a finite array of shape (11,)")
    if (moments_e.shape != (2,) or moments_p.shape != (2,)
            or not np.all(np.isfinite(moments_e))
            or not np.all(np.isfinite(moments_p))):
        raise ValueError("moments_e and moments_p must be finite arrays of shape (2,)")

    c1, c2, c3, c4 = coefficients[7:11]
    return np.array([[c1 * moments_e[0], c2 * moments_e[0]],
                     [c3 * moments_p[0], c4 * moments_p[0]]], dtype=float)

def build_accelerated_operators(coefficients: np.ndarray, moments_e: np.ndarray,
                                        moments_p: np.ndarray, unaccelerated: np.ndarray,
                                        kn_ep: float, v_e: float = 1.0,
                                        v_p: float = 1.0) -> np.ndarray:
    import numpy as np

    co, moments_e, moments_p = (np.asarray(x, float).ravel() for x in (coefficients, moments_e, moments_p))
    unaccelerated = np.asarray(unaccelerated, dtype=float)
    arrays = (co, unaccelerated, moments_e, moments_p)
    if (co.shape != (11,) or unaccelerated.shape != (2, 2)
            or moments_e.shape != (2,) or moments_p.shape != (2,)
            or not all(np.all(np.isfinite(array)) for array in arrays)):
        raise ValueError("inputs must have shapes (11,), (2, 2), (2,) and (2,) and be finite")
    kn_ep, v_e, v_p = (float(v) for v in (kn_ep, v_e, v_p))
    if not all(np.isfinite(v) and v > 0.0 for v in (kn_ep, v_e, v_p)):
        raise ValueError("kn_ep, v_e and v_p must be finite and > 0")
    kn_p, G_e, G_p, K_ee, K_ep, K_pe, K_pp, c1, c2, c3, c4 = co

    left = np.array([[G_e + K_ee, -G_p + K_ep], [-G_e + K_pe, G_p + K_pp]], dtype=float)
    g1, g4 = kn_ep * v_e * v_e, kn_p * v_p * v_p
    scaled = np.array([[g1 * c1, g1 * c2], [g4 * c3, g4 * c4]], dtype=float)
    right = -scaled * np.array([[moments_e[1]], [moments_p[1]]]) + (4.0 * np.pi / 3.0) * (scaled @ unaccelerated)
    return np.stack([left, right]).astype(float)

def compute_spectral_radii(unaccelerated: np.ndarray,
                                   operators: np.ndarray) -> np.ndarray:
    import numpy as np

    unaccelerated = np.asarray(unaccelerated, dtype=float)
    operators = np.asarray(operators, dtype=float)
    if unaccelerated.shape != (2, 2) or not np.all(np.isfinite(unaccelerated)):
        raise ValueError("unaccelerated must be a finite array of shape (2, 2)")
    if operators.shape != (2, 2, 2) or not np.all(np.isfinite(operators)):
        raise ValueError("operators must be a finite array of shape (2, 2, 2)")
    left, right = operators[0], operators[1]
    if np.linalg.det(left) == 0.0:
        raise ValueError("the operator on the new amplitudes must be non-singular")

    rho_plain = float(np.max(np.abs(np.linalg.eigvals(unaccelerated))))
    accelerated = np.linalg.solve(left, right)
    return np.array([rho_plain, float(np.max(np.abs(np.linalg.eigvals(accelerated))))],
                    dtype=float)

def compute_convergence_rates(kn_ep: float, kn_pe: float, kn_pp: float,
                                      C_e: float = 1.0, C_p: float = 1.0,
                                      v_e: float = 1.0, v_p: float = 1.0) -> np.ndarray:
    import numpy as np

    scalars = [float(v) for v in (kn_ep, kn_pe, kn_pp, C_e, C_p, v_e, v_p)]
    if not all(np.isfinite(v) and v > 0.0 for v in scalars):
        raise ValueError("Knudsen numbers, capacities and speeds must be finite and > 0")
    kn_ep, kn_pe, kn_pp, C_e, C_p, v_e, v_p = scalars

    # -- Step 01: every scalar coefficient of the coupled system.
    coefficients = compute_system_coefficients(kn_ep, kn_pe, kn_pp,
                                                       C_e, C_p, v_e, v_p)
    kn_p = float(np.asarray(coefficients, dtype=float).ravel()[0])
    # -- Step 02: the solid-angle integrals at the two distinct arguments.
    moments_e = compute_angular_moments(kn_ep * v_e)
    moments_p = compute_angular_moments(kn_p * v_p)
    # -- Steps 03-05: the three operators and their spectral radii.
    unaccelerated = build_unaccelerated_matrix(coefficients, moments_e, moments_p)
    operators = build_accelerated_operators(coefficients, moments_e, moments_p,
                                                    unaccelerated, kn_ep,
                                                    v_e=v_e, v_p=v_p)
    return compute_spectral_radii(unaccelerated, operators)

def run_stability_sweep(kn_min: float = 1e-2, kn_max: float = 1e2,
                                n_points: int = 41, C_e: float = 1.0, C_p: float = 1.0,
                                v_e: float = 1.0, v_p: float = 1.0) -> float:
    import numpy as np

    scalars = [float(v) for v in (kn_min, kn_max, C_e, C_p, v_e, v_p)]
    if not all(np.isfinite(v) and v > 0.0 for v in scalars):
        raise ValueError("sweep ends, capacities and speeds must be finite and > 0")
    if float(kn_min) > float(kn_max):
        raise ValueError("kn_min must not exceed kn_max")
    if not (isinstance(n_points, (int, np.integer)) and not isinstance(n_points, bool)
            and int(n_points) >= 2):
        raise ValueError("n_points must be an integer >= 2")

    # -- Logarithmic spacing with both ends inclusive.  Step 06 supplies the
    #    chained result; the direct oracle composition keeps every earlier
    #    reference step reachable and verifies that the two paths agree.
    grid = np.logspace(np.log10(float(kn_min)), np.log10(float(kn_max)), int(n_points))
    worst = -np.inf
    for kn in grid:
        kn = float(kn)
        coefficients = compute_system_coefficients(
            kn, kn, kn, C_e, C_p, v_e, v_p)
        kn_p = float(np.asarray(coefficients, dtype=float).ravel()[0])
        moments_e = compute_angular_moments(kn * v_e)
        moments_p = compute_angular_moments(kn_p * v_p)
        unaccelerated = build_unaccelerated_matrix(
            coefficients, moments_e, moments_p)
        operators = build_accelerated_operators(
            coefficients, moments_e, moments_p, unaccelerated, kn,
            v_e=v_e, v_p=v_p)
        direct_rates = compute_spectral_radii(unaccelerated, operators)
        rates = compute_convergence_rates(
            kn, kn, kn, C_e=C_e, C_p=C_p, v_e=v_e, v_p=v_p)
        if not np.allclose(direct_rates, rates, rtol=1.0e-12, atol=1.0e-14):
            raise RuntimeError("direct and chained oracle convergence rates disagree")
        worst = max(worst, float(np.asarray(rates, dtype=float).ravel()[1]))
    return float(worst)
SCICODE_GOLD_EOF
