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
def weighted_mfp_average(MFP: list[float], weights: list[float]) -> float:
    """Reference implementation."""
    MFP = np.asarray(MFP, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if MFP.ndim != 1 or weights.ndim != 1 or MFP.shape[0] != weights.shape[0]:
        raise ValueError("MFP and weights must be 1D arrays of the same length")
    if MFP.shape[0] < 1:
        raise ValueError("MFP must be non-empty")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    if np.sum(weights) == 0:
        raise ValueError("sum of weights must not be zero")
    return float(np.sum(MFP * weights) / np.sum(weights))

import numpy as np
def d0_from_mfp(MFP_avg: float, kappa: float) -> float:
    """Reference implementation."""
    for name, val in [("MFP_avg", MFP_avg), ("kappa", kappa)]:
        if not (isinstance(val, (int, float)) and np.isfinite(val)):
            raise ValueError(f"{name} must be finite")
    if MFP_avg <= 0 or kappa <= 0:
        raise ValueError("MFP_avg and kappa must be positive")
    return float(kappa * MFP_avg)

import numpy as np
def bulk_mobility_temperature_scaling(mu_bulk_ref: float, T_ref: float, alpha: float, T: float) -> float:
    """Reference implementation."""
    for name, val in [("mu_bulk_ref", mu_bulk_ref), ("T_ref", T_ref), ("alpha", alpha), ("T", T)]:
        if not (isinstance(val, (int, float)) and np.isfinite(val)):
            raise ValueError(f"{name} must be finite")
    if mu_bulk_ref <= 0 or T_ref <= 0 or T <= 0:
        raise ValueError("mu_bulk_ref, T_ref, T must be positive")
    return float(mu_bulk_ref * (T / T_ref) ** (-alpha))

import numpy as np
def diameter_dependent_mobility(mu_bulk_T: float, d: float, d0: float, beta: float) -> float:
    """Reference implementation."""
    for name, val in [("mu_bulk_T", mu_bulk_T), ("d", d), ("d0", d0), ("beta", beta)]:
        if not (isinstance(val, (int, float)) and np.isfinite(val)):
            raise ValueError(f"{name} must be finite")
    if d <= d0:
        raise ValueError("d must exceed d0 for this formula to apply")
    return float(mu_bulk_T * (1 - (d / d0) ** (-beta)))

import numpy as np
def surface_limited_mobility(mu_1D: float, mu_bulk_T: float) -> float:
    """Reference implementation."""
    for name, val in [("mu_1D", mu_1D), ("mu_bulk_T", mu_bulk_T)]:
        if not (isinstance(val, (int, float)) and np.isfinite(val)):
            raise ValueError(f"{name} must be finite")
    if mu_1D <= 0.0 or mu_bulk_T <= 0.0:
        raise ValueError("mobilities must be positive")
    if mu_1D >= mu_bulk_T:
        raise ValueError("mu_1D must be smaller than mu_bulk_T")
    denom = 1.0 / mu_1D - 1.0 / mu_bulk_T
    return float(1.0 / denom)

import numpy as np
def surface_mobility_direct_check(mu_bulk_T: float, d: float, d0: float, beta: float) -> float:
    """Reference implementation."""
    for name, val in [("mu_bulk_T", mu_bulk_T), ("d", d), ("d0", d0), ("beta", beta)]:
        if not (isinstance(val, (int, float)) and np.isfinite(val)):
            raise ValueError(f"{name} must be finite")
    if d <= d0:
        raise ValueError("d must exceed d0 for this formula to apply")
    return float(mu_bulk_T * ((d / d0) ** beta - 1))

import numpy as np
def ionized_impurity_screening_parameter(C_b: float, T: float, N_I: float) -> float:
    for name, val in [("C_b", C_b), ("T", T), ("N_I", N_I)]:
        if not (isinstance(val, (int, float)) and np.isfinite(val)):
            raise ValueError(f"{name} must be finite")
    if C_b <= 0 or T <= 0 or N_I <= 0:
        raise ValueError("C_b, T, N_I must be positive")
    return float(C_b * T**2 / N_I)

import numpy as np
def ionized_impurity_mobility(C_i: float, T: float, N_I: float, b: float) -> float:
    for name, val in [("C_i", C_i), ("T", T), ("N_I", N_I), ("b", b)]:
        if not (isinstance(val, (int, float)) and np.isfinite(val)):
            raise ValueError(f"{name} must be finite")
    if C_i <= 0 or T <= 0 or N_I <= 0 or b <= 0:
        raise ValueError("C_i, T, N_I, b must be positive")
    G = np.log(1 + b) - b / (1 + b)
    if G <= 0:
        raise ValueError("G(b) must be positive")
    return float(C_i * T**1.5 / (N_I * G))

import numpy as np
def final_combined_mobility_ratio(MFP: np.ndarray, weights: np.ndarray, kappa: float, mu_bulk_ref: float, T_ref: float, alpha: float, T: float, d: float, beta: float, C_b: float, C_i: float, N_I: float) -> float:
    MFP_avg = weighted_mfp_average(MFP, weights)
    d0 = d0_from_mfp(MFP_avg, kappa)
    mu_bulk_T = bulk_mobility_temperature_scaling(mu_bulk_ref, T_ref, alpha, T)
    mu_1D = diameter_dependent_mobility(mu_bulk_T, d, d0, beta)
    mu_s = surface_limited_mobility(mu_1D, mu_bulk_T)
    mu_s_direct = surface_mobility_direct_check(mu_bulk_T, d, d0, beta)
    if abs(mu_s - mu_s_direct) > 1e-6 * max(abs(mu_s), 1.0):
        raise ValueError("consistency check failed: Matthiessen's-rule and direct Eq.6 surface mobilities disagree")
    b = ionized_impurity_screening_parameter(C_b, T, N_I)
    mu_i = ionized_impurity_mobility(C_i, T, N_I, b)
    mu_total = 1.0 / (1.0 / mu_1D + 1.0 / mu_i)
    return float(mu_total / mu_bulk_T)
SCICODE_GOLD_EOF
