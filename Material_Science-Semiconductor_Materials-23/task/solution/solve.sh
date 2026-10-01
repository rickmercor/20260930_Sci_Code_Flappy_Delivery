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
def compute_scan_auxiliary_constants(
    kappa: float,
) -> "np.ndarray":
    import numpy as np

    if kappa <= 0.0:
        raise ValueError("kappa must be positive")

    mu = 10.0 / 81.0
    b2 = np.sqrt(5913.0 / 405000.0)
    b1 = (511.0 / 13500.0) / (2.0 * b2)
    b3 = 0.5
    b4 = mu**2 / kappa - 1606.0 / 18225.0 - b1**2
    return np.array([b1, b2, b3, b4], dtype=float)

import numpy as np

def compute_h1x_metallic_factor(
    s: float,
    alpha: float,
    kappa: float,
) -> float:
    import numpy as np

    if s <= 0.0:
        raise ValueError("s must be positive")
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")

    b1, b2, b3, b4 = np.asarray(
        compute_scan_auxiliary_constants(kappa), dtype=float
    )

    mu = 10.0 / 81.0
    x = mu * s**2 * (
        1.0 + (b4 * s**2 / mu) * np.exp(-abs(b4) * s**2 / mu)
    ) + (
        b1 * s**2 + b2 * (1.0 - alpha) * np.exp(-b3 * (1.0 - alpha) ** 2)
    ) ** 2
    return float(1.0 + kappa - kappa / (1.0 + x / kappa))

import numpy as np

def compute_fx_enhancement(
    s: float,
    alpha: float,
    kappa: float,
    c1x: float,
) -> float:
    import numpy as np

    if s <= 0.0:
        raise ValueError("s must be positive")
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")

    h1x = float(compute_h1x_metallic_factor(s, alpha, kappa))

    if alpha >= 1.0:
        fx = 0.0
    else:
        fx = float(np.exp(-c1x * alpha / (1.0 - alpha)))

    gx = 1.0 - np.exp(-4.9479 * s ** (-0.5))
    h0x = 1.174
    return float((h1x + fx * (h0x - h1x)) * gx)

import numpy as np

def compute_material_descriptors(
    electronegativity_a: float,
    electronegativity_b: float,
    lattice_a: float,
) -> "np.ndarray":
    import numpy as np

    if lattice_a <= 0.0:
        raise ValueError("lattice_a must be positive")
    delta_x = electronegativity_a - electronegativity_b
    covalency = float(np.exp(-0.25 * delta_x**2))
    bond_length = float(lattice_a * np.sqrt(3.0) / 4.0)
    bond_strength = covalency / bond_length
    ionic_density = 8.0 / lattice_a**3
    return np.array(
        [covalency, bond_length, bond_strength, ionic_density], dtype=float
    )

import numpy as np

def compute_weighted_fx_percent_increase(
    s_grid: "np.ndarray",
    alpha_grid: "np.ndarray",
    weight_matrix: "np.ndarray",
    descriptors: "np.ndarray",
    kappa_sd: float,
    c1x_sd: float,
    kappa_default: float,
    c1x_default: float,
) -> float:
    import numpy as np

    s_grid = np.asarray(s_grid, dtype=float)
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    weight_matrix = np.asarray(weight_matrix, dtype=float)
    descriptors = np.asarray(descriptors, dtype=float)
    if descriptors.shape != (4,):
        raise ValueError("descriptors must have shape (4,)")
    if s_grid.ndim != 1 or alpha_grid.ndim != 1:
        raise ValueError("s_grid and alpha_grid must be one-dimensional")
    if weight_matrix.shape != (s_grid.size, alpha_grid.size):
        raise ValueError("weight_matrix shape must match (len(s_grid), len(alpha_grid))")
    if np.any(s_grid <= 0.0):
        raise ValueError("all s values must be positive")
    if np.any(alpha_grid < 0.0) or np.any(alpha_grid > 1.0):
        raise ValueError("alpha values must lie in [0, 1]")
    if np.any(weight_matrix < 0.0):
        raise ValueError("weights must be non-negative")
    if weight_matrix.sum() == 0.0:
        raise ValueError("weight sum must be positive")

    covalency = float(descriptors[0])
    bond_strength = float(descriptors[2])
    ionic_density = float(descriptors[3])

    ratios = {}
    for i, s in enumerate(s_grid):
        for j, alpha in enumerate(alpha_grid):
            fx_def = compute_fx_enhancement(s, alpha, kappa_default, c1x_default)
            if fx_def == 0.0:
                raise ValueError("default Fx must be non-zero")
            fx_sd = compute_fx_enhancement(s, alpha, kappa_sd, c1x_sd)
            ratios[(i, j)] = (fx_sd - fx_def) / fx_def

    screening = 0.0
    numerator = 0.0
    denominator = 0.0
    for _ in range(3):
        numerator = 0.0
        denominator = 0.0
        for j, alpha in enumerate(alpha_grid):
            for i, s in enumerate(s_grid):
                w = weight_matrix[i, j] * np.exp(
                    -bond_strength * (1.0 - alpha) ** 2
                    - ionic_density * s**2
                    - screening * s**2 * (1.0 - alpha) ** 2
                )
                numerator += w * ratios[(i, j)]
                denominator += w
                if denominator > 0.0:
                    screening = covalency * numerator / denominator
    if denominator <= 0.0:
        raise ValueError("effective weight sum must be positive")
    return float(100.0 * numerator / denominator)

import numpy as np

def calibrate_kappa_to_target(
    target_percent_increase: float,
    s_grid: "np.ndarray",
    alpha_grid: "np.ndarray",
    weight_matrix: "np.ndarray",
    descriptors: "np.ndarray",
    c1x_sd: float,
    kappa_default: float,
    c1x_default: float,
    kappa_low: float = 0.01,
    kappa_high: float = 0.5,
) -> float:
    if kappa_low <= 0.0 or kappa_high <= kappa_low:
        raise ValueError("require 0 < kappa_low < kappa_high")

    def _residual(kappa: float) -> float:
        return (
            compute_weighted_fx_percent_increase(
                s_grid,
                alpha_grid,
                weight_matrix,
                descriptors,
                kappa,
                c1x_sd,
                kappa_default,
                c1x_default,
            )
            - target_percent_increase
        )

    low, high = float(kappa_low), float(kappa_high)
    r_low, r_high = _residual(low), _residual(high)
    if r_low == 0.0:
        return low
    if r_high == 0.0:
        return high
    if r_low * r_high > 0.0:
        raise ValueError("target is not bracketed by [kappa_low, kappa_high]")

    for _ in range(200):
        mid = 0.5 * (low + high)
        r_mid = _residual(mid)
        if r_mid == 0.0 or (high - low) < 1e-12:
            break
        if r_low * r_mid < 0.0:
            high = mid
            r_high = r_mid
        else:
            low = mid
            r_low = r_mid
    return float(0.5 * (low + high))

import numpy as np

def orchestrate_sd_scan_exchange_proxy(
    electronegativity_si: float,
    electronegativity_c: float,
    lattice_a: float,
    s_grid: "np.ndarray",
    alpha_grid: "np.ndarray",
    weight_matrix: "np.ndarray",
    kappa_sd: float,
    c1x_sd: float,
    kappa_default: float,
    c1x_default: float,
) -> float:
    import numpy as np

    descriptors = compute_material_descriptors(
        electronegativity_si, electronegativity_c, lattice_a
    )
    percent_increase = compute_weighted_fx_percent_increase(
        s_grid,
        alpha_grid,
        weight_matrix,
        descriptors,
        kappa_sd,
        c1x_sd,
        kappa_default,
        c1x_default,
    )
    calibrated_kappa = calibrate_kappa_to_target(
        percent_increase,
        s_grid,
        alpha_grid,
        weight_matrix,
        descriptors,
        c1x_sd,
        kappa_default,
        c1x_default,
    )
    return compute_weighted_fx_percent_increase(
        s_grid,
        alpha_grid,
        weight_matrix,
        descriptors,
        calibrated_kappa,
        c1x_sd,
        kappa_default,
        c1x_default,
    )
SCICODE_GOLD_EOF
