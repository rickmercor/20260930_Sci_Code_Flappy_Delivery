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


# Oracle implementation for public function: reduced_dispersion_coefficients
def reduced_dispersion_coefficients(eps_z: float, gamma0: float) -> "np.ndarray":
    if gamma0 < 0.0:
        raise ValueError("gamma0 must be non-negative")
    a_perp = 4.0 - gamma0 ** 2
    a_par = eps_z - gamma0 ** 2
    return np.array([a_perp, a_par])

import numpy as np


# Oracle implementation for public function: equatorial_critical_gamma0_sq
def equatorial_critical_gamma0_sq(eps_z: float) -> float:
    if not (-12.0 < eps_z < -4.0):
        raise ValueError("equatorial critical ring only defined for -12 < eps_z < -4")
    return (eps_z - 4.0) ** 2 / (-8.0 * (eps_z + 4.0))

import numpy as np


# Oracle implementation for public function: classify_ep_sector
def classify_ep_sector(eps_z: float, gamma0: float) -> int:
    if gamma0 < 0.0:
        raise ValueError("gamma0 must be non-negative")
    g0sq = gamma0 ** 2
    if eps_z > 4.0:
        return 2 if (4.0 < g0sq < eps_z) else 0
    elif -12.0 < eps_z < -4.0:
        crit = equatorial_critical_gamma0_sq(eps_z)
        return 2 if g0sq >= crit else 0
    elif eps_z <= -12.0:
        return 2 if g0sq > 4.0 else 0
    else:
        return 0

import numpy as np


# Oracle implementation for public function: ep_ring_kz_squared
def ep_ring_kz_squared(eps_z: float, gamma0: float, omega: float) -> float:
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    g0sq = gamma0 ** 2
    disc = g0sq * (g0sq - 4.0)
    if disc < 0.0:
        raise ValueError("no real exceptional locus: gamma0^2 * (gamma0^2 - 4) < 0")
    s = np.sqrt(disc)
    denom = (eps_z - 4.0) * (3.0 * g0sq + 4.0)
    if denom == 0.0:
        raise ValueError("degenerate denominator for this (eps_z, gamma0) pair")
    return (
        omega ** 2
        * (g0sq - 4.0)
        / denom
        * ((g0sq - 4.0) * (eps_z - 4.0 * g0sq - 4.0) + 2.0 * (eps_z + 2.0 * g0sq + 4.0) * s)
    )

import numpy as np


# Oracle implementation for public function: ep_ring_transverse_squared
def ep_ring_transverse_squared(eps_z: float, gamma0: float, omega: float) -> float:
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    g0sq = gamma0 ** 2
    disc = g0sq * (g0sq - 4.0)
    if disc < 0.0:
        raise ValueError("no real exceptional locus: gamma0^2 * (gamma0^2 - 4) < 0")
    s = np.sqrt(disc)
    denom = (eps_z - 4.0) * (3.0 * g0sq + 4.0)
    if denom == 0.0:
        raise ValueError("degenerate denominator for this (eps_z, gamma0) pair")
    return (
        4.0
        * omega ** 2
        * (g0sq - eps_z)
        / denom
        * (g0sq * (g0sq - 4.0) - (g0sq + 4.0) * s)
    )

import numpy as np


# Oracle implementation for public function: combine_ep_wavevector
def combine_ep_wavevector(kz2: float, rho: float) -> "np.ndarray":
    if kz2 < 0.0 or rho < 0.0:
        raise ValueError("kz2 and rho must be non-negative")
    return np.array([kz2, rho, np.sqrt(kz2 + rho)])

import numpy as np


# Oracle implementation for public function: run_metamaterial_ep_pipeline
def run_metamaterial_ep_pipeline(eps_z: float, gamma0: float, omega: float) -> float:
    if gamma0 < 0.0:
        raise ValueError("gamma0 must be non-negative")
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    coeffs = reduced_dispersion_coefficients(eps_z, gamma0)
    a_perp = float(coeffs[0])
    if not (a_perp < 0.0):
        raise ValueError(
            "physical admissibility check failed: the transverse reduced "
            "coefficient A_perp must be negative (gamma0^2 > 4) for any "
            "finite-frequency EP ring to exist"
        )
    sector = classify_ep_sector(eps_z, gamma0)
    if sector != 2:
        raise ValueError("this (eps_z, gamma0) pair does not admit a finite-frequency EP ring")
    kz2 = ep_ring_kz_squared(eps_z, gamma0, omega)
    rho = ep_ring_transverse_squared(eps_z, gamma0, omega)
    parts = combine_ep_wavevector(kz2, rho)
    return float(parts[2])
SCICODE_GOLD_EOF
