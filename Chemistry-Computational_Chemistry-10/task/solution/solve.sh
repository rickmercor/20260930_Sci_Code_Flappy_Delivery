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


def _real_scalar(name, value):
    """Reject arrays, complex values, bools and non-finite input."""
    if isinstance(value, bool) or isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(name + " must be a real scalar")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " must be a real scalar")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v


def _positive(name, value):
    v = _real_scalar(name, value)
    if v <= 0.0:
        raise ValueError(name + " must be strictly positive")
    return v


def _fraction(name, value):
    v = _real_scalar(name, value)
    if not (0.0 < v < 1.0):
        raise ValueError(name + " must lie strictly inside (0, 1)")
    return v


def _reduce(r1, r2, sigma1, sigma2, alpha):
    r1 = _positive("r1", r1)
    r2 = _positive("r2", r2)
    s1 = _positive("sigma1", sigma1)
    s2 = _positive("sigma2", sigma2)
    al = _positive("alpha", alpha)
    u = s1 / s2
    S = 2.0 * s1 * s2 / (s1 + s2)
    chi = al * S ** 1.5
    r_eff = r1 * r2 * (1.0 + u) / (r2 + u * r1)
    gamma = 1.0 - 1.0 / r_eff
    return u, S, chi, r_eff, gamma


def reduce_polyelectrolyte_pair(r1: float, r2: float, sigma1: float, sigma2: float,
                                alpha: float = 3.655) -> np.ndarray:
    """Reference implementation."""
    u, S, chi, r_eff, gamma = _reduce(r1, r2, sigma1, sigma2, alpha)
    return np.array([u, S, chi, r_eff, gamma], dtype=float)

import numpy as np


def _real_scalar(name, value):
    """Reject arrays, complex values, bools and non-finite input."""
    if isinstance(value, bool) or isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(name + " must be a real scalar")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " must be a real scalar")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v


def _positive(name, value):
    v = _real_scalar(name, value)
    if v <= 0.0:
        raise ValueError(name + " must be strictly positive")
    return v


def _fraction(name, value):
    v = _real_scalar(name, value)
    if not (0.0 < v < 1.0):
        raise ValueError(name + " must lie strictly inside (0, 1)")
    return v


def free_energy_density(phi: float, r_eff: float, chi: float) -> float:
    """Reference implementation."""
    p = _fraction("phi", phi)
    re = _positive("r_eff", r_eff)
    c = _real_scalar("chi", chi)
    if c < 0.0:
        raise ValueError("chi must be non-negative")
    return float(p / re * np.log(p) + (1.0 - p) * np.log(1.0 - p) - c * p ** 1.5)

import numpy as np


def _real_scalar(name, value):
    """Reject arrays, complex values, bools and non-finite input."""
    if isinstance(value, bool) or isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(name + " must be a real scalar")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " must be a real scalar")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v


def _positive(name, value):
    v = _real_scalar(name, value)
    if v <= 0.0:
        raise ValueError(name + " must be strictly positive")
    return v


def _fraction(name, value):
    v = _real_scalar(name, value)
    if not (0.0 < v < 1.0):
        raise ValueError(name + " must lie strictly inside (0, 1)")
    return v


def coexistence_potentials(phi: float, r_eff: float, chi: float) -> np.ndarray:
    """Reference implementation."""
    p = _fraction("phi", phi)
    re = _positive("r_eff", r_eff)
    c = _real_scalar("chi", chi)
    if c < 0.0:
        raise ValueError("chi must be non-negative")
    mu = np.log(p) / re - np.log(1.0 - p) - 1.5 * c * np.sqrt(p)
    Pi = -(1.0 - 1.0 / re) * p - np.log(1.0 - p) - 0.5 * c * p ** 1.5
    return np.array([mu, Pi], dtype=float)

import numpy as np


def _real_scalar(name, value):
    """Reject arrays, complex values, bools and non-finite input."""
    if isinstance(value, bool) or isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(name + " must be a real scalar")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " must be a real scalar")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v


def _positive(name, value):
    v = _real_scalar(name, value)
    if v <= 0.0:
        raise ValueError(name + " must be strictly positive")
    return v


def _fraction(name, value):
    v = _real_scalar(name, value)
    if not (0.0 < v < 1.0):
        raise ValueError(name + " must lie strictly inside (0, 1)")
    return v


def _critical(r_eff):
    re = _positive("r_eff", r_eff)
    phic = 2.0 / (2.0 + re + np.sqrt(re * (re + 8.0)))
    chic = (8.0 / 3.0) * np.sqrt(phic) / (1.0 - phic) ** 2
    return phic, chic


def critical_point(r_eff: float) -> np.ndarray:
    """Reference implementation."""
    phic, chic = _critical(r_eff)
    return np.array([phic, chic], dtype=float)

import numpy as np


def _real_scalar(name, value):
    """Reject arrays, complex values, bools and non-finite input."""
    if isinstance(value, bool) or isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(name + " must be a real scalar")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " must be a real scalar")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v


def _positive(name, value):
    v = _real_scalar(name, value)
    if v <= 0.0:
        raise ValueError(name + " must be strictly positive")
    return v


def _fraction(name, value):
    v = _real_scalar(name, value)
    if not (0.0 < v < 1.0):
        raise ValueError(name + " must lie strictly inside (0, 1)")
    return v


def _landau(phic, chi, r_eff):
    A = 1.0 / (1.0 - phic) + 1.0 / (r_eff * phic) - 0.75 * chi / np.sqrt(phic)
    B = (1.0 / (2.0 * (1.0 - phic) ** 2) - 1.0 / (2.0 * r_eff * phic ** 2)
         + (3.0 / 16.0) * chi / phic ** 1.5)
    C = (1.0 / (3.0 * (1.0 - phic) ** 3) + 1.0 / (3.0 * r_eff * phic ** 3)
         - (3.0 / 32.0) * chi / phic ** 2.5)
    return A, B, C


def _seed(phic, chi, r_eff):
    A, B, C = _landau(phic, chi, r_eff)
    if C == 0.0:
        raise ValueError("cubic coefficient vanishes; no near-critical estimate")
    disc = B * B - 4.0 * A * C
    if not np.isfinite(disc) or disc < 0.0:
        raise ValueError("no real coexistence gap: the near-critical expansion "
                         "is outside its range of validity at this chi")
    root = np.sqrt(disc)
    D = root / C
    phi_I = phic + (-B + root) / (2.0 * C)
    phi_II = phic + (-B - root) / (2.0 * C)
    return A, B, C, D, phi_I, phi_II


def landau_seed(phi_c: float, chi: float, r_eff: float) -> np.ndarray:
    """Reference implementation."""
    pc = _fraction("phi_c", phi_c)
    c = _real_scalar("chi", chi)
    re = _positive("r_eff", r_eff)
    if c < 0.0:
        raise ValueError("chi must be non-negative")
    A, Bc, Cc, D, phi_I, phi_II = _seed(pc, c, re)
    return np.array([A, Bc, Cc, D, phi_I, phi_II], dtype=float)

import numpy as np


def _real_scalar(name, value):
    """Reject arrays, complex values, bools and non-finite input."""
    if isinstance(value, bool) or isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(name + " must be a real scalar")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " must be a real scalar")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v


def _positive(name, value):
    v = _real_scalar(name, value)
    if v <= 0.0:
        raise ValueError(name + " must be strictly positive")
    return v


def _fraction(name, value):
    v = _real_scalar(name, value)
    if not (0.0 < v < 1.0):
        raise ValueError(name + " must lie strictly inside (0, 1)")
    return v


def _H(a, b):
    a = _real_scalar("a", a)
    b = _real_scalar("b", b)
    den = np.sinh(a + b)
    if den == 0.0:
        raise ValueError("degenerate arguments: sinh(a + b) vanishes")
    return np.sinh(a) / den


def _xy(phi_I, phi_II, chi, gamma):
    x = 1.5 * chi * (np.sqrt(phi_I) - np.sqrt(phi_II))
    y = 0.5 * chi * (phi_I ** 1.5 - phi_II ** 1.5) + gamma * (phi_I - phi_II)
    return x, y


def _ab_seed(phic, chi, gamma, r_eff, D):
    a0 = (0.75 * chi * np.sqrt(phic) + gamma) * D / 2.0
    b0 = r_eff * (0.75 * chi * (1.0 - phic) / np.sqrt(phic) - gamma) * D / 2.0
    return a0, b0


def _ab_step(a, b, chi, gamma, r_eff):
    Hp = _H(a, b)
    if not np.isfinite(Hp) or Hp <= 0.0:
        raise ValueError("iterate has left the physical branch (H <= 0)")
    a_new = 0.5 * chi * Hp ** 1.5 * np.sinh(1.5 * b) + gamma * Hp * np.sinh(b)
    b_new = r_eff * (0.5 * chi * (3.0 * np.sqrt(Hp) * np.sinh(0.5 * b)
                                  - Hp ** 1.5 * np.sinh(1.5 * b))
                     - gamma * Hp * np.sinh(b))
    if not (np.isfinite(a_new) and np.isfinite(b_new)):
        raise ValueError("iteration produced non-finite arguments")
    return a_new, b_new


def fixed_point_operators(phi_I: float, phi_II: float, chi: float, gamma: float,
                         r_eff: float) -> np.ndarray:
    """Reference implementation."""
    pI = _fraction("phi_I", phi_I)
    pII = _fraction("phi_II", phi_II)
    c = _real_scalar("chi", chi)
    g = _real_scalar("gamma", gamma)
    re = _positive("r_eff", r_eff)
    if c < 0.0:
        raise ValueError("chi must be non-negative")
    if not (pI > pII):
        raise ValueError("phi_I must be strictly greater than phi_II")
    x, y = _xy(pI, pII, c, g)
    a = y / 2.0
    b = re * (x - y) / 2.0
    return np.array([x, y, a, b, _H(a, b)], dtype=float)

import numpy as np


def _real_scalar(name, value):
    """Reject arrays, complex values, bools and non-finite input."""
    if isinstance(value, bool) or isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(name + " must be a real scalar")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " must be a real scalar")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v


def _positive(name, value):
    v = _real_scalar(name, value)
    if v <= 0.0:
        raise ValueError(name + " must be strictly positive")
    return v


def _fraction(name, value):
    v = _real_scalar(name, value)
    if not (0.0 < v < 1.0):
        raise ValueError(name + " must lie strictly inside (0, 1)")
    return v


def _H(a, b):
    a = _real_scalar("a", a)
    b = _real_scalar("b", b)
    den = np.sinh(a + b)
    if den == 0.0:
        raise ValueError("degenerate arguments: sinh(a + b) vanishes")
    return np.sinh(a) / den


def _xy(phi_I, phi_II, chi, gamma):
    x = 1.5 * chi * (np.sqrt(phi_I) - np.sqrt(phi_II))
    y = 0.5 * chi * (phi_I ** 1.5 - phi_II ** 1.5) + gamma * (phi_I - phi_II)
    return x, y


def _ab_seed(phic, chi, gamma, r_eff, D):
    a0 = (0.75 * chi * np.sqrt(phic) + gamma) * D / 2.0
    b0 = r_eff * (0.75 * chi * (1.0 - phic) / np.sqrt(phic) - gamma) * D / 2.0
    return a0, b0


def _ab_step(a, b, chi, gamma, r_eff):
    Hp = _H(a, b)
    if not np.isfinite(Hp) or Hp <= 0.0:
        raise ValueError("iterate has left the physical branch (H <= 0)")
    a_new = 0.5 * chi * Hp ** 1.5 * np.sinh(1.5 * b) + gamma * Hp * np.sinh(b)
    b_new = r_eff * (0.5 * chi * (3.0 * np.sqrt(Hp) * np.sinh(0.5 * b)
                                  - Hp ** 1.5 * np.sinh(1.5 * b))
                     - gamma * Hp * np.sinh(b))
    if not (np.isfinite(a_new) and np.isfinite(b_new)):
        raise ValueError("iteration produced non-finite arguments")
    return a_new, b_new


def seed_operators(phi_c: float, chi: float, gamma: float, r_eff: float,
                  D: float) -> np.ndarray:
    """Reference implementation."""
    pc = _fraction("phi_c", phi_c)
    c = _real_scalar("chi", chi)
    g = _real_scalar("gamma", gamma)
    re = _positive("r_eff", r_eff)
    d = _real_scalar("D", D)
    if c < 0.0:
        raise ValueError("chi must be non-negative")
    if d < 0.0:
        raise ValueError("D must be non-negative")
    a0, b0 = _ab_seed(pc, c, g, re, d)
    return np.array([a0, b0, _H(a0, b0)], dtype=float)

import numpy as np


def _real_scalar(name, value):
    """Reject arrays, complex values, bools and non-finite input."""
    if isinstance(value, bool) or isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(name + " must be a real scalar")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " must be a real scalar")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v


def _positive(name, value):
    v = _real_scalar(name, value)
    if v <= 0.0:
        raise ValueError(name + " must be strictly positive")
    return v


def _fraction(name, value):
    v = _real_scalar(name, value)
    if not (0.0 < v < 1.0):
        raise ValueError(name + " must lie strictly inside (0, 1)")
    return v


def _H(a, b):
    a = _real_scalar("a", a)
    b = _real_scalar("b", b)
    den = np.sinh(a + b)
    if den == 0.0:
        raise ValueError("degenerate arguments: sinh(a + b) vanishes")
    return np.sinh(a) / den


def _xy(phi_I, phi_II, chi, gamma):
    x = 1.5 * chi * (np.sqrt(phi_I) - np.sqrt(phi_II))
    y = 0.5 * chi * (phi_I ** 1.5 - phi_II ** 1.5) + gamma * (phi_I - phi_II)
    return x, y


def _ab_seed(phic, chi, gamma, r_eff, D):
    a0 = (0.75 * chi * np.sqrt(phic) + gamma) * D / 2.0
    b0 = r_eff * (0.75 * chi * (1.0 - phic) / np.sqrt(phic) - gamma) * D / 2.0
    return a0, b0


def _ab_step(a, b, chi, gamma, r_eff):
    Hp = _H(a, b)
    if not np.isfinite(Hp) or Hp <= 0.0:
        raise ValueError("iterate has left the physical branch (H <= 0)")
    a_new = 0.5 * chi * Hp ** 1.5 * np.sinh(1.5 * b) + gamma * Hp * np.sinh(b)
    b_new = r_eff * (0.5 * chi * (3.0 * np.sqrt(Hp) * np.sinh(0.5 * b)
                                  - Hp ** 1.5 * np.sinh(1.5 * b))
                     - gamma * Hp * np.sinh(b))
    if not (np.isfinite(a_new) and np.isfinite(b_new)):
        raise ValueError("iteration produced non-finite arguments")
    return a_new, b_new


def advance_fixed_point(a: float, b: float, chi: float, gamma: float,
                       r_eff: float) -> np.ndarray:
    """Reference implementation."""
    aa = _real_scalar("a", a)
    bb = _real_scalar("b", b)
    c = _real_scalar("chi", chi)
    g = _real_scalar("gamma", gamma)
    re = _positive("r_eff", r_eff)
    if c < 0.0:
        raise ValueError("chi must be non-negative")
    a_new, b_new = _ab_step(aa, bb, c, g, re)
    return np.array([a_new, b_new, _H(a_new, b_new)], dtype=float)

import numpy as np


def _real_scalar(name, value):
    """Reject arrays, complex values, bools and non-finite input."""
    if isinstance(value, bool) or isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(name + " must be a real scalar")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " must be a real scalar")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v


def _positive(name, value):
    v = _real_scalar(name, value)
    if v <= 0.0:
        raise ValueError(name + " must be strictly positive")
    return v


def _fraction(name, value):
    v = _real_scalar(name, value)
    if not (0.0 < v < 1.0):
        raise ValueError(name + " must lie strictly inside (0, 1)")
    return v


def _reduce(r1, r2, sigma1, sigma2, alpha):
    r1 = _positive("r1", r1)
    r2 = _positive("r2", r2)
    s1 = _positive("sigma1", sigma1)
    s2 = _positive("sigma2", sigma2)
    al = _positive("alpha", alpha)
    u = s1 / s2
    S = 2.0 * s1 * s2 / (s1 + s2)
    chi = al * S ** 1.5
    r_eff = r1 * r2 * (1.0 + u) / (r2 + u * r1)
    gamma = 1.0 - 1.0 / r_eff
    return u, S, chi, r_eff, gamma


def _xi_zeta(r1, r2, u, alpha):
    a23 = alpha ** (2.0 / 3.0)
    xi = (u * r1 - r2) / (u * r1 + r2) * a23
    zeta = u * (r1 - r2) / ((1.0 + u) * (u * r1 + r2)) * a23
    return xi, zeta


def interphase_potential(phi_I: float, phi_II: float, r1: float, r2: float,
                        sigma1: float, sigma2: float, alpha: float = 3.655) -> float:
    """Reference implementation."""
    pI = _fraction("phi_I", phi_I)
    pII = _fraction("phi_II", phi_II)
    if not (pI > pII):
        raise ValueError("phi_I must be strictly greater than phi_II")
    u, S, chi, r_eff, gamma = _reduce(r1, r2, sigma1, sigma2, alpha)
    xi, zeta = _xi_zeta(float(r1), float(r2), u, float(alpha))
    return float(1.5 * xi * chi ** (1.0 / 3.0) * (np.sqrt(pI) - np.sqrt(pII))
                 + 2.0 * zeta * chi ** (-2.0 / 3.0) * np.log((1.0 - pI) / (1.0 - pII)))

import numpy as np


def coacervate_potential_drop(r1: float, r2: float, sigma1: float, sigma2: float,
                             alpha: float = 3.655, order: int = 2) -> float:
    """Reference implementation."""
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
        raise ValueError("order must be an integer")
    n = int(order)
    if n < 1:
        raise ValueError("order must be at least 1")
    # chain the earlier sub-problems rather than reimplementing them
    u, S, chi, r_eff, gamma = reduce_polyelectrolyte_pair(
        r1, r2, sigma1, sigma2, alpha)
    phi_c, chi_c = critical_point(r_eff)
    if chi <= chi_c:
        raise ValueError("no phase separation: chi does not exceed its critical value")
    D = landau_seed(phi_c, chi, r_eff)[3]
    state = seed_operators(phi_c, chi, gamma, r_eff, D)
    for _ in range(n - 1):
        state = advance_fixed_point(state[0], state[1], chi, gamma, r_eff)
    b, Hv = float(state[1]), float(state[2])
    if not np.isfinite(Hv) or Hv <= 0.0:
        raise ValueError("iterate has left the physical branch")
    phi_I = float(np.exp(b) * Hv)
    phi_II = float(np.exp(-b) * Hv)
    if not (0.0 < phi_II < phi_I < 1.0):
        raise ValueError("iterate is not a physical pair of volume fractions")
    # the exact scalar combinations evaluated on the pair the iteration has reached;
    # one further application of the closed-form recursion must reproduce them, which
    # is what ties the two ways of writing the same map together
    ops = fixed_point_operators(phi_I, phi_II, chi, gamma, r_eff)
    nxt = advance_fixed_point(state[0], state[1], chi, gamma, r_eff)
    if not np.allclose(np.asarray(ops)[2:], np.asarray(nxt), rtol=1e-9, atol=1e-12):
        raise ValueError("the two forms of the fixed-point map disagree")
    # the free-energy density and the two coexistence potentials are a Legendre pair,
    # so on each coexisting fraction the pressure must equal phi*(mu - gamma) - f
    for phi in (phi_I, phi_II):
        f_phi = free_energy_density(phi, r_eff, chi)
        mu_phi, pi_phi = coexistence_potentials(phi, r_eff, chi)
        if not np.isclose(pi_phi, phi * (mu_phi - gamma) - f_phi,
                          rtol=1e-9, atol=1e-12):
            raise ValueError("free energy and coexistence potentials are inconsistent")
    return interphase_potential(phi_I, phi_II, r1, r2, sigma1, sigma2, alpha)
SCICODE_GOLD_EOF
