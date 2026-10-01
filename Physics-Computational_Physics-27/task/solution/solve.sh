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


def build_hatano_nelson(n_sites: int, gamma: float, p: float, alpha_bc: float) -> np.ndarray:
    if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 2:
        raise ValueError("n_sites must be an integer >= 2")
    if not np.isfinite(float(gamma)):
        raise ValueError("gamma must be finite")
    if not np.isfinite(float(p)) or abs(float(p)) >= 1.0:
        raise ValueError("p must satisfy |p| < 1")
    if float(alpha_bc) not in (0.0, 1.0):
        raise ValueError("alpha_bc must be 0.0 (OBC) or 1.0 (PBC)")

    n = int(n_sites)
    g = float(gamma)
    pp = float(p)
    bc = float(alpha_bc)

    H = np.zeros((n, n), dtype=complex)
    idx = np.arange(n - 1)
    H[idx, idx + 1] = g * (1.0 + pp)
    H[idx + 1, idx] = g * (1.0 - pp)

    if bc != 0.0:
        # accumulate: on a two-site ring both wrap entries land on the chain bond
        H[n - 1, 0] += bc * g * (1.0 + pp)
        H[0, n - 1] += bc * g * (1.0 - pp)

    return H

import numpy as np


def analytic_spectrum(n_sites: int, gamma: float, p: float, alpha_bc: float) -> np.ndarray:
    if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 2:
        raise ValueError("n_sites must be an integer >= 2")
    if not np.isfinite(float(gamma)):
        raise ValueError("gamma must be finite")
    if not np.isfinite(float(p)) or abs(float(p)) >= 1.0:
        raise ValueError("p must satisfy |p| < 1")
    if float(alpha_bc) not in (0.0, 1.0):
        raise ValueError("alpha_bc must be 0.0 (OBC) or 1.0 (PBC)")

    n = int(n_sites)
    g = float(gamma)
    pp = float(p)
    a = np.arange(1, n + 1)

    if float(alpha_bc) == 1.0:
        theta = 2.0 * np.pi * a / n
        spectrum = g * (1.0 + pp) * np.exp(-1j * theta) + g * (1.0 - pp) * np.exp(1j * theta)
    else:
        amp = 2.0 * g * np.sqrt((1.0 - pp) * (1.0 + pp))
        spectrum = (amp * np.cos(a * np.pi / (n + 1.0))).astype(complex)

    return spectrum

import numpy as np


def bernstein_radius(points: np.ndarray) -> float:
    z = np.asarray(points, dtype=complex).ravel()
    if z.size == 0:
        raise ValueError("points must contain at least one value")
    if not np.all(np.isfinite(z)):
        raise ValueError("points must contain only finite values")

    root = np.sqrt(z * z - 1.0 + 0.0j)
    rho = np.maximum(np.abs(z + root), np.abs(z - root))

    return float(np.max(rho))

import numpy as np


from scipy.special import lambertw


def max_time_step(rho: float, delta_max: float, eps: float = 1.11e-16) -> float:
    r = float(rho)
    dm = float(delta_max)
    ep = float(eps)
    if not np.isfinite(r) or r < 1.0:
        raise ValueError("rho must be finite and >= 1")
    if not np.isfinite(dm) or dm <= 0.0:
        raise ValueError("delta_max must be finite and > 0")
    if not np.isfinite(ep) or ep <= 0.0:
        raise ValueError("eps must be finite and > 0")

    w = float(np.real(lambertw(dm / (4.0 * ep), 0)))

    return float(2.0 * w / r)

import numpy as np


def rounding_error_bound(n_terms: int, dt: float, rho: float, eps: float = 1.11e-16) -> float:
    if not isinstance(n_terms, (int, np.integer)) or int(n_terms) < 1:
        raise ValueError("n_terms must be an integer >= 1")
    t = float(dt)
    r = float(rho)
    ep = float(eps)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("dt must be finite and > 0")
    if not np.isfinite(r) or r < 1.0:
        raise ValueError("rho must be finite and >= 1")
    if not np.isfinite(ep) or ep <= 0.0:
        raise ValueError("eps must be finite and > 0")

    return float(4.0 * ep * int(n_terms) * np.exp(t * r / 2.0))

import numpy as np
from scipy.special import jv


def chebyshev_coefficients(dt: float, max_order: int) -> np.ndarray:
    t = float(dt)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("dt must be finite and > 0")
    if not isinstance(max_order, (int, np.integer)) or int(max_order) < 1:
        raise ValueError("max_order must be an integer >= 1")

    m = np.arange(int(max_order) + 1)
    coefficients = 2.0 * ((-1j) ** m) * jv(m, t)
    coefficients[0] = jv(0, t)

    return coefficients

import numpy as np


def chebyshev_step(
    hamiltonian: np.ndarray,
    psi: np.ndarray,
    dt: float,
    tol: float = 1e-14,
    patience: int = 5,
    max_order: int = 4000,
) -> tuple:
    H = np.asarray(hamiltonian, dtype=complex)
    v = np.asarray(psi, dtype=complex).ravel()
    t = float(dt)

    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] < 1:
        raise ValueError("hamiltonian must be a square 2D array")
    if v.shape[0] != H.shape[0]:
        raise ValueError("psi length must match the hamiltonian dimension")
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("dt must be finite and > 0")
    if not np.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError("tol must be finite and > 0")
    if not isinstance(patience, (int, np.integer)) or int(patience) < 1:
        raise ValueError("patience must be an integer >= 1")
    if not isinstance(max_order, (int, np.integer)) or int(max_order) < 1:
        raise ValueError("max_order must be an integer >= 1")

    cap = int(max_order)

    # --- step 06: the expansion coefficients ---
    try:
        c = chebyshev_coefficients(t, cap)
    except NameError:
        c = chebyshev_coefficients(t, cap)

    t_prev = v.copy()                      # T_0(H)|psi>
    psi_next = c[0] * t_prev
    if cap < 1:
        return psi_next, 1

    t_curr = H @ v                         # T_1(H)|psi>
    psi_next = psi_next + c[1] * t_curr

    n_terms = 2
    run = 0
    for m in range(2, cap + 1):
        t_next = 2.0 * (H @ t_curr) - t_prev
        term = c[m] * t_next
        psi_next = psi_next + term
        n_terms += 1
        if np.linalg.norm(term) < float(tol):
            run += 1
            if run >= int(patience):
                break
        else:
            run = 0
        t_prev, t_curr = t_curr, t_next

    return psi_next, int(n_terms)

import numpy as np


def _gaussian_packet(n_sites: int, momentum: float, sigma: float) -> np.ndarray:
    """Unit-norm Gaussian packet of width sigma centred at (N + 1) / 2."""
    site = np.arange(1, int(n_sites) + 1)
    centre = (int(n_sites) + 1) / 2.0
    psi = np.exp(-((site - centre) ** 2) / (2.0 * float(sigma) ** 2)) * np.exp(
        1j * float(momentum) * site
    )
    return psi / np.linalg.norm(psi)


def evolve_wavepacket(
    n_sites: int,
    gamma: float,
    p: float,
    alpha_bc: float,
    momentum: float,
    sigma: float,
    t_max: float,
    n_steps: int,
    tol: float = 1e-14,
    patience: int = 5,
) -> tuple:
    if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 2:
        raise ValueError("n_sites must be an integer >= 2")
    if not np.isfinite(float(gamma)):
        raise ValueError("gamma must be finite")
    if not np.isfinite(float(p)) or abs(float(p)) >= 1.0:
        raise ValueError("p must satisfy |p| < 1")
    if float(alpha_bc) not in (0.0, 1.0):
        raise ValueError("alpha_bc must be 0.0 (OBC) or 1.0 (PBC)")
    if not np.isfinite(float(sigma)) or float(sigma) <= 0.0:
        raise ValueError("sigma must be finite and > 0")
    if not np.isfinite(float(t_max)) or float(t_max) <= 0.0:
        raise ValueError("t_max must be finite and > 0")
    if not isinstance(n_steps, (int, np.integer)) or int(n_steps) < 1:
        raise ValueError("n_steps must be an integer >= 1")
    if not np.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError("tol must be finite and > 0")
    if not isinstance(patience, (int, np.integer)) or int(patience) < 1:
        raise ValueError("patience must be an integer >= 1")

    # --- step 01: the Hamiltonian ---
    try:
        H = build_hatano_nelson(n_sites, gamma, p, alpha_bc)
    except NameError:
        H = build_hatano_nelson(n_sites, gamma, p, alpha_bc)

    psi = _gaussian_packet(n_sites, momentum, sigma)
    dt = float(t_max) / int(n_steps)

    # --- step 07: one adaptively truncated Chebyshev step, repeated ---
    total_terms = 0
    for _ in range(int(n_steps)):
        try:
            psi, used = chebyshev_step(H, psi, dt, tol, patience, 4000)
        except NameError:
            psi, used = chebyshev_step(H, psi, dt, tol, patience, 4000)
        total_terms += used
        psi = psi / np.linalg.norm(psi)

    return psi, int(total_terms)

import numpy as np


def total_error_budget(
    n_sites: int,
    gamma: float,
    p: float,
    alpha_bc: float,
    momentum: float,
    sigma: float,
    t_max: float,
    delta_max: float,
    eps: float = 1.11e-16,
    tol: float = 1e-14,
    patience: int = 5,
) -> float:
    if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 2:
        raise ValueError("n_sites must be an integer >= 2")
    if not np.isfinite(float(p)) or abs(float(p)) >= 1.0:
        raise ValueError("p must satisfy |p| < 1")
    if float(alpha_bc) not in (0.0, 1.0):
        raise ValueError("alpha_bc must be 0.0 (OBC) or 1.0 (PBC)")
    if not np.isfinite(float(t_max)) or float(t_max) <= 0.0:
        raise ValueError("t_max must be finite and > 0")
    if not np.isfinite(float(delta_max)) or float(delta_max) <= 0.0:
        raise ValueError("delta_max must be finite and > 0")
    if not np.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and > 0")

    # Every stage below is delegated to the earlier sub-problem, preferring the gold
    # binding when the harness provides one and falling back to the public name.

    # Step 01 (the Hamiltonian) and steps 06 and 07 (the expansion coefficients and the
    # single-step propagation) are reached through step 08, which consumes all three and
    # returns the realised term count. They are deliberately not called again here: the
    # orchestrator has no use for their return values of its own, and calling one only to
    # discard it would be a token call rather than a composition.

    # --- step 02: closed-form spectrum ---
    try:
        spectrum = analytic_spectrum(n_sites, gamma, p, alpha_bc)
    except NameError:
        spectrum = analytic_spectrum(n_sites, gamma, p, alpha_bc)

    # --- step 03: smallest Bernstein ellipse containing the spectrum ---
    try:
        rho = bernstein_radius(spectrum)
    except NameError:
        rho = bernstein_radius(spectrum)

    # --- step 04: tolerance -> ceiling on a single step ---
    try:
        dt_ceiling = max_time_step(rho, delta_max, eps)
    except NameError:
        dt_ceiling = max_time_step(rho, delta_max, eps)

    # rounding the count up is what keeps the answer off the input tolerance
    n_steps = int(np.ceil(float(t_max) / dt_ceiling))
    dt = float(t_max) / n_steps

    # --- step 08: propagate, accumulating the realised Chebyshev term count ---
    try:
        _, total_terms = evolve_wavepacket(
            n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, tol, patience
        )
    except NameError:
        _, total_terms = evolve_wavepacket(
            n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, tol, patience
        )

    # --- step 05: accumulated rounding-error budget of the whole run ---
    try:
        budget = rounding_error_bound(total_terms, dt, rho, eps)
    except NameError:
        budget = rounding_error_bound(total_terms, dt, rho, eps)

    return float(np.log10(budget))
SCICODE_GOLD_EOF
