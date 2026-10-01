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


def _sine_transform_pair(f, dr, inverse):
    """Rectangle-rule radial Fourier pair along the last axis (grid r_i = i dr, q_j = j pi/(N dr))."""
    import numpy as np
    from scipy.fft import dst
    f = np.asarray(f, dtype=float)
    n = f.shape[-1] + 1
    k = np.arange(1, n)
    r = dr * k
    dq = np.pi / (n * dr)
    q = dq * k
    if not inverse:
        return (2.0 * np.pi * dr / q) * dst(r * f, type=1, axis=-1)
    return (dq / (4.0 * np.pi ** 2 * r)) * dst(q * f, type=1, axis=-1)


def radial_fourier_transform(f: np.ndarray, dr: float, inverse: bool) -> np.ndarray:
    import numpy as np
    f = np.asarray(f, dtype=float)
    if f.ndim < 1 or f.shape[-1] < 2 or not np.all(np.isfinite(f)):
        raise ValueError("f must be a finite array with at least two radial samples on its last axis")
    dr = float(dr)
    if not np.isfinite(dr) or dr <= 0.0:
        raise ValueError("dr must be a positive finite number")
    if not isinstance(inverse, (bool, np.bool_)):
        raise ValueError("inverse must be a bool")
    return _sine_transform_pair(f, dr, bool(inverse))

import numpy as np


def _check_grid(n_grid, dr):
    import numpy as np
    if isinstance(n_grid, (bool, np.bool_)) or not isinstance(n_grid, (int, np.integer)) or int(n_grid) < 16:
        raise ValueError("n_grid must be an integer of at least 16")
    dr = float(dr)
    if not np.isfinite(dr) or dr <= 0.0:
        raise ValueError("dr must be a positive finite number")
    if (int(n_grid) - 1) * dr < 4.0:
        raise ValueError("the grid must extend to at least r = 4")
    return int(n_grid), dr


def _check_solvent(rho, x, A):
    import numpy as np
    rho = float(rho)
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError("rho must be a positive finite number")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 1 or not np.all(np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("x must be a one dimensional array of non-negative mole fractions")
    if abs(float(np.sum(x)) - 1.0) > 1e-12:
        raise ValueError("the mole fractions must sum to one")
    A = np.asarray(A, dtype=float)
    if A.shape != (x.size, x.size) or not np.all(np.isfinite(A)) or np.any(A < 0.0):
        raise ValueError("A must be a finite non-negative array of shape (m, m)")
    if not np.array_equal(A, A.T):
        raise ValueError("A must be symmetric")
    return rho, x, A


def _dpd_potential(A, r):
    """beta * phi(r) = A (1 - r)^2 / 2 for r < 1 and zero beyond, for every entry of A."""
    import numpy as np
    return np.asarray(A, dtype=float)[..., None] * np.where(r < 1.0, 0.5 * (1.0 - r) ** 2, 0.0)


def _damped_iteration(step_map, gam, tol=1e-12, max_iter=20000):
    """Picard iteration gam -> step_map(gam) with a damping factor that halves whenever the error keeps rising."""
    import numpy as np
    alpha, prev, rises = 0.3, np.inf, 0
    for _ in range(max_iter):
        gnew = step_map(gam)
        err = float(np.max(np.abs(gnew - gam)))
        if not np.isfinite(err):
            raise ValueError("the HNC iteration did not converge")
        if err < tol:
            return gnew
        rises = rises + 1 if err > prev else 0
        if rises >= 3 and alpha > 0.02:
            alpha, rises = 0.5 * alpha, 0
        prev = err
        gam = alpha * gnew + (1.0 - alpha) * gam
    raise ValueError("the HNC iteration did not converge")


def solvent_structure(rho: float, x: np.ndarray, A: np.ndarray, n_grid: int, dr: float) -> np.ndarray:
    import numpy as np
    rho, x, A = _check_solvent(rho, x, A)
    n_grid, dr = _check_grid(n_grid, dr)
    m = x.size
    r = dr * np.arange(1, n_grid)
    bu = _dpd_potential(A, r)
    dens = rho * x
    eye = np.eye(m)

    def _update(gam):
        c = np.expm1(-bu + gam) - gam
        cq = np.moveaxis(radial_fourier_transform(c, dr, False), -1, 0)
        hq = np.linalg.solve(eye[None] - cq * dens[None, None, :], cq)
        g = radial_fourier_transform(np.moveaxis(hq - cq, 0, -1), dr, True)
        return 0.5 * (g + np.swapaxes(g, 0, 1))

    gam = _damped_iteration(_update, np.zeros((m, m, n_grid - 1)))
    c = np.expm1(-bu + gam) - gam
    return np.stack([gam + c, c])

import numpy as np


def _check_solute(a_solute, m):
    import numpy as np
    a = np.asarray(a_solute, dtype=float)
    if a.ndim != 2 or a.shape[0] != m or a.shape[1] < 1 or not np.all(np.isfinite(a)) or np.any(a < 0.0):
        raise ValueError("a_solute must be a finite non-negative array of shape (m, n_solute)")
    return a


def _solute_correlations(rho, x, A, a, n_grid, dr):
    """Infinite-dilution solute-solvent HNC correlations h, c of shape (m, n_s, n_grid - 1), with the solvent h, c."""
    import numpy as np
    hc = solvent_structure(rho, x, A, n_grid, dr)
    h00q = np.moveaxis(radial_fourier_transform(hc[0], dr, False), -1, 0)   # (nq, m, m)
    r = dr * np.arange(1, n_grid)
    bu = _dpd_potential(a, r)
    dens = rho * x

    def _update(gam):
        c = np.expm1(-bu + gam) - gam
        cq = np.moveaxis(radial_fourier_transform(c, dr, False), -1, 0)        # (nq, m, ns)
        g = np.einsum('qmn,n,qns->qms', h00q, dens, cq)
        return radial_fourier_transform(np.moveaxis(g, 0, -1), dr, True)

    gam = _damped_iteration(_update, np.zeros(a.shape + (n_grid - 1,)))
    c = np.expm1(-bu + gam) - gam
    return gam + c, c, hc


def solute_chemical_potential(rho: float, x: np.ndarray, A: np.ndarray, a_solute: np.ndarray,
                                      n_grid: int, dr: float) -> np.ndarray:
    import numpy as np
    rho, x, A = _check_solvent(rho, x, A)
    n_grid, dr = _check_grid(n_grid, dr)
    a = _check_solute(a_solute, x.size)
    h, c, _ = _solute_correlations(rho, x, A, a, n_grid, dr)
    r = dr * np.arange(1, n_grid)
    return rho * np.einsum('m,msr,r->s', x, 0.5 * h * (h - c) - c, 4.0 * np.pi * r * r * dr)

import numpy as np


def _virial_pressure(rho, x, A, hc, dr):
    """beta p from the virial route with the DPD force, rectangle rule on the grid."""
    import numpy as np
    r = dr * np.arange(1, hc.shape[-1] + 1)
    kern = np.where(r < 1.0, r ** 3 * (1.0 - r), 0.0) * dr
    return float(rho + (2.0 * np.pi / 3.0) * rho ** 2 * np.einsum('m,n,mn,mnr,r->', x, x, A, 1.0 + hc[0], kern))


def excluded_volume_functions(rho: float, x: np.ndarray, A: np.ndarray, a_solute: np.ndarray,
                                      n_grid: int, dr: float) -> np.ndarray:
    import numpy as np
    rho, x, A = _check_solvent(rho, x, A)
    n_grid, dr = _check_grid(n_grid, dr)
    a = _check_solute(a_solute, x.size)
    if np.any(x <= 0.0):
        raise ValueError("every solvent mole fraction must be positive")
    h, c, hc = _solute_correlations(rho, x, A, a, n_grid, dr)
    bp = _virial_pressure(rho, x, A, hc, dr)
    hsq = radial_fourier_transform(h, dr, False)                                  # (m, ns, nq)
    h00q = np.moveaxis(radial_fourier_transform(hc[0], dr, False), -1, 0)        # (nq, m, m)
    S = np.diag(x)[None] + rho * x[None, :, None] * h00q * x[None, None, :]
    ev, V = np.linalg.eigh(S)
    if np.any(ev <= 0.0):
        raise ValueError("the partial structure factor matrix is not positive definite")
    Sm12 = np.einsum('qab,qb,qcb->qac', V, ev ** -0.5, V)
    psiq = -np.sqrt(rho / bp) * np.einsum('qmn,n,nsq->msq', Sm12, x, hsq)
    return radial_fourier_transform(psiq, dr, True)

import numpy as np


def solvent_mediated_pmf(rho: float, x: np.ndarray, A: np.ndarray, a_solute: np.ndarray,
                                 n_grid: int, dr: float) -> np.ndarray:
    import numpy as np
    psi = excluded_volume_functions(rho, x, A, a_solute, n_grid, dr)
    rho, x, A = _check_solvent(rho, x, A)
    n_grid, dr = _check_grid(n_grid, dr)
    bp = _virial_pressure(rho, x, A, solvent_structure(rho, x, A, n_grid, dr), dr)
    psiq = radial_fourier_transform(psi, dr, False)
    return radial_fourier_transform(-bp * np.einsum('msq,mtq->stq', psiq, psiq), dr, True)

import numpy as np


def dimer_chemical_potential(rho: float, x: np.ndarray, A: np.ndarray, a_pair: np.ndarray,
                                     k_bond: float, l0: float, n_grid: int, dr: float) -> float:
    import numpy as np
    rho, x, A = _check_solvent(rho, x, A)
    n_grid, dr = _check_grid(n_grid, dr)
    a = _check_solute(a_pair, x.size)
    if a.shape[1] != 2:
        raise ValueError("a_pair must have exactly two columns, one per bead")
    k_bond, l0 = float(k_bond), float(l0)
    if not np.isfinite(k_bond) or k_bond <= 0.0:
        raise ValueError("k_bond must be positive and finite")
    if not np.isfinite(l0) or l0 <= 0.0 or l0 > (n_grid - 1) * dr - 2.0:
        raise ValueError("l0 must be positive and at least 2 inside the end of the grid")
    mu = solute_chemical_potential(rho, x, A, a, n_grid, dr)
    w = solvent_mediated_pmf(rho, x, A, a, n_grid, dr)[0, 1]
    r = dr * np.arange(1, n_grid)
    bond = r * r * np.exp(-k_bond * (r - l0) ** 2)
    return float(mu[0] + mu[1] - np.log(np.sum(bond * np.exp(-w)) / np.sum(bond)))

import numpy as np


def dimer_partition_coefficient(rho_water: float, x_water: np.ndarray, A_water: np.ndarray,
                                        a_water: np.ndarray, rho_oil: float, x_oil: np.ndarray,
                                        A_oil: np.ndarray, a_oil: np.ndarray, k_bond: float, l0: float,
                                        n_grid: int, dr: float) -> float:
    import numpy as np
    mu_water = dimer_chemical_potential(rho_water, x_water, A_water, a_water, k_bond, l0, n_grid, dr)
    mu_oil = dimer_chemical_potential(rho_oil, x_oil, A_oil, a_oil, k_bond, l0, n_grid, dr)
    return float((mu_water - mu_oil) / np.log(10.0))
SCICODE_GOLD_EOF
