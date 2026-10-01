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


def proton_potential(r_grid: np.ndarray, a_barrier: float, r_ref: float,
                             b_bias: float, e_offset: float) -> np.ndarray:
    """Reference implementation."""
    r = np.asarray(r_grid, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("r_grid must be a one-dimensional array with at least two points")
    if not np.isfinite(r).all():
        raise ValueError("r_grid must be finite")
    if not (r_ref > 0.0):
        raise ValueError("r_ref must be positive")
    if not (a_barrier > 0.0):
        raise ValueError("a_barrier must be positive")
    u = r / r_ref
    return a_barrier * (u * u - 1.0) ** 2 + b_bias * u + e_offset

import numpy as np


def _fgh_matrix(n, spacing, hbar2_over_2m):
    ls = np.arange(1, (n - 1) // 2 + 1)
    t_l = hbar2_over_2m * (2.0 * np.pi * ls / (n * spacing)) ** 2
    shifts = np.arange(n)
    row = (2.0 / n) * (np.cos(2.0 * np.pi * np.outer(shifts, ls) / n) * t_l).sum(axis=1)
    idx = np.arange(n)
    return row[np.abs(idx[:, None] - idx[None, :])]


def _check_grid(r_grid, potential, hbar2_over_2m, n_states):
    r = np.asarray(r_grid, dtype=float)
    v = np.asarray(potential, dtype=float)
    if r.ndim != 1 or v.ndim != 1 or r.size != v.size:
        raise ValueError("r_grid and potential must be one-dimensional and the same length")
    if not (np.isfinite(r).all() and np.isfinite(v).all()):
        raise ValueError("r_grid and potential must be finite")
    n = r.size
    if n < 3:
        raise ValueError("the grid needs at least three points")
    if n % 2 == 0:
        raise ValueError("the grid must have an odd number of points")
    if isinstance(n_states, bool) or not isinstance(n_states, (int, np.integer)):
        raise ValueError("n_states must be an integer")
    if not 1 <= int(n_states) <= n:
        raise ValueError("n_states must be between 1 and the number of grid points")
    if not (hbar2_over_2m > 0.0):
        raise ValueError("hbar2_over_2m must be positive")
    dr = np.diff(r)
    if not np.allclose(dr, dr[0], rtol=0.0, atol=1e-12):
        raise ValueError("r_grid must be uniformly spaced")
    spacing = float(dr[0])
    if spacing <= 0.0:
        raise ValueError("r_grid must be strictly increasing")
    return r, v, n, spacing


def vibrational_levels(r_grid: np.ndarray, potential: np.ndarray,
                               hbar2_over_2m: float, n_states: int) -> np.ndarray:
    """Reference implementation."""
    r, v, n, spacing = _check_grid(r_grid, potential, hbar2_over_2m, n_states)
    h = _fgh_matrix(n, spacing, hbar2_over_2m) + np.diag(v)
    return np.linalg.eigvalsh(h)[:int(n_states)]

import numpy as np


def _fgh_matrix(n, spacing, hbar2_over_2m):
    ls = np.arange(1, (n - 1) // 2 + 1)
    t_l = hbar2_over_2m * (2.0 * np.pi * ls / (n * spacing)) ** 2
    shifts = np.arange(n)
    row = (2.0 / n) * (np.cos(2.0 * np.pi * np.outer(shifts, ls) / n) * t_l).sum(axis=1)
    idx = np.arange(n)
    return row[np.abs(idx[:, None] - idx[None, :])]


def vibrational_wavefunctions(r_grid: np.ndarray, potential: np.ndarray,
                                      hbar2_over_2m: float,
                                      n_states: int) -> np.ndarray:
    """Reference implementation."""
    r = np.asarray(r_grid, dtype=float)
    v = np.asarray(potential, dtype=float)
    if r.ndim != 1 or v.ndim != 1 or r.size != v.size:
        raise ValueError("r_grid and potential must be one-dimensional and the same length")
    if not (np.isfinite(r).all() and np.isfinite(v).all()):
        raise ValueError("r_grid and potential must be finite")
    n = r.size
    if n < 3:
        raise ValueError("the grid needs at least three points")
    if n % 2 == 0:
        raise ValueError("the grid must have an odd number of points")
    if isinstance(n_states, bool) or not isinstance(n_states, (int, np.integer)):
        raise ValueError("n_states must be an integer")
    if not 1 <= int(n_states) <= n:
        raise ValueError("n_states must be between 1 and the number of grid points")
    if not (hbar2_over_2m > 0.0):
        raise ValueError("hbar2_over_2m must be positive")
    dr = np.diff(r)
    if not np.allclose(dr, dr[0], rtol=0.0, atol=1e-12):
        raise ValueError("r_grid must be uniformly spaced")
    spacing = float(dr[0])
    if spacing <= 0.0:
        raise ValueError("r_grid must be strictly increasing")
    h = _fgh_matrix(n, spacing, hbar2_over_2m) + np.diag(v)
    _, c = np.linalg.eigh(h)
    out = np.array(c[:, :int(n_states)] / np.sqrt(spacing), dtype=float)
    for k in range(out.shape[1]):
        amplitude = np.abs(out[:, k])
        j = int(np.flatnonzero(amplitude >= amplitude.max() * (1.0 - 1e-8))[0])
        if out[j, k] < 0.0:
            out[:, k] = -out[:, k]
    return out

import numpy as np


def _unit_vector(v, name):
    arr = np.asarray(v, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite vector of length three")
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError(f"{name} must not be the zero vector")
    return arr / norm, norm


def direct_coupling(d_donor: np.ndarray, d_acceptor: np.ndarray,
                            r_vector: np.ndarray, dipole_unit_factor: float,
                            k0: float, beta: float, spin_case: str) -> float:
    """Reference implementation."""
    ud, nd = _unit_vector(d_donor, "d_donor")
    ua, na = _unit_vector(d_acceptor, "d_acceptor")
    ur, r = _unit_vector(r_vector, "r_vector")
    for value, name in ((dipole_unit_factor, "dipole_unit_factor"), (k0, "k0"),
                        (beta, "beta")):
        if not (np.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"{name} must be positive and finite")
    if not isinstance(spin_case, str) or spin_case not in ("singlet", "triplet"):
        raise ValueError('spin_case must be "singlet" or "triplet"')
    kappa = float(ud @ ua - 3.0 * (ud @ ur) * (ua @ ur))
    coulomb = kappa * nd * na / r ** 3 * float(dipole_unit_factor)
    exchange = float(k0) * np.exp(-float(beta) * r)
    if spin_case == "singlet":
        return float(coulomb - exchange)
    return float(-exchange)

import numpy as np


def _level_array(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    return arr


def _bath_pair(reorganization, force_constant):
    lam = float(reorganization)
    k = float(force_constant)
    if not (np.isfinite(lam) and lam > 0.0):
        raise ValueError("reorganization must be positive and finite")
    if not (np.isfinite(k) and k > 0.0):
        raise ValueError("force_constant must be positive and finite")
    return lam, k, np.sqrt(2.0 * lam / k)


def crossing_points(reorganization: float, force_constant: float,
                            e_reactant: np.ndarray, e_product: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    lam, k, q_product = _bath_pair(reorganization, force_constant)
    er = _level_array(e_reactant, "e_reactant")
    ep = _level_array(e_product, "e_product")
    return (lam + ep[None, :] - er[:, None]) / (k * q_product)

import numpy as np


def _virtual_labels(virtual_states):
    virtual = tuple(virtual_states)
    if len(virtual) == 0:
        raise ValueError("at least one unpopulated state is required")
    if len(set(virtual)) != len(virtual):
        raise ValueError("virtual_states must not repeat a label")
    if "I" in virtual or "II" in virtual:
        raise ValueError("the reactant and the product cannot be unpopulated states")
    return virtual


def _waves(wavefunctions, label):
    if label not in wavefunctions:
        raise ValueError(f"the proton vibrational functions of state {label} are missing")
    arr = np.asarray(wavefunctions[label], dtype=float)
    if arr.ndim != 2 or arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError(f"the proton vibrational functions of state {label} must be a "
                         "non-empty finite two-dimensional array")
    return arr


def _coupling(electronic_couplings, a, b):
    for key in ((a, b), (b, a)):
        if key in electronic_couplings:
            value = float(electronic_couplings[key])
            if not np.isfinite(value):
                raise ValueError(f"the electronic coupling between {a} and {b} must be finite")
            return value
    raise ValueError(f"the electronic coupling between {a} and {b} is missing")


def _pair_manifold(crossing, reorganization, force_constant, bath_minima, e_product_level,
                   virtual_levels, virtual_wavefunctions, spacing, virtual_states,
                   electronic_couplings):
    """Energy denominators and one-step matrix of the unpopulated manifold for one pair."""
    lam, k, q_product = _bath_pair(reorganization, force_constant)
    q = float(crossing)
    if not np.isfinite(q) or not np.isfinite(float(e_product_level)):
        raise ValueError("crossing and e_product_level must be finite")
    h = float(spacing)
    if not (np.isfinite(h) and h > 0.0):
        raise ValueError("spacing must be positive and finite")
    if not isinstance(bath_minima, dict) or not isinstance(virtual_levels, dict) \
            or not isinstance(virtual_wavefunctions, dict) or not isinstance(electronic_couplings, dict):
        raise ValueError("bath_minima, virtual_levels, virtual_wavefunctions and "
                         "electronic_couplings must be dictionaries")
    virtual = _virtual_labels(virtual_states)
    waves, energies, minima, basis = {}, [], [], []
    for label in virtual:
        if label not in bath_minima:
            raise ValueError(f"the bath minimum of state {label} is missing")
        q_j = float(bath_minima[label])
        if not np.isfinite(q_j):
            raise ValueError(f"the bath minimum of state {label} must be finite")
        if label not in virtual_levels:
            raise ValueError(f"the vibronic energies of state {label} are missing")
        levels = _level_array(virtual_levels[label], f"the vibronic energies of state {label}")
        waves[label] = _waves(virtual_wavefunctions, label)
        if waves[label].shape[1] != levels.size:
            raise ValueError(f"state {label} has a different number of functions and levels")
        energies.extend(levels.tolist())
        minima.extend([q_j] * levels.size)
        basis.extend((label, i) for i in range(levels.size))
    grid = waves[virtual[0]].shape[0]
    if any(w.shape[0] != grid for w in waves.values()):
        raise ValueError("every state's functions must live on the same grid")
    product_energy = float(e_product_level) + 0.5 * k * (q - q_product) ** 2
    denominators = product_energy - (np.array(energies) + 0.5 * k * (q - np.array(minima)) ** 2)
    if np.any(np.abs(denominators) < 1e-12):
        raise ValueError("a vanishing energy denominator was encountered")
    link = np.zeros((len(basis),) * 2)
    for x, (j, i) in enumerate(basis):
        for y, (kk, l) in enumerate(basis):
            if j != kk:
                link[x, y] = _coupling(electronic_couplings, j, kk) * h * float(
                    waves[j][:, i] @ waves[kk][:, l])
    return denominators, link, basis, waves, h


def pathway_convergence(crossing: float, reorganization: float, force_constant: float,
                                bath_minima: dict, e_product_level: float,
                                virtual_levels: dict, virtual_wavefunctions: dict,
                                spacing: float, virtual_states: tuple,
                                electronic_couplings: dict) -> float:
    """Reference implementation."""
    denominators, link, _, _, _ = _pair_manifold(
        crossing, reorganization, force_constant, bath_minima, e_product_level,
        virtual_levels, virtual_wavefunctions, spacing, virtual_states, electronic_couplings)
    one_pass = link / denominators[:, None]
    return float(np.max(np.abs(np.linalg.eigvals(one_pass))))

import numpy as np


def pair_coupling(crossing: float, reorganization: float, force_constant: float,
                          bath_minima: dict, e_product_level: float, virtual_levels: dict,
                          virtual_wavefunctions: dict, reactant_wavefunction: np.ndarray,
                          product_wavefunction: np.ndarray, spacing: float,
                          virtual_states: tuple, electronic_couplings: dict,
                          direct: float) -> float:
    """Reference implementation."""
    v_direct = float(direct)
    if not np.isfinite(v_direct):
        raise ValueError("direct must be finite")
    radius = pathway_convergence(crossing, reorganization, force_constant,
                                         bath_minima, e_product_level, virtual_levels,
                                         virtual_wavefunctions, spacing, virtual_states,
                                         electronic_couplings)
    if radius >= 1.0:
        raise ValueError("the sum over pathways does not converge for this pair")
    denominators, link, basis, waves, h = _pair_manifold(
        crossing, reorganization, force_constant, bath_minima, e_product_level,
        virtual_levels, virtual_wavefunctions, spacing, virtual_states, electronic_couplings)
    reactant = np.asarray(reactant_wavefunction, dtype=float)
    product = np.asarray(product_wavefunction, dtype=float)
    grid = waves[basis[0][0]].shape[0]
    for arr, name in ((reactant, "reactant_wavefunction"), (product, "product_wavefunction")):
        if arr.ndim != 1 or arr.size != grid or not np.isfinite(arr).all():
            raise ValueError(f"{name} must be a finite one-dimensional array on the shared grid")
    left = np.array([_coupling(electronic_couplings, "I", j) * h * float(reactant @ waves[j][:, i])
                     for j, i in basis])
    right = np.array([_coupling(electronic_couplings, j, "II") * h * float(waves[j][:, i] @ product)
                      for j, i in basis])
    indirect = float(left @ np.linalg.solve(np.diag(denominators) - link, right))
    return float(v_direct * h * float(reactant @ product) + indirect)

import numpy as np


def rate_from_couplings(v_total: np.ndarray, e_reactant: np.ndarray,
                                e_product: np.ndarray, temperature: float,
                                reorganization: float, width: float, hbar: float,
                                kb: float) -> float:
    """Reference implementation."""
    v = np.asarray(v_total, dtype=float)
    er = np.asarray(e_reactant, dtype=float)
    ep = np.asarray(e_product, dtype=float)
    for arr, name in ((er, "e_reactant"), (ep, "e_product")):
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite")
    if v.ndim != 2 or v.shape != (er.size, ep.size):
        raise ValueError("v_total must be two-dimensional and match the level arrays")
    if not np.isfinite(v).all():
        raise ValueError("v_total must be finite")
    if not (temperature > 0.0):
        raise ValueError("temperature must be positive")
    if not (width > 0.0):
        raise ValueError("width must be positive")
    if not (hbar > 0.0):
        raise ValueError("hbar must be positive")
    if not (kb > 0.0):
        raise ValueError("kb must be positive")
    if not np.isfinite(reorganization):
        raise ValueError("reorganization must be finite")

    weights = np.exp(-(er - er.min()) / (kb * float(temperature)))
    populations = weights / weights.sum()

    gap = ep[None, :] - er[:, None]
    spectral = (hbar / (2.0 * float(width) * np.sqrt(np.pi))) * np.exp(
        -(gap + float(reorganization)) ** 2 / (4.0 * float(width) ** 2))

    rate = float((populations[:, None] * v ** 2 * spectral).sum()
                 / (4.0 * np.pi ** 2 * hbar ** 2))
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("the computed rate constant is not a positive finite number")
    return float(np.log10(rate))

import numpy as np


def _state_potential(grid, potentials, label, r_ref, bias_shift=0.0):
    if label not in potentials:
        raise ValueError(f"the potential of state {label} is missing")
    params = tuple(potentials[label])
    if len(params) != 3:
        raise ValueError(f"the potential of state {label} needs (A, B, C)")
    a, b, c = (float(p) for p in params)
    return proton_potential(grid, a, r_ref, b + float(bias_shift), c)


def pcent_log_rate(reorganization: float, r_min: float, r_max: float, n_points: int,
                           r_ref: float, potentials: dict, tilts: dict, virtual_states: tuple,
                           electronic_couplings: dict, direct: float, force_constant: float,
                           bath_minima: dict, temperature: float, width: float,
                           n_states: int, hbar2_over_2m: float, hbar: float,
                           kb: float) -> float:
    """Reference implementation."""
    lam, k, _ = _bath_pair(reorganization, force_constant)
    virtual = _virtual_labels(virtual_states)
    if not isinstance(potentials, dict) or not isinstance(tilts, dict) \
            or not isinstance(bath_minima, dict):
        raise ValueError("potentials, tilts and bath_minima must be dictionaries")
    if not (np.isfinite(float(r_min)) and np.isfinite(float(r_max))
            and float(r_min) < float(r_max)):
        raise ValueError("the grid must satisfy r_min < r_max")
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)):
        raise ValueError("n_points must be an integer")
    grid = np.linspace(float(r_min), float(r_max), int(n_points))
    spacing = float(grid[1] - grid[0]) if grid.size > 1 else 0.0
    reactant = _state_potential(grid, potentials, "I", r_ref)
    product = _state_potential(grid, potentials, "II", r_ref)
    e_reactant = vibrational_levels(grid, reactant, hbar2_over_2m, n_states)
    e_product = vibrational_levels(grid, product, hbar2_over_2m, n_states)
    w_reactant = vibrational_wavefunctions(grid, reactant, hbar2_over_2m, n_states)
    w_product = vibrational_wavefunctions(grid, product, hbar2_over_2m, n_states)
    crossings = crossing_points(lam, k, e_reactant, e_product)
    couplings = np.zeros(crossings.shape)
    for mu in range(crossings.shape[0]):
        for nu in range(crossings.shape[1]):
            q = float(crossings[mu, nu])
            levels, waves = {}, {}
            for label in virtual:
                if label not in tilts:
                    raise ValueError(f"the tilt of state {label} is missing")
                if label not in bath_minima:
                    raise ValueError(f"the bath minimum of state {label} is missing")
                shift = float(tilts[label]) * (q - float(bath_minima[label]))
                if not np.isfinite(shift):
                    raise ValueError(f"the tilted potential of state {label} is not finite")
                tilted = _state_potential(grid, potentials, label, r_ref, shift)
                levels[label] = vibrational_levels(grid, tilted, hbar2_over_2m, n_states)
                waves[label] = vibrational_wavefunctions(grid, tilted, hbar2_over_2m,
                                                                n_states)
            couplings[mu, nu] = pair_coupling(
                q, lam, k, bath_minima, float(e_product[nu]), levels, waves,
                w_reactant[:, mu], w_product[:, nu], spacing, virtual, electronic_couplings,
                direct)
    return rate_from_couplings(couplings, e_reactant, e_product, temperature,
                                       lam, width, hbar, kb)

import numpy as np


def fit_reorganization_energy(k_observed: float, lam_low: float, lam_high: float,
                                      r_min: float, r_max: float, n_points: int,
                                      r_ref: float, potentials: dict, tilts: dict,
                                      virtual_states: tuple, electronic_couplings: dict,
                                      d_donor: np.ndarray, d_acceptor: np.ndarray,
                                      r_vector: np.ndarray, dipole_unit_factor: float,
                                      k0: float, beta: float, spin_case: str,
                                      force_constant: float, bath_minima: dict,
                                      temperature: float, width: float, n_states: int,
                                      hbar2_over_2m: float, hbar: float, kb: float) -> float:
    """Reference implementation."""
    k_obs = float(k_observed)
    if not (np.isfinite(k_obs) and k_obs > 0.0):
        raise ValueError("k_observed must be positive and finite")
    lo, hi = float(lam_low), float(lam_high)
    if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi):
        raise ValueError("the window must satisfy 0 < lam_low < lam_high")
    direct = direct_coupling(d_donor, d_acceptor, r_vector, dipole_unit_factor,
                                     k0, beta, spin_case)
    target = float(np.log10(k_obs))

    def _mismatch(lam):
        return pcent_log_rate(lam, r_min, r_max, n_points, r_ref, potentials, tilts,
                                      virtual_states, electronic_couplings, direct,
                                      force_constant, bath_minima, temperature, width,
                                      n_states, hbar2_over_2m, hbar, kb) - target

    f_lo, f_hi = _mismatch(lo), _mismatch(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError("the observed rate is not bracketed by the window")
    side = 0
    for _ in range(200):
        if hi - lo <= 1e-13:
            break
        if f_hi != f_lo:                        # regula falsi with the Illinois correction
            mid = (lo * f_hi - hi * f_lo) / (f_hi - f_lo)
        else:
            mid = 0.5 * (lo + hi)
        if not (lo + 1e-14 < mid < hi - 1e-14):
            mid = 0.5 * (lo + hi)
        f_mid = _mismatch(mid)
        if f_mid * f_lo > 0.0:
            lo, f_lo = mid, f_mid
            if side == 1:
                f_hi *= 0.5
            side = 1
        else:
            hi, f_hi = mid, f_mid
            if side == -1:
                f_lo *= 0.5
            side = -1
        if abs(f_mid) < 1e-14:
            break
    return float(0.5 * (lo + hi))
SCICODE_GOLD_EOF
