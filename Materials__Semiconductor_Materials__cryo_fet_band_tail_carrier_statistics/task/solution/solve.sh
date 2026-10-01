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

import math


def _finite(value, label):
    """Return an argument as a float once it is known to be finite."""
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("%s must be finite" % label)
    return out


def strained_valley_geometry(
    lattice_si: float,
    lattice_ge: float,
    bowing: float,
    substrate_ge_fraction: float,
    c11: float,
    c12: float,
    valley_fraction: float,
) -> dict:
    """Reference implementation."""
    a_si = _finite(lattice_si, "lattice_si")
    a_ge = _finite(lattice_ge, "lattice_ge")
    b = _finite(bowing, "bowing")
    x = _finite(substrate_ge_fraction, "substrate_ge_fraction")
    stiff11 = _finite(c11, "c11")
    stiff12 = _finite(c12, "c12")
    frac = _finite(valley_fraction, "valley_fraction")
    if a_si <= 0.0 or a_ge <= 0.0:
        raise ValueError("lattice constants must be above zero")
    if stiff11 <= 0.0 or stiff12 <= 0.0:
        raise ValueError("elastic constants must be above zero")
    if stiff12 >= stiff11:
        raise ValueError("C12 must be below C11")
    if not 0.0 <= x <= 1.0:
        raise ValueError("substrate_ge_fraction must lie between zero and one")

    a_sub = a_si + b * x * (1.0 - x) + (a_ge - a_si) * x * x
    eps_par = (a_sub - a_si) / a_si
    eps_zz = -2.0 * stiff12 / stiff11 * eps_par
    zone_vector = 4.0 * math.pi / a_si * (1.0 - eps_zz)
    sector_edge = 0.5 * zone_vector
    k0 = frac * 2.0 * math.pi / a_si
    if not 0.0 < k0 < sector_edge:
        raise ValueError("the valley minimum must lie strictly inside the strained sector")
    return {
        "eps_parallel": eps_par,
        "eps_zz": eps_zz,
        "zone_vector": zone_vector,
        "sector_edge": sector_edge,
        "valley_wavenumber": k0,
        "edge_distance": sector_edge - k0,
    }

import math

import numpy as np


def _finite(value, label):
    """Return an argument as a float once it is known to be finite."""
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("%s must be finite" % label)
    return out


def _positive(value, label):
    """Return an argument as a float once it is known to be finite and above zero."""
    out = _finite(value, label)
    if out <= 0.0:
        raise ValueError("%s must be above zero" % label)
    return out


def _integer(value, label, least):
    """Return an argument as an int once it is known to be an integer of at least the stated size."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be an integer" % label)
    out = int(value)
    if out < least:
        raise ValueError("%s must be at least %d" % (label, least))
    return out


def wiggle_well_supercell(
    zone_vector: float,
    n_bz: int,
    n_fbz: int,
    spacer: float,
    well_monolayers: float,
    lattice_si: float,
    sigma_upper: float,
    sigma_lower: float,
    x_barrier: float,
    x_wiggle: float,
    wiggle_period: float,
    band_offset: float,
    field: float,
) -> dict:
    """Reference implementation."""
    g0 = _positive(zone_vector, "zone_vector")
    zones = _integer(n_bz, "n_bz", 2)
    per_zone = _integer(n_fbz, "n_fbz", 4)
    if per_zone % 2 != 0:
        raise ValueError("n_fbz must be even")
    d = _positive(spacer, "spacer")
    monolayers = _positive(well_monolayers, "well_monolayers")
    a_si = _positive(lattice_si, "lattice_si")
    s_u = _positive(sigma_upper, "sigma_upper")
    s_l = _positive(sigma_lower, "sigma_lower")
    x_b = _finite(x_barrier, "x_barrier")
    x_w = _finite(x_wiggle, "x_wiggle")
    period = _positive(wiggle_period, "wiggle_period")
    offset = _finite(band_offset, "band_offset")
    f = _finite(field, "field")
    if not 0.0 <= x_b <= 1.0:
        raise ValueError("x_barrier must lie between zero and one")
    if not 0.0 <= x_w <= 1.0:
        raise ValueError("x_wiggle must lie between zero and one")

    h = monolayers * a_si / 4.0
    n = zones * per_zone
    length = 2.0 * math.pi * per_zone / g0
    if length <= d + h:
        raise ValueError("the cell must be longer than the well and the spacer together")
    dz = length / n
    z = d - length + dz * np.arange(n)
    wavenumbers = 2.0 * math.pi / length * np.fft.fftfreq(n, d=1.0 / n)

    indicator = 0.5 * (np.tanh((z + h) / s_l) - np.tanh(z / s_u))
    modulation = 0.5 * x_w * (1.0 + np.cos(2.0 * math.pi / period * z)) * indicator
    ge = x_b * (1.0 - indicator) + modulation
    potential = offset * ge - f * z
    return {
        "z": z,
        "wavenumbers": wavenumbers,
        "ge_fraction": ge,
        "well_indicator": indicator,
        "potential": potential,
        "spacing": dz,
        "length": length,
    }

import math

import numpy as np


def _backfold_table(orders, coefficients, label):
    """Return a validated dict from order to coefficient, closed under negation and containing zero."""
    ns = [int(n) for n in orders]
    for n, raw in zip(ns, orders):
        if isinstance(raw, bool) or float(raw) != n:
            raise ValueError("orders must be integers")
    values = [complex(c) for c in coefficients]
    if len(ns) != len(values) or not ns:
        raise ValueError("orders and %s must be non-empty and of equal length" % label)
    if len(set(ns)) != len(ns):
        raise ValueError("orders must not repeat")
    for c in values:
        if not (math.isfinite(c.real) and math.isfinite(c.imag)):
            raise ValueError("%s must be finite" % label)
    table = dict(zip(ns, values))
    if 0 not in table:
        raise ValueError("orders must contain zero")
    for n in ns:
        if -n not in table:
            raise ValueError("orders must be closed under negation")
    return table


def _uniform_grid(z, n_fbz, zone_vector):
    """Check the supercell layout and return the grid as an array with its spacing, length and zone count."""
    grid = np.asarray(z, dtype=float)
    if grid.ndim != 1 or grid.size < 8 or not np.all(np.isfinite(grid)):
        raise ValueError("z must be a finite one-dimensional grid")
    if isinstance(n_fbz, bool) or not isinstance(n_fbz, (int, np.integer)) or n_fbz < 4 or n_fbz % 2:
        raise ValueError("n_fbz must be an even integer of at least four")
    g0 = float(zone_vector)
    if not math.isfinite(g0) or g0 <= 0.0:
        raise ValueError("zone_vector must be finite and above zero")
    steps = np.diff(grid)
    dz = float(steps[0])
    if dz <= 0.0 or np.max(np.abs(steps - dz)) > 1e-9 * dz:
        raise ValueError("z must be uniformly increasing")
    n = grid.size
    if n % int(n_fbz):
        raise ValueError("the number of grid points must be a whole multiple of n_fbz")
    length = n * dz
    if abs(length * g0 / (2.0 * math.pi) - int(n_fbz)) > 1e-7 * int(n_fbz):
        raise ValueError("the cell length must equal 2 pi n_fbz / zone_vector")
    return grid, dz, length, n // int(n_fbz)


def sector_projected_potential(
    z: np.ndarray,
    potential: np.ndarray,
    zone_vector: float,
    n_fbz: int,
    orders,
    backfold,
) -> dict:
    """Reference implementation."""
    grid, dz, length, zones = _uniform_grid(z, n_fbz, zone_vector)
    u = np.asarray(potential)
    if u.shape != grid.shape or np.iscomplexobj(u) or not np.all(np.isfinite(u)):
        raise ValueError("potential must be real, finite and match z")
    u = u.astype(float)
    table = _backfold_table(orders, backfold, "backfold")
    if abs(table[0] - 1.0) > 1e-12:
        raise ValueError("B_0 must equal one")
    for n, value in table.items():
        if abs(table[-n] - value.conjugate()) > 1e-12 * max(1.0, abs(value)):
            raise ValueError("B_(-n) must be the conjugate of B_n")

    n = grid.size
    per_zone = int(n_fbz)
    m = np.arange(1, per_zone // 2)
    reach = (per_zone // 2 - 2) + max(abs(k) for k in table) * per_zone
    if reach >= n // 2:
        raise ValueError("a back-folded wave number falls outside the resolved band")

    transform = np.fft.fft(u) / n
    matrix = np.zeros((m.size, m.size), dtype=complex)
    diff = m[:, None] - m[None, :]
    for order, coeff in table.items():
        if coeff == 0:
            continue
        mu = diff + order * per_zone
        kappa = 2.0 * math.pi / length * mu
        matrix += coeff * np.exp(-1j * kappa * grid[0]) * transform[np.mod(mu, n)]
    defect = float(np.max(np.abs(matrix - matrix.conj().T)))
    matrix = 0.5 * (matrix + matrix.conj().T)
    return {
        "indices": m,
        "wavenumbers": 2.0 * math.pi / length * m,
        "matrix": matrix,
        "hermitian_defect": defect,
    }

import math

import numpy as np

HBAR2_OVER_2M0 = 0.0380998211148596  # eV nm^2, from CODATA 2018 hbar, m0 and e


def nonlocal_valley_ground_state(
    z: np.ndarray,
    sector_wavenumbers: np.ndarray,
    potential_operator: np.ndarray,
    valley_wavenumber: float,
    longitudinal_mass: float,
) -> dict:
    """Reference implementation."""
    grid = np.asarray(z, dtype=float)
    if grid.ndim != 1 or grid.size < 8 or not np.all(np.isfinite(grid)):
        raise ValueError("z must be a finite one-dimensional grid")
    steps = np.diff(grid)
    dz = float(steps[0])
    if dz <= 0.0 or np.max(np.abs(steps - dz)) > 1e-9 * dz:
        raise ValueError("z must be uniformly increasing")
    length = grid.size * dz
    kk = np.asarray(sector_wavenumbers, dtype=float)
    if kk.ndim != 1 or kk.size < 2 or not np.all(np.isfinite(kk)) or kk[0] <= 0.0 or np.any(np.diff(kk) <= 0.0):
        raise ValueError("sector_wavenumbers must be finite, above zero and increasing")
    index = kk * length / (2.0 * math.pi)
    if np.max(np.abs(index - np.round(index))) > 1e-7:
        raise ValueError("sector_wavenumbers must be whole multiples of 2 pi / L")
    matrix = np.asarray(potential_operator)
    if matrix.shape != (kk.size, kk.size) or not np.all(np.isfinite(matrix)):
        raise ValueError("potential_operator must be a finite square matrix matching the sector")
    if np.max(np.abs(matrix - matrix.conj().T)) > 1e-9 * max(1.0, float(np.max(np.abs(matrix)))):
        raise ValueError("potential_operator must be Hermitian")
    k0 = float(valley_wavenumber)
    mass = float(longitudinal_mass)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("longitudinal_mass must be finite and above zero")
    if not math.isfinite(k0) or not kk[0] < k0 < kk[-1]:
        raise ValueError("valley_wavenumber must lie strictly inside the sector")

    hamiltonian = matrix.astype(complex) + np.diag(HBAR2_OVER_2M0 / mass * (kk - k0) ** 2)
    values, vectors = np.linalg.eigh(hamiltonian)
    coeff = vectors[:, 0]

    carrier = np.exp(1j * np.outer(grid, kk)) @ coeff / math.sqrt(length)
    envelope = np.exp(-1j * k0 * grid) * carrier
    envelope = envelope / math.sqrt(dz * float(np.sum(np.abs(envelope) ** 2)))
    peak = int(np.argmax(np.abs(envelope)))
    envelope = envelope * (abs(envelope[peak]) / envelope[peak])

    weight = dz * np.abs(envelope) ** 2
    mean = float(np.sum(weight * grid))
    spread = math.sqrt(max(float(np.sum(weight * grid ** 2)) - mean * mean, 0.0))
    return {
        "energy": float(values[0]),
        "excited_energy": float(values[1]),
        "envelope": envelope,
        "mean_position": mean,
        "spread": spread,
        "sector_size": int(kk.size),
    }

import math

import numpy as np


def intervalley_coupling(
    z: np.ndarray,
    envelope: np.ndarray,
    potential: np.ndarray,
    valley_wavenumber: float,
    zone_vector: float,
    orders,
    coefficients,
) -> dict:
    """Reference implementation."""
    grid = np.asarray(z, dtype=float)
    if grid.ndim != 1 or grid.size < 8 or not np.all(np.isfinite(grid)):
        raise ValueError("z must be a finite one-dimensional grid")
    steps = np.diff(grid)
    dz = float(steps[0])
    if dz <= 0.0 or np.max(np.abs(steps - dz)) > 1e-9 * dz:
        raise ValueError("z must be uniformly increasing")
    f = np.asarray(envelope)
    w = np.asarray(potential)
    if f.shape != grid.shape or not np.all(np.isfinite(f)):
        raise ValueError("envelope must be finite and match z")
    if w.shape != grid.shape or np.iscomplexobj(w) or not np.all(np.isfinite(w)):
        raise ValueError("potential must be real, finite and match z")
    if abs(dz * float(np.sum(np.abs(f) ** 2)) - 1.0) > 1e-8:
        raise ValueError("envelope must be normalised")
    k0 = float(valley_wavenumber)
    g0 = float(zone_vector)
    if not (math.isfinite(k0) and math.isfinite(g0)) or k0 <= 0.0 or g0 <= 0.0:
        raise ValueError("valley_wavenumber and zone_vector must be finite and above zero")
    ns = [int(n) for n in orders]
    for n, raw in zip(ns, orders):
        if isinstance(raw, bool) or float(raw) != n:
            raise ValueError("orders must be integers")
    cs = [complex(c) for c in coefficients]
    if not ns or len(ns) != len(cs) or len(set(ns)) != len(ns):
        raise ValueError("orders and coefficients must be non-empty, of equal length and without repetition")
    if not all(math.isfinite(c.real) and math.isfinite(c.imag) for c in cs):
        raise ValueError("coefficients must be finite")

    density = np.conj(f.astype(complex)) ** 2 * w.astype(float)
    parts = np.array([c * dz * np.sum(np.exp(-1j * (2.0 * k0 + n * g0) * grid) * density)
                      for n, c in zip(ns, cs)])
    delta = complex(np.sum(parts))
    return {
        "delta_real": delta.real,
        "delta_imag": delta.imag,
        "delta_abs": abs(delta),
        "splitting": 2.0 * abs(delta),
        "contribution_real": parts.real,
        "contribution_imag": parts.imag,
        "dominant_order": ns[int(np.argmax(np.abs(parts)))],
    }

import math

import numpy as np
from scipy.linalg import eigh

HBAR2_OVER_2M0 = 0.0380998211148596  # eV nm^2, from CODATA 2018 hbar, m0 and e


def local_valley_ground_state(
    z: np.ndarray,
    potential: np.ndarray,
    longitudinal_mass: float,
    valley_wavenumber: float,
    zone_vector: float,
) -> dict:
    """Reference implementation."""
    grid = np.asarray(z, dtype=float)
    if grid.ndim != 1 or grid.size < 8 or grid.size % 2 or not np.all(np.isfinite(grid)):
        raise ValueError("z must be a finite one-dimensional grid of even length")
    steps = np.diff(grid)
    dz = float(steps[0])
    if dz <= 0.0 or np.max(np.abs(steps - dz)) > 1e-9 * dz:
        raise ValueError("z must be uniformly increasing")
    u = np.asarray(potential)
    if u.shape != grid.shape or np.iscomplexobj(u) or not np.all(np.isfinite(u)):
        raise ValueError("potential must be real, finite and match z")
    mass = float(longitudinal_mass)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("longitudinal_mass must be finite and above zero")
    g0 = float(zone_vector)
    k0 = float(valley_wavenumber)
    if not (math.isfinite(g0) and math.isfinite(k0)) or g0 <= 0.0 or not 0.0 < k0 < 0.5 * g0:
        raise ValueError("valley_wavenumber must lie strictly inside the open sector")
    n = grid.size
    length = n * dz
    per_zone_real = length * g0 / (2.0 * math.pi)
    per_zone = int(round(per_zone_real))
    if per_zone < 4 or per_zone % 2 or abs(per_zone_real - per_zone) > 1e-7 * per_zone:
        raise ValueError("the cell must hold an even whole number of grid wave numbers per zone")

    k = 2.0 * math.pi / length * np.fft.fftfreq(n, d=1.0 / n)
    kinetic = np.real(np.fft.ifft((HBAR2_OVER_2M0 / mass * k ** 2)[:, None] * np.fft.fft(np.eye(n), axis=0), axis=0))
    hamiltonian = 0.5 * (kinetic + kinetic.T) + np.diag(u.astype(float))
    values, vectors = eigh(hamiltonian, subset_by_index=[0, 1])

    f = vectors[:, 0] / math.sqrt(dz)
    if f[int(np.argmax(np.abs(f)))] < 0.0:
        f = -f
    weight = dz * f ** 2
    mean = float(np.sum(weight * grid))
    spread = math.sqrt(max(float(np.sum(weight * grid ** 2)) - mean * mean, 0.0))

    spectrum = np.abs(np.fft.fft(np.exp(1j * k0 * grid) * f)) ** 2
    inside = np.zeros(n, dtype=bool)
    inside[1:per_zone // 2] = True
    leakage = float(np.sum(spectrum[~inside]) / np.sum(spectrum))
    return {
        "energy": float(values[0]),
        "excited_energy": float(values[1]),
        "envelope": f,
        "mean_position": mean,
        "spread": spread,
        "leakage": leakage,
    }

import math

import numpy as np


def spectral_filter_envelope(
    z: np.ndarray,
    envelope: np.ndarray,
    valley_wavenumber: float,
    zone_vector: float,
    orders,
    coefficients,
) -> dict:
    """Reference implementation."""
    grid = np.asarray(z, dtype=float)
    f = np.asarray(envelope)
    ones = np.ones(grid.shape if grid.ndim == 1 else (0,))
    before = intervalley_coupling(grid, f, ones, valley_wavenumber, zone_vector, orders, coefficients)  # noqa: F821

    g0 = float(zone_vector)
    k0 = float(valley_wavenumber)
    if not 0.0 < k0 < 0.5 * g0:
        raise ValueError("valley_wavenumber must lie strictly inside the open sector")
    n = grid.size
    dz = float(grid[1] - grid[0])
    length = n * dz
    per_zone_real = length * g0 / (2.0 * math.pi)
    per_zone = int(round(per_zone_real))
    if per_zone < 4 or per_zone % 2 or abs(per_zone_real - per_zone) > 1e-7 * per_zone:
        raise ValueError("the cell must hold an even whole number of grid wave numbers per zone")

    carrier = np.fft.fft(np.exp(1j * k0 * grid) * f)
    mask = np.zeros(n, dtype=bool)
    mask[1:per_zone // 2] = True
    kept = np.where(mask, carrier, 0.0)
    total = float(np.sum(np.abs(carrier) ** 2))
    retained = float(np.sum(np.abs(kept) ** 2)) / total
    if retained <= 0.0:
        raise ValueError("the projection retains no weight")
    filtered = np.exp(-1j * k0 * grid) * np.fft.ifft(kept)
    filtered = filtered / math.sqrt(dz * float(np.sum(np.abs(filtered) ** 2)))

    after = intervalley_coupling(grid, filtered, ones, valley_wavenumber, zone_vector, orders, coefficients)  # noqa: F821
    return {
        "envelope": filtered,
        "retained_weight": retained,
        "ambiguity_real": before["delta_real"],
        "ambiguity_imag": before["delta_imag"],
        "ambiguity_abs": before["delta_abs"],
        "filtered_ambiguity_abs": after["delta_abs"],
    }

import math

import numpy as np


def exact_valley_splitting(
    lattice_si: float,
    lattice_ge: float,
    bowing: float,
    substrate_ge_fraction: float,
    c11: float,
    c12: float,
    valley_fraction: float,
    longitudinal_mass: float,
    band_offset: float,
    x_barrier: float,
    x_wiggle: float,
    well_monolayers: float,
    sigma_upper: float,
    sigma_lower: float,
    wiggle_period: float,
    field: float,
    spacer: float,
    orders,
    coefficients,
    backfold,
    n_bz: int,
    n_fbz: int,
) -> dict:
    """Reference implementation: reruns every stage from the device description."""
    geometry = strained_valley_geometry(lattice_si, lattice_ge, bowing, substrate_ge_fraction, c11, c12, valley_fraction)  # noqa: F821
    g0 = geometry["zone_vector"]
    k0 = geometry["valley_wavenumber"]
    cell = wiggle_well_supercell(g0, n_bz, n_fbz, spacer, well_monolayers, lattice_si, sigma_upper, sigma_lower, x_barrier, x_wiggle, wiggle_period, band_offset, field)  # noqa: F821
    z = cell["z"]
    u = cell["potential"]

    projected = sector_projected_potential(z, u, g0, n_fbz, orders, backfold)  # noqa: F821
    exact = nonlocal_valley_ground_state(z, projected["wavenumbers"], projected["matrix"], k0, longitudinal_mass)  # noqa: F821
    depth = float(well_monolayers) * float(lattice_si) / 4.0
    if not -depth - 2.0 < exact["mean_position"] < 2.0:
        raise ValueError("the band-limited ground state is not a state of the well")
    coupling = intervalley_coupling(z, exact["envelope"], u, k0, g0, orders, coefficients)  # noqa: F821

    local = local_valley_ground_state(z, u, longitudinal_mass, k0, g0)  # noqa: F821
    local_coupling = intervalley_coupling(z, local["envelope"], u, k0, g0, orders, coefficients)  # noqa: F821
    filtered = spectral_filter_envelope(z, local["envelope"], k0, g0, orders, coefficients)  # noqa: F821
    filtered_coupling = intervalley_coupling(z, filtered["envelope"], u, k0, g0, orders, coefficients)  # noqa: F821
    exact_response = spectral_filter_envelope(z, exact["envelope"], k0, g0, orders, coefficients)  # noqa: F821

    parts = coupling["contribution_real"] + 1j * coupling["contribution_imag"]
    unit = 2.0 * math.pi / float(lattice_si)
    return {
        "splitting_mev": 1.0e3 * coupling["splitting"],
        "ground_energy_mev": 1.0e3 * exact["energy"],
        "excited_energy_mev": 1.0e3 * exact["excited_energy"],
        "mean_position": exact["mean_position"],
        "spread": exact["spread"],
        "eps_zz": geometry["eps_zz"],
        "sector_edge_fraction": geometry["sector_edge"] / unit,
        "long_period_fraction": 2.0 * geometry["edge_distance"] / unit,
        "dominant_order": coupling["dominant_order"],
        "dominant_share": float(np.max(np.abs(parts))) / coupling["delta_abs"],
        "local_splitting_mev": 1.0e3 * local_coupling["splitting"],
        "local_ground_energy_mev": 1.0e3 * local["energy"],
        "local_leakage": local["leakage"],
        "filtered_splitting_mev": 1.0e3 * filtered_coupling["splitting"],
        "local_ambiguity_abs": filtered["ambiguity_abs"],
        "exact_ambiguity_abs": exact_response["ambiguity_abs"],
    }
SCICODE_GOLD_EOF
