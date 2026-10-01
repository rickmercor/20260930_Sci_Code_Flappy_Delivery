"""
With the projected potential operator of the previous stage in hand, the single-valley envelope problem of the +k0 valley is an ordinary Hermitian eigenproblem on the sector wave numbers. Write the band-limited envelope as F(z) = sum over K in S+ of c_K exp(i K z) / sqrt(L). The kinetic energy along the growth direction is diagonal in K, but it is measured from the valley minimum and not from the zone centre: the dispersion near the minimum is hbar^2 (K - k0)^2 / (2 m_l), with m_l the longitudinal effective mass. Constant energies, the band edge E_c and the zero-point energy of the weak in-plane confinement, only shift every eigenvalue together and are left out, so the eigenvalues are growth-direction energies in the zero of U. The Hamiltonian over the sector is therefore

H(K, K') = hbar^2 (K - k0)^2 / (2 m_l) delta(K, K') + M(K, K'),

with M the back-folded projected potential of the previous stage. Its lowest eigenvector is the ground state of the valley; the -k0 valley needs no separate solve, because its envelope on the mirror sector is the complex conjugate of this one under K to -K.

The valley-scale oscillation is carried by F, whose plane waves sit near k0. The coupling between the valleys is written in terms of the slowly varying envelope f(z) = exp(-i k0 z) F(z), which is what this stage returns, sampled on the supercell. Because the sector restriction is enforced on the coefficients, f contains no plane wave that belongs to the other valley. In general f is complex: the potential of an asymmetric stack is not even, so its Fourier components are complex and so is the eigenvector.

Two conventions make the output unique. The envelope is normalised in the continuum sense, dz sum_j |f(z_j)|^2 = 1, which on the supercell is the same as unit norm of the coefficient vector. Its global phase, which no physical quantity depends on, is fixed by making f real and positive at the grid point where |f| is largest. The stage also reports the first excited eigenvalue, the mean position <z> and the spread, the square root of <z^2> - <z>^2, all taken with the weight |f|^2 dz, so that a state that has left the well can be recognised.

Returns
-------
dict holding the floats energy and excited_energy, the two lowest eigenvalues in eV; envelope, the complex slowly varying envelope f on the grid in reciprocal square-root nm; mean_position and spread in nm; and the integer sector_size, the number of sector wave numbers.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nonlocal_valley_ground_state(
    z: np.ndarray,
    sector_wavenumbers: np.ndarray,
    potential_operator: np.ndarray,
    valley_wavenumber: float,
    longitudinal_mass: float,
) -> dict:
    """Solve the sector-restricted single-valley envelope problem and return its ground state.

    Parameters
    ----------
    z : np.ndarray
        Uniform supercell positions in nm.
    sector_wavenumbers : np.ndarray
        Increasing sector wave numbers in reciprocal nm, whole multiples of 2 pi / L.
    potential_operator : np.ndarray
        Hermitian projected potential in eV over those wave numbers.
    valley_wavenumber : float
        Valley minimum k0 in reciprocal nm, inside the range of the sector wave numbers.
    longitudinal_mass : float
        Longitudinal effective mass in free-electron masses, above zero.

    Returns
    -------
    dict
        Under the keys energy, excited_energy, envelope, mean_position, spread and sector_size.

    Raises
    ------
    ValueError
        When the grid fails to be uniform and finite, when the sector wave numbers fail to be finite, increasing, above zero and on the grid of the cell, when the operator fails to be a finite Hermitian matrix of matching size, when the valley wave number fails to lie strictly inside the range of the sector, or when the mass fails to be finite and above zero.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np

HBAR2_OVER_2M0 = 0.0380998211148596  # eV nm^2, from CODATA 2018 hbar, m0 and e


def _oracle_nonlocal_valley_ground_state(
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

# =============================================================================
# TEST CASES
# =============================================================================

FLAT = """
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            out.extend(flat(v))
        return tuple(out)
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    if isinstance(x, bool):
        return (int(x),)
    if isinstance(x, complex):
        return (x.real, x.imag)
    return (x,)
"""

GRID = """
import math
import numpy as np
A = 0.543
G0 = 4.0 * math.pi / A * (1.0 + 0.00884297)
K0 = 0.8394 * 2.0 * math.pi / A
def cell(n_bz, n_fbz, top, h, x_w, period, field, sigma=0.5):
    L = 2.0 * math.pi * n_fbz / G0
    dz = L / (n_bz * n_fbz)
    z = top - L + dz * np.arange(n_bz * n_fbz)
    xi = 0.5 * (np.tanh((z + h) / sigma) - np.tanh(z / sigma))
    x = 0.3 * (1.0 - xi) + 0.5 * x_w * (1.0 + np.cos(2.0 * math.pi / period * z)) * xi
    return z, 0.5 * x - field * z
ORDERS = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
B = (-2.47e-4, -8.55e-5, -5.79e-4, 2.92e-3, 1.0, 2.92e-3, -5.79e-4, -8.55e-5, -2.47e-4)
def digest(out, z):
    f = out["envelope"]
    pick = [int(i) for i in np.argsort(-np.abs(f))[:1]] + [len(z) // 3, len(z) // 2, 2 * len(z) // 3]
    return (round(out["energy"], 9), round(out["excited_energy"], 9), round(out["mean_position"], 7),
            round(out["spread"], 7), out["sector_size"], round(float(np.sum(np.abs(f) ** 2) * (z[1] - z[0])), 10),
            [(round(complex(f[i]).real, 6), round(complex(f[i]).imag, 6)) for i in pick])


def _fx_backfold_table(orders, coefficients, label):
    # Return a validated dict from order to coefficient, closed under negation and containing zero.
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


def _fx_uniform_grid(z, n_fbz, zone_vector):
    # Check the supercell layout and return the grid as an array with its spacing, length and zone count.
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


def _fx_sector_projected_potential(z, potential, zone_vector, n_fbz, orders, backfold):
    # Setup-side reproduction of the sector-projected potential operator (public code, no oracle calls).
    grid, dz, length, zones = _fx_uniform_grid(z, n_fbz, zone_vector)
    u = np.asarray(potential)
    if u.shape != grid.shape or np.iscomplexobj(u) or not np.all(np.isfinite(u)):
        raise ValueError("potential must be real, finite and match z")
    u = u.astype(float)
    table = _fx_backfold_table(orders, backfold, "backfold")
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
"""


def test_cases():
    return [
        {
            # a wiggle well in a field on a reduced cell
            "setup": GRID + """
z, u = cell(10, 128, 12.0, 40 * A / 4.0, 0.15, 3 * A, 3.0e-3)
P = _fx_sector_projected_potential(z, u, G0, 128, ORDERS, B)
""" + FLAT,
            "call": "flat(digest(nonlocal_valley_ground_state(z, P['wavenumbers'], P['matrix'], K0, 0.909), z))",
            "gold_call": "flat(digest(_oracle_nonlocal_valley_ground_state(z, P['wavenumbers'], P['matrix'], K0, 0.909), z))",
        },
        {
            # a wide, abrupt, field-free well without back-folding: the state must sit at the well centre,
            # and its envelope is nearly real, not exactly, because the grid is not symmetric about that centre
            "setup": GRID + """
z, u = cell(10, 128, 17.0, 60 * A / 4.0, 0.0, 1.0, 0.0, sigma=0.05)
P = _fx_sector_projected_potential(z, u, G0, 128, (0,), (1.0,))
def centre(fn):
    out = fn(z, P["wavenumbers"], P["matrix"], K0, 0.909)
    return (round(out["energy"], 7), round(out["mean_position"] + 30 * A / 4.0, 6), round(out["spread"], 6),
            round(float(np.max(np.abs(np.imag(out["envelope"])))), 6))
""" + FLAT,
            "call": "flat(centre(nonlocal_valley_ground_state))",
            "gold_call": "flat(centre(_oracle_nonlocal_valley_ground_state))",
        },
        {
            # a rigid shift of the potential shifts both eigenvalues and nothing else, and a heavier mass
            # lowers the kinetic cost and so the level
            "setup": GRID + """
z, u = cell(10, 64, 6.0, 30 * A / 4.0, 0.1, 3 * A, 2.0e-3)
P = _fx_sector_projected_potential(z, u, G0, 64, ORDERS, B)
Q = _fx_sector_projected_potential(z, u + 0.25, G0, 64, ORDERS, B)
def shift(fn):
    a = fn(z, P["wavenumbers"], P["matrix"], K0, 0.909)
    b = fn(z, Q["wavenumbers"], Q["matrix"], K0, 0.909)
    c = fn(z, P["wavenumbers"], P["matrix"], K0, 1.8)
    return (round(b["energy"] - a["energy"], 11), round(b["excited_energy"] - a["excited_energy"], 11),
            round(float(np.max(np.abs(b["envelope"] - a["envelope"]))), 8), round(a["energy"], 9),
            round(c["energy"], 9), int(c["energy"] < a["energy"]))
""" + FLAT,
            "call": "flat(shift(nonlocal_valley_ground_state))",
            "gold_call": "flat(shift(_oracle_nonlocal_valley_ground_state))",
        },
        {
            "setup": GRID + """
z, u = cell(10, 64, 6.0, 30 * A / 4.0, 0.0, 1.0, 0.0)
P = _fx_sector_projected_potential(z, u, G0, 64, ORDERS, B)
K = P["wavenumbers"]
M = P["matrix"]
skew = M.copy()
skew[0, 1] += 0.01
def verdict(fn, **kw):
    args = dict(z=z, sector_wavenumbers=K, potential_operator=M, valley_wavenumber=K0, longitudinal_mass=0.909)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(nonlocal_valley_ground_state, valley_wavenumber=0.6 * G0), verdict(nonlocal_valley_ground_state, valley_wavenumber=0.0), verdict(nonlocal_valley_ground_state, longitudinal_mass=0.0), verdict(nonlocal_valley_ground_state, potential_operator=skew), verdict(nonlocal_valley_ground_state, potential_operator=M[:-1, :-1]), verdict(nonlocal_valley_ground_state, sector_wavenumbers=1.01 * K), verdict(nonlocal_valley_ground_state, potential_operator=np.full(M.shape, np.nan)), verdict(nonlocal_valley_ground_state)))",
            "gold_call": "flat((verdict(_oracle_nonlocal_valley_ground_state, valley_wavenumber=0.6 * G0), verdict(_oracle_nonlocal_valley_ground_state, valley_wavenumber=0.0), verdict(_oracle_nonlocal_valley_ground_state, longitudinal_mass=0.0), verdict(_oracle_nonlocal_valley_ground_state, potential_operator=skew), verdict(_oracle_nonlocal_valley_ground_state, potential_operator=M[:-1, :-1]), verdict(_oracle_nonlocal_valley_ground_state, sector_wavenumbers=1.01 * K), verdict(_oracle_nonlocal_valley_ground_state, potential_operator=np.full(M.shape, np.nan)), verdict(_oracle_nonlocal_valley_ground_state)))",
        },
    ]
