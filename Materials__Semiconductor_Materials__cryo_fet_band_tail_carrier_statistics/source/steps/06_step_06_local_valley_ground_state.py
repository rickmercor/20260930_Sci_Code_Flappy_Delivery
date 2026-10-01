"""
The conventional envelope-function model of a silicon quantum well drops the valley-sector projection. The truncated delta function of the sector is replaced by an ordinary delta function, the rapidly oscillating Bloch factors are averaged to one over the intravalley matrix elements, and the potential then acts on the envelope by plain multiplication. The single-valley problem becomes the familiar one-dimensional effective-mass equation

E f(z) = -(hbar^2 / (2 m_l)) f''(z) + U(z) f(z),

solved directly in position space, again with constant energies left out. Its Hamiltonian is real, so the ground-state envelope can be chosen real, and the same envelope serves both valleys. Nothing in this equation confines the Fourier content of f to the valley sector: once shifted to the valley, F(z) = exp(i k0 z) f(z) may carry plane waves beyond the strained zone boundary G0z / 2 or below the zone centre, which belong to the other valley. For a slowly varying potential that content is negligible; a germanium profile that varies on the scale of a few monolayers makes it larger.

The equation is solved on the same periodic supercell as the band-limited problem, with the kinetic operator applied spectrally over the full grid: the discrete Fourier transform of f is multiplied by hbar^2 k^2 / (2 m_l) at every grid wave number k and transformed back. The dense real symmetric Hamiltonian that results is diagonalised and its two lowest states are taken. The same two conventions as for the band-limited solve fix the output: dz sum_j f(z_j)^2 = 1, and f positive at the grid point where |f| is largest.

The stage also measures how much of the solution lies outside its sector. With N_FBZ = L G0z / (2 pi) grid wave numbers per zone, the sector is the set of discrete Fourier indices m = 1, ..., N_FBZ / 2 - 1, and the leakage is the fraction of sum_m |F_hat(m)|^2 carried by every other index, where F_hat is the discrete Fourier transform of exp(i k0 z_j) f(z_j).

Returns
-------
dict holding the floats energy and excited_energy in eV; envelope, the real array f in reciprocal square-root nm; mean_position and spread in nm, weighted by f^2 dz; and leakage, the dimensionless fraction of the spectral weight outside the sector.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def local_valley_ground_state(
    z: np.ndarray,
    potential: np.ndarray,
    longitudinal_mass: float,
    valley_wavenumber: float,
    zone_vector: float,
) -> dict:
    """Solve the conventional local effective-mass equation on the supercell and measure its sector leakage.

    Parameters
    ----------
    z : np.ndarray
        Uniform supercell positions in nm, an even number of them.
    potential : np.ndarray
        Real confinement energy in eV on those positions.
    longitudinal_mass : float
        Longitudinal effective mass in free-electron masses, above zero.
    valley_wavenumber : float
        Valley minimum k0 in reciprocal nm, inside the open sector.
    zone_vector : float
        Strained growth-direction reciprocal-lattice vector G0z in reciprocal nm, above zero.

    Returns
    -------
    dict
        Under the keys energy, excited_energy, envelope, mean_position, spread and leakage.

    Raises
    ------
    ValueError
        When the grid fails to be uniform, finite and of even length, when the cell fails to hold an even whole number of grid wave numbers per zone, when the potential fails to be real, finite and of matching length, when the mass fails to be finite and above zero, or when k0 fails to lie strictly inside the open sector.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from scipy.linalg import eigh

HBAR2_OVER_2M0 = 0.0380998211148596  # eV nm^2, from CODATA 2018 hbar, m0 and e


def _oracle_local_valley_ground_state(
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
def digest(out, z):
    f = out["envelope"]
    pick = [int(np.argmax(np.abs(f))), len(z) // 3, len(z) // 2, 2 * len(z) // 3]
    return (round(out["energy"], 9), round(out["excited_energy"], 9), round(out["mean_position"], 7),
            round(out["spread"], 7), round(out["leakage"], 9), [round(float(f[i]), 6) for i in pick])
"""


def test_cases():
    return [
        {
            # a wiggle well in a field on a reduced cell
            "setup": GRID + """
z, u = cell(10, 128, 12.0, 40 * A / 4.0, 0.15, 3 * A, 3.0e-3)
""" + FLAT,
            "call": "flat(digest(local_valley_ground_state(z, u, 0.909, K0, G0), z))",
            "gold_call": "flat(digest(_oracle_local_valley_ground_state(z, u, 0.909, K0, G0), z))",
        },
        {
            # a harmonic confinement resolved well inside the cell: the two lowest levels must be
            # hbar omega / 2 and 3 hbar omega / 2 to spectral accuracy, and the state must be centred
            "setup": GRID + """
L = 2.0 * math.pi * 128 / G0
z = 17.0 - L + (L / 1280) * np.arange(1280)
omega_energy = 0.012
stiffness = omega_energy ** 2 / (4.0 * 0.0380998211148596 / 0.909)
u = stiffness * (z + 0.3) ** 2
def levels(fn):
    out = fn(z, u, 0.909, K0, G0)
    return (round(out["energy"] / omega_energy, 8), round(out["excited_energy"] / omega_energy, 8),
            round(out["mean_position"] + 0.3, 7), int(out["leakage"] < 1e-10))
""" + FLAT,
            "call": "flat(levels(local_valley_ground_state))",
            "gold_call": "flat(levels(_oracle_local_valley_ground_state))",
        },
        {
            # three interface widths: the leakage is set by the whole envelope shape and is not monotone in the
            # width, so it is recorded value by value
            "setup": GRID + """
def trend(fn):
    rows = []
    for sigma in (1.0, 0.3, 0.05):
        z, u = cell(10, 64, 6.0, 30 * A / 4.0, 0.0, 1.0, 0.0, sigma=sigma)
        out = fn(z, u, 0.909, K0, G0)
        rows.append((round(out["energy"], 9), round(out["leakage"], 12)))
    return rows
""" + FLAT,
            "call": "flat(trend(local_valley_ground_state))",
            "gold_call": "flat(trend(_oracle_local_valley_ground_state))",
        },
        {
            "setup": GRID + """
z, u = cell(10, 64, 6.0, 30 * A / 4.0, 0.0, 1.0, 0.0)
def verdict(fn, **kw):
    args = dict(z=z, potential=u, longitudinal_mass=0.909, valley_wavenumber=K0, zone_vector=G0)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(local_valley_ground_state, z=z[:-1], potential=u[:-1]), verdict(local_valley_ground_state, zone_vector=1.03 * G0), verdict(local_valley_ground_state, longitudinal_mass=-0.9), verdict(local_valley_ground_state, valley_wavenumber=0.51 * G0), verdict(local_valley_ground_state, potential=u + 0.0j), verdict(local_valley_ground_state, z=z[::-1]), verdict(local_valley_ground_state)))",
            "gold_call": "flat((verdict(_oracle_local_valley_ground_state, z=z[:-1], potential=u[:-1]), verdict(_oracle_local_valley_ground_state, zone_vector=1.03 * G0), verdict(_oracle_local_valley_ground_state, longitudinal_mass=-0.9), verdict(_oracle_local_valley_ground_state, valley_wavenumber=0.51 * G0), verdict(_oracle_local_valley_ground_state, potential=u + 0.0j), verdict(_oracle_local_valley_ground_state, z=z[::-1]), verdict(_oracle_local_valley_ground_state)))",
        },
    ]
