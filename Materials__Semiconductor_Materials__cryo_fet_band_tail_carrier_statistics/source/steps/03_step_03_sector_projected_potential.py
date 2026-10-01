"""
In a multi-valley envelope-function description without a slowly varying potential approximation, the microscopic wave function is expanded in the Bloch factors at the valley minima, and each valley's envelope is required to contain plane waves only from that valley's own sector of the Brillouin zone. That restriction is what makes the decomposition unique, and it changes how the confinement potential acts. The potential no longer multiplies the envelope point by point: its matrix elements are taken between band-limited functions, so it becomes a non-local integral operator whose kernel is the valley-sector projector sandwiched around U. Reduced to the growth direction for a dot that is large compared with the lattice, the single-valley envelope problem for the +k0 valley is posed on the wave numbers K of the open sector S+ = (0, G0z / 2), and the potential couples two of them through its Fourier components at K - K' + n G0z for every integer n, weighted by the Bloch-factor overlap sums B_n of the valley.

The n = 0 term is the ordinary Fourier component of U between two sector wave numbers. The terms with n not zero carry the short-wavelength content of the potential, at wave numbers a whole reciprocal-lattice vector or more away, back into the sector; this back-folding exists because the Bloch factors are lattice periodic and not constant. B_0 = 1 by the normalisation of the Bloch factors, B_(-n) is the complex conjugate of B_n, and that symmetry is what keeps the operator Hermitian for a real potential.

On the supercell of the previous stage the sector is the set of grid wave numbers k_m = 2 pi m / L with m = 1, ..., N_FBZ / 2 - 1; the zone centre m = 0 and the boundary m = N_FBZ / 2 are excluded, so that the +k0 and -k0 sectors do not share a point. The Fourier component of the potential at a grid wave number kappa is defined on the cell as U_hat(kappa) = (1 / N) sum_j U(z_j) exp(-i kappa z_j), with the actual positions z_j, so that the operator refers to plane waves exp(i K z) in the physical coordinate. The operator then has elements

M(K, K') = sum_n B_n U_hat(K - K' + n G0z),

and every wave number it needs, K - K' + n G0z for the orders supplied, must lie strictly inside the resolved band |kappa| < N_BZ G0z / 2, since a component beyond it would be aliased onto a different one.

Returns
-------
dict holding indices, the integer array of the sector grid indices m; wavenumbers, the sector wave numbers in reciprocal nm; matrix, the complex Hermitian operator in eV over the sector in that order; and the float hermitian_defect, the largest modulus of M - M^H before symmetrisation.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sector_projected_potential(
    z: np.ndarray,
    potential: np.ndarray,
    zone_vector: float,
    n_fbz: int,
    orders,
    backfold,
) -> dict:
    """Assemble the back-folded, valley-sector-projected potential operator on the +k0 sector.

    Parameters
    ----------
    z : np.ndarray
        Uniform supercell positions in nm, N of them, N a multiple of n_fbz.
    potential : np.ndarray
        Real confinement energy in eV on those positions.
    zone_vector : float
        Strained growth-direction reciprocal-lattice vector G0z in reciprocal nm, consistent with the cell.
    n_fbz : int
        Grid wave numbers per zone, even and at least four.
    orders : sequence of int
        Back-folding orders, containing zero and closed under negation.
    backfold : sequence of complex
        Coefficients B_n, with B_0 = 1 and B_(-n) the conjugate of B_n.

    Returns
    -------
    dict
        Under the keys indices, wavenumbers, matrix and hermitian_defect.

    Raises
    ------
    ValueError
        When the grid fails to be uniform or to be a whole number of zones consistent with the zone vector, when the potential fails to be real, finite and of matching length, when n_fbz fails to be an even integer of at least four, when the orders and coefficients differ in length, contain duplicates, lack zero, fail to be closed under negation, have B_0 other than one or B_(-n) other than the conjugate of B_n, or when a required wave number falls outside the resolved band.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_sector_projected_potential(
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
def cell(n_bz, n_fbz, top, h, x_w, period, field):
    L = 2.0 * math.pi * n_fbz / G0
    dz = L / (n_bz * n_fbz)
    z = top - L + dz * np.arange(n_bz * n_fbz)
    xi = 0.5 * (np.tanh((z + h) / 0.5) - np.tanh(z / 0.5))
    x = 0.3 * (1.0 - xi) + 0.5 * x_w * (1.0 + np.cos(2.0 * math.pi / period * z)) * xi
    return z, 0.5 * x - field * z
ORDERS = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
B = (-2.47e-4, -8.55e-5, -5.79e-4, 2.92e-3, 1.0, 2.92e-3, -5.79e-4, -8.55e-5, -2.47e-4)
"""


def test_cases():
    return [
        {
            # a wiggle well on a reduced cell: diagonal, first row and selected entries
            "setup": GRID + """
z, u = cell(10, 128, 12.0, 20 * A / 4.0, 0.15, 3 * A, 3.0e-3)
def digest(out):
    M = out["matrix"]
    return (out["indices"][0], out["indices"][-1], len(out["indices"]),
            round(float(out["wavenumbers"][-1]) / G0, 12),
            [round(float(v), 10) for v in np.real(np.diag(M))[::9]],
            [(round(complex(M[0, j]).real, 10), round(complex(M[0, j]).imag, 10)) for j in (1, 7, 30, 62)],
            (round(complex(M[20, 45]).real, 10), round(complex(M[20, 45]).imag, 10)),
            int(out["hermitian_defect"] < 1e-12))
""" + FLAT,
            "call": "flat(digest(sector_projected_potential(z, u, G0, 128, ORDERS, B)))",
            "gold_call": "flat(digest(_oracle_sector_projected_potential(z, u, G0, 128, ORDERS, B)))",
        },
        {
            # a constant potential c must give c B_0 times the identity: no off-diagonal element and
            # no back-folded contribution survives, whatever the orders
            "setup": GRID + """
z, u = cell(10, 64, 6.0, 12 * A / 4.0, 0.0, 1.0, 0.0)
def digest(fn):
    M = fn(z, np.full(z.shape, 0.37), G0, 64, ORDERS, B)["matrix"]
    return (round(float(np.max(np.abs(M - 0.37 * np.eye(M.shape[0])))), 13), M.shape[0])
""" + FLAT,
            "call": "flat(digest(sector_projected_potential))",
            "gold_call": "flat(digest(_oracle_sector_projected_potential))",
        },
        {
            # a single harmonic one reciprocal-lattice vector away is invisible without back-folding
            # and appears with weight B_1 once back-folding is on; the n = 0 part must be unchanged
            "setup": GRID + """
z, _ = cell(6, 32, 4.0, 1.0, 0.0, 1.0, 0.0)
L = 32 * 2.0 * math.pi / G0
kappa = G0 + 3 * 2.0 * math.pi / L
u = 0.2 * np.cos(kappa * z)
def digest(fn):
    a = fn(z, u, G0, 32, (0,), (1.0,))["matrix"]
    b = fn(z, u, G0, 32, (-1, 0, 1), (0.05, 1.0, 0.05))["matrix"]
    return (round(float(np.max(np.abs(a))), 13), round(float(np.max(np.abs(b))), 12),
            round(complex(b[5, 2]).real, 12), round(complex(b[5, 2]).imag, 12),
            round(complex(b[2, 5]).real, 12), round(complex(b[2, 5]).imag, 12))
""" + FLAT,
            "call": "flat(digest(sector_projected_potential))",
            "gold_call": "flat(digest(_oracle_sector_projected_potential))",
        },
        {
            "setup": GRID + """
z, u = cell(10, 64, 6.0, 12 * A / 4.0, 0.0, 1.0, 0.0)
def verdict(fn, **kw):
    args = dict(z=z, potential=u, zone_vector=G0, n_fbz=64, orders=ORDERS, backfold=B)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(sector_projected_potential, zone_vector=1.1 * G0), verdict(sector_projected_potential, orders=(0, 1, 2), backfold=(1.0, 0.1, 0.1)), verdict(sector_projected_potential, orders=(-1, 0, 1), backfold=(0.1, 2.0, 0.1)), verdict(sector_projected_potential, orders=(-1, 0, 1), backfold=(0.1j, 1.0, 0.1j)), verdict(sector_projected_potential, potential=u[:-1]), verdict(sector_projected_potential, orders=(-5, 0, 5), backfold=(0.1, 1.0, 0.1)), verdict(sector_projected_potential, n_fbz=63), verdict(sector_projected_potential, potential=u + 0.0j), verdict(sector_projected_potential, orders=(-1, 0, 1), backfold=(-0.1j, 1.0, 0.1j))))",
            "gold_call": "flat((verdict(_oracle_sector_projected_potential, zone_vector=1.1 * G0), verdict(_oracle_sector_projected_potential, orders=(0, 1, 2), backfold=(1.0, 0.1, 0.1)), verdict(_oracle_sector_projected_potential, orders=(-1, 0, 1), backfold=(0.1, 2.0, 0.1)), verdict(_oracle_sector_projected_potential, orders=(-1, 0, 1), backfold=(0.1j, 1.0, 0.1j)), verdict(_oracle_sector_projected_potential, potential=u[:-1]), verdict(_oracle_sector_projected_potential, orders=(-5, 0, 5), backfold=(0.1, 1.0, 0.1)), verdict(_oracle_sector_projected_potential, n_fbz=63), verdict(_oracle_sector_projected_potential, potential=u + 0.0j), verdict(_oracle_sector_projected_potential, orders=(-1, 0, 1), backfold=(-0.1j, 1.0, 0.1j))))",
        },
    ]
