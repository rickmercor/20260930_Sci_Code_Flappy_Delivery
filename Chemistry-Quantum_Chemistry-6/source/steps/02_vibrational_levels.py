"""
Solve the one-dimensional proton vibrational problem on a given diabatic potential and return the lowest vibrational energies in ascending order. These energies include the diabatic electronic offset already carried by the potential, so they are vibronic energies rather than vibrational spacings.

Solving the proton vibrational problem on a grid needs a representation of the kinetic energy operator that is accurate for a coordinate the proton tunnels along. Finite- difference stencils converge slowly for tunnelling states because they sample only nearby points. The Fourier grid representation instead maps the position grid onto a conjugate momentum grid, evaluates the kinetic energy exactly there, and transforms back, which makes the kinetic matrix dense but spectrally accurate. Use the periodic, plane-wave form on the odd-point grid: for N points of spacing dr,

T[m][n] = (2 / N) * sum over l = 1 .. (N - 1) / 2 of cos(2 * pi * l * (m - n) / N) * hbar2_over_2m * k_l ** 2, in which k_l = 2 * pi * l / (N * dr),

so the sum runs over the positive half of the momentum grid alone, the cosine having already folded the negative half onto it. Then add the potential on the diagonal and diagonalise. A sinc-function kinetic matrix is a different discretisation of the same operator and does not reproduce these levels. The representation is defined only for an odd number of grid points, which keeps the momentum grid symmetric about zero.

Because the potential passed in already carries the diabatic electronic energy as a constant offset, each eigenvalue is the energy of a complete electron-proton vibronic state and can be compared directly against eigenvalues obtained on a different electronic state. That comparison is what later fixes the crossing points of the reactant and product levels, the energy denominators of the pathways through the unpopulated states and the driving forces entering the spectral factor. The grid must be uniform for the Fourier grid representation to apply, and wide enough that the wave functions of interest have decayed at both ends.

Returns
-------
np.ndarray of shape (n_states,), the lowest vibronic energies in eV in ascending order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def vibrational_levels(r_grid: np.ndarray, potential: np.ndarray,
                       hbar2_over_2m: float, n_states: int) -> np.ndarray:
    '''Lowest proton vibrational energies on one diabatic potential.

    The Hamiltonian is the Fourier grid kinetic energy matrix for this grid plus
    the potential on the diagonal. Energies are returned in ascending order.

    Parameters
    ----------
    r_grid : np.ndarray
        (N,) uniformly spaced, strictly increasing proton positions in angstrom,
        with N odd and at least 3.
    potential : np.ndarray
        (N,) diabatic potential energy in eV at those positions.
    hbar2_over_2m : float
        Positive value of hbar squared over twice the proton mass, in
        eV angstrom squared.
    n_states : int
        Number of lowest states to return, between 1 and N.

    Returns
    -------
    levels : np.ndarray
        (n_states,) vibronic energies in eV, ascending.

    Raises
    ------
    ValueError
        If the grid and potential do not match, if the grid is not uniformly
        spaced and increasing, if N is even or smaller than 3, if n_states is
        out of range, or if hbar2_over_2m is not positive.
    '''
    return levels  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_vibrational_levels(r_grid: np.ndarray, potential: np.ndarray,
                               hbar2_over_2m: float, n_states: int) -> np.ndarray:
    """Reference implementation."""
    r, v, n, spacing = _check_grid(r_grid, potential, hbar2_over_2m, n_states)
    h = _fgh_matrix(n, spacing, hbar2_over_2m) + np.diag(v)
    return np.linalg.eigvalsh(h)[:int(n_states)]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
# normal: the reactant state of the assembly
r_grid = np.linspace(-0.90, 0.90, 201)
u = r_grid / 0.400
v = 0.595 * (u * u - 1.0) ** 2 + 0.180 * u + 3.250
""",
            "call": "vibrational_levels(r_grid, v, 2.07500e-3, 3)",
            "gold_call": "_oracle_vibrational_levels(r_grid, v, 2.07500e-3, 3)",
        },
        {
            "setup": """import numpy as np
# normal: the higher charge-transfer state, whose ladder is more widely spaced
r_grid = np.linspace(-0.90, 0.90, 201)
u = r_grid / 0.400
v = 0.300 * (u * u - 1.0) ** 2 - 0.185 * u + 4.260
""",
            "call": "vibrational_levels(r_grid, v, 2.07500e-3, 3)",
            "gold_call": "_oracle_vibrational_levels(r_grid, v, 2.07500e-3, 3)",
        },
        {
            "setup": """import numpy as np
# boundary: a harmonic well, where the spacing should be nearly uniform
r_grid = np.linspace(-0.60, 0.60, 121)
v = 0.5 * 30.0 * r_grid ** 2
""",
            "call": "vibrational_levels(r_grid, v, 2.07500e-3, 4)",
            "gold_call": "_oracle_vibrational_levels(r_grid, v, 2.07500e-3, 4)",
        },
        {
            "setup": """import numpy as np
# edge: a flat potential, so the levels are those of a free particle on a
# periodic grid; the folding of the momentum grid shows up as exact pairs
r_grid = np.linspace(-0.50, 0.50, 51)
v = np.full_like(r_grid, 1.75)
""",
            "call": "vibrational_levels(r_grid, v, 2.07500e-3, 5)",
            "gold_call": "_oracle_vibrational_levels(r_grid, v, 2.07500e-3, 5)",
        },
        {
            "setup": """import numpy as np
# edge: the smallest grid the representation admits, where the folding factor
# and the zero-point offset are both easiest to get wrong
r_grid = np.linspace(-0.45, 0.45, 3)
v = np.array([2.0, 0.0, 2.0])
""",
            "call": "vibrational_levels(r_grid, v, 2.07500e-3, 3)",
            "gold_call": "_oracle_vibrational_levels(r_grid, v, 2.07500e-3, 3)",
        },
        {
            "setup": """import numpy as np
# edge: a symmetric double well, where the lowest two levels are near degenerate
r_grid = np.linspace(-0.90, 0.90, 201)
u = r_grid / 0.400
v = 0.900 * (u * u - 1.0) ** 2
""",
            "call": "vibrational_levels(r_grid, v, 2.07500e-3, 2)",
            "gold_call": "_oracle_vibrational_levels(r_grid, v, 2.07500e-3, 2)",
        },
        {
            "setup": """import numpy as np
r_grid = np.linspace(-0.9, 0.9, 200)
v = np.zeros_like(r_grid)
def run(f):
    try:
        f(r_grid, v, 2.07500e-3, 3); return 0        # even number of grid points
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(vibrational_levels)",
            "gold_call": "run(_oracle_vibrational_levels)",
        },
        {
            "setup": """import numpy as np
r_grid = np.concatenate([np.linspace(-0.9, 0.0, 10), np.linspace(0.05, 0.9, 11)])
v = np.zeros_like(r_grid)
def run(f):
    try:
        f(r_grid, v, 2.07500e-3, 3); return 0        # non-uniform spacing
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(vibrational_levels)",
            "gold_call": "run(_oracle_vibrational_levels)",
        },
        {
            "setup": """import numpy as np
r_grid = np.linspace(-0.9, 0.9, 21)
v = np.zeros(20)
def run(f):
    try:
        f(r_grid, v, 2.07500e-3, 3); return 0        # mismatched lengths
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(vibrational_levels)",
            "gold_call": "run(_oracle_vibrational_levels)",
        },
    ]
