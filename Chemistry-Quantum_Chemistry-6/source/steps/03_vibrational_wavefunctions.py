"""
Return the lowest proton vibrational wave functions on one diabatic potential, normalised so that the squared amplitude integrates to one over the grid, and with a fixed sign convention. The columns are ordered by increasing energy.

The Hamiltonian is the same one the previous step diagonalised, the Fourier grid kinetic energy matrix plus the potential on the diagonal, so build it the same way.

Grid eigenvectors come out of a diagonalisation normalised as vectors, meaning the sum of their squared entries is one. Turning them into wave functions normalised over the coordinate requires dividing by the square root of the grid spacing, after which an integral becomes a sum times the spacing. Getting this wrong rescales every overlap by a constant and therefore rescales the couplings.

The overall sign of an eigenvector is arbitrary, and the sign a diagonalisation routine returns depends on the library, so the convention has to be imposed by hand. Fix the sign of each column so that the column is positive at the grid point where its amplitude is largest in absolute value.

That point is not always unique. On a potential symmetric about the middle of the grid every state has two extrema of equal magnitude, and which of the two a search returns is then settled by rounding rather than by the physics, so the tie has to be broken by a rule instead. Among the grid points whose amplitude reaches the largest absolute value to within a relative tolerance of 1e-8, take the one with the smallest index.

The convention fixes only the phase of each function. Overlaps and couplings change sign with it while the rate constant does not, but without it the intermediate matrices are not reproducible from one linear algebra library to the next.

Returns
-------
np.ndarray of shape (N, n_states), grid-normalised wave functions in inverse square root angstrom
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def vibrational_wavefunctions(r_grid: np.ndarray, potential: np.ndarray,
                              hbar2_over_2m: float, n_states: int) -> np.ndarray:
    '''Lowest proton vibrational wave functions on one diabatic potential.

    Columns are ordered by increasing energy and normalised so that the sum of
    the squared amplitudes times the grid spacing equals one. The sign of each
    column is fixed so that the column is positive at the grid point where its
    amplitude is largest in absolute value; among the grid points reaching that
    maximum to within a relative tolerance of 1e-8, the smallest index is used.

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
    waves : np.ndarray
        (N, n_states) wave function amplitudes in inverse square root angstrom.

    Raises
    ------
    ValueError
        If the grid and potential do not match, if the grid is not uniformly
        spaced and increasing, if N is even or smaller than 3, if n_states is
        out of range, or if hbar2_over_2m is not positive.
    '''
    return waves  # placeholder

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


def _oracle_vibrational_wavefunctions(r_grid: np.ndarray, potential: np.ndarray,
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
# normal: the product state of the assembly
r_grid = np.linspace(-0.90, 0.90, 201)
u = r_grid / 0.400
v = 0.610 * (u * u - 1.0) ** 2 - 0.175 * u + 3.100
""",
            "call": "vibrational_wavefunctions(r_grid, v, 2.07500e-3, 3)",
            "gold_call": "_oracle_vibrational_wavefunctions(r_grid, v, 2.07500e-3, 3)",
        },
        {
            "setup": """import numpy as np
# normal: check the grid normalisation directly
r_grid = np.linspace(-0.90, 0.90, 201)
u = r_grid / 0.400
v = 0.290 * (u * u - 1.0) ** 2 - 0.080 * u + 3.615
def norms(f):
    w = f(r_grid, v, 2.07500e-3, 3)
    return (w ** 2).sum(axis=0) * (r_grid[1] - r_grid[0])
""",
            "call": "norms(vibrational_wavefunctions)",
            "gold_call": "norms(_oracle_vibrational_wavefunctions)",
        },
        {
            "setup": """import numpy as np
# boundary: a harmonic well, whose ground state is a single positive lobe
r_grid = np.linspace(-0.60, 0.60, 121)
v = 0.5 * 30.0 * r_grid ** 2
""",
            "call": "vibrational_wavefunctions(r_grid, v, 2.07500e-3, 2)",
            "gold_call": "_oracle_vibrational_wavefunctions(r_grid, v, 2.07500e-3, 2)",
        },
        {
            "setup": """import numpy as np
# edge: a symmetric double well, where every state has two extrema of equal
# magnitude and the tie-breaking rule is the only thing that pins the phase.
# Probing fixed grid points inside each well keeps the check independent of
# whichever extremum a search happens to return. The barrier is kept low
# enough that the levels are split by 7.9e-4 eV, so the eigenvectors are
# determined to about 3e-11 and the comparison is not a coin flip.
r_grid = np.linspace(-0.90, 0.90, 201)
u = r_grid / 0.400
v = 0.450 * (u * u - 1.0) ** 2
def probe(f):
    return f(r_grid, v, 2.07500e-3, 3)[[44, 56, 100, 144, 156], :]
""",
            "call": "probe(vibrational_wavefunctions)",
            "gold_call": "probe(_oracle_vibrational_wavefunctions)",
        },
        {
            "setup": """import numpy as np
# edge: two states on a shallower symmetric well, so the rule has to be
# applied to each column separately rather than to the block. A shallower
# barrier is used deliberately: raise it and the two lowest levels approach
# degeneracy, their eigenvectors mix almost freely, and the comparison stops
# measuring the submission and starts measuring the eigensolver.
r_grid = np.linspace(-0.90, 0.90, 201)
u = r_grid / 0.400
v = 0.300 * (u * u - 1.0) ** 2
def probe(f):
    return f(r_grid, v, 2.07500e-3, 2)[[56, 100, 144], :]
""",
            "call": "probe(vibrational_wavefunctions)",
            "gold_call": "probe(_oracle_vibrational_wavefunctions)",
        },
        {
            "setup": """import numpy as np
# edge: a single state on a coarse grid
r_grid = np.linspace(-0.50, 0.50, 25)
v = 0.5 * 30.0 * r_grid ** 2
""",
            "call": "vibrational_wavefunctions(r_grid, v, 2.07500e-3, 1)",
            "gold_call": "_oracle_vibrational_wavefunctions(r_grid, v, 2.07500e-3, 1)",
        },
        {
            "setup": """import numpy as np
r_grid = np.linspace(-0.9, 0.9, 21)
v = np.zeros_like(r_grid)
def run(f):
    try:
        f(r_grid, v, 2.07500e-3, 0); return 0        # zero states requested
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(vibrational_wavefunctions)",
            "gold_call": "run(_oracle_vibrational_wavefunctions)",
        },
        {
            "setup": """import numpy as np
r_grid = np.linspace(0.9, -0.9, 21)
v = np.zeros_like(r_grid)
def run(f):
    try:
        f(r_grid, v, 2.07500e-3, 3); return 0        # decreasing grid
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(vibrational_wavefunctions)",
            "gold_call": "run(_oracle_vibrational_wavefunctions)",
        },
    ]
