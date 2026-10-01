"""
Compute the lowest energy levels of the trigonometric double well exactly, in the reduced energy units of the model.

The trigonometric double well is one of the few smooth double wells, infinite at both ends of its interval, whose Schrödinger equation maps onto a classical special-function eigenproblem, so its levels are available to rounding accuracy rather than from a discretized grid.

Returns
-------
np.ndarray: ascending reduced energies eps_0 ... eps_{n_levels-1} of shape (n_levels,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_well_energy_levels(m: int, p: float, n_levels: int) -> "np.ndarray":
    """Return the lowest energy levels of the trigonometric double well.

    The levels ``eps_0 < eps_1 < ...`` are the eigenvalues of
    ``psi'' + (eps - U(x)) psi = 0`` on ``-pi/2 < x < pi/2`` with
    ``U(x) = (m**2 - 1/4) tan(x)**2 - p**2 sin(x)**2`` and ``psi`` vanishing
    at both ends, in the reduced units in which the barrier top is
    ``U(0) = 0``. Each returned level must be exact up to rounding: relative
    error below ``1e-10`` (absolute error below ``1e-10`` when
    ``|eps| < 1``).

    Parameters
    ----------
    m : int
        Order of the potential, a positive integer.
    p : float
        Positive potential parameter.
    n_levels : int
        Number of levels to return, a positive integer.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_levels,)`` holding ``eps_0, ...,
        eps_{n_levels-1}`` in ascending order.

    Raises
    ------
    ValueError
        If ``m`` or ``n_levels`` is not a positive integer (booleans are
        rejected) or if ``p`` is not a finite positive real number.
    """
    return levels

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_well_energy_levels(m: int, p: float, n_levels: int) -> "np.ndarray":
    """Reference implementation: spheroidal eigenproblem in a normalized associated Legendre basis."""
    import math
    import numpy as np

    for name, value in (("m", m), ("n_levels", n_levels)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(p, bool) or not isinstance(p, (int, float, np.integer, np.floating)):
        raise ValueError("p must be a real number")
    if not (math.isfinite(p) and p > 0.0):
        raise ValueError("p must be finite and positive")
    return _well_spectral_problem(int(m), float(p), int(n_levels))[0]


def _well_spectral_problem(m: int, p: float, n_levels: int) -> tuple:
    """Levels, basis coefficient columns and basis degrees of the lowest n_levels states.

    With eta = sin(x) and psi = sqrt(cos x) S(eta), S solves the oblate angular
    spheroidal equation with eigenvalue eps + m^2 - 1/2 and parameter p. In the
    normalized associated Legendre functions of degrees n = m, m + 1, ... the
    operator is n(n + 1) on the diagonal minus p^2 times the matrix of eta^2,
    which couples only degrees of equal parity, so each parity block is tridiagonal.
    """
    import numpy as np
    from scipy.linalg import eigh_tridiagonal

    size = 2 * n_levels + int(2.0 * p) + 60
    degrees = m + np.arange(size, dtype=float)
    # eta P_n = a_n P_{n+1} + a_{n-1} P_{n-1} for the normalized functions.
    a = np.sqrt(((degrees + 1.0) ** 2 - m * m) / ((2.0 * degrees + 1.0) * (2.0 * degrees + 3.0)))
    a_below = np.concatenate(([0.0], a[:-1]))
    diagonal = degrees * (degrees + 1.0) - p * p * (a * a + a_below * a_below)
    coupling = -p * p * a[:-2] * a[1:-1]
    values, columns = [], []
    for parity in (0, 1):
        index = np.arange(parity, size, 2)
        block_values, block_vectors = eigh_tridiagonal(diagonal[index], coupling[index[:-1]])
        full = np.zeros((size, block_values.size))
        full[index, :] = block_vectors
        values.append(block_values)
        columns.append(full)
    values = np.concatenate(values)
    columns = np.hstack(columns)
    order = np.argsort(values, kind="stable")[:n_levels]
    return values[order] + 0.5 - m * m, columns[:, order], degrees

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    reduce = (
        "import numpy as np\n"
        "def _sig(v):\n"
        "    v = np.asarray(v, dtype=float).ravel()\n"
        "    rank = np.arange(1.0, v.size + 1.0)\n"
        "    return float(v.size + np.sum(rank * v) / np.sum(rank) + np.sqrt(np.sum(v * v)) / v.size)\n"
    )
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": reduce,
            "call": "_sig(compute_well_energy_levels(18, 27.089280949903493, 8))",
            "gold_call": "_sig(_oracle_compute_well_energy_levels(18, 27.089280949903493, 8))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(compute_well_energy_levels(2, 7.82971, 4)[1])",
            "gold_call": "float(_oracle_compute_well_energy_levels(2, 7.82971, 4)[1])",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(compute_well_energy_levels(18, 27.089280949903493, 4)[3])",
            "gold_call": "float(_oracle_compute_well_energy_levels(18, 27.089280949903493, 4)[3])",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(compute_well_energy_levels(1, 3.0, 1)[0])",
            "gold_call": "float(_oracle_compute_well_energy_levels(1, 3.0, 1)[0])",
        },
        {
            "setup": reduce,
            "call": "_sig(compute_well_energy_levels(44, 55.4907031, 60))",
            "gold_call": "_sig(_oracle_compute_well_energy_levels(44, 55.4907031, 60))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_well_energy_levels(0, 27.0, 3))",
            "gold_call": "_status(lambda: _oracle_compute_well_energy_levels(0, 27.0, 3))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_well_energy_levels(18, -1.0, 3))",
            "gold_call": "_status(lambda: _oracle_compute_well_energy_levels(18, -1.0, 3))",
        },
    ]
