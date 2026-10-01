"""
Solve the radial nuclear equation of the exact factorization on the given grid and return that grid together with the ground-state radial function on it.

In the exact factorization the nuclear factor obeys its own equation, which looks like an ordinary Schroedinger equation for a charged particle: the Berry connection of the electronic factor enters as a vector potential and the scalar potential is the Born-Oppenheimer surface together with the correction of the previous step, the whole kinetic term carrying the mass ratio as a prefactor. The equation is written here in the gauge in which the vector potential is purely angular and depends on the radius alone, which is what lets the angular dependence be taken as a single integer angular momentum and reduces the problem to one radial dimension. That radial problem is discretised on an equally spaced grid with the standard three-point second difference and with the radial function held at zero at both ends of the grid, and its ground state is returned together with the grid it lives on.

Returns
-------
np.ndarray of shape (2, n_grid): the radial grid in row 0 and the ground-state radial function of the nuclear factor in row 1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nuclear_radial_state(g: float, K: float, Delta: float, m: int, M0: float,
                         mu: float, Q_min: float, Q_max: float,
                         n_grid: int) -> "np.ndarray":
    '''Ground state of the radial nuclear equation on the given grid.

    Parameters
    ----------
    g : float
        Linear vibronic coupling strength, strictly positive.
    K : float
        Harmonic force constant of the two normal modes.
    Delta : float
        Strength of the constant term that opens the gap, strictly positive.
    m : int
        Integer angular momentum quantum number of the nuclear factor.
    M0 : float
        Nuclear mass carried by the two normal modes, strictly positive.
    mu : float
        Electron-to-nuclear mass ratio, strictly positive.
    Q_min, Q_max : float
        Ends of the radial grid, with 0 < Q_min < Q_max. The radial function is
        held at zero outside them.
    n_grid : int
        Number of equally spaced grid points, at least 16, so that the spacing
        is (Q_max - Q_min) / (n_grid - 1).

    Returns
    -------
    state : np.ndarray
        Real array of shape (2, n_grid). Row 0 is the radial grid and row 1 is
        the radial part of the nuclear factor on it, real, scaled so that its
        largest magnitude is one and so that it is positive there.

    Raises
    ------
    ValueError
        If g, K or Delta is not a finite scalar, if g or Delta is not strictly
        positive, if m is not an integer scalar, if M0 or mu is not a finite
        positive scalar, if the grid ends do not satisfy 0 < Q_min < Q_max, or
        if n_grid is not an integer of at least 16.
    '''
    return state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_point(Q, theta, g, K, Delta):
    vals = [Q, theta, g, K, Delta]
    if any(np.ndim(v) != 0 for v in vals):
        raise ValueError("Q, theta, g, K and Delta must all be scalars")
    Q, theta, g, K, Delta = (float(v) for v in vals)
    if not np.all(np.isfinite([Q, theta, g, K, Delta])):
        raise ValueError("every model parameter must be finite")
    if Q <= 0.0:
        raise ValueError(
            "Q must be strictly positive: the loop may not pass through the origin"
        )
    if g <= 0.0:
        raise ValueError("g must be strictly positive")
    if Delta <= 0.0:
        raise ValueError(
            "Delta must be strictly positive: at Delta = 0 the lower state is "
            "antiperiodic and the phase convention below cannot be imposed"
        )
    return Q, theta, g, K, Delta


def _grad_alpha(Q, theta, g, Delta):
    """(d alpha / dQ, (1/Q) d alpha / d theta) of the azimuth of the field direction."""
    S2 = g * g * Q * Q * np.sin(theta) ** 2 + Delta * Delta
    return np.array([
        Delta * g * np.sin(theta) / S2,
        Delta * g * np.cos(theta) / S2,
    ])


def _check_nuclear(m, M0, mu):
    if np.ndim(m) != 0 or np.ndim(M0) != 0 or np.ndim(mu) != 0:
        raise ValueError("m, M0 and mu must be scalars")
    if int(m) != m:
        raise ValueError("m must be an integer angular momentum quantum number")

    M0, mu = float(M0), float(mu)

    if not np.isfinite(M0) or M0 <= 0.0:
        raise ValueError("M0 must be a finite positive scalar")
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be a finite positive scalar")

    return int(m), M0, mu


def _check_grid(Q_min, Q_max, n_grid):
    if (
        np.ndim(Q_min) != 0
        or np.ndim(Q_max) != 0
        or np.ndim(n_grid) != 0
    ):
        raise ValueError("Q_min, Q_max and n_grid must be scalars")

    Q_min, Q_max = float(Q_min), float(Q_max)

    if (
        not (np.isfinite(Q_min) and np.isfinite(Q_max))
        or Q_min <= 0.0
        or Q_max <= Q_min
    ):
        raise ValueError("require 0 < Q_min < Q_max, both finite")

    if int(n_grid) != n_grid or int(n_grid) < 16:
        raise ValueError("n_grid must be an integer of at least 16")

    return Q_min, Q_max, int(n_grid)


def _ground_state(diag, off):
    """Lowest eigenpair of a real symmetric tridiagonal matrix."""
    try:
        from scipy.linalg import eigh_tridiagonal
    except Exception:
        eigh_tridiagonal = None

    if eigh_tridiagonal is not None:
        val, vec = eigh_tridiagonal(
            diag,
            off,
            select="i",
            select_range=(0, 0),
        )
        return float(val[0]), vec[:, 0]

    H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    val, vec = np.linalg.eigh(H)
    return float(val[0]), vec[:, 0]


def _oracle_nuclear_radial_state(
    g: float,
    K: float,
    Delta: float,
    m: int,
    M0: float,
    mu: float,
    Q_min: float,
    Q_max: float,
    n_grid: int,
) -> "np.ndarray":
    _check_point(1.0, 0.0, g, K, Delta)
    m, M0, mu = _check_nuclear(m, M0, mu)
    Q_min, Q_max, n_grid = _check_grid(Q_min, Q_max, n_grid)

    Q = np.linspace(Q_min, Q_max, n_grid)
    h = Q[1] - Q[0]
    kinetic = mu / (2.0 * M0)

    W = np.sqrt(g * g * Q * Q + Delta * Delta)
    phase = np.pi * (1.0 - Delta / W)
    A_theta = phase / (2.0 * np.pi * Q)
    surface = 0.5 * K * Q * Q - W

    corr = np.array([
        _oracle_diagonal_correction(q, 0.0, g, K, Delta, M0)
        for q in Q
    ])

    V = (
        kinetic
        * (
            (m / Q + A_theta) ** 2
            - 1.0 / (4.0 * Q * Q)
        )
        + surface
        + mu * corr
    )

    diag = 2.0 * kinetic / (h * h) + V
    off = -kinetic / (h * h) * np.ones(n_grid - 1)

    _, u = _ground_state(diag, off)

    if u[int(np.argmax(np.abs(u)))] < 0.0:
        u = -u

    R = u / np.sqrt(Q)
    R = R / np.abs(R).max()

    return np.stack([Q, R])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np\n",
         "call": "np.round(nuclear_radial_state(0.60, 1.00, 0.12, 3, 100.0, 0.01, 0.02, 2.02, 4001)[:, ::400], 9)",
         "gold_call": "np.round(_oracle_nuclear_radial_state(0.60, 1.00, 0.12, 3, 100.0, 0.01, 0.02, 2.02, 4001)[:, ::400], 9)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(nuclear_radial_state(0.60, 1.00, 0.12, 0, 100.0, 0.01, 0.02, 2.02, 1001)[:, ::100], 9)",
         "gold_call": "np.round(_oracle_nuclear_radial_state(0.60, 1.00, 0.12, 0, 100.0, 0.01, 0.02, 2.02, 1001)[:, ::100], 9)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(nuclear_radial_state(0.60, 1.00, 0.12, -3, 100.0, 0.01, 0.02, 2.02, 1001)[:, ::100], 9)",
         "gold_call": "np.round(_oracle_nuclear_radial_state(0.60, 1.00, 0.12, -3, 100.0, 0.01, 0.02, 2.02, 1001)[:, ::100], 9)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(nuclear_radial_state(0.45, 0.80, 0.30, 2, 40.0, 0.05, 0.05, 3.00, 801)[:, ::80], 9)",
         "gold_call": "np.round(_oracle_nuclear_radial_state(0.45, 0.80, 0.30, 2, 40.0, 0.05, 0.05, 3.00, 801)[:, ::80], 9)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(nuclear_radial_state(0.60, 1.00, 0.12, 7, 100.0, 0.01, 0.02, 2.02, 1001)[:, ::100], 9)",
         "gold_call": "np.round(_oracle_nuclear_radial_state(0.60, 1.00, 0.12, 7, 100.0, 0.01, 0.02, 2.02, 1001)[:, ::100], 9)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(nuclear_radial_state(0.90, 1.50, 0.05, 1, 250.0, 0.004, 0.10, 1.60, 401)[:, ::40], 9)",
         "gold_call": "np.round(_oracle_nuclear_radial_state(0.90, 1.50, 0.05, 1, 250.0, 0.004, 0.10, 1.60, 401)[:, ::40], 9)"},
    ]
