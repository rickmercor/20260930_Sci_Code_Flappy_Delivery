"""
Return the nuclear momentum function of the exact-factorization framework at the loop radius, built from the solved nuclear ground state.

The nuclear factor enters the electronic equation of motion only through the nuclear momentum function of the exact-factorization framework, and that is what this step returns, at a single radius and in the same orthonormal polar frame as everything else. It is built from the ground state of the previous step, so the radius asked for has to be one of that grid's points; the radial derivative is taken there with the same three-point central difference the grid already implies. The overall scale of the radial function cancels out, so the normalisation convention of the previous step does not matter here, and the result does not depend on the polar angle.

Returns
-------
np.ndarray of shape (2, 2): real part in row 0 and imaginary part in row 1 of the nuclear momentum function, radial then angular component
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nuclear_momentum(Q: float, g: float, K: float, Delta: float, m: int,
                     M0: float, mu: float, Q_min: float, Q_max: float,
                     n_grid: int) -> "np.ndarray":
    '''Nuclear momentum function at the loop radius.

    Parameters
    ----------
    Q : float
        Loop radius. It must coincide with one of the interior points of the
        radial grid defined by Q_min, Q_max and n_grid.
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
        Ends of the radial grid, with 0 < Q_min < Q_max.
    n_grid : int
        Number of equally spaced grid points, at least 16.

    Returns
    -------
    momentum : "np.ndarray"
        Real array of shape (2, 2). Row 0 holds the real part and row 1 the
        imaginary part; within a row column 0 is the radial component and
        column 1 the angular component, in the orthonormal polar frame.

    Raises
    ------
    ValueError
        If any scalar argument is not finite, if g, Delta, M0 or mu is not
        strictly positive, if m is not an integer scalar, if the grid ends do
        not satisfy 0 < Q_min < Q_max, if n_grid is not an integer of at least
        16, or if Q is not an interior point of that grid.
    '''
    return momentum

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


def _oracle_nuclear_momentum(
    Q: float,
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
    Q, _, g, K, Delta = _check_point(Q, 0.0, g, K, Delta)
    m, M0, mu = _check_nuclear(m, M0, mu)
    Q_min, Q_max, n_grid = _check_grid(Q_min, Q_max, n_grid)

    state = _oracle_nuclear_radial_state(
        g,
        K,
        Delta,
        m,
        M0,
        mu,
        Q_min,
        Q_max,
        n_grid,
    )
    grid, R = state[0], state[1]
    h = grid[1] - grid[0]
    k = int(round((Q - Q_min) / h))

    if (
        k < 1
        or k > n_grid - 2
        or abs(grid[k] - Q) > 1e-9 * max(1.0, abs(Q))
    ):
        raise ValueError("Q must be an interior point of the radial grid")

    log_derivative = (
        (R[k + 1] - R[k - 1])
        / (2.0 * h)
        / R[k]
    )

    W = np.sqrt(g * g * Q * Q + Delta * Delta)
    A_theta = (
        np.pi * (1.0 - Delta / W)
        / (2.0 * np.pi * Q)
    )

    p = np.array(
        [-1j * log_derivative, m / Q + A_theta],
        dtype=complex,
    )

    return np.stack([p.real, p.imag])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np\n",
         "call": "np.round(nuclear_momentum(0.35, 0.60, 1.00, 0.12, 3, 100.0, 0.01, 0.02, 2.02, 4001), 8)",
         "gold_call": "np.round(_oracle_nuclear_momentum(0.35, 0.60, 1.00, 0.12, 3, 100.0, 0.01, 0.02, 2.02, 4001), 8)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(nuclear_momentum(0.62, 0.60, 1.00, 0.12, 3, 100.0, 0.01, 0.02, 2.02, 1001), 8)",
         "gold_call": "np.round(_oracle_nuclear_momentum(0.62, 0.60, 1.00, 0.12, 3, 100.0, 0.01, 0.02, 2.02, 1001), 8)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(nuclear_momentum(0.35, 0.60, 1.00, 0.12, 0, 100.0, 0.01, 0.02, 2.02, 1001), 8)",
         "gold_call": "np.round(_oracle_nuclear_momentum(0.35, 0.60, 1.00, 0.12, 0, 100.0, 0.01, 0.02, 2.02, 1001), 8)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(nuclear_momentum(0.10, 0.60, 1.00, 0.12, -3, 100.0, 0.01, 0.02, 2.02, 1001), 8)",
         "gold_call": "np.round(_oracle_nuclear_momentum(0.10, 0.60, 1.00, 0.12, -3, 100.0, 0.01, 0.02, 2.02, 1001), 8)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(nuclear_momentum(1.00, 0.45, 0.80, 0.30, 2, 40.0, 0.05, 0.05, 3.00, 1181), 8)",
         "gold_call": "np.round(_oracle_nuclear_momentum(1.00, 0.45, 0.80, 0.30, 2, 40.0, 0.05, 0.05, 3.00, 1181), 8)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(nuclear_momentum(0.55, 0.90, 1.50, 0.05, 1, 250.0, 0.004, 0.10, 1.60, 401), 8)",
         "gold_call": "np.round(_oracle_nuclear_momentum(0.55, 0.90, 1.50, 0.05, 1, 250.0, 0.004, 0.10, 1.60, 401), 8)"},
    ]
