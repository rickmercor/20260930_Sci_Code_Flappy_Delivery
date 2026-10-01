"""
Run the whole pipeline and return the correction to the geometric phase as a percentage of the uncorrected phase on the same loop.

The whole pipeline is assembled here. The nuclear factor is solved once, its momentum function is evaluated at the loop radius once, and the two loop integrals are then taken around the same circle with the same equally spaced grid theta_k = 2 pi k / n_theta: the uncorrected geometric phase of the lower adiabatic state, and the correction to it. The reported number is the correction expressed as a percentage of the uncorrected phase, that is one hundred times their ratio. It is negative, because the electron-nuclear correlation reduces the magnitude of the phase, and its size is set by the mass ratio, by the state the nuclei are actually in, and by how far the loop sits outside the region where the gap term dominates the vibronic term.

Returns
-------
float: the leading nonadiabatic correction to the geometric phase of the lower adiabatic state, expressed as a percentage of the uncorrected phase
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nonadiabatic_phase_ratio(Q: float = 0.35, g: float = 0.60,
                             K: float = 1.00, Delta: float = 0.12,
                             m: int = 3, M0: float = 100.0,
                             mu: float = 0.01,
                             Q_min: float = 0.02,
                             Q_max: float = 2.02,
                             n_grid: int = 4001,
                             n_theta: int = 2048) -> float:
    '''Nonadiabatic correction to the geometric phase, as a percentage.

    Parameters
    ----------
    Q : float
        Radius of the circular nuclear contour. It must coincide with one of
        the interior points of the radial grid.
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
        Number of equally spaced radial grid points, at least 16.
    n_theta : int
        Number of equally spaced quadrature points on the loop, at least 8.
        The grid is theta_k = 2 pi k / n_theta for k = 0 .. n_theta - 1.

    Returns
    -------
    ratio : float
        One hundred times the ratio of the correction to the geometric phase
        to the uncorrected geometric phase, both on the same loop.

    Raises
    ------
    ValueError
        If any scalar argument is not finite, if Q, g, Delta, M0 or mu is not
        strictly positive, if m is not an integer scalar, if the grid ends do
        not satisfy 0 < Q_min < Q_max, if n_grid is not an integer of at least
        16, if n_theta is not an integer of at least 8, or if Q is not an
        interior point of the radial grid.
    '''
    return ratio

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


def _oracle_nonadiabatic_phase_ratio(
    Q: float = 0.35,
    g: float = 0.6,
    K: float = 1.0,
    Delta: float = 0.12,
    m: int = 3,
    M0: float = 100.0,
    mu: float = 0.01,
    Q_min: float = 0.02,
    Q_max: float = 2.02,
    n_grid: int = 4001,
    n_theta: int = 2048,
) -> float:
    Q, _, g, K, Delta = _check_point(Q, 0.0, g, K, Delta)
    m, M0, mu = _check_nuclear(m, M0, mu)
    Q_min, Q_max, n_grid = _check_grid(
        Q_min,
        Q_max,
        n_grid,
    )

    if (
        np.ndim(n_theta) != 0
        or int(n_theta) != n_theta
        or int(n_theta) < 8
    ):
        raise ValueError("n_theta must be an integer of at least 8")

    n = int(n_theta)

    momentum = _oracle_nuclear_momentum(
        Q,
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

    zeroth = _oracle_bo_geometric_phase(
        Q,
        g,
        K,
        Delta,
        n,
    )

    grid = 2.0 * np.pi * np.arange(n) / n
    total = 0.0

    for th in grid:
        total += _oracle_connection_correction(
            Q,
            th,
            g,
            K,
            Delta,
            M0,
            mu,
            momentum,
        )[1]

    first = total * Q * 2.0 * np.pi / n
    return float(100.0 * first / zeroth)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np\n",
         "call": "round(nonadiabatic_phase_ratio(), 8)",
         "gold_call": "round(_oracle_nonadiabatic_phase_ratio(), 8)"},
        {"setup": "import numpy as np\n",
         "call": "round(nonadiabatic_phase_ratio(0.35, 0.60, 1.00, 0.12, 3, 100.0, 0.01, 0.02, 2.02, 4001, 512), 8)",
         "gold_call": "round(_oracle_nonadiabatic_phase_ratio(0.35, 0.60, 1.00, 0.12, 3, 100.0, 0.01, 0.02, 2.02, 4001, 512), 8)"},
        {"setup": "import numpy as np\n",
         "call": "round(nonadiabatic_phase_ratio(0.35, 0.60, 1.00, 0.12, 0, 100.0, 0.01, 0.02, 2.02, 1001, 512), 8)",
         "gold_call": "round(_oracle_nonadiabatic_phase_ratio(0.35, 0.60, 1.00, 0.12, 0, 100.0, 0.01, 0.02, 2.02, 1001, 512), 8)"},
        {"setup": "import numpy as np\n",
         "call": "round(nonadiabatic_phase_ratio(0.10, 0.60, 1.00, 0.12, 3, 100.0, 0.01, 0.02, 2.02, 1001, 512), 8)",
         "gold_call": "round(_oracle_nonadiabatic_phase_ratio(0.10, 0.60, 1.00, 0.12, 3, 100.0, 0.01, 0.02, 2.02, 1001, 512), 8)"},
        {"setup": "import numpy as np\n",
         "call": "round(nonadiabatic_phase_ratio(1.00, 0.45, 0.80, 0.30, 2, 40.0, 0.05, 0.05, 3.00, 1181, 512), 8)",
         "gold_call": "round(_oracle_nonadiabatic_phase_ratio(1.00, 0.45, 0.80, 0.30, 2, 40.0, 0.05, 0.05, 3.00, 1181, 512), 8)"},
        {"setup": "import numpy as np\n",
         "call": "round(nonadiabatic_phase_ratio(0.55, 0.90, 1.50, 0.05, 1, 250.0, 0.004, 0.10, 1.60, 401, 512), 8)",
         "gold_call": "round(_oracle_nonadiabatic_phase_ratio(0.55, 0.90, 1.50, 0.05, 1, 250.0, 0.004, 0.10, 1.60, 401, 512), 8)"},
    ]
