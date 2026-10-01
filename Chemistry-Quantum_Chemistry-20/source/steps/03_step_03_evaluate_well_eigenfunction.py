"""
Evaluate a normalized eigenfunction of the trigonometric double well and its coordinate derivative at given points, exactly up to rounding.

Flux and transmission constructions built on exact stationary states need the wave function and its slope at particular points of the well, so the eigenfunction must be evaluated from its exact representation with a fixed normalization and sign.

Returns
-------
np.ndarray: shape (2, x.size), row 0 the normalized eigenfunction psi_q(x) and row 1 its derivative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_well_eigenfunction(m: int, p: float, q: int, x: "np.ndarray") -> "np.ndarray":
    """Return the q-th eigenfunction of the trigonometric double well and its derivative.

    ``psi_q`` is the eigenfunction of ``psi'' + (eps - U(x)) psi = 0`` on
    ``-pi/2 < x < pi/2``, with ``U(x) = (m**2 - 1/4) tan(x)**2 -
    p**2 sin(x)**2`` and ``psi`` vanishing at both ends, that belongs to the
    level ``eps_q`` of ``compute_well_energy_levels`` (``q = 0`` is the
    ground state). It is normalized so that the integral of ``psi_q**2`` over
    the interval is one, and its sign is chosen so that ``psi_q(x_min) > 0``
    at the minimum ``x_min > 0`` of ``U``. Returned values must be exact up
    to rounding: relative error below ``1e-10`` (absolute error below
    ``1e-10`` for magnitudes below one).

    Parameters
    ----------
    m : int
        Order of the potential, a positive integer.
    p : float
        Potential parameter with ``p**2 > m**2 - 1/4``, so that ``U`` is a
        double well.
    q : int
        Level index, a nonnegative integer.
    x : np.ndarray
        Nonempty one-dimensional array of points with ``|x| < pi/2``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(2, x.size)``: row 0 holds ``psi_q(x)`` and row
        1 holds ``d psi_q / dx``.

    Raises
    ------
    ValueError
        If ``m`` is not a positive integer or ``q`` a nonnegative integer
        (booleans are rejected), if ``p`` is not finite with
        ``p**2 > m**2 - 1/4``, or if ``x`` is empty, not one-dimensional,
        not finite or has a point with ``|x| >= pi/2``.
    """
    return values

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_evaluate_well_eigenfunction(m: int, p: float, q: int, x: "np.ndarray") -> "np.ndarray":
    """Reference implementation: psi_q = sqrt(cos x) S(sin x) from the Legendre-basis coefficients."""
    import math
    import numpy as np

    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("m must be a positive integer")
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q < 0:
        raise ValueError("q must be a nonnegative integer")
    if isinstance(p, bool) or not isinstance(p, (int, float, np.integer, np.floating)):
        raise ValueError("p must be a real number")
    if not (math.isfinite(p) and p * p > m * m - 0.25):
        raise ValueError("p must satisfy p^2 > m^2 - 1/4")
    points = np.asarray(x, dtype=float)
    if points.ndim != 1 or points.size == 0 or not np.all(np.abs(points) < 0.5 * math.pi):
        raise ValueError("x must be a nonempty 1-D array strictly inside (-pi/2, pi/2)")
    m, p, q = int(m), float(p), int(q)
    _, columns, degrees = _well_spectral_problem(m, p, q + 1)
    x_min = math.acos(((m * m - 0.25) / (p * p)) ** 0.25)
    grid = np.append(points, x_min)
    legendre, slope = _normalized_legendre(m, degrees.size, np.sin(grid))
    angular = columns[:, q] @ legendre
    angular_slope = columns[:, q] @ slope
    cosine = np.cos(grid)
    psi = np.sqrt(cosine) * angular
    dpsi = -np.sin(grid) * angular / (2.0 * np.sqrt(cosine)) + cosine ** 1.5 * angular_slope
    sign = 1.0 if psi[-1] > 0.0 else -1.0
    return sign * np.vstack([psi[:-1], dpsi[:-1]])


def _normalized_legendre(m: int, size: int, eta: "np.ndarray") -> tuple:
    """Unit-norm associated Legendre functions of degrees m .. m + size - 1 and their eta-derivatives."""
    import math
    import numpy as np

    one_minus = 1.0 - eta * eta
    values = np.zeros((size, eta.size))
    log_first = (0.5 * math.log(m + 0.5) + 0.5 * math.lgamma(2 * m + 1)
                 - m * math.log(2.0) - math.lgamma(m + 1))
    values[0] = np.exp(log_first + 0.5 * m * np.log(one_minus))
    if size > 1:
        values[1] = eta * math.sqrt(2 * m + 3.0) * values[0]
    for k in range(2, size):
        n = m + k - 1
        up = math.sqrt(((n + 1.0) ** 2 - m * m) / ((2 * n + 1.0) * (2 * n + 3.0)))
        down = math.sqrt((n * n - m * m) / ((2 * n - 1.0) * (2 * n + 1.0)))
        values[k] = (eta * values[k - 1] - down * values[k - 2]) / up
    slopes = np.empty_like(values)
    for k in range(size):
        n = m + k
        slopes[k] = -n * eta * values[k]
        if k > 0:
            ratio = math.sqrt((2 * n + 1.0) * (n - m) / ((2 * n - 1.0) * (n + m)))
            slopes[k] += (n + m) * ratio * values[k - 1]
    return values, slopes / one_minus

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    reduce = (
        "import numpy as np\n"
        "def _sig(v):\n"
        "    v = np.asarray(v, dtype=float)\n"
        "    if v.ndim != 2 or v.shape[0] != 2:\n"
        "        return -1.0\n"
        "    w = np.cos(np.arange(v.size, dtype=float) + 1.0)\n"
        "    flat = 1.0e2 * v.ravel()\n"
        "    return float(v.shape[1] + np.sum(np.abs(flat)) + flat @ w)\n"
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
            "call": "_sig(evaluate_well_eigenfunction(18, 27.089280949903493, 0, np.array([-1.2, -0.618, 0.0, 0.25, 0.618, 1.2])))",
            "gold_call": "_sig(_oracle_evaluate_well_eigenfunction(18, 27.089280949903493, 0, np.array([-1.2, -0.618, 0.0, 0.25, 0.618, 1.2])))",
        },
        {
            "setup": reduce,
            "call": "_sig(evaluate_well_eigenfunction(18, 27.089280949903493, 2, np.array([-0.9, -0.3, 0.1, 0.35, 0.618, 0.9])))",
            "gold_call": "_sig(_oracle_evaluate_well_eigenfunction(18, 27.089280949903493, 2, np.array([-0.9, -0.3, 0.1, 0.35, 0.618, 0.9])))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(evaluate_well_eigenfunction(18, 27.089280949903493, 1, np.array([0.0]))[1, 0])",
            "gold_call": "float(_oracle_evaluate_well_eigenfunction(18, 27.089280949903493, 1, np.array([0.0]))[1, 0])",
        },
        {
            "setup": reduce,
            "call": "_sig(evaluate_well_eigenfunction(2, 7.82971, 3, np.array([-1.3, -0.5, 0.2, 1.05, 1.4])))",
            "gold_call": "_sig(_oracle_evaluate_well_eigenfunction(2, 7.82971, 3, np.array([-1.3, -0.5, 0.2, 1.05, 1.4])))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(1.0e3 * evaluate_well_eigenfunction(44, 55.4907031, 1, np.array([0.35]))[0, 0])",
            "gold_call": "float(1.0e3 * _oracle_evaluate_well_eigenfunction(44, 55.4907031, 1, np.array([0.35]))[0, 0])",
        },
        {
            "setup": status,
            "call": "_status(lambda: evaluate_well_eigenfunction(18, 27.089280949903493, 0, np.array([0.2, 1.5707963267948966])))",
            "gold_call": "_status(lambda: _oracle_evaluate_well_eigenfunction(18, 27.089280949903493, 0, np.array([0.2, 1.5707963267948966])))",
        },
        {
            "setup": status,
            "call": "_status(lambda: evaluate_well_eigenfunction(18, 4.0, 0, np.array([0.2])))",
            "gold_call": "_status(lambda: _oracle_evaluate_well_eigenfunction(18, 4.0, 0, np.array([0.2])))",
        },
    ]
