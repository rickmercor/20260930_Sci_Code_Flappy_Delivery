"""
Locate every equilibrium of a gene class's slow chromatin-mark flow and report the linear rate at which the flow grows or decays about each one.

A saturating self-reinforcing mark obeys a scalar flow that carries either one equilibrium or a pair of attractors separated by a repelling watershed, depending on the feedback strength, the gain and the external input.

Returns
-------
np.ndarray: float array of shape (M, 2), rows [equilibrium mark, linear rate], sorted by mark.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_epigenetic_equilibria(alpha: float, beta: float, c: float,
                                scan_points: int = 4001) -> "np.ndarray":
    """Return the equilibria of the slow mark flow together with their linear rates.

    The equilibria are the solutions of ``alpha * tanh(beta * (theta + c)) =
    theta``. The residual ``alpha * tanh(beta * (theta + c)) - theta`` is
    sampled at ``scan_points`` uniformly spaced locations over the interval
    ``[-(alpha + 1), alpha + 1]``, which always contains all equilibria. When
    the residual is non-monotone, its analytic stationary points are added to
    the scan so that close root pairs and double roots at folds are retained.
    Each sign-changing bracket is refined to double precision with Brent's
    method, and a sampled zero is taken as an equilibrium directly. Two
    equilibria closer together than ``1e-10`` are reported once.

    Row ``k`` of the result holds ``[theta_k, rate_k]``, where ``rate_k`` is the
    derivative ``alpha * beta / cosh(beta * (theta_k + c))**2 - 1`` of the
    residual at that equilibrium, so a negative rate marks an attractor and a
    positive one a repelling watershed. Rows are ordered by increasing
    ``theta_k``.

    Parameters
    ----------
    alpha : float
        Nonnegative finite feedback strength.
    beta : float
        Positive finite response gain.
    c : float
        Finite external input shared by the whole class.
    scan_points : int
        Number of bracketing samples, at least 3 (booleans are rejected).

    Returns
    -------
    np.ndarray
        Float array of shape ``(M, 2)`` with ``M >= 1`` equilibria.

    Raises
    ------
    ValueError
        If ``alpha`` is negative or not finite, if ``beta`` is not positive and
        finite, if ``c`` is not finite, if ``scan_points`` is not an integer of
        at least 3, or if the scan brackets no equilibrium.
    """
    return table

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _epi_require_scalar(value: object, name: str, low: float, strict: bool) -> float:
    """Return ``value`` as a finite float that respects the lower bound ``low``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " must be a real number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(name + " must be finite")
    if strict and not number > low:
        raise ValueError(name + " must be greater than " + repr(low))
    if not strict and number < low:
        raise ValueError(name + " must be at least " + repr(low))
    return number

def _epi_require_vector(values: object, name: str) -> "np.ndarray":
    """Return ``values`` as a non-empty one-dimensional array of finite floats."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(name + " must be an array of real numbers") from error
    if array.ndim != 1 or array.size == 0 or not bool(np.all(np.isfinite(array))):
        raise ValueError(name + " must be a non-empty 1-D array of finite numbers")
    return array

def _epi_require_count(value: object, name: str, low: int) -> int:
    """Return ``value`` as a Python integer of at least ``low`` (booleans are rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    count = int(value)
    if count < low:
        raise ValueError(name + " must be at least " + str(low))
    return count

def _oracle_solve_epigenetic_equilibria(alpha: float, beta: float, c: float,
                                        scan_points: int = 4001) -> "np.ndarray":
    """Reference implementation using monotonic intervals and Brent refinement."""
    import numpy as np

    alpha = _epi_require_scalar(alpha, "alpha", 0.0, False)
    beta = _epi_require_scalar(beta, "beta", 0.0, True)
    c = _epi_require_scalar(c, "c", -np.inf, False)
    samples = _epi_require_count(scan_points, "scan_points", 3)

    def _residual(value):
        return alpha * np.tanh(beta * (value + c)) - value

    span = alpha + 1.0  # every solution obeys |theta| <= alpha
    grid = np.linspace(-span, span, samples)
    gain = alpha * beta
    if gain > 1.0:
        displacement = np.arccosh(np.sqrt(gain)) / beta
        stationary = np.array([-c - displacement, -c + displacement], dtype=float)
        stationary = stationary[(stationary >= -span) & (stationary <= span)]
        grid = np.unique(np.concatenate((grid, stationary)))
    values = _residual(grid)

    roots = [float(node) for node, value in zip(grid, values) if value == 0.0]
    for index in np.nonzero(values[:-1] * values[1:] < 0.0)[0]:
        roots.append(float(brentq(_residual, grid[index], grid[index + 1],
                                  xtol=1e-15, rtol=4.0 * float(np.finfo(float).eps))))
    if not roots:
        raise ValueError("the scan brackets no equilibrium; increase scan_points")

    marks = np.array(sorted(roots), dtype=float)
    if marks.size > 1:
        keep = np.concatenate(([True], np.diff(marks) > 1e-10))
        marks = marks[keep]
    rates = alpha * beta / np.cosh(beta * (marks + c)) ** 2 - 1.0
    return np.column_stack([marks, rates])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _digest(values, rows, cols):\n"
        "    array = np.asarray(values, dtype=float)\n"
        "    if cols == 0:\n"
        "        if array.ndim != 1 or array.shape[0] != rows:\n"
        "            return -1.0\n"
        "    elif array.shape != (rows, cols):\n"
        "        return -1.0\n"
        "    flat = array.ravel()\n"
        "    phase = np.cos(np.arange(flat.size, dtype=float) + 1.0)\n"
        "    return float(flat.size) + float(np.sum(np.abs(flat)) + flat @ phase)\n"
    )
    fold_setup = digest + (
        "c_fold = 0.5 * np.sqrt(1.0 - 1.0 / 2.0) "
        "- np.arccosh(np.sqrt(2.0)) / 4.0\n"
    )
    status = (
        "import numpy as np\n"
        "def _status(action):\n"
        "    try:\n"
        "        action()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # --- Normal: bistable class with its watershed away from the origin ---
        {
            "setup": digest,
            "call": "_digest(solve_epigenetic_equilibria(0.5, 4.0, 0.11), 3, 2)",
            "gold_call": "_digest(_oracle_solve_epigenetic_equilibria(0.5, 4.0, 0.11), 3, 2)",
        },
        # --- Normal: input beyond the fold, a single attractor survives ---
        {
            "setup": digest,
            "call": "_digest(solve_epigenetic_equilibria(0.5, 4.0, -0.32), 1, 2)",
            "gold_call": "_digest(_oracle_solve_epigenetic_equilibria(0.5, 4.0, -0.32), 1, 2)",
        },
        # --- Boundary: only seven bracketing samples still resolve all three roots ---
        {
            "setup": digest,
            "call": "_digest(solve_epigenetic_equilibria(1.6, 2.0, -0.55, 7), 3, 2)",
            "gold_call": "_digest(_oracle_solve_epigenetic_equilibria(1.6, 2.0, -0.55, 7), 3, 2)",
        },
        # --- Edge: feedback switched off, the only equilibrium sits at zero ---
        {
            "setup": digest,
            "call": "_digest(solve_epigenetic_equilibria(0.0, 4.0, 0.04), 1, 2)",
            "gold_call": "_digest(_oracle_solve_epigenetic_equilibria(0.0, 4.0, 0.04), 1, 2)",
        },
        # --- Edge: feedback too weak to bifurcate at this gain ---
        {
            "setup": digest,
            "call": "_digest(solve_epigenetic_equilibria(0.2, 4.0, 0.03), 1, 2)",
            "gold_call": "_digest(_oracle_solve_epigenetic_equilibria(0.2, 4.0, 0.03), 1, 2)",
        },
        # --- Normal: symmetric class, watershed at the origin ---
        {
            "setup": digest,
            "call": "_digest(solve_epigenetic_equilibria(0.5, 4.0, 0.0), 3, 2)",
            "gold_call": "_digest(_oracle_solve_epigenetic_equilibria(0.5, 4.0, 0.0), 3, 2)",
        },
        # --- Boundary: a close stable/unstable pair lies inside one uniform scan cell ---
        {
            "setup": fold_setup,
            "call": "_digest(solve_epigenetic_equilibria(0.5, 4.0, c_fold - 1e-10), 3, 2)",
            "gold_call": "_digest(_oracle_solve_epigenetic_equilibria(0.5, 4.0, c_fold - 1e-10), 3, 2)",
        },
        # --- Boundary: the double equilibrium at the fold is retained ---
        {
            "setup": fold_setup,
            "call": "_digest(solve_epigenetic_equilibria(0.5, 4.0, c_fold), 2, 2)",
            "gold_call": "_digest(_oracle_solve_epigenetic_equilibria(0.5, 4.0, c_fold), 2, 2)",
        },
        # --- Invalid: negative feedback strength ---
        {
            "setup": status,
            "call": "_status(lambda: solve_epigenetic_equilibria(-0.5, 4.0, 0.0))",
            "gold_call": "_status(lambda: _oracle_solve_epigenetic_equilibria(-0.5, 4.0, 0.0))",
        },
        # --- Invalid: too few bracketing samples ---
        {
            "setup": status,
            "call": "_status(lambda: solve_epigenetic_equilibria(0.5, 4.0, 0.0, 2))",
            "gold_call": "_status(lambda: _oracle_solve_epigenetic_equilibria(0.5, 4.0, 0.0, 2))",
        },
    ]
