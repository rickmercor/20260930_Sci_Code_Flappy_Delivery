"""
Forms the unconstrained conditional projection slope together with the feasibility metrics that decide whether the positivity requirement forces a constrained fallback.

Unconstrained projection slope and its feasibility metrics.

The conditional projection of the variance increment onto the integrated variance sets the slope equal to the ratio of the cross-moment to the conditional mean, so that the first two conditional moments of the increment are matched exactly. Whether that slope can be used is decided by the requirement that the updated variance stays non-negative for every realization of the draw: the slope must be positive, it must not exceed the ceiling implied by the positivity constraint, and the constraint evaluated at the intercept of the projection must be non-negative. The intercept is assembled from the current factor state, the per-factor conditional mean, and the initial variance curve, and the ceiling is formed from the lift weights, the decay speeds, and the cross-moment vector. This step returns the slope and all three feasibility quantities so the next step can apply the fallback if needed, and the feasibility test is inclusive at both boundaries with no numerical tolerance.

Returns
-------
beta, betaL, C0, c, feasible : tuple -- unconstrained slope, its upper admissible bound, the positivity constraint at the intercept, the intercept, and the feasibility flag (see docstring).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def projection_slope(lam: float, nu: float, v0: float, theta: float, omega: np.ndarray, x: np.ndarray, U_s: np.ndarray, s: float, t: float, mu: np.ndarray, alpha: float, kappa: np.ndarray, EXZ: float) -> tuple:
    """Form the unconstrained projection slope and its feasibility metrics.

    Parameters
    ----------
    lam : float
        Mean-reversion strength of the variance in the lifted model.
    nu : float
        Volatility-of-variance coefficient.
    v0 : float
        Long-run variance level of the initial curve.
    theta : float
        Long-run mean of the variance.
    omega : (N,) float array
        Lift weights of the N factors.
    x : (N,) float array
        Mean-reversion speeds of the N factors.
    U_s : (N,) float array
        Factor state at the start of the step.
    s : float
        Start time of the step.
    t : float
        End time of the step.
    mu : (N,) float array
        Per-factor conditional mean vector over the step.
    alpha : float
        Conditional mean of the integrated variance (must be non-zero).
    kappa : (N,) float array
        Per-factor cross-moment vector over the step.
    EXZ : float
        Conditional cross-moment of the integrated variance (must be
        non-zero).

    Returns
    -------
    beta : float
        Unconstrained projection slope.
    betaL : float
        Upper admissible slope implied by the positivity requirement.
    C0 : float
        Value of the positivity constraint at the intercept.
    c : float
        Intercept of the projection constraint.
    feasible : bool
        True when the unconstrained slope satisfies every positivity
        condition.

    Raises
    ------
    ValueError
        If alpha is zero, if EXZ is zero, if a scalar argument is not a real
        finite number, or if the array shapes are inconsistent.
    """
    return beta, betaL, C0, c, feasible

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math


def _g0_t6_05(t, omega, x, lam, v0, theta):
    """Initial variance curve evaluated at t."""
    return float(v0 + lam * theta * np.sum(omega / x * (1.0 - np.exp(-x * t))))


def _real_scalar_05(value) -> bool:
    """True when value is a real scalar (not a bool, not an array)."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _oracle_projection_slope(lam: float, nu: float, v0: float, theta: float, omega: np.ndarray, x: np.ndarray, U_s: np.ndarray, s: float, t: float, mu: np.ndarray, alpha: float, kappa: np.ndarray, EXZ: float) -> tuple:
    """Reference implementation of projection_slope."""
    for name, value in (("lam", lam), ("nu", nu), ("v0", v0), ("theta", theta)):
        if not _real_scalar_05(value) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a real finite number")
    for name, value in (("s", s), ("t", t)):
        if not _real_scalar_05(value):
            raise ValueError(f"{name} must be a real number")
    if not _real_scalar_05(alpha) or float(alpha) == 0.0:
        raise ValueError("alpha must be a real number != 0")
    if not _real_scalar_05(EXZ) or float(EXZ) == 0.0:
        raise ValueError("EXZ must be a real number != 0")
    om = np.asarray(omega, dtype=float)
    xs = np.asarray(x, dtype=float)
    U_s = np.asarray(U_s, dtype=float)
    mu = np.asarray(mu, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    for name, arr in (("omega", om), ("x", xs), ("U_s", U_s), ("mu", mu), ("kappa", kappa)):
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array")
    if not (om.shape[0] == xs.shape[0] == U_s.shape[0] == mu.shape[0] == kappa.shape[0]):
        raise ValueError("omega, x, U_s, mu, and kappa must all have the same length")
    if om.shape[0] < 1:
        raise ValueError("omega, x, U_s, mu, and kappa must all have the same length")

    lam = float(lam); nu = float(nu); v0 = float(v0); theta = float(theta)
    s = float(s); t = float(t)
    alpha = float(alpha); EXZ = float(EXZ)

    beta = EXZ / alpha
    betaL = nu * float(np.sum(om)) / float(np.sum(om * (xs * kappa / EXZ + lam)))
    X0 = mu - alpha * (kappa / EXZ)
    c = float(om @ U_s - om @ (xs * X0) + _g0_t6_05(t, om, xs, lam, v0, theta))
    C0 = c - nu * float(np.sum(om)) * alpha / beta
    feasible = bool(beta > 0.0 and beta <= betaL and C0 >= 0.0)
    return beta, betaL, C0, c, feasible

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 4 cases: the pinned benchmark step-0 metrics, a single-factor fixture
    with a feasible slope, a boundary with the slope exactly at its ceiling,
    and two invalid-input edges (a zero conditional mean, a zero
    cross-moment).
    """
    pinned = (
        "import numpy as np\n"
        "import math\n"
        "H = 0.3\n"
        "N = 5\n"
        "rN = 1.0 + 10.0 * N ** (-0.9)\n"
        "n = np.arange(1, N + 1)\n"
        "omega = ((rN ** (0.5 - H) - 1.0) * rN ** ((H - 0.5) * (1.0 + 0.5 * N)) / (math.gamma(H + 0.5) * math.gamma(1.5 - H)) * rN ** ((0.5 - H) * n))\n"
        "x = ((0.5 - H) / (1.5 - H) * (rN ** (1.5 - H) - 1.0) / (rN ** (0.5 - H) - 1.0) * rN ** (n - 1.0 - N / 2.0))\n"
        "lam = 0.25\n"
        "nu = 0.1\n"
        "v0 = 0.02\n"
        "theta = 0.5\n"
        "U_s = np.zeros(N)\n"
        "s = 0.0\n"
        "t = 0.5\n"
        "mu = np.array([-0.0011146368711955363, -0.0010775036228335088, -0.0009662764647468681, -0.00070228899205464, -0.00034297436914833285])\n"
        "alpha = 0.021119426636698066\n"
        "kappa = np.array([0.000431892369694564, 0.00041741917529622267, 0.0003740863796720897, 0.00027138278108762234, 0.00013208160225628565])\n"
        "EXZ = 0.00034639714095345076"
    )
    feasible = (
        "import numpy as np\n"
        "omega = np.array([1.0])\n"
        "x = np.array([1.0])\n"
        "lam = 0.25\n"
        "nu = 0.2\n"
        "v0 = 0.3\n"
        "theta = 0.5\n"
        "U_s = np.array([0.0])\n"
        "s = 0.0\n"
        "t = 0.5\n"
        "mu = np.array([0.0])\n"
        "alpha = 1.0\n"
        "kappa = np.array([0.16])\n"
        "EXZ = 0.16"
    )
    multi = (
        "import numpy as np\n"
        "import math\n"
        "H = 0.3\n"
        "N = 5\n"
        "rN = 1.0 + 10.0 * N ** (-0.9)\n"
        "n = np.arange(1, N + 1)\n"
        "omega = ((rN ** (0.5 - H) - 1.0) * rN ** ((H - 0.5) * (1.0 + 0.5 * N)) / (math.gamma(H + 0.5) * math.gamma(1.5 - H)) * rN ** ((0.5 - H) * n))\n"
        "x = ((0.5 - H) / (1.5 - H) * (rN ** (1.5 - H) - 1.0) / (rN ** (0.5 - H) - 1.0) * rN ** (n - 1.0 - N / 2.0))\n"
        "lam = 0.25\n"
        "nu = 0.1\n"
        "v0 = 0.02\n"
        "theta = 0.5\n"
        "U_s = np.full(5, -0.04974472876518553)\n"
        "s = 0.0\n"
        "t = 0.5\n"
        "mu = np.zeros(5)\n"
        "alpha = 0.16\n"
        "kappa = np.full(5, 0.003)\n"
        "EXZ = 0.0036025836205343053"
    )
    guard = (
        "\n"
        "def _guard(thunk):\n"
        "    try:\n"
        "        thunk()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 2\n"
        "    except Exception:\n"
        "        return 1"
    )

    return [
        {
            # pinned benchmark step 0: the positivity constraint is violated
            "setup": pinned,
            "call": "float(np.sum(projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ)) + float(bool(projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ)[4])))",
            "gold_call": "float(np.sum(_oracle_projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ)) + float(bool(_oracle_projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ)[4])))",
        },
        {
            # multi-factor fixture with omega_bar != 1, straddling the
            # feasibility boundary: the constraint value and the verdict
            # discriminate the summed-weight contraction of the noise term
            "setup": multi,
            "call": "float(np.sum(projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ)) + float(bool(projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ)[4])))",
            "gold_call": "float(np.sum(_oracle_projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ)) + float(bool(_oracle_projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ)[4])))",
        },
        {
            # single-factor fixture with the slope exactly at its ceiling: the
            # inclusive feasibility boundaries both hold
            "setup": feasible,
            "call": "float(np.sum(projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ)) + float(bool(projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ)[4])))",
            "gold_call": "float(np.sum(_oracle_projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ)) + float(bool(_oracle_projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ)[4])))",
        },
        {
            # boundary: a slope above its ceiling is infeasible
            "setup": feasible + "\nEXZ = 0.20",
            "call": "float(np.sum(projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ)) + float(bool(projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ)[4])))",
            "gold_call": "float(np.sum(_oracle_projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ)) + float(bool(_oracle_projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ)[4])))",
        },
        {
            # edge: a zero conditional mean is invalid
            "setup": pinned.replace("alpha = 0.021119426636698066", "alpha = 0.0") + guard,
            "call": "_guard(lambda: projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ))",
            "gold_call": "_guard(lambda: _oracle_projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ))",
        },
        {
            # edge: a zero cross-moment is invalid
            "setup": pinned.replace("EXZ = 0.00034639714095345076", "EXZ = 0.0") + guard,
            "call": "_guard(lambda: projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ))",
            "gold_call": "_guard(lambda: _oracle_projection_slope(lam, nu, v0, theta, omega, x, U_s, s, t, mu, alpha, kappa, EXZ))",
        },
    ]
