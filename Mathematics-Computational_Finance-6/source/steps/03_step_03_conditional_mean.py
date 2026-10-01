"""
Computes the conditional mean of the integrated variance over a time step and the per-factor conditional mean vector of the lifted state, by integrating the linear forcing equation and exponentiating the state matrix.

Conditional mean of the integrated variance.

Over a step, the drift of the factor vector is linear in the state with a deterministic forcing term supplied by the initial variance curve, so the conditional expectation of the state increment splits into a homogeneous part driven by the current state and an inhomogeneous part driven purely by the curve. The homogeneous part is the action of the matrix exponential of the state matrix on the current state, while the forcing part is the solution of a linear ordinary differential equation with a known right-hand side. Together they give the per-factor conditional mean; contracting the result with the lift weights and adding the integrated variance curve yields the conditional mean of the integrated variance itself. This scalar is the first moment the projection step matches, and the per-factor vector is needed to advance the state consistently.

Returns
-------
alpha, mu : tuple -- conditional mean of the integrated variance (scalar) and the per-factor conditional mean vector (N,) (see docstring).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def conditional_mean(lam: float, v0: float, theta: float, omega: np.ndarray, x: np.ndarray, A: np.ndarray, s: float, t: float, U_s: np.ndarray, n_grid: int = 4001) -> tuple:
    """Compute the conditional mean of the integrated variance over a step.


    Definitions
    -----------
    g0(u)=v0+lam*theta*sum(omega/x*(1-exp(-x*u))) and
    G0(s,u)=integral_s^u g0(r) dr. Let P(h)=A^{-1}(exp(A*h)-I).
    xi'=A*xi-lam*G0(s,u)*ones, xi(s)=0.
    mu=P(t-s)*U_s+xi(t); alpha=omega@mu+G0(s,t).

    Numerical domain
    ----------------
    Support short positive steps down to 1e-8 and positive factor speeds down
    to 1e-10, including nonzero U_s and clustered speeds. Return the unscaled
    moments. Tests also compare alpha and mu divided by (t-s), or EXZ and
    kappa divided by (t-s)**2, to check the moment densities in these regimes.
    The formulas denote real mathematical quantities; use numerically stable
    equivalent evaluations when differences of exponentials lose precision.
    The forcing equations use classical RK4 on n_grid uniform points. Evaluate
    the homogeneous matrix-integral contribution to numerical precision.

    Parameters
    ----------
    lam : float
        Mean-reversion strength of the variance in the lifted model.
    v0 : float
        Long-run variance level of the initial curve.
    theta : float
        Long-run mean of the variance.
    omega : (N,) float array
        Lift weights of the N factors.
    x : (N,) float array
        Mean-reversion speeds of the N factors.
    A : (N, N) float array
        State matrix of the lifted system.
    s : float
        Start time of the step.
    t : float
        End time of the step (must satisfy t > s).
    U_s : (N,) float array
        Factor state at the start of the step.
    n_grid : int, optional
        Number of uniform grid points used by the fixed-step integrator
        (must be an integer >= 2; default 4001).

    Returns
    -------
    alpha : float
        Conditional mean of the integrated variance over the step.
    mu : (N,) float array
        Per-factor conditional mean vector of the lifted state over the step.

    Raises
    ------
    ValueError
        If t <= s, if n_grid < 2, if lam, v0, or theta is not a real finite
        number, or if the array shapes are inconsistent.
    """
    return alpha, mu

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math


def _oracle_conditional_mean(lam: float, v0: float, theta: float, omega: np.ndarray, x: np.ndarray, A: np.ndarray, s: float, t: float, U_s: np.ndarray, n_grid: int = 4001) -> tuple:
    """Reference implementation of conditional_mean."""
    np = __import__("numpy")
    math = __import__("math")
    expm = __import__("scipy.linalg", fromlist=["expm"]).expm
    exprel = __import__("scipy.special", fromlist=["exprel"]).exprel

    def curve_integral(elapsed):
        # G0(s,s+elapsed), evaluated without subtracting almost equal terms.
        a = -xs * elapsed
        small = abs(a) < 0.05
        phi2 = np.empty_like(a)
        aa = a[small]
        value = np.full_like(aa, 1.0 / math.factorial(12))
        for j in range(11, 1, -1):
            value = 1.0 / math.factorial(j) + aa * value
        phi2[small] = value
        phi2[~small] = (np.expm1(a[~small]) - a[~small]) / a[~small]**2
        initial = v0 + lam * theta * np.sum(om * s * exprel(-xs * s))
        return initial * elapsed + lam * theta * np.sum(om * np.exp(-xs*s) * elapsed**2 * phi2)
    if not isinstance(n_grid, (int, np.integer)) or isinstance(n_grid, bool) or int(n_grid) < 2:
        raise ValueError("n_grid must be an integer >= 2")
    for name, value in (("lam", lam), ("v0", v0), ("theta", theta)):
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a real finite number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a real finite number")
    for name, value in (("s", s), ("t", t)):
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a real number")
    if not (float(t) > float(s)):
        raise ValueError("t must be strictly greater than s")
    om = np.asarray(omega, dtype=float)
    xs = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    U_s = np.asarray(U_s, dtype=float)
    if om.ndim != 1 or xs.ndim != 1 or om.shape[0] != xs.shape[0] or om.shape[0] < 1:
        raise ValueError("omega and x must be one-dimensional arrays of the same length >= 1")
    n = om.shape[0]
    if A.ndim != 2 or A.shape != (n, n):
        raise ValueError("A must be a square matrix of shape (N, N)")
    if U_s.ndim != 1 or U_s.shape[0] != n:
        raise ValueError("U_s must be a one-dimensional array of length N")

    lam = float(lam); v0 = float(v0); theta = float(theta)
    s = float(s); t = float(t); n_grid = int(n_grid)
    one = np.ones(n)
    dt = t - s
    h = dt / (n_grid - 1)
    xi = np.zeros(n)
    for k in range(n_grid - 1):
        uk = s + k * h
        g1 = curve_integral(k * h)
        k1 = A @ xi - lam * g1 * one
        g2 = curve_integral((k + 0.5) * h)
        k2 = A @ (xi + 0.5 * h * k1) - lam * g2 * one
        g3 = curve_integral((k + 0.5) * h)
        k3 = A @ (xi + 0.5 * h * k2) - lam * g3 * one
        g4 = curve_integral((k + 1) * h)
        k4 = A @ (xi + h * k3) - lam * g4 * one
        xi = xi + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    augmented = np.zeros((n + 1, n + 1))
    augmented[:n, :n] = A
    augmented[:n, n] = U_s
    eta = expm(augmented * dt)[:n, n]
    mu = eta + xi
    alpha = float(om @ mu + curve_integral(dt))
    return alpha, mu

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 4 cases: the pinned benchmark configuration, a coarser integration
    grid, a single-factor boundary, and two invalid-input edges (a
    non-advancing step, a too-small grid).
    """
    fixtures = (
        "import numpy as np\n"
        "import math\n"
        "H = 0.3\n"
        "N = 5\n"
        "rN = 1.0 + 10.0 * N ** (-0.9)\n"
        "n = np.arange(1, N + 1)\n"
        "omega = ((rN ** (0.5 - H) - 1.0) * rN ** ((H - 0.5) * (1.0 + 0.5 * N)) / (math.gamma(H + 0.5) * math.gamma(1.5 - H)) * rN ** ((0.5 - H) * n))\n"
        "x = ((0.5 - H) / (1.5 - H) * (rN ** (1.5 - H) - 1.0) / (rN ** (0.5 - H) - 1.0) * rN ** (n - 1.0 - N / 2.0))\n"
        "lam = 0.25\n"
        "v0 = 0.02\n"
        "theta = 0.5\n"
        "A = -lam * np.outer(np.ones(N), omega) - np.diag(x)\n"
        "U_s = np.zeros(N)\n"
        "s = 0.0\n"
        "t = 0.5"
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
    single = (
        "import numpy as np\n"
        "omega = np.array([0.5])\n"
        "x = np.array([2.0])\n"
        "lam = 0.25\n"
        "v0 = 0.02\n"
        "theta = 0.5\n"
        "A = -lam * np.outer(np.ones(1), omega) - np.diag(x)\n"
        "U_s = np.array([0.0])\n"
        "s = 0.0\n"
        "t = 0.5"
    )
    return [
        {'setup': 'import numpy as np\nomega=np.array([0.2,0.3,0.5])\nx=np.array([0.13, 0.7, 3.0])\nlam=0.25\nnu=0.1\nv0=0.02\ntheta=0.5\ns=0.0\nt=1e-08\nU_s=np.array([0.04, -0.01, 0.02])\nA=-lam*np.outer(np.ones(3),omega)-np.diag(x)\ndef flatten(out):\n    return np.concatenate(([out[0]],out[1])) / (t-s)**1\n', 'call': 'flatten(conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 401))', 'gold_call': 'flatten(_oracle_conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 401))'},
        {'setup': 'import numpy as np\nomega=np.array([0.2,0.3,0.5])\nx=np.array([1e-10, 2e-10, 7e-10])\nlam=1e-09\nnu=0.1\nv0=0.02\ntheta=0.5\ns=0.25\nt=0.75\nU_s=np.array([0.04, -0.01, 0.02])\nA=-lam*np.outer(np.ones(3),omega)-np.diag(x)\ndef flatten(out):\n    return np.concatenate(([out[0]],out[1])) / (t-s)**1\n', 'call': 'flatten(conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 401))', 'gold_call': 'flatten(_oracle_conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 401))'},
        {'setup': 'import numpy as np\nomega=np.array([0.2,0.3,0.5])\nx=np.array([0.13, 0.13+1e-12, 3.0])\nlam=0.25\nnu=0.1\nv0=0.02\ntheta=0.5\ns=2.0\nt=2.00001\nU_s=np.array([0.04, -0.01, 0.02])\nA=-lam*np.outer(np.ones(3),omega)-np.diag(x)\ndef flatten(out):\n    return np.concatenate(([out[0]],out[1])) / (t-s)**1\n', 'call': 'flatten(conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 401))', 'gold_call': 'flatten(_oracle_conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 401))'},

        {
            # pinned benchmark configuration on the full integration grid
            "setup": fixtures,
            "call": "float(conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 4001)[0] + np.sum(conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 4001)[1]))",
            "gold_call": "float(_oracle_conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 4001)[0] + np.sum(_oracle_conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 4001)[1]))",
        },
        {
            # coarser integration grid: the conditional mean is grid-insensitive
            "setup": fixtures,
            "call": "float(conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 201)[0] + np.sum(conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 201)[1]))",
            "gold_call": "float(_oracle_conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 201)[0] + np.sum(_oracle_conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 201)[1]))",
        },
        {
            # boundary: a single lifted factor
            "setup": single,
            "call": "float(conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 4001)[0] + np.sum(conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 4001)[1]))",
            "gold_call": "float(_oracle_conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 4001)[0] + np.sum(_oracle_conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 4001)[1]))",
        },
        {
            # edge: a non-advancing step is invalid
            "setup": fixtures + guard,
            "call": "_guard(lambda: conditional_mean(lam, v0, theta, omega, x, A, 0.5, 0.5, U_s, 4001))",
            "gold_call": "_guard(lambda: _oracle_conditional_mean(lam, v0, theta, omega, x, A, 0.5, 0.5, U_s, 4001))",
        },
        {
            # edge: a grid with fewer than two points is invalid
            "setup": fixtures + guard,
            "call": "_guard(lambda: conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 1))",
            "gold_call": "_guard(lambda: _oracle_conditional_mean(lam, v0, theta, omega, x, A, s, t, U_s, 1))",
        },
    ]
