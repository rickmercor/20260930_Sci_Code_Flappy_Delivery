"""
Computes the conditional cross-moment between the integrated variance and the driving noise over a time step, together with the per-factor cross-moment vector, from the defining covariance integral.

Conditional cross-moment of the integrated variance with the noise.

The projection slope that matches the conditional first two moments of the variance increment is the ratio of a cross-moment between the integrated variance and the driving noise to the conditional mean. That cross-moment is an integral over the step of the state transition operator applied to a rank-one weight matrix, together with a forcing term built from the conditional mean path. In the eigenbasis of the state matrix the transition kernel diagonalizes, so the covariance integral collapses to an elementwise exponential expression whose off-diagonal entries carry the difference of the two decay rates and whose diagonal entries carry the step length; the forcing contribution is a companion vector obtained from the same linear system that produces the conditional mean. Contracting the sum with the lift weights gives the scalar cross-moment, and the per-factor vector is returned for the state update. The defining integral is the authoritative definition of this quantity.

Returns
-------
EXZ, kappa : tuple -- conditional cross-moment of the integrated variance (scalar) and the per-factor cross-moment vector (N,) whose weighted sum equals EXZ (see docstring).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def cross_moment(lam: float, nu: float, v0: float, theta: float, omega: np.ndarray, x: np.ndarray, A: np.ndarray, s: float, t: float, U_s: np.ndarray, n_grid: int = 4001) -> tuple:
    """Compute the conditional cross-moment of the integrated variance.


    Definitions
    -----------
    g0(u)=v0+lam*theta*sum(omega/x*(1-exp(-x*u))) and
    G0(s,u)=integral_s^u g0(r) dr. Let P(h)=A^{-1}(exp(A*h)-I).
    xi'=A*xi-lam*G0(s,u)*ones, xi(s)=0.
    I1=integral_s^t exp(A*(t-u))*outer(ones,omega)*exp(A*(u-s)) du.
    psi'=A*psi+nu*(omega@xi+G0(s,u))*ones, psi(s)=0.
    kappa=nu*(I1-P(t-s)*outer(ones,omega))*A^{-1}*U_s+psi(t);
    EXZ=omega@kappa. Integrate xi and psi jointly with RK4.

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
    EXZ : float
        Conditional cross-moment of the integrated variance with the noise
        over the step.
    kappa : (N,) float array
        Per-factor cross-moment vector whose weighted sum equals EXZ.

    Raises
    ------
    ValueError
        If t <= s, if n_grid < 2, if a scalar argument is not a real finite
        number, if the array shapes are inconsistent, or if the resulting
        cross-moment is not finite and non-zero.
    """
    return EXZ, kappa

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math


def _oracle_cross_moment(lam: float, nu: float, v0: float, theta: float, omega: np.ndarray, x: np.ndarray, A: np.ndarray, s: float, t: float, U_s: np.ndarray, n_grid: int = 4001) -> tuple:
    """Reference implementation of cross_moment."""
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
    for name, value in (("lam", lam), ("nu", nu), ("v0", v0), ("theta", theta)):
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

    lam = float(lam); nu = float(nu); v0 = float(v0); theta = float(theta)
    s = float(s); t = float(t); n_grid = int(n_grid)
    one = np.ones(n)
    dt = t - s
    h = dt / (n_grid - 1)

    # joint RK4 for the conditional-mean forcing (xi) and the companion (psi)
    def _rhs_t6_04(uu, yy):
        xi, psi = yy[:n], yy[n:]
        g = curve_integral(uu)
        return np.concatenate([A @ xi - lam * g * one,
                               A @ psi + nu * (one * (om @ xi) + g) * one])

    y = np.zeros(2 * n)
    for k in range(n_grid - 1):
        uk = k * h
        k1 = _rhs_t6_04(uk, y)
        k2 = _rhs_t6_04(uk + 0.5 * h, y + 0.5 * h * k1)
        k3 = _rhs_t6_04(uk + 0.5 * h, y + 0.5 * h * k2)
        k4 = _rhs_t6_04(uk + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    xi, psi = y[:n], y[n:]

    # y'=Ay+U_s, chi'=A chi+nu*1*omega*y integrates the
    # homogeneous covariance contribution from its defining integral.
    augmented = np.zeros((2*n + 1, 2*n + 1))
    augmented[:n, :n] = A
    augmented[:n, -1] = U_s
    augmented[n:2*n, :n] = nu * np.outer(one, om)
    augmented[n:2*n, n:2*n] = A
    chi = expm(augmented * dt)[n:2*n, -1]
    kappa = chi + psi
    EXZ = float(om @ kappa)
    if not math.isfinite(EXZ) or abs(EXZ) < 1e-300:
        raise ValueError("the conditional cross-moment must be finite and non-zero")
    return EXZ, kappa

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 4 cases: the pinned benchmark configuration, a coarser integration
    grid, single-factor and near-degenerate spectral boundaries, and two
    invalid-input edges (a non-advancing step, a too-small grid).
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
        "nu = 0.1\n"
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
        "nu = 0.1\n"
        "v0 = 0.02\n"
        "theta = 0.5\n"
        "A = -lam * np.outer(np.ones(1), omega) - np.diag(x)\n"
        "U_s = np.array([0.0])\n"
        "s = 0.0\n"
        "t = 0.5"
    )
    near_degenerate = (
        "import numpy as np\n"
        "omega = np.array([0.5, 0.5])\n"
        "x = np.array([1.0, 1.0 + 1e-14])\n"
        "lam = 0.25\n"
        "nu = 0.1\n"
        "v0 = 0.02\n"
        "theta = 0.5\n"
        "A = np.diag(np.array([-1.0, -1.0 - 1e-14]))\n"
        "U_s = np.array([0.02, -0.01])\n"
        "s = 0.0\n"
        "t = 0.5"
    )
    return [
        {'setup': 'import numpy as np\nomega=np.array([0.2,0.3,0.5])\nx=np.array([0.13, 0.7, 3.0])\nlam=0.25\nnu=0.1\nv0=0.02\ntheta=0.5\ns=0.0\nt=1e-08\nU_s=np.array([0.04, -0.01, 0.02])\nA=-lam*np.outer(np.ones(3),omega)-np.diag(x)\ndef flatten(out):\n    return np.concatenate(([out[0]],out[1])) / (t-s)**2\n', 'call': 'flatten(cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 401))', 'gold_call': 'flatten(_oracle_cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 401))'},
        {'setup': 'import numpy as np\nomega=np.array([0.2,0.3,0.5])\nx=np.array([1e-10, 2e-10, 7e-10])\nlam=1e-09\nnu=0.1\nv0=0.02\ntheta=0.5\ns=0.25\nt=0.75\nU_s=np.array([0.04, -0.01, 0.02])\nA=-lam*np.outer(np.ones(3),omega)-np.diag(x)\ndef flatten(out):\n    return np.concatenate(([out[0]],out[1])) / (t-s)**2\n', 'call': 'flatten(cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 401))', 'gold_call': 'flatten(_oracle_cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 401))'},
        {'setup': 'import numpy as np\nomega=np.array([0.2,0.3,0.5])\nx=np.array([0.13, 0.13+1e-12, 3.0])\nlam=0.25\nnu=0.1\nv0=0.02\ntheta=0.5\ns=2.0\nt=2.00001\nU_s=np.array([0.04, -0.01, 0.02])\nA=-lam*np.outer(np.ones(3),omega)-np.diag(x)\ndef flatten(out):\n    return np.concatenate(([out[0]],out[1])) / (t-s)**2\n', 'call': 'flatten(cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 401))', 'gold_call': 'flatten(_oracle_cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 401))'},

        {
            # pinned benchmark configuration on the full integration grid
            "setup": fixtures,
            "call": "float(cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 4001)[0] + np.sum(cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 4001)[1]))",
            "gold_call": "float(_oracle_cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 4001)[0] + np.sum(_oracle_cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 4001)[1]))",
        },
        {
            # coarser integration grid: the cross-moment is grid-insensitive
            "setup": fixtures,
            "call": "float(cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 201)[0] + np.sum(cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 201)[1]))",
            "gold_call": "float(_oracle_cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 201)[0] + np.sum(_oracle_cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 201)[1]))",
        },
        {
            # boundary: a single lifted factor
            "setup": single,
            "call": "float(cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 4001)[0] + np.sum(cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 4001)[1]))",
            "gold_call": "float(_oracle_cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 4001)[0] + np.sum(_oracle_cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 4001)[1]))",
        },
        {
            # boundary: nearly repeated eigenvalues use a stable divided difference
            "setup": near_degenerate,
            "call": "float(cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 401)[0] + np.sum(cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 401)[1]))",
            "gold_call": "float(_oracle_cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 401)[0] + np.sum(_oracle_cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 401)[1]))",
        },
        {
            # edge: a non-advancing step is invalid
            "setup": fixtures + guard,
            "call": "_guard(lambda: cross_moment(lam, nu, v0, theta, omega, x, A, 0.5, 0.5, U_s, 4001))",
            "gold_call": "_guard(lambda: _oracle_cross_moment(lam, nu, v0, theta, omega, x, A, 0.5, 0.5, U_s, 4001))",
        },
        {
            # edge: a grid with fewer than two points is invalid
            "setup": fixtures + guard,
            "call": "_guard(lambda: cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 1))",
            "gold_call": "_guard(lambda: _oracle_cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U_s, 1))",
        },
    ]
