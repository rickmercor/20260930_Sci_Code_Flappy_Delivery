"""
Advances the lifted-Heston state by one time step: draws the variance increment from the inverse-Gaussian family with the full two-root transform, then updates the factor vector and the variance level.

One-step state update.

Each step of the march turns a standard normal draw and a uniform draw into one realization of the variance increment, then propagates the factor vector and the variance level. The increment is sampled from the inverse-Gaussian family whose mean is the conditional mean of the increment and whose shape follows from the selected slope; the sampling uses the exact two-root construction, in which a candidate root is accepted or replaced by its reciprocal according to the uniform draw, so the resulting sample has the intended inverse-Gaussian law rather than the biased single-root approximation. The realized increment then enters the affine state update: the factor vector is shifted by the scaled increment, the variance level is rebuilt from the updated factors, and the increment is accumulated into the running total that forms the final answer. All quantities are deterministic functions of the inputs, so the same draws always produce the same state.

Returns
-------
Xhat, Zhat, U_next, V_next : tuple -- realized variance increment, standardized increment, updated (N,) factor state, and updated variance level (see docstring).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def simulate_step(alpha: float, beta_tilde: float, mu: np.ndarray, kappa: np.ndarray, EXZ: float, x: np.ndarray, omega: np.ndarray, lam: float, nu: float, U_s: np.ndarray, z_k: float, u_k: float, g0_t: float) -> tuple:
    """Advance the lifted-Heston state by one time step.

    Parameters
    ----------
    alpha : float
        Conditional mean of the variance increment (must be > 0).
    beta_tilde : float
        Slope used for the draw (must be > 0).
    mu : (N,) float array
        Conditional mean vector of the factor state.
    kappa : (N,) float array
        Conditional cross-moment vector of the factor state.
    EXZ : float
        Conditional cross-moment scalar (must be non-zero).
    x : (N,) float array
        Mean-reversion speeds of the N factors.
    omega : (N,) float array
        Lift weights of the N factors.
    lam : float
        Mean-reversion speed of the variance level.
    nu : float
        Volatility-of-variance coefficient.
    U_s : (N,) float array
        Factor state at the start of the step.
    z_k : float
        Standard normal draw for this step.
    u_k : float
        Uniform draw for this step.
    g0_t : float
        Initial variance curve evaluated at the end of the step.

    Returns
    -------
    Xhat : float
        Realized variance increment for this step.
    Zhat : float
        Standardized increment used in the state update.
    U_next : (N,) float array
        Factor state at the end of the step.
    V_next : float
        Variance level at the end of the step.

    Raises
    ------
    ValueError
        If alpha is not > 0, if beta_tilde is not > 0, if EXZ is zero, or if the
        vector inputs are not one-dimensional arrays of equal length.
    """
    return Xhat, Zhat, U_next, V_next

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math


def _real_scalar_07(value) -> bool:
    """True when value is a real scalar (not a bool, not an array)."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _as_vector_07(value, name: str) -> np.ndarray:
    """Coerce value to a one-dimensional float array, or raise ValueError."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or arr.size < 1:
        raise ValueError(f"{name} must be a one-dimensional array with at least one entry")
    return arr


def _oracle_simulate_step(alpha: float, beta_tilde: float, mu: np.ndarray, kappa: np.ndarray, EXZ: float, x: np.ndarray, omega: np.ndarray, lam: float, nu: float, U_s: np.ndarray, z_k: float, u_k: float, g0_t: float) -> tuple:
    """Reference implementation of simulate_step."""
    if not _real_scalar_07(alpha) or not (float(alpha) > 0.0):
        raise ValueError("alpha must be a real number > 0")
    if not _real_scalar_07(beta_tilde) or not (float(beta_tilde) > 0.0):
        raise ValueError("beta_tilde must be a real number > 0")
    if not _real_scalar_07(EXZ) or float(EXZ) == 0.0:
        raise ValueError("EXZ must be a real number != 0")
    mu_v = _as_vector_07(mu, "mu")
    kappa_v = _as_vector_07(kappa, "kappa")
    x_v = _as_vector_07(x, "x")
    omega_v = _as_vector_07(omega, "omega")
    U_s_v = _as_vector_07(U_s, "U_s")
    n = mu_v.size
    if not (kappa_v.size == n and x_v.size == n and omega_v.size == n and U_s_v.size == n):
        raise ValueError("mu, kappa, x, omega and U_s must all have the same length")
    for name, value in (("lam", lam), ("nu", nu), ("z_k", z_k), ("u_k", u_k), ("g0_t", g0_t)):
        if not _real_scalar_07(value):
            raise ValueError(f"{name} must be a real number")
    alpha = float(alpha)
    beta_tilde = float(beta_tilde)
    EXZ = float(EXZ)
    lam = float(lam)
    nu = float(nu)
    z_k = float(z_k)
    u_k = float(u_k)
    g0_t = float(g0_t)
    gamma = (alpha / beta_tilde) ** 2
    V = z_k ** 2
    X1 = (alpha + alpha ** 2 * V / (2.0 * gamma)
          - (alpha / (2.0 * gamma)) * math.sqrt(4.0 * alpha * gamma * V + alpha ** 2 * V ** 2))
    Xhat = X1 if u_k <= alpha / (alpha + X1) else alpha ** 2 / X1
    Zhat = (Xhat - alpha) / beta_tilde
    Xn = mu_v + (kappa_v / EXZ) * (Xhat - alpha)
    U_next = U_s_v - x_v * Xn - lam * Xhat + nu * Zhat
    V_next = float(omega_v @ U_next) + g0_t
    return Xhat, Zhat, U_next, V_next

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 4 cases: the pinned benchmark step, a second benchmark step, a draw
    just above the root-selection threshold, a minimal one-factor step, and
    an invalid-input edge.
    """
    fixture = (
        "import numpy as np\n"
        "x = np.array([0.09693399818571485, 0.32465501918211553, 1.0873468901819494, 3.6417834000130815, 12.197198936570796])\n"
        "omega = np.array([0.13979052887176874, 0.17801924180267542, 0.2267024147327589, 0.2886990435709822, 0.36764997786658365])\n"
        "def _reduce_07(out):\n"
        "    return float(out[0] + out[1] + np.sum(out[2]) + out[3])\n"
    )
    minimal = (
        "import numpy as np\n"
        "x = np.array([1.0])\n"
        "omega = np.array([1.0])\n"
        "mu = np.array([1.0])\n"
        "kappa = np.array([0.5])\n"
        "U_s = np.array([0.0])\n"
        "def _reduce_07(out):\n"
        "    return float(out[0] + out[1] + np.sum(out[2]) + out[3])\n"
    )
    guard_def = (
        "def _guard(thunk):\n"
        "    try:\n"
        "        thunk()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 2\n"
        "    except Exception:\n"
        "        return 1\n"
    )
    return [
        {
            # pinned benchmark step 0: the uniform draw selects the first root
            "setup": fixture + (
                "mu = np.array([-0.0011146368711955363, -0.0010775036228335088, -0.0009662764647468681, -0.00070228899205464, -0.00034297436914833285])\n"
                "kappa = np.array([0.000431892369694564, 0.00041741917529622267, 0.0003740863796720897, 0.00027138278108762234, 0.00013208160225628565])\n"
                "U_s = np.zeros(5)\n"
            ),
            "call": "_reduce_07(simulate_step(0.021119426636698066, 0.020233810047406237, mu, kappa, 0.00034639714095345076, x, omega, 0.25, 0.1, U_s, -1.4238250364546312, 0.248245714629571, 0.06179296633696958))",
            "gold_call": "_reduce_07(_oracle_simulate_step(0.021119426636698066, 0.020233810047406237, mu, kappa, 0.00034639714095345076, x, omega, 0.25, 0.1, U_s, -1.4238250364546312, 0.248245714629571, 0.06179296633696958))",
        },
        {
            # pinned benchmark step 1: the uniform draw selects the reciprocal root
            "setup": fixture + (
                "mu = np.array([-0.012736208248811469, -0.011492139714410538, -0.008281807755447039, -0.003306171070555185, -0.0006116935724147324])\n"
                "kappa = np.array([0.0006773301821425944, 0.0006532902510640252, 0.0005816665179400856, 0.00041441964564438726, 0.00019557049706775515])\n"
                "U_s = np.array([-0.022507611488912515, -0.02124056271574814, -0.017570193244091065, -0.009696576820776088, -0.0012535973437224242])\n"
            ),
            "call": "_reduce_07(simulate_step(0.030468875990134425, 0.021913318651889988, mu, kappa, 0.0005343918279931116, x, omega, 0.25, 0.1, U_s, 1.2637284581291104, 0.9488811518333182, 0.0863481802499916))",
            "gold_call": "_reduce_07(_oracle_simulate_step(0.030468875990134425, 0.021913318651889988, mu, kappa, 0.0005343918279931116, x, omega, 0.25, 0.1, U_s, 1.2637284581291104, 0.9488811518333182, 0.0863481802499916))",
        },
        {
            # boundary: the uniform draw sits just above the root-selection threshold
            "setup": fixture + (
                "mu = np.array([-0.0011146368711955363, -0.0010775036228335088, -0.0009662764647468681, -0.00070228899205464, -0.00034297436914833285])\n"
                "kappa = np.array([0.000431892369694564, 0.00041741917529622267, 0.0003740863796720897, 0.00027138278108762234, 0.00013208160225628565])\n"
                "U_s = np.zeros(5)\n"
            ),
            "call": "_reduce_07(simulate_step(0.021119426636698066, 0.020233810047406237, mu, kappa, 0.00034639714095345076, x, omega, 0.25, 0.1, U_s, -1.4238250364546312, 0.5493185399734597, 0.06179296633696958))",
            "gold_call": "_reduce_07(_oracle_simulate_step(0.021119426636698066, 0.020233810047406237, mu, kappa, 0.00034639714095345076, x, omega, 0.25, 0.1, U_s, -1.4238250364546312, 0.5493185399734597, 0.06179296633696958))",
        },
        {
            # boundary: the minimal one-factor step
            "setup": minimal,
            "call": "_reduce_07(simulate_step(2.0, 1.0, mu, kappa, 1.0, x, omega, 0.1, 0.2, U_s, 1.0, 0.5, 0.0))",
            "gold_call": "_reduce_07(_oracle_simulate_step(2.0, 1.0, mu, kappa, 1.0, x, omega, 0.1, 0.2, U_s, 1.0, 0.5, 0.0))",
        },
        {
            # edge: a non-positive slope is invalid
            "setup": minimal + guard_def,
            "call": "_guard(lambda: simulate_step(2.0, -1.0, mu, kappa, 1.0, x, omega, 0.1, 0.2, U_s, 1.0, 0.5, 0.0))",
            "gold_call": "_guard(lambda: _oracle_simulate_step(2.0, -1.0, mu, kappa, 1.0, x, omega, 0.1, 0.2, U_s, 1.0, 0.5, 0.0))",
        },
    ]
