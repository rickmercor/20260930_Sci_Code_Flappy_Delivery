"""
Evaluate the discrete time-uniform interpolation error of the degree-n extremal configuration associated with a candidate pole interval.

For the given interval [c, d], construct the extremal pole and node configuration of degree n as in the previous steps. For each supplied time t, form the partial-fraction interpolant of exp(-t z) determined by that configuration, evaluate it at every supplied sample point z, and take the largest absolute deviation from exp(-t z). Return the largest such deviation over all supplied times, as a single float.

The interpolation systems that determine the residues inside this objective are solved in standard double-precision floating-point arithmetic.

The inputs are the two interval endpoints, the degree, a one-dimensional array of times, and a one-dimensional array of sample points on the nonnegative real axis.

The function raises ValueError if either endpoint is not finite, if the ordering c < d < 0 does not hold, if n is not a positive integer, if either input array is not one-dimensional and non-empty, if any time is not finite and strictly positive, or if any sample point is not finite and nonnegative.

The quantity being measured is a discretization of the continuous time-uniform error, the supremum over the time interval of the supremum over the nonnegative real axis of the deviation between the rational family and the exponential. That continuous quantity bounds the approximation error for the matrix problem uniformly over every symmetric positive semidefinite argument, which is what makes the resulting propagator mesh-independent, but it is not directly computable and must be replaced by a maximum over finitely many samples.

Replacing the continuous supremum by a discrete one is safe here for two reasons. The deviation is uniformly continuous on the nonnegative axis for every time in the interval, so refining the sample set closes the gap between the discrete and continuous maxima at a rate governed by the modulus of continuity. Beyond the largest sample point the deviation is controlled by the decay of the exponential together with a bound involving the residue magnitudes divided by that largest sample point, so extending the sample range also closes the remaining gap. Both corrections vanish in the limit of dense and far-reaching samples, which justifies using the discrete quantity as the optimization criterion.

Because this objective is evaluated repeatedly inside a search over the interval endpoints, and because the poles and nodes must be regenerated from scratch at every candidate interval, its cost per evaluation determines whether the refinement is practical at all. The accuracy demanded of the objective is only what is needed to locate the minimizer, which is far less than the accuracy demanded of the final approximation.

Returns
-------
float, the largest absolute deviation of the interpolant from exp(-t z) over all supplied times and sample points, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def discrete_uniform_error(c: float, d: float, n: int, times: np.ndarray,
                           z_samples: np.ndarray) -> float:
    '''Discrete time-uniform interpolation error of the degree-n configuration on [c,d].

    Parameters
    ----------
    c : float
        Left endpoint of the candidate pole interval. Must be finite with c < d < 0.
    d : float
        Right endpoint of the candidate pole interval. Must be finite with c < d < 0.
    n : int
        Degree of the configuration. Must be a positive integer.
    times : np.ndarray
        One-dimensional non-empty array of finite, strictly positive times.
    z_samples : np.ndarray
        One-dimensional non-empty array of finite, nonnegative sample points.

    Returns
    -------
    err : float
        The largest absolute deviation of the interpolant from exp(-t z) over all
        supplied times and sample points, as a native Python float.

    Raises
    ------
    ValueError
        If either endpoint is not finite, if the ordering c < d < 0 does not hold,
        if n is not a positive integer, if either input array is not one-dimensional
        and non-empty, if any time is not finite and strictly positive, or if any
        sample point is not finite and nonnegative.
    '''
    return err  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import ellipk, ellipj


def _h_config(c, d, n):
    """Extremal poles and interpolation nodes of degree n for [c,d] u [0,inf)."""
    varsigma = c + np.sqrt(c * (c - d))
    varrho = 2.0 * c - varsigma
    eta1 = (varsigma - d) / (d - varrho)
    eta2 = (varsigma - c) / (c - varrho)
    mu = 1.0 - (eta1 / eta2) ** 2
    if not (mu < 1.0):
        raise ValueError("reduced modulus is not strictly below one in working precision")
    bigj = ellipk(mu)
    poles = np.empty(n, dtype=float)
    nodes = np.empty(n, dtype=float)
    for i in range(1, n + 1):
        w = eta2 * ellipj((2 * n - 2 * i + 1) * bigj / (2 * n), mu)[2]
        poles[i - 1] = (varsigma + varrho * w) / (1.0 + w)
        nodes[i - 1] = (varsigma - varrho * w) / (1.0 - w)
    return poles, nodes


def _oracle_discrete_uniform_error(c: float, d: float, n: int, times: np.ndarray,
                                   z_samples: np.ndarray) -> float:
    """Reference implementation."""
    c = float(c)
    d = float(d)
    if not (np.isfinite(c) and np.isfinite(d)):
        raise ValueError("c and d must be finite")
    if not (c < d < 0.0):
        raise ValueError("endpoints must satisfy c < d < 0")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)

    times = np.asarray(times, dtype=float)
    z_samples = np.asarray(z_samples, dtype=float)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("times must be a one-dimensional non-empty array")
    if z_samples.ndim != 1 or z_samples.size < 1:
        raise ValueError("z_samples must be a one-dimensional non-empty array")
    if not np.all(np.isfinite(times)) or not np.all(times > 0.0):
        raise ValueError("times must be finite and strictly positive")
    if not np.all(np.isfinite(z_samples)) or not np.all(z_samples >= 0.0):
        raise ValueError("z_samples must be finite and nonnegative")

    poles, nodes = _h_config(c, d, n)
    cauchy = 1.0 / (nodes[:, None] - poles[None, :])

    worst = 0.0
    for t in times:
        alpha = np.linalg.solve(cauchy, np.exp(-t * nodes))
        approx = (alpha[None, :] / (z_samples[:, None] - poles[None, :])).sum(axis=1)
        dev = float(np.abs(approx - np.exp(-t * z_samples)).max())
        if dev > worst:
            worst = dev
    return float(worst)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: refined production interval on the production grids ---
        {
            "setup": """import numpy as np
c = -551.5183157669
d = -10.1118823417
n = 21
times = np.logspace(-2.0, 0.0, 40)
z_samples = np.concatenate([[0.0], np.logspace(-6.0, 6.0, 3000)])
""",
            "call": "discrete_uniform_error(c, d, n, times, z_samples)",
            "gold_call": "_oracle_discrete_uniform_error(c, d, n, times, z_samples)",
            "tol": 1e-5,
        },
        # --- normal: unrefined initial interval on the production grids ---
        {
            "setup": """import numpy as np
c = -1484.9242404917
d = -14.8492424049
n = 21
times = np.logspace(-2.0, 0.0, 40)
z_samples = np.concatenate([[0.0], np.logspace(-6.0, 6.0, 3000)])
""",
            "call": "discrete_uniform_error(c, d, n, times, z_samples)",
            "gold_call": "_oracle_discrete_uniform_error(c, d, n, times, z_samples)",
            "tol": 1e-5,
        },
        # --- normal: a better-conditioned competing interval that fails the production bound ---
        {
            "setup": """import numpy as np
c = -800.0
d = -10.0
n = 21
times = np.logspace(-2.0, 0.0, 40)
z_samples = np.concatenate([[0.0], np.logspace(-6.0, 6.0, 3000)])
""",
            "call": "discrete_uniform_error(c, d, n, times, z_samples)",
            "gold_call": "_oracle_discrete_uniform_error(c, d, n, times, z_samples)",
            "tol": 1e-7,
        },
        # --- normal: coarse sample grids on the refined interval ---
        {
            "setup": """import numpy as np
c = -551.5183157669
d = -10.1118823417
n = 21
times = np.logspace(-2.0, 0.0, 5)
z_samples = np.concatenate([[0.0], np.logspace(-4.0, 4.0, 200)])
""",
            "call": "discrete_uniform_error(c, d, n, times, z_samples)",
            "gold_call": "_oracle_discrete_uniform_error(c, d, n, times, z_samples)",
            "tol": 1e-5,
        },
        # --- normal: low degree, error of order one ---
        {
            "setup": """import numpy as np
c = -2.0
d = -1.0
n = 5
times = np.logspace(-2.0, 0.0, 40)
z_samples = np.concatenate([[0.0], np.logspace(-6.0, 6.0, 3000)])
""",
            "call": "discrete_uniform_error(c, d, n, times, z_samples)",
            "gold_call": "_oracle_discrete_uniform_error(c, d, n, times, z_samples)",
        },
        # --- boundary: degree one ---
        {
            "setup": """import numpy as np
c = -2.0
d = -1.0
n = 1
times = np.logspace(-2.0, 0.0, 40)
z_samples = np.concatenate([[0.0], np.logspace(-6.0, 6.0, 3000)])
""",
            "call": "discrete_uniform_error(c, d, n, times, z_samples)",
            "gold_call": "_oracle_discrete_uniform_error(c, d, n, times, z_samples)",
        },
        # --- boundary: single time, two samples including the origin ---
        {
            "setup": """import numpy as np
c = -2.0
d = -1.0
n = 3
times = np.array([1.0])
z_samples = np.array([0.0, 1.0])
""",
            "call": "discrete_uniform_error(c, d, n, times, z_samples)",
            "gold_call": "_oracle_discrete_uniform_error(c, d, n, times, z_samples)",
        },
        # --- invalid: a nonpositive time ---
        {
            "setup": """import numpy as np
c = -551.5183157669
d = -10.1118823417
n = 21
times = np.array([0.0, 1.0])
z_samples = np.array([0.0, 1.0])
def run_model():
    try:
        discrete_uniform_error(c, d, n, times, z_samples)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_discrete_uniform_error(c, d, n, times, z_samples)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: a negative sample point ---
        {
            "setup": """import numpy as np
c = -551.5183157669
d = -10.1118823417
n = 21
times = np.array([1.0])
z_samples = np.array([-1.0, 1.0])
def run_model():
    try:
        discrete_uniform_error(c, d, n, times, z_samples)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_discrete_uniform_error(c, d, n, times, z_samples)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: empty sample array ---
        {
            "setup": """import numpy as np
c = -2.0
d = -1.0
n = 4
times = np.array([1.0])
z_samples = np.array([])
def run_model():
    try:
        discrete_uniform_error(c, d, n, times, z_samples)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_discrete_uniform_error(c, d, n, times, z_samples)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: interval ordering violated ---
        {
            "setup": """import numpy as np
c = -1.0
d = -8.0
n = 4
times = np.array([1.0])
z_samples = np.array([0.0, 1.0])
def run_model():
    try:
        discrete_uniform_error(c, d, n, times, z_samples)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_discrete_uniform_error(c, d, n, times, z_samples)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
