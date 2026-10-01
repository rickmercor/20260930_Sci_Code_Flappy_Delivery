"""
Construct the extremal pole and interpolation-node configuration of degree n from the reduced condenser parameters.



Given the five parameters produced by the condenser reduction and a degree n of at least one, solve the Zolotarev problem on the reduced symmetric condenser using Jacobi elliptic functions with the reduced parameter mu (the squared modulus, as taken by scipy.special.ellipj), and transport the resulting configuration back to the original condenser through the inverse of the Mobius map those parameters define. The n poles lie in the original pole interval on the negative real axis and the n interpolation nodes lie on the nonnegative real axis; the two sets are recovered from the same symmetric-condenser quantity by two different inverse maps.



Return a two-row array: row 0 holds the poles and row 1 the interpolation nodes. The columns are ordered so that the pole entries are strictly decreasing, and column i pairs the i-th pole with the i-th node.



The function raises ValueError if the parameter array is not one-dimensional of length 5, if it contains a non-finite entry, if the parameter entry mu is not at least zero and strictly below one, if the second symmetric-condenser endpoint is zero, or if n is not a positive integer.

The third Zolotarev problem on a symmetric condenser has a classical closed-form solution: the extremal rational function of degree n has its zeros and poles at points obtained by dividing the period of an elliptic integral into equal parts and evaluating a Jacobi elliptic function at the resulting abscissae. The division is offset by a half step, so the abscissae interlace the endpoints of the period rather than landing on them, which is what makes the extremal function equioscillate on both plates.



In the application to time-uniform exponential approximation, this configuration is used not for its own sake but to control the growth of the rational nodal function whose zeros are the interpolation nodes and whose poles are the poles of the approximant. Bounding the ratio of the maximum modulus of that nodal function on the nonnegative axis to its minimum modulus on the pole interval bounds the interpolation error uniformly in the time parameter, because the time dependence enters the Hermite integral error formula only through the interpolated function and not through the nodal function. This is why a single set of shared poles can serve an entire time interval, and why the poles must be placed by an extremal criterion rather than by matching any particular time.



The reduced parameter mu approaches unity as the pole interval widens. Complete and incomplete elliptic integrals grow logarithmically in that limit and generic numerical quadrature loses accuracy there, so evaluation of the elliptic quantities requires algorithms that remain accurate for parameters close to one.

Returns
-------
np.ndarray, a float64 array of shape (2, n) whose first row holds the poles in strictly decreasing order and whose second row holds the paired interpolation nodes
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def zolotarev_poles_nodes(params: np.ndarray, n: int) -> np.ndarray:
    '''Construct extremal poles and interpolation nodes from reduced condenser parameters.

    Parameters
    ----------
    params : np.ndarray
        One-dimensional array of length 5 holding, in order, the two Möbius
        parameters varsigma and varrho, the two symmetric-condenser endpoints
        eta1 and eta2, and the elliptic parameter mu.
    n : int
        Degree of the configuration. Must be a positive integer.

    Returns
    -------
    config : np.ndarray
        Float array of shape (2, n). Row 0 holds the poles in strictly decreasing
        order; row 1 holds the paired interpolation nodes.

    Raises
    ------
    ValueError
        If the parameter array is not one-dimensional of length 5, if it contains a
        non-finite entry, if the parameter entry mu is not at least zero and
        strictly below one, if the second symmetric-condenser endpoint is zero, or
        if n is not a positive integer.
    '''
    return config  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import ellipk, ellipj


def _oracle_zolotarev_poles_nodes(params: np.ndarray, n: int) -> np.ndarray:
    """Reference implementation."""
    params = np.asarray(params, dtype=float)
    if params.ndim != 1 or params.size != 5:
        raise ValueError("params must be a one-dimensional array of length 5")
    if not np.all(np.isfinite(params)):
        raise ValueError("params must be finite")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)

    varsigma, varrho, eta1, eta2, mu = (float(params[0]), float(params[1]),
                                        float(params[2]), float(params[3]),
                                        float(params[4]))
    if not (0.0 <= mu < 1.0):
        raise ValueError("modulus must lie in [0, 1)")
    if eta2 == 0.0:
        raise ValueError("second symmetric-condenser endpoint must be nonzero")

    bigj = ellipk(mu)
    poles = np.empty(n, dtype=float)
    nodes = np.empty(n, dtype=float)
    for i in range(1, n + 1):
        v = (2 * n - 2 * i + 1) * bigj / (2 * n)
        w = eta2 * ellipj(v, mu)[2]
        poles[i - 1] = (varsigma + varrho * w) / (1.0 + w)
        nodes[i - 1] = (varsigma - varrho * w) / (1.0 - w)

    return np.vstack([poles, nodes])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: refined production interval, production degree ---
        {
            "setup": """import numpy as np
def _cp(c, d):
    vs = c + np.sqrt(c*(c-d)); rho = 2.0*c - vs
    e1 = (vs-d)/(d-rho); e2 = (vs-c)/(c-rho)
    return np.array([vs, rho, e1, e2, 1.0-(e1/e2)**2])
params = _cp(-551.5183157669, -10.1118823417)
n = 21
""",
            "call": "zolotarev_poles_nodes(params, n)",
            "gold_call": "_oracle_zolotarev_poles_nodes(params, n)",
        },
        # --- normal: unrefined initial interval, production degree ---
        {
            "setup": """import numpy as np
def _cp(c, d):
    vs = c + np.sqrt(c*(c-d)); rho = 2.0*c - vs
    e1 = (vs-d)/(d-rho); e2 = (vs-c)/(c-rho)
    return np.array([vs, rho, e1, e2, 1.0-(e1/e2)**2])
params = _cp(-1484.9242404917, -14.8492424049)
n = 21
""",
            "call": "zolotarev_poles_nodes(params, n)",
            "gold_call": "_oracle_zolotarev_poles_nodes(params, n)",
        },
        # --- normal: small interval and low degree ---
        {
            "setup": """import numpy as np
def _cp(c, d):
    vs = c + np.sqrt(c*(c-d)); rho = 2.0*c - vs
    e1 = (vs-d)/(d-rho); e2 = (vs-c)/(c-rho)
    return np.array([vs, rho, e1, e2, 1.0-(e1/e2)**2])
params = _cp(-2.0, -1.0)
n = 5
""",
            "call": "zolotarev_poles_nodes(params, n)",
            "gold_call": "_oracle_zolotarev_poles_nodes(params, n)",
        },
        # --- boundary: degree one ---
        {
            "setup": """import numpy as np
def _cp(c, d):
    vs = c + np.sqrt(c*(c-d)); rho = 2.0*c - vs
    e1 = (vs-d)/(d-rho); e2 = (vs-c)/(c-rho)
    return np.array([vs, rho, e1, e2, 1.0-(e1/e2)**2])
params = _cp(-2.0, -1.0)
n = 1
""",
            "call": "zolotarev_poles_nodes(params, n)",
            "gold_call": "_oracle_zolotarev_poles_nodes(params, n)",
        },
        # --- boundary: very narrow interval, modulus far from one ---
        {
            "setup": """import numpy as np
def _cp(c, d):
    vs = c + np.sqrt(c*(c-d)); rho = 2.0*c - vs
    e1 = (vs-d)/(d-rho); e2 = (vs-c)/(c-rho)
    return np.array([vs, rho, e1, e2, 1.0-(e1/e2)**2])
params = _cp(-1.0000001, -1.0)
n = 4
""",
            "call": "zolotarev_poles_nodes(params, n)",
            "gold_call": "_oracle_zolotarev_poles_nodes(params, n)",
            "tol": 1e-7,
        },
        # --- edge: interval far from the origin ---
        {
            "setup": """import numpy as np
def _cp(c, d):
    vs = c + np.sqrt(c*(c-d)); rho = 2.0*c - vs
    e1 = (vs-d)/(d-rho); e2 = (vs-c)/(c-rho)
    return np.array([vs, rho, e1, e2, 1.0-(e1/e2)**2])
params = _cp(-1.0e8, -9.9e7)
n = 3
""",
            "call": "zolotarev_poles_nodes(params, n)",
            "gold_call": "_oracle_zolotarev_poles_nodes(params, n)",
        },
        # --- edge: higher degree on the production interval ---
        {
            "setup": """import numpy as np
def _cp(c, d):
    vs = c + np.sqrt(c*(c-d)); rho = 2.0*c - vs
    e1 = (vs-d)/(d-rho); e2 = (vs-c)/(c-rho)
    return np.array([vs, rho, e1, e2, 1.0-(e1/e2)**2])
params = _cp(-551.5183157669, -10.1118823417)
n = 35
""",
            "call": "zolotarev_poles_nodes(params, n)",
            "gold_call": "_oracle_zolotarev_poles_nodes(params, n)",
        },
        # --- structural: poles inside the interval, nodes nonnegative, pairing ordered ---
        {
            "setup": """import numpy as np
def _cp(c, d):
    vs = c + np.sqrt(c*(c-d)); rho = 2.0*c - vs
    e1 = (vs-d)/(d-rho); e2 = (vs-c)/(c-rho)
    return np.array([vs, rho, e1, e2, 1.0-(e1/e2)**2])
c_lo = -551.5183157669
d_hi = -10.1118823417
params = _cp(c_lo, d_hi)
n = 21
def probe(fn):
    r = fn(params, n)
    p, t = r[0], r[1]
    return [int(np.all((p >= c_lo) & (p <= d_hi))),
            int(np.all(t >= 0.0)),
            int(np.all(np.diff(p) < 0.0)),
            int(np.all(np.diff(t) > 0.0)),
            int(r.shape[0]), int(r.shape[1])]
""",
            "call": "probe(zolotarev_poles_nodes)",
            "gold_call": "probe(_oracle_zolotarev_poles_nodes)",
        },
        # --- invalid: modulus saturated at one ---
        {
            "setup": """import numpy as np
params = np.array([-0.0005, -1999999.9995, 2.5e-10, 1.0, 1.0])
n = 6
def run_model():
    try:
        zolotarev_poles_nodes(params, n)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_zolotarev_poles_nodes(params, n)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: modulus outside [0, 1) ---
        {
            "setup": """import numpy as np
params = np.array([-0.5857864376, -3.4142135624, 0.1715728753, 1.0, -0.25])
n = 4
def run_model():
    try:
        zolotarev_poles_nodes(params, n)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_zolotarev_poles_nodes(params, n)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: parameter array of the wrong length ---
        {
            "setup": """import numpy as np
params = np.array([-0.5857864376, -3.4142135624, 0.1715728753, 1.0])
n = 4
def run_model():
    try:
        zolotarev_poles_nodes(params, n)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_zolotarev_poles_nodes(params, n)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-positive degree ---
        {
            "setup": """import numpy as np
def _cp(c, d):
    vs = c + np.sqrt(c*(c-d)); rho = 2.0*c - vs
    e1 = (vs-d)/(d-rho); e2 = (vs-c)/(c-rho)
    return np.array([vs, rho, e1, e2, 1.0-(e1/e2)**2])
params = _cp(-2.0, -1.0)
n = 0
def run_model():
    try:
        zolotarev_poles_nodes(params, n)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_zolotarev_poles_nodes(params, n)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-finite parameter entry ---
        {
            "setup": """import numpy as np
params = np.array([-0.5857864376, -3.4142135624, 0.1715728753, 1.0, np.nan])
n = 4
def run_model():
    try:
        zolotarev_poles_nodes(params, n)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_zolotarev_poles_nodes(params, n)
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
