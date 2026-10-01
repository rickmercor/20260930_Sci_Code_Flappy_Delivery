"""
Compute the residues of the partial-fraction rational interpolant of the exponential exp(-t z) determined by a given pole set and interpolation-node set, at a single time t.

The interpolant is a sum over j of alpha_j(t) divided by (z - sigma_j), with the poles sigma_j fixed and independent of t. The residues alpha_j(t) are determined by requiring that the interpolant agree with exp(-t z) at every interpolation node. Return the residues as a one-dimensional array of length n, with entry j the residue attached to pole j, in the same column order as the inputs.

This system is to be solved in precision beyond double, using at least fifty significant decimal digits throughout the elimination, with the residues rounded to double only on return. The matrix entries are to be formed in that same extended precision from the given poles and nodes rather than computed in double and promoted. Carry out that arithmetic with Python's standard-library decimal module; third-party multiprecision packages such as mpmath are not installed.

The inputs are the pole array and the node array from the extremal configuration, both one-dimensional and of equal length, and the time t.

The function raises ValueError if the two input arrays are not one-dimensional of equal positive length, if any entry of either array is not finite, if any pole is not strictly negative, if any node is negative, or if t is not finite and strictly positive.

A rational function with prescribed poles that interpolates a given function at prescribed points has residues fixed by a linear system. Evaluating the partial-fraction form at each node and matching the target function produces a matrix whose entries are reciprocals of differences between a node and a pole. Matrices of this Cauchy type have displacement structure, and their singular values are known to decay at a rate governed by the Zolotarev numbers of the same condenser that determined the node and pole placement in the first place. The same condenser therefore governs both the interpolation error and the conditioning of this system, and any configuration of this degree on it gives a severely ill-conditioned system.



The residues themselves carry practical significance beyond determining the interpolant. In the matrix setting the approximation is formed as a linear combination of shifted linear-system solutions weighted by these residues, so their magnitudes propagate directly into the rounding-error behaviour of the final evaluation. Bounds on that error involve the sum of residue magnitudes divided by the corresponding pole magnitudes, which is why residue growth is tracked alongside the approximation error when assessing whether a given degree is usable in practice. Residues grow as the poles cluster, and for shared real poles on a wide interval they can exceed the magnitude of the final result by many orders, so the linear combination that forms the result involves substantial cancellation.

Returns
-------
np.ndarray, a one-dimensional float64 array of length n giving the residue attached to each pole, in the input pole order, obtained from a solve carried out in at least fifty significant decimal digits and rounded to double on return
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interpolation_residues(poles: np.ndarray, nodes: np.ndarray, t: float) -> np.ndarray:
    '''Compute residues of the partial-fraction interpolant of exp(-t z) at given nodes.

    Parameters
    ----------
    poles : np.ndarray
        One-dimensional array of n strictly negative, finite poles.
    nodes : np.ndarray
        One-dimensional array of n nonnegative, finite interpolation nodes.
    t : float
        Time parameter. Must be finite and strictly positive.

    Returns
    -------
    alpha : np.ndarray
        One-dimensional float array of length n; entry j is the residue attached
        to poles[j].

    Raises
    ------
    ValueError
        If the two input arrays are not one-dimensional of equal positive length, if
        any entry of either array is not finite, if any pole is not strictly
        negative, if any node is negative, or if t is not finite and strictly
        positive.
    '''
    return alpha  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from decimal import Decimal, localcontext


def _oracle_interpolation_residues(poles: np.ndarray, nodes: np.ndarray,
                                   t: float) -> np.ndarray:
    """Reference implementation."""
    poles = np.asarray(poles, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    if poles.ndim != 1 or nodes.ndim != 1:
        raise ValueError("poles and nodes must be one-dimensional")
    if poles.size != nodes.size or poles.size < 1:
        raise ValueError("poles and nodes must have equal positive length")
    if not (np.all(np.isfinite(poles)) and np.all(np.isfinite(nodes))):
        raise ValueError("poles and nodes must be finite")
    if not np.all(poles < 0.0):
        raise ValueError("all poles must be strictly negative")
    if not np.all(nodes >= 0.0):
        raise ValueError("all nodes must be nonnegative")
    t = float(t)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("t must be finite and strictly positive")

    n = poles.size
    with localcontext() as ctx:
        ctx.prec = 60
        dt = Decimal(repr(t))
        dnodes = [Decimal(repr(float(x))) for x in nodes]
        dpoles = [Decimal(repr(float(x))) for x in poles]

        # Augmented system built and eliminated entirely in extended precision.
        aug = [[Decimal(1) / (dnodes[i] - dpoles[j]) for j in range(n)]
               + [(-dt * dnodes[i]).exp()] for i in range(n)]

        for k in range(n):
            piv = max(range(k, n), key=lambda r: abs(aug[r][k]))
            if aug[piv][k] == 0:
                raise ValueError("interpolation system is singular")
            aug[k], aug[piv] = aug[piv], aug[k]
            for r in range(k + 1, n):
                fac = aug[r][k] / aug[k][k]
                for cc in range(k, n + 1):
                    aug[r][cc] -= fac * aug[k][cc]

        sol = [Decimal(0)] * n
        for i in range(n - 1, -1, -1):
            acc = aug[i][n] - sum(aug[i][j] * sol[j] for j in range(i + 1, n))
            sol[i] = acc / aug[i][i]

        return np.asarray([float(v) for v in sol], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: production configuration at the evaluation time ---
        {
            "setup": """import numpy as np
from scipy.special import ellipk, ellipj
def _cfg(c, d, n):
    vs = c + np.sqrt(c*(c-d)); rho = 2.0*c - vs
    e1 = (vs-d)/(d-rho); e2 = (vs-c)/(c-rho); mu = 1.0-(e1/e2)**2
    J = ellipk(mu); P = np.empty(n); T = np.empty(n)
    for i in range(1, n+1):
        w = e2*ellipj((2*n-2*i+1)*J/(2*n), mu)[2]
        P[i-1] = (vs+rho*w)/(1.0+w); T[i-1] = (vs-rho*w)/(1.0-w)
    return P, T
poles, nodes = _cfg(-551.5183157669, -10.1118823417, 21)
t = 1.0
""",
            "call": "interpolation_residues(poles, nodes, t)",
            "gold_call": "_oracle_interpolation_residues(poles, nodes, t)",
        },
        # --- normal: production configuration at the left endpoint time ---
        {
            "setup": """import numpy as np
from scipy.special import ellipk, ellipj
def _cfg(c, d, n):
    vs = c + np.sqrt(c*(c-d)); rho = 2.0*c - vs
    e1 = (vs-d)/(d-rho); e2 = (vs-c)/(c-rho); mu = 1.0-(e1/e2)**2
    J = ellipk(mu); P = np.empty(n); T = np.empty(n)
    for i in range(1, n+1):
        w = e2*ellipj((2*n-2*i+1)*J/(2*n), mu)[2]
        P[i-1] = (vs+rho*w)/(1.0+w); T[i-1] = (vs-rho*w)/(1.0-w)
    return P, T
poles, nodes = _cfg(-551.5183157669, -10.1118823417, 21)
t = 0.01
""",
            "call": "interpolation_residues(poles, nodes, t)",
            "gold_call": "_oracle_interpolation_residues(poles, nodes, t)",
        },
        # --- normal: unrefined initial interval ---
        {
            "setup": """import numpy as np
from scipy.special import ellipk, ellipj
def _cfg(c, d, n):
    vs = c + np.sqrt(c*(c-d)); rho = 2.0*c - vs
    e1 = (vs-d)/(d-rho); e2 = (vs-c)/(c-rho); mu = 1.0-(e1/e2)**2
    J = ellipk(mu); P = np.empty(n); T = np.empty(n)
    for i in range(1, n+1):
        w = e2*ellipj((2*n-2*i+1)*J/(2*n), mu)[2]
        P[i-1] = (vs+rho*w)/(1.0+w); T[i-1] = (vs-rho*w)/(1.0-w)
    return P, T
poles, nodes = _cfg(-1484.9242404917, -14.8492424049, 21)
t = 1.0
""",
            "call": "interpolation_residues(poles, nodes, t)",
            "gold_call": "_oracle_interpolation_residues(poles, nodes, t)",
        },
        # --- normal: low degree on a small interval ---
        {
            "setup": """import numpy as np
from scipy.special import ellipk, ellipj
def _cfg(c, d, n):
    vs = c + np.sqrt(c*(c-d)); rho = 2.0*c - vs
    e1 = (vs-d)/(d-rho); e2 = (vs-c)/(c-rho); mu = 1.0-(e1/e2)**2
    J = ellipk(mu); P = np.empty(n); T = np.empty(n)
    for i in range(1, n+1):
        w = e2*ellipj((2*n-2*i+1)*J/(2*n), mu)[2]
        P[i-1] = (vs+rho*w)/(1.0+w); T[i-1] = (vs-rho*w)/(1.0-w)
    return P, T
poles, nodes = _cfg(-2.0, -1.0, 5)
t = 1.0
""",
            "call": "interpolation_residues(poles, nodes, t)",
            "gold_call": "_oracle_interpolation_residues(poles, nodes, t)",
        },
        # --- edge: extended degree on the production interval ---
        {
            "setup": """import numpy as np
from scipy.special import ellipk, ellipj
def _cfg(c, d, n):
    vs = c + np.sqrt(c*(c-d)); rho = 2.0*c - vs
    e1 = (vs-d)/(d-rho); e2 = (vs-c)/(c-rho); mu = 1.0-(e1/e2)**2
    J = ellipk(mu); P = np.empty(n); T = np.empty(n)
    for i in range(1, n+1):
        w = e2*ellipj((2*n-2*i+1)*J/(2*n), mu)[2]
        P[i-1] = (vs+rho*w)/(1.0+w); T[i-1] = (vs-rho*w)/(1.0-w)
    return P, T
poles, nodes = _cfg(-551.5183157669, -10.1118823417, 35)
t = 1.0
""",
            "call": "interpolation_residues(poles, nodes, t)",
            "gold_call": "_oracle_interpolation_residues(poles, nodes, t)",
        },
        # --- boundary: single pole and node ---
        {
            "setup": """import numpy as np
poles = np.array([-1.4142135623730951])
nodes = np.array([1.4142135623730951])
t = 0.5
""",
            "call": "interpolation_residues(poles, nodes, t)",
            "gold_call": "_oracle_interpolation_residues(poles, nodes, t)",
        },
        # --- boundary: a node placed exactly at the origin ---
        {
            "setup": """import numpy as np
poles = np.array([-4.0, -2.0, -1.0])
nodes = np.array([0.0, 3.0, 40.0])
t = 0.25
""",
            "call": "interpolation_residues(poles, nodes, t)",
            "gold_call": "_oracle_interpolation_residues(poles, nodes, t)",
        },
        # --- invalid: a nonnegative pole ---
        {
            "setup": """import numpy as np
poles = np.array([-2.0, 0.0])
nodes = np.array([1.0, 5.0])
t = 1.0
def run_model():
    try:
        interpolation_residues(poles, nodes, t)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_interpolation_residues(poles, nodes, t)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: mismatched lengths ---
        {
            "setup": """import numpy as np
poles = np.array([-2.0, -1.0])
nodes = np.array([1.0, 5.0, 9.0])
t = 1.0
def run_model():
    try:
        interpolation_residues(poles, nodes, t)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_interpolation_residues(poles, nodes, t)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: nonpositive time ---
        {
            "setup": """import numpy as np
poles = np.array([-2.0, -1.0])
nodes = np.array([1.0, 5.0])
t = 0.0
def run_model():
    try:
        interpolation_residues(poles, nodes, t)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_interpolation_residues(poles, nodes, t)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: negative node ---
        {
            "setup": """import numpy as np
poles = np.array([-2.0, -1.0])
nodes = np.array([-1.0, 5.0])
t = 1.0
def run_model():
    try:
        interpolation_residues(poles, nodes, t)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_interpolation_residues(poles, nodes, t)
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
