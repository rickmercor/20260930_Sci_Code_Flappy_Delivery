"""
Return the nodes and weights of the N-point Gauss-Laguerre rule associated with a scaled exponential weight. The rule is the generalized Gauss-Laguerre rule with zero algebraic exponent, and it is obtained from the standard rule by a single change of scale that acts on nodes and weights alike. Nodes come back in increasing order so that callers can index the smallest and largest without sorting.

Gauss rules are built from the three-term recurrence of the orthogonal polynomials of their weight, and the nodes are the eigenvalues of the symmetric tridiagonal matrix that recurrence defines, with the weights read off the first components of the eigenvectors. For the Laguerre weight the recurrence coefficients are integers, so the whole construction is exact input to an eigenvalue solver and needs no root finding.

Changing the weight from exp(-u) to exp(-sigma u) is a substitution rather than a new family. The nodes contract by the scale and the weights contract by it too, because the substitution carries a Jacobian. That second contraction is the part that is easy to omit: a rule with rescaled nodes and unrescaled weights integrates the right function against the wrong measure and is off by exactly the scale factor. The visible symptom is that the weights no longer sum to the integral of the weight function, which is the reciprocal of the scale rather than one.

Raising the scale pulls the nodes towards the origin and lowers the reach of the rule; lowering it spreads them out. Neither is better in the abstract, which is why the scale is matched to the integrand rather than fixed.

Returns
-------
np.ndarray of shape (2*N,) packed as nodes then weights, nodes increasing. The weights sum to 1/sigma, the first moment is 1/sigma2 and the second is 2/sigma3. A one-point rule at unit scale is the single node 1 with weight 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def scaled_laguerre_rule(N, sigma):
    """Nodes and weights of the scaled Gauss-Laguerre quadrature rule.

    The rule is exact for polynomials of degree up to 2N-1 against the weight
    exp(-sigma*u) on (0, inf); it is the generalized Gauss-Laguerre rule with
    zero algebraic exponent.  Its nodes and weights follow from those of the
    standard rule, which corresponds to sigma = 1.  Increasing sigma
    concentrates the nodes near the origin, decreasing it spreads them
    towards larger u.

    Args:
        N (int): number of quadrature points, at least 1.
        sigma (float): positive scaling factor of the exponential weight.

    Expected return:
        np.ndarray of shape (2*N,) packed as
        [u_1, ..., u_N, w_1, ..., w_N], nodes in increasing order.
        Raises:
        ValueError: if N < 1 or if sigma <= 0.
    """
    return np.zeros(2 * int(N))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_scaled_laguerre_rule(N, sigma):
    def _standard(N):
        k = np.arange(N, dtype=float)
        diag = 2.0 * k + 1.0
        off = -(k[1:])
        J = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        x, V = np.linalg.eigh(J)
        w = V[0, :] ** 2
        order = np.argsort(x)
        return x[order], w[order]

    N = int(N)
    if N < 1:
        raise ValueError("N must be at least 1")
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    x, w = _standard(N)
    return np.concatenate((x / sigma, w / sigma))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "v = scaled_laguerre_rule(1, 1.0)",
            "call": "v",
            "gold_call": "np.array([1.0, 1.0])",
        },
        {
            "setup": (
                "N, sig = 6, 1.6742400931298613\n"
                "v = scaled_laguerre_rule(N, sig)\n"
                "w = v[N:]"
            ),
            "call": "round(float(w.sum()) - 1.0 / sig, 12)",
            "gold_call": "0.0",
        },
        {
            "setup": (
                "N, sig = 8, 2.5\n"
                "v = scaled_laguerre_rule(N, sig)\n"
                "u, w = v[:N], v[N:]\n"
                "first = float(np.sum(w * u))\n"
                "second = float(np.sum(w * u * u))"
            ),
            "call": "np.round([first - 1.0 / sig ** 2, second - 2.0 / sig ** 3], 12)",
            "gold_call": "np.zeros(2)",
        },
        {
            "setup": (
                "N = 5\n"
                "a = scaled_laguerre_rule(N, 1.0)\n"
                "b = scaled_laguerre_rule(N, 4.0)"
            ),
            "call": "np.round(b[:N] * 4.0 - a[:N], 12)",
            "gold_call": "np.zeros(5)",
        },
        {
            "setup": (
                "N = 10\n"
                "v = scaled_laguerre_rule(N, 1.6742400931298613)\n"
                "u = v[:N]"
            ),
            "call": "np.round(np.array([u[0], u[-1]]), 12)",
            "gold_call": "np.array([0.082302096997, 17.871210428571])",
            "tol": 1e-9,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        scaled_laguerre_rule(4, 0.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_scaled_laguerre_rule(4, 0.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
