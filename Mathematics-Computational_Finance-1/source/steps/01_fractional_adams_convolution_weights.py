"""
Produce the two triangular weight arrays the fractional Adams predictor-corrector scheme uses to advance the Volterra form of the fractional Riccati equation on a uniform grid. Step j+1 consumes a corrector row of length j+2 and a predictor row of length j+1, so both arrays are square and lower triangular with row 0 unused. The weights depend only on the fractional order, the step size and the number of steps, never on the Fourier argument, so they are built once per discretization level and reused at every quadrature node; that reuse is what keeps the multilevel estimator affordable.

A fractional derivative has memory. Where a classical one-step method needs only the previous value, the fractional Adams scheme evaluates a convolution against the whole history at every step, which is what makes each characteristic-function evaluation cost $O(M^2)$ rather than $O(M)$ and fixes the cost exponent that the work model downstream depends on.

The two rows come from different interpolations of the forcing term inside the fractional integral. The predictor uses a piecewise-constant reading, so its weights are plain first differences of a power; the corrector uses a piecewise-linear one, so its weights are second differences of the next power up. The corrector's opening weight is the one place where the pattern breaks: at the left endpoint the linear interpolant has only one neighbour, and the resulting expression is not what the interior formula produces there. Extending the interior formula to the edge is the commonest way to get a scheme that looks right and converges a full order too slowly.

Both arrays carry the step size to the power of the fractional order, which is where the non-integer order first enters the arithmetic and why halving the step does not halve the weights.

Returns
-------
np.ndarray of shape (2*(M+1)**2,). The first (M+1)**2 entries are the corrector array flattened row-major, the next (M+1)**2 the predictor array. Both are lower triangular with a zero first row; every corrector diagonal entry equals $\Delta t^{\alpha}/(\alpha(\alpha+1))$, and predictor row $j$ telescopes to $\Delta t^{\alpha} j^{\alpha}/\alpha$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def adams_convolution_weights(alpha, dt, M):
    """Convolution weights of the fractional Adams predictor-corrector scheme.

    The scheme advances the Volterra form of the fractional Riccati equation
    on the uniform grid t_j = j*dt, j = 0..M.  Step j+1 uses a corrector row
    a[j+1, 0..j+1] and a predictor row b[j+1, 0..j].  Row 0 of both is unused
    and is returned as zeros.

    Args:
        alpha (float): fractional order in (0, 1).
        dt (float): time-step size.
        M (int): number of steps.

    Expected return:
        np.ndarray of shape (2*(M+1)**2,).  The first (M+1)**2 entries are the
        corrector array a flattened row-major with shape (M+1, M+1); the next
        (M+1)**2 entries are the predictor array b in the same layout.  Entries
        outside the stated index ranges are zero.
     Raises:
        ValueError: if alpha is outside (0, 1), if dt <= 0, or if
        M < 1.
    """
    return np.zeros(2 * (M + 1) ** 2)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_adams_convolution_weights(alpha, dt, M):
    def _rows(alpha, dt, M):
        ca = dt ** alpha / (alpha * (alpha + 1.0))
        cb = dt ** alpha / alpha
        a = np.zeros((M + 1, M + 1))
        b = np.zeros((M + 1, M + 1))
        for jp in range(1, M + 1):
            j = jp - 1
            a[jp, 0] = ca * (j ** (alpha + 1.0)
                             - (j - alpha) * (j + 1.0) ** alpha)
            if j >= 1:
                k = np.arange(1, j + 1)
                a[jp, 1:j + 1] = ca * ((j - k + 2.0) ** (alpha + 1.0)
                                       + (j - k) ** (alpha + 1.0)
                                       - 2.0 * (j - k + 1.0) ** (alpha + 1.0))
            a[jp, jp] = ca
            k = np.arange(0, j + 1)
            b[jp, 0:j + 1] = cb * ((j + 1.0 - k) ** alpha - (j - k) ** alpha)
        return a, b

    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive")
    M = int(M)
    if M < 1:
        raise ValueError("M must be at least 1")
    a, b = _rows(float(alpha), float(dt), M)
    return np.concatenate((a.reshape(-1), b.reshape(-1)))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": (
                "w = adams_convolution_weights(0.62, 0.25, 4)\n"
                "n = 5\n"
                "a = w[:n * n].reshape(n, n)\n"
                "b = w[n * n:].reshape(n, n)"
            ),
            "call": "np.array([a[1, 0], a[1, 1], b[1, 0]])",
            "gold_call": ("np.array([0.26134114579090345, 0.4215179770821023, "
                          "0.6828591228730058])"),
        },
        {
            "setup": (
                "w = adams_convolution_weights(0.62, 0.25, 4)\n"
                "n = 5\n"
                "a = w[:n * n].reshape(n, n)"
            ),
            "call": "a[4, :5]",
            "gold_call": ("np.array([0.12934716687833486, 0.28027605198798844, "
                          "0.32915694913380683, 0.45260508072421984, "
                          "0.4215179770821023])"),
        },
        {
            "setup": (
                "alpha, dt, M = 0.55, 0.1, 6\n"
                "w = adams_convolution_weights(alpha, dt, M)\n"
                "n = M + 1\n"
                "a = w[:n * n].reshape(n, n)\n"
                "diag = np.array([a[j, j] for j in range(1, n)])\n"
                "want = dt ** alpha / (alpha * (alpha + 1.0))"
            ),
            "call": "diag - want",
            "gold_call": "np.zeros(6)",
        },
        {
            "setup": (
                "alpha, dt, M = 0.8, 0.5, 5\n"
                "w = adams_convolution_weights(alpha, dt, M)\n"
                "n = M + 1\n"
                "b = w[n * n:].reshape(n, n)\n"
                "rowsum = float(b[3].sum())\n"
                "want = dt ** alpha * 3.0 ** alpha / alpha"
            ),
            "call": "round(rowsum - want, 12)",
            "gold_call": "0.0",
        },
        {
            "setup": (
                "alpha, dt, M = 0.62, 0.25, 4\n"
                "w = adams_convolution_weights(alpha, dt, M)\n"
                "n = M + 1\n"
                "a = w[:n * n].reshape(n, n)\n"
                "b = w[n * n:].reshape(n, n)\n"
                "upper = float(np.abs(np.triu(a, 2)).sum() "
                "+ np.abs(np.triu(b, 1)).sum() + np.abs(a[0]).sum())"
            ),
            "call": "round(upper, 14)",
            "gold_call": "0.0",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        adams_convolution_weights(1.4, 0.25, 4)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_adams_convolution_weights(1.4, 0.25, 4)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        adams_convolution_weights(0.62, -0.25, 4)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_adams_convolution_weights(0.62, -0.25, 4)\n"
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
