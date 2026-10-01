"""
Assemble the reduced convection matrix, the projection of the first-order drift term of the pricing equation onto the basis, in the form the source paper's reduction uses. Entry (m, n) pairs mode m with mode n; which of the two is differentiated, and whether a derivative is moved across by parts, is the paper's convention and is not free to choose here.

Unlike its diffusion counterpart the matrix carries no time dependence and no volatility, so it is built once and reused at every time level. It is also strongly structured, and the structure is worth predicting before computing anything: the drift term does not raise polynomial degree, and the basis is graded by degree, so most of the matrix is forced to vanish and part of what survives is fixed by leading coefficients alone. An implementation that produces a dense or symmetric matrix has the pairing or the index order wrong.

The integrand is a polynomial of bounded degree, so a Gauss-Legendre rule with enough nodes evaluates it exactly rather than approximately. Once the node count clears that bound the answer stops changing, which means the node count is not a tuning parameter and the matrix is determined to machine precision by the basis alone.

The nodes and weights supplied by a standard Gauss-Legendre routine live on the reference interval and must be mapped affinely onto the price interval, the weights picking up the same scaling factor as the interval length. Forgetting the weight scaling leaves every entry short by a constant factor.

Projecting a differential operator onto a basis turns it into a matrix, and the structure of that matrix records how the operator interacts with the grading of the basis by degree. An operator that does not raise degree produces a triangular matrix against a degree-graded orthogonal basis, and one that preserves leading coefficients up to a factor produces a diagonal determined by that factor. Which triangle, and which factor, depends on how the two modes are paired inside the integral.

Knowing this before computing anything is what makes an assembled operator checkable. Exact structural facts survive floating-point arithmetic to rounding, so an implementation that produces a dense matrix where theory says triangular has an error that no amount of refinement will remove, and one whose diagonal is off by a constant factor has almost certainly mis-scaled either the basis or the quadrature weights.

Gauss-Legendre quadrature with a given node count integrates polynomials up to a degree roughly twice that count without error. When the integrand is known in advance to be polynomial of bounded degree, this converts a numerical approximation into an exact evaluation, which removes quadrature error from the error budget entirely and makes the result reproducible across implementations that choose different node counts.

Returns
-------
np.ndarray of shape ((N+1)**2,), the convection matrix flattened row-major, entry (m, n) at index m*(N+1)+n.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def reduced_convection_matrix(N, smax, nq):
    """Projection of the first-order drift term onto the reduced basis.

    Entry (m, n) is the projection of the first-order drift term onto the
    reduction basis, in the pairing the source paper's reduction specifies,
    evaluated by nq-node Gauss-Legendre quadrature mapped affinely onto
    [0, smax].

    Args:
        N (int): highest mode index, so modes n = 0..N.
        smax (float): right end of the price interval.
        nq (int): number of Gauss-Legendre nodes.

    Expected return:
        np.ndarray of shape ((N+1)**2,), the matrix flattened row-major,
        entry (m, n) at index m*(N+1)+n.

    Raises:
        ValueError: if N < 0, if smax <= 0, or if nq < 1.
    """
    return np.zeros((N + 1) ** 2)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reduced_convection_matrix(N, smax, nq):
    def _tab(nmax, pts, right):
        x = 2.0 * pts / right - 1.0
        P = np.zeros((nmax + 1, x.size))
        dP = np.zeros((nmax + 1, x.size))
        P[0] = 1.0
        if nmax >= 1:
            P[1] = x
            dP[1] = 1.0
        for n in range(1, nmax):
            P[n + 1] = ((2.0 * n + 1.0) * x * P[n] - n * P[n - 1]) / (n + 1.0)
            dP[n + 1] = ((2.0 * n + 1.0) * (P[n] + x * dP[n])
                         - n * dP[n - 1]) / (n + 1.0)
        nrm = np.sqrt((2.0 * np.arange(nmax + 1) + 1.0) / right).reshape(-1, 1)
        return nrm * P, nrm * dP * (2.0 / right)

    N = int(N)
    smax = float(smax)
    nq = int(nq)
    if N < 0:
        raise ValueError("N must be non-negative")
    if smax <= 0.0:
        raise ValueError("smax must be positive")
    if nq < 1:
        raise ValueError("nq must be at least 1")
    xg, wg = np.polynomial.legendre.leggauss(nq)
    Sq = 0.5 * smax * (xg + 1.0)
    wq = 0.5 * smax * wg
    L, dL = _tab(N, Sq, smax)
    B = (L * (wq * Sq).reshape(1, -1)) @ dL.T
    return B.reshape(-1)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "pass",
            "call": "reduced_convection_matrix(4, 10.0, 20)",
            "gold_call": "_oracle_reduced_convection_matrix(4, 10.0, 20)",
        },
        {
            "setup": "pass",
            "call": "reduced_convection_matrix(6, 10.0, 30)",
            "gold_call": "_oracle_reduced_convection_matrix(6, 10.0, 30)",
        },
        {
            "setup": "pass",
            "call": "reduced_convection_matrix(5, 7.5, 25)",
            "gold_call": "_oracle_reduced_convection_matrix(5, 7.5, 25)",
        },
        {
            "setup": "pass",
            "call": "reduced_convection_matrix(4, 10.0, 12)",
            "gold_call": "_oracle_reduced_convection_matrix(4, 10.0, 40)",
        },
        {
            "setup": "pass",
            "call": "reduced_convection_matrix(0, 4.0, 5)",
            "gold_call": "_oracle_reduced_convection_matrix(0, 4.0, 5)",
        },
        {
            "setup": "def run_model():\n    try:\n        reduced_convection_matrix(4, 10.0, 0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_reduced_convection_matrix(4, 10.0, 0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "def run_model():\n    try:\n        reduced_convection_matrix(-1, 10.0, 20)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_reduced_convection_matrix(-1, 10.0, 20)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
