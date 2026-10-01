"""
Assemble the reduced diffusion matrix at a given time, the projection of the second-order term of the pricing equation onto the basis, in the form the source paper's reduction uses. Entry (m, n) weights the pairing of modes m and n by the squared volatility and the squared asset price. Which mode carries the second derivative, and whether the expression is left as written or transformed before integrating, is the paper's convention.

That choice is the single most consequential thing about this step. The two natural readings differ by a boundary term that does not vanish here, because the coefficient of the second-order term degenerates at one end of the interval and not at the other, and they produce matrices with different symmetry. Both give a well-behaved computation; only one gives the paper's method.

The time dependence enters only through the volatility, which is a smile in the asset price whose curvature decays over the horizon. Because the squared volatility is polynomial in the asset price, the whole integrand is polynomial too, of degree bounded by the truncation level, so a Gauss-Legendre rule with enough nodes is exact and the entry is determined to machine precision. Evaluating the smile at the wrong instant, at a grid node instead of the interval midpoint the caller asks for, is a more likely error than any quadrature issue.

Two columns of the matrix vanish identically whatever the volatility, for reasons that follow from the lowest two modes alone. Those are the quickest check that the derivative table feeding this step is correct.

The second-order term of a pricing equation carries a coefficient that vanishes at zero asset price, because the diffusion of a geometric process shuts off as the price approaches zero. On a grid this degeneracy is awkward: the equation loses its parabolic character at the left edge and schemes need special treatment there. Projecting onto a global polynomial basis sidesteps the issue, because the basis functions do not see the edge as a special location and the degeneracy simply appears as a weight inside an integral that is perfectly well behaved.

Whether to move a derivative across by parts is a genuine modelling decision rather than an algebraic convenience. The weak form obtained that way is the standard route for a Galerkin method, because it lowers the smoothness required of the trial space and yields a symmetric operator for a self-adjoint problem. Neither advantage necessarily applies: a polynomial trial space is already infinitely smooth, and an operator with a first-order term is not self-adjoint anyway. What the transformation always does is generate a boundary term, and with a coefficient that degenerates at one end of the interval and not at the other, that term is not zero. Which form the method intends is therefore a statement the source has to make.

A state-dependent volatility smile is what distinguishes this from a constant-coefficient exercise. It makes the diffusion matrix depend on time, so the reduced system has a time-varying coefficient matrix and cannot be solved by a single matrix exponential, which is exactly why the reconstruction is posed as a minimisation over the whole trajectory.

Returns
-------
np.ndarray of shape ((N+1)**2,), the diffusion matrix at time t flattened row-major, entry (m, n) at index m*(N+1)+n.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def reduced_diffusion_matrix(t, N, smax, nq, sigma0, eta, sref, T):
    """Projection of the second-order term onto the reduced basis at time t.

    Entry (m, n) is the projection of the second-order term onto the
    reduction basis at time t, in the form the source paper's reduction
    specifies, evaluated by nq-node Gauss-Legendre quadrature mapped affinely
    onto [0, smax].  The volatility is
    sigma(t,S) = sigma0 * sqrt(1 + eta * exp(-t/T) * ((S - sref)/sref)**2).

    Args:
        t (float): time at which the volatility is evaluated.
        N (int): highest mode index, so modes n = 0..N.
        smax (float): right end of the price interval.
        nq (int): number of Gauss-Legendre nodes.
        sigma0 (float): base volatility level.
        eta (float): smile curvature parameter.
        sref (float): reference price of the smile.
        T (float): horizon setting the decay of the smile.

    Expected return:
        np.ndarray of shape ((N+1)**2,), the matrix flattened row-major,
        entry (m, n) at index m*(N+1)+n.

    Raises:
        ValueError: if N < 0, if smax <= 0, if sref <= 0, if T <= 0, if
        nq < 1, or if eta < 0.
    """
    return np.zeros((N + 1) ** 2)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reduced_diffusion_matrix(t, N, smax, nq, sigma0, eta, sref, T):
    def _tab(nmax, pts, right):
        x = 2.0 * pts / right - 1.0
        P = np.zeros((nmax + 1, x.size))
        dP = np.zeros((nmax + 1, x.size))
        d2P = np.zeros((nmax + 1, x.size))
        P[0] = 1.0
        if nmax >= 1:
            P[1] = x
            dP[1] = 1.0
        for n in range(1, nmax):
            P[n + 1] = ((2.0 * n + 1.0) * x * P[n] - n * P[n - 1]) / (n + 1.0)
            dP[n + 1] = ((2.0 * n + 1.0) * (P[n] + x * dP[n])
                         - n * dP[n - 1]) / (n + 1.0)
            d2P[n + 1] = ((2.0 * n + 1.0) * (2.0 * dP[n] + x * d2P[n])
                          - n * d2P[n - 1]) / (n + 1.0)
        nrm = np.sqrt((2.0 * np.arange(nmax + 1) + 1.0) / right).reshape(-1, 1)
        return nrm * P, nrm * d2P * (2.0 / right) ** 2

    t = float(t)
    N = int(N)
    smax = float(smax)
    nq = int(nq)
    sigma0 = float(sigma0)
    eta = float(eta)
    sref = float(sref)
    T = float(T)
    if N < 0:
        raise ValueError("N must be non-negative")
    if smax <= 0.0 or sref <= 0.0 or T <= 0.0:
        raise ValueError("smax, sref and T must be positive")
    if nq < 1:
        raise ValueError("nq must be at least 1")
    if eta < 0.0:
        raise ValueError("eta must be non-negative")
    xg, wg = np.polynomial.legendre.leggauss(nq)
    Sq = 0.5 * smax * (xg + 1.0)
    wq = 0.5 * smax * wg
    L, d2L = _tab(N, Sq, smax)
    s2 = sigma0 ** 2 * (1.0 + eta * np.exp(-t / T) * ((Sq - sref) / sref) ** 2)
    A = (L * (wq * s2 * Sq ** 2).reshape(1, -1)) @ d2L.T
    return A.reshape(-1)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "pass",
            "call": "reduced_diffusion_matrix(0.0125, 4, 10.0, 20, 0.2, 0.25, 5.0, 1.0)",
            "gold_call": "_oracle_reduced_diffusion_matrix(0.0125, 4, 10.0, 20, 0.2, 0.25, 5.0, 1.0)",
        },
        {
            "setup": "pass",
            "call": "reduced_diffusion_matrix(0.5, 6, 10.0, 30, 0.2, 0.25, 5.0, 1.0)",
            "gold_call": "_oracle_reduced_diffusion_matrix(0.5, 6, 10.0, 30, 0.2, 0.25, 5.0, 1.0)",
        },
        {
            "setup": "pass",
            "call": "reduced_diffusion_matrix(0.5, 3, 10.0, 20, 0.2, 0.0, 5.0, 1.0)",
            "gold_call": "_oracle_reduced_diffusion_matrix(0.5, 3, 10.0, 20, 0.2, 0.0, 5.0, 1.0)",
        },
        {
            "setup": "pass",
            "call": "reduced_diffusion_matrix(0.3, 4, 10.0, 14, 0.2, 0.25, 5.0, 1.0)",
            "gold_call": "_oracle_reduced_diffusion_matrix(0.3, 4, 10.0, 40, 0.2, 0.25, 5.0, 1.0)",
        },
        {
            "setup": "pass",
            "call": "reduced_diffusion_matrix(0.0, 2, 10.0, 20, 0.2, 0.25, 5.0, 1.0)",
            "gold_call": "_oracle_reduced_diffusion_matrix(0.0, 2, 10.0, 20, 0.2, 0.25, 5.0, 1.0)",
        },
        {
            "setup": "def run_model():\n    try:\n        reduced_diffusion_matrix(0.1, 4, 10.0, 20, 0.2, 0.25, 0.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_reduced_diffusion_matrix(0.1, 4, 10.0, 20, 0.2, 0.25, 0.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "def run_model():\n    try:\n        reduced_diffusion_matrix(0.1, 4, 10.0, 20, 0.2, -1.0, 5.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_reduced_diffusion_matrix(0.1, 4, 10.0, 20, 0.2, -1.0, 5.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
