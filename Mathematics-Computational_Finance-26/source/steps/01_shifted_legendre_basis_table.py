"""
Evaluate the shifted, normalised Legendre basis and its first two derivatives at a given set of asset prices, and return the three tables packed into one flat array. Every later step consumes this table: the projection of the observed profile needs the functions themselves, the convection integral needs the first derivatives, the diffusion integral needs the second, and the final expansion back to a price needs the functions again. Building it once, from a stable recurrence, is what keeps the whole pipeline consistent.

Evaluate the basis that the source paper's dimension reduction is built on, together with its first two derivatives in the asset price, at a given set of points, and return the three tables packed into one flat array. Every later step consumes this table: the projection of the observed profile needs the functions themselves, the convection integral needs the first derivatives, the diffusion integral needs the second, and the final expansion back to a price needs the functions again. Building it once, from a stable recurrence, is what keeps the whole pipeline consistent.

The basis is the Legendre family carried from its reference interval onto the price interval and rescaled. The rescaling is not free to choose: the paper fixes a particular constant for each mode, and that constant is what makes the reduction behave the way the rest of the method assumes. Every reduced matrix entry downstream carries the product of two of these constants, so a basis that is merely orthogonal rather than scaled the paper's way produces matrices that are wrong by a different factor in every row and column, and no later step recovers.

The change of variable onto the price interval is affine, so each differentiation contributes a constant factor from the chain rule. Those factors are easy to drop, and dropping them leaves a table that looks plausible mode by mode while being wrong by a fixed power throughout. The second-derivative table is where the damage shows first, because the reduced diffusion operator is built from it.

Evaluate the polynomials by the three-term recurrence rather than by forming explicit coefficients, and differentiate the recurrence itself to get the two derivative tables. Building monomial coefficients and differentiating those loses several digits by the eighth mode.

Legendre polynomials are the orthogonal family for the unweighted inner product on an interval, which makes them the natural basis for approximating a function when no part of the domain is privileged over another. On the reference interval they satisfy a three-term recurrence, and because the recurrence is an identity in the variable it can be differentiated term by term as many times as needed, giving recurrences for the derivatives that are as stable as the original.

Moving to a different interval is an affine substitution. Orthogonality survives it; other properties do not automatically, and what a particular method needs from its basis depends on how the method uses it. Reductions of this kind are usually set up so that the projection of the identity operator is the identity matrix, because otherwise a mass matrix appears on the left of the reduced system and has to be inverted at every step. Which normalisation delivers that is a property of the basis and the interval together, and it is fixed by the source paper rather than chosen here.

The reason the second derivatives matter is that the reduced diffusion operator integrates them against the basis functions in whatever form the paper's reduction specifies. Their accuracy therefore feeds straight into every diffusion entry, and an error in the chain-rule factors is indistinguishable from an error in the volatility model once the two are multiplied together.

Returns
-------
np.ndarray of shape (3*(N+1)*len(S),). The first (N+1)*len(S) entries are the basis values in row-major order over (mode, point), the next block the first derivatives and the last block the second derivatives, in the same layout.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def shifted_legendre_basis_table(S, N, smax):
    """Reduction basis and its first two derivatives on the price interval.

    Evaluates the Legendre-type basis that the source paper's dimension
    reduction uses, at the given asset prices, together with dell/dS and
    d2ell/dS2, and packs the three tables into a single flat array.  The
    per-mode scaling constant is the paper's.

    Args:
        S (np.ndarray): asset prices, each in [0, smax].
        N (int): highest mode index, so modes n = 0..N.
        smax (float): right end of the price interval.

    Expected return:
        np.ndarray of shape (3*(N+1)*len(S),).  The first (N+1)*len(S)
        entries are the basis values in row-major order over (mode, point),
        the next block the first derivatives and the last block the second
        derivatives, in the same layout.

    Raises:
        ValueError: if N < 0, if smax <= 0, if S is empty, or if any entry
        of S lies outside [0, smax].
    """
    return np.zeros(3 * (N + 1) * np.asarray(S).size)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_shifted_legendre_basis_table(S, N, smax):
    def _legendre(nmax, x):
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
        return P, dP, d2P

    S = np.asarray(S, dtype=float).reshape(-1)
    N = int(N)
    smax = float(smax)
    if N < 0:
        raise ValueError("N must be non-negative")
    if smax <= 0.0:
        raise ValueError("smax must be positive")
    if S.size == 0:
        raise ValueError("S must contain at least one point")
    if np.any(S < 0.0) or np.any(S > smax):
        raise ValueError("every S must lie in [0, smax]")
    x = 2.0 * S / smax - 1.0
    P, dP, d2P = _legendre(N, x)
    nrm = np.sqrt((2.0 * np.arange(N + 1) + 1.0) / smax).reshape(-1, 1)
    lv = nrm * P
    l1 = nrm * dP * (2.0 / smax)
    l2 = nrm * d2P * (2.0 / smax) ** 2
    return np.concatenate((lv.reshape(-1), l1.reshape(-1), l2.reshape(-1)))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "pass",
            "call": "shifted_legendre_basis_table(np.array([0.0, 2.5, 5.0, 7.5, 10.0]), 3, 10.0)[5:10]",
            "gold_call": "_oracle_shifted_legendre_basis_table(np.array([0.0, 2.5, 5.0, 7.5, 10.0]), 3, 10.0)[5:10]",
        },
        {
            "setup": "pass",
            "call": "shifted_legendre_basis_table(np.array([3.0]), 2, 10.0)",
            "gold_call": "_oracle_shifted_legendre_basis_table(np.array([3.0]), 2, 10.0)",
        },
        {
            "setup": "def run_model():\n    S = np.linspace(0.0, 4.0, 401)\n    t = shifted_legendre_basis_table(S, 3, 4.0)\n    L = t[:4 * 401].reshape(4, 401)\n    g = L[2] * L[3] + L[2] * L[2]\n    return float(np.sum(0.5 * (g[1:] + g[:-1]) * (S[1:] - S[:-1])))\ndef run_gold():\n    S = np.linspace(0.0, 4.0, 401)\n    t = _oracle_shifted_legendre_basis_table(S, 3, 4.0)\n    L = t[:4 * 401].reshape(4, 401)\n    g = L[2] * L[3] + L[2] * L[2]\n    return float(np.sum(0.5 * (g[1:] + g[:-1]) * (S[1:] - S[:-1])))",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "pass",
            "call": "shifted_legendre_basis_table(np.array([2.0]), 0, 10.0)",
            "gold_call": "_oracle_shifted_legendre_basis_table(np.array([2.0]), 0, 10.0)",
        },
        {
            "setup": "pass",
            "call": "shifted_legendre_basis_table(np.array([1.0, 4.0, 9.0]), 5, 10.0)",
            "gold_call": "_oracle_shifted_legendre_basis_table(np.array([1.0, 4.0, 9.0]), 5, 10.0)",
        },
        {
            "setup": "def run_model():\n    try:\n        shifted_legendre_basis_table(np.array([1.0]), 3, -1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_shifted_legendre_basis_table(np.array([1.0]), 3, -1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "def run_model():\n    try:\n        shifted_legendre_basis_table(np.array([12.0]), 3, 10.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_shifted_legendre_basis_table(np.array([12.0]), 3, 10.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
