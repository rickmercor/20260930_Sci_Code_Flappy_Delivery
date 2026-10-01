"""
Distribute the quadrature points over the hierarchy so that the total work is smallest among all allocations meeting the quadrature tolerance. Half the tolerance is reserved for the level-zero term and half for the level differences taken together, and each relaxed optimum is rounded up. The per-node cost of a level difference is the sum of the costs at the two levels it spans, because one evaluation needs a solve at each.

Splitting an integrand into a coarse term plus a telescoping sum of differences only pays if the differences are both smaller and cheaper to resolve than the original. Here they are smaller because consecutive discretizations agree to a power of the step size, and that is what lets the expensive fine levels be visited at a handful of nodes while the cheap coarse level carries the bulk of the value.

The optimisation is a constrained minimisation of a linear work functional under two inverse-power constraints, and it has a closed-form solution. The exponents in it are not the ones a multilevel Monte Carlo argument would produce: there the error model is a variance and the optimal counts go like the square root of the ratio of variance to cost, whereas here the error model is an algebraic quadrature bound with a smoothness index, and the counts go like a ratio raised to two over the index plus two. Using the square-root rule gives an allocation that is not merely different but qualitatively so, front-loading the first correction instead of tapering.

The split of the tolerance between the level-zero term and the corrections is a convention rather than an optimum, and it is the source of the factor of two inside the level-zero count. Omitting it lowers that count by enough to change the answer.

Returns
-------
np.ndarray of shape (L+1,) holding integer counts as floats, each at least one. For the task configuration the allocation is (10, 8, 4, 2). Counts are non-increasing whenever the level constants fall faster than the pairwise costs rise, and a level whose constant is negligible receives the floor of one point.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def allocate_quadrature_points(A, s0, s, eps_quad, W):
    """Number of quadrature points to place at each level of the hierarchy.

    The multilevel estimator applies the scaled Gauss-Laguerre rule with N_0
    points to the level-zero integrand and with N_l points to each level
    difference, l = 1..L.  Evaluating a level difference at one node costs one
    Riccati solve at level l and one at level l-1, so the total work is

        W_ML = W_0*N_0 + sum_{l=1..L} (W_l + W_{l-1}) * N_l.

    The algebraic quadrature error model bounds the level-zero remainder by
    A_0 * N_0**(-s0/2) and the level-l remainder by A_l * N_l**(-s/2) with a
    common correction smoothness index s.  The counts minimise the work
    subject to the quadrature part of the tolerance being met, using the
    allocation the source paper derives and the split of eps_quad it
    prescribes between the level-zero term and the level differences.  Each
    relaxed optimum is rounded up to the next integer.

    Args:
        A: (L+1,) array of algebraic quadrature constants, A[0] for the
            level-zero integrand and A[l] for the l-th level difference.
        s0 (float): smoothness index of the level-zero integrand.
        s (float): common smoothness index of the level differences.
        eps_quad (float): quadrature part of the prescribed tolerance.
        W: (L+1,) array of per-node solver costs, W[l] at level l.

    Expected return:
        np.ndarray of shape (L+1,) holding the integer counts N_0..N_L as
        floats, each at least 1.
        Raises:
        ValueError: if A and W differ in length, if A is empty, if
        eps_quad <= 0, if s0 <= 0 or s <= 0, or if any entry of A is not
        positive.
    """
    return np.zeros(np.asarray(A, dtype=float).size)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_allocate_quadrature_points(A, s0, s, eps_quad, W):
    A = np.asarray(A, dtype=float).reshape(-1)
    W = np.asarray(W, dtype=float).reshape(-1)
    s0, s, eps_quad = float(s0), float(s), float(eps_quad)
    if A.size != W.size:
        raise ValueError("A and W must have the same length")
    if A.size < 1:
        raise ValueError("A must hold at least the level-zero constant")
    if eps_quad <= 0.0:
        raise ValueError("eps_quad must be positive")
    if s0 <= 0.0 or s <= 0.0:
        raise ValueError("smoothness indices must be positive")
    if np.any(A <= 0.0):
        raise ValueError("quadrature constants must be positive")

    L = A.size - 1
    N0 = (2.0 * A[0] / eps_quad) ** (2.0 / s0)
    out = np.empty(L + 1)
    out[0] = np.ceil(N0)
    if L >= 1:
        pair = W[1:] + W[:-1]
        Ak = A[1:]
        total = np.sum(Ak ** (2.0 / (s + 2.0)) * pair ** (s / (s + 2.0)))
        lead = ((2.0 / eps_quad) * total) ** (2.0 / s)
        out[1:] = np.ceil((Ak / pair) ** (2.0 / (s + 2.0)) * lead)
    return np.maximum(out, 1.0)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": (
                "dt = np.array([1.0 / 32, 1.0 / 64, 1.0 / 128, 1.0 / 256])\n"
                "W = dt ** -2.0\n"
                "A = np.concatenate(([0.43], 4.0 * dt[1:] ** 1.62))"
            ),
            "call": "allocate_quadrature_points(A, 6.0, 4.0, 1e-3, W)",
            "gold_call": "np.array([10.0, 8.0, 4.0, 2.0])",
        },
        {
            "setup": "",
            "call": ("float(allocate_quadrature_points(np.array([0.43]), 6.0, "
                     "4.0, 1e-3, np.array([1024.0]))[0])"),
            "gold_call": "10.0",
        },
        {
            "setup": (
                "A = np.array([0.5, 0.01, 0.004])\n"
                "W = np.array([100.0, 400.0, 1600.0])\n"
                "N = allocate_quadrature_points(A, 4.0, 4.0, 1e-2, W)\n"
                "lhs0 = A[0] * N[0] ** -2.0\n"
                "rest = float(np.sum(A[1:] * N[1:] ** -2.0))"
            ),
            "call": "bool(lhs0 <= 5e-3 + 1e-15 and rest <= 5e-3 + 1e-15)",
            "gold_call": "True",
        },
        {
            "setup": (
                "A = np.array([0.5, 0.02, 0.01, 0.005])\n"
                "W = np.array([1.0, 4.0, 16.0, 64.0])\n"
                "N = allocate_quadrature_points(A, 4.0, 4.0, 1e-2, W)"
            ),
            "call": "bool(np.all(np.diff(N) <= 0.0))",
            "gold_call": "True",
        },
        {
            "setup": (
                "A = np.array([0.5, 1e-9, 1e-10])\n"
                "W = np.array([1.0, 4.0, 16.0])\n"
                "N = allocate_quadrature_points(A, 4.0, 4.0, 1e-1, W)"
            ),
            "call": "np.array([float(N[1]), float(N[2])])",
            "gold_call": "np.array([1.0, 1.0])",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        allocate_quadrature_points(np.array([0.5, 0.01]), 4.0, "
                "4.0, 1e-2, np.array([1.0]))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_allocate_quadrature_points(np.array([0.5, 0.01]), "
                "4.0, 4.0, 1e-2, np.array([1.0]))\n"
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
                "        allocate_quadrature_points(np.array([0.5, 0.0]), 4.0, "
                "4.0, 1e-2, np.array([1.0, 4.0]))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_allocate_quadrature_points(np.array([0.5, 0.0]), "
                "4.0, 4.0, 1e-2, np.array([1.0, 4.0]))\n"
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
