"""
Given the four blocks of the monolithic tangent and the two residuals, eliminate the internal-variable increment at the level of the linearised system and return the displacement increment. The elimination requires one solve against the internal block with the coupling matrix and the internal residual as simultaneous right-hand sides, followed by one solve of the reduced system. The internal increment itself is recovered afterwards by the caller from the same two factors.

Once the evolution law is treated as an algebraic constraint rather than something to be resolved separately, the Newton system for the step has two unknown fields and a two-by-two block tangent. That tangent is not symmetric: the sensitivity of the internal forces to the internal variable and the sensitivity of the evolution equation to the deformation are unrelated objects, because the evolution law is a constitutive equation rather than the stationarity condition of a potential.

Eliminating the internal increment produces the Schur complement of the internal block, and the reduction has a clean geometric reading. The linearised evolution equation defines the tangent space of the manifold of admissible states; a column operator built from the internal block spans the null space of that constraint, and the reduced operator is the restriction of the full tangent to it. Because the tangent is non-symmetric, the operator used on the right differs from the one used on the left: one parameterises admissible increments, the other selects test directions.

The right-hand side is where this scheme parts company with the classical one. A nested Gauss-point scheme solves the internal equation to convergence before assembling the global step, so its internal residual is zero by construction and the reduced right-hand side is the momentum residual alone. Here the internal residual is generally non-zero during the global iteration, and its contribution is carried into the displacement equation. Both schemes converge to the same solution; they take different paths to it, and this step is where that difference is created.

Returns
-------
np.ndarray of shape (nq,), the displacement increment. With the internal residual set to zero the result reduces to the classical condensed step. The recovered pair of increments satisfies both block rows of the original system exactly.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def condensed_correction(Kqq, KqC, KCq, KCC, Rq, RC):
    """Condensed Newton correction of the monolithic block system.

    The monolithic system couples the displacement increment and the
    internal-variable increment through a generally non-symmetric block
    tangent.  The internal increment is eliminated at the level of the
    linearised system; the reduction keeps whatever the internal residual
    contributes to the reduced right-hand side, which is what distinguishes
    this scheme from the classical nested Gauss-point condensation.

    Args:
        Kqq: (nq, nq); KqC: (nq, nc); KCq: (nc, nq); KCC: (nc, nc).
        Rq: (nq,); RC: (nc,).

    Raises:
        numpy.linalg.LinAlgError: if KCC or the Schur complement is singular.

    Expected return:
        np.ndarray of shape (nq,): the displacement increment.
    """
    return np.zeros(np.asarray(Rq).shape[0])

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_condensed_correction(Kqq, KqC, KCq, KCC, Rq, RC):
    Kqq = np.asarray(Kqq, dtype=float)
    KqC = np.asarray(KqC, dtype=float)
    KCq = np.asarray(KCq, dtype=float)
    KCC = np.asarray(KCC, dtype=float)
    Rq = np.asarray(Rq, dtype=float).reshape(-1)
    RC = np.asarray(RC, dtype=float).reshape(-1)
    W = np.linalg.solve(KCC, np.column_stack([KCq, RC.reshape(-1, 1)]))
    return np.linalg.solve(Kqq - KqC @ W[:, :-1], -(Rq - KqC @ W[:, -1]))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": (
                "Kqq = np.eye(2) * 2.0\n"
                "KqC = np.zeros((2, 2))\n"
                "KCq = np.zeros((2, 2))\n"
                "KCC = np.eye(2)\n"
                "Rq = np.array([2.0, -4.0])\n"
                "RC = np.zeros(2)"
            ),
            "call": "condensed_correction(Kqq, KqC, KCq, KCC, Rq, RC)",
            "gold_call": "np.array([-1.0, 2.0])",
        },
        {
            "setup": (
                "Kqq = np.array([[4.0, 1.0], [0.0, 3.0]])\n"
                "KqC = np.array([[1.0, 0.0], [0.0, 2.0]])\n"
                "KCq = np.array([[2.0, 0.0], [0.0, 1.0]])\n"
                "KCC = np.array([[2.0, 0.0], [0.0, 4.0]])\n"
                "Rq = np.array([1.0, 1.0])\n"
                "RC = np.zeros(2)"
            ),
            "call": "condensed_correction(Kqq, KqC, KCq, KCC, Rq, RC)",
            "gold_call": "np.linalg.solve(Kqq - KqC @ np.linalg.solve(KCC, KCq), -Rq)",
        },
        {
            "setup": (
                "Kqq = np.array([[4.0, 1.0], [0.0, 3.0]])\n"
                "KqC = np.array([[1.0, 0.0], [0.0, 2.0]])\n"
                "KCq = np.array([[2.0, 0.0], [0.0, 1.0]])\n"
                "KCC = np.array([[2.0, 0.0], [0.0, 4.0]])\n"
                "Rq = np.array([1.0, 1.0])\n"
                "RC = np.array([0.5, -0.25])\n"
                "a = condensed_correction(Kqq, KqC, KCq, KCC, Rq, RC)\n"
                "b = condensed_correction(Kqq, KqC, KCq, KCC, Rq, np.zeros(2))"
            ),
            "call": "float(np.linalg.norm(a - b) > 1e-9)",
            "gold_call": "1.0",
        },
        {
            "setup": (
                "rng_free = np.array([[3.0, 0.5, 0.0], [0.2, 4.0, 1.0], [0.0, 0.3, 5.0]])\n"
                "KqC = np.array([[0.4, 0.1], [0.0, 0.2], [0.3, 0.0]])\n"
                "KCq = np.array([[0.5, 0.0, 0.1], [0.2, 0.3, 0.0]])\n"
                "KCC = np.array([[3.0, 0.4], [0.1, 2.0]])\n"
                "Rq = np.array([1.0, -2.0, 0.5])\n"
                "RC = np.array([0.2, 0.7])\n"
                "dq = condensed_correction(rng_free, KqC, KCq, KCC, Rq, RC)\n"
                "dC = -np.linalg.solve(KCC, RC) - np.linalg.solve(KCC, KCq) @ dq"
            ),
            "call": "RC + KCq @ dq + KCC @ dC",
            "gold_call": "np.zeros(2)",
        },
        {
            "setup": (
                "Kqq = np.eye(2)\n"
                "KqC = np.zeros((2, 2))\n"
                "KCq = np.zeros((2, 2))\n"
                "KCC = np.zeros((2, 2))\n"
                "Rq = np.ones(2)\n"
                "RC = np.ones(2)\n"
                "def run_model():\n"
                "    try:\n"
                "        condensed_correction(Kqq, KqC, KCq, KCC, Rq, RC)\n"
                "        return 0\n"
                "    except np.linalg.LinAlgError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_condensed_correction(Kqq, KqC, KCq, KCC, Rq, RC)\n"
                "        return 0\n"
                "    except np.linalg.LinAlgError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
