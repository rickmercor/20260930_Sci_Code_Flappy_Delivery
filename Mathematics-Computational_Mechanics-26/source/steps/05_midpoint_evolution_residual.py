"""
Form the discrete residual of the internal evolution law at one quadrature point over one time step. The internal variable is supplied at both ends of the step in packed six-component form; its rate is the backward difference quotient, and the constitutive quantities are evaluated at the arithmetic midpoint of the two endpoint values. The residual is the rate minus the flow term, projected onto symmetric tensors before being packed back into six components.

Discretising an evolution law in time turns a constitutive differential equation into an algebraic constraint on the endpoint state. The midpoint rule is chosen here because it mirrors the underlying energy-dissipation structure more faithfully than a fully implicit rule; the source paper designs its scheme around preserving that structure as closely as the discrete setting allows.

Two details decide the answer and neither is inferable from the equation alone. The first is which quantities are averaged and how: the internal variable is averaged arithmetically between endpoints, and evaluating it at the end of the step instead turns the scheme into backward Euler. The second is the test space. The weak form of the evolution law is tested with symmetric tensor-valued test functions, so only the symmetric part of the pointwise equation is enforced. Since the flow term is not symmetric in general, enforcing all nine components instead of six over-constrains the internal variable and shifts every downstream quantity.

The packing convention matters for the same reason. Storing a symmetric tensor as six numbers and mirroring on unpacking means a perturbation of an off-diagonal slot moves two tensor entries at once, which is what the finite-difference tangent in the final step differentiates.

Returns
-------
np.ndarray of shape (6,) packed as (R11, R22, R33, R12, R13, R23). A state in which the internal variable does not change and the elastic measure is the identity gives exactly zero. When the elastic measure is the identity but the internal variable does change, the flow term drops out and the residual is the bare difference quotient.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evolution_residual(C_mid, ci_new, ci_old, h):
    """Midpoint-discrete evolution residual at one quadrature point.

    A symmetric 3x3 tensor is packed as the 6-vector
    (A11, A22, A33, A12, A13, A23) and unpacked by mirroring.
    The internal variable is evaluated at the arithmetic midpoint of its
    endpoint values; the rate is the backward difference quotient.  The
    residual is tested with symmetric test functions.

    Args:
        C_mid: (3, 3) midpoint right Cauchy-Green tensor.
        ci_new, ci_old: (6,) packed internal variable at t_{n+1} and t_n.
        h (float): time-step size.

    Raises:
        numpy.linalg.LinAlgError: if the midpoint internal variable is singular.

    Expected return:
        np.ndarray of shape (6,): the packed residual.
    """
    return np.zeros(6)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_evolution_residual(C_mid, ci_new, ci_old, h):
    LAM, MU, LAMV, MUV = 30000.0, 7500.0, 30000.0, 7500.0
    V_DEV, V_VOL = 10000.0, 50000.0
    SYM = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

    def _vec_to_sym(v):
        A = np.zeros((3, 3))
        for k, (i, j) in enumerate(SYM):
            A[i, j] = v[k]
            A[j, i] = v[k]
        return A

    def _sym_to_vec(A):
        return np.array([A[i, j] for (i, j) in SYM])
    def _d_psi(A, m, l):
        Jd = np.sqrt(np.linalg.det(A))
        AinvT = np.linalg.inv(A).T
        return 0.5 * (m * (np.eye(3) - AinvT)
                      + l * (np.log(Jd) + Jd * (Jd - 1.0)) * AinvT)
    def _constitutive(C, Ci):
        S_eq = 2.0 * _d_psi(C, MU, LAM)
        Ci_inv = np.linalg.inv(Ci)
        S_neq = 2.0 * _d_psi(C @ Ci_inv, MUV, LAMV) @ Ci_inv
        return S_eq + S_neq, S_neq @ C
    def _flow(M, Ci):
        A = M.T
        vol = np.trace(A) / 3.0 * np.eye(3)
        return 2.0 * Ci @ (vol / V_VOL + (A - vol) / (2.0 * V_DEV))

    Ci_new = _vec_to_sym(np.asarray(ci_new, dtype=float))
    Ci_old = _vec_to_sym(np.asarray(ci_old, dtype=float))
    Ci_mid = 0.5 * (Ci_new + Ci_old)
    _, M = _constitutive(np.asarray(C_mid, dtype=float), Ci_mid)
    R = (Ci_new - Ci_old) / h - _flow(M, Ci_mid)
    return _sym_to_vec(0.5 * (R + R.T))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "e = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])",
            "call": "evolution_residual(np.eye(3), e, e, 2.0)",
            "gold_call": "np.zeros(6)",
        },
        {
            "setup": (
                "e = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])\n"
                "n = np.array([1.02, 1.0, 1.0, 0.0, 0.0, 0.0])\n"
                "Cm = np.diag([1.01, 1.0, 1.0])"
            ),
            "call": "evolution_residual(Cm, n, e, 2.0)",
            "gold_call": "np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])",
        },
        {
            "setup": (
                "e = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])\n"
                "C = np.array([[1.3, 0.2, 0.0], [0.2, 0.9, 0.0], [0.0, 0.0, 1.0]])\n"
                "r = evolution_residual(C, e, e, 2.0)"
            ),
            "call": "int(r.shape[0])",
            "gold_call": "6",
        },
        {
            "setup": (
                "e = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])\n"
                "C = np.array([[1.3, 0.2, 0.0], [0.2, 0.9, 0.0], [0.0, 0.0, 1.0]])"
            ),
            "call": "evolution_residual(C, e, e, 1.0)",
            "gold_call": "_oracle_evolution_residual(C, e, e, 1.0)",
        },
        {
            "setup": (
                "z = np.zeros(6)\n"
                "def run_model():\n"
                "    try:\n"
                "        v = evolution_residual(np.eye(3), z, z, 2.0)\n"
                "        return 0 if np.all(np.isfinite(v)) else 1\n"
                "    except np.linalg.LinAlgError:\n"
                "        return 2\n"
                "    except Exception:\n"
                "        return 3\n"
                "def run_gold():\n"
                "    try:\n"
                "        v = _oracle_evolution_residual(np.eye(3), z, z, 2.0)\n"
                "        return 0 if np.all(np.isfinite(v)) else 1\n"
                "    except np.linalg.LinAlgError:\n"
                "        return 2\n"
                "    except Exception:\n"
                "        return 3"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
