"""
Evaluate the second Piola-Kirchhoff stress and the mixed driving force at one material point, given the right Cauchy-Green tensor and the current internal variable. The free energy splits into an equilibrium part evaluated at C and a non-equilibrium part evaluated at the elastic measure C_e = C C_i^-1; both parts use the same Simo-Taylor form and, in this task, the same pair of constants. The stress is the sum of the two contributions. The driving force is the non-equilibrium stress post-multiplied by C, and it is deliberately returned unsymmetrised.

Finite viscoelasticity of this kind rests on a multiplicative split of the deformation gradient into elastic and inelastic factors. The inelastic factor is not tracked directly; what is stored is the strain-like internal variable C_i built from it, and the elastic response is measured by the mixed tensor C C_i^-1, which lives entirely in the reference configuration and shares its principal invariants with the elastic right Cauchy-Green tensor. Isotropy is what makes that substitution legitimate.

The energy itself is the compressible Simo-Taylor model: a neo-Hookean term plus a volumetric penalty built from the logarithm of the volume ratio and from the ratio minus one. Its derivative has a compact closed form, a multiple of the identity plus a multiple of the inverse transpose, which is why the whole stress evaluation reduces to a determinant, an inverse and two scalar coefficients.

The driving force deserves attention. Thermodynamics does not conjugate the internal variable to a symmetric stress here; the dissipation is the product of the mixed tensor M with the viscous rate, and M is symmetric only when C and C_i happen to commute. Symmetrising it looks harmless and is not: it changes what the evolution law transports.

Returns
-------
np.ndarray of shape (18,) packed as [S.ravel(), M.ravel()], both tensors flattened row-major. At C = C_i = I both halves are exactly zero, which is the stress-free reference state. For a non-commuting pair the off-diagonal gap M[0,1] - M[1,0] is a direct measure of the asymmetry; at the test state used below it is 693.4660.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def constitutive_response(C, Ci):
    """Second Piola-Kirchhoff stress and the mixed viscous driving force.

    The free energy splits as Psi = Psi_eq(C) + Psi_neq(Ce) with
    Ce = C Ci^{-1}, and both parts take the same Simo-Taylor form
    Psi(A) = m/2 (tr A - d - 2 log Jd) + l/2 ((log Jd)^2 + (Jd - 1)^2)
    with Jd the determinant of the corresponding deformation gradient,
    i.e. the square root of the determinant of the argument.  Material constants are
    lambda = lambda_visc = 30000, mu = mu_visc = 7500.

    Args:
        C: (3, 3) right Cauchy-Green tensor.
        Ci: (3, 3) internal (viscous) strain-like variable.

    Expected return:
        np.ndarray of shape (18,) packed as [S.ravel(), M.ravel()], where
        S = S_eq + S_neq and M is the mixed driving force conjugate to the
        viscous rate.  M is in general non-symmetric.
    """
    return np.zeros(18)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_constitutive_response(C, Ci):
    LAM, MU, LAMV, MUV = 30000.0, 7500.0, 30000.0, 7500.0
    V_DEV, V_VOL = 10000.0, 50000.0
    def _d_psi(A, m, l):
        Jd = np.sqrt(np.linalg.det(A))
        AinvT = np.linalg.inv(A).T
        return 0.5 * (m * (np.eye(3) - AinvT)
                      + l * (np.log(Jd) + Jd * (Jd - 1.0)) * AinvT)

    C = np.asarray(C, dtype=float)
    Ci = np.asarray(Ci, dtype=float)
    S_eq = 2.0 * _d_psi(C, MU, LAM)
    Ci_inv = np.linalg.inv(Ci)
    S_neq = 2.0 * _d_psi(C @ Ci_inv, MUV, LAMV) @ Ci_inv
    M = S_neq @ C
    return np.concatenate(((S_eq + S_neq).reshape(-1), M.reshape(-1)))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "I = np.eye(3)",
            "call": "constitutive_response(I, I)[:9]",
            "gold_call": "np.zeros(9)",
        },
        {
            "setup": "I = np.eye(3)",
            "call": "constitutive_response(I, I)[9:]",
            "gold_call": "np.zeros(9)",
        },
        {
            "setup": (
                "C = np.diag([1.21, 0.9, 1.0])\n"
                "Ci = np.diag([1.05, 0.98, 1.0])\n"
                "v = constitutive_response(C, Ci)"
            ),
            "call": "v[:9].reshape(3, 3) - v[:9].reshape(3, 3).T",
            "gold_call": "np.zeros((3, 3))",
        },
        {
            "setup": (
                "C = np.array([[1.4, 0.3, 0.0], [0.3, 0.9, 0.0], [0.0, 0.0, 1.0]])\n"
                "Ci = np.array([[1.0, 0.12, 0.0], [0.12, 1.15, 0.0], [0.0, 0.0, 1.0]])\n"
                "M = constitutive_response(C, Ci)[9:].reshape(3, 3)"
            ),
            "call": "round(float(M[0, 1] - M[1, 0]), 4)",
            "gold_call": "693.4660",
            "tol": 1e-6,
        },
        {
            "setup": (
                "bad = np.zeros((3, 3))\n"
                "def run_model():\n"
                "    try:\n"
                "        v = constitutive_response(np.eye(3), bad)\n"
                "        return 0 if np.all(np.isfinite(v)) else 1\n"
                "    except Exception:\n"
                "        return 1\n"
                "def run_gold():\n"
                "    try:\n"
                "        v = _oracle_constitutive_response(np.eye(3), bad)\n"
                "        return 0 if np.all(np.isfinite(v)) else 1\n"
                "    except Exception:\n"
                "        return 1"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
