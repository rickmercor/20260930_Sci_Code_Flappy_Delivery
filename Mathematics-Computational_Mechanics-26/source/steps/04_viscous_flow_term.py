"""
Apply the inverse viscosity operator to the transpose of the driving force and pre-multiply by twice the internal variable. This is the whole right-hand side of the internal evolution law at a single material point. The viscosity operator is isotropic and built from the volumetric and deviatoric projectors with separate coefficients, so inverting it is a matter of dividing each projection by its own coefficient rather than inverting any matrix. The volumetric projector uses a factor of one third, corresponding to three dimensions, even in the plane-strain setting.

The model belongs to the class of generalized standard materials: the reversible response comes from a free-energy potential and the irreversible one from a convex dissipation potential. Taking that potential quadratic in the viscous rate gives a linear relation between the driving force and the rate, with a fourth-order viscosity tensor in between. Positive coefficients on both projections guarantee non-negative dissipation, which is what makes the model thermodynamically admissible.

Splitting the operator into volumetric and deviatoric parts lets a material resist shape change and volume change at different rates, which is the physically interesting case: most polymers relax deviatorically far faster than they relax volumetrically. Here the volumetric viscosity is five times the deviatoric one.

The coefficient on the deviatoric projector is a convention, not a derivation. Writing the operator with a factor of two there is standard in this literature because it mirrors how the shear modulus enters an elasticity tensor, but the convention has to be read off the source rather than guessed, since the parameter value alone does not disclose it.

Returns
-------
np.ndarray of shape (9,), the flow tensor flattened row-major. A zero driving force gives a zero flow. A purely volumetric driving force of magnitude 3 with C_i = I returns 2*3/V_vol on each diagonal entry and nothing off it.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def viscous_flow(M, Ci):
    """The viscous flow term appearing in the internal evolution law.

    The isotropic viscosity operator is built from the volumetric and
    deviatoric projectors with V_vol = 50000 and V_dev = 10000, using the
    volumetric-deviatoric split stated in the source paper.  The volumetric
    part uses d = 3.

    Args:
        M: (3, 3) mixed driving force.
        Ci: (3, 3) internal variable.

    Raises:
        ValueError: if M does not have shape (3, 3).

    Expected return:
        np.ndarray of shape (9,): the flow tensor flattened row-major.
    """
    return np.zeros(9)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_viscous_flow(M, Ci):
    LAM, MU, LAMV, MUV = 30000.0, 7500.0, 30000.0, 7500.0
    V_DEV, V_VOL = 10000.0, 50000.0

    M = np.asarray(M, dtype=float)
    Ci = np.asarray(Ci, dtype=float)
    A = M.T
    vol = np.trace(A) / 3.0 * np.eye(3)
    Vinv_A = vol / V_VOL + (A - vol) / (2.0 * V_DEV)
    return (2.0 * Ci @ Vinv_A).reshape(-1)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "z = np.zeros((3, 3))",
            "call": "viscous_flow(z, np.eye(3))",
            "gold_call": "np.zeros(9)",
        },
        {
            "setup": "M = np.eye(3) * 3.0",
            "call": "viscous_flow(M, np.eye(3))",
            "gold_call": "(2.0 * np.eye(3) * (3.0 / 50000.0)).reshape(-1)",
        },
        {
            "setup": (
                "M = np.array([[1.0, 2.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])\n"
                "f = viscous_flow(M, np.eye(3)).reshape(3, 3)"
            ),
            "call": "float(f[1, 0])",
            "gold_call": "2.0 * 2.0 / (2.0 * 10000.0)",
        },
        {
            "setup": (
                "M = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])\n"
                "Ci = np.diag([1.1, 0.95, 1.0])\n"
                "f = viscous_flow(M, Ci).reshape(3, 3)"
            ),
            "call": "float(np.trace(np.linalg.inv(Ci) @ f) - 2.0 * np.trace(M) / 3.0 / 50000.0 * 3.0)",
            "gold_call": "0.0",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        viscous_flow(np.zeros((2, 2)), np.eye(3))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_viscous_flow(np.zeros((2, 2)), np.eye(3))\n"
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
