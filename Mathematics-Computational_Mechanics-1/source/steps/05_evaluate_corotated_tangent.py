"""
Differentiate the fixed corotated first Piola-Kirchhoff stress with respect to the deformation gradient to obtain the fourth-order material tangent.

The tangent of a corotated model contains the derivative of the polar rotation, which is not the derivative of a smooth algebraic expression in the entries of the deformation gradient but follows from differentiating the polar decomposition itself. In two dimensions, the rotation has a single degree of freedom, so its derivative is a rank-one object built from the rotation and the trace of the symmetric stretch.

Returns
-------
np.ndarray of shape (n_particles, 2, 2, 2, 2): the derivative of the first Piola-Kirchhoff stress with respect to the deformation gradient of every particle.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_corotated_tangent(deformation_gradients: "np.ndarray", shear_modulus: float,
                               lame_first: float) -> "np.ndarray":
    """Build the fourth-order tangent of the fixed corotated stress.

    The returned array holds the derivative of the first Piola-Kirchhoff
    stress with respect to the deformation gradient, so that entry
    ``(p, a, b, c, d)`` is the derivative of the stress component ``(a, b)`` of
    particle ``p`` with respect to the deformation gradient component
    ``(c, d)`` of the same particle. The tangent is symmetric under exchange of
    the index pair ``(a, b)`` with the index pair ``(c, d)``, since it is the
    second derivative of the strain energy density.

    Parameters
    ----------
    deformation_gradients : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the deformation gradients;
        every determinant must be > 0.
    shear_modulus : float
        Shear modulus in pascals (shear_modulus > 0).
    lame_first : float
        First Lame constant in pascals (lame_first >= 0).

    Returns
    -------
    tangents : "np.ndarray"
        Array of shape (n_particles, 2, 2, 2, 2) holding the material tangent
        of every particle, in pascals.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return tangents  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_corotated_tangent(deformation_gradients: "np.ndarray", shear_modulus: float,
                                       lame_first: float) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    gradients = np.asarray(deformation_gradients, dtype=float)
    if gradients.ndim != 3 or gradients.shape[1:] != (2, 2):
        raise ValueError("deformation_gradients must have shape (n_particles, 2, 2)")
    if gradients.shape[0] == 0:
        raise ValueError("deformation_gradients must hold at least one particle")
    if not np.all(np.isfinite(gradients)):
        raise ValueError("deformation_gradients must contain only finite entries")
    if not (isinstance(shear_modulus, (int, float)) and np.isfinite(shear_modulus)
            and float(shear_modulus) > 0.0):
        raise ValueError("shear_modulus must be a finite number > 0")
    if not (isinstance(lame_first, (int, float)) and np.isfinite(lame_first)
            and float(lame_first) >= 0.0):
        raise ValueError("lame_first must be a finite number >= 0")

    shear = float(shear_modulus)
    lame = float(lame_first)
    n_particles = gradients.shape[0]
    determinant = (gradients[:, 0, 0] * gradients[:, 1, 1]
                   - gradients[:, 0, 1] * gradients[:, 1, 0])
    if np.any(determinant <= 0.0):
        raise ValueError("every deformation gradient must have a determinant > 0")

    trace_part = gradients[:, 0, 0] + gradients[:, 1, 1]
    spin_part = gradients[:, 0, 1] - gradients[:, 1, 0]
    scale = np.sqrt(trace_part * trace_part + spin_part * spin_part)
    if np.any(scale <= 0.0):
        raise ValueError("a deformation gradient has no admissible polar rotation")
    rotation = np.empty_like(gradients)
    rotation[:, 0, 0] = trace_part / scale
    rotation[:, 0, 1] = spin_part / scale
    rotation[:, 1, 0] = -spin_part / scale
    rotation[:, 1, 1] = trace_part / scale

    # -- Derivative of the polar rotation. Writing the rotation increment as an
    #    angle times the rotation turned by a quarter turn, the angle increment
    #    follows from the antisymmetric part of the rotation transpose times the
    #    increment of the deformation gradient, divided by the trace of the
    #    symmetric stretch.
    stretch_trace = np.einsum('pba,pba->p', rotation, gradients)
    if np.any(stretch_trace <= 0.0):
        raise ValueError("a deformation gradient has a non-positive stretch trace")
    quarter_turn = np.stack([rotation[:, :, 1], -rotation[:, :, 0]], axis=2)
    angle_derivative = np.empty((n_particles, 2, 2))
    angle_derivative[:, :, 0] = rotation[:, :, 1] / stretch_trace[:, None]
    angle_derivative[:, :, 1] = -rotation[:, :, 0] / stretch_trace[:, None]
    rotation_derivative = np.einsum('pab,pcd->pabcd', quarter_turn, angle_derivative)

    identity = np.zeros((2, 2, 2, 2))
    for first in range(2):
        for second in range(2):
            identity[first, second, first, second] = 1.0

    cofactor = np.empty_like(gradients)
    cofactor[:, 0, 0] = gradients[:, 1, 1]
    cofactor[:, 0, 1] = -gradients[:, 1, 0]
    cofactor[:, 1, 0] = -gradients[:, 0, 1]
    cofactor[:, 1, 1] = gradients[:, 0, 0]

    # -- Derivative of the cofactor, which in two dimensions is a constant
    #    fourth-order array exchanging the two diagonal and the two
    #    off-diagonal entries with a sign.
    cofactor_derivative = np.zeros((2, 2, 2, 2))
    cofactor_derivative[0, 0, 1, 1] = 1.0
    cofactor_derivative[0, 1, 1, 0] = -1.0
    cofactor_derivative[1, 0, 0, 1] = -1.0
    cofactor_derivative[1, 1, 0, 0] = 1.0

    tangents = 2.0 * shear * (identity[None] - rotation_derivative)
    tangents = tangents + lame * np.einsum('pab,pcd->pabcd', cofactor, cofactor)
    tangents = tangents + lame * (determinant - 1.0)[:, None, None, None, None] * cofactor_derivative[None]
    return tangents

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    import numpy as np
    """Return list of test case specifications."""
    # Every case returns a plain float: the valid cases reduce the returned arrays
    # through a position weighted digest, the invalid cases return a status code.
    return [
        # --- Valid: benchmark material over a spread of distortions (normal scenario) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(51)
gradients = np.eye(2)[None] + 0.25 * rng.standard_normal((9, 2, 2))
shear, lame = 43200.0, 148800.0

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "pin(evaluate_corotated_tangent(gradients, shear, lame))",
            "gold_call": "pin(_oracle_evaluate_corotated_tangent(gradients, shear, lame))",
        },
        # --- Valid: the tangent against a central difference of the stress ---
        # The tangent must reproduce the derivative of the fixed corotated stress,
        # and it must be symmetric under exchange of its two index pairs.
        {
            "setup": """import numpy as np
rng = np.random.default_rng(52)
gradients = np.eye(2)[None] + 0.2 * rng.standard_normal((6, 2, 2))
shear, lame = 43200.0, 148800.0
nudge = 1.0e-6

def stress(F):
    trace = F[:, 0, 0] + F[:, 1, 1]
    spin = F[:, 0, 1] - F[:, 1, 0]
    scale = np.sqrt(trace ** 2 + spin ** 2)
    R = np.empty_like(F)
    R[:, 0, 0] = trace / scale
    R[:, 0, 1] = spin / scale
    R[:, 1, 0] = -spin / scale
    R[:, 1, 1] = trace / scale
    C = np.empty_like(F)
    C[:, 0, 0] = F[:, 1, 1]
    C[:, 0, 1] = -F[:, 1, 0]
    C[:, 1, 0] = -F[:, 0, 1]
    C[:, 1, 1] = F[:, 0, 0]
    J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    return 2.0 * shear * (F - R) + lame * (J - 1.0)[:, None, None] * C

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def consistency(fn):
    exact = fn(gradients, shear, lame)
    numeric = np.zeros_like(exact)
    for c in range(2):
        for d in range(2):
            up = gradients.copy(); up[:, c, d] += nudge
            down = gradients.copy(); down[:, c, d] -= nudge
            numeric[:, :, :, c, d] = (stress(up) - stress(down)) / (2.0 * nudge)
    swapped = np.einsum('pabcd->pcdab', exact)
    flags = float(int(np.abs(exact - numeric).max() < 1.0e-6 * np.abs(numeric).max())
                  + 2 * int(np.abs(exact - swapped).max() < 1.0e-9 * np.abs(exact).max()))
    return flags + pin(exact) + 3.0 * pin(numeric)
""",
            "call": "consistency(evaluate_corotated_tangent)",
            "gold_call": "consistency(_oracle_evaluate_corotated_tangent)",
        },
        # --- Boundary: the undeformed state, where the tangent takes its reference value ---
        {
            "setup": """import numpy as np
gradients = np.tile(np.eye(2), (3, 1, 1))
shear, lame = 43200.0, 148800.0

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "pin(evaluate_corotated_tangent(gradients, shear, lame))",
            "gold_call": "pin(_oracle_evaluate_corotated_tangent(gradients, shear, lame))",
        },
        # --- Valid: the isochoric pre-strain of the benchmark under a rigid rotation ---
        {
            "setup": """import numpy as np
angle = 0.61
turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
stretch = np.array([1.20, 1.35])
gradients = np.stack([turn @ np.diag([s, 1.0 / s]) for s in stretch])
shear, lame = 43200.0, 148800.0

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "pin(evaluate_corotated_tangent(gradients, shear, lame))",
            "gold_call": "pin(_oracle_evaluate_corotated_tangent(gradients, shear, lame))",
        },
        # --- Edge: a nearly incompressible material with a large first Lame constant ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(53)
gradients = np.eye(2)[None] + 0.15 * rng.standard_normal((5, 2, 2))
shear, lame = 43200.0, 4.32e7

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "pin(evaluate_corotated_tangent(gradients, shear, lame))",
            "gold_call": "pin(_oracle_evaluate_corotated_tangent(gradients, shear, lame))",
        },
        # --- Invalid: a deformation gradient array of the wrong rank ---
        {
            "setup": """import numpy as np
gradients = np.eye(2)
def run_model():
    try:
        evaluate_corotated_tangent(gradients, 43200.0, 148800.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_corotated_tangent(gradients, 43200.0, 148800.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative first Lame constant ---
        {
            "setup": """import numpy as np
gradients = np.tile(np.eye(2), (2, 1, 1))
def run_model():
    try:
        evaluate_corotated_tangent(gradients, 43200.0, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_corotated_tangent(gradients, 43200.0, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
