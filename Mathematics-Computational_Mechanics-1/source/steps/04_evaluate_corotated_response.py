"""
Evaluate the fixed corotated strain energy density and the first Piola-Kirchhoff stress of every material point from its deformation gradient.

The fixed corotated model measures distortion against the rotation carried by the polar decomposition of the deformation gradient, so it is frame indifferent and reduces to linear elasticity at small strain with the Lame constants as its parameters. Its stress separates into a term proportional to the departure of the deformation gradient from that rotation and a volumetric term proportional to the cofactor.

Returns
-------
tuple of two np.ndarray: strain-energy densities of shape (n_particles,) and first Piola-Kirchhoff stresses of shape (n_particles, 2, 2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_corotated_response(deformation_gradients: "np.ndarray", shear_modulus: float,
                                lame_first: float) -> tuple:
    """Evaluate the fixed corotated energy density and first Piola stress.

    With ``R`` the rotation of the polar decomposition of the deformation
    gradient ``F`` and ``J`` its determinant, the strain energy density is

        ``psi = mu * ||F - R||_F^2 + (lambda / 2) * (J - 1)^2``,

    where ``||.||_F`` is the Frobenius norm, ``mu`` is the shear modulus and
    ``lambda`` is the first Lame constant. The first Piola-Kirchhoff stress is
    the derivative of that density with respect to the deformation gradient.

    In two dimensions the rotation is fixed by the requirement that ``R`` be
    orthogonal with unit determinant and that ``R^T F`` be symmetric, which
    determines it uniquely whenever the determinant of the deformation
    gradient is strictly positive.

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
    energy_densities : "np.ndarray"
        Array of shape (n_particles,) holding the strain energy density of
        every particle, in joules per cubic metre.
    first_piola : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the first Piola-Kirchhoff
        stress of every particle, in pascals.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return energy_densities, first_piola  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_corotated_response(deformation_gradients: "np.ndarray", shear_modulus: float,
                                        lame_first: float) -> tuple:
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
    determinant = (gradients[:, 0, 0] * gradients[:, 1, 1]
                   - gradients[:, 0, 1] * gradients[:, 1, 0])
    if np.any(determinant <= 0.0):
        raise ValueError("every deformation gradient must have a determinant > 0")

    # -- Rotation of the polar decomposition. In two dimensions the symmetric
    #    part of the trace and the antisymmetric part of the off-diagonal fix
    #    the rotation angle without any eigenvalue computation.
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

    # -- Cofactor, which is the derivative of the determinant.
    cofactor = np.empty_like(gradients)
    cofactor[:, 0, 0] = gradients[:, 1, 1]
    cofactor[:, 0, 1] = -gradients[:, 1, 0]
    cofactor[:, 1, 0] = -gradients[:, 0, 1]
    cofactor[:, 1, 1] = gradients[:, 0, 0]

    departure = gradients - rotation
    energy_densities = (shear * np.sum(departure * departure, axis=(1, 2))
                        + 0.5 * lame * (determinant - 1.0) ** 2)
    first_piola = (2.0 * shear * departure
                   + lame * (determinant - 1.0)[:, None, None] * cofactor)
    return energy_densities, first_piola

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
rng = np.random.default_rng(41)
gradients = np.eye(2)[None] + 0.25 * rng.standard_normal((9, 2, 2))
shear, lame = 43200.0, 148800.0

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def pin_all(parts):
    total = 1.0
    for order, part in enumerate(parts):
        total *= (1.0 + pin(part)) ** (1.0 / (order + 1.0))
    return float(total)
""",
            "call": "pin_all(evaluate_corotated_response(gradients, shear, lame))",
            "gold_call": "pin_all(_oracle_evaluate_corotated_response(gradients, shear, lame))",
        },
        # --- Valid: the isochoric pre-strain of the benchmark, whose volumetric term vanishes ---
        {
            "setup": """import numpy as np
stretch = np.array([1.05, 1.20, 1.40])
gradients = np.zeros((3, 2, 2))
gradients[:, 0, 0] = stretch
gradients[:, 1, 1] = 1.0 / stretch
shear, lame = 43200.0, 148800.0

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def pin_all(parts):
    total = 1.0
    for order, part in enumerate(parts):
        total *= (1.0 + pin(part)) ** (1.0 / (order + 1.0))
    return float(total)
""",
            "call": "pin_all(evaluate_corotated_response(gradients, shear, lame))",
            "gold_call": "pin_all(_oracle_evaluate_corotated_response(gradients, shear, lame))",
        },
        # --- Valid: frame indifference and the stress-free reference state ---
        # A rigid rotation of the deformation gradient leaves the energy density
        # unchanged and rotates the stress with it, and the identity carries no
        # energy and no stress at all.
        {
            "setup": """import numpy as np
rng = np.random.default_rng(42)
gradients = np.eye(2)[None] + 0.2 * rng.standard_normal((7, 2, 2))
shear, lame = 43200.0, 148800.0
angle = 0.83
turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def indifference(fn):
    plain, stress = fn(gradients, shear, lame)
    turned, turned_stress = fn(turn @ gradients, shear, lame)
    rest, rest_stress = fn(np.tile(np.eye(2), (2, 1, 1)), shear, lame)
    flags = float(int(np.abs(turned - plain).max() < 1.0e-9 * np.abs(plain).max())
                  + 2 * int(np.abs(turned_stress - turn @ stress).max()
                            < 1.0e-9 * np.abs(stress).max())
                  + 4 * int(np.abs(rest).max() < 1.0e-18 and np.abs(rest_stress).max() < 1.0e-9))
    return flags + pin(plain) + 3.0 * pin(stress) + 5.0 * pin(turned_stress)
""",
            "call": "indifference(evaluate_corotated_response)",
            "gold_call": "indifference(_oracle_evaluate_corotated_response)",
        },
        # --- Boundary: a vanishing first Lame constant, so only the shear term survives ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(43)
gradients = np.eye(2)[None] + 0.3 * rng.standard_normal((5, 2, 2))
shear, lame = 43200.0, 0.0

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def pin_all(parts):
    total = 1.0
    for order, part in enumerate(parts):
        total *= (1.0 + pin(part)) ** (1.0 / (order + 1.0))
    return float(total)
""",
            "call": "pin_all(evaluate_corotated_response(gradients, shear, lame))",
            "gold_call": "pin_all(_oracle_evaluate_corotated_response(gradients, shear, lame))",
        },
        # --- Edge: strongly compressed and strongly rotated states ---
        {
            "setup": """import numpy as np
angle = 1.9
turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
gradients = np.stack([turn @ np.diag([0.55, 0.62]),
                      turn @ np.array([[1.8, 0.4], [-0.3, 0.9]]),
                      np.diag([0.999, 1.001])])
shear, lame = 43200.0, 148800.0

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def pin_all(parts):
    total = 1.0
    for order, part in enumerate(parts):
        total *= (1.0 + pin(part)) ** (1.0 / (order + 1.0))
    return float(total)
""",
            "call": "pin_all(evaluate_corotated_response(gradients, shear, lame))",
            "gold_call": "pin_all(_oracle_evaluate_corotated_response(gradients, shear, lame))",
        },
        # --- Invalid: an inverted element, whose determinant is negative ---
        {
            "setup": """import numpy as np
gradients = np.stack([np.eye(2), np.diag([-1.0, 1.0])])
def run_model():
    try:
        evaluate_corotated_response(gradients, 43200.0, 148800.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_corotated_response(gradients, 43200.0, 148800.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive shear modulus ---
        {
            "setup": """import numpy as np
gradients = np.tile(np.eye(2), (2, 1, 1))
def run_model():
    try:
        evaluate_corotated_response(gradients, 0.0, 148800.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_corotated_response(gradients, 0.0, 148800.0)
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
