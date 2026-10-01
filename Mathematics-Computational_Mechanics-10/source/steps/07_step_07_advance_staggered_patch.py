"""
Advance the fixed two-cell patch by one staggered load increment.

One classical staggered sweep first solves degraded mechanics from the phase available at the start of the increment, then updates the irreversible volumetric-deviatoric history, and finally solves the phase equation. Cell unknowns are eliminated locally in both solves and recovered after the global face systems are solved. The signed load is imposed on the constant horizontal mode of the left face; positive load means outward tension.

Returns
-------
np.ndarray of shape (18,) containing phase[9], history.ravel()[8], and normalized reaction
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def advance_staggered_patch(
    displacement_load: float,
    previous_phase: np.ndarray,
    previous_history: np.ndarray,
    material: np.ndarray,
) -> np.ndarray:
    r"""Advance one load increment on the prescribed two-cell plane-strain patch.

    Parameters
    ----------
    displacement_load : float
        Finite signed left-boundary displacement magnitude in millimetres;
        the imposed constant horizontal coefficient is its negative.
    previous_phase : np.ndarray, shape (9,)
        Two cell values followed by global face values $F_0$ through $F_6$.
    previous_history : np.ndarray, shape (2, 4)
        Cell-by-quadrature nonnegative tensile-history values.
    material : np.ndarray, shape (6,)
        $[\lambda,\mu,G_c,\ell,\eta,\Delta t]$ in the stated unit system.

    Returns
    -------
    np.ndarray, shape (18,)
        Updated phase values, flattened updated history, and the normalized
        mechanics reaction from this sweep, in that order.

    Raises
    ------
    ValueError
        If the state or material contracts fail, or a condensed global solve
        is singular or produces a nonfinite value.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_advance_staggered_patch(
    displacement_load, previous_phase, previous_history, material
):
    """Reference one-sweep patch advance using the preceding oracle steps."""
    

    phase = np.asarray(previous_phase, dtype=float)
    history = np.asarray(previous_history, dtype=float)
    material = np.asarray(material, dtype=float)
    if not np.isscalar(displacement_load) or not np.isfinite(displacement_load):
        raise ValueError("displacement_load must be a finite scalar")
    if phase.shape != (9,) or not np.all(np.isfinite(phase)):
        raise ValueError("previous_phase must be finite with shape (9,)")
    if np.any(phase < 0.0) or np.any(phase >= 1.0):
        raise ValueError("previous_phase values must lie in [0, 1)")
    if history.shape != (2, 4) or not np.all(np.isfinite(history)):
        raise ValueError("previous_history must be finite with shape (2, 4)")
    if np.any(history < 0.0):
        raise ValueError("previous_history must be nonnegative")
    if material.shape != (6,) or not np.all(np.isfinite(material)):
        raise ValueError("material must be finite with shape (6,)")
    lame_lambda, shear_modulus, toughness, length_scale, viscosity, time_step = material
    if shear_modulus <= 0.0 or lame_lambda + 2.0 * shear_modulus / 3.0 <= 0.0:
        raise ValueError("material must define positive bulk and shear moduli")
    if toughness <= 0.0 or length_scale <= 0.0 or time_step <= 0.0:
        raise ValueError("Gc, ell, and dt must be positive")
    if viscosity < 0.0:
        raise ValueError("eta must be nonnegative")

    global_faces = np.array(
        [
            [[0.0, 0.0], [0.5, 0.0]],
            [[0.5, 0.0], [0.5, 1.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.5, 1.0]],
            [[0.5, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.5, 1.0], [0.0, 1.0]],
        ]
    )
    cell_bounds = [
        np.array([0.0, 0.5, 0.0, 1.0]),
        np.array([0.5, 1.0, 0.0, 1.0]),
    ]
    cell_face_ids = [np.array([0, 1, 6, 5]), np.array([4, 2, 3, 1])]
    normals = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    weights = np.full(4, 0.125)

    strain_operators = []
    mechanics_data = []
    phase_operators = []
    for cell in range(2):
        local_faces = global_faces[cell_face_ids[cell]]
        strain_operator = _oracle_build_affine_strain_reconstruction(
            cell_bounds[cell], local_faces, normals
        )
        stabilization = _oracle_build_quadratic_stabilization(
            cell_bounds[cell], local_faces, normals, strain_operator
        )
        mechanics = _oracle_condense_degraded_mechanics(
            strain_operator,
            weights,
            stabilization,
            phase[cell],
            lame_lambda,
            shear_modulus,
        )
        phase_operator = _oracle_build_phase_reconstruction(
            cell_bounds[cell], local_faces, normals
        )
        strain_operators.append(strain_operator)
        mechanics_data.append(mechanics)
        phase_operators.append(phase_operator)

    face_mechanics = np.zeros((28, 28))
    for cell in range(2):
        local_global = np.concatenate(
            [4 * face + np.arange(4) for face in cell_face_ids[cell]]
        )
        face_mechanics[np.ix_(local_global, local_global)] += mechanics_data[cell][0:16]

    face_displacement = np.zeros(28)
    left_dofs = 4 * 5 + np.arange(4)
    right_dofs = 4 * 2 + np.arange(4)
    fixed_dofs = np.concatenate([left_dofs, right_dofs])
    free_dofs = np.setdiff1d(np.arange(28), fixed_dofs)
    face_displacement[left_dofs[0]] = -float(displacement_load)
    try:
        face_displacement[free_dofs] = np.linalg.solve(
            face_mechanics[np.ix_(free_dofs, free_dofs)],
            -face_mechanics[np.ix_(free_dofs, fixed_dofs)]
            @ face_displacement[fixed_dofs],
        )
    except np.linalg.LinAlgError as error:
        raise ValueError("condensed mechanics system is singular") from error
    mechanics_residual = face_mechanics @ face_displacement
    normalized_reaction = abs(mechanics_residual[left_dofs[0]]) / shear_modulus

    local_displacements = []
    updated_history = np.empty_like(history)
    for cell in range(2):
        local_global = np.concatenate(
            [4 * face + np.arange(4) for face in cell_face_ids[cell]]
        )
        local_face_values = face_displacement[local_global]
        local_cell_values = mechanics_data[cell][16:22] @ local_face_values
        local_displacement = np.concatenate([local_cell_values, local_face_values])
        local_displacements.append(local_displacement)
        updated_history[cell] = _oracle_update_volumetric_deviatoric_history(
            strain_operators[cell],
            local_displacement,
            history[cell],
            lame_lambda,
            shear_modulus,
        )

    face_phase_matrix = np.zeros((7, 7))
    face_phase_rhs = np.zeros(7)
    phase_data = []
    for cell in range(2):
        condensed_phase = _oracle_condense_phase_field(
            phase_operators[cell],
            weights,
            updated_history[cell],
            phase[cell],
            length_scale,
            toughness,
            viscosity,
            time_step,
        )
        ids = cell_face_ids[cell]
        face_phase_matrix[np.ix_(ids, ids)] += condensed_phase[0:4, 0:4]
        face_phase_rhs[ids] += condensed_phase[0:4, 4]
        phase_data.append(condensed_phase)
    try:
        updated_face_phase = np.linalg.solve(face_phase_matrix, face_phase_rhs)
    except np.linalg.LinAlgError as error:
        raise ValueError("condensed phase system is singular") from error

    updated_cell_phase = np.empty(2)
    for cell in range(2):
        recovery = phase_data[cell][4]
        updated_cell_phase[cell] = (
            recovery[0:4] @ updated_face_phase[cell_face_ids[cell]] + recovery[4]
        )
    updated_phase = np.concatenate([updated_cell_phase, updated_face_phase])
    result = np.concatenate(
        [updated_phase, updated_history.ravel(), [normalized_reaction]]
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("the staggered update produced a nonfinite result")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return tensile, reverse-load, viscous, and invalid-state cases."""
    return [
        {
            "setup": """import numpy as np
load = 0.002
phase = np.zeros(9)
history = np.zeros((2, 4))
material = np.array([121.15, 80.77, 2.7e-3, 0.0075, 0.0, 1.0])
""",
            "call": "advance_staggered_patch(load, phase, history, material)",
            "gold_call": "_oracle_advance_staggered_patch(load, phase, history, material)",
        },
        {
            "setup": """import numpy as np
material = np.array([121.15, 80.77, 2.7e-3, 0.0075, 0.0, 1.0])
phase = np.zeros(9)
history = np.zeros((2, 4))
# State after the unchanged tensile preload sequence [0.002, 0.020].
phase = np.array([0.20523297478873131, 0.20523297478873134, 0.2052329747887314, 0.2052329747887314, 0.20523297478873143, 0.2052329747887314, 0.2052329747887314, 0.2052329747887314, 0.20523297478873137], dtype=np.float64)
history = np.array([[0.04376957407407111, 0.04376957407407109, 0.04919335541372836, 0.04919335541372834], [0.04919335541372835, 0.04919335541372836, 0.043769574074071046, 0.043769574074071046]], dtype=np.float64)
load = -0.035
""",
            "call": "advance_staggered_patch(load, phase, history, material)",
            "gold_call": "_oracle_advance_staggered_patch(load, phase, history, material)",
        },
        {
            "setup": """import numpy as np
load = 0.01
phase = np.full(9, 0.05)
history = np.full((2, 4), 0.002)
material = np.array([100.0, 60.0, 3.0e-3, 0.01, 1.0e-5, 0.5])
""",
            "call": "advance_staggered_patch(load, phase, history, material)",
            "gold_call": "_oracle_advance_staggered_patch(load, phase, history, material)",
        },
        {
            "setup": """import numpy as np
load = 0.002
phase = np.zeros(9)
phase[0] = 1.0
history = np.zeros((2, 4))
material = np.array([121.15, 80.77, 2.7e-3, 0.0075, 0.0, 1.0])
def run_model():
    try:
        advance_staggered_patch(load, phase, history, material)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_advance_staggered_patch(load, phase, history, material)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
