"""
Run the full load path and return the final normalized reaction.

The phase and history produced by one staggered sweep become the state of the next pseudo-time increment. This state transfer is essential on unloading and reverse loading: resetting the history would allow the phase to heal and would change the later mechanics solve. The returned scalar is $R_n/(\mu A)$ with $A=1\,\mathrm{mm}^2$.

Returns
-------
one finite float equal to the normalized reaction from the last load increment
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_irreversible_patch_response(
    load_path: np.ndarray | None = None,
    material: np.ndarray | None = None,
) -> float:
    r"""Return the final normalized reaction for the fixed two-cell patch.

    Parameters
    ----------
    load_path : np.ndarray, shape (n_steps,), optional
        Nonempty finite signed displacement sequence in millimetres. ``None``
        selects $[0.002,0.020,-0.035,0.004,0.015]\,\mathrm{mm}$.
    material : np.ndarray, shape (6,), optional
        $[\lambda,\mu,G_c,\ell,\eta,\Delta t]$ in the stated unit system.
        ``None`` selects $[121.15,80.77,2.7\times10^{-3},0.0075,0,1]$.

    Returns
    -------
    float
        Finite final normalized reaction after all increments.

    Raises
    ------
    ValueError
        If the load path or material is invalid, or any staggered step fails.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_irreversible_patch_response(load_path=None, material=None):
    """Reference load-path orchestration with irreversible state transfer."""
    

    if load_path is None:
        load_path = np.array([0.002, 0.020, -0.035, 0.004, 0.015])
    if material is None:
        material = np.array([121.15, 80.77, 2.7e-3, 0.0075, 0.0, 1.0])
    loads = np.asarray(load_path, dtype=float)
    material = np.asarray(material, dtype=float)
    if loads.ndim != 1 or loads.shape[0] < 1 or not np.all(np.isfinite(loads)):
        raise ValueError("load_path must be a nonempty finite vector")
    if material.shape != (6,) or not np.all(np.isfinite(material)):
        raise ValueError("material must be finite with shape (6,)")

    # Build and validate the fixed patch contract through every preceding
    # numerical stage before advancing its state. These values are also built
    # inside the one-increment step, but the direct chain keeps the final
    # orchestrator sensitive to every public stage in the ordered pipeline.
    bounds = np.array([0.0, 0.5, 0.0, 1.0])
    faces = np.array(
        [
            [[0.0, 0.0], [0.5, 0.0]],
            [[0.5, 0.0], [0.5, 1.0]],
            [[0.5, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ]
    )
    normals = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    weights = np.full(4, 0.125)
    strain = _oracle_build_affine_strain_reconstruction(bounds, faces, normals)
    stabilization = _oracle_build_quadratic_stabilization(
        bounds, faces, normals, strain
    )
    phase_operator = _oracle_build_phase_reconstruction(bounds, faces, normals)
    zero_history = _oracle_update_volumetric_deviatoric_history(
        strain, np.zeros(22), np.zeros(4), material[0], material[1]
    )
    mechanics_contract = _oracle_condense_degraded_mechanics(
        strain, weights, stabilization, 0.0, material[0], material[1]
    )
    phase_contract = _oracle_condense_phase_field(
        phase_operator,
        weights,
        zero_history,
        0.0,
        material[3],
        material[2],
        material[4],
        material[5],
    )
    contract_norms = np.array(
        [
            np.linalg.norm(strain),
            np.linalg.norm(stabilization),
            np.linalg.norm(phase_operator),
            np.linalg.norm(mechanics_contract),
            np.linalg.norm(phase_contract),
        ]
    )
    if not np.all(np.isfinite(contract_norms)):
        raise ValueError("the fixed patch contract produced a nonfinite operator")

    phase = np.zeros(9)
    history = np.zeros((2, 4))
    final_reaction = 0.0
    for load in loads:
        state = _oracle_advance_staggered_patch(load, phase, history, material)
        phase = state[0:9]
        history = state[9:17].reshape(2, 4)
        final_reaction = float(state[17])
    if not np.isfinite(final_reaction):
        raise ValueError("the final normalized reaction is not finite")
    return final_reaction

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return zero, single-step, two-step, and invalid path cases."""
    return [
        {
            "setup": """import numpy as np
loads = np.array([0.0])
material = np.array([121.15, 80.77, 2.7e-3, 0.0075, 0.0, 1.0])
""",
            "call": "compute_irreversible_patch_response(loads, material)",
            "gold_call": "_oracle_compute_irreversible_patch_response(loads, material)",
        },
        {
            "setup": """import numpy as np
loads = np.array([0.002])
material = np.array([121.15, 80.77, 2.7e-3, 0.0075, 0.0, 1.0])
""",
            "call": "compute_irreversible_patch_response(loads, material)",
            "gold_call": "_oracle_compute_irreversible_patch_response(loads, material)",
        },
        {
            "setup": """import numpy as np
loads = np.array([0.002, 0.020])
material = np.array([121.15, 80.77, 2.7e-3, 0.0075, 0.0, 1.0])
""",
            "call": "compute_irreversible_patch_response(loads, material)",
            "gold_call": "_oracle_compute_irreversible_patch_response(loads, material)",
        },
        {
            "setup": """import numpy as np
loads = np.array([])
material = np.array([121.15, 80.77, 2.7e-3, 0.0075, 0.0, 1.0])
def run_model():
    try:
        compute_irreversible_patch_response(loads, material)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_compute_irreversible_patch_response(loads, material)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
