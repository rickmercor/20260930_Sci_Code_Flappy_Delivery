"""
Assemble and statically condense one degraded mechanics cell.

The local elastic form is the quadrature sum of



$$2\mu\,E(\mathbf{u}):E(\mathbf{v}) +\lambda\,\operatorname{tr}(E(\mathbf{u})) \operatorname{tr}(E(\mathbf{v}))$$



plus the stabilization contribution $2\mu S_T$. The degradation $g(\phi_T)=(1-\phi_T)^2$ multiplies the complete local form. With zero cell load, eliminating the first $n_{\mathrm{cell}}$ coefficients gives a face Schur matrix and a linear cell-recovery operator.

Returns
-------
np.ndarray stacking the face Schur matrix above the zero-load cell recovery map
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def condense_degraded_mechanics(
    strain_reconstruction: np.ndarray,
    quadrature_weights: np.ndarray,
    stabilization: np.ndarray,
    cell_phase: float,
    lame_lambda: float,
    shear_modulus: float,
    n_cell: int = 6,
) -> np.ndarray:
    r"""Return the condensed matrix stacked above the cell recovery matrix.

    Parameters
    ----------
    strain_reconstruction : np.ndarray, shape (n_q, 4, n_dof)
        Local strain matrices in $(xx,yy,zz,xy)$ order.
    quadrature_weights : np.ndarray, shape (n_q,)
        Finite positive cell quadrature weights.
    stabilization : np.ndarray, shape (n_dof, n_dof)
        Finite symmetric positive-semidefinite local stabilization.
    cell_phase : float
        Finite phase value in $[0,1)$.
    lame_lambda : float
        Finite Lamé first parameter.
    shear_modulus : float
        Finite positive shear modulus.
    n_cell : int, optional
        Number of leading cell coefficients.

    Returns
    -------
    np.ndarray, shape (n_dof, n_dof - n_cell)
        The face Schur matrix in the first $n_{\mathrm{face}}$ rows and the
        recovery map $\mathbf{u}_C=R\mathbf{u}_F$ in the last
        $n_{\mathrm{cell}}$ rows.

    Raises
    ------
    ValueError
        If input contracts, symmetry, positivity, or the cell solve fail.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_condense_degraded_mechanics(
    strain_reconstruction,
    quadrature_weights,
    stabilization,
    cell_phase,
    lame_lambda,
    shear_modulus,
    n_cell=6,
):
    """Reference degraded assembly and cell-block Schur complement."""
    

    strain = np.asarray(strain_reconstruction, dtype=float)
    weights = np.asarray(quadrature_weights, dtype=float)
    stabilization = np.asarray(stabilization, dtype=float)
    if strain.ndim != 3 or strain.shape[1] != 4 or strain.shape[0] < 1:
        raise ValueError("strain_reconstruction must have shape (n_q, 4, n_dof)")
    n_dof = strain.shape[2]
    if weights.shape != (strain.shape[0],) or np.any(weights <= 0.0):
        raise ValueError("quadrature_weights must be positive with shape (n_q,)")
    if stabilization.shape != (n_dof, n_dof):
        raise ValueError("stabilization shape must match the local dof count")
    if (
        not np.all(np.isfinite(strain))
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(stabilization))
    ):
        raise ValueError("array inputs must be finite")
    if not np.allclose(stabilization, stabilization.T, atol=1e-12, rtol=0.0):
        raise ValueError("stabilization must be symmetric")
    if np.min(np.linalg.eigvalsh(stabilization)) < -1e-10:
        raise ValueError("stabilization must be positive semidefinite")
    if not np.isfinite(cell_phase) or cell_phase < 0.0 or cell_phase >= 1.0:
        raise ValueError("cell_phase must lie in [0, 1)")
    if not np.isfinite(lame_lambda) or not np.isfinite(shear_modulus):
        raise ValueError("elastic parameters must be finite")
    if shear_modulus <= 0.0 or lame_lambda + 2.0 * shear_modulus / 3.0 <= 0.0:
        raise ValueError(
            "elastic parameters must define positive bulk and shear moduli"
        )
    if not isinstance(n_cell, (int, np.integer)) or not 0 < n_cell < n_dof:
        raise ValueError("n_cell must be an integer strictly between zero and n_dof")

    elasticity = np.zeros((4, 4))
    elasticity[0:3, 0:3] = lame_lambda
    elasticity[0, 0] += 2.0 * shear_modulus
    elasticity[1, 1] += 2.0 * shear_modulus
    elasticity[2, 2] += 2.0 * shear_modulus
    elasticity[3, 3] = 4.0 * shear_modulus
    local_matrix = np.zeros((n_dof, n_dof))
    for operator, weight in zip(strain, weights):
        local_matrix += weight * (operator.T @ elasticity @ operator)
    local_matrix += 2.0 * shear_modulus * stabilization
    local_matrix *= (1.0 - cell_phase) ** 2
    local_matrix = 0.5 * (local_matrix + local_matrix.T)

    cell_block = local_matrix[:n_cell, :n_cell]
    coupling = local_matrix[:n_cell, n_cell:]
    face_block = local_matrix[n_cell:, n_cell:]
    try:
        solved_coupling = np.linalg.solve(cell_block, coupling)
    except np.linalg.LinAlgError as error:
        raise ValueError("cell mechanics block is singular") from error
    schur = face_block - coupling.T @ solved_coupling
    recovery = -solved_coupling
    return np.vstack([0.5 * (schur + schur.T), recovery])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return undamaged, degraded, and invalid-phase cell cases."""
    return [
        {
            "setup": """import numpy as np
bounds = np.array([0.0, 0.5, 0.0, 1.0])
faces = np.array([[[0.0,0.0],[0.5,0.0]], [[0.5,0.0],[0.5,1.0]], [[0.5,1.0],[0.0,1.0]], [[0.0,1.0],[0.0,0.0]]])
normals = np.array([[0.0,-1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]])
B = _oracle_build_affine_strain_reconstruction(bounds, faces, normals)
S = _oracle_build_quadratic_stabilization(bounds, faces, normals, B)
w = np.full(4, 0.125)
""",
            "call": "condense_degraded_mechanics(B, w, S, 0.0, 121.15, 80.77)",
            "gold_call": "_oracle_condense_degraded_mechanics(B, w, S, 0.0, 121.15, 80.77)",
        },
        {
            "setup": """import numpy as np
bounds = np.array([0.5, 1.0, 0.0, 1.0])
faces = np.array([[[0.5,0.0],[1.0,0.0]], [[1.0,0.0],[1.0,1.0]], [[1.0,1.0],[0.5,1.0]], [[0.5,1.0],[0.5,0.0]]])
normals = np.array([[0.0,-1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]])
B = _oracle_build_affine_strain_reconstruction(bounds, faces, normals)
S = _oracle_build_quadratic_stabilization(bounds, faces, normals, B)
w = np.full(4, 0.125)
""",
            "call": "condense_degraded_mechanics(B, w, S, 0.6, 121.15, 80.77)",
            "gold_call": "_oracle_condense_degraded_mechanics(B, w, S, 0.6, 121.15, 80.77)",
        },
        {
            "setup": """import numpy as np
B = np.zeros((1, 4, 2))
w = np.ones(1)
S = np.eye(2)
def run_model():
    try:
        condense_degraded_mechanics(B, w, S, 1.0, 1.0, 1.0, n_cell=1)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_condense_degraded_mechanics(B, w, S, 1.0, 1.0, 1.0, n_cell=1)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
