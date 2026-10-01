"""
Assemble and statically condense one local phase-field equation.

The local phase form combines the reconstructed-gradient term, the reaction $\ell^{-2}+2H/(\ell G_c)$, the face-average jump matrix, and the optional backward-Euler coefficient $\eta/(\ell G_c\,\Delta t)$. Only the constant cell phase enters the reaction and source terms. Eliminating it produces a four-face Schur system and an affine recovery rule for the cell value.

Returns
-------
np.ndarray of shape (5, 5) packing the four-face system and cell recovery rule
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def condense_phase_field(
    phase_reconstruction: np.ndarray,
    quadrature_weights: np.ndarray,
    history: np.ndarray,
    previous_cell_phase: float,
    length_scale: float,
    fracture_toughness: float,
    viscosity: float = 0.0,
    time_step: float = 1.0,
) -> np.ndarray:
    r"""Return the packed condensed phase system and cell recovery rule.

    Parameters
    ----------
    phase_reconstruction : np.ndarray, shape (7, 5)
        Gradient rows followed by the five-by-five jump matrix.
    quadrature_weights : np.ndarray, shape (n_q,)
        Finite positive cell quadrature weights.
    history : np.ndarray, shape (n_q,)
        Finite nonnegative quadrature-node history values.
    previous_cell_phase : float
        Finite previous cell phase in $[0,1]$.
    length_scale : float
        Finite positive regularization length.
    fracture_toughness : float
        Finite positive critical energy-release rate.
    viscosity : float, optional
        Finite nonnegative viscous coefficient.
    time_step : float, optional
        Finite positive pseudo-time increment.

    Returns
    -------
    np.ndarray, shape (5, 5)
        Rows $0{:}4$ contain $[A^{\mathrm{sc}}\mid\mathbf{b}^{\mathrm{sc}}]$;
        the last row stores $\mathbf{r}$ followed by $c$ in the recovery rule
        $\phi_T=\mathbf{r}^{T}\boldsymbol{\phi}_F+c$.

    Raises
    ------
    ValueError
        If any shape, finiteness, sign, or symmetry contract is violated.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_condense_phase_field(
    phase_reconstruction,
    quadrature_weights,
    history,
    previous_cell_phase,
    length_scale,
    fracture_toughness,
    viscosity=0.0,
    time_step=1.0,
):
    """Reference diffusion-reaction assembly and scalar cell elimination."""
    

    operator = np.asarray(phase_reconstruction, dtype=float)
    weights = np.asarray(quadrature_weights, dtype=float)
    history = np.asarray(history, dtype=float)
    if operator.shape != (7, 5) or not np.all(np.isfinite(operator)):
        raise ValueError("phase_reconstruction must be finite with shape (7, 5)")
    if weights.ndim != 1 or weights.shape[0] < 1 or np.any(weights <= 0.0):
        raise ValueError("quadrature_weights must be a nonempty positive vector")
    if history.shape != weights.shape:
        raise ValueError("history and quadrature_weights must have matching shapes")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(history)):
        raise ValueError("quadrature data must be finite")
    if np.any(history < 0.0):
        raise ValueError("history must be nonnegative")
    if not np.isfinite(previous_cell_phase) or not 0.0 <= previous_cell_phase <= 1.0:
        raise ValueError("previous_cell_phase must lie in [0, 1]")
    parameters = [length_scale, fracture_toughness, viscosity, time_step]
    if not np.all(np.isfinite(parameters)):
        raise ValueError("scalar parameters must be finite")
    if length_scale <= 0.0 or fracture_toughness <= 0.0 or time_step <= 0.0:
        raise ValueError(
            "length_scale, fracture_toughness, and time_step must be positive"
        )
    if viscosity < 0.0:
        raise ValueError("viscosity must be nonnegative")

    gradient = operator[0:2]
    jump = operator[2:7]
    if not np.allclose(jump, jump.T, atol=1e-12, rtol=0.0):
        raise ValueError("the packed jump matrix must be symmetric")
    area = float(np.sum(weights))
    cell_selector = np.zeros(5)
    cell_selector[0] = 1.0
    reaction = np.sum(
        weights
        * (1.0 / length_scale**2 + 2.0 * history / (length_scale * fracture_toughness))
    )
    source = np.sum(weights * (2.0 * history / (length_scale * fracture_toughness)))
    viscous_coefficient = (
        area * viscosity / (length_scale * fracture_toughness * time_step)
    )
    local_matrix = area * (gradient.T @ gradient) + jump
    local_matrix += (reaction + viscous_coefficient) * np.outer(
        cell_selector, cell_selector
    )
    local_rhs = (source + viscous_coefficient * previous_cell_phase) * cell_selector

    cell_diagonal = local_matrix[0, 0]
    if not np.isfinite(cell_diagonal) or cell_diagonal <= 0.0:
        raise ValueError("phase cell block must be positive")
    coupling = local_matrix[0, 1:]
    schur = local_matrix[1:, 1:] - np.outer(coupling, coupling) / cell_diagonal
    condensed_rhs = local_rhs[1:] - coupling * local_rhs[0] / cell_diagonal
    recovery_coefficients = -coupling / cell_diagonal
    recovery_constant = local_rhs[0] / cell_diagonal
    packed = np.empty((5, 5))
    packed[0:4, 0:4] = 0.5 * (schur + schur.T)
    packed[0:4, 4] = condensed_rhs
    packed[4, 0:4] = recovery_coefficients
    packed[4, 4] = recovery_constant
    return packed

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return rate-independent, viscous, and invalid-history phase cases."""
    return [
        {
            "setup": """import numpy as np
bounds = np.array([0.0, 0.5, 0.0, 1.0])
faces = np.array([[[0.0,0.0],[0.5,0.0]], [[0.5,0.0],[0.5,1.0]], [[0.5,1.0],[0.0,1.0]], [[0.0,1.0],[0.0,0.0]]])
normals = np.array([[0.0,-1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]])
P = _oracle_build_phase_reconstruction(bounds, faces, normals)
w = np.full(4, 0.125)
H = np.array([0.01, 0.02, 0.015, 0.025])
""",
            "call": "condense_phase_field(P, w, H, 0.1, 0.0075, 2.7e-3)",
            "gold_call": "_oracle_condense_phase_field(P, w, H, 0.1, 0.0075, 2.7e-3)",
        },
        {
            "setup": """import numpy as np
bounds = np.array([0.5, 1.0, 0.0, 1.0])
faces = np.array([[[0.5,0.0],[1.0,0.0]], [[1.0,0.0],[1.0,1.0]], [[1.0,1.0],[0.5,1.0]], [[0.5,1.0],[0.5,0.0]]])
normals = np.array([[0.0,-1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]])
P = _oracle_build_phase_reconstruction(bounds, faces, normals)
w = np.full(4, 0.125)
H = np.zeros(4)
""",
            "call": "condense_phase_field(P, w, H, 0.4, 0.01, 0.003, 1e-5, 0.25)",
            "gold_call": "_oracle_condense_phase_field(P, w, H, 0.4, 0.01, 0.003, 1e-5, 0.25)",
        },
        {
            "setup": """import numpy as np
P = np.zeros((7, 5))
w = np.ones(2)
H = np.array([0.0, -1.0])
def run_model():
    try:
        condense_phase_field(P, w, H, 0.0, 0.01, 0.003)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_condense_phase_field(P, w, H, 0.0, 0.01, 0.003)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
