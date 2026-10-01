"""
The complete one-interval regression chains every earlier step into one deterministic backward BSDE update: advancing the controlled states, evaluating the continuation tensor train, forming right-endpoint targets, building derivative-aware features, and performing two ALS micro-steps connected by the adaptive core shift. Advances controlled factor states, evaluates the continuation tensor train, forms right-endpoint BSDE targets, and builds the derivative-aware left-endpoint features. A middle-core ridge update is followed by residual-scaled regularization, an orthogonal core shift, and a last-core update. The resulting value gradient gives the feedback control, whose first component is the reported scalar.

Inputs
------
states, noises: Arrays of shape (K, 3).
drift_matrix, drift_bias: Affine drift parameters.
sigma_diag: Positive diagonal diffusion entries.
policy_matrix, policy_bias: Affine trajectory-policy parameters.
next_cores: Three compatible continuation TT cores with basis mode three.
initial_cores: Three compatible current-time TT cores with an orthonormal first and last component.
dt, tau_initial, gamma: Positive scalar parameters.
query: Three-factor evaluation state.

Returns
-------
feedback_component: Native float, the first component of the adapted feedback at query.

Returns
-------
float, the first component of -sigma_diag * gradient V at query
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_full_pipeline(
    states: np.ndarray,
    noises: np.ndarray,
    drift_matrix: np.ndarray,
    drift_bias: np.ndarray,
    sigma_diag: np.ndarray,
    policy_matrix: np.ndarray,
    policy_bias: np.ndarray,
    next_cores: list[np.ndarray],
    initial_cores: list[np.ndarray],
    dt: float,
    tau_initial: float,
    gamma: float,
    query: np.ndarray,
) -> float:
    """Run the adaptive tensor-train BSDE regression pipeline.

    Parameters
    ----------
    states : np.ndarray
        Left-endpoint states with shape (K, 3).
    noises : np.ndarray
        Fixed innovations with shape (K, 3).
    drift_matrix : np.ndarray
        Affine drift matrix with shape (3, 3).
    drift_bias : np.ndarray
        Affine drift bias with shape (3,).
    sigma_diag : np.ndarray
        Positive diagonal entries of the diffusion matrix.
    policy_matrix : np.ndarray
        Affine trajectory-policy matrix with shape (3, 3).
    policy_bias : np.ndarray
        Affine trajectory-policy bias with shape (3,).
    next_cores : list[np.ndarray]
        Three compatible continuation TT cores with basis mode three.
    initial_cores : list[np.ndarray]
        Three compatible current-time TT cores with orthonormal exterior components.
    dt : float
        Finite positive time increment.
    tau_initial : float
        Finite positive initial ridge magnitude.
    gamma : float
        Finite positive relative regularization weight.
    query : np.ndarray
        Three-factor state at which the feedback is evaluated.

    Raises
    ------
    ValueError
        If the state dimension is not three, any array shape or TT rank is
        incompatible, an input is non-finite, an exterior current-time core is
        not orthonormal, or `dt`, `tau_initial`, or `gamma` is not positive.

    Returns
    -------
    feedback_component : float
        First adapted feedback component at `query` as a native float.
    """
    return feedback_component  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_run_full_pipeline(
    states: np.ndarray,
    noises: np.ndarray,
    drift_matrix: np.ndarray,
    drift_bias: np.ndarray,
    sigma_diag: np.ndarray,
    policy_matrix: np.ndarray,
    policy_bias: np.ndarray,
    next_cores: list[np.ndarray],
    initial_cores: list[np.ndarray],
    dt: float,
    tau_initial: float,
    gamma: float,
    query: np.ndarray,
) -> float:
    """Reference implementation."""
    x = np.asarray(states, dtype=float)
    xi = np.asarray(noises, dtype=float)
    matrix = np.asarray(drift_matrix, dtype=float)
    drift_offset = np.asarray(drift_bias, dtype=float)
    sigma = np.asarray(sigma_diag, dtype=float)
    policy = np.asarray(policy_matrix, dtype=float)
    policy_offset = np.asarray(policy_bias, dtype=float)
    evaluation_point = np.asarray(query, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != 3:
        raise ValueError("states must have non-empty shape (K, 3)")
    if xi.shape != x.shape:
        raise ValueError("noises must have the same shape as states")
    if matrix.shape != (3, 3) or policy.shape != (3, 3):
        raise ValueError("drift_matrix and policy_matrix must have shape (3, 3)")
    if drift_offset.shape != (3,) or policy_offset.shape != (3,) or sigma.shape != (3,):
        raise ValueError("biases and sigma_diag must have shape (3,)")
    if evaluation_point.shape != (3,):
        raise ValueError("query must have shape (3,)")
    arrays = (x, xi, matrix, drift_offset, sigma, policy, policy_offset, evaluation_point)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("all direct array inputs must be finite")
    if np.any(sigma <= 0.0):
        raise ValueError("sigma_diag must be positive")
    scalars = (dt, tau_initial, gamma)
    if any(not np.isfinite(value) or value <= 0.0 for value in scalars):
        raise ValueError("dt, tau_initial, and gamma must be finite and positive")
    if not isinstance(next_cores, (list, tuple)) or len(next_cores) != 3:
        raise ValueError("next_cores must contain three TT cores")
    if not isinstance(initial_cores, (list, tuple)) or len(initial_cores) != 3:
        raise ValueError("initial_cores must contain three TT cores")
    continuation_cores = [np.asarray(core, dtype=float) for core in next_cores]
    current_cores = [np.asarray(core, dtype=float) for core in initial_cores]
    for core_set in (continuation_cores, current_cores):
        if any(core.ndim != 3 or core.shape[1] != 3 for core in core_set):
            raise ValueError("every TT core must have basis mode three")
        if core_set[0].shape[0] != 1 or core_set[-1].shape[2] != 1:
            raise ValueError("exterior TT ranks must equal one")
        if core_set[0].shape[2] != core_set[1].shape[0] or core_set[1].shape[2] != core_set[2].shape[0]:
            raise ValueError("adjacent TT ranks must agree")
        if any(not np.all(np.isfinite(core)) for core in core_set):
            raise ValueError("TT cores must be finite")
    first_matrix = current_cores[0].reshape(3, current_cores[0].shape[2])
    last_matrix = current_cores[2].reshape(current_cores[2].shape[0], 3)
    if not np.allclose(first_matrix.T @ first_matrix, np.eye(first_matrix.shape[1]), atol=1e-12):
        raise ValueError("the first current-time core must be left orthonormal")
    if not np.allclose(last_matrix @ last_matrix.T, np.eye(last_matrix.shape[0]), atol=1e-12):
        raise ValueError("the last current-time core must be right orthonormal")

    next_states = _oracle_advance_controlled_diffusion(  # noqa: F821
        x, xi, matrix, drift_offset, sigma, policy, policy_offset, dt
    )
    continuation = _oracle_evaluate_tt_continuation(  # noqa: F821
        next_states, continuation_cores
    )
    targets = _oracle_form_bsde_targets(  # noqa: F821
        continuation,
        next_states,
        policy,
        policy_offset,
        float(np.trace(matrix)),
        sigma,
        dt,
    )
    weighted_features = _oracle_build_weighted_features(x, xi, sigma, dt)  # noqa: F821
    middle_design = _oracle_assemble_local_system(  # noqa: F821
        weighted_features, current_cores, 1
    )
    updated_middle = _oracle_solve_ridge_core(  # noqa: F821
        middle_design, targets, tau_initial, current_cores[1].shape
    )
    adaptive_state = _oracle_advance_adaptive_sweep(  # noqa: F821
        weighted_features,
        middle_design,
        targets,
        current_cores[0],
        updated_middle,
        current_cores[2],
        gamma,
    )
    feedback = _oracle_recover_feedback_control(  # noqa: F821
        evaluation_point, current_cores[0], adaptive_state, sigma
    )
    return float(feedback[0])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
states = np.array([[-0.8,-0.5,0.2],[-0.6,0.3,-0.4],[-0.3,0.7,0.6],[-0.1,-0.8,-0.7],[0.2,-0.2,0.9],[0.4,0.5,-0.1],[0.7,-0.6,0.4],[0.9,0.1,-0.8],[0.5,0.9,0.7],[-0.7,0.8,-0.2]])
noises = np.array([[0.2,-1.1,0.5],[-0.7,0.4,1.2],[1.1,-0.3,-0.8],[-1.3,0.9,0.2],[0.6,1.0,-1.0],[-0.2,-0.7,0.9],[0.8,0.2,-0.5],[-0.9,-1.2,0.7],[0.3,0.6,1.1],[1.2,-0.5,-0.3]])
F = np.array([[0.05,-0.08,0.02],[0.03,-0.12,0.04],[-0.01,0.06,-0.09]])
b = np.array([0.01,-0.02,0.015])
sigma = np.array([0.35,0.22,0.18])
U = np.array([[-0.25,0.08,-0.04],[0.05,-0.18,0.06],[-0.03,0.04,-0.12]])
c = np.array([0.02,-0.01,0.015])
n1 = np.array([[[0.75,-0.20],[0.10,0.35],[-0.08,0.12]]])
n2 = np.array([[[0.90,-0.15],[0.20,0.25],[-0.10,0.18]],[[0.05,0.70],[-0.30,0.12],[0.22,-0.08]]])
n3 = np.array([[[0.80],[0.15],[-0.05]],[[-0.20],[0.60],[0.10]]])
q = 0.7071067811865476
c1 = np.array([[[q,0.0],[0.0,1.0],[q,0.0]]])
c2 = np.array([[[0.4,-0.1],[0.15,0.35],[-0.05,0.12]],[[0.08,0.3],[-0.22,0.1],[0.18,-0.07]]])
c3 = np.array([[[q],[0.0],[q]],[[0.0],[1.0],[0.0]]])
next_cores = [n1,n2,n3]
initial_cores = [c1,c2,c3]
query = np.array([0.35,-0.15,0.55])
""",
            "call": "round(run_full_pipeline(states, noises, F, b, sigma, U, c, next_cores, initial_cores, 0.15, 0.07, 0.2, query), 12)",
            "gold_call": "round(_oracle_run_full_pipeline(states, noises, F, b, sigma, U, c, next_cores, initial_cores, 0.15, 0.07, 0.2, query), 12)",
        },
        {
            "setup": """import numpy as np
states = np.array([[0.0,0.0,0.0],[0.2,-0.1,0.3]])
noises = np.array([[0.2,-0.1,0.3],[-0.4,0.5,-0.2]])
F = np.zeros((3,3))
b = np.array([0.01,-0.02,0.03])
sigma = np.array([0.2,0.25,0.3])
U = np.zeros((3,3))
c = np.zeros(3)
n1 = np.array([[[1.0],[0.2],[0.1]]])
n2 = np.array([[[0.8],[-0.1],[0.05]]])
n3 = np.array([[[1.1],[0.15],[-0.08]]])
c1 = np.array([[[1.0],[0.0],[0.0]]])
c2 = np.array([[[0.3],[0.1],[-0.1]]])
c3 = np.array([[[1.0],[0.0],[0.0]]])
query = np.zeros(3)
""",
            "call": "round(run_full_pipeline(states, noises, F, b, sigma, U, c, [n1,n2,n3], [c1,c2,c3], 0.1, 0.05, 0.1, query), 12)",
            "gold_call": "round(_oracle_run_full_pipeline(states, noises, F, b, sigma, U, c, [n1,n2,n3], [c1,c2,c3], 0.1, 0.05, 0.1, query), 12)",
        },
        {
            "setup": """import numpy as np
states = np.zeros((2,2))
noises = np.zeros((2,2))
F = np.zeros((3,3))
b = np.zeros(3)
sigma = np.ones(3)
U = np.zeros((3,3))
c = np.zeros(3)
core = np.zeros((1,3,1))
def run_model():
    try:
        run_full_pipeline(states, noises, F, b, sigma, U, c, [core]*3, [core]*3, 0.1, 0.1, 0.1, np.zeros(3))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_run_full_pipeline(states, noises, F, b, sigma, U, c, [core]*3, [core]*3, 0.1, 0.1, 0.1, np.zeros(3))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
