"""
Assemble the frozen three-segment compound-BSDE calculation.

The configured surrogate evaluates independent value and control heads for the

successive BSDE components. The complete calculation couples adjacent forward

segments at the exercise dates and reduces all residual conditions to the

constant-free a posteriori certificate. This orchestrator is the final step in

Studio.

Inputs

------

seed: Nonnegative seed controlling the complete Brownian tensor.

Returns

-------

certificate: Native float reported as the final answer.

Returns
-------
float, the constant-free a posteriori certificate as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_full_pipeline(seed: int = 260118634) -> float:
    """Run the frozen three-segment pipeline and return its certificate.

    Parameters
    ----------
    seed : int
        Nonnegative seed controlling the complete Brownian tensor.

    Raises
    ------
    ValueError
        If seed is not a nonnegative integer.

    Returns
    -------
    certificate : float
        Constant-free a posteriori certificate for the configured instance.
    """
    return certificate  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402


def _oracle_run_full_pipeline(seed: int = 260118634) -> float:
    """Reference implementation chaining every earlier step."""
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    h = 0.125
    segment_steps = np.array([2, 2, 2, 2, 2], dtype=int)
    x0 = np.array([48.0, 52.0])
    x_scale = np.array([50.0, 50.0])
    q = np.array([0.01, 0.015])
    sigma = np.array([[0.24, 0.05], [0.08, 0.20]])
    r = 0.06
    strikes = np.array([51.0, 50.0, 49.0, 48.0, 47.0])
    theta = np.array([0.45, -0.30])
    feature_matrix = np.array([[0.70, -0.30], [-0.40, 0.60], [0.25, 0.50], [-0.55, -0.20]])
    feature_bias = np.array([0.10, -0.15, 0.05, 0.20])
    value_weights = np.array([[1.20, -0.40, 0.70, 0.30], [0.90, -0.60, 0.50, -0.20], [0.60, -0.30, 0.40, 0.10], [0.50, -0.25, 0.35, 0.05], [0.40, -0.20, 0.30, 0.15]])
    value_bias = np.array([3.40, 3.00, 2.60, 2.20, 1.80])
    control_bias = np.array([[-1.80, -1.40], [-1.70, -1.35], [-1.60, -1.25], [-1.50, -1.15], [-1.40, -1.05], [-1.30, -0.95], [-1.20, -0.85], [-1.10, -0.75], [-1.00, -0.65], [-0.90, -0.55]])
    control_weights = np.array([[[0.30, -0.20, 0.15, 0.10], [-0.10, 0.25, 0.20, -0.15]], [[0.25, -0.15, 0.10, 0.12], [-0.08, 0.22, 0.18, -0.12]], [[0.22, -0.18, 0.14, 0.08], [-0.12, 0.20, 0.16, -0.10]], [[0.20, -0.14, 0.12, 0.06], [-0.10, 0.18, 0.14, -0.08]], [[0.18, -0.12, 0.10, 0.05], [-0.08, 0.16, 0.12, -0.06]], [[0.16, -0.10, 0.08, 0.04], [-0.06, 0.14, 0.10, -0.05]], [[0.14, -0.08, 0.06, 0.03], [-0.05, 0.12, 0.08, -0.04]], [[0.12, -0.06, 0.05, 0.02], [-0.04, 0.10, 0.07, -0.03]], [[0.10, -0.05, 0.04, 0.02], [-0.03, 0.09, 0.06, -0.02]], [[0.08, -0.04, 0.03, 0.01], [-0.02, 0.08, 0.05, -0.02]]])
    n_steps = int(np.sum(segment_steps))
    increments = _oracle_generate_brownian_increments(seed, n_steps, 7, 2, h)  # noqa: F821
    paths = _oracle_simulate_gbm_paths(x0, q, sigma, r, h, increments)  # noqa: F821
    starts = np.concatenate((np.array([0]), np.cumsum(segment_steps)[:-1]))
    value_starts = _oracle_evaluate_value_heads(  # noqa: F821
        paths, starts, x_scale, feature_matrix, feature_bias, value_weights, value_bias
    )
    controls = _oracle_evaluate_control_heads(  # noqa: F821
        paths, x_scale, feature_matrix, feature_bias, control_weights, control_bias
    )
    terminals = _oracle_propagate_bsde_segments(  # noqa: F821
        value_starts, controls, increments, segment_steps, r, h, theta
    )
    targets = _oracle_construct_bermudan_targets(  # noqa: F821
        paths, value_starts, segment_steps, strikes
    )
    loss_summary = _oracle_compute_joint_objective(terminals, targets)  # noqa: F821
    return _oracle_compute_error_certificate(loss_summary, h)  # noqa: F821

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "seed = 260118634\n",
            "call": "float(run_full_pipeline(seed=seed))",
            "gold_call": "float(_oracle_run_full_pipeline(seed=seed))",
        },
        {
            "setup": "seed = 0\n",
            "call": "float(run_full_pipeline(seed=seed))",
            "gold_call": "float(_oracle_run_full_pipeline(seed=seed))",
        },
        {
            "setup": """def run_model():
    try:
        run_full_pipeline(seed=-1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_run_full_pipeline(seed=-1)
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
