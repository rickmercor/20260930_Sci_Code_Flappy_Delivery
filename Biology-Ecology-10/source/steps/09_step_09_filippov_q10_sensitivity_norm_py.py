"""
Run the complete Filippov periodic-sensitivity workflow and return the Frobenius norm of its 10-by-6 log-Q10 elasticity matrix.

Call Step 8 with the supplied inputs, take its elasticity matrix $E$, and return $\lVert E\rVert_F=\left(\sum_{j=1}^{10}\sum_{k=1}^{6}E_{jk}^2\right)^{1/2}$ as a native Python float. The norm aggregates how strongly the periodic oxygen, carbon, nitrogen, and flux diagnostics, and the timing of the ammonium-sliding onset, respond to the six process-specific temperature sensitivities.

Returns
-------
return sensitivity_norm
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def filippov_q10_sensitivity_norm(
    initial_cycle_start: "np.ndarray",
    reference_q10: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
    renewal_state: "np.ndarray",
    retention_fraction: float,
    time_step: float,
    closure_tolerance: float,
    max_cycles: int,
) -> float:
    """Return the Frobenius norm of the Step-8 elasticity matrix.
 
    Parameters:
        initial_cycle_start: Seven finite nonnegative initial states.
        reference_q10: Six finite positive Q10 values in the declared process order.
        parameters: The 22-value finite positive parameter vector.
        forcing_parameters: Nine finite forcing values in the Step-5 order.
        renewal_state: Seven finite nonnegative renewal-water states.
        retention_fraction: Finite renewal retention fraction, 0 < rho < 1.
        time_step: Positive RK4 step that divides the period under Step 7.
        closure_tolerance: Finite positive infinity-norm closure tolerance.
        max_cycles: Positive integer maximum number of renewal-map iterations.
 
    Returns:
        The Frobenius norm of the 10-by-6 elasticity matrix as a native float.
 
    Raises:
        ValueError: If an input contract of Step 8 fails.
        RuntimeError: If Step 8 raises RuntimeError.
    """
    return sensitivity_norm

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
 
def _oracle_filippov_q10_sensitivity_norm(
    initial_cycle_start: "np.ndarray",
    reference_q10: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
    renewal_state: "np.ndarray",
    retention_fraction: float,
    time_step: float,
    closure_tolerance: float,
    max_cycles: int,
) -> float:
    elasticity_matrix = _oracle_filippov_periodic_q10_elasticities(
        initial_cycle_start,
        reference_q10,
        parameters,
        forcing_parameters,
        renewal_state,
        retention_fraction,
        time_step,
        closure_tolerance,
        max_cycles,
    )[2]
    return float(np.linalg.norm(elasticity_matrix))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return three valid end-to-end cases and one documented error case."""
    base = (
        "import numpy as np\n"
        "initial=np.array([220.,.2,25.,40.,30.,.4,40.])\n"
        "renewal_state=np.array([220.,.2,25.,40.,30.,.4,40.])\n"
        "q=np.array([1.4,4.4,1.1,1.2,1.12,1.04])\n"
        "parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n"
        "forcing=np.array([23.,2.,4.,90.,60.,.2,.05,340.,4.])\n"
        "rho=.65;dt=.125;tol=1.e-8;cycles=500\n"
    )
    args = "(initial.copy(),q.copy(),parameters.copy(),forcing.copy(),renewal_state.copy(),rho,dt,tol,cycles)"
    status = (
        "def status(fn):\n"
        "    try: fn(); return 0\n"
        "    except ValueError: return 1\n"
        "    except RuntimeError: return 2\n"
        "    except Exception: return 3\n"
    )
    return [
        {"setup": base + "q[2]=1.3", "call": "filippov_q10_sensitivity_norm" + args, "gold_call": "_oracle_filippov_q10_sensitivity_norm" + args, "tol": 1.0e-6},
        {"setup": base + "q[1]=4.6", "call": "filippov_q10_sensitivity_norm" + args, "gold_call": "_oracle_filippov_q10_sensitivity_norm" + args, "tol": 1.0e-6},
        {"setup": base + "renewal_state[1]=.5", "call": "filippov_q10_sensitivity_norm" + args, "gold_call": "_oracle_filippov_q10_sensitivity_norm" + args, "tol": 1.0e-6},
        {
            "setup": base + "rho=1.2\n" + status,
            "call": "status(lambda: filippov_q10_sensitivity_norm" + args + ")",
            "gold_call": "status(lambda: _oracle_filippov_q10_sensitivity_norm" + args + ")",
        },
    ]
