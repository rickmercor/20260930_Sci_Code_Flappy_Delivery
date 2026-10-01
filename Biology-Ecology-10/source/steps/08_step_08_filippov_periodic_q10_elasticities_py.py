"""
Converge the renewed Filippov cycle to its periodic fixed point and return its ten cycle diagnostics, their total log-Q10 elasticities, and the stability radius of the renewal map.

Let $M(\mathbf{y},\mathbf{q})$ be the Step-7 renewal map and $\mathbf{y}^*(\mathbf{q})$ its fixed point, $\mathbf{y}^*=M(\mathbf{y}^*,\mathbf{q})$. On the periodic cycle started from $\mathbf{y}^*$, the diagnostics are

$X=[\overline{O_2},\overline{POC_L+POC_S},\overline{\chi},\overline{NO_3},\overline{P},\overline{R},\overline{N_{O_2}},\overline{M_{O_2}},\varphi_s,t_{\mathrm{off}}]$. Each overbar is the inclusive trapezoidal mean over the $N+1$ grid values of one period, $\overline{x}=\frac{\Delta t}{\mathcal{P}}\left(\frac{x_0}{2}+\sum_{i=1}^{N-1}x_i+\frac{x_N}{2}\right)$. With $t_{\mathrm{on}}$ the time of the first Step-7 event with code 0 (start of sliding) and $t_{\mathrm{off}}$ the time of the first exit event (code $\pm1$) after it, $\varphi_s=(t_{\mathrm{off}}-t_{\mathrm{on}})/\mathcal{P}$ is the fraction of the period during which ammonium is held at the critical concentration. The ammonium uptake share $\chi_i$ at a grid state is the Step-6 equivalent share $\chi_{\mathrm{eq}}$ when the state lies on the surface ($NH_4=c$), $NH_4/(NH_4+NO_3)$ when $NH_4<c$, and 1 when $NH_4>c$; it measures the fraction of phytoplankton nitrogen uptake supplied by ammonium.

 

The elasticity of diagnostic $j$ with respect to process $k$ is the total derivative along the periodic branch, $E_{jk}=\frac{1}{X_j}\frac{dX_j}{d\log Q_{10,k}}$, including the response of the periodic start $\mathbf{y}^*(\mathbf{q})$ to the Q10 change. The periodic cycle's arrival and exit are transversal events, so the renewed discrete Filippov map and the event times are differentiable at the benchmark fixed point. The spectral radius $\max|\operatorname{eig}(\partial M/\partial\mathbf{y})|$ at $\mathbf{y}^*$ certifies that the fixed point is locally attracting.

Returns
-------
return cycle_start, reference_diagnostics, elasticity_matrix, spectral_radius
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def filippov_periodic_q10_elasticities(
    initial_cycle_start: "np.ndarray",
    reference_q10: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
    renewal_state: "np.ndarray",
    retention_fraction: float,
    time_step: float,
    closure_tolerance: float,
    max_cycles: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, float]":
    """Return the periodic start, diagnostics, elasticities, and map radius.
 
    The periodic start is obtained by iterating the Step-7 renewal map from
    initial_cycle_start and accepting the first update y_(n+1) with
    max |y_(n+1) - y_(n)| <= closure_tolerance. All derivatives are exact
    derivatives of the discrete Step-7 map; any method that reproduces them
    to a relative accuracy of 1e-7 or better is acceptable.
 
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
        cycle_start, reference_diagnostics, elasticity_matrix, spectral_radius,
        where cycle_start is the accepted periodic start, shape (7,);
        reference_diagnostics is X in the declared order, shape (10,);
        elasticity_matrix is E with shape (10, 6), rows following X and columns
        following the Q10 order; and spectral_radius is the float spectral
        radius of the renewal-map state Jacobian at cycle_start.
 
    Raises:
        ValueError: If an input contract fails.
        RuntimeError: If the renewal map does not close within max_cycles, if
            the periodic cycle does not start sliding at a positive time or
            does not leave the surface afterwards,
            if Step 7 raises RuntimeError, if I - dM/dy is singular, or if a
            diagnostic is zero or nonfinite.
    """
    return cycle_start, reference_diagnostics, elasticity_matrix, spectral_radius

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
 
def _cycle_outputs(start, q10_values, parameters, forcing_parameters, renewal_state, retention_fraction, time_step):
    """Renewed state (7) followed by the ten cycle diagnostics (10)."""
    times, states, fluxes, next_start, events = _oracle_integrate_filippov_cycle_rk4(
        start, q10_values, parameters, forcing_parameters, renewal_state, retention_fraction, time_step
    )
    onset = [time for time, code in events if code == 0.0]
    if not onset:
        raise RuntimeError("the cycle never starts sliding")
    release = [time for time, code in events if abs(code) == 1.0 and time > onset[0]]
    if not release:
        raise RuntimeError("the cycle never leaves the surface after sliding starts")
    shares = np.empty(times.size)
    for index, (time, state) in enumerate(zip(times, states)):
        if state[5] == _critical_ammonium():
            shares[index] = _oracle_ammonium_sliding_field(
                time, state, q10_values, parameters, forcing_parameters
            )[3]
        elif state[5] > _critical_ammonium():
            shares[index] = 1.0
        else:
            shares[index] = state[5] / (state[5] + state[6])
    series = np.column_stack(
        [states[:, 0], states[:, 2] + states[:, 3], shares, states[:, 6], fluxes]
    )
    weights = np.ones(times.size)
    weights[0] = weights[-1] = 0.5
    means = time_step * (weights @ series) / forcing_parameters[2]
    sliding_fraction = (release[0] - onset[0]) / forcing_parameters[2]
    return np.concatenate([next_start, means, [sliding_fraction, release[0]]])
 
 
def _centered_derivative(function, base, direction, step):
    """Second-order centered derivative of function at base along direction."""
    return (function(base + step * direction) - function(base - step * direction)) / (2.0 * step)
 
 
def _oracle_filippov_periodic_q10_elasticities(
    initial_cycle_start: "np.ndarray",
    reference_q10: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
    renewal_state: "np.ndarray",
    retention_fraction: float,
    time_step: float,
    closure_tolerance: float,
    max_cycles: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, float]":
    closure_tolerance = float(closure_tolerance)
    if not np.isfinite(closure_tolerance) or closure_tolerance <= 0.0:
        raise ValueError("closure_tolerance must be finite and positive")
    if isinstance(max_cycles, (bool, np.bool_)) or not isinstance(max_cycles, (int, np.integer)) or max_cycles < 1:
        raise ValueError("max_cycles must be a positive integer")
    (
        start,
        reference_q10,
        parameters,
        forcing_parameters,
        renewal_state,
        retention_fraction,
        time_step,
        _,
    ) = _validate_filippov_cycle_inputs(
        initial_cycle_start,
        reference_q10,
        parameters,
        forcing_parameters,
        renewal_state,
        retention_fraction,
        time_step,
    )
    fixed = (parameters, forcing_parameters, renewal_state, retention_fraction, time_step)
 
    cycle_start = start.copy()
    for _ in range(int(max_cycles)):
        next_start = _oracle_integrate_filippov_cycle_rk4(cycle_start, reference_q10, *fixed)[3]
        closure = np.max(np.abs(next_start - cycle_start))
        cycle_start = next_start
        if closure <= closure_tolerance:
            break
    else:
        raise RuntimeError("the renewal map did not close within max_cycles")
 
    reference = _cycle_outputs(cycle_start, reference_q10, *fixed)
    diagnostics = reference[7:]
    onset_time = diagnostics[9] - diagnostics[8] * forcing_parameters[2]
    if not onset_time > 0.0:
        raise RuntimeError("the periodic cycle must start sliding at a positive time")
    if not np.all(np.isfinite(diagnostics)) or np.any(diagnostics == 0.0):
        raise RuntimeError("diagnostics must be finite and nonzero")
 
    log_q10 = np.log(reference_q10)
    state_jacobian = np.empty((17, 7))
    log_q10_jacobian = np.empty((17, 6))
    for j in range(7):
        step = 1.0e-5 * max(1.0, abs(cycle_start[j]))
        state_jacobian[:, j] = _centered_derivative(
            lambda y: _cycle_outputs(y, reference_q10, *fixed), cycle_start, np.eye(7)[j], step
        )
    for k in range(6):
        log_q10_jacobian[:, k] = _centered_derivative(
            lambda ell: _cycle_outputs(cycle_start, np.exp(ell), *fixed), log_q10, np.eye(6)[k], 1.0e-5
        )
 
    map_state = state_jacobian[:7]
    try:
        start_sensitivity = np.linalg.solve(np.eye(7) - map_state, log_q10_jacobian[:7])
    except np.linalg.LinAlgError as error:
        raise RuntimeError("I - dM/dy is singular") from error
    total = state_jacobian[7:] @ start_sensitivity + log_q10_jacobian[7:]
    elasticity_matrix = total / diagnostics[:, None]
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(map_state))))
    return cycle_start, diagnostics, elasticity_matrix, spectral_radius

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return three valid periodic cases and two documented error cases."""
    base = (
        "import numpy as np\n"
        "initial=np.array([220.,.2,25.,40.,30.,.4,40.])\n"
        "renewal_state=np.array([220.,.2,25.,40.,30.,.4,40.])\n"
        "q=np.array([1.4,4.4,1.1,1.2,1.12,1.04])\n"
        "parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n"
        "forcing=np.array([23.,2.,4.,90.,60.,.2,.05,340.,4.])\n"
        "rho=.65;dt=.125;tol=1.e-8;cycles=500\n"
    )
    flatten = "(lambda r:np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))"
    args = "(initial.copy(),q.copy(),parameters.copy(),forcing.copy(),renewal_state.copy(),rho,dt,tol,cycles)"
    call = flatten + "(filippov_periodic_q10_elasticities" + args + ")"
    gold = flatten + "(_oracle_filippov_periodic_q10_elasticities" + args + ")"
    status = (
        "def status(fn):\n"
        "    try: fn(); return 0\n"
        "    except ValueError: return 1\n"
        "    except RuntimeError: return 2\n"
        "    except Exception: return 3\n"
    )
    return [
        {"setup": base, "call": call, "gold_call": gold, "tol": 1.0e-6},
        {"setup": base + "q[5]=1.1", "call": call, "gold_call": gold, "tol": 1.0e-6},
        {"setup": base + "rho=.6", "call": call, "gold_call": gold, "tol": 1.0e-6},
        {
            "setup": base + "cycles=3\n" + status,
            "call": "status(lambda: filippov_periodic_q10_elasticities" + args + ")",
            "gold_call": "status(lambda: _oracle_filippov_periodic_q10_elasticities" + args + ")",
        },
        {
            "setup": base + "dt=.03\n" + status,
            "call": "status(lambda: filippov_periodic_q10_elasticities" + args + ")",
            "gold_call": "status(lambda: _oracle_filippov_periodic_q10_elasticities" + args + ")",
        },
    ]
