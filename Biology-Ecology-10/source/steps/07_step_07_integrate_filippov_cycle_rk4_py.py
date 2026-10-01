"""
Integrate one forcing period of the Filippov solution of the reduced OxyPOM model, handling arrival at, sliding on, crossing of, and exit from the critical-ammonium surface, and apply the terminal renewal map.

The Step-5 right-hand side is discontinuous across $\Sigma=\{NH_4=c\}$ with $c=0.7$. Its Filippov solution moves in one of three modes: below $\Sigma$ with the smooth branch $\mathbf f^-$ ($\chi^-=NH_4/(NH_4+NO_3)$), above $\Sigma$ with $\mathbf f^+$ ($\chi^+=1$), or sliding on $\Sigma$ with the Step-6 field $\mathbf f_s$. With the Step-6 normal components $f^\pm_{NH_4}$ evaluated at the current time and state, the mode changes as follows. A state that reaches $\Sigma$ from below slides if $f^+_{NH_4}<0$ and otherwise crosses into the above mode; a state that reaches $\Sigma$ from above slides if $f^-_{NH_4}>0$ and otherwise crosses into the below mode. A sliding state leaves $\Sigma$ when the surface stops attracting: into the below mode when $f^-_{NH_4}\le0$, or into the above mode when $f^+_{NH_4}\ge0$. A cycle that starts on $\Sigma$ is classified with the same rules: it slides if $f^-_{NH_4}>0>f^+_{NH_4}$, moves above if both components are positive, and moves below if both are negative.

 

The cycle uses classical RK4 on the uniform grid $t_i=i\Delta t$, $N\Delta t=\mathcal P$, and every RK4 stage of a step uses the field of the mode in which that step starts, even when a stage state lies across $\Sigma$ or outside the sliding region (the Step-6 extension). An event inside a grid interval is located at the offset $u$ where it first occurs along a single RK4 step of length $u$ from the start of the current sub-interval. For arrival this is the root of $[\Psi(u)]_{NH_4}=c$; for exit it is the root of the violated normal component evaluated at time $t+u$ and state $\Psi(u)$. At an event the ammonium entry is set exactly to $c$, the event is recorded, and the remainder of the grid interval is integrated in the new mode. Arrival is tested only for sub-intervals that start strictly off $\Sigma$, so a state that has just left $\Sigma$ is not immediately recaptured. The period ends with the renewal map $\mathbf y_{\mathrm{next}}=\rho\,\mathbf y_N+(1-\rho)\,\mathbf y_{\mathrm{renew}}$.

Returns
-------
return times, states, fluxes, next_cycle_start, events
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def integrate_filippov_cycle_rk4(
    cycle_start: "np.ndarray",
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
    renewal_state: "np.ndarray",
    retention_fraction: float,
    time_step: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Integrate one Filippov forcing cycle and apply the renewal map.
 
    Parameters:
        cycle_start: Seven finite nonnegative initial states in the declared order.
        q10_values: Six finite positive Q10 values in the declared process order.
        parameters: The 22-value finite positive parameter vector.
        forcing_parameters: Nine finite forcing values in the Step-5 order;
            entry 2 is the period P.
        renewal_state: Seven finite nonnegative renewal-water states.
        retention_fraction: Finite renewal retention fraction rho, 0 < rho < 1.
        time_step: Finite positive step; N = round(P / time_step) must satisfy
            N >= 1 and |N * time_step - P| <= 1e-12.
 
    Returns:
        times, states, fluxes, next_cycle_start, events, where times has shape
        (N + 1,); states holds the Filippov states at the grid times, shape
        (N + 1, 7); fluxes holds the Step-5 diagnostic fluxes [P, R, N_O2,
        M_O2] at every grid state, shape (N + 1, 4); next_cycle_start is
        rho * states[-1] + (1 - rho) * renewal_state, shape (7,); and events
        is a float64 array of shape (m, 2) listing, in time order, [time, code]
        for every mode change in the cycle (m may be 0). Codes: 0 = start of
        sliding (arrival, or a sliding start at t = 0); -1 = exit from sliding
        to below; 1 = exit from sliding to above; -2 = crossing downward
        through the surface; 2 = crossing upward through the surface. Event
        times must be located to an absolute accuracy of 1e-12 days or better.
 
    Raises:
        ValueError: If an input contract fails.
        RuntimeError: If a cycle starts on the surface with f_minus_NH4 <= 0
            <= f_plus_NH4 (repelling), if Step 6 cannot form a sliding field
            during a sliding step, if more than four events occur within one
            grid interval, or if an accepted grid state is nonfinite or negative.
    """
    return times, states, fluxes, next_cycle_start, events

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq
 
 
def _validate_filippov_cycle_inputs(
    cycle_start,
    q10_values,
    parameters,
    forcing_parameters,
    renewal_state,
    retention_fraction,
    time_step,
):
    cycle_start = np.asarray(cycle_start, dtype=np.float64)
    q10_values = np.asarray(q10_values, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    forcing_parameters = np.asarray(forcing_parameters, dtype=np.float64)
    renewal_state = np.asarray(renewal_state, dtype=np.float64)
    retention_fraction = float(retention_fraction)
    time_step = float(time_step)
    for name, vector in (("cycle_start", cycle_start), ("renewal_state", renewal_state)):
        if vector.shape != (7,) or not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
            raise ValueError(f"{name} must contain seven finite nonnegative values")
    if q10_values.shape != (6,) or not np.all(np.isfinite(q10_values)) or np.any(q10_values <= 0.0):
        raise ValueError("q10_values must contain six finite positive values")
    if parameters.shape != (22,) or not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
        raise ValueError("parameters must contain 22 finite positive values")
    if forcing_parameters.shape != (9,) or not np.all(np.isfinite(forcing_parameters)) or forcing_parameters[2] <= 0.0:
        raise ValueError("forcing_parameters must contain nine finite values with positive period")
    if not np.isfinite(retention_fraction) or not 0.0 < retention_fraction < 1.0:
        raise ValueError("retention_fraction must lie strictly between zero and one")
    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time_step must be finite and positive")
    number_of_steps = int(round(forcing_parameters[2] / time_step))
    if number_of_steps < 1 or abs(number_of_steps * time_step - forcing_parameters[2]) > 1.0e-12:
        raise ValueError("forcing period must be an integer multiple of time_step")
    return (
        cycle_start,
        q10_values,
        parameters,
        forcing_parameters,
        renewal_state,
        retention_fraction,
        time_step,
        number_of_steps,
    )
 
 
def _branch_field(time, state, side, q10_values, parameters, forcing_parameters):
    """Smooth one-sided Step-5 branch: side -1 uses chi = NH4/DIN, side +1 uses chi = 1."""
    derivative = _oracle_oxypom_reduced_rhs(
        time, state, q10_values, parameters, forcing_parameters
    )[0].copy()
    ammonium, nitrate = state[5], state[6]
    chi_step5 = 1.0 if ammonium >= _critical_ammonium() else ammonium / (ammonium + nitrate)
    chi_branch = 1.0 if side > 0 else ammonium / (ammonium + nitrate)
    if chi_branch != chi_step5:
        temperature, light = _environment(time, forcing_parameters)
        rates = _oracle_phytoplankton_process_rates(
            temperature, light, state[1], ammonium, nitrate, q10_values, parameters
        )
        total_uptake = rates[4] + rates[5]
        derivative[5] -= (chi_branch - chi_step5) * total_uptake
        derivative[6] += (chi_branch - chi_step5) * total_uptake
    return derivative
 
 
def _surface_rates(time, state, q10_values, parameters, forcing_parameters):
    try:
        return _oracle_ammonium_sliding_field(time, state, q10_values, parameters, forcing_parameters)
    except ValueError as error:
        raise RuntimeError("no sliding field exists on the critical-ammonium surface") from error
 
 
def _rk4_mode_step(time, state, step_size, mode, q10_values, parameters, forcing_parameters):
    if step_size == 0.0:
        return state.copy()
    if mode == 0:
        def _field(t, y):
            return _surface_rates(t, y, q10_values, parameters, forcing_parameters)[4:]
    else:
        def _field(t, y):
            return _branch_field(t, y, mode, q10_values, parameters, forcing_parameters)
    k1 = _field(time, state)
    k2 = _field(time + 0.5 * step_size, state + 0.5 * step_size * k1)
    k3 = _field(time + 0.5 * step_size, state + 0.5 * step_size * k2)
    k4 = _field(time + step_size, state + step_size * k3)
    return state + step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
 
 
def _event_offset(function, length):
    """First offset in [0, length] at which a scalar event function reaches zero.
 
    The function is negative before the event; a nonnegative value at offset 0
    means the event has already occurred when the sub-interval starts.
    """
    if function(0.0) >= 0.0:
        return 0.0
    if function(length) == 0.0:
        return length
    return brentq(function, 0.0, length, xtol=1.0e-15, rtol=4.0 * np.finfo(float).eps, maxiter=200)
 
 
def _oracle_integrate_filippov_cycle_rk4(
    cycle_start: "np.ndarray",
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
    renewal_state: "np.ndarray",
    retention_fraction: float,
    time_step: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    (
        cycle_start,
        q10_values,
        parameters,
        forcing_parameters,
        renewal_state,
        retention_fraction,
        time_step,
        number_of_steps,
    ) = _validate_filippov_cycle_inputs(
        cycle_start,
        q10_values,
        parameters,
        forcing_parameters,
        renewal_state,
        retention_fraction,
        time_step,
    )
    critical = _critical_ammonium()
    args = (q10_values, parameters, forcing_parameters)
    times = np.linspace(0.0, forcing_parameters[2], number_of_steps + 1)
    states = np.empty((number_of_steps + 1, 7), dtype=np.float64)
    fluxes = np.empty((number_of_steps + 1, 4), dtype=np.float64)
    states[0] = cycle_start
    events = []
 
    if cycle_start[5] < critical:
        mode = -1
    elif cycle_start[5] > critical:
        mode = 1
    else:
        f_minus, f_plus = _surface_rates(0.0, cycle_start, *args)[:2]
        if f_minus > 0.0 and f_plus < 0.0:
            mode = 0
            events.append((0.0, 0.0))
        elif f_minus > 0.0 and f_plus > 0.0:
            mode = 1
        elif f_minus < 0.0 and f_plus < 0.0:
            mode = -1
        else:
            raise RuntimeError("the cycle starts on a repelling part of the surface")
 
    for index in range(number_of_steps):
        time, state = times[index], states[index].copy()
        remaining = time_step
        count = 0
        while remaining > 0.0:
            if count > 4:
                raise RuntimeError("more than four events in one grid interval")
            trial = _rk4_mode_step(time, state, remaining, mode, *args)
            if mode == 0:
                end_minus, end_plus = _surface_rates(time + remaining, trial, *args)[:2]
                if end_minus > 0.0 and end_plus < 0.0:
                    state, remaining = trial, 0.0
                    continue
 
                def _normal(offset, which):
                    end = _rk4_mode_step(time, state, offset, 0, *args)
                    return _surface_rates(time + offset, end, *args)[which]
 
                candidates = []
                if end_minus <= 0.0:
                    candidates.append((_event_offset(lambda u: -_normal(u, 0), remaining), -1))
                if end_plus >= 0.0:
                    candidates.append((_event_offset(lambda u: _normal(u, 1), remaining), 1))
                offset, mode = min(candidates)
                state = _rk4_mode_step(time, state, offset, 0, *args)
                state[5] = critical
                time, remaining = time + offset, remaining - offset
                events.append((time, float(mode)))
            else:
                reached = trial[5] >= critical if mode < 0 else trial[5] <= critical
                if not (state[5] != critical and reached):
                    state, remaining = trial, 0.0
                    continue
                offset = _event_offset(
                    lambda u: mode * (critical - _rk4_mode_step(time, state, u, mode, *args)[5]), remaining
                )
                state = _rk4_mode_step(time, state, offset, mode, *args)
                state[5] = critical
                time, remaining = time + offset, remaining - offset
                f_minus, f_plus = _surface_rates(time, state, *args)[:2]
                if mode < 0:
                    if f_plus < 0.0:
                        mode, code = 0, 0.0
                    else:
                        mode, code = 1, 2.0
                else:
                    if f_minus > 0.0:
                        mode, code = 0, 0.0
                    else:
                        mode, code = -1, -2.0
                events.append((time, code))
            count += 1
        if not np.all(np.isfinite(state)) or np.any(state < 0.0):
            raise RuntimeError("integration produced a nonfinite or negative state")
        states[index + 1] = state
 
    for index in range(number_of_steps + 1):
        fluxes[index] = _oracle_oxypom_reduced_rhs(times[index], states[index], *args)[1]
    next_cycle_start = retention_fraction * states[-1] + (1.0 - retention_fraction) * renewal_state
    event_array = np.array(events, dtype=np.float64).reshape(-1, 2)
    return times, states, fluxes, next_cycle_start, event_array

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return arrival/exit, above-start, crossing, on-surface, no-event, and invalid-input cases."""
    base = (
        "import numpy as np\n"
        "renewal_state=np.array([220.,.2,25.,40.,30.,.4,40.])\n"
        "q=np.array([1.4,4.4,1.1,1.2,1.12,1.04])\n"
        "parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n"
        "forcing=np.array([23.,2.,4.,90.,60.,.2,.05,340.,4.])\n"
        "rho=.65;dt=.04\n"
    )
    flatten = "(lambda r:np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))"
    call = flatten + "(integrate_filippov_cycle_rk4(start.copy(),q.copy(),parameters.copy(),forcing.copy(),renewal_state.copy(),rho,dt))"
    gold = flatten + "(_oracle_integrate_filippov_cycle_rk4(start.copy(),q.copy(),parameters.copy(),forcing.copy(),renewal_state.copy(),rho,dt))"
    status = (
        "def status(fn):\n"
        "    try: fn(); return 0\n"
        "    except ValueError: return 1\n"
        "    except RuntimeError: return 2\n"
        "    except Exception: return 3\n"
    )
    return [
        # Arrival from below, sliding, then exit to below as nitrate is drawn down.
        {"setup": base + "start=np.array([259.9,58.,108.,43.,62.,.595,20.3])", "call": call, "gold_call": gold},
        # Ammonium-rich start above the surface on a coarser admissible grid.
        {"setup": base + "start=np.array([250.,20.,60.,40.,50.,1.6,30.])\ndt=.05", "call": call, "gold_call": gold},
        # Weak initial uptake: upward crossing, then arrival from above into sliding.
        {"setup": base + "start=np.array([240.,6.,80.,40.,60.,.6,25.])", "call": call, "gold_call": gold},
        # Cycle that starts exactly on an attracting part of the surface.
        {"setup": base + "start=np.array([259.97,45.19,70.77,62.45,54.26,.7,14.9])", "call": call, "gold_call": gold},
        # Ammonium stays below the surface for the whole cycle: no events.
        {"setup": base + "start=np.array([220.,.2,25.,40.,30.,.05,40.])", "call": call, "gold_call": gold},
        # Documented input contract: retention fraction must lie strictly inside (0, 1).
        {
            "setup": base + "start=np.array([259.9,58.,108.,43.,62.,.595,20.3])\nrho=1.\n" + status,
            "call": "status(lambda: integrate_filippov_cycle_rk4(start.copy(),q.copy(),parameters.copy(),forcing.copy(),renewal_state.copy(),rho,dt))",
            "gold_call": "status(lambda: _oracle_integrate_filippov_cycle_rk4(start.copy(),q.copy(),parameters.copy(),forcing.copy(),renewal_state.copy(),rho,dt))",
        },
    ]
