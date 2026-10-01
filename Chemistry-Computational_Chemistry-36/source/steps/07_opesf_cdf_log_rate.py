"""
Compute the conventional OPES-f comparator from rescaled event times.

Trajectory-specific acceleration rescaling and a censored empirical CDF form
the conventional reference against which the multi-set rate is compared.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def opesf_cdf_log_rate(
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    events: 'np.ndarray',
    dt: float,
    temperature: float,
    indices: 'np.ndarray',
) -> float:
    """Return the pooled conventional OPES-f CDF-fit log-rate.

    Rescale every selected trajectory duration by its own time-average of
    ``exp(beta*V)``.  Pool the selected trajectories, sort only transitioned
    rescaled times, use empirical ordinates ``1/N,...,M/N`` with all ``N``
    pooled trajectories in the denominator, and fit ``1-exp(-k*t)`` by
    unweighted nonlinear least squares using ``ftol=xtol=gtol=1e-12`` and
    ``maxfev=100000``.  Return ``log(k)``.

    Raises ``ValueError`` for invalid inputs, unsafe exponentiation, or a
    nonpositive/nonfinite fit.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import curve_fit


def _oracle_opesf_cdf_log_rate(
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    events: 'np.ndarray',
    dt: float,
    temperature: float,
    indices: 'np.ndarray',
) -> float:
    try:
        bias = np.asarray(restored_bias)
        lens = np.asarray(lengths)
        ev = np.asarray(events)
        chosen = np.asarray(indices)
        step = float(dt)
        temp = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("OPES-f inputs must be numeric") from exc
    if (
        bias.ndim != 3 or lens.shape != bias.shape[:2] or ev.shape != lens.shape
        or chosen.ndim != 1 or chosen.size < 1
        or not np.issubdtype(chosen.dtype, np.integer)
    ):
        raise ValueError("incompatible OPES-f input shapes")
    if (
        not np.issubdtype(bias.dtype, np.number) or not np.isrealobj(bias)
        or not np.issubdtype(lens.dtype, np.integer)
        or not np.issubdtype(ev.dtype, np.number)
        or np.any(chosen < 0) or np.any(chosen >= bias.shape[0])
        or len(np.unique(chosen)) != len(chosen)
        or not np.isfinite(step) or step <= 0.0
        or not np.isfinite(temp) or temp <= 0.0
    ):
        raise ValueError("invalid OPES-f arrays or controls")
    bias = bias.astype(float, copy=False)
    ev = ev.astype(float, copy=False)
    if np.any((ev != 0.0) & (ev != 1.0)) or np.any(lens < 1) or np.any(lens > bias.shape[2]):
        raise ValueError("events must be binary and lengths in range")
    beta = 1.0 / (0.008314462618 * temp)
    log_times = []
    pooled_events = []
    for set_index in chosen:
        for trajectory in range(bias.shape[1]):
            count = int(lens[set_index, trajectory])
            active = bias[set_index, trajectory, :count]
            if np.any(~np.isfinite(active)):
                raise ValueError("active bias samples must be finite")
            log_acceleration = _eatr_logmeanexp(beta * active)
            log_times.append(np.log(step * count) + log_acceleration)
            pooled_events.append(bool(ev[set_index, trajectory]))
    log_times = np.asarray(log_times, dtype=float)
    pooled_events = np.asarray(pooled_events, dtype=bool)
    if np.count_nonzero(pooled_events) < 2:
        raise ValueError("CDF fitting requires at least two transitioned trajectories")
    if np.max(log_times) >= np.log(np.finfo(float).max):
        raise ValueError("rescaled time exceeded the float64 range")
    times = np.exp(log_times)
    event_times = np.sort(times[pooled_events])
    ordinates = np.arange(1, len(event_times) + 1, dtype=float) / len(times)
    initial = float(np.count_nonzero(pooled_events) / np.sum(times))
    try:
        parameters, _ = curve_fit(
            lambda time, rate: 1.0 - np.exp(-rate * time),
            event_times,
            ordinates,
            p0=initial,
            maxfev=100000,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        raise ValueError("OPES-f CDF fit failed") from exc
    rate = float(parameters[0])
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("OPES-f CDF fit returned a nonpositive rate")
    result = float(np.log(rate))
    if not np.isfinite(result):
        raise ValueError("OPES-f log-rate is not finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\nv=np.array([[[1.,1.2,0.],[1.1,1.3,1.4],[.9,1.,0.]],[[2.,2.2,2.3],[2.1,2.3,0.],[1.9,2.,2.1]],[[3.,3.2,0.],[3.1,3.3,3.4],[2.9,3.,3.1]]]); l=np.array([[2,3,2],[3,2,3],[2,3,3]]); e=np.array([[1,0,1],[1,1,0],[1,0,1]])"
    return [
        {"setup": setup, "call": "opesf_cdf_log_rate(v,l,e,.1,300.,np.array([0,1,2]))", "gold_call": "_oracle_opesf_cdf_log_rate(v,l,e,.1,300.,np.array([0,1,2]))", "tol": 1e-4},
        {"setup": setup, "call": "opesf_cdf_log_rate(v,l,e,.1,300.,np.array([2,0]))", "gold_call": "_oracle_opesf_cdf_log_rate(v,l,e,.1,300.,np.array([2,0]))", "tol": 1e-4},
        {"setup": "import numpy as np\nv=np.zeros((3,2,2)); l=np.full((3,2),2); e=np.ones((3,2),dtype=int)", "call": "opesf_cdf_log_rate(v,l,e,1e-4,1e6,np.arange(3))", "gold_call": "_oracle_opesf_cdf_log_rate(v,l,e,1e-4,1e6,np.arange(3))", "tol": 1e-4},
        {"setup": "import numpy as np\ndef check(fn):\n v=np.zeros((3,2,2)); l=np.full((3,2),2); out=[]\n for e,idx in ((np.zeros((3,2)),np.arange(3)),(np.ones((3,2)),np.array([0,0])),(np.ones((3,2)),np.array([3]))):\n  try: fn(v,l,e,.1,300.,idx)\n  except ValueError: out.append(1)\n  except Exception: out.append(2)\n  else: out.append(0)\n return np.array(out)", "call": "check(opesf_cdf_log_rate)", "gold_call": "check(_oracle_opesf_cdf_log_rate)", "tol": 0.0},
    ]
