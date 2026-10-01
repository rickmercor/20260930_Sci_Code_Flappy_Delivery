"""
Fit a common flooding efficiency and unbiased log-rate across sets.

Agreement among differently biased conditions identifies a shared efficiency
without assuming the collective variable is kinetically perfect.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_eatr_flooding(
    log_k_observed: 'np.ndarray',
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    temperature: float,
    indices: 'np.ndarray',
    gamma_min: float = 0.0,
    gamma_max: float = 1.0,
) -> 'np.ndarray':
    """Return ``[gamma, mean_log_k0, variance]`` for selected sets.

    Minimize the population variance of the selected per-set corrected
    log-rates with bounded scalar minimization using ``xatol=1e-12`` and
    ``maxiter=10000``, then report their mean and variance at the optimum.
    At least three distinct set indices are needed.

    Raises ``ValueError`` for invalid selections, bounds, arrays, or a
    nonfinite optimization result.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import minimize_scalar


def _eatr_prefix_variance(
    gamma: float,
    logs: 'np.ndarray',
    bias: 'np.ndarray',
    lens: 'np.ndarray',
    temp: float,
    chosen: 'np.ndarray',
) -> float:
    acceleration = _oracle_ensemble_time_log_acceleration(
        bias[chosen], lens[chosen], temp, float(gamma)
    )
    corrected = logs[chosen] - acceleration
    return float(np.var(corrected))


def _oracle_fit_eatr_flooding(
    log_k_observed: 'np.ndarray',
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    temperature: float,
    indices: 'np.ndarray',
    gamma_min: float = 0.0,
    gamma_max: float = 1.0,
) -> 'np.ndarray':
    try:
        logs = np.asarray(log_k_observed)
        bias = np.asarray(restored_bias)
        lens = np.asarray(lengths)
        chosen = np.asarray(indices)
        lower = float(gamma_min)
        upper = float(gamma_max)
        temp = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("fit inputs must be numeric") from exc
    if (
        bias.ndim != 3 or logs.shape != (bias.shape[0],)
        or lens.shape != bias.shape[:2] or chosen.ndim != 1 or chosen.size < 3
        or not np.issubdtype(chosen.dtype, np.integer)
    ):
        raise ValueError("incompatible fit shapes or selection")
    if (
        not np.issubdtype(logs.dtype, np.number) or not np.isrealobj(logs)
        or np.any(~np.isfinite(logs.astype(float, copy=False)))
        or np.any(chosen < 0) or np.any(chosen >= bias.shape[0])
        or len(np.unique(chosen)) != len(chosen)
        or not np.isfinite(temp) or temp <= 0.0
        or not np.isfinite(lower) or not np.isfinite(upper)
        or not 0.0 <= lower < upper <= 1.0
    ):
        raise ValueError("invalid fit data, indices, temperature, or bounds")
    logs = logs.astype(float, copy=False)
    optimum = minimize_scalar(
        _eatr_prefix_variance,
        args=(logs, bias, lens, temp, chosen),
        bounds=(lower, upper),
        options={'xatol': 1e-12, 'maxiter': 10000},
        method="bounded",
    )
    gamma = float(optimum.x)
    corrected = logs[chosen] - _oracle_ensemble_time_log_acceleration(
        bias[chosen], lens[chosen], temp, gamma
    )
    result = np.array([gamma, np.mean(corrected), np.var(corrected)], dtype=float)
    if not bool(optimum.success) or np.any(~np.isfinite(result)):
        raise ValueError("bounded flooding fit did not produce a finite optimum")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\nv=np.array([[[1.,1.2],[1.1,1.3]],[[2.,2.2],[2.1,2.3]],[[3.,3.2],[3.1,3.3]],[[4.,4.2],[4.1,4.3]]]); l=np.full((4,2),2); y=np.array([-1.1,-.8,-.5,-.2])"
    return [
        {"setup": setup, "call": "fit_eatr_flooding(y,v,l,300.,np.array([0,1,2,3]))", "gold_call": "_oracle_fit_eatr_flooding(y,v,l,300.,np.array([0,1,2,3]))", "tol": 1e-10},
        {"setup": setup, "call": "fit_eatr_flooding(y,v,l,300.,np.array([0,1,2]),.15,.85)", "gold_call": "_oracle_fit_eatr_flooding(y,v,l,300.,np.array([0,1,2]),.15,.85)", "tol": 1e-10},
        {"setup": "import numpy as np\nv=np.zeros((3,2,2)); l=np.full((3,2),2); y=np.array([-2.,-2.,-2.])", "call": "fit_eatr_flooding(y,v,l,300.,np.arange(3))", "gold_call": "_oracle_fit_eatr_flooding(y,v,l,300.,np.arange(3))", "tol": 1e-10},
        {"setup": "import numpy as np\ndef check(fn):\n v=np.zeros((4,2,2)); l=np.full((4,2),2); y=np.zeros(4); out=[]\n for idx,a,b in ((np.array([0,0,1]),0.,1.),(np.array([0,1]),0.,1.),(np.array([0,1,4]),0.,1.),(np.array([0,1,2]),.8,.2)):\n  try: fn(y,v,l,300.,idx,a,b)\n  except ValueError: out.append(1)\n  except Exception: out.append(2)\n  else: out.append(0)\n return np.array(out)", "call": "check(fit_eatr_flooding)", "gold_call": "check(_oracle_fit_eatr_flooding)", "tol": 0.0},
    ]
