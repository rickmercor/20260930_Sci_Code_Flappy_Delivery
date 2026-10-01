"""
Optimize one plan under all periodic climate conditions.

The objective is the minimum of three smooth but changing condition branches. Evaluate the fixed 129-point inclusive scan. Refine every usable interior grid value that is at least both neighbours over its two-neighbour bracket with scipy.optimize.minimize_scalar(method='bounded', xatol=1e-11, maxiter=120), and include both endpoints. Compare robust values with a 1e-12 tolerance and choose the earlier effort within that tolerance. Return effort, the three condition scores, their minimum, the zero-based active condition, that condition's limiting species, and the maximum Floquet radius. If no effort is usable, return [-1, -1e12, -1e12, -1e12, -1e12, -1, -1, -1].

Returns
-------
return a float64 array of shape (8,): optimized effort, three condition scores, robust score, active condition, limiting species, and maximum Floquet radius
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def continuous_plan_profile(competition: "np.ndarray", slope: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", pulse_decay: "np.ndarray", climate_covariance: "np.ndarray", idiosyncratic_noise: "np.ndarray", durations: "np.ndarray", exposure: "np.ndarray", interval: "np.ndarray", policy: "np.ndarray", cost: float, contraction_gain: float, stability_limit: float, chi_radius: float) -> "np.ndarray":
    """Optimize one plan on its closed certificate interval and return the common-effort profile.
 
    Returns
    -------
    The numerical object stated in the step contract.
 
    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import minimize_scalar
 
 
def _oracle_continuous_plan_profile(competition: "np.ndarray", slope: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", pulse_decay: "np.ndarray", climate_covariance: "np.ndarray", idiosyncratic_noise: "np.ndarray", durations: "np.ndarray", exposure: "np.ndarray", interval: "np.ndarray", policy: "np.ndarray", cost: float, contraction_gain: float, stability_limit: float, chi_radius: float) -> "np.ndarray":
    bounds = np.asarray(interval, dtype=np.float64)
    rule = np.asarray(policy, dtype=np.float64)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] <= 0.0 or bounds[1] < bounds[0]:
        raise ValueError("interval must be a positive closed interval")
    if rule.shape != (5,) or not np.isfinite(rule).all() or not np.isfinite(cost) or cost <= 0.0:
        raise ValueError("invalid policy or cost")
    gain, effort_penalty, covariance_penalty, floquet_penalty, curvature = rule
 
    def _evaluate(a):
        panel = _oracle_periodic_risk_panel(competition, slope, loadings, reserves, pulse_decay, climate_covariance, idiosyncratic_noise, durations, exposure, float(a), contraction_gain, stability_limit, chi_radius)
        if not np.all(panel[:, 5] == 1.0):
            return None
        shared = gain * np.log1p(float(a)) - effort_penalty * float(a) - curvature * (float(a) - 6.1) ** 2
        scores = (panel[:, 3] - covariance_penalty * np.log1p(panel[:, 2]) - floquet_penalty * panel[:, 1] + shared) / float(cost)
        return scores, panel
 
    grid = np.linspace(bounds[0], bounds[1], 129, dtype=np.float64)
    values = np.full(129, -1.0e12, dtype=np.float64)
    for index, effort in enumerate(grid):
        outcome = _evaluate(float(effort))
        if outcome is not None:
            values[index] = float(np.min(outcome[0]))
    peak_indices = [index for index in range(1, 128) if values[index] > -1.0e11 and values[index] >= values[index - 1] and values[index] >= values[index + 1]]
    candidates = [float(grid[index]) for index in peak_indices]
    candidates.extend([float(grid[0]), float(grid[-1])])
    for index in peak_indices:
        lo, hi = float(grid[index - 1]), float(grid[index + 1])
 
        def _objective(a):
            outcome = _evaluate(float(a))
            return 1.0e12 if outcome is None else -float(np.min(outcome[0]))
 
        fitted = minimize_scalar(_objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-11, "maxiter": 120})
        if fitted.success and fitted.fun < 1.0e11:
            candidates.append(float(fitted.x))
    best = None
    for effort in candidates:
        outcome = _evaluate(effort)
        if outcome is None:
            continue
        scores, panel = outcome
        robust = float(np.min(scores))
        if best is None or robust > best[0] + 1e-12 or (abs(robust - best[0]) <= 1e-12 and effort < best[1]):
            best = (robust, effort, scores, panel)
    if best is None:
        unavailable = np.full(8, -1.0e12, dtype=np.float64)
        unavailable[[0, 5, 6, 7]] = -1.0
        return unavailable
    robust, effort, scores, panel = best
    active = int(np.argmin(scores))
    return np.array([effort, scores[0], scores[1], scores[2], robust, float(active), panel[active, 4], float(np.max(panel[:, 1]))], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = 'import numpy as np\nB = np.array([[[0.0,0.08,0.05],[0.06,0.0,0.07],[0.05,0.06,0.0]],[[0.0,0.09,0.04],[0.05,0.0,0.08],[0.06,0.05,0.0]],[[0.0,0.07,0.06],[0.07,0.0,0.05],[0.04,0.08,0.0]]])\nd = np.array([0.9,1.1,1.0])\nU = np.array([[[0.05,0.03],[-0.04,0.05],[0.03,-0.02]],[[0.04,-0.03],[0.02,0.05],[-0.05,0.03]],[[0.03,0.04],[-0.03,0.02],[0.04,-0.05]]])\nreserve = np.array([0.015,0.016,0.014])\npulse = np.array([0.35,0.4,0.32])\nclimate = np.array([[[1.0,0.25],[0.25,1.1]],[[0.9,-0.2],[-0.2,1.2]],[[1.2,0.3],[0.3,0.95]]])\nidio = np.full((3,3),0.001)\nduration = np.array([[0.45,0.55],[0.5,0.5],[0.6,0.4]])\nexposure = np.array([0.01,-0.02,0.015])' + "\ninterval=np.array([0.4,3.0]); policy=np.array([1.5,0.1,0.1,0.15,0.04])"
    return [
        {"setup": common, "call": "continuous_plan_profile(B,d,U,reserve,pulse,climate,idio,duration,exposure,interval,policy,1.0,0.04,0.92,2.447746830680816)", "gold_call": "_oracle_continuous_plan_profile(B,d,U,reserve,pulse,climate,idio,duration,exposure,interval,policy,1.0,0.04,0.92,2.447746830680816)", "tol": 1e-8},
        {"setup": common + "\ninterval=np.array([1.2,1.2])", "call": "continuous_plan_profile(B,d,U,reserve,pulse,climate,idio,duration,exposure,interval,policy,1.0,0.04,0.92,2.447746830680816)", "gold_call": "_oracle_continuous_plan_profile(B,d,U,reserve,pulse,climate,idio,duration,exposure,interval,policy,1.0,0.04,0.92,2.447746830680816)", "tol": 1e-8},
        {"setup": common + "\npolicy[-1]=0.2", "call": "continuous_plan_profile(B,d,U,reserve,pulse,climate,idio,duration,exposure,interval,policy,1.0,0.04,0.92,2.447746830680816)", "gold_call": "_oracle_continuous_plan_profile(B,d,U,reserve,pulse,climate,idio,duration,exposure,interval,policy,1.0,0.04,0.92,2.447746830680816)", "tol": 1e-8},
        {"setup": common, "call": "continuous_plan_profile(B,d,U,reserve,pulse,climate,idio,duration,exposure,interval,policy,1.0,0.04,0.10,2.447746830680816)", "gold_call": "_oracle_continuous_plan_profile(B,d,U,reserve,pulse,climate,idio,duration,exposure,interval,policy,1.0,0.04,0.10,2.447746830680816)", "tol": 1e-8},
        {"setup": common + "\ninterval=np.array([2.0,1.0])\ndef caught(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    except Exception:\n        return np.array([-1.0])\n    return np.array([0.0])", "call": "caught(lambda: continuous_plan_profile(B,d,U,reserve,pulse,climate,idio,duration,exposure,interval,policy,1.0,0.04,0.92,2.447746830680816))", "gold_call": "caught(lambda: _oracle_continuous_plan_profile(B,d,U,reserve,pulse,climate,idio,duration,exposure,interval,policy,1.0,0.04,0.92,2.447746830680816))", "tol": 0.0},
    ]
