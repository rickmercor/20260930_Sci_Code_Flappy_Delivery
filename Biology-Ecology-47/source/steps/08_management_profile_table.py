"""
Calculate the optimized profile table for every plan.

The function takes the buffered lower certificate from each plan's last certificate column, pairs it with the supplied upper effort, and calls the one-plan optimizer without moving cost normalization outside that optimizer.

Returns
-------
return a float64 matrix of shape (plans, 8), one optimized common-effort profile per plan.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def management_profile_table(competition: "np.ndarray", slopes: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", pulse_decay: "np.ndarray", climate_covariance: "np.ndarray", idiosyncratic_noise: "np.ndarray", durations: "np.ndarray", exposure: "np.ndarray", certificates: "np.ndarray", upper_efforts: "np.ndarray", policies: "np.ndarray", costs: "np.ndarray", contraction_gain: float, stability_limit: float, chi_radius: float) -> "np.ndarray":
    """Calculate the continuous common-effort profile for every management plan.
 
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
 
 
def _oracle_management_profile_table(competition: "np.ndarray", slopes: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", pulse_decay: "np.ndarray", climate_covariance: "np.ndarray", idiosyncratic_noise: "np.ndarray", durations: "np.ndarray", exposure: "np.ndarray", certificates: "np.ndarray", upper_efforts: "np.ndarray", policies: "np.ndarray", costs: "np.ndarray", contraction_gain: float, stability_limit: float, chi_radius: float) -> "np.ndarray":
    D = np.asarray(slopes, dtype=np.float64)
    P = np.asarray(pulse_decay, dtype=np.float64)
    cert = np.asarray(certificates, dtype=np.float64)
    upper = np.asarray(upper_efforts, dtype=np.float64)
    rules = np.asarray(policies, dtype=np.float64)
    price = np.asarray(costs, dtype=np.float64)
    if D.ndim != 2 or P.shape != D.shape or cert.ndim != 2 or cert.shape[0] != D.shape[0]:
        raise ValueError("unaligned plan table inputs")
    plans = D.shape[0]
    if upper.shape != (plans,) or rules.shape != (plans, 5) or price.shape != (plans,) or np.any(price <= 0.0):
        raise ValueError("unaligned plan policy inputs")
    rows = []
    for plan in range(plans):
        interval = np.array([cert[plan, -1], upper[plan]], dtype=np.float64)
        rows.append(_oracle_continuous_plan_profile(competition, D[plan], loadings, reserves, P[plan], climate_covariance, idiosyncratic_noise, durations, exposure, interval, rules[plan], price[plan], contraction_gain, stability_limit, chi_radius))
    return np.asarray(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = 'import numpy as np\nB = np.array([[[0.0,0.08,0.05],[0.06,0.0,0.07],[0.05,0.06,0.0]],[[0.0,0.09,0.04],[0.05,0.0,0.08],[0.06,0.05,0.0]],[[0.0,0.07,0.06],[0.07,0.0,0.05],[0.04,0.08,0.0]]])\nd = np.array([0.9,1.1,1.0])\nU = np.array([[[0.05,0.03],[-0.04,0.05],[0.03,-0.02]],[[0.04,-0.03],[0.02,0.05],[-0.05,0.03]],[[0.03,0.04],[-0.03,0.02],[0.04,-0.05]]])\nreserve = np.array([0.015,0.016,0.014])\npulse = np.array([0.35,0.4,0.32])\nclimate = np.array([[[1.0,0.25],[0.25,1.1]],[[0.9,-0.2],[-0.2,1.2]],[[1.2,0.3],[0.3,0.95]]])\nidio = np.full((3,3),0.001)\nduration = np.array([[0.45,0.55],[0.5,0.5],[0.6,0.4]])\nexposure = np.array([0.01,-0.02,0.015])' + "\nD=np.array([[0.9,1.1,1.0],[1.0,0.95,1.05],[1.1,1.0,0.9]]); pulses=np.array([[0.35,0.4,0.32],[0.38,0.34,0.41],[0.31,0.43,0.36]]); cert=np.column_stack([np.zeros((3,6)),np.array([0.4,0.45,0.5])]); upper=np.array([3.0,3.1,2.9]); policies=np.tile(np.array([1.5,0.1,0.1,0.15,0.04]),(3,1)); costs=np.array([1.0,1.05,0.97])"
    call = "management_profile_table(B,D,U,reserve,pulses,climate,idio,duration,exposure,cert,upper,policies,costs,0.04,0.92,2.447746830680816)"
    gold = "_oracle_management_profile_table(B,D,U,reserve,pulses,climate,idio,duration,exposure,cert,upper,policies,costs,0.04,0.92,2.447746830680816)"
    return [
        {"setup": common, "call": call, "gold_call": gold, "tol": 1e-8},
        {"setup": common + "\nupper[2]=0.5", "call": call, "gold_call": gold, "tol": 1e-8},
        {"setup": common + "\ncosts=np.array([1.0,0.8,1.2])", "call": call, "gold_call": gold, "tol": 1e-8},
        {"setup": common + "\ncosts[0]=-1.0\ndef caught(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    except Exception:\n        return np.array([-1.0])\n    return np.array([0.0])", "call": "caught(lambda: management_profile_table(B,D,U,reserve,pulses,climate,idio,duration,exposure,cert,upper,policies,costs,0.04,0.92,2.447746830680816))", "gold_call": "caught(lambda: _oracle_management_profile_table(B,D,U,reserve,pulses,climate,idio,duration,exposure,cert,upper,policies,costs,0.04,0.92,2.447746830680816))", "tol": 0.0},
    ]
