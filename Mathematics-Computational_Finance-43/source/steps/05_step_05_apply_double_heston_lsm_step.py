"""
Apply one in-the-money least-squares Bermudan stopping decision.

Continuation is estimated from discounted future cashflows using spot and both variance factors as the Markov state.

Returns
-------
np.ndarray, updated cashflows and stopping indices with shape (2, n_paths).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apply_double_heston_lsm_step(
    spot: "np.ndarray", variance_one: "np.ndarray", variance_two: "np.ndarray",
    strike: float, current_step: int, cashflows: "np.ndarray",
    exercise_steps: "np.ndarray", rate: float, dt: float,
) -> "np.ndarray":
    r"""Return updated cashflows and stopping indices after one exercise date.

    Parameters
    ----------
    spot, variance_one, variance_two, cashflows, exercise_steps : numpy.ndarray
        Equal path vectors holding current and selected stopping states.
    strike, rate, dt : float
        Positive strike and interval length, and a finite interest rate.
    current_step : int
        Nonnegative current exercise index.
    Returns
    -------
    state : numpy.ndarray
        Two rows containing updated cashflows and stopping indices.

    Raises
    ------
    ValueError
        If path vectors, stopping states, or scalar date inputs are invalid.
    """
    return state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_apply_double_heston_lsm_step(
    spot: "np.ndarray", variance_one: "np.ndarray", variance_two: "np.ndarray",
    strike: float, current_step: int, cashflows: "np.ndarray",
    exercise_steps: "np.ndarray", rate: float, dt: float,
) -> "np.ndarray":
    arrays = [np.asarray(value, dtype=float) for value in
              (spot, variance_one, variance_two, cashflows, exercise_steps)]
    if (arrays[0].ndim != 1 or arrays[0].size == 0
            or any(value.shape != arrays[0].shape for value in arrays[1:])):
        raise ValueError("path inputs must be equal nonempty vectors")
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("all path inputs must be finite")
    if np.any(arrays[1] < 0.0) or np.any(arrays[2] < 0.0):
        raise ValueError("variance states must be nonnegative")
    if not np.isfinite([strike, rate, dt]).all() or strike <= 0.0 or dt <= 0.0:
        raise ValueError("strike and dt must be positive finite values")
    spot, variance_one, variance_two, cashflows, exercise_steps = arrays
    if (isinstance(current_step, bool)
            or not isinstance(current_step, (int, np.integer))
            or current_step < 0 or np.any(exercise_steps < current_step)):
        raise ValueError("current_step and stored stopping dates are invalid")
    x = spot / strike
    design = np.column_stack([
        np.ones_like(x), x, variance_one, variance_two, x * x,
        x * variance_one, x * variance_two, variance_one * variance_one,
        variance_one * variance_two, variance_two * variance_two,
    ])
    payoff = np.maximum(strike - spot, 0.0); itm = payoff > 0.0
    continuation = np.zeros(spot.size, dtype=float)
    if np.any(itm):
        target = cashflows[itm] * np.exp(-rate * dt * (exercise_steps[itm] - current_step))
        beta = np.linalg.lstsq(design[itm], target, rcond=None)[0]; continuation[itm] = design[itm] @ beta
    exercise = itm & (payoff > continuation)
    return np.vstack([np.where(exercise, payoff, cashflows),
                      np.where(exercise, current_step, exercise_steps)]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, collinear, and full-rank specifications."""
    return [
        {"setup": "import numpy as np\ns=np.linspace(75.,110.,18); v1=np.linspace(.08,.2,18); v2=np.linspace(.35,.12,18); c=np.linspace(20.,2.,18); e=np.full(18,5.)",
         "call": "float(np.sum(apply_double_heston_lsm_step(s,v1,v2,100.,3,c,e,.03,.1)*np.array([[1.],[.01]])))",
         "gold_call": "float(np.sum(_oracle_apply_double_heston_lsm_step(s,v1,v2,100.,3,c,e,.03,.1)*np.array([[1.],[.01]])))"},
        {"setup": "import numpy as np\ns=np.array([100.,115.]); v1=np.zeros(2); v2=np.zeros(2); c=np.zeros(2); e=np.full(2,2.)",
         "call": "float(np.sum(apply_double_heston_lsm_step(s,v1,v2,100.,1,c,e,0.,.25)))", "gold_call": "float(np.sum(_oracle_apply_double_heston_lsm_step(s,v1,v2,100.,1,c,e,0.,.25)))"},
        {"setup": "import numpy as np\ns=np.array([90.]); v1=np.array([.2]); v2=np.array([.3]); c=np.array([8.]); e=np.array([1.])",
         "call": "float(np.sum(apply_double_heston_lsm_step(s,v1,v2,100.,1,c,e,0.,.25)))", "gold_call": "float(np.sum(_oracle_apply_double_heston_lsm_step(s,v1,v2,100.,1,c,e,0.,.25)))"},
        {"setup": "import numpy as np\ns=np.linspace(82.,112.,12); v1=np.linspace(.03,.24,12); v2=np.linspace(.4,.08,12); c=np.array([19.,16.,14.,11.,10.,8.,7.,5.,4.,3.,2.,1.]); e=np.array([6.,5.,6.,4.,5.,3.,6.,4.,5.,6.,3.,4.])",
         "call": "float(np.sum(apply_double_heston_lsm_step(s,v1,v2,100.,2,c,e,.04,.08)*np.array([[1.],[.001]])))", "gold_call": "float(np.sum(_oracle_apply_double_heston_lsm_step(s,v1,v2,100.,2,c,e,.04,.08)*np.array([[1.],[.001]])))"},
        {"setup": "import numpy as np\nrng=np.random.default_rng(1); n=28; s=rng.uniform(70.,120.,n); v1=rng.uniform(.02,.5,n); v2=rng.uniform(.03,.65,n); c=rng.uniform(1.,35.,n); e=rng.integers(3,8,n).astype(float); w=np.vstack([np.linspace(.7,2.3,n),.001*np.arange(n,0,-1)])",
         "call": "float(np.sum(apply_double_heston_lsm_step(s.copy(),v1.copy(),v2.copy(),100.,2,c.copy(),e.copy(),.04,.08)*w))",
         "gold_call": "float(np.sum(_oracle_apply_double_heston_lsm_step(s.copy(),v1.copy(),v2.copy(),100.,2,c.copy(),e.copy(),.04,.08)*w))"},
    ]
