"""
Discount each selected stopping payoff to time zero and average across paths.

A Bermudan path contributes one payoff at its selected stopping date, so discounting is path-specific.

Returns
-------
float, the time-zero discounted Monte Carlo mean as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def price_discounted_cashflows(
    cashflows: "np.ndarray", exercise_steps: "np.ndarray", rate: float, dt: float,
) -> float:
    r"""Return the time-zero Monte Carlo mean of stopping cashflows.

    Parameters
    ----------
    cashflows : numpy.ndarray
        Finite nonnegative realized payoff per path.
    exercise_steps : numpy.ndarray
        Nonnegative stopping-step index per path.
    rate : float
        Finite continuously compounded risk-free rate.
    dt : float
        Positive interval length.

    Returns
    -------
    price : float
        Discounted path mean as a native Python float.
    Raises
    ------
    ValueError
        If cashflow states or discounting inputs violate the stated contract.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_price_discounted_cashflows(
    cashflows: "np.ndarray", exercise_steps: "np.ndarray", rate: float, dt: float,
) -> float:
    import math
    try:
        cashflows = [float(value) for value in cashflows]
        exercise_steps = [float(value) for value in exercise_steps]
    except (TypeError, ValueError) as exc:
        raise ValueError("cashflow state must contain real scalars") from exc
    if not cashflows or len(exercise_steps) != len(cashflows):
        raise ValueError("cashflows and exercise_steps must be equal nonempty vectors")
    values = cashflows + exercise_steps
    if not all(math.isfinite(value) for value in values):
        raise ValueError("cashflow state must be finite")
    if any(value < 0.0 for value in values):
        raise ValueError("cashflows and exercise steps must be nonnegative")
    if not math.isfinite(rate) or not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("rate must be finite and dt must be positive")
    discounted = (cash * math.exp(-rate * dt * step)
                  for cash, step in zip(cashflows, exercise_steps))
    return float(sum(discounted) / len(cashflows))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge test specifications."""
    return [
        {
            "setup": """import numpy as np
c=np.array([5.,10.,3.]); e=np.array([1.,2.,4.])
""",
            "call": "price_discounted_cashflows(c,e,.03,.25)",
            "gold_call": "_oracle_price_discounted_cashflows(c,e,.03,.25)",
        },
        {
            "setup": """import numpy as np
c=np.array([1.,2.,3.]); e=np.array([0.,5.,9.])
""",
            "call": "price_discounted_cashflows(c,e,0.,.1)",
            "gold_call": "_oracle_price_discounted_cashflows(c,e,0.,.1)",
        },
        {
            "setup": """import numpy as np
c=np.zeros(4); e=np.array([1.,2.,3.,4.])
""",
            "call": "price_discounted_cashflows(c,e,-.01,.5)",
            "gold_call": "_oracle_price_discounted_cashflows(c,e,-.01,.5)",
        },
    ]
