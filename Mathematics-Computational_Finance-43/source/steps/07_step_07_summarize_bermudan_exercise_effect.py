"""
Summarize the maturity-only value and the contribution produced by early exercise.

The diagnostic separates state-simulation effects from the stopping rule without changing the Bermudan estimator.

Returns
-------
np.ndarray, [maturity-only value, early-stopping count, mean increment per early-stopping path] as a float array with shape (3,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def summarize_bermudan_exercise_effect(
    bermudan_price: float, terminal_payoffs: "np.ndarray",
    exercise_steps: "np.ndarray", rate: float, dt: float, maturity_step: int,
) -> "np.ndarray":
    r"""Return three diagnostics that reconstruct the Bermudan price.

    Parameters
    ----------
    bermudan_price : float
        Finite nonnegative time-zero stopping-policy mean.
    terminal_payoffs, exercise_steps : numpy.ndarray
        Equal nonempty vectors of maturity payoffs and selected stopping indices.
    rate, dt, maturity_step : float, float, int
        Finite rate, positive interval length, and positive terminal index.

    Returns
    -------
    diagnostics : numpy.ndarray
        ``[maturity_only_value, early_count, mean_early_increment]``.

    Raises
    ------
    ValueError
        If arrays, scalars, or stopping indices violate the stated contract.
    """
    return diagnostics

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_summarize_bermudan_exercise_effect(
    bermudan_price: float, terminal_payoffs: "np.ndarray",
    exercise_steps: "np.ndarray", rate: float, dt: float, maturity_step: int,
) -> "np.ndarray":
    import numpy as np
    terminal = np.asarray(terminal_payoffs, dtype=float)
    stopping = np.asarray(exercise_steps, dtype=float)
    if terminal.ndim != 1 or terminal.size == 0 or stopping.shape != terminal.shape:
        raise ValueError("payoff and stopping arrays must be equal nonempty vectors")
    if not np.all(np.isfinite(terminal)) or not np.all(np.isfinite(stopping)):
        raise ValueError("payoff and stopping arrays must be finite")
    if np.any(terminal < 0.0) or np.any(stopping < 0.0):
        raise ValueError("payoffs and stopping indices must be nonnegative")
    if not np.isfinite(bermudan_price) or bermudan_price < 0.0:
        raise ValueError("bermudan_price must be finite and nonnegative")
    if not np.isfinite(rate) or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("rate must be finite and dt must be positive")
    if isinstance(maturity_step, bool) or not isinstance(maturity_step, (int, np.integer)):
        raise ValueError("maturity_step must be an integer")
    if maturity_step < 1 or np.any(stopping > maturity_step):
        raise ValueError("stopping indices cannot exceed a positive maturity_step")
    maturity_steps = np.full(terminal.size, maturity_step, dtype=float)
    maturity_value = _oracle_price_discounted_cashflows(
        terminal, maturity_steps, rate, dt
    )
    early_count = int(np.count_nonzero(stopping < maturity_step))
    premium = float(bermudan_price - maturity_value)
    mean_increment = 0.0 if early_count == 0 else premium * terminal.size / early_count
    return np.array([maturity_value, early_count, mean_increment], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and mixed-stopping specifications."""
    return [
        {"setup": "import numpy as np\nt=np.array([4.,7.,0.,3.]); e=np.array([1.,3.,3.,2.]); w=np.array([2.,.1,.5])",
         "call": "float(np.dot(summarize_bermudan_exercise_effect(4.8,t,e,.03,.1,3),w))",
         "gold_call": "float(np.dot(_oracle_summarize_bermudan_exercise_effect(4.8,t,e,.03,.1,3),w))"},
        {"setup": "import numpy as np\nt=np.array([1.,2.]); e=np.array([2.,2.]); w=np.array([1.,2.,3.])",
         "call": "float(np.dot(summarize_bermudan_exercise_effect(1.5,t,e,0.,.2,2),w))",
         "gold_call": "float(np.dot(_oracle_summarize_bermudan_exercise_effect(1.5,t,e,0.,.2,2),w))"},
        {"setup": "import numpy as np\nt=np.array([0.,9.,1.]); e=np.array([0.,4.,2.]); w=np.array([.5,1.,2.])",
         "call": "float(np.dot(summarize_bermudan_exercise_effect(3.7,t,e,-.02,.25,4),w))",
         "gold_call": "float(np.dot(_oracle_summarize_bermudan_exercise_effect(3.7,t,e,-.02,.25,4),w))"},
        {"setup": "import numpy as np\nt=np.array([2.,5.,1.,4.,8.]); e=np.array([5.,3.,5.,1.,4.]); w=np.array([1.3,.07,-.2])",
         "call": "float(np.dot(summarize_bermudan_exercise_effect(4.2,t,e,.07,.05,5),w))",
         "gold_call": "float(np.dot(_oracle_summarize_bermudan_exercise_effect(4.2,t,e,.07,.05,5),w))"},
    ]
