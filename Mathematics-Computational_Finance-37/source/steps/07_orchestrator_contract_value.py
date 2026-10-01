"""
Orchestrates the full four-stage chain: calibrate the SPX-VIX joint law (the equity-index process driving the fund), marginalize it over the volatility coordinates to obtain the (S1,S2,S3) law the contract actually uses, compute the survival probability at one and two years (the mortality process governing the death-benefit weighting), and combine both into the surrenderable contract's backward obstacle recursion. The available step functions are spx_vix_calibration, marginalize_to_sss, survival_probability, and contract_value; determining the correct arguments and composition order for each is part of this step.

This step introduces no new scientific content of its own; it wires together the outputs of the three preceding steps into the final scalar. Using each step's own function only (not re-deriving any of their internals) is itself the discriminating requirement here.

Returns
-------
value : float, the contract's fair initial value, a single deterministic real number.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_contract_value() -> float:
    """Run the full pipeline and return the fair initial value of the
    3-year surrenderable equity-linked contract, using the task's exact
    instance inputs (see the problem statement for all numeric values).

    Returns
    -------
    value : float
        The contract's fair initial value, a single deterministic real
        number.

    Raises
    ------
    TypeError
        If called with any argument (this function takes none; the task
        instance is fixed).
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

# NOTE for Studio authoring: the model-facing compute_contract_value (once
# implemented by a candidate) should call the PUBLIC function names from
# sub_problems 02, 03, 05, 06 (spx_vix_calibration, marginalize_to_sss,
# survival_probability, contract_value), since only those are available to
# it. The GOLD _oracle_compute_contract_value below must instead call the
# _oracle_-prefixed versions of those same steps -- calling the public
# names there would let a candidate's own buggy step implementations leak
# into what is supposed to be the independent reference answer, letting a
# broken pipeline pass this step's comparison.


def _oracle_compute_contract_value() -> float:
    import numpy as np
    S_grid = np.array([0.7046880897187134, 0.8394570207692074, 1.0, 1.191246216612358, 1.4190675485932571])
    V_grid = np.array([0.12, 0.22, 0.32])
    tau = 30 / 365
    muS = np.array([0.06, 0.2, 0.48, 0.2, 0.06])
    muS = muS / muS.sum()
    muV = np.array([0.3, 0.45, 0.25])
    pi = _oracle_spx_vix_calibration(S_grid, V_grid, tau, muS, muV, Kout=6, Kin=40)
    joint123 = _oracle_marginalize_to_sss(pi)

    k, sigma = 0.3, 0.35
    theta = -2.900422093749666
    x0 = -3.101092789211817
    Q1 = _oracle_survival_probability(k, sigma, theta, x0, 1.0)
    Q2 = _oracle_survival_probability(k, sigma, theta, x0, 2.0)

    D, gm, alpha_s = 100.0, 0.02, 1.0
    return _oracle_contract_value(joint123, S_grid, D, gm, alpha_s, Q1, Q2)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal case: the actual task instance, full end-to-end run ---
        {
            "setup": """def run_model():
    return compute_contract_value()
def run_gold():
    return _oracle_compute_contract_value()
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 8.0,
        },
        # --- Boundary case: determinism. compute_contract_value() takes no
        # arguments, so its output cannot be meaningfully varied per test
        # case without changing the task instance itself; the property this
        # case checks instead is that repeated calls agree exactly, the
        # boundary of what "a single deterministic real number" (per the
        # docstring) requires -- no unseeded randomness or hidden state
        # leaking across calls. ---
        {
            "setup": """def run_model():
    a = compute_contract_value()
    b = compute_contract_value()
    return abs(a - b)
def run_gold():
    return 0.0
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 1e-9,
        },
        # --- Edge case: compute_contract_value takes zero arguments by
        # design (the task instance is fixed); calling it with an
        # unexpected positional argument must raise TypeError rather than
        # silently accepting or ignoring it. ---
        {
            "setup": """def run_model():
    try:
        compute_contract_value(1)
        return 0
    except TypeError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_contract_value(1)
        return 0
    except TypeError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
