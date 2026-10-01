"""
Combine the reconstructed excursion length with the crossing probability into the transition rate.

Interface sampling factorises a rate into a frequency and a probability. The frequency is the conditional flux out of the reactant state: the reciprocal of the mean interval between two successive upward crossings of the reactant boundary, counting only the time the trajectory was last in the reactant state. That interval is the time spent per excursion below the boundary plus the mean duration of a full excursion above it, and only the first of the two is measured directly by a partial-path scheme. The probability is the chance that a given upward crossing reaches the product state instead of falling back, and the product of frequency and probability is the rate.

The factorisation matters beyond bookkeeping because it is an independent route to the same number. The mean duration of one visit to the reactant state can also be obtained in a single pass as a mean first passage time to the product state, and the reciprocal of that duration is the rate as well. The two routes share the transition matrix but not the boundary conditions, the destination sets or the arithmetic, so their agreement to machine precision is a strong test that the chain was wired consistently and that the overlap accounting neither lost nor duplicated time. A discrepancy of even a percent points to a misassigned boundary state rather than to numerical noise.

Formulas:

    f = 1 / ((tau_reactant + tau_transit) * time_step)

    k = f * P_A(lambda_B | lambda_A)

Returns
-------
float: the rate constant in the reciprocal time unit implied by time_step, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_rate_constant(reactant_time: float, conditional_passage_time: float,
                          crossing_probability: float, time_step: float) -> float:
    """Combine the reconstructed excursion length and the crossing probability.

    Parameters
    ----------
    reactant_time : float
        Mean time spent per excursion below the reactant boundary, in phase
        points. Must be non-negative.
    conditional_passage_time : float
        Mean duration of one full excursion above the reactant boundary, in
        phase points. Must be positive.
    crossing_probability : float
        Probability that an upward crossing of the reactant boundary reaches the
        product state before falling back. Must lie in (0, 1].
    time_step : float
        Physical duration of one phase point, expressed in the time unit whose
        reciprocal the rate is reported in. Must be positive.

    Returns
    -------
    rate_constant : float
        Rate constant of the transition into the product state, in the
        reciprocal of the time unit of time_step, as a native Python float.

    Raises
    ------
    ValueError
        If crossing_probability is not in (0, 1] or if time_step is not
        positive.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_rate_constant(reactant_time: float, conditional_passage_time: float,
                                  crossing_probability: float, time_step: float) -> float:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    values = {"reactant_time": reactant_time,
              "conditional_passage_time": conditional_passage_time,
              "crossing_probability": crossing_probability,
              "time_step": time_step}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if float(reactant_time) < 0.0:
        raise ValueError("reactant_time must be non-negative")
    if float(conditional_passage_time) <= 0.0:
        raise ValueError("conditional_passage_time must be positive")
    if not 0.0 < float(crossing_probability) <= 1.0:
        raise ValueError("crossing_probability must lie in (0, 1]")
    if float(time_step) <= 0.0:
        raise ValueError("time_step must be positive")

    # The interval between two successive upward crossings of the reactant
    # boundary spans one excursion below it and one full excursion above it.
    interval = (float(reactant_time) + float(conditional_passage_time)) * float(time_step)
    flux = 1.0 / interval

    return float(flux * float(crossing_probability))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the measured configuration, rate in inverse picoseconds ---
        {
            "setup": """import numpy as np
reactant_time = 43.7
conditional_passage_time = 118.37185470338312
crossing_probability = 0.030623080431117063
time_step = 0.02
""",
            "call": "compute_rate_constant(reactant_time, conditional_passage_time, crossing_probability, time_step)",
            "gold_call": "_oracle_compute_rate_constant(reactant_time, conditional_passage_time, crossing_probability, time_step)",
        },
        # --- Valid: a deep reactant basin, where the wait dominates ---
        {
            "setup": """import numpy as np
reactant_time = 3061.4
conditional_passage_time = 123.2072941835
crossing_probability = 0.0051837
time_step = 0.02
""",
            "call": "compute_rate_constant(reactant_time, conditional_passage_time, crossing_probability, time_step)",
            "gold_call": "_oracle_compute_rate_constant(reactant_time, conditional_passage_time, crossing_probability, time_step)",
        },
        # --- Boundary: no time below the boundary, every crossing commits ---
        {
            "setup": """import numpy as np
reactant_time = 0.0
conditional_passage_time = 40.0
crossing_probability = 1.0
time_step = 0.02
""",
            "call": "compute_rate_constant(reactant_time, conditional_passage_time, crossing_probability, time_step)",
            "gold_call": "_oracle_compute_rate_constant(reactant_time, conditional_passage_time, crossing_probability, time_step)",
        },
        # --- Edge: a very rare event and a different phase-point spacing ---
        {
            "setup": """import numpy as np
reactant_time = 5000.0
conditional_passage_time = 900.0
crossing_probability = 2.47e-14
time_step = 0.005
""",
            "call": "compute_rate_constant(reactant_time, conditional_passage_time, crossing_probability, time_step)",
            "gold_call": "_oracle_compute_rate_constant(reactant_time, conditional_passage_time, crossing_probability, time_step)",
        },
        # --- Invalid: a crossing probability of zero has no defined rate ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_rate_constant(43.7, 118.4, 0.0, 0.02)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_rate_constant(43.7, 118.4, 0.0, 0.02)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive phase-point duration ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_rate_constant(43.7, 118.4, 0.03, -0.02)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_rate_constant(43.7, 118.4, 0.03, -0.02)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
