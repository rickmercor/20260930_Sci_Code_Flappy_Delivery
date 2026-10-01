"""
Return the mean and the relative scatter of each level's capacitance deflection over the temperatures of the active window, from the deflections recovered at every measured temperature.

A capacitance deflection is a property of the defect population, not of the temperature at which it happens to be measured, so a correctly recovered deflection is constant across temperature. That constancy is the built-in consistency check of the whole procedure: a level whose recovered deflection drifts is either being fitted with a wrong emission rate, or is being measured outside the range in which the acquisition carries information about it, and averaging over temperature is what converts a family of per-temperature estimates into one number with a meaningful uncertainty.




The range in which the acquisition does carry that information is bounded at both ends by the sampling grid rather than by the physics of the defect. Below it, the emission rate is so small that the level's exponential has barely begun to decay by the last sample, the fitted amplitude is degenerate with the quiescent capacitance, and the recovered deflection runs away. Above it, the emission rate is so large that the level has already fully emitted before the first sample, the exponential is indistinguishable from a step, and the amplitude is again undetermined. Between those bounds lies the active window, and for two levels of different depth it is the intersection of the two individual windows, so it is narrower than either.




Averaging outside the window does not merely add noise, it adds bias, because the failures at both ends are systematic and one-sided. The relative scatter inside the window is therefore reported alongside the mean: a scatter comparable to the measurement noise confirms that the window was chosen inside the informative range, while a scatter far exceeding it signals that the window still contains temperatures at which one of the levels was not properly resolved.

Returns
-------
np.ndarray of shape (n_levels, 2), float: window-mean capacitance deflection (pF) and its relative standard deviation (dimensionless).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def average_active_window(temperature: np.ndarray, deflection: np.ndarray,
                          window_low: float, window_high: float) -> np.ndarray:
    """Return the mean and relative scatter of each level's deflection in a window.

    Parameters
    ----------
    temperature : np.ndarray
        Temperatures in K at which the deflections were recovered, of length
        n_temperatures.
    deflection : np.ndarray
        Array of shape (n_temperatures, n_levels) holding the magnitude of the
        capacitance deflection of each level at each temperature, in pF. An
        identically zero level is valid and must not raise an exception.
    window_low : float
        Lower bound of the active window in K, inclusive.
    window_high : float
        Upper bound of the active window in K, inclusive
        (window_high >= window_low).

    Returns
    -------
    summary : np.ndarray
        Array of shape (n_levels, 2) whose columns are the mean capacitance
        deflection of the level over the window in pF and the standard
        deviation of that deflection divided by its mean. If a level's window
        mean is exactly zero, its relative scatter is np.inf.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.
    """
    return summary  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_average_active_window(temperature: np.ndarray, deflection: np.ndarray,
                                  window_low: float, window_high: float) -> np.ndarray:
    import numpy as np

    temperature = np.asarray(temperature, dtype=float)
    deflection = np.asarray(deflection, dtype=float)

    if deflection.ndim != 2:
        raise ValueError("deflection must have shape (n_temperatures, n_levels)")
    if temperature.ndim != 1 or temperature.size != deflection.shape[0]:
        raise ValueError("temperature length must match the first axis of deflection")
    for name, value in (("window_low", window_low), ("window_high", window_high)):
        if (isinstance(value, bool)
                or not isinstance(value, (int, float, np.floating, np.integer))
                or not np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
    if float(window_high) < float(window_low):
        raise ValueError("window_high must not be below window_low")

    inside = (temperature >= float(window_low)) & (temperature <= float(window_high))
    if not np.any(inside):
        raise ValueError("the active window contains no measured temperature")

    selected = deflection[inside]

    mean = selected.mean(axis=0)
    # Population standard deviation, so a window holding one temperature gives
    # a scatter of exactly zero rather than an undefined value.
    scatter = selected.std(axis=0)

    summary = np.empty((deflection.shape[1], 2), dtype=float)
    summary[:, 0] = mean
    # Guard the division so a level recovered as identically zero reports an
    # infinite relative scatter rather than raising a divide warning.
    nonzero = mean != 0.0
    summary[:, 1] = np.inf
    summary[nonzero, 1] = scatter[nonzero] / np.abs(mean[nonzero])

    return summary

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark window over the full measured temperature span ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3)
temperature = np.arange(300.0, 480.0 + 1e-9, 3.0)
inside = (temperature >= 370.0) & (temperature <= 450.0)
deflection = np.column_stack([
    np.where(inside, 0.7506, 0.7506 * np.exp(0.4 * rng.standard_normal(temperature.size))),
    np.where(inside, 0.0538, 0.0538 * np.exp(0.8 * rng.standard_normal(temperature.size))),
])
""",
            "call": "average_active_window(temperature, deflection, 370.0, 450.0) + 1000.0",
            "gold_call": "_oracle_average_active_window(temperature, deflection, 370.0, 450.0) + 1000.0",
        },
        # --- Valid: a window spilling outside the informative range inflates scatter ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3)
temperature = np.arange(300.0, 480.0 + 1e-9, 3.0)
inside = (temperature >= 370.0) & (temperature <= 450.0)
deflection = np.column_stack([
    np.where(inside, 0.7506, 0.7506 * np.exp(0.4 * rng.standard_normal(temperature.size))),
    np.where(inside, 0.0538, 0.0538 * np.exp(0.8 * rng.standard_normal(temperature.size))),
])
""",
            "call": "average_active_window(temperature, deflection, 300.0, 480.0) + 1000.0",
            "gold_call": "_oracle_average_active_window(temperature, deflection, 300.0, 480.0) + 1000.0",
        },
        # --- Valid: the deflection ratio of the two levels inside the window ---
        {
            "setup": """import numpy as np
temperature = np.arange(360.0, 460.0 + 1e-9, 5.0)
deflection = np.column_stack([
    0.75 + 0.002 * np.sin(temperature),
    0.054 + 0.001 * np.cos(temperature),
])
""",
            "call": "average_active_window(temperature, deflection, 370.0, 450.0) + 1000.0",
            "gold_call": "_oracle_average_active_window(temperature, deflection, 370.0, 450.0) + 1000.0",
        },
        # --- Boundary: a window holding exactly one temperature ---
        {
            "setup": """import numpy as np
temperature = np.arange(300.0, 480.0 + 1e-9, 3.0)
deflection = np.column_stack([np.full(temperature.size, 0.75), np.full(temperature.size, 0.05)])
""",
            "call": "average_active_window(temperature, deflection, 399.0, 399.0) + 1000.0",
            "gold_call": "_oracle_average_active_window(temperature, deflection, 399.0, 399.0) + 1000.0",
        },
        # --- Edge: an identically zero level is valid and has infinite scatter ---
        {
            "setup": """import numpy as np
temperature = np.arange(370.0, 450.0 + 1e-9, 10.0)
deflection = np.column_stack([
    np.full(temperature.size, 0.75),
    np.full(temperature.size, 0.05),
    np.zeros(temperature.size),
])
""",
            "call": "float(np.sum(average_active_window(temperature, deflection, 370.0, 450.0)[:2]))",
            "gold_call": "float(np.sum(_oracle_average_active_window(temperature, deflection, 370.0, 450.0)[:2]))",
        },
        # --- Invalid: an empty active window ---
        {
            "setup": """import numpy as np
temperature = np.arange(300.0, 480.0 + 1e-9, 3.0)
deflection = np.column_stack([np.full(temperature.size, 0.75), np.full(temperature.size, 0.05)])
def run_model():
    try:
        average_active_window(temperature, deflection, 500.0, 520.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_average_active_window(temperature, deflection, 500.0, 520.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: inverted window bounds ---
        {
            "setup": """import numpy as np
temperature = np.arange(300.0, 480.0 + 1e-9, 3.0)
deflection = np.column_stack([np.full(temperature.size, 0.75), np.full(temperature.size, 0.05)])
def run_model():
    try:
        average_active_window(temperature, deflection, 450.0, 370.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_average_active_window(temperature, deflection, 450.0, 370.0)
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
