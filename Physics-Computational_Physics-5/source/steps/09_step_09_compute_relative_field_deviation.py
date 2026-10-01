"""
Measure how far one recorded field history departs from a reference history over the whole run, as a single dimensionless number.

Two closures that damp at slightly different rates and oscillate at slightly different frequencies separate in both amplitude and phase, so a fidelity measure that keeps the complex amplitude sees the phase drift that an envelope comparison discards. Normalising by the reference over the same levels makes the measure independent of the amplitude of the initial perturbation.

Returns
-------
float: the dimensionless relative departure of the test history from the reference, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_relative_field_deviation(reference_history: np.ndarray,
                                     test_history: np.ndarray) -> float:
    """Measure the relative departure of a field history from a reference.

    The measure is the root mean square, taken over all recorded time levels,
    of the magnitude of the difference between the two histories, divided by
    the root mean square of the magnitude of the reference history over the
    same levels.

    Parameters
    ----------
    reference_history : np.ndarray
        Complex array of shape (n_levels,) holding the reference field
        amplitude at each time level, as returned by sub-problem 08.
    test_history : np.ndarray
        Complex array of the same shape holding the field amplitude of the
        history being assessed.

    Returns
    -------
    deviation : float
        The dimensionless relative departure, as a native Python float.

    Raises
    ------
    ValueError
        If either input is not a finite one-dimensional array with at least one
        entry, if the two shapes differ, or if the reference history vanishes at
        every level.
    """
    return deviation  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_relative_field_deviation(reference_history: np.ndarray,
                                             test_history: np.ndarray) -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    reference = np.asarray(reference_history, dtype=complex)
    test = np.asarray(test_history, dtype=complex)
    for name, array in (("reference_history", reference), ("test_history", test)):
        if array.ndim != 1 or array.size < 1:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
    if reference.shape != test.shape:
        raise ValueError("the two histories must have the same shape")

    # The level count cancels between the two mean squares, so the measure is
    # the ratio of the two accumulated squared magnitudes.
    reference_power = float(np.sum(np.abs(reference) ** 2))
    if reference_power <= 0.0:
        raise ValueError("reference_history must not vanish at every level")
    difference_power = float(np.sum(np.abs(test - reference) ** 2))
    return float(np.sqrt(difference_power / reference_power))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: two damped oscillations differing in both rate and frequency
        #     (normal scenario) ---
        {
            "setup": """import numpy as np
levels = np.arange(400, dtype=float) * 0.05
reference_history = 0.05 * np.exp((-0.066 - 1.285j) * levels)
test_history = 0.05 * np.exp((-0.058 - 1.190j) * levels)
""",
            "call": "round(compute_relative_field_deviation(reference_history, test_history), 9)",
            "gold_call": "round(_oracle_compute_relative_field_deviation(reference_history, test_history), 9)",
        },
        # --- Valid: histories differing only in damping rate, so the measure sees
        #     no phase drift at all ---
        {
            "setup": """import numpy as np
levels = np.arange(250, dtype=float) * 0.02
reference_history = np.exp(-0.30 * levels) * np.exp(-2.0j * levels)
test_history = np.exp(-0.42 * levels) * np.exp(-2.0j * levels)
""",
            "call": "round(compute_relative_field_deviation(reference_history, test_history), 9)",
            "gold_call": "round(_oracle_compute_relative_field_deviation(reference_history, test_history), 9)",
        },
        # --- Boundary: identical histories, for which the measure must vanish ---
        {
            "setup": """import numpy as np
levels = np.arange(30, dtype=float)
reference_history = 0.02 * np.exp(-0.1j * levels)
test_history = reference_history.copy()
""",
            "call": "round(compute_relative_field_deviation(reference_history, test_history), 9)",
            "gold_call": "round(_oracle_compute_relative_field_deviation(reference_history, test_history), 9)",
        },
        # --- Boundary: a vanishing test history, for which the measure is one ---
        {
            "setup": """import numpy as np
levels = np.arange(30, dtype=float)
reference_history = 0.02 * np.exp(-0.1j * levels)
test_history = np.zeros(30, dtype=complex)
""",
            "call": "round(compute_relative_field_deviation(reference_history, test_history), 9)",
            "gold_call": "round(_oracle_compute_relative_field_deviation(reference_history, test_history), 9)",
        },
        # --- Edge: a single level, where the measure reduces to a plain
        #     magnitude ratio ---
        {
            "setup": """import numpy as np
reference_history = np.array([0.02 - 0.01j])
test_history = np.array([0.014 + 0.003j])
""",
            "call": "round(compute_relative_field_deviation(reference_history, test_history), 9)",
            "gold_call": "round(_oracle_compute_relative_field_deviation(reference_history, test_history), 9)",
        },
        # --- Invalid: mismatched history lengths ---
        {
            "setup": """import numpy as np
reference_history = np.ones(5, dtype=complex)
test_history = np.ones(6, dtype=complex)
def run_model():
    try:
        compute_relative_field_deviation(reference_history, test_history)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_relative_field_deviation(reference_history, test_history)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a reference that vanishes at every level, which leaves the
        #     relative measure undefined ---
        {
            "setup": """import numpy as np
reference_history = np.zeros(8, dtype=complex)
test_history = np.ones(8, dtype=complex)
def run_model():
    try:
        compute_relative_field_deviation(reference_history, test_history)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_relative_field_deviation(reference_history, test_history)
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
