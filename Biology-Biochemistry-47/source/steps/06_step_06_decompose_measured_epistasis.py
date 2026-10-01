"""
Split the epistasis measured in a real double mutant into the part manufactured by the catalytic cycle and the part that requires a genuine interaction between the two substitutions.

A laboratory dataset that reports every microscopic rate constant for a wild type, two single mutants and their combination carries strictly more information than the steady-state parameters alone, and that surplus is what makes a decomposition possible: an expectation for the double mutant can be built at the level of the observable, as published epistasis values almost always are, and another can be built one level lower, from the elementary steps themselves.




The step reports, for each of the four steady-state observables, the epistasis measured against the conventional expectation, the component of it that the mechanism manufactures out of substitutions that do not interact, and the component that survives once the manufactured part is accounted for. The practical consequence is that a conventional value can understate, overstate or even reverse the true interaction, according to whether the two components push the same way or against each other.

Returns
-------
np.ndarray of shape (4, 3), float: for each of the four kinetic parameters, the conventional epistasis, the mechanism-manufactured component and the genuine-interaction component.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def decompose_measured_epistasis(rate_table) -> np.ndarray:
    """Split measured double-mutant epistasis into its two components.

    Parameters
    ----------
    rate_table : array_like
        Array of shape (4, 5) whose rows are the measured microscopic rate
        constants of the wild type, the first single mutant, the second
        single mutant and the double mutant, each ordered as association,
        dissociation, forward chemical, reverse chemical and release.

    Returns
    -------
    decomposition : np.ndarray
        Array of shape (4, 3) whose rows are the dissociation constant, the
        turnover number, the Michaelis constant and the specificity constant,
        and whose columns are the epistasis against the conventional null
        model, the component manufactured by the catalytic cycle, and the
        component requiring a genuine interaction.

    Raises
    ------
    ValueError
        If rate_table does not have shape (4, 5), holds a non-finite or
        negative entry, or holds a zero among the wild-type rate constants,
        which would leave the fold-changes undefined.
    """
    return decomposition  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np 

def _oracle_decompose_measured_epistasis(rate_table) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _parameters(rates):
        """Steady-state observables of the three-step cycle."""
        k_on, k_off = rates[..., 0], rates[..., 1]
        k_chem, k_rev, k_rel = rates[..., 2], rates[..., 3], rates[..., 4]
        partition = k_chem + k_rev + k_rel
        capture = k_chem * k_rel + k_off * k_rev + k_off * k_rel
        if np.any(partition <= 0.0) or np.any(capture <= 0.0):
            raise ValueError("the cycle must carry non-zero steady-state flux")
        return np.stack([k_off / k_on,
                         k_chem * k_rel / partition,
                         capture / (k_on * partition),
                         k_on * k_chem * k_rel / capture], axis=-1)

    table = np.asarray(rate_table, dtype=float)
    if table.shape != (4, 5):
        raise ValueError("rate_table must have shape (4, 5)")
    if not np.all(np.isfinite(table)):
        raise ValueError("rate_table must be finite")
    if np.any(table < 0.0):
        raise ValueError("rate constants must be non-negative")
    if np.any(table[:, 0] <= 0.0):
        raise ValueError("every association rate constant must be > 0")
    if np.any(table[0] <= 0.0):
        raise ValueError("every wild-type rate constant must be > 0 to define fold-changes")

    wild, first, second = table[0], table[1], table[2]

    # Expectation built one level below the observables: free energy
    # additivity multiplies the rate-constant fold-changes step by step.
    additive_rates = wild * (first / wild) * (second / wild)

    parameters = _parameters(table)
    wt_parameters = parameters[0]
    conventional = wt_parameters * (parameters[1] / wt_parameters) * (parameters[2] / wt_parameters)
    additive = _parameters(additive_rates)
    observed = parameters[3]

    total = observed / conventional
    manufactured = additive / conventional
    genuine = observed / additive

    return np.stack([total, manufactured, genuine], axis=-1)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np 

def test_cases():
    """Return list of test case specifications."""
    import numpy as np  # noqa: F401  (keeps this field self-contained)

    return [
        # --- Valid: the characterised double mutant of the task (normal scenario) ---
        {
            "setup": """import numpy as np
rate_table = np.array([[1.20e6, 850.0, 340.0, 4.50, 95.0],
                       [7.80e5, 1190.0, 6.12, 5.40, 5.70],
                       [2.82e6, 595.0, 34.0, 3.82, 14.2],
                       [2.75e6, 2500.0, 2.45, 5.97, 2.22]])
""",
            "call": "decompose_measured_epistasis(rate_table)",
            "gold_call": "_oracle_decompose_measured_epistasis(rate_table)",
        },
        # --- Valid: a strictly additive double mutant, no genuine interaction anywhere ---
        {
            "setup": """import numpy as np
wild = np.array([3.0e5, 60.0, 12.0, 0.014, 0.40])
first = wild * np.array([0.4, 2.2, 0.09, 1.7, 0.35])
second = wild * np.array([1.9, 0.6, 2.7, 0.4, 3.1])
rate_table = np.stack([wild, first, second, wild * (first / wild) * (second / wild)])
""",
            "call": "decompose_measured_epistasis(rate_table)",
            "gold_call": "_oracle_decompose_measured_epistasis(rate_table)",
        },
        # --- Boundary: two neutral substitutions, every row identical to the wild type ---
        {
            "setup": """import numpy as np
wild = np.array([1.20e6, 850.0, 340.0, 4.50, 95.0])
rate_table = np.stack([wild, wild, wild, wild])
""",
            "call": "decompose_measured_epistasis(rate_table)",
            "gold_call": "_oracle_decompose_measured_epistasis(rate_table)",
        },
        # --- Edge: an all but irreversible chemical step, and a release-limited cycle ---
        {
            "setup": """import numpy as np
rate_table = np.array([[5.0e6, 2.0e4, 1.0e4, 1.0e-8, 12.0],
                       [5.0e6, 2.0e4, 30.0, 4.0e-8, 11.0],
                       [1.0e6, 8.0e3, 9.0e3, 2.0e-9, 0.15],
                       [9.0e5, 3.0e4, 55.0, 5.0e-9, 0.30]])
""",
            "call": "decompose_measured_epistasis(rate_table)",
            "gold_call": "_oracle_decompose_measured_epistasis(rate_table)",
        },
        # --- Invalid: table with the wrong number of variants ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        decompose_measured_epistasis(np.ones((3, 5)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_decompose_measured_epistasis(np.ones((3, 5)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a wild-type rate constant of zero, so fold-changes are undefined ---
        {
            "setup": """import numpy as np
bad = np.array([[1.20e6, 850.0, 340.0, 0.0, 95.0],
                [7.80e5, 1190.0, 6.12, 5.40, 5.70],
                [2.82e6, 595.0, 34.0, 3.82, 14.2],
                [2.75e6, 2500.0, 2.45, 5.97, 2.22]])
def run_model():
    try:
        decompose_measured_epistasis(bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_decompose_measured_epistasis(bad)
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
