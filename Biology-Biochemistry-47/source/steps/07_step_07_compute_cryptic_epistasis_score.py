"""
Express a measured enzyme's manufactured epistasis in units of the manufactured epistasis its own mechanism generates for typical non-interacting variant pairs.

A fold departure from a null model is meaningless until it is compared with something. Quoting a manufactured epistasis of a given size says nothing about whether the mechanism is unusually deceptive for this particular pair of substitutions or merely behaving as it does for any pair, because the size of the artefact depends on where the wild-type rate constants sit relative to one another and on how strongly the two substitutions perturb the elementary steps. The reference library supplies the missing denominator, being a sweep of variant pairs that are guaranteed by construction not to interact and are evaluated on the same wild-type reaction coordinate.




Both quantities are fold-changes and are therefore compared on the logarithmic scale, taking the magnitude so that a diminishing artefact and an enhancing one of the same size score alike. A score near one says the measured pair is unremarkable for this mechanism; a score well above one says the artefact is exceptional, which is the case worth reporting, because the mechanistic interpretation attached to the conventional epistasis value is then correspondingly unsafe.

Returns
-------
float: the measured manufactured epistasis, on a base-ten logarithmic magnitude scale, in units of the reference library median, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def compute_cryptic_epistasis_score(manufactured_epistasis: float,
                                    median_reference_magnitude: float) -> float:
    """Score a measured artefact against the mechanism's own reference scale.

    Parameters
    ----------
    manufactured_epistasis : float
        The component of the measured double mutant's epistasis that the
        catalytic cycle manufactures, expressed as a fold-change
        (manufactured_epistasis > 0).
    median_reference_magnitude : float
        The median absolute base-ten logarithm of the epistasis produced by
        the reference library of non-interacting variant pairs
        (median_reference_magnitude > 0).

    Returns
    -------
    score : float
        The absolute base-ten logarithm of the manufactured epistasis divided
        by the median reference magnitude, as a native Python float.

    Raises
    ------
    ValueError
        If either argument is not a finite number greater than zero, which
        would leave the score undefined.
    """
    return score  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np 

def _oracle_compute_cryptic_epistasis_score(manufactured_epistasis: float,
                                            median_reference_magnitude: float) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(manufactured_epistasis, (int, float, np.floating, np.integer))
            and not isinstance(manufactured_epistasis, bool)
            and np.isfinite(manufactured_epistasis) and float(manufactured_epistasis) > 0.0):
        raise ValueError("manufactured_epistasis must be a finite number > 0")
    if not (isinstance(median_reference_magnitude, (int, float, np.floating, np.integer))
            and not isinstance(median_reference_magnitude, bool)
            and np.isfinite(median_reference_magnitude)
            and float(median_reference_magnitude) > 0.0):
        raise ValueError("median_reference_magnitude must be a finite number > 0")

    magnitude = abs(np.log10(float(manufactured_epistasis)))

    return float(magnitude / float(median_reference_magnitude))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np 

def test_cases():
    """Return list of test case specifications."""
    import numpy as np  # noqa: F401  (keeps this field self-contained)

    return [
        # --- Valid: a diminishing artefact against a typical reference scale (normal scenario) ---
        {
            "setup": """import numpy as np
manufactured_epistasis = 0.4375
median_reference_magnitude = 0.1825
""",
            "call": "compute_cryptic_epistasis_score(manufactured_epistasis, median_reference_magnitude)",
            "gold_call": "_oracle_compute_cryptic_epistasis_score(manufactured_epistasis, median_reference_magnitude)",
        },
        # --- Valid: an enhancing artefact of the same logarithmic size ---
        {
            "setup": """import numpy as np
manufactured_epistasis = 1.0 / 0.4375
median_reference_magnitude = 0.1825
""",
            "call": "compute_cryptic_epistasis_score(manufactured_epistasis, median_reference_magnitude)",
            "gold_call": "_oracle_compute_cryptic_epistasis_score(manufactured_epistasis, median_reference_magnitude)",
        },
        # --- Boundary: a measured pair sitting exactly on the null value ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_cryptic_epistasis_score(1.0, 0.2)",
            "gold_call": "_oracle_compute_cryptic_epistasis_score(1.0, 0.2)",
        },
        # --- Edge: an almost non-deceptive mechanism, so the reference scale is tiny ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_cryptic_epistasis_score(2.5, 1.0e-9)",
            "gold_call": "_oracle_compute_cryptic_epistasis_score(2.5, 1.0e-9)",
        },
        # --- Invalid: a non-positive manufactured epistasis ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_cryptic_epistasis_score(0.0, 0.13)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_cryptic_epistasis_score(0.0, 0.13)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a reference scale of zero, so the score is undefined ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_cryptic_epistasis_score(2.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_cryptic_epistasis_score(2.0, 0.0)
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
