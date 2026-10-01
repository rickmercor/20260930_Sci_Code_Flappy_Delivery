"""
Chain the sub-problem functions 01-07 end to end and return the cryptic epistasis score of the characterised double mutant.

Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (compute_cycle_observables, simulate_mutant_free_energies, compute_variant_fold_changes, compute_double_mutant_epistasis, summarize_epistasis, decompose_measured_epistasis, compute_cryptic_epistasis_score) rather than reimplementing them.

This step runs the whole measurement end to end on one reaction coordinate, one reference library and one measured dataset, and returns a single number.




The returned scalar answers a question that neither half of the calculation can answer alone. The decomposition on its own gives a fold-change but no yardstick, since a given artefact may be typical of the mechanism or wildly atypical, and the two cases have opposite implications for how much weight the conventional epistasis value can bear. The simulation on its own gives a yardstick but no measurement. Putting them together states how far outside its own habitual behaviour the catalytic cycle was pushed by this particular pair of substitutions, and therefore how badly a structural interpretation drawn from the conventional epistasis value in this dataset would be misled.

Returns
-------
float: the cryptic epistasis score of the characterised double mutant for this configuration, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def run_cryptic_epistasis_pipeline(wt_free_energies=(0.0, 10.0, -5.0, 11.0, -9.0, 9.0),
                                   rate_table=None, n_mutants: int = 1000,
                                   amplitude: float = 2.0, seed: int = 314159,
                                   temperature: float = 298.15,
                                   threshold: float = 1.5,
                                   parameter_index: int = 3) -> float:
    """Run the full cryptic-epistasis measurement end to end.

    Parameters
    ----------
    wt_free_energies : array_like
        Six wild-type sub-state free energies in kcal/mol defining the
        reference reaction coordinate, in reaction coordinate order.
    rate_table : array_like or None
        Array of shape (4, 5) of measured microscopic rate constants for the
        wild type, both single mutants and the double mutant. None selects
        the characterised dataset of this task.
    n_mutants : int
        Number of single mutants in the reference library (n_mutants >= 1).
    amplitude : float
        Half-width in kcal/mol of the sub-state perturbation window
        (amplitude > 0).
    seed : int
        Seed of numpy's default bit generator (seed >= 0).
    temperature : float
        Absolute temperature in kelvin (temperature > 0).
    threshold : float
        Fold-change at which a departure from the null model counts as
        significant (threshold > 1).
    parameter_index : int
        Kinetic parameter to score, indexed as 1 for the turnover number, 2
        for the Michaelis constant and 3 for the specificity constant. The
        dissociation constant, index 0, is not admissible.

    Returns
    -------
    score : float
        The manufactured component of the measured double mutant's epistasis,
        on a base-ten logarithmic magnitude scale, in units of the reference
        library median, as a native Python float.

    Raises
    ------
    ValueError
        If parameter_index is not an integer in 1..3, if the reference
        reaction coordinate does not give finite observables, or if any
        other argument violates the constraints stated above.
    """
    return score  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_cryptic_epistasis_pipeline(
    wt_free_energies=(0.0, 10.0, -5.0, 11.0, -9.0, 9.0),
    rate_table=None,
    n_mutants: int = 1000,
    amplitude: float = 2.0,
    seed: int = 314159,
    temperature: float = 298.15,
    threshold: float = 1.5,
    parameter_index: int = 3,
) -> float:
    """Run the complete cryptic-epistasis calculation."""
    import numpy as np

    if rate_table is None:
        rate_table = np.array(
            [
                [1.20e6, 850.0, 340.0, 4.50, 95.0],
                [7.80e5, 1190.0, 6.12, 5.40, 5.70],
                [2.82e6, 595.0, 34.0, 3.82, 14.2],
                [2.75e6, 2500.0, 2.45, 5.97, 2.22],
            ],
            dtype=float,
        )

    if not (
        isinstance(parameter_index, (int, np.integer))
        and not isinstance(parameter_index, bool)
        and 1 <= int(parameter_index) <= 3
    ):
        raise ValueError("parameter_index must be an integer in 1..3")

    index = int(parameter_index)

    # Step 01: wild-type microscopic rates and observables.
    rate_constants_wt, wt_parameters = _oracle_compute_cycle_observables(
        wt_free_energies,
        temperature,
    )
    if not np.all(np.isfinite(wt_parameters)):
        raise ValueError(
            "the reference reaction coordinate must give finite observables"
        )

    # Steps 02–03: construct the reference single-mutant library.
    mutant_free_energies = _oracle_simulate_mutant_free_energies(
        wt_free_energies,
        n_mutants,
        amplitude,
        seed,
    )
    rate_folds, _ = _oracle_compute_variant_fold_changes(
        wt_free_energies,
        mutant_free_energies,
        temperature,
    )

    # Steps 04–05: derive observable expectations internally from microscopic
    # folds, then select and summarize the requested epistasis distribution.
    all_epistasis = _oracle_compute_double_mutant_epistasis(
        rate_constants_wt,
        rate_folds,
    )
    epistasis = all_epistasis[index]
    summary = _oracle_summarize_epistasis(epistasis, threshold)

    # Step 06: split measured double-mutant epistasis into components.
    decomposition = _oracle_decompose_measured_epistasis(rate_table)

    # Step 07: normalize the manufactured component by the library median.
    score = _oracle_compute_cryptic_epistasis_score(
        float(decomposition[index, 1]),
        float(summary[1]),
    )

    return float(score)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return whole-pipeline test specifications."""
    return [
        {
            # Normal: default specificity-constant score.
            "setup": """import numpy as np
n_mutants = 60
""",
            "call": "run_cryptic_epistasis_pipeline(n_mutants=n_mutants)",
            "gold_call": "_oracle_run_cryptic_epistasis_pipeline(n_mutants=n_mutants)",
        },
        {
            # Normal: Michaelis-constant score, selected after Step 04 returns
            # all four epistasis matrices.
            "setup": """import numpy as np
n_mutants = 80
parameter_index = 2
""",
            "call": (
                "run_cryptic_epistasis_pipeline("
                "n_mutants=n_mutants, parameter_index=parameter_index)"
            ),
            "gold_call": (
                "_oracle_run_cryptic_epistasis_pipeline("
                "n_mutants=n_mutants, parameter_index=parameter_index)"
            ),
        },
        {
            # Boundary: narrow perturbations and a strict significance threshold.
            "setup": """import numpy as np
""",
            "call": (
                "run_cryptic_epistasis_pipeline("
                "n_mutants=40, amplitude=0.25, seed=11, threshold=5.0)"
            ),
            "gold_call": (
                "_oracle_run_cryptic_epistasis_pipeline("
                "n_mutants=40, amplitude=0.25, seed=11, threshold=5.0)"
            ),
        },
        {
            # Edge: alternate reaction coordinate, measured table, and temperature.
            "setup": """import numpy as np
wt_free_energies = [0.0, 9.0, -3.5, 13.0, -6.0, 11.5]
rate_table = np.array([
    [3.0e5, 60.0, 12.0, 0.014, 0.40],
    [1.2e5, 132.0, 1.08, 0.0238, 0.14],
    [5.7e5, 36.0, 32.4, 0.0056, 1.24],
    [3.1e5, 95.0, 6.5, 0.0071, 0.62],
])
""",
            "call": (
                "run_cryptic_epistasis_pipeline("
                "wt_free_energies, rate_table, 50, 1.5, 20260311, 310.15)"
            ),
            "gold_call": (
                "_oracle_run_cryptic_epistasis_pipeline("
                "wt_free_energies, rate_table, 50, 1.5, 20260311, 310.15)"
            ),
        },
        {
            # Invalid: index is outside the three scoreable observables.
            "setup": """import numpy as np

def run_model():
    try:
        run_cryptic_epistasis_pipeline(n_mutants=20, parameter_index=7)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_run_cryptic_epistasis_pipeline(
            n_mutants=20,
            parameter_index=7,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            # Invalid: KD is excluded because its reference scale is zero.
            "setup": """import numpy as np

def run_model():
    try:
        run_cryptic_epistasis_pipeline(n_mutants=20, parameter_index=0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_run_cryptic_epistasis_pipeline(
            n_mutants=20,
            parameter_index=0,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            # Invalid: the perturbation window must be positive.
            "setup": """import numpy as np

def run_model():
    try:
        run_cryptic_epistasis_pipeline(n_mutants=20, amplitude=0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_run_cryptic_epistasis_pipeline(
            n_mutants=20,
            amplitude=0.0,
        )
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
