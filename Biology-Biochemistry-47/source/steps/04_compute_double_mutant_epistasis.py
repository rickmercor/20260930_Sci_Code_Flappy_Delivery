"""
Compute mechanism-manufactured epistasis for every ordered pair of single mutants and for all four steady-state kinetic observables, without receiving precomputed observable-level fold changes.



Under the free-energy-additive null model, the two single-mutant microscopic rate-constant fold vectors multiply componentwise with the wild-type microscopic rate vector. Independently reconstruct each single mutant’s KD, kcat, KM, and kcat/KM from its microscopic rates, so that the conventional observable-level expectation is derived rather than supplied. Evaluate the additive-null double mutant from its combined microscopic rates and divide each observable by that reconstructed expectation.



Return a tensor ordered as KD, kcat, KM, and kcat/KM. Retain diagonal pairs: combining a variant with itself is a valid additive-null double-mutant construction.

This step implements the paper’s free-energy-additive null model for mutational interaction. For two independently acting substitutions, the energy shifts of the two variants add at every ground state and transition state of the reaction coordinate. Under transition-state theory, this means the two variants’ fold-changes in each microscopic rate constant multiply.



The resulting double mutant is therefore free of any explicit energetic coupling or residue-level interaction. Nevertheless, the steady-state observables of a catalytic cycle are nonlinear functions of several microscopic rate constants. A double mutant constructed under this additive null can consequently deviate from the conventional expectation formed by multiplying the two single-mutant fold-changes in \(K_D\), \(k_{\mathrm{cat}}\), \(K_M\), or \(k_{\mathrm{cat}}/K_M\).



For every ordered pair of library variants, calculate the additive-null double-mutant rate constants, evaluate all four steady-state observables, and divide each by its conventional multiplicative expectation. The result is a tensor whose first axis is ordered as \(K_D\), \(k_{\mathrm{cat}}\), \(K_M\), and \(k_{\mathrm{cat}}/K_M\). Diagonal pairs are retained because pairing a mutation with itself is still a valid additive-null combination.

Returns
-------
return epistasis
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_double_mutant_epistasis(
    rate_constants_wt,
    rate_folds,
) -> np.ndarray:
    """Compute additive-null epistasis from microscopic-rate fold changes.

    Reconstruct single-mutant observable folds internally, construct each
    ordered-pair additive-null double mutant at the microscopic-rate level,
    and return observed-over-expected epistasis for KD, kcat, KM, and kcat/KM.
    """
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_double_mutant_epistasis(
    rate_constants_wt,
    rate_folds,
) -> np.ndarray:
    """Compute all four additive-null epistasis matrices from microscopic folds."""
    import numpy as np

    def _parameters(rates):
        """Return KD, kcat, KM, and kcat/KM on the final axis."""
        k_on = rates[..., 0]
        k_off = rates[..., 1]
        k_chem = rates[..., 2]
        k_rev = rates[..., 3]
        k_rel = rates[..., 4]

        partition = k_chem + k_rev + k_rel
        capture = k_chem * k_rel + k_off * k_rev + k_off * k_rel
        if np.any(partition <= 0.0) or np.any(capture <= 0.0):
            raise ValueError("the cycle must carry non-zero steady-state flux")

        return np.stack(
            [
                k_off / k_on,
                k_chem * k_rel / partition,
                capture / (k_on * partition),
                k_on * k_chem * k_rel / capture,
            ],
            axis=-1,
        )

    wt = np.asarray(rate_constants_wt, dtype=float)
    folds = np.asarray(rate_folds, dtype=float)

    if wt.shape != (5,):
        raise ValueError("rate_constants_wt must have shape (5,)")
    if folds.ndim != 2 or folds.shape[0] < 1 or folds.shape[1] != 5:
        raise ValueError(
            "rate_folds must have shape (n_mutants, 5), n_mutants >= 1"
        )
    if not (np.all(np.isfinite(wt)) and np.all(np.isfinite(folds))):
        raise ValueError("inputs must be finite")
    if np.any(wt <= 0.0) or np.any(folds <= 0.0):
        raise ValueError("rate constants and fold changes must be positive")

    # Reconstruct the observable-level expectation from each single mutant.
    wt_parameters = _parameters(wt)
    single_rates = wt[None, :] * folds
    single_parameters = _parameters(single_rates)
    single_parameter_folds = single_parameters / wt_parameters[None, :]

    # Construct every ordered additive-null double mutant microscopically.
    double_rates = (
        wt[None, None, :]
        * folds[:, None, :]
        * folds[None, :, :]
    )
    double_parameters = _parameters(double_rates)

    expected_parameters = (
        wt_parameters[None, None, :]
        * single_parameter_folds[:, None, :]
        * single_parameter_folds[None, :, :]
    )
    if np.any(expected_parameters <= 0.0):
        raise ValueError("the reconstructed null expectation must be positive")

    epistasis = np.moveaxis(
        double_parameters / expected_parameters,
        -1,
        0,
    )
    return epistasis

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for microscopic additive-null epistasis."""
    return [
        {
            "setup": """import numpy as np
rate_constants_wt = np.array([1.20e6, 850.0, 340.0, 4.50, 95.0])
rate_folds = np.array([
    [0.8, 1.2, 0.4, 1.1, 0.6],
    [1.5, 0.7, 2.0, 0.5, 1.3],
    [0.9, 1.8, 0.6, 1.4, 0.8],
])
""",
            "call": (
                "compute_double_mutant_epistasis("
                "rate_constants_wt, rate_folds)[3]"
            ),
            "gold_call": (
                "_oracle_compute_double_mutant_epistasis("
                "rate_constants_wt, rate_folds)[3]"
            ),
        },
        {
            "setup": """import numpy as np
rate_constants_wt = np.array([1.20e6, 850.0, 340.0, 4.50, 95.0])
rate_folds = np.array([
    [0.8, 1.2, 0.4, 1.1, 0.6],
    [1.5, 0.7, 2.0, 0.5, 1.3],
    [0.9, 1.8, 0.6, 1.4, 0.8],
])
""",
            "call": (
                "compute_double_mutant_epistasis("
                "rate_constants_wt, rate_folds)[2]"
            ),
            "gold_call": (
                "_oracle_compute_double_mutant_epistasis("
                "rate_constants_wt, rate_folds)[2]"
            ),
        },
        {
            "setup": """import numpy as np
rate_constants_wt = np.array([1.20e6, 850.0, 340.0, 4.50, 95.0])
rate_folds = np.array([
    [0.65, 1.40, 0.018, 1.20, 0.060],
])
""",
            "call": (
                "compute_double_mutant_epistasis("
                "rate_constants_wt, rate_folds)[1, 0, 0]"
            ),
            "gold_call": (
                "_oracle_compute_double_mutant_epistasis("
                "rate_constants_wt, rate_folds)[1, 0, 0]"
            ),
        },
        {
            "setup": """import numpy as np
rate_constants_wt = np.array([2.50e5, 14.0, 0.85, 90.0, 0.13])
rate_folds = np.array([
    [1.0e-5, 2.5e3, 4.0e-4, 8.0e2, 3.0e-5],
    [7.0e2, 3.0e-4, 9.0e3, 2.0e-5, 4.0e2],
])
""",
            "call": (
                "compute_double_mutant_epistasis("
                "rate_constants_wt, rate_folds)"
            ),
            "gold_call": (
                "_oracle_compute_double_mutant_epistasis("
                "rate_constants_wt, rate_folds)"
            ),
        },
        {
            "setup": """import numpy as np
generator = np.random.default_rng(806)
rate_constants_wt = np.array([8.4e5, 73.0, 15.0, 2.7, 41.0])
rate_folds = np.exp(generator.uniform(-5.0, 5.0, size=(7, 5)))
""",
            "call": (
                "compute_double_mutant_epistasis("
                "rate_constants_wt, rate_folds)[0]"
            ),
            "gold_call": (
                "_oracle_compute_double_mutant_epistasis("
                "rate_constants_wt, rate_folds)[0]"
            ),
        },
    ]
