"""
Draw the donor level cohort in which ligand expression, receptor expression and receiver pathway activity are all loaded by one latent confounder, and return it as a single centred matrix that every later step reads.

A cell-cell communication screen works on donor level quantities. For an ordered pair of cell types, one acting as sender and one as receiver, each donor contributes a ligand expression value measured in the sender cell type, a receptor expression value measured in the receiver cell type, and a scalar pathway activity score summarising a receiver gene set. The structural question is whether variation in the ligand drives the pathway score, and the answer is confounded from the outset: donor age, ancestry, batch, cell composition and unmeasured cell state programmes load on all three quantities at once, so a co-expression screen sees a signal whether or not any signalling occurs. A germline genotype does not share that fate. Cis-eQTL dosages near the ligand gene and near the receptor gene are fixed at conception, so they are plausibly independent of the latent donor factor, which is what makes them usable as instruments for the two expression traits.




A surrogate cohort that is fair to both the associational and the causal analysis therefore needs four ingredients. Instruments for the ligand and instruments for the receptor, each acting on its own exposure only. Observed donor covariates that act on all three quantities, standing for the age, sex and genotype principal components that a real analysis adjusts for. A single unmeasured donor factor loading on the ligand, on the receptor and on the pathway score, which is the confounder the instruments are meant to defeat. And a pathway equation in which the ligand acts both on its own and in product with the receptor, because the strength of a communication signal in the receiver depends on how much receptor is available to engage. Setting both of those ligand terms to zero while keeping the confounder loading nonzero produces a cohort in which there is no communication at all and yet every pairwise association is present, which is the setting that separates a causal analysis from an associational one.




Centring every observed column afterwards is a convenience of the structural model rather than a transformation of the science. The ligand and receptor equations carry no intercept once their inputs are centred, and the pathway equation is written with an explicit intercept because the product of two centred variables does not have mean zero. Note that the pathway score is formed from the uncentred ligand and receptor values and only then centred with everything else, so the product term retains the covariance that the intercept has to absorb.

Returns
-------
np.ndarray of shape (n_donors, 2 * n_instruments + n_covariates + 3), float: the centred cohort whose columns are the ligand instruments, the receptor instruments, the covariates, ligand expression, receptor expression and pathway activity.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def simulate_communication_dataset(n_donors: int, n_instruments: int, n_covariates: int,
                                   instrument_strength: float, covariate_effect: float,
                                   receptor_effect: float, confounder_loading: float,
                                   ligand_effect: float, interaction_effect: float,
                                   seed: int) -> np.ndarray:
    '''Draw one donor level cohort from the structural communication model.

    Every instrument, covariate, confounder and residual entry is an
    independent standard normal. Randomness comes from
    numpy.random.default_rng(seed), which draws, in this order, the ligand
    instrument matrix of shape (n_donors, n_instruments), the receptor
    instrument matrix of the same shape, the covariate matrix of shape
    (n_donors, n_covariates), the confounder vector of length n_donors, and
    then the ligand, receptor and pathway residual vectors, each of length
    n_donors.

    The ligand instruments act on ligand expression only and the receptor
    instruments on receptor expression only, every instrument carrying the
    same coefficient instrument_strength. The covariates act on all three
    observed quantities with the common coefficient covariate_effect. The
    confounder acts on all three with the common coefficient
    confounder_loading. Pathway activity is built from the uncentred ligand
    and receptor values, receiving ligand_effect times the ligand,
    receptor_effect times the receptor and interaction_effect times their
    elementwise product. Every column of the returned matrix is then centred
    to mean zero.

    Parameters
    ----------
    n_donors : int
        Number of donors, at least 2 * n_instruments + n_covariates + 5.
    n_instruments : int
        Number of cis-eQTL instruments for each of the two exposures,
        n_instruments >= 1.
    n_covariates : int
        Number of observed donor covariates, n_covariates >= 1.
    instrument_strength : float
        Common coefficient of every instrument on its own exposure.
    covariate_effect : float
        Common coefficient of every covariate on each observed quantity.
    receptor_effect : float
        Coefficient of receptor expression on pathway activity.
    confounder_loading : float
        Common coefficient of the unmeasured donor factor on each observed
        quantity.
    ligand_effect : float
        Coefficient of ligand expression on pathway activity.
    interaction_effect : float
        Coefficient of the ligand by receptor product on pathway activity.
    seed : int
        Seed of the random generator.

    Returns
    -------
    data : np.ndarray
        Array of shape (n_donors, 2 * n_instruments + n_covariates + 3) whose
        columns are, in order, the n_instruments ligand instruments, the
        n_instruments receptor instruments, the n_covariates covariates,
        ligand expression, receptor expression and pathway activity. Every
        column has mean zero.

    Raises
    ------
    ValueError
        If n_instruments or n_covariates is below one, if n_donors is below
        2 * n_instruments + n_covariates + 5, if seed is not an integer, or if
        any coefficient argument is not a finite real number.
    '''
    return data  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np 

def _oracle_simulate_communication_dataset(n_donors: int, n_instruments: int, n_covariates: int,
                                           instrument_strength: float, covariate_effect: float,
                                           receptor_effect: float, confounder_loading: float,
                                           ligand_effect: float, interaction_effect: float,
                                           seed: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _check_int(name, value, low):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if int(value) < low:
            raise ValueError(f"{name} must be at least {low}")
        return int(value)

    def _check_real(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        return float(value)

    n_instrument = _check_int("n_instruments", n_instruments, 1)
    n_covariate = _check_int("n_covariates", n_covariates, 1)
    smallest = 2 * n_instrument + n_covariate + 5
    n_row = _check_int("n_donors", n_donors, smallest)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    pi_value = _check_real("instrument_strength", instrument_strength)
    alpha_value = _check_real("covariate_effect", covariate_effect)
    beta_receptor = _check_real("receptor_effect", receptor_effect)
    lambda_value = _check_real("confounder_loading", confounder_loading)
    beta_ligand = _check_real("ligand_effect", ligand_effect)
    beta_interaction = _check_real("interaction_effect", interaction_effect)

    rng = np.random.default_rng(int(seed))
    ligand_instruments = rng.standard_normal((n_row, n_instrument))
    receptor_instruments = rng.standard_normal((n_row, n_instrument))
    covariates = rng.standard_normal((n_row, n_covariate))
    confounder = rng.standard_normal(n_row)
    ligand_residual = rng.standard_normal(n_row)
    receptor_residual = rng.standard_normal(n_row)
    pathway_residual = rng.standard_normal(n_row)

    instrument_vector = np.full(n_instrument, pi_value)
    covariate_vector = np.full(n_covariate, alpha_value)
    covariate_signal = covariates @ covariate_vector

    # The two exposure equations of the structural model.
    ligand = (ligand_instruments @ instrument_vector + covariate_signal
              + lambda_value * confounder + ligand_residual)
    receptor = (receptor_instruments @ instrument_vector + covariate_signal
                + lambda_value * confounder + receptor_residual)

    # The outcome equation, with the receptor modulated ligand effect entering
    # as the elementwise product of the two uncentred exposures.
    pathway = (beta_ligand * ligand + beta_receptor * receptor
               + beta_interaction * ligand * receptor + covariate_signal
               + lambda_value * confounder + pathway_residual)

    data = np.column_stack([ligand_instruments, receptor_instruments, covariates,
                            ligand, receptor, pathway])
    return np.asarray(data - data.mean(axis=0), dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np 

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: null cohort with confounding, the benchmark shape at reduced size ---
        {
            "setup": """import numpy as np
n_donors, n_instruments, n_covariates = 60, 4, 3
instrument_strength, covariate_effect = 0.45, 0.25
receptor_effect, confounder_loading = 0.5, 0.7
ligand_effect, interaction_effect, seed = 0.0, 0.0, 20260826
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(simulate_communication_dataset(n_donors, n_instruments, n_covariates, instrument_strength, covariate_effect, receptor_effect, confounder_loading, ligand_effect, interaction_effect, seed))",
            "gold_call": "digest(_oracle_simulate_communication_dataset(n_donors, n_instruments, n_covariates, instrument_strength, covariate_effect, receptor_effect, confounder_loading, ligand_effect, interaction_effect, seed))",
        },
        # --- Valid: communication present, with receptor modulation ---
        {
            "setup": """import numpy as np
n_donors, n_instruments, n_covariates = 45, 5, 3
instrument_strength, covariate_effect = 0.5, 0.3
receptor_effect, confounder_loading = 0.5, 0.7
ligand_effect, interaction_effect, seed = 0.3, 0.3, 11
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(simulate_communication_dataset(n_donors, n_instruments, n_covariates, instrument_strength, covariate_effect, receptor_effect, confounder_loading, ligand_effect, interaction_effect, seed))",
            "gold_call": "digest(_oracle_simulate_communication_dataset(n_donors, n_instruments, n_covariates, instrument_strength, covariate_effect, receptor_effect, confounder_loading, ligand_effect, interaction_effect, seed))",
        },
        # --- Boundary: no confounding at all, so the associational and causal targets coincide ---
        {
            "setup": """import numpy as np
n_donors, n_instruments, n_covariates = 30, 1, 1
instrument_strength, covariate_effect = 0.8, 0.2
receptor_effect, confounder_loading = 0.5, 0.0
ligand_effect, interaction_effect, seed = 0.3, 0.0, 3
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(simulate_communication_dataset(n_donors, n_instruments, n_covariates, instrument_strength, covariate_effect, receptor_effect, confounder_loading, ligand_effect, interaction_effect, seed))",
            "gold_call": "digest(_oracle_simulate_communication_dataset(n_donors, n_instruments, n_covariates, instrument_strength, covariate_effect, receptor_effect, confounder_loading, ligand_effect, interaction_effect, seed))",
        },
        # --- Edge: the smallest admissible cohort for this instrument count ---
        {
            "setup": """import numpy as np
n_donors, n_instruments, n_covariates = 8, 1, 1
instrument_strength, covariate_effect = 0.5, 0.3
receptor_effect, confounder_loading = 0.5, 0.7
ligand_effect, interaction_effect, seed = 0.0, 0.0, 0
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(simulate_communication_dataset(n_donors, n_instruments, n_covariates, instrument_strength, covariate_effect, receptor_effect, confounder_loading, ligand_effect, interaction_effect, seed))",
            "gold_call": "digest(_oracle_simulate_communication_dataset(n_donors, n_instruments, n_covariates, instrument_strength, covariate_effect, receptor_effect, confounder_loading, ligand_effect, interaction_effect, seed))",
        },
        # --- Invalid: an exposure with no instrument at all ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        simulate_communication_dataset(50, 0, 3, 0.5, 0.3, 0.5, 0.7, 0.0, 0.0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_simulate_communication_dataset(50, 0, 3, 0.5, 0.3, 0.5, 0.7, 0.0, 0.0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: fewer donors than the first stage regressions can support ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        simulate_communication_dataset(7, 1, 1, 0.5, 0.3, 0.5, 0.7, 0.0, 0.0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_simulate_communication_dataset(7, 1, 1, 0.5, 0.3, 0.5, 0.7, 0.0, 0.0, 1)
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
