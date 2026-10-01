"""
Score the cohort the way a co-expression screen does, by regressing pathway activity on the observed ligand, the observed receptor, their product and the covariates, and turning the joint test of the two ligand terms into a communication score on the logit scale.

The paradigm this task is testing scores a candidate ligand receptor pathway triplet by how strongly the observed expression values track the pathway score. Written as a regression, that means putting pathway activity on the left and the observed ligand, the observed receptor, their elementwise product and the donor covariates on the right, with an intercept, and asking whether the two coefficients that carry the ligand are jointly nonzero. The product term has to be present even in the associational analysis, because the alternative being entertained is that ligand action is modulated by receptor abundance, and a test that omits it would be testing a different hypothesis from the causal analysis it is being compared with.




The hypothesis is therefore two-dimensional, and the natural test is the joint one. With a contrast matrix that picks the ligand coefficient and the ligand by receptor coefficient out of the fitted vector, the Wald statistic divided by its two degrees of freedom is an F ratio against the residual degrees of freedom of the fit, and the corresponding upper tail probability is the significance of the pair of terms taken together. Testing the two coefficients one at a time instead would answer a different and weaker question and would also leave an unresolved multiplicity.




Reporting the result as a probability throws away resolution exactly where it matters. Under a latent donor factor that loads on all three quantities, the fitted ligand coefficient is displaced from its structural value by an amount that does not shrink with cohort size, so the tail probability collapses towards zero as donors accumulate and a score expressed on the unit interval saturates at one. The logit of that score, the base ten logarithm of the ratio of the tail probability's complement to the tail probability itself, keeps the same ordering and stays finite and informative deep into the tail, which is what allows an associational verdict and a posterior probability to be compared as odds. Computing it needs the logarithm of both tails of the F distribution rather than the probability itself, since forming the ratio after rounding the tail probability to a double precision zero loses the answer.

Returns
-------
np.ndarray of four floats: the F statistic, the base ten logarithm of the associational communication odds, the fitted ligand coefficient and the fitted ligand by receptor coefficient.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_naive_communication_score(data: np.ndarray, n_instruments: int,
                                      n_covariates: int) -> np.ndarray:
    '''Score a cohort with the associational regression that ignores the instruments.

    Pathway activity is regressed by ordinary least squares on a design whose
    columns are, in order, an intercept, ligand expression, receptor
    expression, the elementwise product of ligand and receptor expression, and
    the covariates. The two ligand carrying coefficients are tested jointly by
    the Wald F ratio on two numerator degrees of freedom and on the residual
    degrees of freedom of the fit, using the usual homoscedastic coefficient
    covariance estimate. The score is reported on the logit scale as the base
    ten logarithm of the ratio of the lower tail of that F distribution to its
    upper tail at the observed value.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_donors, 2 * n_instruments + n_covariates + 3) laid
        out as returned by the cohort simulation step.
    n_instruments : int
        Number of cis-eQTL instruments for each exposure, n_instruments >= 1.
    n_covariates : int
        Number of observed donor covariates, n_covariates >= 1.

    Returns
    -------
    summary : np.ndarray
        Array of four floats holding, in order, the observed F statistic, the
        base ten logarithm of the associational communication odds, the fitted
        ligand coefficient and the fitted ligand by receptor coefficient.

    Raises
    ------
    ValueError
        If data is not a two dimensional finite array with the column count
        implied by n_instruments and n_covariates, if either count is below
        one, if the cohort leaves no residual degrees of freedom, or if the
        design or the contrast covariance is numerically singular.
    '''
    return summary  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np 

def _oracle_compute_naive_communication_score(data: np.ndarray, n_instruments: int,
                                              n_covariates: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
    from scipy.stats import f as f_distribution

    def _check_int(name, value, low):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if int(value) < low:
            raise ValueError(f"{name} must be at least {low}")
        return int(value)

    n_instrument = _check_int("n_instruments", n_instruments, 1)
    n_covariate = _check_int("n_covariates", n_covariates, 1)
    table = np.asarray(data, dtype=float)
    if table.ndim != 2:
        raise ValueError("data must be a two dimensional array")
    if table.shape[1] != 2 * n_instrument + n_covariate + 3:
        raise ValueError("data has the wrong number of columns for these counts")
    if not np.all(np.isfinite(table)):
        raise ValueError("data must be finite")

    n_row = table.shape[0]
    offset = 2 * n_instrument + n_covariate
    covariates = table[:, offset - n_covariate:offset]
    ligand = table[:, offset]
    receptor = table[:, offset + 1]
    pathway = table[:, offset + 2]

    design = np.column_stack([np.ones(n_row), ligand, receptor,
                              ligand * receptor, covariates])
    n_parameter = design.shape[1]
    residual_degrees = n_row - n_parameter
    if residual_degrees < 1:
        raise ValueError("the cohort leaves no residual degrees of freedom")

    gram = design.T @ design
    if not np.isfinite(np.linalg.cond(gram)) or np.linalg.cond(gram) > 1.0e12:
        raise ValueError("the associational design is numerically singular")
    gram_inverse = np.linalg.inv(gram)
    coefficients = gram_inverse @ (design.T @ pathway)
    residual = pathway - design @ coefficients
    residual_variance = float(residual @ residual) / residual_degrees
    coefficient_covariance = residual_variance * gram_inverse

    # The contrast picks the ligand coefficient and the ligand by receptor
    # coefficient, the two terms that carry communication under this model.
    contrast = np.zeros((2, n_parameter))
    contrast[0, 1] = 1.0
    contrast[1, 3] = 1.0
    contrast_effect = contrast @ coefficients
    contrast_covariance = contrast @ coefficient_covariance @ contrast.T
    if np.linalg.cond(contrast_covariance) > 1.0e12:
        raise ValueError("the contrast covariance is numerically singular")
    statistic = float(contrast_effect @ np.linalg.solve(contrast_covariance,
                                                        contrast_effect)) / 2.0
    if not np.isfinite(statistic) or statistic <= 0.0:
        raise ValueError("the joint F statistic is not a positive finite number")

    # Both tails are taken in logarithmic form so that the odds survive a tail
    # probability far below the smallest representable double.
    log_upper = float(f_distribution.logsf(statistic, 2, residual_degrees))
    log_lower = float(f_distribution.logcdf(statistic, 2, residual_degrees))
    log10_odds = (log_lower - log_upper) / float(np.log(10.0))
    if not np.isfinite(log10_odds):
        raise ValueError("the associational communication odds are not finite")

    return np.asarray([statistic, log10_odds, coefficients[1], coefficients[3]],
                      dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np 

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: confounded null cohort, where the association is entirely spurious ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(5)
n_donors, n_instruments, n_covariates = 220, 4, 3
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
confounder = rng.standard_normal(n_donors)
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.45) + 0.7 * confounder + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.45) + 0.7 * confounder + rng.standard_normal(n_donors)
pathway = 0.5 * receptor + covariates @ np.full(n_covariates, 0.25) + 0.7 * confounder + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(compute_naive_communication_score(data, n_instruments, n_covariates))",
            "gold_call": "digest(_oracle_compute_naive_communication_score(data, n_instruments, n_covariates))",
        },
        # --- Valid: genuine communication with receptor modulation and no confounding ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(17)
n_donors, n_instruments, n_covariates = 150, 2, 2
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.6) + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.6) + rng.standard_normal(n_donors)
pathway = 0.3 * ligand + 0.5 * receptor + 0.3 * ligand * receptor + covariates @ np.full(n_covariates, 0.3) + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(compute_naive_communication_score(data, n_instruments, n_covariates))",
            "gold_call": "digest(_oracle_compute_naive_communication_score(data, n_instruments, n_covariates))",
        },
        # --- Boundary: an unconfounded null cohort, where the odds should be modest ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(41)
n_donors, n_instruments, n_covariates = 90, 1, 1
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
pathway = 0.5 * receptor + covariates @ np.full(n_covariates, 0.3) + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(compute_naive_communication_score(data, n_instruments, n_covariates))",
            "gold_call": "digest(_oracle_compute_naive_communication_score(data, n_instruments, n_covariates))",
        },
        # --- Edge: a very small cohort, leaving few residual degrees of freedom ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(2)
n_donors, n_instruments, n_covariates = 12, 1, 1
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.5) + 0.7 * rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.5) + 0.7 * rng.standard_normal(n_donors)
pathway = 0.5 * receptor + covariates @ np.full(n_covariates, 0.3) + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(compute_naive_communication_score(data, n_instruments, n_covariates))",
            "gold_call": "digest(_oracle_compute_naive_communication_score(data, n_instruments, n_covariates))",
        },
        # --- Invalid: a column count that does not match the declared counts ---
        {
            "setup": """import numpy as np
data = np.zeros((40, 9))
def run_model():
    try:
        compute_naive_communication_score(data, 4, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_naive_communication_score(data, 4, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: fewer donors than the associational design has columns ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(1)
data = rng.standard_normal((5, 8))
data = data - data.mean(axis=0)
def run_model():
    try:
        compute_naive_communication_score(data, 2, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_naive_communication_score(data, 2, 1)
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
