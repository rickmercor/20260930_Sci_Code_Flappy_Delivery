"""
Draw the two-element communication effect vector from its spike and slab conditional, then draw the inclusion indicator that decides which component the vector belongs to, then refresh the inclusion probability from its own conditional.

The quantity this analysis exists to report is a pair, not a single number. A ligand can act on a receiver pathway on its own, and it can act in a way that depends on how much receptor the receiver is carrying, and the second mechanism is exactly what a purely additive causal model cannot express. Because the two effects are two coordinates of one biological claim, that either there is communication along this ligand receptor pathway triplet or there is not, they are selected jointly by a single indicator rather than screened one at a time. Screening them separately would license the incoherent verdict that a ligand has no effect of its own and yet modulates the pathway through the receptor, and it would multiply the number of tests without any gain in interpretability.




Selection here is not a label attached to a fitted vector after the fact. The two components of the prior differ only by a scale factor on their covariance, but that factor is small enough that the vector drawn under one component and the vector drawn under the other live on entirely different scales; the chain therefore moves the effect vector between two regimes rather than merely recording which regime it prefers. That is what makes the fraction of sweeps in which the indicator is on interpretable as a probability that communication is present, in the strict sense in which the complement of a frequentist tail probability is not.

Two things about this step are easy to get wrong and consequential. The first is that the comparison between the two components is not a comparison of exponents alone. A multivariate normal density carries the determinant of its covariance, and here the two covariances differ by a scale factor applied in two dimensions, so the comparison carries a term that favours the smaller component by a large factor before any data are seen. Dropping it produces a screen that includes very nearly everything, which is the failure mode a selection prior is supposed to prevent. The second is arithmetic: the two exponents differ by four orders of magnitude in scale, so a comparison formed by exponentiating them directly will overflow on one side or underflow on the other for any effect vector that is not close to the boundary between them.

Returns
-------
np.ndarray of five floats: the drawn ligand main effect, the drawn ligand by receptor interaction effect, the drawn inclusion indicator, the conditional inclusion probability it was drawn from, and the refreshed inclusion probability.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def update_communication_state(data: np.ndarray, n_instruments: int, n_covariates: int,
                               state: np.ndarray, g_prior: float, nu_spike: float,
                               a_rho: float, b_rho: float, ridge: float,
                               rng: np.random.Generator) -> np.ndarray:
    '''Draw the communication effect vector, the inclusion indicator and the inclusion probability.

    The three conditionals are visited in the order in which they are
    returned. The effect vector is drawn against the inclusion indicator held
    in state, and the indicator is then drawn against the effect vector that
    has just been drawn rather than against the one held on entry. No entry of
    the two exposure blocks and no entry of the outcome nuisance block is
    modified. The spike and slab scale factor is one when the indicator being
    conditioned on is at least one half and nu_spike otherwise. Every matrix
    inverted along the way carries ridge added to its diagonal beforehand.

    Randomness comes from rng, which draws one vector of two standard normals,
    then one uniform variate, then one beta variate. The effect vector is its
    conditional mean plus the lower Cholesky factor of its conditional
    covariance times the standard normal vector. The indicator is one when the
    uniform variate is strictly below the conditional inclusion probability
    and zero otherwise. The beta variate has its first shape raised by the
    drawn indicator and its second by the complement of that indicator.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_donors, 2 * n_instruments + n_covariates + 3) laid
        out as returned by the cohort simulation step.
    n_instruments : int
        Number of cis-eQTL instruments for each exposure, n_instruments >= 1.
    n_covariates : int
        Number of observed donor covariates, n_covariates >= 1.
    state : np.ndarray
        Array of 2 * n_instruments + 3 * n_covariates + 9 floats laid out as
        returned by the state initialisation step.
    g_prior : float
        Zellner g value of the communication prior, strictly positive.
    nu_spike : float
        Scale factor of the spike component, 0 < nu_spike <= 1.
    a_rho : float
        First shape of the beta prior on the inclusion probability, a_rho > 0.
    b_rho : float
        Second shape of the beta prior on the inclusion probability, b_rho > 0.
    ridge : float
        Non negative diagonal regularisation added before any inversion.
    rng : np.random.Generator
        Generator supplying the draws, in the order described above.

    Returns
    -------
    communication_state : np.ndarray
        Array of five floats holding the drawn ligand main effect, the drawn
        ligand by receptor interaction effect, the drawn inclusion indicator,
        the conditional inclusion probability that indicator was drawn from,
        and the refreshed inclusion probability, in that order.

    Raises
    ------
    ValueError
        If data is not a two dimensional finite array with the column count
        implied by n_instruments and n_covariates, if either count is below
        one, if state is not a finite one dimensional array of the implied
        length, if the outcome residual variance held in state is not strictly
        positive, if the inclusion indicator held in state is neither zero nor
        one, if the inclusion probability held in state is not strictly inside
        the unit interval, if g_prior, a_rho or b_rho is not a positive finite
        number, if nu_spike is outside the half open interval from zero to
        one, if ridge is negative, or if the conditional covariance of the
        effect vector is not positive definite.
    '''
    return communication_state  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_update_communication_state(data: np.ndarray, n_instruments: int, n_covariates: int,
                                       state: np.ndarray, g_prior: float, nu_spike: float,
                                       a_rho: float, b_rho: float, ridge: float,
                                       rng: np.random.Generator) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _check_int(name, value, low):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if int(value) < low:
            raise ValueError(f"{name} must be at least {low}")
        return int(value)

    def _check_positive(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not (np.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"{name} must be a positive finite number")
        return float(value)

    n_instrument = _check_int("n_instruments", n_instruments, 1)
    n_covariate = _check_int("n_covariates", n_covariates, 1)
    g_value = _check_positive("g_prior", g_prior)
    spike = _check_positive("nu_spike", nu_spike)
    if spike > 1.0:
        raise ValueError("nu_spike must not exceed one")
    shape_a = _check_positive("a_rho", a_rho)
    shape_b = _check_positive("b_rho", b_rho)
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float, np.integer, np.floating)):
        raise ValueError("ridge must be a real number")
    if not (np.isfinite(float(ridge)) and float(ridge) >= 0.0):
        raise ValueError("ridge must be a non negative finite number")
    ridge_value = float(ridge)
    table = np.asarray(data, dtype=float)
    if table.ndim != 2 or table.shape[1] != 2 * n_instrument + n_covariate + 3:
        raise ValueError("data has the wrong shape for these counts")
    if not np.all(np.isfinite(table)):
        raise ValueError("data must be finite")
    parameters = np.asarray(state, dtype=float).ravel()
    if parameters.size != 2 * n_instrument + 3 * n_covariate + 9:
        raise ValueError("state has the wrong length for these counts")
    if not np.all(np.isfinite(parameters)):
        raise ValueError("state must be finite")

    offset = 2 * n_instrument + n_covariate
    covariates = table[:, offset - n_covariate:offset]
    pathway = table[:, offset + 2]
    cursor = n_instrument + n_covariate
    ligand_representation = (table[:, :n_instrument] @ parameters[0:n_instrument]
                             + covariates @ parameters[n_instrument:cursor])
    receptor_representation = (
        table[:, n_instrument:2 * n_instrument] @ parameters[cursor + 1:cursor + 1 + n_instrument]
        + covariates @ parameters[cursor + 1 + n_instrument:2 * cursor + 1])
    intercept = float(parameters[2 * n_instrument + 2 * n_covariate + 2])
    outcome_covariate_slice = slice(2 * n_instrument + 2 * n_covariate + 3,
                                    2 * n_instrument + 3 * n_covariate + 3)
    receptor_effect = float(parameters[outcome_covariate_slice.stop])
    outcome_variance = float(parameters[outcome_covariate_slice.stop + 1])
    if outcome_variance <= 0.0:
        raise ValueError("state carries an outcome variance that is not positive")
    inclusion = float(parameters[outcome_covariate_slice.stop + 4])
    if inclusion not in (0.0, 1.0):
        raise ValueError("state carries an inclusion indicator that is neither zero nor one")
    prior_probability = float(parameters[outcome_covariate_slice.stop + 5])
    if not 0.0 < prior_probability < 1.0:
        raise ValueError("state carries an inclusion probability outside the unit interval")

    # The two columns the communication claim is carried by, both built from
    # the instrument based representation rather than from observed expression.
    communication_design = np.column_stack(
        [ligand_representation, ligand_representation * receptor_representation])
    gram = communication_design.T @ communication_design + ridge_value * np.eye(2)

    # Under the spike the effective g value collapses by four orders of
    # magnitude, which is what pins the drawn vector near zero.
    effective_g = g_value * (1.0 if inclusion >= 0.5 else spike)
    shrinkage = effective_g / (1.0 + effective_g)
    left = (pathway - intercept - receptor_effect * receptor_representation
            - covariates @ parameters[outcome_covariate_slice])
    try:
        gram_inverse = np.linalg.inv(gram)
        factor = np.linalg.cholesky(shrinkage * outcome_variance * gram_inverse)
    except np.linalg.LinAlgError as error:
        raise ValueError("the communication covariance is not positive definite") from error
    communication = (shrinkage * (gram_inverse @ (communication_design.T @ left))
                     + factor @ rng.standard_normal(2))

    # The determinant of a two dimensional spike covariance contributes the
    # reciprocal of the spike factor, which is the penalty the slab must
    # overcome, and the comparison is normalised on the log scale because the
    # two exponents differ by orders of magnitude.
    quadratic = float(communication @ gram @ communication)
    log_slab = (-0.5 * quadratic / (g_value * outcome_variance)
                + np.log(prior_probability))
    log_spike = (-0.5 * quadratic / (g_value * spike * outcome_variance)
                 + np.log1p(-prior_probability) - np.log(spike))
    largest = max(log_slab, log_spike)
    slab_weight = np.exp(log_slab - largest)
    spike_weight = np.exp(log_spike - largest)
    conditional = float(slab_weight / (slab_weight + spike_weight))

    indicator = 1.0 if float(rng.random()) < conditional else 0.0
    refreshed = float(rng.beta(shape_a + indicator, shape_b + 1.0 - indicator))
    return np.asarray([communication[0], communication[1], indicator, conditional,
                       refreshed], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np
def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the indicator on and an effect vector large enough for the slab to hold ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(109)
n_donors, n_instruments, n_covariates = 100, 4, 3
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
pathway = 0.4 * ligand + 0.5 * receptor + 0.35 * ligand * receptor + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
state = np.full(2 * n_instruments + 3 * n_covariates + 9, 0.3)
state[n_instruments + n_covariates] = 1.0
state[2 * n_instruments + 2 * n_covariates + 1] = 1.0
state[2 * n_instruments + 3 * n_covariates + 4] = 1.0
state[2 * n_instruments + 3 * n_covariates + 5] = 0.3
state[2 * n_instruments + 3 * n_covariates + 6] = 0.3
state[2 * n_instruments + 3 * n_covariates + 7] = 1.0
state[2 * n_instruments + 3 * n_covariates + 8] = 0.75
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(update_communication_state(data, n_instruments, n_covariates, state, 100.0, 1e-4, 3.0, 1.0, 1e-6, np.random.default_rng(16)))",
            "gold_call": "digest(_oracle_update_communication_state(data, n_instruments, n_covariates, state, 100.0, 1e-4, 3.0, 1.0, 1e-6, np.random.default_rng(16)))",
        },
        # --- Valid: the indicator off, so the vector is drawn on the spike scale ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(113)
n_donors, n_instruments, n_covariates = 80, 3, 2
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.45) + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.45) + rng.standard_normal(n_donors)
pathway = 0.5 * receptor + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
state = np.linspace(-0.2, 0.6, 2 * n_instruments + 3 * n_covariates + 9)
state[n_instruments + n_covariates] = 1.0
state[2 * n_instruments + 2 * n_covariates + 1] = 1.0
state[2 * n_instruments + 3 * n_covariates + 4] = 1.4
state[2 * n_instruments + 3 * n_covariates + 5] = 0.001
state[2 * n_instruments + 3 * n_covariates + 6] = 0.002
state[2 * n_instruments + 3 * n_covariates + 7] = 0.0
state[2 * n_instruments + 3 * n_covariates + 8] = 0.35
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(update_communication_state(data, n_instruments, n_covariates, state, 100.0, 1e-4, 3.0, 1.0, 1e-6, np.random.default_rng(23)))",
            "gold_call": "digest(_oracle_update_communication_state(data, n_instruments, n_covariates, state, 100.0, 1e-4, 3.0, 1.0, 1e-6, np.random.default_rng(23)))",
        },
        # --- Boundary: a spike scale factor of one, where the two components coincide ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(127)
n_donors, n_instruments, n_covariates = 60, 2, 2
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.5) + rng.standard_normal(n_donors)
pathway = 0.5 * receptor + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
state = np.full(2 * n_instruments + 3 * n_covariates + 9, 0.2)
state[n_instruments + n_covariates] = 1.0
state[2 * n_instruments + 2 * n_covariates + 1] = 1.0
state[2 * n_instruments + 3 * n_covariates + 4] = 1.0
state[2 * n_instruments + 3 * n_covariates + 7] = 0.0
state[2 * n_instruments + 3 * n_covariates + 8] = 0.5
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(update_communication_state(data, n_instruments, n_covariates, state, 100.0, 1.0, 3.0, 1.0, 1e-6, np.random.default_rng(11)))",
            "gold_call": "digest(_oracle_update_communication_state(data, n_instruments, n_covariates, state, 100.0, 1.0, 3.0, 1.0, 1e-6, np.random.default_rng(11)))",
        },
        # --- Edge: an inclusion probability close to the boundary of the unit interval ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(137)
n_donors, n_instruments, n_covariates = 30, 1, 1
instruments = rng.standard_normal((n_donors, 2 * n_instruments))
covariates = rng.standard_normal((n_donors, n_covariates))
ligand = instruments[:, :n_instruments] @ np.full(n_instruments, 0.6) + rng.standard_normal(n_donors)
receptor = instruments[:, n_instruments:] @ np.full(n_instruments, 0.6) + rng.standard_normal(n_donors)
pathway = 0.5 * receptor + rng.standard_normal(n_donors)
data = np.column_stack([instruments, covariates, ligand, receptor, pathway])
data = data - data.mean(axis=0)
state = np.linspace(-0.3, 0.7, 2 * n_instruments + 3 * n_covariates + 9)
state[n_instruments + n_covariates] = 1.0
state[2 * n_instruments + 2 * n_covariates + 1] = 1.0
state[2 * n_instruments + 3 * n_covariates + 4] = 1.0
state[2 * n_instruments + 3 * n_covariates + 7] = 1.0
state[2 * n_instruments + 3 * n_covariates + 8] = 1e-6
def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(update_communication_state(data, n_instruments, n_covariates, state, 100.0, 1e-4, 3.0, 1.0, 1e-6, np.random.default_rng(19)))",
            "gold_call": "digest(_oracle_update_communication_state(data, n_instruments, n_covariates, state, 100.0, 1e-4, 3.0, 1.0, 1e-6, np.random.default_rng(19)))",
        },
        # --- Invalid: a state whose inclusion probability sits at one ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(1)
data = rng.standard_normal((40, 2 * 2 + 2 + 3))
state = np.full(2 * 2 + 3 * 2 + 9, 0.3)
state[2 + 2] = 1.0
state[2 * 2 + 2 * 2 + 1] = 1.0
state[2 * 2 + 3 * 2 + 4] = 1.0
state[2 * 2 + 3 * 2 + 7] = 1.0
state[2 * 2 + 3 * 2 + 8] = 1.0
def run_model():
    try:
        update_communication_state(data, 2, 2, state, 100.0, 1e-4, 3.0, 1.0, 1e-6, np.random.default_rng(1))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_update_communication_state(data, 2, 2, state, 100.0, 1e-4, 3.0, 1.0, 1e-6, np.random.default_rng(1))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative beta shape ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(1)
data = rng.standard_normal((40, 2 * 2 + 2 + 3))
state = np.full(2 * 2 + 3 * 2 + 9, 0.3)
state[2 + 2] = 1.0
state[2 * 2 + 2 * 2 + 1] = 1.0
state[2 * 2 + 3 * 2 + 4] = 1.0
state[2 * 2 + 3 * 2 + 7] = 1.0
state[2 * 2 + 3 * 2 + 8] = 0.75
def run_model():
    try:
        update_communication_state(data, 2, 2, state, 100.0, 1e-4, -3.0, 1.0, 1e-6, np.random.default_rng(1))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_update_communication_state(data, 2, 2, state, 100.0, 1e-4, -3.0, 1.0, 1e-6, np.random.default_rng(1))
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
