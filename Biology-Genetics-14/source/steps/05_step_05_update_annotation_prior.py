"""
Update the coefficients of the TSS-informed SNP inclusion prior.

For SNP j, ell_j = alpha + kappa * a_j and pi_j = sigmoid(ell_j), where a_j is

its TSS distance in megabases. Conditional on the current block state, only SNPs

inside active blocks contribute Bernoulli observations to the annotation

likelihood; zeros forced by inactive blocks are structural and are not evidence

about alpha or kappa. The log target is the active-set sum

gamma_j ell_j - log(1 + exp(ell_j)) plus independent Gaussian log priors for

alpha and kappa. Each proposal adds alpha_step * z_alpha and

kappa_step * z_kappa to the latest accepted state, and it is accepted when the

log of its supplied uniform is below the change in log posterior. Updating the

current state after every acceptance is essential because the proposal row is a

Metropolis path, not a collection of comparisons with the sweep's initial

coefficients.

Inputs

------

alpha, kappa : current annotation-prior coefficients

distances_mb, gamma : SNP annotations and inclusion indicators

zeta, block_indices : block activity and the SNP partition

proposal_normals, proposal_uniforms : fixed random-walk and acceptance variates

alpha_prior_mean, kappa_prior_mean : Gaussian prior centers

alpha_prior_var, kappa_prior_var : Gaussian prior variances

alpha_step, kappa_step : random-walk scales

Returns

-------

alpha, kappa : coefficients after the ordered proposal sequence

accepted : Boolean acceptance indicators in proposal order

log_posterior : log target at the final coefficient state

Annotation coefficients are inferred from inclusion states inside active blocks; structural zeros forced by inactive blocks do not supply annotation evidence. This step applies the supplied random-walk updates to those coefficients.

Returns
-------
float64 vector of length q + 3: alpha, kappa, the numeric acceptance indicators in proposal order, and the final log posterior
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def update_annotation_prior(
    alpha: float,
    kappa: float,
    distances_mb: np.ndarray,
    gamma: np.ndarray,
    zeta: np.ndarray,
    block_indices: list,
    proposal_normals: np.ndarray,
    proposal_uniforms: np.ndarray,
    alpha_prior_mean: float,
    kappa_prior_mean: float,
    alpha_prior_var: float,
    kappa_prior_var: float,
    alpha_step: float,
    kappa_step: float,
) -> np.ndarray:
    """Apply ordered random-walk Metropolis updates to annotation coefficients.

    Parameters
    ----------
    alpha, kappa : float
        Current logistic-prior coefficients.
    distances_mb : np.ndarray
        Nonnegative TSS distances with shape (p,).
    gamma : np.ndarray
        Binary SNP inclusion vector with shape (p,).
    zeta : np.ndarray
        Binary block activity vector.
    block_indices : list
        Ordered list of disjoint integer SNP-index arrays.
    proposal_normals : np.ndarray
        Standard-normal proposal increments with shape (q, 2).
    proposal_uniforms : np.ndarray
        Uniform acceptance variates with shape (q,).
    alpha_prior_mean, kappa_prior_mean : float
        Gaussian prior means.
    alpha_prior_var, kappa_prior_var : float
        Positive Gaussian prior variances.
    alpha_step, kappa_step : float
        Positive proposal scales.

    Raises
    ------
    ValueError
        If state dimensions, block membership, priors, or variates are invalid.

    Returns
    -------
    result : np.ndarray
        Float64 vector containing updated alpha, updated kappa, q numeric
        acceptance indicators, and the final log posterior, in that order.
    """
    return result  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_update_annotation_prior(
    alpha: float,
    kappa: float,
    distances_mb: np.ndarray,
    gamma: np.ndarray,
    zeta: np.ndarray,
    block_indices: list,
    proposal_normals: np.ndarray,
    proposal_uniforms: np.ndarray,
    alpha_prior_mean: float,
    kappa_prior_mean: float,
    alpha_prior_var: float,
    kappa_prior_var: float,
    alpha_step: float,
    kappa_step: float,
) -> np.ndarray:
    """Reference implementation."""
    distances = np.asarray(distances_mb, dtype=np.float64)
    indicators_raw = np.asarray(gamma)
    block_state_raw = np.asarray(zeta)
    proposals = np.asarray(proposal_normals, dtype=np.float64)
    uniforms = np.asarray(proposal_uniforms, dtype=np.float64)
    if distances.ndim != 1 or distances.size == 0 or not np.all(np.isfinite(distances)) or np.any(distances < 0.0):
        raise ValueError("distances_mb must be a nonempty finite nonnegative vector")
    if indicators_raw.shape != distances.shape or not np.all(
        (indicators_raw == 0) | (indicators_raw == 1)
    ):
        raise ValueError("gamma must be a binary vector of length p")
    if block_state_raw.shape != (len(block_indices),) or not np.all(
        (block_state_raw == 0) | (block_state_raw == 1)
    ):
        raise ValueError("zeta must have one binary entry per block")
    if proposals.ndim != 2 or proposals.shape[1] != 2 or proposals.shape[0] == 0 or not np.all(np.isfinite(proposals)):
        raise ValueError("proposal_normals must be a nonempty finite array with shape (q, 2)")
    if uniforms.shape != (proposals.shape[0],) or np.any(uniforms <= 0.0) or np.any(uniforms >= 1.0):
        raise ValueError("proposal_uniforms must contain q values strictly between zero and one")

    active_mask = np.zeros(distances.size, dtype=bool)
    seen = np.zeros(distances.size, dtype=bool)
    block_state = block_state_raw.astype(bool)
    for block_number, block in enumerate(block_indices):
        indices = np.asarray(block)
        if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("each block must be a nonempty integer vector")
        indices = indices.astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= distances.size) or np.any(seen[indices]):
            raise ValueError("blocks must contain disjoint valid SNP indices")
        seen[indices] = True
        if block_state[block_number]:
            active_mask[indices] = True
    if not np.all(seen):
        raise ValueError("block_indices must cover every SNP")
    indicators = indicators_raw.astype(np.float64)
    if np.any(indicators_raw[~active_mask] != 0):
        raise ValueError("gamma must be zero outside active blocks")

    scalar_values = [
        alpha,
        kappa,
        alpha_prior_mean,
        kappa_prior_mean,
        alpha_prior_var,
        kappa_prior_var,
        alpha_step,
        kappa_step,
    ]
    try:
        scalar_values = [float(value) for value in scalar_values]
    except (TypeError, ValueError) as exc:
        raise ValueError("coefficient, prior, and step parameters must be scalar") from exc
    if not np.all(np.isfinite(scalar_values)):
        raise ValueError("coefficient, prior, and step parameters must be finite")
    (
        current_alpha,
        current_kappa,
        alpha_mean,
        kappa_mean,
        alpha_variance,
        kappa_variance,
        alpha_scale,
        kappa_scale,
    ) = scalar_values
    if alpha_variance <= 0.0 or kappa_variance <= 0.0:
        raise ValueError("prior variances must be positive")
    if alpha_scale <= 0.0 or kappa_scale <= 0.0:
        raise ValueError("proposal scales must be positive")

    def log_posterior(candidate_alpha, candidate_kappa):
        linear_predictor = (
            candidate_alpha + candidate_kappa * distances[active_mask]
        )
        log_likelihood = np.sum(
            indicators[active_mask] * linear_predictor
            - np.logaddexp(0.0, linear_predictor)
        )
        log_prior = (
            -0.5 * (candidate_alpha - alpha_mean) ** 2 / alpha_variance
            - 0.5 * (candidate_kappa - kappa_mean) ** 2 / kappa_variance
        )
        return float(log_likelihood + log_prior)

    accepted = np.zeros(proposals.shape[0], dtype=bool)
    current_log_posterior = log_posterior(current_alpha, current_kappa)
    for proposal_number in range(proposals.shape[0]):
        candidate_alpha = current_alpha + alpha_scale * proposals[proposal_number, 0]
        candidate_kappa = current_kappa + kappa_scale * proposals[proposal_number, 1]
        candidate_log_posterior = log_posterior(candidate_alpha, candidate_kappa)
        if np.log(uniforms[proposal_number]) < candidate_log_posterior - current_log_posterior:
            current_alpha = candidate_alpha
            current_kappa = candidate_kappa
            current_log_posterior = candidate_log_posterior
            accepted[proposal_number] = True
    return np.concatenate(
        (
            np.array([current_alpha, current_kappa], dtype=np.float64),
            accepted.astype(np.float64),
            np.array([current_log_posterior], dtype=np.float64),
        )
    )

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
distances = np.array([0.001, 0.0, 0.05, 0.2, 0.7, 1.2])
gamma = np.array([1, 0, 0, 0, 0, 0])
zeta = np.array([1, 0])
blocks = [np.array([0, 1, 2]), np.array([3, 4, 5])]
proposals = np.array([[0.4, -0.6], [-1.1, 0.3], [0.7, -0.2], [0.2, 0.9]])
uniforms = np.array([0.99, 0.95, 0.4, 0.97])
""",
            "call": "update_annotation_prior(-1.55, -0.9, distances, gamma, zeta, blocks, proposals, uniforms, -2.9444389791664403, 0.0, 4.0, 4.0, 0.3, 0.5)",
            "gold_call": "_oracle_update_annotation_prior(-1.55, -0.9, distances, gamma, zeta, blocks, proposals, uniforms, -2.9444389791664403, 0.0, 4.0, 4.0, 0.3, 0.5)",
        },
        {
            "setup": """import numpy as np
distances = np.array([0.0])
gamma = np.array([0])
zeta = np.array([0])
blocks = [np.array([0])]
proposals = np.array([[1.0, -1.0]])
uniforms = np.array([0.5])
""",
            "call": "update_annotation_prior(-2.0, 0.0, distances, gamma, zeta, blocks, proposals, uniforms, -2.0, 0.0, 4.0, 4.0, 0.3, 0.5)",
            "gold_call": "_oracle_update_annotation_prior(-2.0, 0.0, distances, gamma, zeta, blocks, proposals, uniforms, -2.0, 0.0, 4.0, 4.0, 0.3, 0.5)",
        },
        {
            "setup": """import numpy as np
distances = np.array([0.0])
gamma = np.array([0])
zeta = np.array([1])
blocks = [np.array([0])]
proposals = np.array([[0.0, 0.0]])
uniforms = np.array([0.5])
def run_model():
    try:
        update_annotation_prior(0.0, 0.0, distances, gamma, zeta, blocks, proposals, uniforms, 0.0, 0.0, 0.0, 4.0, 0.3, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_update_annotation_prior(0.0, 0.0, distances, gamma, zeta, blocks, proposals, uniforms, 0.0, 0.0, 0.0, 4.0, 0.3, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
