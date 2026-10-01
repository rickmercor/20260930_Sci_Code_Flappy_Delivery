#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np  # noqa: E402, F811


def compute_annotation_probabilities(
    positions_bp: np.ndarray,
    tss_bp: float,
    alpha: float,
    kappa: float,
) -> np.ndarray:
    """Reference implementation."""
    positions = np.asarray(positions_bp, dtype=np.float64)
    if positions.ndim != 1 or positions.size == 0 or not np.all(np.isfinite(positions)):
        raise ValueError("positions_bp must be a nonempty finite one-dimensional array")
    try:
        tss = float(tss_bp)
        intercept = float(alpha)
        coefficient = float(kappa)
    except (TypeError, ValueError) as exc:
        raise ValueError("tss_bp, alpha, and kappa must be finite scalars") from exc
    if not np.all(np.isfinite([tss, intercept, coefficient])):
        raise ValueError("tss_bp, alpha, and kappa must be finite scalars")

    distances_mb = np.abs(positions - tss) / 1_000_000.0
    linear_predictor = intercept + coefficient * distances_mb
    inclusion_probabilities = np.exp(-np.logaddexp(0.0, -linear_predictor))
    return np.vstack((distances_mb, inclusion_probabilities))

import numpy as np  # noqa: E402, F811


def compute_collapsed_block_probability(
    X_rot: np.ndarray,
    block_indices: np.ndarray,
    residual_without_block: np.ndarray,
    sigma2: float,
    eta_beta: float,
    inclusion_probabilities: np.ndarray,
    pi_block: float,
) -> np.ndarray:
    """Reference implementation."""
    X = np.asarray(X_rot, dtype=np.float64)
    indices = np.asarray(block_indices)
    residual = np.asarray(residual_without_block, dtype=np.float64)
    probabilities = np.asarray(inclusion_probabilities, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or not np.all(np.isfinite(X)):
        raise ValueError("X_rot must be a nonempty finite matrix")
    if residual.shape != (X.shape[0],) or not np.all(np.isfinite(residual)):
        raise ValueError("residual_without_block must be a finite vector of length n")
    if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("block_indices must be a nonempty integer vector")
    indices = indices.astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= X.shape[1]) or np.unique(indices).size != indices.size:
        raise ValueError("block_indices must be unique valid column indices")
    if probabilities.shape != (X.shape[1],) or not np.all(np.isfinite(probabilities)):
        raise ValueError("inclusion_probabilities must have shape (p,)")
    if np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
        raise ValueError("all SNP probabilities must lie strictly between zero and one")
    try:
        residual_variance = float(sigma2)
        slab_ratio = float(eta_beta)
        block_prior = float(pi_block)
    except (TypeError, ValueError) as exc:
        raise ValueError("variance parameters and pi_block must be scalar") from exc
    if not np.isfinite(residual_variance) or residual_variance <= 0.0:
        raise ValueError("sigma2 must be positive and finite")
    if not np.isfinite(slab_ratio) or slab_ratio <= 0.0:
        raise ValueError("eta_beta must be positive and finite")
    if not np.isfinite(block_prior) or not 0.0 < block_prior < 1.0:
        raise ValueError("pi_block must lie strictly between zero and one")

    log_bayes_factors = np.empty(indices.size, dtype=np.float64)
    evidence_terms = np.empty(indices.size, dtype=np.float64)
    for offset, j in enumerate(indices):
        x_j = X[:, j]
        xtx = float(x_j @ x_j)
        xty = float(x_j @ residual)
        variance = 1.0 / (
            xtx / residual_variance + 1.0 / (residual_variance * slab_ratio)
        )
        mean = variance * xty / residual_variance
        log_bayes_factor = (
            0.5 * np.log(variance)
            - 0.5 * np.log(residual_variance * slab_ratio)
            + 0.5 * mean * mean / variance
        )
        log_bayes_factors[offset] = log_bayes_factor
        evidence_terms[offset] = np.logaddexp(
            np.log1p(-probabilities[j]),
            np.log(probabilities[j]) + log_bayes_factor,
        )

    log_evidence_increment = float(np.sum(evidence_terms))
    block_log_odds = (
        np.log(block_prior) - np.log1p(-block_prior) + log_evidence_increment
    )
    block_probability = np.exp(-np.logaddexp(0.0, -block_log_odds))
    return np.concatenate(
        (
            np.array(
                [log_evidence_increment, float(block_probability)],
                dtype=np.float64,
            ),
            log_bayes_factors,
        )
    )

import numpy as np  # noqa: E402, F811


def scan_active_block(
    X_rot: np.ndarray,
    block_indices: np.ndarray,
    residual: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma2: float,
    eta_beta: float,
    inclusion_probabilities: np.ndarray,
    u_gate: np.ndarray,
    z_normal: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    X = np.asarray(X_rot, dtype=np.float64)
    indices = np.asarray(block_indices)
    running_residual = np.asarray(residual, dtype=np.float64).copy()
    effects = np.asarray(beta, dtype=np.float64).copy()
    indicators_raw = np.asarray(gamma)
    probabilities = np.asarray(inclusion_probabilities, dtype=np.float64)
    uniforms = np.asarray(u_gate, dtype=np.float64)
    normals = np.asarray(z_normal, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or not np.all(np.isfinite(X)):
        raise ValueError("X_rot must be a nonempty finite matrix")
    if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("block_indices must be a nonempty integer vector")
    indices = indices.astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= X.shape[1]) or np.unique(indices).size != indices.size:
        raise ValueError("block_indices must be unique valid column indices")
    if running_residual.shape != (X.shape[0],) or not np.all(np.isfinite(running_residual)):
        raise ValueError("residual must be a finite vector of length n")
    if effects.shape != (X.shape[1],) or not np.all(np.isfinite(effects)):
        raise ValueError("beta must be a finite vector of length p")
    if indicators_raw.shape != (X.shape[1],) or not np.all(
        (indicators_raw == 0) | (indicators_raw == 1)
    ):
        raise ValueError("gamma must be a binary vector of length p")
    indicators = indicators_raw.astype(bool).copy()
    if np.any(effects[~indicators] != 0.0):
        raise ValueError("beta must be zero wherever gamma is zero")
    if probabilities.shape != (X.shape[1],) or np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
        raise ValueError("inclusion_probabilities must have shape (p,) and lie in (0, 1)")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("inclusion_probabilities must be finite")
    if uniforms.shape != (indices.size,) or np.any(uniforms < 0.0) or np.any(uniforms >= 1.0):
        raise ValueError("u_gate must contain one variate in [0, 1) per block SNP")
    if normals.shape != (indices.size,) or not np.all(np.isfinite(normals)):
        raise ValueError("z_normal must contain one finite variate per block SNP")
    try:
        residual_variance = float(sigma2)
        slab_ratio = float(eta_beta)
    except (TypeError, ValueError) as exc:
        raise ValueError("sigma2 and eta_beta must be scalar") from exc
    if not np.isfinite(residual_variance) or residual_variance <= 0.0:
        raise ValueError("sigma2 must be positive and finite")
    if not np.isfinite(slab_ratio) or slab_ratio <= 0.0:
        raise ValueError("eta_beta must be positive and finite")

    scan_probabilities = np.empty(indices.size, dtype=np.float64)
    for offset, j in enumerate(indices):
        if indicators[j]:
            running_residual += X[:, j] * effects[j]
            indicators[j] = False
            effects[j] = 0.0

        x_j = X[:, j]
        xtx = float(x_j @ x_j)
        xty = float(x_j @ running_residual)
        variance = 1.0 / (
            xtx / residual_variance + 1.0 / (residual_variance * slab_ratio)
        )
        mean = variance * xty / residual_variance
        log_bayes_factor = (
            0.5 * np.log(variance)
            - 0.5 * np.log(residual_variance * slab_ratio)
            + 0.5 * mean * mean / variance
        )
        log_odds = (
            np.log(probabilities[j])
            - np.log1p(-probabilities[j])
            + log_bayes_factor
        )
        inclusion_probability = np.exp(-np.logaddexp(0.0, -log_odds))
        scan_probabilities[offset] = inclusion_probability
        if uniforms[offset] < inclusion_probability:
            indicators[j] = True
            effects[j] = mean + np.sqrt(variance) * normals[offset]
            running_residual -= X[:, j] * effects[j]

    return np.concatenate(
        (
            running_residual,
            effects,
            indicators.astype(np.float64),
            scan_probabilities,
        )
    )

import numpy as np


def sample_polygenic_background(
    y_rot: np.ndarray,
    X_rot: np.ndarray,
    beta: np.ndarray,
    eigenvalues: np.ndarray,
    sigma2: float,
    eta_g: float,
    z_normal: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    expression = np.asarray(y_rot, dtype=np.float64)
    X = np.asarray(X_rot, dtype=np.float64)
    effects = np.asarray(beta, dtype=np.float64)
    eigvals = np.asarray(eigenvalues, dtype=np.float64)
    normals = np.asarray(z_normal, dtype=np.float64)

    if (
        X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not np.all(np.isfinite(X))
    ):
        raise ValueError("X_rot must be a nonempty finite matrix")

    if expression.shape != (X.shape[0],) or not np.all(
        np.isfinite(expression)
    ):
        raise ValueError("y_rot must be a finite vector of length n")

    if effects.shape != (X.shape[1],) or not np.all(np.isfinite(effects)):
        raise ValueError("beta must be a finite vector of length p")

    if (
        eigvals.shape != (X.shape[0],)
        or not np.all(np.isfinite(eigvals))
        or np.any(eigvals < 0.0)
    ):
        raise ValueError(
            "eigenvalues must be a finite nonnegative vector of length n"
        )

    if normals.shape != (X.shape[0],) or not np.all(np.isfinite(normals)):
        raise ValueError("z_normal must be a finite vector of length n")

    try:
        residual_variance = float(sigma2)
        polygenic_ratio = float(eta_g)
    except (TypeError, ValueError) as exc:
        raise ValueError("sigma2 and eta_g must be scalar") from exc

    if not np.isfinite(residual_variance) or residual_variance <= 0.0:
        raise ValueError("sigma2 must be positive and finite")

    if not np.isfinite(polygenic_ratio) or polygenic_ratio <= 0.0:
        raise ValueError("eta_g must be positive and finite")

    rotated_cis_residual = expression - X @ effects
    effective_eigenvalues = np.where(eigvals < 1e-12, 0.0, eigvals)

    shrinkage = (
        polygenic_ratio
        * effective_eigenvalues
        / (1.0 + polygenic_ratio * effective_eigenvalues)
    )
    conditional_variance = residual_variance * shrinkage
    g_rot = (
        shrinkage * rotated_cis_residual
        + np.sqrt(conditional_variance) * normals
    )

    return np.vstack((g_rot, shrinkage, conditional_variance))

import numpy as np  # noqa: E402, F811


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

import numpy as np  # noqa: E402, F811


def summarize_sparse_prediction(
    beta_samples: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    samples = np.asarray(beta_samples, dtype=np.float64)
    test_genotypes = np.asarray(X_test, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError("beta_samples must be a nonempty two-dimensional array")
    if test_genotypes.ndim != 2 or test_genotypes.shape[0] == 0:
        raise ValueError("X_test must be a nonempty two-dimensional array")
    if samples.shape[1] != test_genotypes.shape[1]:
        raise ValueError("beta_samples and X_test must have the same SNP dimension")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(test_genotypes)):
        raise ValueError("beta_samples and X_test must contain only finite values")

    beta_mean = np.mean(samples, axis=0)
    predictions = test_genotypes @ beta_mean
    return np.concatenate(
        (beta_mean, predictions, np.array([predictions[0]], dtype=np.float64))
    )

import numpy as np  # noqa: E402, F811


def run_full_pipeline(
    X_rot: np.ndarray,
    y_rot: np.ndarray,
    eigenvalues: np.ndarray,
    positions_bp: np.ndarray,
    tss_bp: float,
    block_indices: list,
    X_test: np.ndarray,
    n_sweeps: int,
    u_block: np.ndarray,
    u_snp: np.ndarray,
    z_beta: np.ndarray,
    z_g: np.ndarray,
    mh_proposal_normals: np.ndarray,
    mh_uniforms: np.ndarray,
    sigma2: float,
    eta_beta: float,
    eta_g: float,
    pi_block: float,
    alpha_init: float,
    kappa_init: float,
    alpha_prior_mean: float,
    kappa_prior_mean: float,
    alpha_prior_var: float,
    kappa_prior_var: float,
    alpha_step: float,
    kappa_step: float,
) -> float:
    """Reference implementation chaining every earlier step."""
    X = np.asarray(X_rot, dtype=np.float64)
    expression = np.asarray(y_rot, dtype=np.float64)
    eigvals = np.asarray(eigenvalues, dtype=np.float64)
    positions = np.asarray(positions_bp, dtype=np.float64)
    test_genotypes = np.asarray(X_test, dtype=np.float64)
    block_uniforms = np.asarray(u_block, dtype=np.float64)
    snp_uniforms = np.asarray(u_snp, dtype=np.float64)
    slab_normals = np.asarray(z_beta, dtype=np.float64)
    dense_normals = np.asarray(z_g, dtype=np.float64)
    annotation_normals = np.asarray(mh_proposal_normals, dtype=np.float64)
    annotation_uniforms = np.asarray(mh_uniforms, dtype=np.float64)
    if isinstance(n_sweeps, bool) or not isinstance(n_sweeps, (int, np.integer)):
        raise ValueError("n_sweeps must be a positive integer")
    sweeps = int(n_sweeps)
    if sweeps <= 0:
        raise ValueError("n_sweeps must be a positive integer")
    if X.ndim != 2:
        raise ValueError("X_rot must be two dimensional")
    n, p = X.shape
    n_blocks = len(block_indices)
    expected_shapes = [
        (block_uniforms, (sweeps, n_blocks), "u_block"),
        (snp_uniforms, (sweeps, p), "u_snp"),
        (slab_normals, (sweeps, p), "z_beta"),
        (dense_normals, (sweeps, n), "z_g"),
    ]
    for values, expected_shape, name in expected_shapes:
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite with shape {expected_shape}")
    if annotation_normals.ndim != 3 or annotation_normals.shape[0] != sweeps or annotation_normals.shape[2] != 2:
        raise ValueError("mh_proposal_normals must have shape (n_sweeps, q, 2)")
    if annotation_uniforms.shape != annotation_normals.shape[:2]:
        raise ValueError("mh_uniforms must have shape (n_sweeps, q)")
    if not np.all(np.isfinite(annotation_normals)) or not np.all(np.isfinite(annotation_uniforms)):
        raise ValueError("annotation variates must be finite")
    if expression.shape != (n,) or eigvals.shape != (n,) or positions.shape != (p,):
        raise ValueError("training inputs have incompatible dimensions")
    if test_genotypes.ndim != 2 or test_genotypes.shape[1] != p or test_genotypes.shape[0] == 0:
        raise ValueError("X_test must be nonempty with p columns")

    beta = np.zeros(p, dtype=np.float64)
    gamma = np.zeros(p, dtype=bool)
    zeta = np.zeros(n_blocks, dtype=bool)
    g_rot = np.zeros(n, dtype=np.float64)
    alpha = float(alpha_init)
    kappa = float(kappa_init)
    beta_samples = []

    for sweep in range(sweeps):
        annotation = compute_annotation_probabilities(  # noqa: F821
            positions, tss_bp, alpha, kappa
        )
        distances_mb = annotation[0]
        inclusion_probabilities = annotation[1]
        residual = expression - X @ beta - g_rot

        for block_number, block in enumerate(block_indices):
            indices = np.asarray(block, dtype=np.int64)
            residual_without_block = residual + X[:, indices] @ beta[indices]
            block_update = compute_collapsed_block_probability(  # noqa: F821
                X,
                indices,
                residual_without_block,
                sigma2,
                eta_beta,
                inclusion_probabilities,
                pi_block,
            )
            zeta[block_number] = (
                block_uniforms[sweep, block_number]
                < block_update[1]
            )
            if zeta[block_number]:
                scan = scan_active_block(  # noqa: F821
                    X,
                    indices,
                    residual,
                    beta,
                    gamma,
                    sigma2,
                    eta_beta,
                    inclusion_probabilities,
                    snp_uniforms[sweep, indices],
                    slab_normals[sweep, indices],
                )
                residual = scan[:n].copy()
                beta = scan[n : n + p].copy()
                gamma = scan[n + p : n + 2 * p].astype(bool)
            else:
                residual = residual_without_block
                beta[indices] = 0.0
                gamma[indices] = False

        dense_update = sample_polygenic_background(  # noqa: F821
            expression,
            X,
            beta,
            eigvals,
            sigma2,
            eta_g,
            dense_normals[sweep],
        )
        g_rot = dense_update[0]
        annotation_update = update_annotation_prior(  # noqa: F821
            alpha,
            kappa,
            distances_mb,
            gamma,
            zeta,
            block_indices,
            annotation_normals[sweep],
            annotation_uniforms[sweep],
            alpha_prior_mean,
            kappa_prior_mean,
            alpha_prior_var,
            kappa_prior_var,
            alpha_step,
            kappa_step,
        )
        alpha = float(annotation_update[0])
        kappa = float(annotation_update[1])
        beta_samples.append(beta.copy())

    summary = summarize_sparse_prediction(  # noqa: F821
        np.asarray(beta_samples), test_genotypes
    )
    return float(summary[-1])
SCICODE_GOLD_EOF
