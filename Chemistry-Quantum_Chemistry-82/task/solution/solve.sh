#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import math

import numpy as np

def build_shg_frequency_grid(omega_max: float = 0.36) -> np.ndarray:
    if not np.isscalar(omega_max) or not np.isfinite(omega_max) or float(omega_max) <= 0.0:
        raise ValueError("omega_max must be a positive finite scalar")
    return float(omega_max) * np.arange(5, dtype=float) / 8.0

import math

import numpy as np

def compute_effective_hyperpolarizability(
    beta: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    beta = np.asarray(beta, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if beta.shape != (3, 3, 3) or not np.all(np.isfinite(beta)):
        raise ValueError("beta must be a finite (3,3,3) tensor")
    if directions.ndim != 2 or directions.shape[1] != 3 or directions.shape[0] < 4:
        raise ValueError("directions must have shape (n,3) with n at least 4")
    if not np.all(np.isfinite(directions)):
        raise ValueError("directions must be finite")
    norms = np.linalg.norm(directions, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("every direction must be a unit vector")
    return np.einsum("kij,ni,nj->nk", beta, directions, directions)

import math

import numpy as np

def compute_relative_rms_total(
    beta_basis: np.ndarray,
    beta_mra: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray,
) -> tuple:
    directions = np.asarray(directions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (directions.shape[0],) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must match the direction count and be finite")
    if np.any(weights <= 0.0) or not np.isclose(np.sum(weights), 4.0 * np.pi, rtol=0.0, atol=1e-10):
        raise ValueError("weights must be positive and sum to 4*pi")
    basis_eff = compute_effective_hyperpolarizability(beta_basis, directions)
    mra_eff = compute_effective_hyperpolarizability(beta_mra, directions)
    error = basis_eff - mra_eff
    mra_scale = float(np.dot(weights, np.linalg.norm(mra_eff, axis=1)) / (4.0 * np.pi))
    if mra_scale <= 1e-14:
        raise ValueError("the MRA response norm must be nonzero")
    rms_error = math.sqrt(float(np.sum(weights[:, None] * error**2) / (4.0 * np.pi)))
    coefficients = np.arange(1, error.size + 1, dtype=float).reshape(error.shape)
    checksum = float(np.sum(coefficients * error))
    return float(rms_error / mra_scale), float(mra_scale), checksum

import math

import numpy as np

def compute_signed_projection_metrics(
    beta_basis: np.ndarray,
    beta_mra: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray,
) -> tuple:
    directions = np.asarray(directions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (directions.shape[0],) or np.any(~np.isfinite(weights)):
        raise ValueError("weights must match the direction count and be finite")
    if np.any(weights <= 0.0) or not np.isclose(np.sum(weights), 4.0 * np.pi, rtol=0.0, atol=1e-10):
        raise ValueError("weights must be positive and sum to 4*pi")
    basis_eff = compute_effective_hyperpolarizability(beta_basis, directions)
    mra_eff = compute_effective_hyperpolarizability(beta_mra, directions)
    denom = np.sum(mra_eff * mra_eff, axis=1)
    if np.any(denom <= 1e-14):
        raise ValueError("every sampled MRA effective response must be nonzero")
    signed = np.sum(basis_eff * mra_eff, axis=1) / denom - 1.0
    error_norm_sq = np.sum((basis_eff - mra_eff) ** 2, axis=1)
    mra_scale = float(np.dot(weights, np.linalg.norm(mra_eff, axis=1)) / (4.0 * np.pi))
    if mra_scale <= 1e-14:
        raise ValueError("the MRA response norm must be nonzero")
    positive = signed > 0.0
    negative = signed < 0.0
    positive_weight = float(np.sum(weights[positive]))
    negative_weight = float(np.sum(weights[negative]))
    positive_rms = (
        math.sqrt(float(np.sum(weights[positive] * error_norm_sq[positive]) / positive_weight)) / mra_scale
        if positive_weight > 0.0
        else 0.0
    )
    negative_rms = (
        math.sqrt(float(np.sum(weights[negative] * error_norm_sq[negative]) / negative_weight)) / mra_scale
        if negative_weight > 0.0
        else 0.0
    )
    checksum = float(np.dot(np.arange(1, signed.size + 1, dtype=float), signed))
    return (
        float(positive_rms),
        float(negative_rms),
        checksum,
        float(positive_weight / (4.0 * np.pi)),
        float(negative_weight / (4.0 * np.pi)),
    )

import math

import numpy as np

def compute_component_rms_anisotropy(
    beta_basis: np.ndarray,
    beta_mra: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray,
) -> tuple:
    directions = np.asarray(directions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (directions.shape[0],) or np.any(~np.isfinite(weights)):
        raise ValueError("weights must match the direction count and be finite")
    if np.any(weights <= 0.0) or not np.isclose(np.sum(weights), 4.0 * np.pi, rtol=0.0, atol=1e-10):
        raise ValueError("weights must be positive and sum to 4*pi")
    basis_eff = compute_effective_hyperpolarizability(beta_basis, directions)
    mra_eff = compute_effective_hyperpolarizability(beta_mra, directions)
    mra_scale = float(np.dot(weights, np.linalg.norm(mra_eff, axis=1)) / (4.0 * np.pi))
    if mra_scale <= 1e-14:
        raise ValueError("the MRA response norm must be nonzero")
    error = basis_eff - mra_eff
    component = np.sqrt(np.sum(weights[:, None] * error**2, axis=0) / (4.0 * np.pi)) / mra_scale
    if np.min(component) <= 1e-14:
        raise ValueError("all component errors must be nonzero for the anisotropy ratio")
    anisotropy = float(np.max(component) / np.min(component))
    checksum = float(np.dot(np.arange(1, 4, dtype=float), component))
    return component.astype(float), anisotropy, checksum

import math

import numpy as np

def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.sum(np.exp(values - maximum), axis=axis)
    )


def _fit_full_gmm(features: np.ndarray, clusters: int, regularization: float) -> tuple:
    n_samples, dimension = features.shape
    center = np.mean(features, axis=0)
    selected = [int(np.argmax(np.sum((features - center) ** 2, axis=1)))]
    while len(selected) < clusters:
        distances = np.min(
            np.stack(
                [np.sum((features - features[index]) ** 2, axis=1) for index in selected],
                axis=1,
            ),
            axis=1,
        )
        distances[selected] = -1.0
        selected.append(int(np.argmax(distances)))
    means = features[selected].copy()
    global_covariance = np.cov(features, rowvar=False, bias=True)
    if dimension == 1:
        global_covariance = np.array([[float(global_covariance)]], dtype=float)
    global_covariance = np.asarray(global_covariance, dtype=float) + regularization * np.eye(dimension)
    covariances = np.repeat(global_covariance[None, :, :], clusters, axis=0)
    mixing = np.full(clusters, 1.0 / clusters, dtype=float)
    previous = -np.inf
    responsibilities = np.full((n_samples, clusters), 1.0 / clusters, dtype=float)
    for _ in range(250):
        log_probability = np.empty((n_samples, clusters), dtype=float)
        for cluster in range(clusters):
            sign, logdet = np.linalg.slogdet(covariances[cluster])
            if sign <= 0:
                raise ValueError("covariance is not positive definite")
            delta = features - means[cluster]
            solved = np.linalg.solve(covariances[cluster], delta.T).T
            quadratic = np.sum(delta * solved, axis=1)
            log_probability[:, cluster] = (
                np.log(mixing[cluster])
                - 0.5 * (dimension * np.log(2.0 * np.pi) + logdet + quadratic)
            )
        normalizer = _logsumexp(log_probability, axis=1)
        log_likelihood = float(np.sum(normalizer))
        responsibilities = np.exp(log_probability - normalizer[:, None])
        effective = np.sum(responsibilities, axis=0) + 1e-15
        mixing = effective / n_samples
        means = (responsibilities.T @ features) / effective[:, None]
        for cluster in range(clusters):
            delta = features - means[cluster]
            covariances[cluster] = (
                (responsibilities[:, cluster, None] * delta).T @ delta / effective[cluster]
                + regularization * np.eye(dimension)
            )
        if np.isfinite(previous) and abs(log_likelihood - previous) <= 1e-10 * (1.0 + abs(previous)):
            break
        previous = log_likelihood
    parameters = (clusters - 1) + clusters * dimension + clusters * dimension * (dimension + 1) / 2
    bic = float(-2.0 * log_likelihood + parameters * np.log(n_samples))
    return bic, log_likelihood, responsibilities, means


def select_convergence_clusters(
    features: np.ndarray, max_clusters: int = 4, regularization: float = 1e-6
) -> tuple:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[0] < 8 or features.shape[1] < 2:
        raise ValueError("features must have shape (n,d) with n at least 8 and d at least 2")
    if not np.all(np.isfinite(features)) or np.any(np.std(features, axis=0) <= 1e-12):
        raise ValueError("feature columns must be finite and nonconstant")
    if not isinstance(max_clusters, (int, np.integer)) or not 2 <= int(max_clusters) <= min(4, features.shape[0] - 1):
        raise ValueError("max_clusters must be an integer from 2 through min(4,n-1)")
    if not np.isscalar(regularization) or not np.isfinite(regularization) or float(regularization) <= 0.0:
        raise ValueError("regularization must be a positive finite scalar")
    bics = []
    fits = []
    for clusters in range(1, int(max_clusters) + 1):
        fit = _fit_full_gmm(features, clusters, float(regularization))
        bics.append(fit[0])
        fits.append(fit)
    bics_array = np.asarray(bics, dtype=float)
    best_index = int(np.argmin(bics_array))
    best_clusters = best_index + 1
    responsibilities = fits[best_index][2]
    means = fits[best_index][3]
    order = np.lexsort(means[:, ::-1].T)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(best_clusters)
    labels = inverse[np.argmax(responsibilities, axis=1)]
    clipped = np.clip(responsibilities, 1e-300, 1.0)
    entropy = float(-np.mean(np.sum(clipped * np.log(clipped), axis=1)))
    sorted_bics = np.sort(bics_array)
    bic_gap = float(sorted_bics[1] - sorted_bics[0]) if sorted_bics.size > 1 else 0.0
    return int(best_clusters), labels.astype(float), bics_array, entropy, bic_gap

import math

import numpy as np

def _fixture(seed: int, molecule_count: int, direction_count: int) -> tuple:
    rng = np.random.default_rng(int(seed))
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    indices = np.arange(direction_count, dtype=float)
    z = 1.0 - 2.0 * (indices + 0.5) / direction_count
    radius = np.sqrt(1.0 - z * z)
    azimuth = 2.0 * np.pi * indices / golden_ratio
    directions = np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    weights = np.full(direction_count, 4.0 * np.pi / direction_count, dtype=float)
    reference = rng.normal(0.0, 0.42, size=(molecule_count, 3, 3, 3))
    reference = 0.5 * (reference + np.swapaxes(reference, 2, 3))
    modes = rng.normal(0.0, 1.0, size=(molecule_count, 3, 3, 3))
    modes = 0.5 * (modes + np.swapaxes(modes, 2, 3))
    for molecule in range(molecule_count):
        reference[molecule, 0, 0, 0] += 1.05 + 0.06 * molecule
        reference[molecule, 1, 1, 1] -= 0.72 - 0.025 * molecule
        reference[molecule, 2, 2, 2] += 0.48 + 0.035 * (molecule % 5)
        modes[molecule] /= np.linalg.norm(modes[molecule])
    return directions, weights, reference, modes


def compute_directional_basis_audit(
    seed: int = 82026,
    molecule_count: int = 68,
    direction_count: int = 86,
    omega_max: float = 0.36,
    max_clusters: int = 4,
) -> tuple:
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if not isinstance(molecule_count, (int, np.integer)) or int(molecule_count) < 8:
        raise ValueError("molecule_count must be an integer at least 8")
    if not isinstance(direction_count, (int, np.integer)) or int(direction_count) < 24:
        raise ValueError("direction_count must be an integer at least 24")
    frequencies = build_shg_frequency_grid(omega_max)
    directions, weights, reference, modes = _fixture(seed, int(molecule_count), int(direction_count))
    basis_count = 6
    relative = np.empty((molecule_count, basis_count, frequencies.size), dtype=float)
    positive = np.empty_like(relative)
    negative = np.empty_like(relative)
    anisotropy = np.empty_like(relative)
    groups = np.arange(molecule_count) % 4
    base_bias = np.array([0.24, 0.075, 0.12, 0.026, 0.052, 0.006], dtype=float)
    anis_scale = np.array([0.19, 0.105, 0.12, 0.065, 0.07, 0.028], dtype=float)
    group_sign = np.array(
        [
            [1.0, -0.85, 1.0, 0.15],
            [0.75, -0.65, -0.45, 0.05],
            [0.85, -0.72, 0.8, 0.10],
            [0.55, -0.52, -0.35, 0.02],
            [0.68, -0.60, 0.55, 0.08],
            [0.34, -0.28, -0.22, 0.01],
        ],
        dtype=float,
    )
    group_anisotropy = np.array([0.38, 0.42, 0.72, 2.25], dtype=float)
    frequency_scale = 1.0 + 0.24 * frequencies / float(omega_max) + 0.11 * (frequencies / float(omega_max)) ** 2
    for molecule in range(int(molecule_count)):
        for basis in range(basis_count):
            for frequency_index, scale in enumerate(frequency_scale):
                beta_mra = reference[molecule] * scale + 0.035 * frequency_index * modes[molecule]
                bias = base_bias[basis] * group_sign[basis, groups[molecule]]
                modulation = 1.0 + 0.09 * math.sin((molecule + 1) * (frequency_index + 1))
                beta_basis = (1.0 + bias * modulation) * beta_mra + anis_scale[basis] * group_anisotropy[
                    groups[molecule]
                ] * (0.58 + 0.07 * frequency_index) * modes[molecule]
                relative[molecule, basis, frequency_index] = compute_relative_rms_total(
                    beta_basis, beta_mra, directions, weights
                )[0]
                signed = compute_signed_projection_metrics(beta_basis, beta_mra, directions, weights)
                positive[molecule, basis, frequency_index] = signed[0]
                negative[molecule, basis, frequency_index] = signed[1]
                anisotropy[molecule, basis, frequency_index] = compute_component_rms_anisotropy(
                    beta_basis, beta_mra, directions, weights
                )[1]
    features = np.empty((molecule_count, 2 * basis_count), dtype=float)
    features[:, 0::2] = np.mean(positive, axis=2)
    features[:, 1::2] = np.mean(negative, axis=2)
    best_clusters, labels, bics, entropy, bic_gap = select_convergence_clusters(
        features, max_clusters=max_clusters, regularization=1e-6
    )
    mean_relative = float(np.mean(relative))
    mean_positive = float(np.mean(positive))
    mean_negative = float(np.mean(negative))
    mean_anisotropy = float(np.mean(anisotropy))
    central_relative = float(relative[molecule_count // 2, 3, 2])
    coefficients = np.arange(1, features.size + 1, dtype=float).reshape(features.shape)
    feature_checksum = float(np.sum(coefficients * features))
    frequency_drift = float(np.mean(relative[:, :, -1]) - np.mean(relative[:, :, 0]))
    j_value = (
        mean_relative
        + 0.35 * (mean_positive + mean_negative)
        + 0.025 * mean_anisotropy
        + 0.02 * entropy
        + 0.0005 * bic_gap
        + 0.1 * abs(frequency_drift)
        + 0.005 * best_clusters
    )
    return (
        float(j_value),
        mean_relative,
        mean_positive,
        mean_negative,
        mean_anisotropy,
        central_relative,
        feature_checksum,
        int(best_clusters),
        float(entropy),
        float(bic_gap),
        frequency_drift,
        frequencies,
        relative,
        features,
        labels,
        bics,
    )
SCICODE_GOLD_EOF
