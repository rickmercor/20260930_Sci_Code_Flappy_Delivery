#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
from scipy.special import logsumexp, xlogy

def single_effect_posterior(residual_cross: 'np.ndarray | list', diagonal: 'np.ndarray | list', noise: float, prior_variance: float, prior_mean: 'np.ndarray | list') -> 'np.ndarray':
    residual_cross, diagonal, prior_mean = map(np.asarray, (residual_cross, diagonal, prior_mean))
    sampling = noise / diagonal
    variance = sampling * prior_variance / (sampling + prior_variance)
    mean = variance * (residual_cross / noise + prior_mean / prior_variance)
    estimate = residual_cross / diagonal
    logbf = -0.5 * np.log1p(prior_variance / sampling)
    logbf += 0.5 * estimate ** 2 / sampling - 0.5 * (estimate - prior_mean) ** 2 / (sampling + prior_variance)
    inclusion = np.exp(logbf - logsumexp(logbf))
    return np.stack((inclusion, mean, variance))

import numpy as np
from scipy.special import logsumexp, xlogy

def expected_residual_ss(gram: 'np.ndarray | list', cross: 'np.ndarray | list', yty: float, posterior: 'np.ndarray | list') -> float:
    gram, cross, posterior = map(np.asarray, (gram, cross, posterior))
    alpha, mean, variance = (posterior[:, 0], posterior[:, 1], posterior[:, 2])
    component_mean = alpha * mean
    total = component_mean.sum(axis=0)
    second_moment = alpha * (mean ** 2 + variance)
    return float(yty - 2 * total @ cross + total @ gram @ total + np.sum(second_moment * np.diag(gram)) - np.einsum('li,ij,lj->', component_mean, gram, component_mean))

import numpy as np
from scipy.special import logsumexp, xlogy

def variational_objective(gram: 'np.ndarray | list', cross: 'np.ndarray | list', yty: float, n: int, noise: float, prior_variance: float, prior_mean: 'np.ndarray | list', posterior: 'np.ndarray | list') -> float:
    posterior, prior_mean = map(np.asarray, (posterior, prior_mean))
    alpha, mean, variance = (posterior[:, 0], posterior[:, 1], posterior[:, 2])
    normal_kl = 0.5 * ((variance + (mean - prior_mean) ** 2) / prior_variance - 1 + np.log(prior_variance / variance))
    kl = np.sum(xlogy(alpha, alpha * posterior.shape[-1]) + alpha * normal_kl)
    erss = expected_residual_ss(gram, cross, yty, posterior)
    return float(-0.5 * n * np.log(2 * np.pi * noise) - 0.5 * erss / noise - kl)

import numpy as np
from scipy.special import logsumexp, xlogy

def fit_single_setting(gram: 'np.ndarray | list', cross: 'np.ndarray | list', yty: float, n: int, annotation: 'np.ndarray | list', scale: float, prior_variance: float, effects: int=2, noise: float=1.0, sweeps: int=300) -> 'tuple[np.ndarray, float, np.ndarray]':
    gram, cross, annotation = map(np.asarray, (gram, cross, annotation))
    p = len(cross)
    posterior = np.empty((effects, 3, p))
    component_mean = np.zeros((effects, p))
    prior_mean = scale * annotation
    for _ in range(sweeps):
        for ell in range(effects):
            residual_cross = cross - gram @ (component_mean.sum(axis=0) - component_mean[ell])
            posterior[ell] = single_effect_posterior(residual_cross, np.diag(gram), noise, prior_variance, prior_mean)
            component_mean[ell] = posterior[ell, 0] * posterior[ell, 1]
    pip = 1 - np.prod(1 - posterior[:, 0], axis=0)
    elbo = variational_objective(gram, cross, yty, n, noise, prior_variance, prior_mean, posterior)
    return (pip, elbo, posterior)

import numpy as np
from scipy.special import logsumexp, xlogy

def partition_fits(pips: 'np.ndarray | list', cutoff: float=0.05) -> 'np.ndarray':
    pips = np.asarray(pips)
    distances = np.max(np.maximum(pips[:, None], pips[None, :]) * np.abs(pips[:, None] - pips[None, :]), axis=2)
    groups = [(i,) for i in range(len(pips))]
    while len(groups) > 1:
        distance, left, right = min(((float(distances[np.ix_(g, h)].max()), g, h) for i, g in enumerate(groups) for h in groups[i + 1:]))
        if distance > cutoff:
            break
        groups.remove(left)
        groups.remove(right)
        groups.append(tuple(sorted(left + right)))
        groups.sort()
    labels = np.empty(len(pips), dtype=int)
    for label, group in enumerate(groups):
        labels[list(group)] = label
    return labels

import numpy as np
from scipy.special import logsumexp, xlogy

def aggregate_ensemble(pips: 'np.ndarray | list', elbos: 'np.ndarray | list', labels: 'np.ndarray | list') -> 'tuple[np.ndarray, np.ndarray]':
    pips, elbos, labels = map(np.asarray, (pips, elbos, labels))
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    peaks = np.array([elbos[group].max() for group in groups])
    masses = np.exp(peaks - logsumexp(peaks))
    weights = np.zeros(len(pips))
    for group, mass in zip(groups, masses):
        weights[group] = mass * np.exp(elbos[group] - logsumexp(elbos[group]))
    return (weights @ pips, weights)

import numpy as np
from scipy.special import logsumexp, xlogy

def annotation_sensitivity(gram: 'np.ndarray | list', cross: 'np.ndarray | list', yty: float, n: int, annotation: 'np.ndarray | list', variant: int, scales: 'np.ndarray | list', variances: 'np.ndarray | list', effects: int=2, noise: float=1.0, sweeps: int=300, cutoff: float=0.05) -> float:
    values = []
    for flipped in (False, True):
        current = np.array(annotation, dtype=float, copy=True)
        if flipped:
            current[variant] *= -1
        fits = [fit_single_setting(gram, cross, yty, n, current, scale, variance, effects, noise, sweeps) for scale in scales for variance in variances]
        pips = np.array([fit[0] for fit in fits])
        elbos = np.array([fit[1] for fit in fits])
        labels = partition_fits(pips, cutoff)
        aggregate, _ = aggregate_ensemble(pips, elbos, labels)
        values.append(aggregate[variant])
    return float(values[1] - values[0])
SCICODE_GOLD_EOF
