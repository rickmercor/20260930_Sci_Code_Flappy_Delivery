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
from scipy.special import logsumexp, ndtr

def _bg11_array(value, ndim):
    a = np.asarray(value, dtype=float)
    if a.ndim != ndim or not np.all(np.isfinite(a)) or any(s == 0 for s in a.shape):
        raise ValueError('invalid array dimensions or nonfinite entries')
    return a

def _bg11_cov(value, size=None, pd=True):
    a = _bg11_array(value, 2)
    if a.shape[0] != a.shape[1] or (size is not None and a.shape != (size, size)):
        raise ValueError('invalid covariance shape')
    if not np.allclose(a, a.T, rtol=0, atol=1e-10):
        raise ValueError('covariance is not symmetric')
    eigen = np.linalg.eigvalsh(a)
    if (pd and eigen[0] <= 0) or (not pd and eigen[0] < -1e-10):
        raise ValueError('invalid covariance definiteness')
    return a

def _bg11_prob(value, size):
    a = _bg11_array(value, 1)
    if a.shape != (size,) or np.any(a <= 0) or not np.isclose(a.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError('expected a strictly positive probability vector')
    return a / a.sum()

def _bg11_prior(prior, r):
    a = _bg11_array(prior, 2)
    if a.shape[1] != 1+r*r:
        raise ValueError('invalid packed prior')
    w = _bg11_prob(a[:, 0], len(a))
    u = a[:, 1:].reshape(-1, r, r)
    for h in u:
        _bg11_cov(h, r, pd=False)
    return w, u

def _bg11_integer(value, lower=1):
    if not np.isscalar(value) or not np.isfinite(value) or value != int(value) or value < lower:
        raise ValueError('invalid integer')
    return int(value)

def _bg11_state(state):
    s = _bg11_array(state, 3)
    l, rows, width = s.shape
    r = (width-3)//2
    j = rows-1
    if r < 1 or width != 2*r+3 or j < 1:
        raise ValueError('invalid posterior state dimensions')
    a = s[:, :j, 0]
    f = s[:, :j, r+1:2*r+1]
    if np.any(a < 0) or not np.allclose(a.sum(axis=1), 1, rtol=0, atol=1e-8):
        raise ValueError('invalid posterior probabilities')
    if np.any(f < -1e-12) or np.any(f > 1+1e-12) or np.any(s[:, j, 0] < 0):
        raise ValueError('invalid sign probabilities or scales')
    return s, l, j, r

def estimate_residual_correlation(weak_z, cutoff):
    z = _bg11_array(weak_z, 2)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('cutoff must be positive')
    retained = z[np.max(np.abs(z), axis=1) < cutoff]
    if not len(retained):
        raise ValueError('empty weak association set')
    moment = retained.T @ retained / len(retained)
    if np.any(np.diag(moment) <= 0):
        raise ValueError('zero marginal second moment')
    v = moment / np.sqrt(np.outer(np.diag(moment), np.diag(moment)))
    return _bg11_cov(v, z.shape[1])

import numpy as np
from scipy.special import logsumexp

def update_covariance_mixture(training_z, residual_covariance, prior):
    x = _bg11_array(training_z, 2)
    n, r = x.shape
    v = _bg11_cov(residual_covariance, r)
    w, u = _bg11_prior(prior, r)
    k = len(w)
    log_q = np.empty((n, k))
    moments = np.empty((n, k, r, r))
    for h in range(k):
        total = u[h] + v
        solved = np.linalg.solve(total, x.T).T
        log_q[:, h] = np.log(w[h]) - 0.5*(np.linalg.slogdet(total)[1] + np.sum(x*solved, axis=1))
        mean = solved @ u[h]
        covariance = u[h] - u[h] @ np.linalg.solve(total, u[h])
        moments[:, h] = covariance + np.einsum('ni,nj->nij', mean, mean)
    q = np.exp(log_q - logsumexp(log_q, axis=1)[:, None])
    counts = q.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError('mixture component has zero numerical mass')
    updated = np.einsum('nk,nkij->kij', q, moments) / counts[:, None, None]
    updated = (updated + updated.transpose(0, 2, 1))/2
    return np.column_stack((counts/n, updated.reshape(k, r*r)))

import numpy as np
from scipy.special import logsumexp, ndtr

def learn_covariance_prior(training_z, residual_covariance, initial_prior, iterations, ridge):
    x = _bg11_array(training_z, 2)
    r = x.shape[1]
    _bg11_cov(residual_covariance, r)
    _bg11_prior(initial_prior, r)
    count = _bg11_integer(iterations, 0)
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError('ridge must be nonnegative')
    result = np.array(initial_prior, dtype=float, copy=True)
    for _ in range(count):
        result = update_covariance_mixture(x, residual_covariance, result)
    u = result[:, 1:].reshape(-1, r, r).copy()
    u += ridge*np.eye(r)
    result[:, 1:] = u.reshape(len(u), r*r)
    return result

import numpy as np
from scipy.special import logsumexp, ndtr

def fit_single_effect(information, residual_crossproduct, residual_covariance, prior, variant_prior, scale):
    d = _bg11_array(information, 1)
    xy = _bg11_array(residual_crossproduct, 2)
    j, r = xy.shape
    if d.shape != (j,) or np.any(d <= 0) or not np.isfinite(scale) or scale < 0:
        raise ValueError('invalid information or scale')
    v = _bg11_cov(residual_covariance, r)
    w, u = _bg11_prior(prior, r)
    pi = _bg11_prob(variant_prior, j)
    b = xy/d[:, None]
    k = len(w)
    log_bf = np.empty((j, k))
    means = np.empty((j, k, r))
    positive = np.zeros((j, k, r))
    negative = np.zeros_like(positive)
    for a in range(j):
        noise = v/d[a]
        for h in range(k):
            covariance = scale*u[h]
            total = noise+covariance
            solved = np.linalg.solve(total, b[a])
            mean = covariance @ solved
            posterior_covariance = covariance - covariance @ np.linalg.solve(total, covariance)
            log_bf[a, h] = 0.5*(np.linalg.slogdet(noise)[1] - np.linalg.slogdet(total)[1]
                                  + b[a] @ (np.linalg.solve(noise, b[a])-solved))
            means[a, h] = mean
            # A zero diagonal of a PSD prior is an exact zero-valued trait.
            supported = np.diag(covariance) > 0
            variance = np.diag(posterior_covariance)
            if np.any(variance[supported] <= 0):
                raise ValueError('posterior variance lost positive precision')
            z = mean[supported]/np.sqrt(variance[supported])
            positive[a, h, supported] = ndtr(z)
            negative[a, h, supported] = ndtr(-z)
    local_bf = logsumexp(log_bf+np.log(w), axis=1)
    log_evidence = float(logsumexp(local_bf+np.log(pi)))
    alpha = np.exp(local_bf+np.log(pi)-log_evidence)
    omega = np.exp(log_bf+np.log(w)-local_bf[:, None])
    out = np.zeros((j+1, 2*r+3))
    out[:j, 0] = alpha
    out[:j, 1:r+1] = alpha[:, None]*np.einsum('jk,jkr->jr', omega, means)
    out[:j, r+1:2*r+1] = 1-np.maximum(np.einsum('jk,jkr->jr', omega, positive), np.einsum('jk,jkr->jr', omega, negative))
    out[:j, 2*r+1] = local_bf
    out[j, :2] = [scale, log_evidence]
    return out

import numpy as np
from scipy.special import logsumexp, ndtr

def select_effect_scale(information, residual_crossproduct, residual_covariance, prior, variant_prior, scale_grid, initial_candidate=None):
    grid = _bg11_array(scale_grid, 1)
    if grid[0] < 0 or np.any(np.diff(grid) <= 0):
        raise ValueError('scale grid must be nonnegative and strictly increasing')
    best = None
    for index, scale in enumerate(grid):
        if index == 0 and initial_candidate is not None:
            candidate = _bg11_array(initial_candidate, 2).copy()
            j, r = np.asarray(residual_crossproduct).shape
            if candidate.shape != (j+1, 2*r+3) or candidate[-1, 0] != scale:
                raise ValueError('invalid initial scale candidate')
        else:
            candidate = fit_single_effect(information, residual_crossproduct, residual_covariance, prior, variant_prior, float(scale))
        if best is None or candidate[-1, 1] > best[-1, 1]:
            best = candidate
    return best

import numpy as np
from scipy.special import logsumexp, ndtr

def fit_additive_effects(genotype_crossproduct, genotype_trait_crossproduct, residual_covariance, prior, variant_prior, scale_grid, effects, tolerance, max_sweeps, first_component_state=None):
    xy = _bg11_array(genotype_trait_crossproduct, 2)
    j, r = xy.shape
    xx = _bg11_cov(genotype_crossproduct, j)
    _bg11_cov(residual_covariance, r)
    _bg11_prior(prior, r)
    pi = _bg11_prob(variant_prior, j)
    l = _bg11_integer(effects)
    cap = _bg11_integer(max_sweeps)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be positive')
    grid = _bg11_array(scale_grid, 1)
    if grid[0] != 0 or np.any(np.diff(grid) <= 0):
        raise ValueError('IBSS scale grid must start at zero and be strictly increasing')
    state = np.zeros((l, j+1, 2*r+3))
    state[:, :j, 0] = pi
    state[:, :j, r+1:2*r+1] = 1
    previous_delta = 0.0
    for sweep in range(1, cap+1):
        old_means = state[:, :j, 1:r+1].copy()
        old_scales = state[:, j, 0].copy()
        for a in range(l):
            means = state[:, :j, 1:r+1]
            residual = xy - xx @ (means.sum(axis=0)-means[a])
            if sweep == 1 and a == 0 and first_component_state is not None:
                supplied = _bg11_array(first_component_state, 2)
                if supplied.shape != state[a].shape:
                    raise ValueError('invalid first component state')
                state[a] = supplied
            else:
                state[a] = select_effect_scale(np.diag(xx), residual, residual_covariance, prior, pi, grid)
        delta = float(np.max(np.abs(state[:, :j, 1:r+1]-old_means)))
        if delta <= tolerance and np.array_equal(state[:, j, 0], old_scales):
            state[:, j, 2:5] = [delta, previous_delta, sweep]
            return state
        previous_delta = delta
    raise ValueError('no convergence within max_sweeps')

import numpy as np
from scipy.special import logsumexp, ndtr

def resolve_credible_sets(state, ld, coverage, min_purity):
    s, l, j, r = _bg11_state(state)
    correlation = _bg11_cov(ld, j)
    if not np.allclose(np.diag(correlation), 1, rtol=0, atol=1e-10):
        raise ValueError('LD must be a correlation matrix')
    if not np.isfinite(coverage) or not 0 < coverage <= 1 or not np.isfinite(min_purity) or not 0 <= min_purity <= 1:
        raise ValueError('invalid credible-set thresholds')
    out = np.zeros((l, j+3))
    seen = set()
    for a in range(l):
        probabilities = s[a, :j, 0]
        order = np.argsort(-probabilities, kind='stable')
        count = j if coverage == 1 else min(j, int(np.searchsorted(np.cumsum(probabilities[order]), coverage, side='left'))+1)
        members = order[:count]
        pure = 1.0 if count == 1 else float(np.min(np.abs(correlation[np.ix_(members, members)][np.triu_indices(count, 1)])))
        key = tuple(sorted(members.tolist()))
        retained = s[a, j, 0] > 0 and pure >= min_purity and key not in seen
        if retained:
            seen.add(key)
        out[a, :3] = [float(retained), probabilities[members].sum(), pure]
        out[a, 3+members] = 1
    return out

import numpy as np
from scipy.special import logsumexp, ndtr

def average_trait_sign_uncertainty(state):
    if np.asarray(state).ndim != 3:
        raise ValueError('expected a stacked posterior state')
    s, l, j, r = _bg11_state(state)
    return np.einsum('lj,ljr->lr', s[:, :j, 0], s[:, :j, r+1:2*r+1])

import numpy as np
from scipy.special import logsumexp, ndtr

def prioritize_variant(state, credible_sets, average_lfsr, trait_index, sign_threshold, priority):
    s, l, j, r = _bg11_state(state)
    cs = _bg11_array(credible_sets, 2)
    averages = _bg11_array(average_lfsr, 2)
    ranking = _bg11_array(priority, 1)
    trait = _bg11_integer(trait_index, 0)
    if cs.shape != (l, j+3) or averages.shape != (l, r) or ranking.shape != (j,) or trait >= r:
        raise ValueError('incompatible prioritization dimensions')
    if not np.isfinite(sign_threshold) or not 0 <= sign_threshold <= 1:
        raise ValueError('invalid sign threshold')
    if np.any((cs[:, 0] != 0) & (cs[:, 0] != 1)) or np.any((cs[:, 3:] != 0) & (cs[:, 3:] != 1)) or np.any(averages < 0) or np.any(averages > 1):
        raise ValueError('invalid membership or average-lfsr state')
    eligible = (cs[:, 0] == 1) & (averages[:, trait] < sign_threshold)
    candidates = np.flatnonzero(np.any(cs[eligible, 3:] == 1, axis=0))
    if not len(candidates):
        return np.array([-1.0, 0.0])
    chosen = int(candidates[np.argmax(ranking[candidates])])
    active = s[:, j, 0] > 0
    pip = float(1-np.prod(1-s[active, chosen, 0]))
    return np.array([float(chosen), pip])

import numpy as np
from scipy.special import logsumexp, ndtr

def compute_prioritized_pip(weak_z, training_z, initial_prior, genotype_crossproduct, genotype_trait_crossproduct, variant_prior, scale_grid, priority, effects, ed_iterations, ridge, weak_cutoff, coverage, min_purity, trait_index, sign_threshold, tolerance, max_sweeps):
    weak = _bg11_array(weak_z, 2)
    training = _bg11_array(training_z, 2)
    if weak.shape[1] != training.shape[1]:
        raise ValueError('training panels must have the same aligned traits')
    count = _bg11_integer(ed_iterations, 0)
    v = estimate_residual_correlation(weak, weak_cutoff)
    # Reuse completed work at the three natural stage boundaries.
    current = initial_prior
    if count:
        current = update_covariance_mixture(training, v, current)
    prior = learn_covariance_prior(training, v, current, max(count-1, 0), ridge)
    d = np.diag(np.asarray(genotype_crossproduct, dtype=float))
    initial = fit_single_effect(d, genotype_trait_crossproduct, v, prior, variant_prior, float(_bg11_array(scale_grid, 1)[0]))
    first = select_effect_scale(d, genotype_trait_crossproduct, v, prior, variant_prior, scale_grid, initial_candidate=initial)
    state = fit_additive_effects(genotype_crossproduct, genotype_trait_crossproduct, v, prior, variant_prior, scale_grid, effects, tolerance, max_sweeps, first_component_state=first)
    xx = np.asarray(genotype_crossproduct, dtype=float)
    ld = xx / np.sqrt(np.outer(np.diag(xx), np.diag(xx)))
    cs = resolve_credible_sets(state, ld, coverage, min_purity)
    average = average_trait_sign_uncertainty(state)
    result = prioritize_variant(state, cs, average, trait_index, sign_threshold, priority)
    return float(result[1])
SCICODE_GOLD_EOF
