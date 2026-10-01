#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

"""Step 1: draw one entry state, physical realization, and nuisance point."""
import numpy as np
_MEAN = np.array([0.0, 0.0, 100000.0, 12400.0, 180.0, -5250.0])
_COVARIANCE = np.array([[420.0 ** 2, 0.35 * 420.0 * 260.0, 0.0, 0.0, 0.0, 0.0], [0.35 * 420.0 * 260.0, 260.0 ** 2, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 95.0 ** 2, 0.0, 0.0, -0.25 * 95.0 * 18.0], [0.0, 0.0, 0.0, 26.0 ** 2, 0.2 * 26.0 * 11.0, 0.0], [0.0, 0.0, 0.0, 0.2 * 26.0 * 11.0, 11.0 ** 2, 0.0], [0.0, 0.0, -0.25 * 95.0 * 18.0, 0.0, 0.0, 18.0 ** 2]])

def sample_entry(rng=None):
    if rng is None:
        rng = np.random.Generator(np.random.PCG64(761903))
    if not hasattr(rng, 'standard_normal') or not hasattr(rng, 'uniform'):
        raise ValueError('rng must provide standard_normal and uniform')
    state6 = _MEAN + np.linalg.cholesky(_COVARIANCE) @ rng.standard_normal(6)
    density_scale = rng.uniform(0.92, 1.08)
    state7 = np.concatenate((state6, [185000.0 * density_scale]))
    strength = rng.uniform(420000.0, 720000.0)
    dust = rng.uniform(0.18, 0.34)
    largest = rng.uniform(0.25, 0.38)
    beta = rng.uniform(0.48, 0.66)
    wind_scale = rng.uniform(0.88, 1.12)
    cd_scale = rng.uniform(0.9, 1.1)
    sigma_abl = rng.uniform(7.2e-08, 8.8e-08)
    nuisance = np.array([(strength - 420000.0) / 300000.0, (dust - 0.18) / 0.16, (beta - 0.48) / 0.18, (wind_scale - 0.88) / 0.24])
    physical = np.array([strength, dust, largest, beta, wind_scale, cd_scale, sigma_abl])
    return np.concatenate((state7, physical, nuisance))

"""Step 2: evaluate local atmosphere, loading, drag, gravity, and ablation."""
import math
import numpy as np
_G0 = 9.80665
_R_EARTH = 6371000.0
_RHO_AIR_0 = 1.225
_H_AIR = 7200.0
_RHO_BODY = 3050.0
_OMEGA = 7.2921159e-05
_MACH_NODES = np.array([0.0, 0.5, 0.8, 1.0, 1.2, 2.0, 3.0, 4.0])
_CD_NODES = np.array([0.47, 0.48, 0.55, 0.92, 1.08, 0.91, 0.72, 0.64])

def atmospheric_rhs(state, wind_scale=1.0, cd_scale=1.0, sigma_abl=8e-08):
    state = np.asarray(state, dtype=float)
    if state.shape != (7,) or not np.all(np.isfinite(state)) or state[6] <= 0.0:
        raise ValueError('state must be a finite seven-vector with positive mass')
    if not np.all(np.isfinite([wind_scale, cd_scale, sigma_abl])):
        raise ValueError('physical scales must be finite')
    if wind_scale < 0.0 or cd_scale <= 0.0 or sigma_abl < 0.0:
        raise ValueError('physical scales are outside their domains')
    position = state[:3]
    velocity = state[3:6]
    altitude = max(float(position[2]), 0.0)
    rho = _RHO_AIR_0 * math.exp(-altitude / _H_AIR)
    temperature = 216.65 + 71.0 * math.exp(-((altitude - 9000.0) / 8500.0) ** 2)
    sound = math.sqrt(1.4 * 287.0 * temperature)
    wind = wind_scale * np.array([8.0 + 17.0 * math.exp(-((altitude - 12000.0) / 7000.0) ** 2), -3.0 + 24.0 * math.exp(-((altitude - 10000.0) / 6500.0) ** 2) - 8.0 * math.exp(-((altitude - 31000.0) / 9000.0) ** 2), 0.0])
    relative = velocity - wind
    speed = float(np.linalg.norm(relative))
    mach = speed / sound
    cd = cd_scale * float(np.interp(min(mach, 4.0), _MACH_NODES, _CD_NODES))
    radius = (3.0 * state[6] / (4.0 * math.pi * _RHO_BODY)) ** (1.0 / 3.0)
    area = math.pi * radius * radius
    drag = -0.5 * rho * cd * area / state[6] * speed * relative
    gravity = np.array([0.0, 0.0, -_G0 * (_R_EARTH / (_R_EARTH + altitude)) ** 2])
    coriolis = np.array([2.0 * _OMEGA * velocity[1], -2.0 * _OMEGA * velocity[0], 0.0])
    mass_rate = -0.5 * sigma_abl * rho * area * speed ** 3 if speed >= 3000.0 else 0.0
    derivative = np.concatenate((velocity, drag + gravity + coriolis, [mass_rate]))
    ram_pressure = 0.5 * rho * speed * speed
    return np.concatenate((derivative, [rho, sound], wind, [speed, ram_pressure, cd, area]))

"""Step 3: propagate one fragment to impact, depletion, or breakup."""
import numpy as np
from scipy.integrate import solve_ivp

def propagate_fragment(fragment, wind_scale, cd_scale, sigma_abl, rtol=2e-10, max_step=0.35, max_depth=2):
    fragment = np.asarray(fragment, dtype=float)
    if fragment.shape != (9,) or not np.all(np.isfinite(fragment)):
        raise ValueError('fragment must contain state, strength, and depth')
    if fragment[6] <= 0.0 or fragment[7] <= 0.0 or fragment[8] < 0.0 or (fragment[8] != np.floor(fragment[8])):
        raise ValueError('fragment mass, strength, and depth are invalid')
    if not np.all(np.isfinite([rtol, max_step, max_depth])):
        raise ValueError('integration controls must be finite')
    if rtol <= 0.0 or max_step <= 0.0 or max_depth < 0 or (int(max_depth) != max_depth) or isinstance(max_depth, (bool, np.bool_)):
        raise ValueError('integration controls are invalid')
    state0 = fragment[:7]
    strength = float(fragment[7])
    depth = int(round(fragment[8]))

    def rhs(_, state):
        return atmospheric_rhs(state, wind_scale, cd_scale, sigma_abl)[:7]

    def ground(_, state):
        return state[2]

    def depleted(_, state):
        return state[6] - 0.001
    ground.terminal = True
    ground.direction = -1
    depleted.terminal = True
    depleted.direction = -1
    events = [ground, depleted]
    if depth < int(max_depth):

        def breakup(_, state):
            return atmospheric_rhs(state, wind_scale, cd_scale, sigma_abl)[13] - strength
        breakup.terminal = True
        breakup.direction = 1
        events.append(breakup)
    solution = solve_ivp(rhs, (0.0, 5000.0), state0, method='RK45', rtol=rtol, atol=np.array([1e-05, 1e-05, 1e-05, 1e-06, 1e-06, 1e-06, 1e-12]), max_step=max_step, events=events)
    if not solution.success:
        raise RuntimeError(solution.message)
    for event_index, event_states in enumerate(solution.y_events):
        if len(event_states):
            terminal = event_states[0].copy()
            if event_index == 0:
                terminal[2] = 0.0
                return np.concatenate(([0.0], terminal))
            if event_index == 1:
                return np.concatenate(([1.0], terminal))
            return np.concatenate(([2.0], terminal))
    raise RuntimeError('trajectory terminated without a terminal event')

"""Step 4: partition a disrupted parent and assign transverse kicks."""
import math
import numpy as np
_RHO_BODY = 3050.0

def partition_fragment_cloud(fragment, dust_fraction, largest_fraction, beta, wind_scale, rng=None, alpha=0.16, strength_cap=15000000.0, minimum_mass=0.2, kick_power=2.0, momentum_correct=True):
    fragment = np.asarray(fragment, dtype=float)
    if fragment.shape != (9,) or not np.all(np.isfinite(fragment)):
        raise ValueError('fragment must contain state, strength, and depth')
    if not np.all(np.isfinite([dust_fraction, largest_fraction, beta, wind_scale, alpha, strength_cap, minimum_mass, kick_power])):
        raise ValueError('fragment controls must be finite')
    if not (0.0 <= dust_fraction < 1.0 and 0.0 < largest_fraction < 1.0 and (0.0 < beta < 1.0)):
        raise ValueError('fragment-distribution parameters are invalid')
    if fragment[6] <= 0.0 or fragment[7] <= 0.0 or minimum_mass <= 0.0:
        raise ValueError('masses and strength must be positive')
    if fragment[8] < 0 or fragment[8] != np.floor(fragment[8]) or np.linalg.norm(fragment[3:6]) == 0.0:
        raise ValueError('depth must be a nonnegative integer and velocity must be nonzero')
    if wind_scale < 0 or alpha < 0 or strength_cap <= 0 or (kick_power < 0):
        raise ValueError('wind, alpha, and kick power must be nonnegative and strength cap positive')
    if not isinstance(momentum_correct, (bool, np.bool_)):
        raise ValueError('momentum_correct must be boolean')
    if rng is None:
        rng = np.random.Generator(np.random.PCG64(741))
    if not hasattr(rng, 'uniform'):
        raise ValueError('rng must provide uniform')
    available = fragment[6] * (1.0 - dust_fraction)
    largest = largest_fraction * available
    masses = [largest]
    remaining = available - largest
    rank = 2
    while remaining >= minimum_mass and rank < 8:
        normalized = (largest_fraction ** (-beta) + beta / (1.0 - beta) * largest_fraction ** (1.0 - beta) * (rank - 1)) ** (-1.0 / beta)
        proposed = normalized * available
        if proposed < minimum_mass:
            break
        value = min(proposed, remaining)
        masses.append(value)
        remaining -= value
        if remaining < minimum_mass:
            break
        rank += 1
    if remaining >= minimum_mass:
        masses.append(remaining)
    masses = np.asarray(masses, dtype=float)
    parent_radius = (3.0 * fragment[6] / (4.0 * math.pi * _RHO_BODY)) ** (1.0 / 3.0)
    diagnostics = atmospheric_rhs(fragment[:7], wind_scale, 1.0, 0.0)
    rho = diagnostics[7]
    relative_speed = diagnostics[12]
    velocity = fragment[3:6]
    unit = velocity / np.linalg.norm(velocity)
    seed_axis = np.array([0.0, 0.0, 1.0]) if abs(unit[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    basis1 = np.cross(unit, seed_axis)
    basis1 /= np.linalg.norm(basis1)
    basis2 = np.cross(unit, basis1)
    angles = rng.uniform(0.0, 2.0 * math.pi, len(masses))
    kicks = []
    for mass, angle in zip(masses, angles):
        radius = (3.0 * mass / (4.0 * math.pi * _RHO_BODY)) ** (1.0 / 3.0)
        magnitude = math.sqrt(3.0 * rho / (2.0 * _RHO_BODY)) * (parent_radius / radius) ** (0.5 * kick_power) * relative_speed
        kicks.append(magnitude * (math.cos(angle) * basis1 + math.sin(angle) * basis2))
    kicks = np.asarray(kicks)
    if momentum_correct:
        kicks -= np.sum(masses[:, None] * kicks, axis=0) / np.sum(masses)
    children = []
    for mass, kick in zip(masses, kicks):
        child = fragment.copy()
        child[3:6] += kick
        child[6] = mass
        child[7] = min(fragment[7] * (fragment[6] / mass) ** alpha, strength_cap)
        child[8] = fragment[8] + 1.0
        children.append(child)
    return np.asarray(children)

"""Step 5: propagate one queue-based cascading-fragment realization."""
import numpy as np

def simulate_realization(rng=None, kick_power=2.0, momentum_correct=True, cascade=True, rtol=2e-10, max_step=0.35, max_depth=2):
    if not np.all(np.isfinite([kick_power, rtol, max_step, max_depth])):
        raise ValueError('simulation controls must be finite')
    if kick_power < 0 or rtol <= 0 or max_step <= 0 or (max_depth < 0) or (int(max_depth) != max_depth) or isinstance(max_depth, (bool, np.bool_)):
        raise ValueError('invalid simulation control domain')
    if not isinstance(momentum_correct, (bool, np.bool_)) or not isinstance(cascade, (bool, np.bool_)):
        raise ValueError('momentum_correct and cascade must be boolean')
    if rng is None:
        rng = np.random.Generator(np.random.PCG64(761903))
    if not hasattr(rng, 'standard_normal') or not hasattr(rng, 'uniform'):
        raise ValueError('rng must provide standard_normal and uniform')
    entry = sample_entry(rng)
    strength, dust, largest, beta, wind_scale, cd_scale, sigma_abl = entry[7:14]
    nuisance = entry[14:18]
    queue = [np.concatenate((entry[:7], [strength, 0.0]))]
    impacts = []
    breakup_count = 0
    while queue:
        current = queue.pop(0)
        terminal = propagate_fragment(current, wind_scale, cd_scale, sigma_abl, rtol, max_step, max_depth)
        event_code = int(round(terminal[0]))
        if event_code == 2:
            breakup_count += 1
            parent_depth = current[8] if cascade else float(max_depth)
            parent = np.concatenate((terminal[1:8], [current[7], parent_depth]))
            children = partition_fragment_cloud(parent, dust, largest, beta, wind_scale, rng, 0.16, 15000000.0, 0.2, kick_power, momentum_correct)
            queue.extend(children)
        elif event_code == 0 and terminal[7] >= 0.01:
            impacts.append([terminal[1], terminal[2], terminal[7]])
    if not impacts:
        raise RuntimeError('realization produced no surviving impacts')
    impacts = np.asarray(impacts, dtype=float)
    repeated_nuisance = np.repeat(nuisance[None, :], len(impacts), axis=0)
    repeated_breakups = np.full((len(impacts), 1), float(breakup_count))
    return np.hstack((impacts, repeated_nuisance, repeated_breakups))

def _reported_impacts(value):
    """Apply only the disclosed measurement-resolution comparison map."""
    result = np.asarray(value, dtype=float).copy()
    if result.ndim == 2 and result.shape[1] >= 3:
        result[:, :2] = np.round(result[:, :2], 1)
        result[:, 2] = np.round(result[:, 2], 6)
    return result

"""Step 6: assemble the proposal cloud and retain realization metadata."""
import numpy as np

def build_impact_cloud(seed=761903, runs=8, kick_power=2.0, momentum_correct=True, cascade=True, rtol=2e-10, max_step=0.35, max_depth=2):
    if not np.all(np.isfinite([seed, runs])):
        raise ValueError('seed and runs must be finite integers')
    if isinstance(seed, (bool, np.bool_)) or isinstance(runs, (bool, np.bool_)) or int(seed) != seed or (int(runs) != runs) or (seed < 0):
        raise ValueError('seed must be a nonnegative integer and runs an integer')
    seed = int(seed)
    runs = int(runs)
    if runs < 1 or runs > 1000:
        raise ValueError('runs must be between one and one thousand')
    if not np.isfinite(seed):
        raise ValueError('seed must be finite')
    rng = np.random.Generator(np.random.PCG64(seed))
    rows = []
    for realization in range(runs):
        impacts = simulate_realization(rng, kick_power, momentum_correct, cascade, rtol, max_step, max_depth)
        labels = np.full((len(impacts), 1), float(realization))
        rows.append(np.hstack((impacts[:, :3], labels, impacts[:, 3:])))
    return np.vstack(rows)

"""Step 7: importance-reweight the proposal cloud for three populations."""
import math
import numpy as np
from scipy.special import betaln, logsumexp
_BETA_SHAPES = (((2.0, 5.0), (5.0, 2.0), (2.0, 2.0), (2.0, 5.0)), ((2.0, 2.0), (2.0, 2.0), (2.0, 2.0), (2.0, 2.0)), ((5.0, 2.0), (2.0, 5.0), (5.0, 2.0), (5.0, 2.0)))

def compute_population_weights(impact_cloud, balance_realizations=False):
    if not isinstance(balance_realizations, (bool, np.bool_)):
        raise ValueError('balance_realizations must be boolean')
    cloud = np.asarray(impact_cloud, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] < 8 or len(cloud) < 1:
        raise ValueError('impact_cloud must have at least eight columns and one row')
    if not np.all(np.isfinite(cloud)):
        raise ValueError('impact_cloud must be finite')
    nuisance = cloud[:, 4:8]
    if np.any(nuisance <= 0.0) or np.any(nuisance >= 1.0):
        raise ValueError('nuisance coordinates must lie strictly inside (0,1)')
    labels = cloud[:, 3]
    rounded = np.rint(labels).astype(int)
    if np.any(labels != rounded) or np.any(rounded < 0):
        raise ValueError('realization labels must be nonnegative integers')
    unique, inverse, counts = np.unique(rounded, return_inverse=True, return_counts=True)
    del unique
    rows = []
    for shapes in _BETA_SHAPES:
        log_weight = np.zeros(len(cloud), dtype=float)
        for coordinate, (a, b) in enumerate(shapes):
            value = nuisance[:, coordinate]
            log_weight += (a - 1.0) * np.log(value) + (b - 1.0) * np.log1p(-value) - betaln(a, b)
        if bool(balance_realizations):
            log_weight -= np.log(counts[inverse])
        rows.append(np.exp(log_weight - logsumexp(log_weight)))
    result = np.asarray(rows)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise RuntimeError('importance weights are not finite and positive')
    return result

"""Step 8: fit a weighted three-dimensional mass-location Gaussian mixture."""
import math
import numpy as np
from scipy.special import logsumexp
_FEATURE_CENTER = np.array([198000.0, 5000.0, math.log(0.05)])
_FEATURE_SCALE = np.array([4000.0, 2000.0, 1.0])

def fit_ground_gmm(impact_cloud, sample_weights, components=4, iterations=120, regularization=(0.0064, 0.01, 0.0049)):
    cloud = np.asarray(impact_cloud, dtype=float)
    weights_in = np.asarray(sample_weights, dtype=float)
    diagonal = np.asarray(regularization, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] < 3 or len(cloud) < 1:
        raise ValueError('impact_cloud must contain east, north, and mass')
    if not np.all(np.isfinite(cloud)) or np.any(cloud[:, 2] <= 0.0):
        raise ValueError('impact rows must be finite with positive mass')
    if weights_in.shape != (len(cloud),) or not np.all(np.isfinite(weights_in)):
        raise ValueError('sample_weights must be one finite value per impact')
    if np.any(weights_in < 0.0) or np.sum(weights_in) <= 0.0:
        raise ValueError('sample_weights must be nonnegative with positive sum')
    if not np.all(np.isfinite([components, iterations])):
        raise ValueError('GMM controls must be finite')
    if isinstance(components, (bool, np.bool_)) or int(components) != components:
        raise ValueError('components must be an integer')
    if isinstance(iterations, (bool, np.bool_)) or int(iterations) != iterations:
        raise ValueError('iterations must be an integer')
    components = int(components)
    iterations = int(iterations)
    if components < 1 or len(cloud) < components or iterations < 1:
        raise ValueError('GMM controls are invalid')
    if diagonal.shape != (3,) or not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise ValueError('regularization must be three positive diagonal entries')
    physical = np.column_stack((cloud[:, 0], cloud[:, 1], np.log(cloud[:, 2])))
    points = (physical - _FEATURE_CENTER) / _FEATURE_SCALE
    weights_in = weights_in / np.sum(weights_in)
    score = 0.72 * points[:, 0] - 0.28 * points[:, 1] + 0.35 * points[:, 2]
    groups = np.array_split(np.argsort(score, kind='mergesort'), components)
    means = np.empty((components, 3), dtype=float)
    for index, group in enumerate(groups):
        local_weight = weights_in[group]
        if np.sum(local_weight) <= 0.0:
            raise ValueError('every deterministic initialization block needs positive weight')
        means[index] = local_weight @ points[group] / np.sum(local_weight)
    global_mean = weights_in @ points
    centered = points - global_mean
    common_covariance = centered.T * weights_in @ centered
    regularizer = np.diag(diagonal)
    covariances = np.repeat((common_covariance + regularizer)[None, :, :], components, axis=0)
    mixture_weights = np.full(components, 1.0 / components)

    def log_density(values, mean, covariance):
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0.0:
            raise RuntimeError('non-positive GMM covariance')
        delta = values - mean
        quadratic = np.einsum('ni,ij,nj->n', delta, np.linalg.inv(covariance), delta)
        return -0.5 * (3.0 * math.log(2.0 * math.pi) + logdet + quadratic)
    for _ in range(iterations):
        log_components = np.column_stack([math.log(mixture_weights[j]) + log_density(points, means[j], covariances[j]) for j in range(components)])
        responsibilities = np.exp(log_components - logsumexp(log_components, axis=1)[:, None])
        effective = weights_in[:, None] * responsibilities
        counts = np.sum(effective, axis=0)
        if np.any(counts <= 0.0):
            raise RuntimeError('empty GMM component')
        mixture_weights = counts / np.sum(counts)
        means = effective.T @ points / counts[:, None]
        for j in range(components):
            delta = points - means[j]
            covariances[j] = delta.T * effective[:, j] @ delta / counts[j] + regularizer
    if np.any(np.linalg.eigvalsh(covariances) <= 0.0):
        raise RuntimeError('fitted covariance is not positive definite')
    return np.concatenate((mixture_weights, means.ravel(), covariances.ravel()))

"""Step 9: infer second-campaign recovery from a mixed first catalog."""
import math
import numpy as np
from scipy.special import expit, gammaln, logsumexp, roots_jacobi, betaln

def _campaign_array(value, shape=None):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('numeric array required') from exc
    if not np.all(np.isfinite(a)) or (shape is not None and a.shape != shape):
        raise ValueError('invalid shape or nonfinite array')
    return a

def _campaign_statistics(models, finds, measurement, response, center, axes, angle, bounds, orders):
    gh, nr, na, nm = orders
    hx, hw = np.polynomial.hermite.hermgauss(gh)
    hw = hw / np.sqrt(np.pi)
    xr, wr = np.polynomial.legendre.leggauss(nr)
    r, wr = ((xr + 1) / 2, wr / 2)
    xa, wa = np.polynomial.legendre.leggauss(na)
    phi, wa = (np.pi * (xa + 1), np.pi * wa)
    xm, wm = np.polynomial.legendre.leggauss(nm)
    low, high = np.log(bounds / 0.05)
    zm = (low + high) / 2 + (high - low) * xm / 2
    wm = wm * (high - low) / 2
    rr, pp = np.meshgrid(r, phi, indexing='ij')
    t = np.deg2rad(angle)
    xx = center[0] + axes[0] * rr * np.cos(pp) * np.cos(t) - axes[1] * rr * np.sin(pp) * np.sin(t)
    yy = center[1] + axes[0] * rr * np.cos(pp) * np.sin(t) + axes[1] * rr * np.sin(pp) * np.cos(t)
    grid = np.column_stack(((xx.ravel() - 198000) / 4000, (yy.ravel() - 5000) / 2000))
    wxy = (wr[:, None] * wa[None, :] * axes[0] * axes[1] * rr / (4000 * 2000)).ravel()
    standardized = finds.copy()
    standardized[:, :2] = (finds[:, :2] - [198000, 5000]) / [4000, 2000]
    standardized[:, 2] = np.log(finds[:, 2] / 0.05)

    def first(z):
        return expit((z - np.log(response[0] / 0.05)) / response[1])

    def second(z):
        return expit((z - np.log(response[2] / 0.05)) / response[3])
    output = []
    for row in models:
        c = len(row) // 13
        weights = row[:c]
        means = row[c:4 * c].reshape(c, 3)
        covs = row[4 * c:].reshape(c, 3, 3)
        A = B = D = 0.0
        logmarks = []
        for weight, mean, cov in zip(weights, means, covs):
            A += weight * np.dot(hw, first(mean[2] + np.sqrt(2 * cov[2, 2]) * hx))
            v = cov + measurement
            inverse = np.linalg.inv(v)
            delta = standardized - mean
            gain = cov @ inverse
            conditional_mean = mean[2] + delta @ gain[2]
            conditional_variance = cov[2, 2] - (gain @ cov)[2, 2]
            if conditional_variance < -1e-12:
                raise RuntimeError('invalid conditional measurement variance')
            conditional_variance = max(conditional_variance, 0.0)
            logpdf = -0.5 * (3 * np.log(2 * np.pi) + np.linalg.slogdet(v)[1] + np.einsum('ni,ij,nj->n', delta, inverse, delta))
            selection = first(conditional_mean[:, None] + np.sqrt(2 * conditional_variance) * hx) @ hw
            logmarks.append(np.log(weight) + logpdf + np.log(selection))
            xy_inverse = np.linalg.inv(cov[:2, :2])
            delta = grid - mean[:2]
            xy_density = np.exp(-0.5 * np.einsum('ni,ij,nj->n', delta, xy_inverse, delta)) / (2 * np.pi * np.sqrt(np.linalg.det(cov[:2, :2])))
            mass_mean = mean[2] + delta @ xy_inverse @ cov[:2, 2]
            mass_variance = cov[2, 2] - cov[2, :2] @ xy_inverse @ cov[:2, 2]
            if mass_variance <= 0:
                raise RuntimeError('invalid conditional mass variance')
            mass_density = np.exp(-0.5 * (zm[None, :] - mass_mean[:, None]) ** 2 / mass_variance) / np.sqrt(2 * np.pi * mass_variance)
            B += weight * np.dot(wxy * xy_density, mass_density @ (wm * second(zm)))
            D += weight * np.dot(wxy * xy_density, mass_density @ (wm * first(zm) * second(zm)))
        output.append(np.r_[A, B, D, logsumexp(logmarks, axis=0)])
    result = np.asarray(output)
    A, B, D = result[:, :3].T
    if not np.all(np.isfinite(result)) or np.any(A < 0) or np.any(A > 1) or np.any(B < 0) or np.any(B > 1) or np.any(D < 0) or np.any(D > np.minimum(A, B) + 1e-12) or np.any(1 - A - B + D < -1e-12):
        raise RuntimeError('invalid campaign quadrature probabilities')
    return result

def _campaign_posterior(inventory, priors, statistics, finds, calibration, beta_prior, population_prior, efficiency_nodes, background_model, background_prior, background_control, background_scale):
    n = len(finds)
    H, R = priors.shape
    A, B, D = statistics[:, :3].T
    logmarks = statistics[:, 3:]
    shapes = np.broadcast_to(beta_prior, (H, 2))
    a = shapes[:, 0] + calibration[:, 1].sum()
    b = shapes[:, 1] + (calibration[:, 0] - calibration[:, 1]).sum()
    calibration_count = np.sum(gammaln(calibration[:, 0] + 1) - gammaln(calibration[:, 1] + 1) - gammaln(calibration[:, 0] - calibration[:, 1] + 1))
    log_calibration = betaln(a, b) - betaln(shapes[:, 0], shapes[:, 1]) + calibration_count
    standardized = finds.copy()
    standardized[:, :2] = (finds[:, :2] - [198000, 5000]) / [4000, 2000]
    standardized[:, 2] = np.log(finds[:, 2] / 0.05)
    covariance = background_model[3:].reshape(3, 3)
    delta = standardized - background_model[:3]
    log_background_marks = -0.5 * (3 * np.log(2 * np.pi) + np.linalg.slogdet(covariance)[1] + np.einsum('ni,ij,nj->n', delta, np.linalg.inv(covariance), delta))
    u = background_prior[0] + background_control[0]
    v = background_prior[1] + background_control[1]
    masks = np.arange(2 ** n)[:, None] >> np.arange(n) & 1
    sizes = masks.sum(axis=1)
    log_background = np.full(n + 1, -np.inf)
    for j in range(n + 1):
        if background_scale == 0:
            log_background[j] = 0.0 if j == 0 else -np.inf
        else:
            log_background[j] = j * np.log(background_scale) + gammaln(u + j) - gammaln(u) + u * np.log(v) - (u + j) * np.log(v + background_scale)
    log_terms = np.full((H, R, len(masks)), -np.inf)
    recovery = np.zeros_like(log_terms)
    for K in range(H):
        for k in range(n + 1):
            chosen = np.flatnonzero(sizes == k)
            x, w = roots_jacobi(efficiency_nodes, b[K] - 1, a[K] + k - 1)
            p, w = ((x + 1) / 2, w / w.sum())
            remaining = np.maximum(inventory - k, 0)
            log_first_void = remaining[:, None] * np.log1p(-p * A[:, None])
            log_both_void = remaining[:, None] * np.log1p(-p * (A + B)[:, None] + p * p * D[:, None])
            logJ = logsumexp(log_first_void + np.log(w), axis=1)
            logJ0 = logsumexp(log_both_void + np.log(w), axis=1)
            log_beta_moment = betaln(a[K] + k, b[K]) - betaln(a[K], b[K])
            log_count = gammaln(inventory + 1) - gammaln(remaining + 1) - gammaln(n + 1)
            base = np.log(population_prior[K]) + log_calibration[K] + np.log(priors[K]) + log_count + log_beta_moment + logJ + log_background[n - k]
            base = np.where(inventory >= k, base, -np.inf)
            for index in chosen:
                selected = masks[index].astype(bool)
                log_terms[K, :, index] = base + logmarks[:, selected].sum(axis=1) + log_background_marks[~selected].sum()
                recovery[K, :, index] = -np.expm1(logJ0 - logJ)
    normalizer = logsumexp(log_terms)
    if not np.isfinite(normalizer):
        raise RuntimeError('nonfinite catalog evidence')
    joint = np.exp(log_terms - normalizer)
    posterior = joint.sum(axis=(1, 2))
    association = joint.sum(axis=(0, 1)) @ masks
    population_recovery = (joint * recovery).sum(axis=(1, 2))
    result = np.r_[population_recovery.sum(), posterior, association, population_recovery]
    if not np.all(np.isfinite(result)) or np.any(result < -1e-12) or np.any(result > 1 + 1e-12):
        raise RuntimeError('nonfinite or invalid posterior probability')
    return result

def infer_campaign_recovery(population_models, inventory, scenario_priors, observations=((196900.0, 4550.0, 0.095), (199650.0, 5250.0, 0.022), (198150.0, 5050.0, 0.058)), observation_covariance=((0.0875 ** 2, 0.25 * 0.0875 * 0.125, 0.0), (0.25 * 0.0875 * 0.125, 0.125 ** 2, 0.0), (0.0, 0.0, 0.18 ** 2)), calibration=((13, 2), (17, 5), (11, 3), (19, 6)), beta_prior=((2.3, 5.7), (4.1, 10.3), (1.2, 3.8)), population_prior=(0.3, 0.45, 0.25), survey_response=(0.06, 0.75, 0.035, 0.9), ellipse_center=(198500.0, 4850.0), ellipse_axes=(2600.0, 1300.0), ellipse_angle_degrees=22.0, mass_bounds=(0.025, 0.18), hermite_nodes=64, radial_nodes=56, angular_nodes=88, mass_nodes=48, efficiency_nodes=64, background_model=(0.1, -0.15, 0.05, 0.7, 0.08, 0.12, 0.08, 0.5, -0.06, 0.12, -0.06, 0.9), background_prior=(2.4, 1.6), background_control=(4, 3.0), background_scale=1.0):
    models = _campaign_array(population_models)
    if models.ndim != 2 or models.shape[0] < 1 or models.shape[1] < 13 or models.shape[1] % 13:
        raise ValueError('models must have shape (R,13*C)')
    R, width = models.shape
    C = width // 13
    if np.any(models[:, :C] <= 0) or not np.allclose(models[:, :C].sum(axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError('invalid mixture probabilities')
    covs = models[:, 4 * C:].reshape(R, C, 3, 3)
    if not np.allclose(covs, covs.swapaxes(-1, -2), rtol=0, atol=1e-12) or np.any(np.linalg.eigvalsh(covs) <= 0):
        raise ValueError('covariances must be symmetric positive definite')
    finds = _campaign_array(observations)
    if finds.ndim != 2 or finds.shape[1] != 3 or len(finds) > 12 or np.any(finds[:, 2] <= 0):
        raise ValueError('invalid catalog')
    counts = _campaign_array(inventory, (R,))
    if any((isinstance(x, (bool, np.bool_)) for x in np.asarray(inventory, dtype=object).flat)) or np.any(counts != np.floor(counts)) or np.any(counts < 0):
        raise ValueError('inventory must contain nonnegative integer counts')
    priors = _campaign_array(scenario_priors)
    if priors.ndim != 2 or priors.shape[0] < 1 or priors.shape[1] != R or np.any(priors <= 0) or (not np.allclose(priors.sum(axis=1), 1.0, rtol=0, atol=1e-12)):
        raise ValueError('invalid scenario priors')
    prior = _campaign_array(population_prior, (len(priors),))
    if np.any(prior <= 0) or not np.isclose(prior.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError('invalid population prior')
    calibration_array = _campaign_array(calibration)
    if calibration_array.ndim != 2 or calibration_array.shape[1] != 2 or len(calibration_array) < 1 or any((isinstance(x, (bool, np.bool_)) for x in np.asarray(calibration, dtype=object).flat)) or np.any(calibration_array != np.floor(calibration_array)) or np.any(calibration_array[:, 0] < 1) or np.any(calibration_array[:, 1] < 0) or np.any(calibration_array[:, 1] > calibration_array[:, 0]):
        raise ValueError('invalid binomial calibration')
    shapes = _campaign_array(beta_prior)
    if shapes.shape not in [(2,), (len(priors), 2)]:
        raise ValueError('Beta prior must be a pair or one pair per population')
    background = _campaign_array(background_model, (12,))
    background_covariance = background[3:].reshape(3, 3)
    if not np.allclose(background_covariance, background_covariance.T, rtol=0, atol=1e-12) or np.any(np.linalg.eigvalsh(background_covariance) <= 0):
        raise ValueError('background covariance must be symmetric positive definite')
    gamma_prior = _campaign_array(background_prior, (2,))
    control = _campaign_array(background_control, (2,))
    scale = _campaign_array(background_scale, ())
    if np.any(gamma_prior <= 0) or control[0] < 0 or control[0] != np.floor(control[0]) or (control[1] <= 0) or any((isinstance(x, (bool, np.bool_)) for x in np.asarray(background_control, dtype=object).flat)) or (scale < 0):
        raise ValueError('invalid background rate or control')
    if scale == 0 and np.all(counts < len(finds)):
        raise ValueError('catalog impossible without background')
    response = _campaign_array(survey_response, (4,))
    center = _campaign_array(ellipse_center, (2,))
    axes = _campaign_array(ellipse_axes, (2,))
    bounds = _campaign_array(mass_bounds, (2,))
    angle = _campaign_array(ellipse_angle_degrees, ())
    if np.any(shapes <= 0) or np.any(response <= 0) or np.any(axes <= 0) or (bounds[0] <= 0) or (bounds[0] >= bounds[1]):
        raise ValueError('invalid positive parameter or interval')
    measurement = _campaign_array(observation_covariance, (3, 3))
    if not np.allclose(measurement, measurement.T, rtol=0, atol=1e-12):
        raise ValueError('measurement covariance must be positive semidefinite')
    measurement = (measurement + measurement.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(measurement)
    if np.any(eigenvalues < -1e-12):
        raise ValueError('measurement covariance must be positive semidefinite')
    if np.any(eigenvalues < 0):
        measurement = eigenvectors * np.maximum(eigenvalues, 0.0) @ eigenvectors.T
    orders = (hermite_nodes, radial_nodes, angular_nodes, mass_nodes, efficiency_nodes)
    for order in orders:
        try:
            valid = not isinstance(order, (bool, np.bool_)) and np.isscalar(order) and np.isfinite(order) and (int(order) == order) and (order >= 2)
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError('quadrature orders must be integers >=2')
    stats = _campaign_statistics(models, finds, measurement, response, center, axes, float(angle), bounds, tuple(map(int, orders[:4])))
    return _campaign_posterior(counts, priors, stats, finds, calibration_array, shapes, prior, int(efficiency_nodes), background, gamma_prior, control, float(scale))

def _reference_test_cases():
    """Literal positive, boundary, and input-domain fixtures."""
    return [{'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])', 'call': 'infer_campaign_recovery(models, [7, 11], [[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))', 'gold_call': 'infer_campaign_recovery(models, [7, 11], [[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.r_[1.,-.4,.2,-.3,cov.ravel()][None,:]', 'call': 'infer_campaign_recovery(models, [2], [[1.0]], population_prior=(1.0,), observation_covariance=np.zeros((3, 3)), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))', 'gold_call': 'infer_campaign_recovery(models, [2], [[1.0]], population_prior=(1.0,), observation_covariance=np.zeros((3, 3)), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.r_[1.,-.4,.2,-.3,cov.ravel()][None,:]', 'call': 'infer_campaign_recovery(models, [0], [[1.0]], observations=np.empty((0, 3)), population_prior=(1.0,), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))', 'gold_call': 'infer_campaign_recovery(models, [0], [[1.0]], observations=np.empty((0, 3)), population_prior=(1.0,), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])', 'call': 'infer_campaign_recovery(models, [9, 17], [[0.8, 0.2], [0.2, 0.8]], observations=((198200.0, 4900.0, 0.08),), observation_covariance=np.diag([0.02, 0.03, 0.07]), calibration=((7, 0), (11, 11)), beta_prior=(0.9, 1.7), population_prior=(0.7, 0.3), survey_response=(0.08, 0.6, 0.02, 1.2), ellipse_center=(197700.0, 5100.0), ellipse_axes=(2100.0, 1600.0), ellipse_angle_degrees=-31.0, mass_bounds=(0.02, 0.24), hermite_nodes=32, radial_nodes=18, angular_nodes=28, mass_nodes=24, efficiency_nodes=24)', 'gold_call': 'infer_campaign_recovery(models, [9, 17], [[0.8, 0.2], [0.2, 0.8]], observations=((198200.0, 4900.0, 0.08),), observation_covariance=np.diag([0.02, 0.03, 0.07]), calibration=((7, 0), (11, 11)), beta_prior=(0.9, 1.7), population_prior=(0.7, 0.3), survey_response=(0.08, 0.6, 0.02, 1.2), ellipse_center=(197700.0, 5100.0), ellipse_axes=(2100.0, 1600.0), ellipse_angle_degrees=-31.0, mass_bounds=(0.02, 0.24), hermite_nodes=32, radial_nodes=18, angular_nodes=28, mass_nodes=24, efficiency_nodes=24)'}, {'setup': 'models=np.r_[1.,0.,0.,0.,np.eye(3).ravel()][None,:]\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models, [-1], [[1.0]], population_prior=(1.0,), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(models, [-1], [[1.0]], population_prior=(1.0,), beta_prior=(2.3, 5.7)))'}, {'setup': 'models=np.r_[1.,0.,0.,0.,np.eye(3).ravel()][None,:]\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models, [7], [[1.0]], population_prior=(1.0,), survey_response=(0.06, np.nan, 0.035, 0.9), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(models, [7], [[1.0]], population_prior=(1.0,), survey_response=(0.06, np.nan, 0.035, 0.9), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])', 'call': 'infer_campaign_recovery(models, [9, 17], [[0.3, 0.7], [0.6, 0.4]], observations=np.empty((0, 3)), population_prior=(0.4, 0.6), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))', 'gold_call': 'infer_campaign_recovery(models, [9, 17], [[0.3, 0.7], [0.6, 0.4]], observations=np.empty((0, 3)), population_prior=(0.4, 0.6), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[.3,.7,-.7,.2,-.4,.4,-.1,.6,cov.ravel(),(cov*1.2).ravel()],np.r_[.6,.4,-.3,-.4,.2,.8,.5,-.5,(cov*1.3).ravel(),(cov*.7).ravel()]])', 'call': 'infer_campaign_recovery(models, [84, 51], [[0.2, 0.8], [0.75, 0.25]], population_prior=(0.55, 0.45), beta_prior=(0.9, 1.4), calibration=((1, 0),), hermite_nodes=4, radial_nodes=5, angular_nodes=7, mass_nodes=4, efficiency_nodes=2)', 'gold_call': 'infer_campaign_recovery(models, [84, 51], [[0.2, 0.8], [0.75, 0.25]], population_prior=(0.55, 0.45), beta_prior=(0.9, 1.4), calibration=((1, 0),), hermite_nodes=4, radial_nodes=5, angular_nodes=7, mass_nodes=4, efficiency_nodes=2)'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[True, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[True, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[2.5, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[2.5, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.8], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.8], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.0, 1.0], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.0, 1.0], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), observations=((1.0, 2.0, 0.0),), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), observations=((1.0, 2.0, 0.0),), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), observation_covariance=np.diag([1.0, 1.0, -1.0]), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), observation_covariance=np.diag([1.0, 1.0, -1.0]), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), calibration=((2, 3),), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), calibration=((2, 3),), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), calibration=((2, True),), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), calibration=((2, True),), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), calibration=((0, 0),), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), calibration=((0, 0),), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(0.0, 1.0)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(0.0, 1.0)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.7), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.7), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), survey_response=(0.06, 0.0, 0.035, 0.9), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), survey_response=(0.06, 0.0, 0.035, 0.9), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), ellipse_axes=(0.0, 1.0), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), ellipse_axes=(0.0, 1.0), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), mass_bounds=(0.2, 0.1), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), mass_bounds=(0.2, 0.1), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), efficiency_nodes=True, beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), efficiency_nodes=True, beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), hermite_nodes=1, beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), hermite_nodes=1, beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': "rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), angular_nodes='bad', beta_prior=(2.3, 5.7)))", 'gold_call': "rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), angular_nodes='bad', beta_prior=(2.3, 5.7)))"}, {'setup': 'models=np.r_[1.,0.,0.,0.,np.eye(3).ravel()][None,:]', 'call': 'infer_campaign_recovery(models, [7], [[1.0]], population_prior=(1.0,), observation_covariance=np.full((3, 3), 0.01), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))', 'gold_call': 'infer_campaign_recovery(models, [7], [[1.0]], population_prior=(1.0,), observation_covariance=np.full((3, 3), 0.01), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])', 'call': 'infer_campaign_recovery(models,[5,9],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=((.9,1.4),(2.7,3.2)),hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24,background_model=(-.2,.3,-.5,.8,.1,-.07,.1,.6,.09,-.07,.09,.5),background_prior=(1.3,2.2),background_control=(2,1.5),background_scale=.4)', 'gold_call': 'infer_campaign_recovery(models,[5,9],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=((.9,1.4),(2.7,3.2)),hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24,background_model=(-.2,.3,-.5,.8,.1,-.07,.1,.6,.09,-.07,.09,.5),background_prior=(1.3,2.2),background_control=(2,1.5),background_scale=.4)'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])', 'call': 'infer_campaign_recovery(models,[5,9],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=((.9,1.4),(2.7,3.2)),hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24,background_scale=0.)', 'gold_call': 'infer_campaign_recovery(models,[5,9],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=((.9,1.4),(2.7,3.2)),hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24,background_scale=0.)'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])', 'call': 'infer_campaign_recovery(models,[0,0],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=((.9,1.4),(2.7,3.2)),hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24)', 'gold_call': 'infer_campaign_recovery(models,[0,0],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=((.9,1.4),(2.7,3.2)),hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24)'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])', 'call': 'infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=((.9,1.4),(2.7,3.2)),hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24,observations=((198150.,5050.,.058),(196900.,4550.,.095),(199650.,5250.,.022)))', 'gold_call': 'infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=((.9,1.4),(2.7,3.2)),hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24,observations=((198150.,5050.,.058),(196900.,4550.,.095),(199650.,5250.,.022)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_prior=(0.,1.)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_prior=(0.,1.)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_control=(1.5,2.)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_control=(1.5,2.)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_control=(True,2.)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_control=(True,2.)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_control=(1,0.)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_control=(1,0.)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_scale=-1.))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_scale=-1.))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_model=np.zeros(12)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_model=np.zeros(12)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_model=np.ones(11)))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_model=np.ones(11)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=np.ones((3,2))))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=np.ones((3,2))))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),observations=np.tile((1.,2.,.03),(13,1))))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),observations=np.tile((1.,2.,.03),(13,1))))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_scale=0.))', 'gold_call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_scale=0.))'}]

"""Step 10: compose the complete posterior predictive recovery calculation."""
import numpy as np

def solve_strewn_probability(seed=761903, runs=8, kick_power=2.0, momentum_correct=True, cascade=True, rtol=2e-10, max_step=0.35, max_depth=2, components=4, iterations=120, regularization=(0.0064, 0.01, 0.0049), balance_realizations=True, survey_response=(0.06, 0.75, 0.035, 0.9), calibration=((13, 2), (17, 5), (11, 3), (19, 6)), hermite_nodes=64, radial_nodes=56, angular_nodes=88, mass_nodes=48, efficiency_nodes=64, beta_prior=((2.3, 5.7), (4.1, 10.3), (1.2, 3.8)), background_model=(0.1, -0.15, 0.05, 0.7, 0.08, 0.12, 0.08, 0.5, -0.06, 0.12, -0.06, 0.9), background_prior=(2.4, 1.6), background_control=(4, 3.0), background_scale=1.0):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    cloud = build_impact_cloud(seed, runs, kick_power, momentum_correct, cascade, rtol, max_step, max_depth)
    row_weights = compute_population_weights(cloud, balance_realizations)
    labels = np.unique(cloud[:, 3])
    inventory = np.array([np.count_nonzero(cloud[:, 3] == label) for label in labels])
    scenario_priors = np.array([[np.sum(row[cloud[:, 3] == label]) for label in labels] for row in row_weights])
    models = np.vstack([fit_ground_gmm(cloud[cloud[:, 3] == label], np.ones(inventory[i]), components, iterations, regularization) for i, label in enumerate(labels)])
    inference = infer_campaign_recovery(models, inventory, scenario_priors, survey_response=survey_response, calibration=calibration, hermite_nodes=hermite_nodes, radial_nodes=radial_nodes, angular_nodes=angular_nodes, mass_nodes=mass_nodes, efficiency_nodes=efficiency_nodes, beta_prior=beta_prior, background_model=background_model, background_prior=background_prior, background_control=background_control, background_scale=background_scale)
    return float(inference[0])
SCICODE_GOLD_EOF
