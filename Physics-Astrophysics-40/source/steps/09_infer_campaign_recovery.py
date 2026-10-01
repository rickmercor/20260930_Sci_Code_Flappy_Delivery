"""
Step 9: infer second-campaign recovery from a mixed first catalog.

Decode each model row as all $C$ weights, all $C$ three-vector means, then all $C$ row-major `3x3` covariances. Rows are alternative scenarios of one fall, with fixed inventories and population-conditional scenario priors. Within a scenario, survivor marks are iid from its normalized continuous GMM. Campaign A covers all position and positive mass; campaign B covers the supplied true-position ellipse and true-mass interval. Each eligible new-fall object has the stated logistic response multiplied by the single population-linked visibility $p$, shared across both campaigns and all calibration batches. Campaign B applies to the same true marks remaining after campaign A; only associated new-fall detections deplete the inventory.



The complete campaign-A catalog is a uniformly ordered merge of binomially detected new-fall objects and independently Poisson-distributed accumulated meteorites. New-fall marks use the scenario density convolved with the validated Gaussian reporting covariance and include campaign-A selection. Accumulated marks use the supplied already-selected Gaussian density without another response or convolution. The accumulated-only control and field background share a Gamma-distributed rate. Population priors precede the efficiency calibration. Marginalize all unknown origin labels, scenarios, populations, $p$, and the accumulated rate under this joint count-and-mark model; do not condition away either source count.



Use the Stage-8 standardized coordinates. Validate and symmetrize measurement covariance at tolerance $1e-12$, reject eigenvalues below $-1e-12$, and set smaller negative eigenvalues to zero. Continuous GMM support is all standardized space. Use Gauss-Hermite for full-line normal/logistic expectations, independent Gauss-Legendre rules for ellipse radius, angle, and finite log mass, and Gauss-Jacobi mapped by $p=(x+1)/2$ for normalized Beta-weighted visibility integrals. The ellipse angle is counterclockwise from east, its area map retains the stated radial Jacobian, and reporting error does not alter the true campaign-B target mask. Empty products and zero powers equal one, including exhausted inventories and empty catalogs.

Returns
-------
For `R` model rows, `H` rows of scenario priors, and `n` observed catalog items, return a finite vector of length `1+2H+n` ordered `(posterior_predictive_scalar, posterior[0:H], association[0:n], joint_population_recovery[0:H])`. Absolute log evidence is not a returned diagnostic. The signature defines complete parameter domains, including empty catalogs, zero background exposure, and inventories smaller than the mixed catalog.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Step 9: infer second-campaign recovery from a mixed first catalog."""
import math
import numpy as np
from scipy.special import expit, gammaln, logsumexp, roots_jacobi, betaln

def infer_campaign_recovery(population_models, inventory, scenario_priors, observations=((196900.0, 4550.0, 0.095), (199650.0, 5250.0, 0.022), (198150.0, 5050.0, 0.058)), observation_covariance=((0.0875 ** 2, 0.25 * 0.0875 * 0.125, 0.0), (0.25 * 0.0875 * 0.125, 0.125 ** 2, 0.0), (0.0, 0.0, 0.18 ** 2)), calibration=((13, 2), (17, 5), (11, 3), (19, 6)), beta_prior=((2.3, 5.7), (4.1, 10.3), (1.2, 3.8)), population_prior=(0.3, 0.45, 0.25), survey_response=(0.06, 0.75, 0.035, 0.9), ellipse_center=(198500.0, 4850.0), ellipse_axes=(2600.0, 1300.0), ellipse_angle_degrees=22.0, mass_bounds=(0.025, 0.18), hermite_nodes=64, radial_nodes=56, angular_nodes=88, mass_nodes=48, efficiency_nodes=64, background_model=(0.1, -0.15, 0.05, 0.7, 0.08, 0.12, 0.08, 0.5, -0.06, 0.12, -0.06, 0.9), background_prior=(2.4, 1.6), background_control=(4, 3.0), background_scale=1.0):
    """Infer new-fall recovery from a complete catalog with unknown fall associations.

Parameters
----------
population_models : array_like, shape (R,13*C)
    The column count must be a positive multiple of 13; set
    C=population_models.shape[1]//13. Each row stores weights in columns
    [0:C], row-major standardized means in [C:4*C] reshaped to (C,3),
    and row-major 3x3 covariances in [4*C:13*C] reshaped to (C,3,3).
    R,C>=1; weights are positive and sum to one within absolute 1e-12,
    and covariances are symmetric positive-definite with symmetry
    tolerance 1e-12.
inventory : array_like, shape (R,)
    Finite nonboolean integer N_r>=0. Counts smaller than the catalog are
    valid because other detections can be accumulated meteorites.
scenario_priors : array_like, shape (H,R)
    H>=1 positive probabilities, each row sums to one within 1e-12.
observations : array_like, shape (n,3)
    Complete merged campaign-A catalog, 0<=n<=12, meters east/north and
    positive kilograms. Empty (0,3) valid. A value-independent random
    permutation orders the union of new-fall and accumulated detections.
observation_covariance : array_like, shape (3,3)
    New-fall Gaussian reporting covariance in standardized coordinates.
    Symmetry tolerance 1e-12; symmetrize, reject eigenvalues below -1e-12,
    set other negative eigenvalues to zero. Zero covariance is valid.
calibration : array_like, shape (J,2)
    J>=1 independent binomial (trials,successes), nonboolean integers,
    trials>=1, 0<=successes<=trials. Response exactly one. Calibration
    analogs share the unknown new-fall visibility p, not the background rate.
beta_prior : array_like, shape (H,2) or (2,)
    Positive finite Beta shapes for p conditional on population K; a single
    pair broadcasts over H. Population prior precedes calibration, so its
    normalized marginal likelihood updates the population weights too.
population_prior : array_like, shape (H,)
    Positive pre-calibration probabilities summing to one within 1e-12.
survey_response : array_like, shape (4,)
    Positive finite (m_A,w_A,m_B,w_B), in kg and natural-log-mass width.
    New-fall Bernoulli probability p*expit(log(m/m_j)/w_j); A covers all
    space and positive mass, B the target. Trials independent conditional
    on true mark and p. Only actual new-fall A detections deplete N_r.
ellipse_center, ellipse_axes : array_like, shape (2,)
    Finite center in meters and positive semiaxes in meters.
ellipse_angle_degrees : float
    Finite counterclockwise angle from east to the first semiaxis.
mass_bounds : array_like, shape (2,)
    Positive strictly increasing closed true target mass interval in kg.
hermite_nodes, radial_nodes, angular_nodes, mass_nodes, efficiency_nodes : int
    Nonboolean finite integer orders >=2. Hermite exp(-x*x); Legendre
    radius/angle/log-mass; Jacobi efficiency weight includes k associated
    detections for each latent subset, not the full catalog size n.
background_model : array_like, shape (12,)
    Accumulated-meteorite reported-mark density in the SAME standardized
    coordinates: mean followed by row-major symmetric positive-definite
    3x3 covariance. Symmetry tolerance 1e-12. Already includes reporting
    error and detection selection; do not convolve or select it again.
background_prior : array_like, shape (2,)
    Positive Gamma shape/rate (u0,v0) for accumulated detection intensity
    lambda per unit exposure; independent of K, r, p and new-fall marks.
background_control : array_like, shape (2,)
    Finite (count,exposure), nonboolean integer count>=0, exposure>0,
    neither entry boolean. Count is Poisson(exposure*lambda), in an
    independent accumulated-only control survey with the same rate.
background_scale : float
    Finite nonnegative campaign-A exposure relative to the control unit.
    Background A count is Poisson(background_scale*lambda). Zero disables
    background, in which case at least one N_r must be >=n.

Returns
-------
ndarray, shape (1+2*H+n,)
    [P_search, posterior_K, association_i, population_recovery_K].
    Association is probability catalog item i belongs to the witnessed
    fall. P_search concerns at least one NEW witnessed-fall recovery in B,
    never an accumulated meteorite. Population recovery is the joint
    probability of population K AND at least one new recovery; these H
    entries sum to P_search. All conditioning includes both
    calibration data sets and the complete merged count and marks.

Raises
------
ValueError
    For nonfinite inputs or stated shape/domain violations.
RuntimeError
    If quadrature or floating arithmetic yields nonfinite evidence or
    invalid posterior probabilities.

Notes
-----
The H populations label physical and visibility distributions; r and p are
shared by all witnessed-fall objects. Conditional on r, its N_r latent
marks are iid from the continuous GMM. Campaign B searches the same true
marks remaining after campaign A. No absolute log evidence is returned.
The runtime uses NumPy 2.x, which has no ``np.math`` namespace; use the
imported standard-library ``math`` module or the supplied SciPy special functions.
Return deterministic rtol=1e-10, atol=1e-12 agreement; these code-test
tolerances differ from reasoning tolerances.
"""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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

def _oracle_infer_campaign_recovery(population_models, inventory, scenario_priors, observations=((196900.0, 4550.0, 0.095), (199650.0, 5250.0, 0.022), (198150.0, 5050.0, 0.058)), observation_covariance=((0.0875 ** 2, 0.25 * 0.0875 * 0.125, 0.0), (0.25 * 0.0875 * 0.125, 0.125 ** 2, 0.0), (0.0, 0.0, 0.18 ** 2)), calibration=((13, 2), (17, 5), (11, 3), (19, 6)), beta_prior=((2.3, 5.7), (4.1, 10.3), (1.2, 3.8)), population_prior=(0.3, 0.45, 0.25), survey_response=(0.06, 0.75, 0.035, 0.9), ellipse_center=(198500.0, 4850.0), ellipse_axes=(2600.0, 1300.0), ellipse_angle_degrees=22.0, mass_bounds=(0.025, 0.18), hermite_nodes=64, radial_nodes=56, angular_nodes=88, mass_nodes=48, efficiency_nodes=64, background_model=(0.1, -0.15, 0.05, 0.7, 0.08, 0.12, 0.08, 0.5, -0.06, 0.12, -0.06, 0.9), background_prior=(2.4, 1.6), background_control=(4, 3.0), background_scale=1.0):
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
    return [{'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])', 'call': 'infer_campaign_recovery(models, [7, 11], [[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))', 'gold_call': '_oracle_infer_campaign_recovery(models, [7, 11], [[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.r_[1.,-.4,.2,-.3,cov.ravel()][None,:]', 'call': 'infer_campaign_recovery(models, [2], [[1.0]], population_prior=(1.0,), observation_covariance=np.zeros((3, 3)), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))', 'gold_call': '_oracle_infer_campaign_recovery(models, [2], [[1.0]], population_prior=(1.0,), observation_covariance=np.zeros((3, 3)), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.r_[1.,-.4,.2,-.3,cov.ravel()][None,:]', 'call': 'infer_campaign_recovery(models, [0], [[1.0]], observations=np.empty((0, 3)), population_prior=(1.0,), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))', 'gold_call': '_oracle_infer_campaign_recovery(models, [0], [[1.0]], observations=np.empty((0, 3)), population_prior=(1.0,), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])', 'call': 'infer_campaign_recovery(models, [9, 17], [[0.8, 0.2], [0.2, 0.8]], observations=((198200.0, 4900.0, 0.08),), observation_covariance=np.diag([0.02, 0.03, 0.07]), calibration=((7, 0), (11, 11)), beta_prior=(0.9, 1.7), population_prior=(0.7, 0.3), survey_response=(0.08, 0.6, 0.02, 1.2), ellipse_center=(197700.0, 5100.0), ellipse_axes=(2100.0, 1600.0), ellipse_angle_degrees=-31.0, mass_bounds=(0.02, 0.24), hermite_nodes=32, radial_nodes=18, angular_nodes=28, mass_nodes=24, efficiency_nodes=24)', 'gold_call': '_oracle_infer_campaign_recovery(models, [9, 17], [[0.8, 0.2], [0.2, 0.8]], observations=((198200.0, 4900.0, 0.08),), observation_covariance=np.diag([0.02, 0.03, 0.07]), calibration=((7, 0), (11, 11)), beta_prior=(0.9, 1.7), population_prior=(0.7, 0.3), survey_response=(0.08, 0.6, 0.02, 1.2), ellipse_center=(197700.0, 5100.0), ellipse_axes=(2100.0, 1600.0), ellipse_angle_degrees=-31.0, mass_bounds=(0.02, 0.24), hermite_nodes=32, radial_nodes=18, angular_nodes=28, mass_nodes=24, efficiency_nodes=24)'}, {'setup': 'models=np.r_[1.,0.,0.,0.,np.eye(3).ravel()][None,:]\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models, [-1], [[1.0]], population_prior=(1.0,), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(models, [-1], [[1.0]], population_prior=(1.0,), beta_prior=(2.3, 5.7)))'}, {'setup': 'models=np.r_[1.,0.,0.,0.,np.eye(3).ravel()][None,:]\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models, [7], [[1.0]], population_prior=(1.0,), survey_response=(0.06, np.nan, 0.035, 0.9), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(models, [7], [[1.0]], population_prior=(1.0,), survey_response=(0.06, np.nan, 0.035, 0.9), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])', 'call': 'infer_campaign_recovery(models, [9, 17], [[0.3, 0.7], [0.6, 0.4]], observations=np.empty((0, 3)), population_prior=(0.4, 0.6), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))', 'gold_call': '_oracle_infer_campaign_recovery(models, [9, 17], [[0.3, 0.7], [0.6, 0.4]], observations=np.empty((0, 3)), population_prior=(0.4, 0.6), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[.3,.7,-.7,.2,-.4,.4,-.1,.6,cov.ravel(),(cov*1.2).ravel()],np.r_[.6,.4,-.3,-.4,.2,.8,.5,-.5,(cov*1.3).ravel(),(cov*.7).ravel()]])', 'call': 'infer_campaign_recovery(models, [84, 51], [[0.2, 0.8], [0.75, 0.25]], population_prior=(0.55, 0.45), beta_prior=(0.9, 1.4), calibration=((1, 0),), hermite_nodes=4, radial_nodes=5, angular_nodes=7, mass_nodes=4, efficiency_nodes=2)', 'gold_call': '_oracle_infer_campaign_recovery(models, [84, 51], [[0.2, 0.8], [0.75, 0.25]], population_prior=(0.55, 0.45), beta_prior=(0.9, 1.4), calibration=((1, 0),), hermite_nodes=4, radial_nodes=5, angular_nodes=7, mass_nodes=4, efficiency_nodes=2)'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[True, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[True, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[2.5, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[2.5, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.8], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.8], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.0, 1.0], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.0, 1.0], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), observations=((1.0, 2.0, 0.0),), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), observations=((1.0, 2.0, 0.0),), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), observation_covariance=np.diag([1.0, 1.0, -1.0]), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), observation_covariance=np.diag([1.0, 1.0, -1.0]), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), calibration=((2, 3),), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), calibration=((2, 3),), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), calibration=((2, True),), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), calibration=((2, True),), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), calibration=((0, 0),), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), calibration=((0, 0),), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(0.0, 1.0)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), beta_prior=(0.0, 1.0)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.7), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.7), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), survey_response=(0.06, 0.0, 0.035, 0.9), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), survey_response=(0.06, 0.0, 0.035, 0.9), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), ellipse_axes=(0.0, 1.0), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), ellipse_axes=(0.0, 1.0), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), mass_bounds=(0.2, 0.1), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), mass_bounds=(0.2, 0.1), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), efficiency_nodes=True, beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), efficiency_nodes=True, beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), hermite_nodes=1, beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), hermite_nodes=1, beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': "rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), angular_nodes='bad', beta_prior=(2.3, 5.7)))", 'gold_call': "rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), angular_nodes='bad', beta_prior=(2.3, 5.7)))"}, {'setup': 'models=np.r_[1.,0.,0.,0.,np.eye(3).ravel()][None,:]', 'call': 'infer_campaign_recovery(models, [7], [[1.0]], population_prior=(1.0,), observation_covariance=np.full((3, 3), 0.01), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))', 'gold_call': '_oracle_infer_campaign_recovery(models, [7], [[1.0]], population_prior=(1.0,), observation_covariance=np.full((3, 3), 0.01), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])', 'call': 'infer_campaign_recovery(models,[5,9],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=((.9,1.4),(2.7,3.2)),hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24,background_model=(-.2,.3,-.5,.8,.1,-.07,.1,.6,.09,-.07,.09,.5),background_prior=(1.3,2.2),background_control=(2,1.5),background_scale=.4)', 'gold_call': '_oracle_infer_campaign_recovery(models,[5,9],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=((.9,1.4),(2.7,3.2)),hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24,background_model=(-.2,.3,-.5,.8,.1,-.07,.1,.6,.09,-.07,.09,.5),background_prior=(1.3,2.2),background_control=(2,1.5),background_scale=.4)'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])', 'call': 'infer_campaign_recovery(models,[5,9],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=((.9,1.4),(2.7,3.2)),hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24,background_scale=0.)', 'gold_call': '_oracle_infer_campaign_recovery(models,[5,9],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=((.9,1.4),(2.7,3.2)),hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24,background_scale=0.)'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])', 'call': 'infer_campaign_recovery(models,[0,0],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=((.9,1.4),(2.7,3.2)),hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24)', 'gold_call': '_oracle_infer_campaign_recovery(models,[0,0],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=((.9,1.4),(2.7,3.2)),hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24)'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])', 'call': 'infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=((.9,1.4),(2.7,3.2)),hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24,observations=((198150.,5050.,.058),(196900.,4550.,.095),(199650.,5250.,.022)))', 'gold_call': '_oracle_infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=((.9,1.4),(2.7,3.2)),hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24,observations=((198150.,5050.,.058),(196900.,4550.,.095),(199650.,5250.,.022)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_prior=(0.,1.)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_prior=(0.,1.)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_control=(1.5,2.)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_control=(1.5,2.)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_control=(True,2.)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_control=(True,2.)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_control=(1,0.)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_control=(1,0.)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_scale=-1.))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_scale=-1.))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_model=np.zeros(12)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_model=np.zeros(12)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_model=np.ones(11)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_model=np.ones(11)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=np.ones((3,2))))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=np.ones((3,2))))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),observations=np.tile((1.,2.,.03),(13,1))))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),observations=np.tile((1.,2.,.03),(13,1))))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_scale=0.))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_scale=0.))'}]

# =============================================================================
# TEST CASES
# =============================================================================

import math
import numpy as np
from scipy.special import expit, gammaln, logsumexp, roots_jacobi, betaln

def test_cases():
    """Representative catalog, boundary, and domain fixtures.

    The complete public pipeline exercises the four-component default model and
    the three-observation finite-inventory case in the integration suite.
    """
    return [{'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.r_[1.,-.4,.2,-.3,cov.ravel()][None,:]', 'call': 'infer_campaign_recovery(models, [0], [[1.0]], observations=np.empty((0, 3)), population_prior=(1.0,), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))', 'gold_call': '_oracle_infer_campaign_recovery(models, [0], [[1.0]], observations=np.empty((0, 3)), population_prior=(1.0,), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])', 'call': 'infer_campaign_recovery(models, [9, 17], [[0.8, 0.2], [0.2, 0.8]], observations=((198200.0, 4900.0, 0.08),), observation_covariance=np.diag([0.02, 0.03, 0.07]), calibration=((7, 0), (11, 11)), beta_prior=(0.9, 1.7), population_prior=(0.7, 0.3), survey_response=(0.08, 0.6, 0.02, 1.2), ellipse_center=(197700.0, 5100.0), ellipse_axes=(2100.0, 1600.0), ellipse_angle_degrees=-31.0, mass_bounds=(0.02, 0.24), hermite_nodes=32, radial_nodes=18, angular_nodes=28, mass_nodes=24, efficiency_nodes=24)', 'gold_call': '_oracle_infer_campaign_recovery(models, [9, 17], [[0.8, 0.2], [0.2, 0.8]], observations=((198200.0, 4900.0, 0.08),), observation_covariance=np.diag([0.02, 0.03, 0.07]), calibration=((7, 0), (11, 11)), beta_prior=(0.9, 1.7), population_prior=(0.7, 0.3), survey_response=(0.08, 0.6, 0.02, 1.2), ellipse_center=(197700.0, 5100.0), ellipse_axes=(2100.0, 1600.0), ellipse_angle_degrees=-31.0, mass_bounds=(0.02, 0.24), hermite_nodes=32, radial_nodes=18, angular_nodes=28, mass_nodes=24, efficiency_nodes=24)'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])', 'call': 'infer_campaign_recovery(models, [9, 17], [[0.3, 0.7], [0.6, 0.4]], observations=np.empty((0, 3)), population_prior=(0.4, 0.6), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))', 'gold_call': '_oracle_infer_campaign_recovery(models, [9, 17], [[0.3, 0.7], [0.6, 0.4]], observations=np.empty((0, 3)), population_prior=(0.4, 0.6), hermite_nodes=24, radial_nodes=16, angular_nodes=24, mass_nodes=20, efficiency_nodes=20, beta_prior=(2.3, 5.7))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), observation_covariance=np.diag([1.0, 1.0, -1.0]), beta_prior=(2.3, 5.7)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), observation_covariance=np.diag([1.0, 1.0, -1.0]), beta_prior=(2.3, 5.7)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': "rejects(lambda: infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), angular_nodes='bad', beta_prior=(2.3, 5.7)))", 'gold_call': "rejects(lambda: _oracle_infer_campaign_recovery(population_models=models, inventory=[7, 11], scenario_priors=[[0.3, 0.7], [0.6, 0.4]], population_prior=(0.4, 0.6), angular_nodes='bad', beta_prior=(2.3, 5.7)))"}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_model=np.zeros(12)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_model=np.zeros(12)))'}, {'setup': 'cov=np.array([[.7,.04,.18],[.04,.4,-.09],[.18,-.09,.8]])\nmodels=np.vstack([np.r_[1.,-.4,.2,-.3,cov.ravel()],np.r_[1.,.5,-.1,.4,(cov*1.2).ravel()]])\ndef rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_model=np.ones(11)))', 'gold_call': 'rejects(lambda: _oracle_infer_campaign_recovery(models,[1,2],[[.3,.7],[.6,.4]],population_prior=(.4,.6),beta_prior=(2.3,5.7),background_model=np.ones(11)))'}]
