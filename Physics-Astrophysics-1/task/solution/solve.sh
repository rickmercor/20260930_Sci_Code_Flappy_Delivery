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
from scipy.integrate import quad


def compute_cosmic_age(z: float) -> float:
    if not np.isscalar(z) or isinstance(z, (bool, np.bool_, str)) or not np.isrealobj(z):
        raise ValueError("z must be a finite scalar with z >= 0")
    z = float(z)
    if not np.isfinite(z) or z < 0.0:
        raise ValueError("z must be a finite scalar with z >= 0")
    _H0_KM_S_MPC, _OMEGA_M, _GYR_PER_INV_KM_S_MPC = 67.7, 0.31, 977.79222168
    omega_l = 1.0 - _OMEGA_M

    inv_power = (1.0 + z) ** -1.5
    age = 2.0 * np.arcsinh(np.sqrt(omega_l / _OMEGA_M) * inv_power)
    age /= 3.0 * np.sqrt(omega_l)
    return float(age * _GYR_PER_INV_KM_S_MPC / _H0_KM_S_MPC)

import numpy as np
from scipy.integrate import quad, cumulative_simpson


def build_time_redshift_grid(n_samples: int) -> np.ndarray:
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise ValueError("n_samples must be an integer >= 16")
    if n_samples < 16:
        raise ValueError("n_samples must be an integer >= 16")
    t0 = compute_cosmic_age(0.0)
    t_cap = compute_cosmic_age(60.0)
    t_grid = np.linspace(0.0, t0, int(n_samples))
    omega_m = 0.31
    omega_l = 1.0 - omega_m
    hubble_time = 977.79222168 / 67.7
    active = t_grid > t_cap
    z_grid = np.full(t_grid.shape, 60.0)
    scaled_time = 1.5 * np.sqrt(omega_l) * t_grid[active] / hubble_time
    z_grid[active] = (np.sqrt(omega_l / omega_m) / np.sinh(scaled_time)) ** (2.0 / 3.0) - 1.0
    z_grid[-1] = 0.0
    return np.vstack([t_grid, z_grid])

import numpy as np

def affine_merger_family(z_grid: np.ndarray, pivot_rate: float, r_asym: float,
                                 amplitudes: np.ndarray, kappa: float = 3.2) -> np.ndarray:
    if not np.isrealobj(z_grid) or not np.isrealobj(amplitudes):
        raise ValueError("grid and amplitudes must be real")
    try:
        z = np.asarray(z_grid, dtype=float)
        amp = np.asarray(amplitudes, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("grid and amplitudes must be finite real arrays") from exc
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError("redshifts must be nonempty, finite and nonnegative")
    if amp.shape != (3,) or not np.all(np.isfinite(amp)) or np.any(amp < 0) or amp.sum() >= 1:
        raise ValueError("three nonnegative amplitudes with sum below one required")
    vals = (pivot_rate, r_asym, kappa)
    if any(not np.isscalar(v) or not np.isrealobj(v) for v in vals):
        raise ValueError("rate and exponent parameters must be real scalars")
    try:
        rp, ra, exponent = map(float, vals)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("rate and exponent parameters must be finite real scalars") from exc
    if not np.all(np.isfinite([rp,ra,exponent])) or rp <= 0 or ra < 0:
        raise ValueError("positive pivot and nonnegative boundary required")
    result = np.zeros((4,z.size))
    result[0] = ra
    active = z < 1.55
    za = z[active]
    low = .5*(1-np.cos(np.pi*np.clip(za/.1,0,1)))
    blend = .5*(1-np.cos(np.pi*np.clip((za-1.5)/.05,0,1)))
    with np.errstate(over='ignore',invalid='ignore'):
        core = low*(1-blend)*rp*np.exp(exponent*(np.log1p(za)-np.log(1.2)))
        result[0,active] = core + low*blend*ra
        result[1,active] = core*amp[0]
        result[2,active] = core*amp[1]*np.exp(-.5*((za-.35)/.12)**2)
        result[3,active] = core*amp[2]*np.exp(-.5*((za-.95)/.17)**2)
    if not np.all(np.isfinite(result)):
        raise ValueError("evaluated history family must be finite")
    return result

import numpy as np

def delay_time_distribution(tau_grid: np.ndarray, alpha: float,
                                    tau_min: float, tau_max: float) -> np.ndarray:
    if not np.isrealobj(tau_grid):
        raise ValueError("delay samples must be real")
    try:
        tau = np.asarray(tau_grid,dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("delay samples must be finite real values") from exc
    if tau.ndim != 1 or tau.size == 0 or not np.all(np.isfinite(tau)) or np.any(tau<0):
        raise ValueError("nonempty finite nonnegative delay grid required")
    if any(not np.isscalar(v) or not np.isrealobj(v) for v in (alpha,tau_min,tau_max)):
        raise ValueError("exponent and limits must be real scalars")
    try:
        exponent, lo, hi = map(float,(alpha,tau_min,tau_max))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("exponent and limits must be finite real scalars") from exc
    if not np.all(np.isfinite([exponent,lo,hi])) or not 0<lo<hi:
        raise ValueError("finite exponent and 0 < lower < upper support required")
    width = np.log(hi)-np.log(lo)
    beta = exponent+1
    x = beta*width
    if not np.isfinite(x):
        raise ValueError("exponent-support product must be finite")
    if x == 0:
        relative = 0.0
    elif x > 50:
        relative = x+np.log1p(-np.exp(-x))-np.log(x)
    elif x < -50:
        relative = np.log1p(-np.exp(x))-np.log(-x)
    else:
        relative = np.log(np.expm1(x)/x)
    log_z = beta*np.log(lo)+np.log(width)+relative
    inside = (tau>=lo)&(tau<=hi)
    result = np.zeros_like(tau)
    with np.errstate(over='ignore',invalid='ignore'):
        result[inside] = np.exp(exponent*np.log(tau[inside])-log_z)
    if not np.all(np.isfinite(result)):
        raise ValueError("density must evaluate finitely")
    return result

import numpy as np

def wiener_formation_jet(merger_family: np.ndarray,
                                 steep_density: np.ndarray,
                                 shallow_density: np.ndarray,
                                 fraction: float, dt: float) -> np.ndarray:
    values = (merger_family, steep_density, shallow_density, fraction, dt)
    if any(not np.isrealobj(v) for v in values):
        raise ValueError("all inputs must be real")
    try:
        family, steep, shallow = [np.asarray(v, dtype=float) for v in values[:3]]
        if not np.isscalar(fraction) or not np.isscalar(dt):
            raise ValueError("fraction and spacing must be scalars")
        f, spacing = float(fraction), float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("finite real arrays and scalars required") from exc
    if family.ndim != 2 or family.shape[0] != 4 or family.shape[1] < 2:
        raise ValueError("merger_family must have shape (4,N), N >= 2")
    n = family.shape[1]
    if steep.shape != (n,) or shallow.shape != (n,):
        raise ValueError("component densities must match the time grid")
    if not all(np.all(np.isfinite(v)) for v in (family, steep, shallow)):
        raise ValueError("all arrays must be finite")
    if any(np.any(v < 0.0) or not np.any(v > 0.0) for v in (steep, shallow)):
        raise ValueError("component densities must be nonnegative and nonzero")
    if not np.isfinite(f) or not 0.0 <= f <= 1.0 or not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("fraction in [0,1] and finite positive spacing required")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        density = (1.0 - f) * steep + f * shallow
        delta = shallow - steep
        kernel = np.fft.rfft(density * spacing, n=2*n)
        direction = np.fft.rfft(delta * spacing, n=2*n)
        rate = np.fft.rfft(family, n=2*n, axis=1)
        # A nonnegative kernel has maximal Fourier modulus at zero frequency.
        dc, dc_first = kernel[0].real, direction[0].real
        penalty = 1e-3 * dc**2
        penalty_first = 2e-3 * dc * dc_first
        penalty_second = 2e-3 * dc_first**2
        denominator = np.abs(kernel)**2 + penalty
        first_denominator = 2.0 * np.real(kernel.conj() * direction) + penalty_first
        second_denominator = 2.0 * np.abs(direction)**2 + penalty_second
        value = rate * kernel.conj() / denominator
        first = (rate * direction.conj() - first_denominator * value) / denominator
        second = -(second_denominator * value + 2.0 * first_denominator * first) / denominator
        result = np.fft.irfft(np.stack((value, first, second)), n=2*n, axis=2)[:, :, :n]
    if not np.all(np.isfinite(result)):
        raise ValueError("the formation response must be finite")
    return result

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection

def physicality_vertex_jet(formation_family_jet: np.ndarray,
                                    window_mask: np.ndarray, lower: np.ndarray,
                                    upper: np.ndarray) -> np.ndarray:
    if any(not np.isrealobj(value) for value in (formation_family_jet, lower, upper)):
        raise ValueError("formation jets and box bounds must be real")
    try:
        family, lo, hi = [np.asarray(value, dtype=float)
                          for value in (formation_family_jet, lower, upper)]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("formation jets and box bounds must be finite real arrays") from exc
    if family.ndim != 3 or family.shape[:2] != (3, 4) or family.shape[2] < 1:
        raise ValueError("formation_family_jet must have shape (3,4,N), N >= 1")
    if lo.shape != (3,) or hi.shape != (3,) or not np.all(lo < hi):
        raise ValueError("box bounds must have shape (3,) with lower < upper")
    if not all(np.all(np.isfinite(value)) for value in (family, lo, hi)):
        raise ValueError("formation jets and box bounds must be finite")
    mask = np.asarray(window_mask)
    if mask.shape != (family.shape[2],) or mask.dtype != np.bool_ or not np.any(mask):
        raise ValueError("window_mask must be boolean, matching and nonempty")
    midpoint = lo + (hi - lo) / 2.0
    halfwidth = (hi - lo) / 2.0
    selected = family[:, :, mask]
    intercepts = selected[:, 0] + np.einsum("kan,a->kn", selected[:, 1:], midpoint)
    coefficients = selected[:, 1:].transpose(0, 2, 1) * halfwidth
    zero = np.all(coefficients[0] == 0.0, axis=1)
    if np.any(intercepts[0, zero] < 0.0):
        return np.empty((3, 0, 3), dtype=float)
    necessary = (~zero) & (intercepts[0] - np.sum(np.abs(coefficients[0]), axis=1) < 0.0)
    coefficients, intercepts = coefficients[:, necessary], intercepts[:, necessary]
    count = coefficients.shape[1]
    planes = np.zeros((3, count + 6, 4), dtype=float)
    planes[:, :count, :3] = -coefficients
    planes[:, :count, 3] = -intercepts
    planes[0, count:, :3] = np.vstack((np.eye(3), -np.eye(3)))
    planes[0, count:, 3] = -1.0
    scale = np.linalg.norm(planes[0, :, :3], axis=1)
    planes = planes / scale[None, :, None]
    if not np.all(np.isfinite(planes)):
        raise ValueError("scaled constraint jets must be finite")
    base = planes[0]
    fit = linprog(np.array([0.0, 0.0, 0.0, -1.0]),
                  A_ub=np.column_stack((base[:, :3], np.ones(len(base)))),
                  b_ub=-base[:, 3], bounds=[(None, None)] * 3 + [(0.0, None)], method="highs")
    if fit.status == 2:
        return np.empty((3, 0, 3), dtype=float)
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError("could not determine a finite interior point")
    if fit.x[3] <= 0.0:
        return np.empty((3, 0, 3), dtype=float)
    try:
        region = HalfspaceIntersection(base, fit.x[:3])
        if not np.any(family[1:]):
            positions = np.unique(midpoint + halfwidth * region.intersections, axis=0)
            result = np.zeros((3, len(positions), 3), dtype=float)
            result[0] = positions
            if not np.all(np.isfinite(result)):
                raise ValueError("static vertices must be finite")
            return result
        vertices = []
        for active in region.dual_facets:
            if len(active) != 3:
                raise ValueError("supported moving vertices must be simple")
            indices = np.asarray(active, dtype=int)
            matrix = planes[:, indices, :3]
            constant = planes[:, indices, 3]
            value = np.linalg.solve(matrix[0], -constant[0])
            first = np.linalg.solve(matrix[0], -constant[1] - matrix[1] @ value)
            second = np.linalg.solve(matrix[0], -constant[2] - matrix[2] @ value - 2.0 * matrix[1] @ first)
            vertices.append(np.array([midpoint + halfwidth * value,
                                      halfwidth * first, halfwidth * second]))
    except Exception as exc:
        raise ValueError("the moving polytope could not be evaluated") from exc
    result = np.asarray(vertices).transpose(1, 0, 2)
    if not np.all(np.isfinite(result)):
        raise ValueError("vertex jets must be finite")
    order = np.lexsort((result[0, :, 2], result[0, :, 1], result[0, :, 0]))
    return result[:, order]

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial import ConvexHull

def gaussian_polytope_moment_jet(vertex_jet: np.ndarray, mean: np.ndarray,
                                         covariance: np.ndarray,
                                         tol: float = 1e-8) -> np.ndarray:
    if any(not np.isrealobj(value) for value in (vertex_jet, mean, covariance)):
        raise ValueError("vertex jets, mean and covariance must be real")
    try:
        vertices, mu, cov = [np.asarray(value, dtype=float) for value in (vertex_jet, mean, covariance)]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("vertex jets, mean and covariance must be finite real arrays") from exc
    if vertices.ndim != 3 or vertices.shape[0] != 3 or vertices.shape[2] != 3 or mu.shape != (3,) or cov.shape != (3, 3):
        raise ValueError("expected shapes (3,V,3), (3,), and (3,3)")
    if not all(np.all(np.isfinite(value)) for value in (vertices, mu, cov)):
        raise ValueError("vertex jets, mean and covariance must be finite")
    if not np.allclose(cov, cov.T, rtol=0.0, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    cov = (cov + cov.T) / 2.0
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues[-1] / eigenvalues[0] > 1e6:
        raise ValueError("covariance condition number must be at most 1e6")
    inverse_chol = np.linalg.inv(chol)
    if vertices.shape[1] and np.any(np.sum(((vertices[0] - mu) @ inverse_chol.T) ** 2, axis=1) > 144.0):
        raise ValueError("base vertices must lie within Mahalanobis distance 12 of the mean")
    if not np.isscalar(tol) or not np.isrealobj(tol):
        raise ValueError("tol must be a finite real accuracy target")
    try:
        tolerance = float(tol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("tol must be a finite real accuracy target") from exc
    if not np.isfinite(tolerance) or not 1e-10 <= tolerance <= 1e-5:
        raise ValueError("tol must lie in [1e-10,1e-5]")
    if vertices.shape[1] < 4 or np.linalg.matrix_rank(vertices[0] - vertices[0].mean(axis=0)) < 3:
        return np.zeros((3, 10), dtype=float)
    try:
        hull = ConvexHull(vertices[0])
    except Exception as exc:
        raise ValueError("the base convex region could not be constructed") from exc
    center = vertices[:, hull.vertices].mean(axis=1)
    static_region = not np.any(vertices[1:])
    precision = inverse_chol.T @ inverse_chol
    normalization = (2.0 * np.pi) ** (-1.5) / np.prod(np.diag(chol))

    def _moving_gaussian_integral(order):
        nodes, weights = leggauss(order)
        nodes, weights = (nodes + 1.0) / 2.0, weights / 2.0
        a, b, c = np.meshgrid(nodes, nodes, nodes, indexing="ij")
        wa, wb, wc = np.meshgrid(weights, weights, weights, indexing="ij")
        barycentric = np.column_stack((a.ravel(), ((1.0 - a) * b).ravel(),
                                       ((1.0 - a) * (1.0 - b) * c).ravel()))
        quadrature_weights = (wa * wb * wc * (1.0 - a) ** 2 * (1.0 - b)).ravel()
        total = np.zeros((3, 10), dtype=float)
        for face in hull.simplices:
            if static_region:
                base = (vertices[0, face] - center[0]).T
                samples = center[0] + barycentric @ base.T
                standardized = (samples - mu) @ inverse_chol.T
                weight = (quadrature_weights * abs(np.linalg.det(base)) * normalization
                          * np.exp(-0.5 * np.sum(standardized ** 2, axis=1)))
                x, y, z = samples.T
                total[0, 0] += np.sum(weight)
                total[0, 1:4] += weight @ samples
                total[0, 4:] += weight @ np.column_stack((x*x, x*y, x*z, y*y, y*z, z*z))
                continue
            basis = (vertices[:, face] - center[:, None, :]).transpose(0, 2, 1)
            inverse_basis = np.linalg.inv(basis[0])
            first_matrix = inverse_basis @ basis[1]
            jacobian_first = np.trace(first_matrix)
            jacobian_second = np.trace(inverse_basis @ basis[2] - first_matrix @ first_matrix)
            determinant = abs(np.linalg.det(basis[0]))
            samples = center[:, None, :] + np.einsum("pi,kji->kpj", barycentric, basis)
            displacement = samples[0] - mu
            gaussian_first = -np.einsum("pi,ij,pj->p", samples[1], precision, displacement)
            gaussian_second = (-np.einsum("pi,ij,pj->p", samples[2], precision, displacement)
                               -np.einsum("pi,ij,pj->p", samples[1], precision, samples[1]))
            logarithmic_first = jacobian_first + gaussian_first
            logarithmic_second = jacobian_second + gaussian_second
            standardized = displacement @ inverse_chol.T
            weight = quadrature_weights * determinant * normalization * np.exp(-0.5 * np.sum(standardized ** 2, axis=1))
            weight_jet = np.array([weight, weight * logarithmic_first,
                                   weight * (logarithmic_first ** 2 + logarithmic_second)])
            monomials = np.zeros((3, len(barycentric), 10), dtype=float)
            monomials[0, :, 0] = 1.0
            monomials[:, :, 1:4] = samples
            for column, (i, j) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)), 4):
                monomials[0, :, column] = samples[0, :, i] * samples[0, :, j]
                monomials[1, :, column] = samples[1, :, i] * samples[0, :, j] + samples[0, :, i] * samples[1, :, j]
                monomials[2, :, column] = (samples[2, :, i] * samples[0, :, j]
                                          + 2.0 * samples[1, :, i] * samples[1, :, j]
                                          + samples[0, :, i] * samples[2, :, j])
            total[0] += weight_jet[0] @ monomials[0]
            total[1] += weight_jet[1] @ monomials[0] + weight_jet[0] @ monomials[1]
            total[2] += (weight_jet[2] @ monomials[0] + 2.0 * weight_jet[1] @ monomials[1]
                         + weight_jet[0] @ monomials[2])
        return total

    previous = _moving_gaussian_integral(8)
    for order in (12, 16, 24, 32, 48, 64):
        current = _moving_gaussian_integral(order)
        if not np.all(np.isfinite(current)):
            raise ValueError("the Gaussian response is nonfinite")
        if np.all(np.abs(current - previous) <= 0.2 * tolerance * (1.0 + np.abs(current))):
            return current
        previous = current
    raise ValueError("moving Gaussian integration did not converge to the requested accuracy")

import numpy as np

def condition_gaussian_mixture_jet(accepted_moment_jets: np.ndarray,
                                           prior_moments: np.ndarray,
                                           weights: np.ndarray) -> np.ndarray:
    if any(not np.isrealobj(v) for v in (accepted_moment_jets, prior_moments, weights)):
        raise ValueError("moments and weights must be real")
    try:
        accepted, prior, weight = [np.asarray(v, dtype=float)
                                  for v in (accepted_moment_jets, prior_moments, weights)]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("finite real moments and weights required") from exc
    if accepted.ndim != 3 or accepted.shape[1:] != (3,10) or accepted.shape[0] < 1:
        raise ValueError("accepted_moment_jets must have shape (G,3,10)")
    if prior.shape != (accepted.shape[0],10) or weight.shape != (accepted.shape[0],):
        raise ValueError("prior moments and weights must match the components")
    if not all(np.all(np.isfinite(v)) for v in (accepted, prior, weight)):
        raise ValueError("all inputs must be finite")
    if np.any(accepted[:,0,0] < 0) or np.any(prior[:,0] < 0) or np.any(weight < 0) or not np.any(weight > 0):
        raise ValueError("nonnegative masses and nonnegative nonzero weights required")
    weight = weight / np.max(weight)
    denominator = float(weight @ prior[:,0])
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("the weighted box mass must be positive")
    raw = np.einsum('g,gij->ij', weight, accepted)
    if raw[0,0] == 0:
        return np.zeros((3,10))
    normalized = np.empty((3,9))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        normalized[0] = raw[0,1:] / raw[0,0]
        normalized[1] = (raw[1,1:] - raw[1,0]*normalized[0]) / raw[0,0]
        normalized[2] = (raw[2,1:] - raw[2,0]*normalized[0] - 2*raw[1,0]*normalized[1]) / raw[0,0]
        result = np.column_stack((raw[:,0]/denominator, normalized))
        pairs = ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))
        for j, (a,b) in enumerate(pairs):
            ma, mb = normalized[:,a], normalized[:,b]
            product = np.array([ma[0]*mb[0], ma[1]*mb[0]+ma[0]*mb[1],
                                ma[2]*mb[0]+2*ma[1]*mb[1]+ma[0]*mb[2]])
            result[:,4+j] -= product
    if not np.all(np.isfinite(result)):
        raise ValueError("the conditional response must be finite")
    return result

import numpy as np

def reconvolution_error(r_form: np.ndarray, p_tau: np.ndarray,
                                r_merge: np.ndarray, dt: float,
                                window_mask: np.ndarray) -> float:
    values = (r_form, p_tau, r_merge)
    if any(not np.isrealobj(v) for v in values):
        raise ValueError("all histories and densities must be real")
    try:
        formation, density, merger = [np.asarray(v, dtype=float) for v in values]
        if not np.isscalar(dt):
            raise ValueError("the sample spacing must be a scalar")
        spacing = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("finite real arrays and a scalar spacing required") from exc
    if formation.ndim != 1 or formation.size < 2:
        raise ValueError("r_form must be one-dimensional with at least two samples")
    n = formation.size
    mask = np.asarray(window_mask)
    if density.shape != (n,) or merger.shape != (n,) or mask.shape != (n,):
        raise ValueError("densities, histories and the mask must match the time grid")
    if mask.dtype != np.bool_:
        raise ValueError("window_mask must be a boolean array")
    if not all(np.all(np.isfinite(v)) for v in (formation, density, merger)):
        raise ValueError("all arrays must be finite")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("finite positive spacing required")
    if not np.any(mask):
        raise ValueError("window_mask must select at least one sample")
    if np.any(merger[mask] == 0.0):
        raise ValueError("r_merge must be nonzero on every selected sample")
    reconvolved = np.fft.irfft(np.fft.rfft(formation, n=2*n)
                               * np.fft.rfft(density * spacing, n=2*n), n=2*n)[:n]
    error = float(np.max(np.abs(reconvolved[mask] - merger[mask]) / np.abs(merger[mask])))
    if not np.isfinite(error):
        raise ValueError("the reconvolution diagnostic must be finite")
    return error

import numpy as np

def run_formation_response_pipeline(pivot_rates: np.ndarray, r_asym: float,
                                               n_samples: int, amplitudes: np.ndarray,
                                               weights: np.ndarray, means: np.ndarray,
                                               covariances: np.ndarray,
                                               required_probability: float = 0.5,
                                               alpha_steep: float = -1.73,
                                               alpha_shallow: float = -0.99,
                                               kappa: float = 3.2,
                                               max_reconvolution_error: float = 0.1) -> np.ndarray:
    values=(pivot_rates,amplitudes,weights,means,covariances)
    if any(not np.isrealobj(v) for v in values):
        raise ValueError("all rate and posterior arrays must be real")
    try:
        rates,amp,weight,mu,cov=[np.asarray(v,dtype=float) for v in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("all arrays must contain finite real values") from exc
    if rates.ndim!=1 or rates.size==0 or not np.all(np.isfinite(rates)) or np.any(rates<=0):
        raise ValueError("nonempty positive finite pivot rates required")
    if weight.ndim!=1 or weight.size==0 or mu.shape!=(weight.size,3) or cov.shape!=(weight.size,3,3):
        raise ValueError("posterior weight, mean and covariance shapes are inconsistent")
    if not all(np.all(np.isfinite(v)) for v in (weight,mu,cov)) or np.any(weight<0) or not np.any(weight>0) or np.any(np.abs(mu)>1):
        raise ValueError("nonnegative nonzero weights and means inside [-1,1]^3 required")
    for matrix in cov:
        if not np.allclose(matrix,matrix.T,rtol=0,atol=1e-12):
            raise ValueError("posterior covariances must be symmetric")
        eigenvalues=np.linalg.eigvalsh((matrix+matrix.T)/2)
        if eigenvalues[0]<.1 or eigenvalues[-1]>2:
            raise ValueError("posterior covariance eigenvalues must lie in [0.1,2]")
    if any(not np.isscalar(v) or not np.isrealobj(v) for v in (required_probability,alpha_steep,alpha_shallow)):
        raise ValueError("threshold and delay indices must be real scalars")
    try:
        target,steep,shallow=map(float,(required_probability,alpha_steep,alpha_shallow))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("threshold and delay indices must be finite real scalars") from exc
    if not np.all(np.isfinite([target,steep,shallow])) or not .05<=target<=.95 or not steep<shallow:
        raise ValueError("threshold in [0.05,0.95] and ordered finite delay indices required")
    if not np.isscalar(max_reconvolution_error) or not np.isrealobj(max_reconvolution_error):
        raise ValueError("the reconvolution bound must be a real scalar")
    try:
        reconvolution_bound=float(max_reconvolution_error)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("the reconvolution bound must be a finite real scalar") from exc
    if not np.isfinite(reconvolution_bound) or reconvolution_bound<=0.0:
        raise ValueError("a finite positive reconvolution bound is required")
    t0 = compute_cosmic_age(0.0)
    grid = build_time_redshift_grid(n_samples)
    time, redshift = grid
    dt = float(time[1] - time[0])
    window = (redshift >= .1) & (redshift <= 1.5)
    if not np.any(window):
        raise ValueError("the physicality window must contain a sample")
    ps = delay_time_distribution(time, steep, .01, t0)
    pl = delay_time_distribution(time, shallow, .01, t0)
    families = [affine_merger_family(redshift, float(rate), r_asym, amp, kappa)
                for rate in rates]
    for family in families:
        for endpoint, density in ((0.0, ps), (1.0, pl)):
            reconstruction = wiener_formation_jet(family, ps, pl, endpoint, dt)[0, 0]
            if reconvolution_error(reconstruction, density, family[0], dt,
                                           window) > reconvolution_bound:
                raise ValueError("the regularized inverse fails endpoint reconvolution certification")
    joint_window = np.tile(window, rates.size)
    lower_box, upper_box = -np.ones(3), np.ones(3)
    box = np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
    box_jet = np.stack((box, np.zeros_like(box), np.zeros_like(box)))
    prior = np.array([gaussian_polytope_moment_jet(box_jet, center, matrix)[0]
                      for center, matrix in zip(mu, cov)])

    def _posterior_response_at_fraction(fraction, response=False):
        formation = np.concatenate([wiener_formation_jet(family, ps, pl, fraction, dt)
                                    for family in families], axis=2)
        if not response:
            formation = np.stack((formation[0], np.zeros_like(formation[0]), np.zeros_like(formation[0])))
        vertices = physicality_vertex_jet(formation, joint_window, lower_box, upper_box)
        if vertices.shape[1] == 0:
            accepted = np.zeros((weight.size,3,10))
        elif vertices.shape[1] == 8 and np.all(vertices[1:] == 0.0) and np.allclose(
                vertices[0][np.lexsort((vertices[0,:,2],vertices[0,:,1],vertices[0,:,0]))],
                box, rtol=0.0, atol=1e-12):
            accepted = np.zeros((weight.size,3,10))
            accepted[:,0] = prior
        else:
            accepted = np.array([gaussian_polytope_moment_jet(vertices, center, matrix)
                                 for center, matrix in zip(mu,cov)])
        return condition_gaussian_mixture_jet(accepted, prior, weight)

    lower_summary = _posterior_response_at_fraction(0.0)
    if lower_summary[0,0] <= target:
        raise ValueError("the all-steep endpoint must exceed the required probability")
    upper_summary = _posterior_response_at_fraction(1.0)
    result = np.zeros((3,11))
    if upper_summary[0,0] >= target:
        result[0] = np.r_[1.0, upper_summary[0]]
        return result
    lower, upper = 0.0, 1.0
    while upper - lower > 1e-11:
        middle = (lower + upper) / 2.0
        summary = _posterior_response_at_fraction(middle)
        if summary[0,0] >= target:
            lower, lower_summary = middle, summary
        else:
            upper = middle
    result[0] = np.r_[lower, lower_summary[0]]
    if np.all(amp == 0.0):
        return result
    summary = _posterior_response_at_fraction(lower, response=True)
    slope, curvature = summary[1,0], summary[2,0]
    if not np.isfinite(slope) or slope >= 0.0:
        raise ValueError("the smooth interior critical fraction requires a negative probability slope")
    inverse_slope = 1.0 / slope
    inverse_curvature = -curvature / slope**3
    result[0,1:] = summary[0]
    result[1] = np.r_[inverse_slope, summary[1]*inverse_slope]
    result[2] = np.r_[inverse_curvature,
                      summary[2]*inverse_slope**2 + summary[1]*inverse_curvature]
    if not np.all(np.isfinite(result)):
        raise ValueError("the threshold response must be finite")
    return result
SCICODE_GOLD_EOF
