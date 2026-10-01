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

def _epr_d(a, axis):
    return (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis)) / 2.0

def spectral_rays(tau: np.ndarray, fractions: np.ndarray, attenuation: np.ndarray, eta: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Construct single-material spectral intensities and dimensionless ray shifts.

    Parameters
    ----------
    tau : float ndarray, shape (ny,nx)
        Finite reference optical depth (dimensionless), ny,nx >= 3.
    fractions : float ndarray, shape (S,)
        Strictly positive incident line fractions summing to one.
    attenuation : float ndarray, shape (S,)
        Positive absorption coefficients relative to the reference coefficient.
    eta : float ndarray, shape (S,)
        Nonnegative L/(k_s*p**2), in squared-pixel units normalized by p**2.
    db : float ndarray, shape (S,)
        Nonnegative delta_s/beta_s. Phase is db_s*log(I_s/f_s)/2.

    Returns
    -------
    out : float ndarray, shape (S,3,ny,nx)
        Axis 1 is [intensity, y displacement, x displacement]; intensity is
        relative to incident total flux, displacements are in pixels. Spatial
        derivatives are centered periodic first differences on unit pitch.

    Raises
    ------
    ValueError
        If tau is not two-dimensional or the spectral vector lengths differ.
    """
    t = np.asarray(tau, dtype=float)
    (f, g, e, b) = (np.asarray(v, dtype=float) for v in (fractions, attenuation, eta, db))
    if t.ndim != 2 or not f.ndim == g.ndim == e.ndim == b.ndim == 1 or (not f.shape == g.shape == e.shape == b.shape):
        raise ValueError('Expected a two-dimensional depth and equal-length spectral vectors')
    intensity = f[:, None, None] * np.exp(-g[:, None, None] * t)
    phase = -0.5 * (b * g)[:, None, None] * t
    return np.stack((intensity, e[:, None, None] * _epr_d(phase, -2), e[:, None, None] * _epr_d(phase, -1)), axis=1)

import numpy as np

def _epr_d(a, axis):
    return (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis)) / 2.0

def local_transport(intensity: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Evaluate the local second-order WKB0 intensity closure.

    Parameters
    ----------
    intensity : float ndarray, shape (S,ny,nx)
        Positive object-plane intensities, normalized to total incident flux.
    shifts : float ndarray, shape (S,2,ny,nx)
        Per-line dimensionless displacements ordered [y,x].

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Local WKB0 detector predictions in incident-flux units, through
        quadratic order in shifts. Every derivative is the centered periodic
        first difference; second derivatives compose that same operator.
        This output excludes the separately evaluated pressure correction.

    Raises
    ------
    ValueError
        If intensity and shifts do not have the documented compatible shapes.
    """
    i = np.asarray(intensity, dtype=float)
    if i.ndim != 3 or np.shape(shifts) != (i.shape[0], 2, *i.shape[1:]):
        raise ValueError('Expected spectral intensities and [y,x] shifts on the same grid')
    (v, u) = (np.asarray(shifts, dtype=float)[:, 0], np.asarray(shifts, dtype=float)[:, 1])
    return i - _epr_d(i * v, -2) - _epr_d(i * u, -1) + 0.5 * _epr_d(_epr_d(i * v * v, -2), -2) + _epr_d(_epr_d(i * v * u, -2), -1) + 0.5 * _epr_d(_epr_d(i * u * u, -1), -1)

import numpy as np

def _epr_d(a, axis):
    return (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis)) / 2.0

def _epr_lap(a):
    return sum((np.roll(a, -1, axis=j) + np.roll(a, 1, axis=j) - 2 * a for j in (-2, -1)))

def diffraction_pressure(intensity: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Evaluate the source-plane leading WKB1 diffraction-pressure correction.

    Parameters
    ----------
    intensity : float ndarray, shape (S,ny,nx)
        Strictly positive spectral object intensities in incident-flux units.
    eta : float ndarray, shape (S,)
        Nonnegative dimensionless propagation parameters L/(k_s*p**2).

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Signed additive O(eta**2) intensity correction. Q is the nearest-
        neighbor five-point periodic Laplacian of sqrt(I), divided by sqrt(I).
        Gradient and divergence use centered periodic first differences.
        The result is in incident-flux units on the source grid.

    Raises
    ------
    ValueError
        If intensity is not a positive three-dimensional array or eta length differs.
    """
    i = np.asarray(intensity, dtype=float)
    if i.ndim != 3 or np.shape(eta) != (i.shape[0],) or np.any(i <= 0):
        raise ValueError('Pressure requires positive spectral intensities and matching eta')
    root = np.sqrt(i)
    q = _epr_lap(root) / root
    div = _epr_d(i * _epr_d(q, -2), -2) + _epr_d(i * _epr_d(q, -1), -1)
    return -0.25 * np.asarray(eta, dtype=float)[:, None, None] ** 2 * div

import numpy as np

def _epr_stencil(shifts):
    (s, _, ny, nx) = shifts.shape
    (y, x) = np.indices((ny, nx))
    (yy, xx) = (y + shifts[:, 0], x + shifts[:, 1])
    (iy, ix) = (np.floor(yy).astype(int), np.floor(xx).astype(int))
    (fy, fx) = (yy - iy, xx - ix)
    return [(np.arange(s)[:, None, None], (iy + dy) % ny, (ix + dx) % nx, (fy if dy else 1 - fy) * (fx if dx else 1 - fx)) for (dy, dx) in ((0, 0), (0, 1), (1, 0), (1, 1))]

def ray_push(source: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Conservatively redistribute source mass along the finite eikonal map.

    Parameters
    ----------
    source : float ndarray, shape (S,ny,nx)
        Signed spectral source masses in incident-flux units. Signed values
        permit transporting an additive diffraction-pressure correction.
    shifts : float ndarray, shape (S,2,ny,nx)
        Finite mapped displacements in pixels, ordered [y,x].

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Detector masses, accumulated with tensor-product linear weights at
        floor(y+uy), floor(x+ux) and their upper neighbors, modulo ny,nx.
        Repeated target indices accumulate. Units match source.

    Raises
    ------
    ValueError
        If source and shifts do not have the documented compatible shapes.
    """
    (a, u) = (np.asarray(source, dtype=float), np.asarray(shifts, dtype=float))
    if a.ndim != 3 or u.shape != (a.shape[0], 2, *a.shape[1:]):
        raise ValueError('Source and displacement shapes are incompatible')
    out = np.zeros_like(a)
    for (s, y, x, w) in _epr_stencil(u):
        np.add.at(out, (s, y, x), a * w)
    return out

import numpy as np

def _epr_stencil(shifts):
    (s, _, ny, nx) = shifts.shape
    (y, x) = np.indices((ny, nx))
    (yy, xx) = (y + shifts[:, 0], x + shifts[:, 1])
    (iy, ix) = (np.floor(yy).astype(int), np.floor(xx).astype(int))
    (fy, fx) = (yy - iy, xx - ix)
    return [(np.arange(s)[:, None, None], (iy + dy) % ny, (ix + dx) % nx, (fy if dy else 1 - fy) * (fx if dx else 1 - fx)) for (dy, dx) in ((0, 0), (0, 1), (1, 0), (1, 1))]

def ray_adjoint(detector: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Apply the exact Euclidean transpose of the frozen bilinear ray map.

    Parameters
    ----------
    detector : float ndarray, shape (S,ny,nx)
        Signed detector residuals in incident-flux units.
    shifts : float ndarray, shape (S,2,ny,nx)
        Forward source-to-detector displacements in pixels, ordered [y,x].

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Source-grid residuals in incident-flux units, using the periodic
        tensor-product linear map and the Euclidean discrete inner product.

    Raises
    ------
    ValueError
        If detector and shifts do not have the documented compatible shapes.
    """
    (r, u) = (np.asarray(detector, dtype=float), np.asarray(shifts, dtype=float))
    if r.ndim != 3 or u.shape != (r.shape[0], 2, *r.shape[1:]):
        raise ValueError('Detector and displacement shapes are incompatible')
    out = np.zeros_like(r)
    for (s, y, x, w) in _epr_stencil(u):
        out += w * r[s, y, x]
    return out

import numpy as np



def spectral_residual(measured: np.ndarray, predicted: np.ndarray, weight_floor: float) -> np.ndarray:
    """Allocate an incoherent detector mismatch using positive predicted flux.

    Parameters
    ----------
    measured : float ndarray, shape (ny,nx)
        Positive total measured intensity in incident-flux units.
    predicted : float ndarray, shape (S,ny,nx)
        Signed per-line forward predictions in the same units.
    weight_floor : float
        Strictly positive intensity added once to the positive-flux denominator.

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Per-line detector residuals in incident-flux units, using the measured
        minus incoherent predicted total and normalized positive predictions.

    Raises
    ------
    ValueError
        If the array shapes are incompatible or weight_floor is not positive.
    """
    p = np.asarray(predicted, dtype=float)
    if p.ndim != 3 or np.shape(measured) != p.shape[1:] or weight_floor <= 0:
        raise ValueError('Expected matching detector grids and a positive denominator floor')
    positive = np.maximum(p, 0)
    return positive / (positive.sum(axis=0) + weight_floor) * (measured - p.sum(axis=0))

import numpy as np



def energy_inverse(residual: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Apply the energy-resolved FFT-diagonal homogeneous-object preconditioner.

    Parameters
    ----------
    residual : float ndarray, shape (S,ny,nx)
        Signed source residuals in incident-flux units.
    coefficients : float ndarray, shape (S,)
        Nonnegative a_s=eta_s*db_s/2, measured in pixel squared units.

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Real filtered residuals, same units. The multiplier is the inverse of
        1+a_s*(qx**2+qy**2), qj=2*pi*fftfreq(nj), with NumPy FFT normalization.

    Raises
    ------
    ValueError
        If residual is not three-dimensional, coefficients have the wrong length,
        or a coefficient is negative.
    """
    r = np.asarray(residual, dtype=float)
    if r.ndim != 3 or np.shape(coefficients) != (r.shape[0],) or np.any(np.asarray(coefficients) < 0):
        raise ValueError('Expected a spectral residual and matching nonnegative coefficients')
    (ny, nx) = r.shape[-2:]
    q2 = (2 * np.pi * np.fft.fftfreq(ny))[:, None] ** 2 + (2 * np.pi * np.fft.fftfreq(nx))[None, :] ** 2
    return np.fft.ifft2(np.fft.fft2(r) / (1 + np.asarray(coefficients)[:, None, None] * q2)).real

import numpy as np



def thickness_projection(components: np.ndarray, fractions: np.ndarray, attenuation: np.ndarray, intensity_floor: float) -> np.ndarray:
    """Project updated spectral components to one fraction-weighted optical depth.

    Parameters
    ----------
    components : float ndarray, shape (S,ny,nx)
        Finite updated spectral intensities; signed inputs are allowed.
    fractions : float ndarray, shape (S,)
        Positive incident spectral fractions summing to one.
    attenuation : float ndarray, shape (S,)
        Positive absorption coefficients relative to the reference coefficient.
    intensity_floor : float
        Positive lower bound, less than one, on each transmission I_s/f_s.

    Returns
    -------
    out : float ndarray, shape (S+1,ny,nx)
        Row 0 is dimensionless optical depth: the f-weighted mean of
        -log(max(I_s/f_s,intensity_floor))/attenuation_s. Rows 1..S are
        f_s*exp(-attenuation_s*out[0]) in incident-flux units. Transmission
        may exceed one, corresponding to a negative reconstructed depth.

    Raises
    ------
    ValueError
        If component/vector shapes disagree, a fraction or attenuation is nonpositive,
        or intensity_floor is outside (0,1).
    """
    (f, g) = (np.asarray(fractions), np.asarray(attenuation))
    if np.ndim(components) != 3 or f.shape != (np.shape(components)[0],) or g.shape != f.shape or np.any(f <= 0) or np.any(g <= 0) or (not 0 < intensity_floor < 1):
        raise ValueError('Expected matching positive spectral parameters and a transmission floor in (0,1)')
    ts = np.maximum(np.asarray(components) / f[:, None, None], intensity_floor)
    tau = np.sum(-f[:, None, None] * np.log(ts) / g[:, None, None], axis=0) / f.sum()
    return np.concatenate((tau[None], f[:, None, None] * np.exp(-g[:, None, None] * tau)))

import numpy as np
def inverse_calibration_jet(calibration_jacobian: np.ndarray, calibration_hessians: np.ndarray, response_gradient: np.ndarray, response_hessian: np.ndarray) -> np.ndarray:
    """Transform the response jet through a locally invertible calibration map.

    Parameters
    ----------
    calibration_jacobian : float ndarray, shape (n,n)
        Nonsingular dF_i/dtheta_j; rows are measured observables.
    calibration_hessians : float ndarray, shape (n,n,n)
        Entry [i,j,k] is d2F_i/(dtheta_j*dtheta_k).
    response_gradient : float ndarray, shape (n,)
        Gradient of the scalar response B with respect to theta.
    response_hessian : float ndarray, shape (n,n)
        Symmetric Hessian of B with respect to theta.

    Returns
    -------
    out : float ndarray, shape (n+1,n)
        Row zero is the gradient of B composed with the local inverse of F;
        the remaining n rows are its Hessian, both in measurement coordinates.
        Units are response per measurement, and response per measurement squared.
    """
    (j, f2, g, b2) = map(np.asarray, (calibration_jacobian, calibration_hessians, response_gradient, response_hessian))
    n = g.size
    if j.shape != (n, n) or f2.shape != (n, n, n) or b2.shape != (n, n):
        raise ValueError('Incompatible calibration and response jets')
    a = np.linalg.solve(j.T, g)
    reduced = b2 - np.einsum('i,ijk->jk', a, f2)
    right = np.linalg.solve(j.T, reduced.T).T
    hessian = np.linalg.solve(j.T, right)
    return np.vstack((a, (hessian + hessian.T) / 2))

import numpy as np
def gaussian_quadratic_bound(value: float, gradient: np.ndarray, hessian: np.ndarray, covariance: np.ndarray, quantile: float) -> np.ndarray:
    """Evaluate Gaussian moments and a lower normal-equivalent bound of a quadratic.

    Parameters
    ----------
    value : float
        Response at zero perturbation, measured in percentage points.
    gradient : float ndarray, shape (n,)
        Linear coefficients with respect to calibration measurements.
    hessian : float ndarray, shape (n,n)
        Symmetric second derivatives with respect to those measurements.
    covariance : float ndarray, shape (n,n)
        Symmetric positive-semidefinite covariance of zero-mean Gaussian noise.
    quantile : float
        Nonnegative normal-equivalent standard-deviation multiplier.

    Returns
    -------
    out : float ndarray, shape (4,)
        Ordered [mean, standard deviation, lower bound, curvature bias].
        All four entries are in percentage points. Moments are those of the
        complete quadratic Taylor polynomial, including its quadratic variance.
    """
    (g, h, s) = map(np.asarray, (gradient, hessian, covariance))
    n = g.size
    if h.shape != (n, n) or s.shape != (n, n) or quantile < 0:
        raise ValueError('Incompatible quadratic uncertainty inputs')
    bias = 0.5 * np.trace(h @ s)
    variance = g @ s @ g + 0.5 * np.trace(h @ s @ h @ s)
    sd = np.sqrt(max(0.0, float(variance)))
    mean = value + bias
    return np.array([mean, sd, mean - quantile * sd, bias])

import numpy as np
from scipy.optimize import root

def _hard_grid(ny, nx, calibration=False, strength=1.0):
    (y, x) = np.indices((ny, nx))
    (X, Y) = (2 * np.pi * x / nx, 2 * np.pi * y / ny)
    if calibration:
        depth = 0.93 + 0.21 * np.cos(X) + 0.14 * np.sin(Y) + 0.11 * np.cos(X + Y) + 0.075 * np.sin(2 * X - Y)
    else:
        depth = 0.72 + strength * (0.22 * np.cos(X) + 0.13 * np.sin(Y) + 0.09 * np.cos(X + Y) + 0.04 * np.sin(2 * X - Y))
    return (depth, X, Y)

def _hard_spectrum(theta):
    weights = np.exp(np.array([theta[0], theta[1], 0.0]))
    return (weights / weights.sum(), np.array([1.6, 1.0, 0.6]), np.exp(theta[2]) * np.array([4.2, 2.5, 1.1]), np.array([16.0, 22.0, 30.0]))

def _hard_collect(function, theta, spacing):
    n = len(theta)
    axes = np.diag(spacing)
    c = np.asarray(function(theta))
    axial = np.array([[function(theta - axes[i]), function(theta + axes[i])] for i in range(n)])
    mixed = np.array([[function(theta + si * axes[i] + sj * axes[j]) for (si, sj) in ((-1, -1), (-1, 1), (1, -1), (1, 1))] for i in range(n) for j in range(i + 1, n)])
    return (c, axial, mixed)

def _hard_sampled_jet(center, axial, mixed, spacing):
    """Recover a second-order multivariate jet from a centered sampling stencil.

    Parameters
    ----------
    center : float ndarray, shape (m,)
        Values of m dimensionless observables at the expansion point.
    axial : float ndarray, shape (n,2,m)
        Values at theta-h_i*e_i and theta+h_i*e_i, in that order.
    mixed : float ndarray, shape (n*(n-1)//2,4,m)
        Pairs i<j in lexicographic order; signs (--,-+,+-,++).
    spacing : float ndarray, shape (n,)
        Positive coordinate increments, in the units of theta.

    Returns
    -------
    out : float ndarray, shape (m,1+n+n*n)
        Each row packs [value, gradient, Hessian flattened in C order].
        Three-point centered axial and four-corner mixed differences define
        the derivatives. Gradient/Hessian units include inverse theta units.
    """
    (c, a, v, h) = map(np.asarray, (center, axial, mixed, spacing))
    (m, n) = (c.size, h.size)
    if c.shape != (m,) or a.shape != (n, 2, m) or v.shape != (n * (n - 1) // 2, 4, m) or np.any(h <= 0):
        raise ValueError('Incompatible centered-jet stencil')
    gradient = ((a[:, 1] - a[:, 0]) / (2 * h[:, None])).T
    hessian = np.zeros((m, n, n))
    for i in range(n):
        hessian[:, i, i] = (a[i, 1] - 2 * c + a[i, 0]) / h[i] ** 2
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            hessian[:, i, j] = hessian[:, j, i] = (v[k, 3] - v[k, 2] - v[k, 1] + v[k, 0]) / (4 * h[i] * h[j])
            k += 1
    return np.concatenate((c[:, None], gradient, hessian.reshape(m, n * n)), axis=1)

def calibrated_contrast_bound(observed: np.ndarray=None, covariance: np.ndarray=None, noise_scale: float=1.0, strength: float=1.0, steps: int=12, relax: float=0.4, quantile: float=1.645) -> float:
    """Combine paper-grounded transport, nonlinear calibration and uncertainty.

    Parameters
    ----------
    observed : float ndarray, shape (3,), or None
        Calibration moments in incident-total-flux units; None selects the
        three fixed data values stated in the problem. The root is interior
        to the specified calibration box and has nonsingular Jacobian.
    covariance : float ndarray, shape (3,3), or None
        Positive-semidefinite calibration-measurement covariance; None selects
        the stated correlated matrix, in squared incident-total-flux units.
    noise_scale : float
        Nonnegative multiplier of standard deviations, at most 2.
    strength : float
        Test-phantom modulation multiplier in [0.2,1.3]; calibration is unchanged.
    steps : int
        Retrieval iteration count, 1..20, with simultaneous line corrections.
    relax : float
        Retrieval relaxation in (0,0.6].
    quantile : float
        Nonnegative normal-equivalent lower-bound multiplier.

    Returns
    -------
    out : float
        The lower normal-equivalent bound of the quadratic approximation to
        B(F^{-1}(z+epsilon)), in percentage points, for the declared covariance.
        The public implementation must call and use every preceding public
        function, including through its nested calibration/retrieval evaluations.
    """
    if not (0 <= noise_scale <= 2 and 0.2 <= strength <= 1.3 and (0 < relax <= 0.6) and (quantile >= 0)) or not isinstance(steps, (int, np.integer)) or (not 1 <= steps <= 20):
        raise ValueError('Out-of-domain retrieval/uncertainty settings')
    z = np.array([0.41517240210080686, -0.06127696643781137, -0.05212904606788967]) if observed is None else np.asarray(observed, dtype=float)
    cov = 3e-09 * np.array([[4.0, 0.8, -0.6], [0.8, 1.0, 0.25], [-0.6, 0.25, 0.64]]) if covariance is None else np.asarray(covariance, dtype=float)
    if z.shape != (3,) or cov.shape != (3, 3):
        raise ValueError('Three calibration moments and a 3 by 3 covariance are required')
    (cal, Xc, Yc) = _hard_grid(35, 41, True)
    (truth, X, Y) = _hard_grid(35, 41, strength=strength)
    weights = np.array([np.ones_like(cal), np.cos(Xc), np.sin(2 * Xc - Yc)])
    w = np.cos(X + Y) + 0.3 * np.sin(2 * X - Y)

    def calibration(theta):
        (f, g, e, b) = _hard_spectrum(theta)
        rays = spectral_rays(cal, f, g, e, b)
        (obj, u) = (rays[:, 0], rays[:, 1:])
        p = diffraction_pressure(obj, e)
        detector = ray_push(obj + p, u).sum(axis=0)
        return np.mean(detector[None] * weights, axis=(1, 2))
    solution = root(lambda theta: calibration(theta) - z, np.array([0.0, 0.3, -0.05]), tol=1e-11)
    theta = solution.x
    if np.max(np.abs(calibration(theta) - z)) > 2e-12 or np.any(theta < [-0.8, -0.2, -0.2]) or np.any(theta > [0.3, 0.9, 0.1]):
        raise ValueError('Calibration root is outside the declared branch or unresolved')

    def response(theta):
        (f, g, e, b) = _hard_spectrum(theta)
        rays = spectral_rays(truth, f, g, e, b)
        (obj, u) = (rays[:, 0], rays[:, 1:])
        pressure = diffraction_pressure(obj, e)
        measured = ray_push(obj + pressure, u).sum(axis=0)
        a0 = np.sum(f * e * b / 2)
        initial = -np.log(np.maximum(energy_inverse(measured[None], np.array([a0]))[0], 1e-12))
        contrasts = []
        for mapped in (True, False):
            depth = initial.copy()
            for _ in range(steps):
                rays = spectral_rays(depth, f, g, e, b)
                (obj, u) = (rays[:, 0], rays[:, 1:])
                pressure = diffraction_pressure(obj, e)
                pred = ray_push(obj + pressure, u) if mapped else local_transport(obj, u) + pressure
                split = spectral_residual(measured, pred, 1e-12)
                back = ray_adjoint(split, u) if mapped else split
                update = obj + relax * energy_inverse(back, e * b / 2)
                depth = thickness_projection(update, f, g, 1e-12)[0]
            contrasts.append(np.mean(depth * w) / np.mean(w * w))
        value = 100 * (contrasts[0] - contrasts[1]) / (np.mean(truth * w) / np.mean(w * w))
        return np.array([value])
    spacing = np.array([5e-05, 5e-05, 5e-06])
    (c, ax, mix) = _hard_collect(calibration, theta, spacing)
    calibration_jet = _hard_sampled_jet(c, ax, mix, spacing)
    (c, ax, mix) = _hard_collect(response, theta, spacing)
    response_jet = _hard_sampled_jet(c, ax, mix, spacing)[0]
    pulled = inverse_calibration_jet(calibration_jet[:, 1:4], calibration_jet[:, 4:].reshape(3, 3, 3), response_jet[1:4], response_jet[4:].reshape(3, 3))
    moments = gaussian_quadratic_bound(response_jet[0], pulled[0], pulled[1:], cov * noise_scale ** 2, quantile)
    certificate = np.array([*theta, response_jet[0], moments[3], moments[1], moments[2]])
    return float(moments[2])
SCICODE_GOLD_EOF
