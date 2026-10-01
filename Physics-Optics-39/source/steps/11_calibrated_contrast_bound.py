"""
Evaluate the bound R for the parameterized benchmark. Compose and use every preceding public function; nested evaluations are permitted.

All input data, numerical conventions, output definition and parameterized variations are given in the main prompt and background.

Returns
-------
return out  # out : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def calibrated_contrast_bound(observed: np.ndarray=None, covariance: np.ndarray=None, noise_scale: float=1.0, strength: float=1.0, steps: int=12, relax: float=0.4, quantile: float=1.645) -> float:
    """Evaluate the benchmark's lower normal-equivalent bound.

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
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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

def _oracle_calibrated_contrast_bound(observed: np.ndarray=None, covariance: np.ndarray=None, noise_scale: float=1.0, strength: float=1.0, steps: int=12, relax: float=0.4, quantile: float=1.645) -> float:
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
        rays = _oracle_spectral_rays(cal, f, g, e, b)
        (obj, u) = (rays[:, 0], rays[:, 1:])
        p = _oracle_diffraction_pressure(obj, e)
        detector = _oracle_ray_push(obj + p, u).sum(axis=0)
        return np.mean(detector[None] * weights, axis=(1, 2))
    solution = root(lambda theta: calibration(theta) - z, np.array([0.0, 0.3, -0.05]), tol=1e-11)
    theta = solution.x
    if np.max(np.abs(calibration(theta) - z)) > 2e-12 or np.any(theta < [-0.8, -0.2, -0.2]) or np.any(theta > [0.3, 0.9, 0.1]):
        raise ValueError('Calibration root is outside the declared branch or unresolved')

    def response(theta):
        (f, g, e, b) = _hard_spectrum(theta)
        rays = _oracle_spectral_rays(truth, f, g, e, b)
        (obj, u) = (rays[:, 0], rays[:, 1:])
        pressure = _oracle_diffraction_pressure(obj, e)
        measured = _oracle_ray_push(obj + pressure, u).sum(axis=0)
        a0 = np.sum(f * e * b / 2)
        initial = -np.log(np.maximum(_oracle_energy_inverse(measured[None], np.array([a0]))[0], 1e-12))
        contrasts = []
        for mapped in (True, False):
            depth = initial.copy()
            for _ in range(steps):
                rays = _oracle_spectral_rays(depth, f, g, e, b)
                (obj, u) = (rays[:, 0], rays[:, 1:])
                pressure = _oracle_diffraction_pressure(obj, e)
                pred = _oracle_ray_push(obj + pressure, u) if mapped else _oracle_local_transport(obj, u) + pressure
                split = _oracle_spectral_residual(measured, pred, 1e-12)
                back = _oracle_ray_adjoint(split, u) if mapped else split
                update = obj + relax * _oracle_energy_inverse(back, e * b / 2)
                depth = _oracle_thickness_projection(update, f, g, 1e-12)[0]
            contrasts.append(np.mean(depth * w) / np.mean(w * w))
        value = 100 * (contrasts[0] - contrasts[1]) / (np.mean(truth * w) / np.mean(w * w))
        return np.array([value])
    spacing = np.array([5e-05, 5e-05, 5e-06])
    (c, ax, mix) = _hard_collect(calibration, theta, spacing)
    calibration_jet = _hard_sampled_jet(c, ax, mix, spacing)
    (c, ax, mix) = _hard_collect(response, theta, spacing)
    response_jet = _hard_sampled_jet(c, ax, mix, spacing)[0]
    pulled = _oracle_inverse_calibration_jet(calibration_jet[:, 1:4], calibration_jet[:, 4:].reshape(3, 3, 3), response_jet[1:4], response_jet[4:].reshape(3, 3))
    moments = _oracle_gaussian_quadratic_bound(response_jet[0], pulled[0], pulled[1:], cov * noise_scale ** 2, quantile)
    certificate = np.array([*theta, response_jet[0], moments[3], moments[1], moments[2]])
    return float(moments[2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'canonical_correlated_uncertainty',
      'setup': 'import numpy as np',
      'call': 'calibrated_contrast_bound()',
      'gold_call': '_oracle_calibrated_contrast_bound()',
      'tol': 0.0005},
     {'name': 'deterministic_calibration_limit',
      'setup': 'import numpy as np',
      'call': 'calibrated_contrast_bound(noise_scale=0)',
      'gold_call': '_oracle_calibrated_contrast_bound(noise_scale=0)',
      'tol': 0.0005},
     {'name': 'reduced_contrast_finite_iteration',
      'setup': 'import numpy as np',
      'call': 'calibrated_contrast_bound(strength=.65,steps=5,relax=.3)',
      'gold_call': '_oracle_calibrated_contrast_bound(strength=.65,steps=5,relax=.3)',
      'tol': 0.0005},
     {'name': 'rank_one_calibration_noise',
      'setup': 'import numpy as np',
      'call': 'calibrated_contrast_bound(covariance=1e-9*np.outer([1.,-.4,.7],[1.,-.4,.7]))',
      'gold_call': '_oracle_calibrated_contrast_bound(covariance=1e-9*np.outer([1.,-.4,.7],[1.,-.4,.7]))',
      'tol': 0.0005}]
