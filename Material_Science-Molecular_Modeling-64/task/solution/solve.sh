#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def gap_block_statistics(block_gaps):
    """Estimate correlated lambda-window means and covariance of their mean."""
    import numpy as np
    values = np.asarray(block_gaps, dtype=float)
    if values.ndim != 2 or values.shape[0] < values.shape[1] + 1 or values.shape[1] < 3:
        raise ValueError("block_gaps must have shape (B,L), with L>=3 and B>=L+1")
    if not np.all(np.isfinite(values)):
        raise ValueError("block_gaps must be finite")
    means = np.mean(values, axis=0)
    covariance = np.cov(values, rowvar=False, ddof=1) / values.shape[0]
    covariance = np.atleast_2d(covariance).astype(float)
    if np.min(np.linalg.eigvalsh(covariance)) <= 1.0e-16:
        raise ValueError("covariance of the mean must be positive definite")
    return np.vstack((means, covariance)).astype(float)

def quadratic_gls_ti(lambdas, statistics):
    """Fit a quadratic energy-gap curve by GLS and integrate it from 0 to 1."""
    import numpy as np
    lam = np.asarray(lambdas, dtype=float)
    stats = np.asarray(statistics, dtype=float)
    if lam.ndim != 1 or lam.size < 3 or stats.shape != (lam.size + 1, lam.size):
        raise ValueError("statistics must have shape (L+1,L) for a length-L lambda vector")
    if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(stats)):
        raise ValueError("inputs must be finite")
    if np.any(lam < 0.0) or np.any(lam > 1.0) or len(np.unique(lam)) != lam.size:
        raise ValueError("lambda values must be distinct and lie in [0,1]")
    means = stats[0]
    covariance = stats[1:]
    if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
        raise ValueError("covariance must be positive definite")
    design = np.column_stack((np.ones(lam.size), lam, lam * lam))
    inverse = np.linalg.inv(covariance)
    normal = design.T @ inverse @ design
    coefficient_covariance = np.linalg.inv(normal)
    coefficients = coefficient_covariance @ design.T @ inverse @ means
    integral_weights = np.array([1.0, 0.5, 1.0 / 3.0], dtype=float)
    integral = float(integral_weights @ coefficients)
    standard_error = float(np.sqrt(integral_weights @ coefficient_covariance @ integral_weights))
    curvature = float(2.0 * coefficients[2])
    return np.array([integral, standard_error, curvature], dtype=float)

def alchemical_mass_correction(x_solute, temperature_k, mass_reference_u, mass_solute_u):
    """Return total, ideal-configurational, and kinetic-mass Helmholtz terms."""
    import math
    import numpy as np
    x = float(x_solute)
    temperature = float(temperature_k)
    mass_reference = float(mass_reference_u)
    mass_solute = float(mass_solute_u)
    if not all(math.isfinite(v) for v in (x, temperature, mass_reference, mass_solute)):
        raise ValueError("inputs must be finite")
    if x <= 0.0 or x > 1.0 or temperature <= 0.0 or mass_reference <= 0.0 or mass_solute <= 0.0:
        raise ValueError("require 0<x<=1 and positive temperature and masses")
    k_b_ev_per_k = 8.617333262145e-5
    ideal = 0.0 if x == 1.0 else k_b_ev_per_k * temperature * (
        x * math.log(x) + (1.0 - x) * math.log(1.0 - x)
    )
    kinetic = 1.5 * k_b_ev_per_k * temperature * x * math.log(mass_reference / mass_solute)
    return np.array([ideal + kinetic, ideal, kinetic], dtype=float)

def birch_murnaghan_pv(volume_solution_a3, volume_reference_a3,
                               pressure_target_gpa, bulk_modulus_gpa,
                               bulk_derivative, volume_zero_a3,
                               quadrature_order=64):
    """Evaluate the paper's EOS free-energy correction with Gauss-Legendre quadrature."""
    import math
    import numpy as np
    volume_solution = float(volume_solution_a3)
    volume_reference = float(volume_reference_a3)
    pressure_target = float(pressure_target_gpa)
    bulk_modulus = float(bulk_modulus_gpa)
    bulk_derivative = float(bulk_derivative)
    volume_zero = float(volume_zero_a3)
    order = int(quadrature_order)
    values = (volume_solution, volume_reference, pressure_target,
              bulk_modulus, bulk_derivative, volume_zero)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("inputs must be finite")
    if min(volume_solution, volume_reference, bulk_modulus, volume_zero) <= 0.0 or order < 8:
        raise ValueError("volumes and modulus must be positive and quadrature_order>=8")

    def pressure(volume):
        eta = (volume_zero / volume) ** (1.0 / 3.0)
        return (1.5 * bulk_modulus * (eta ** 7 - eta ** 5)
                * (1.0 + 0.75 * (bulk_derivative - 4.0) * (eta * eta - 1.0)))

    nodes, weights = np.polynomial.legendre.leggauss(order)
    mapped = 0.5 * (volume_solution - volume_reference) * nodes + 0.5 * (
        volume_solution + volume_reference
    )
    integral_gpa_a3 = 0.5 * (volume_solution - volume_reference) * float(
        weights @ np.array([pressure(v) for v in mapped], dtype=float)
    )
    gpa_a3_to_ev = 0.006241509074460763
    correction = (pressure_target * (volume_solution - volume_reference)
                  - integral_gpa_a3) * gpa_a3_to_ev
    endpoint_pressure = pressure(volume_solution)
    reference_residual = pressure(volume_reference) - pressure_target
    return np.array([correction, endpoint_pressure, reference_residual], dtype=float)

def mixing_curve_covariance(compositions, gibbs_certificates, temperature_k):
    """Construct excess-mixing responses and shared-endmember covariance."""
    import math
    import numpy as np
    x = np.asarray(compositions, dtype=float)
    certificates = np.asarray(gibbs_certificates, dtype=float)
    temperature = float(temperature_k)
    if x.ndim != 1 or x.size < 5 or certificates.shape != (x.size, 5):
        raise ValueError("need C>=5 compositions and Gibbs certificates of shape (C,5)")
    if not math.isfinite(temperature) or temperature <= 0.0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(certificates)):
        raise ValueError("inputs must be finite and temperature positive")
    if np.any(x <= 0.0) or np.any(x > 1.0) or np.any(np.diff(x) <= 0.0) or x[-1] != 1.0:
        raise ValueError("compositions must increase strictly in (0,1] and end at 1")
    if np.any(certificates[:, 1] <= 0.0):
        raise ValueError("all Gibbs standard errors must be positive")
    k_b_ev_per_k = 8.617333262145e-5
    interior = x[:-1]
    ideal = k_b_ev_per_k * temperature * (
        interior * np.log(interior) + (1.0 - interior) * np.log(1.0 - interior)
    )
    response = certificates[:-1, 0] - interior * certificates[-1, 0] - ideal
    variances = certificates[:, 1] ** 2
    covariance = np.diag(variances[:-1]) + variances[-1] * np.outer(interior, interior)
    return np.column_stack((interior, response, covariance)).astype(float)

def augment_finite_size_covariance(mixing_certificate,
                                           calibration_compositions,
                                           calibration_residuals_ev,
                                           sigma_grid_ev,
                                           length_grid,
                                           noise_sigma_ev):
    """Profile a finite-size GP kernel and transform it through endmember subtraction."""
    import math
    import numpy as np
    certificate = np.asarray(mixing_certificate, dtype=float)
    calibration_x = np.asarray(calibration_compositions, dtype=float)
    residuals = np.asarray(calibration_residuals_ev, dtype=float)
    sigma_grid = np.asarray(sigma_grid_ev, dtype=float)
    length_candidates = np.asarray(length_grid, dtype=float)
    noise = float(noise_sigma_ev)
    if certificate.ndim != 2 or certificate.shape[1] != certificate.shape[0] + 2:
        raise ValueError("mixing_certificate must have shape (N,N+2)")
    if calibration_x.ndim != 1 or residuals.shape != calibration_x.shape or calibration_x.size < 3:
        raise ValueError("finite-size calibration vectors must have one common length >=3")
    if sigma_grid.ndim != 1 or length_candidates.ndim != 1 or sigma_grid.size < 2 or length_candidates.size < 2:
        raise ValueError("both hyperparameter grids must contain at least two values")
    arrays = (certificate, calibration_x, residuals, sigma_grid, length_candidates)
    if any(not np.all(np.isfinite(value)) for value in arrays) or not math.isfinite(noise):
        raise ValueError("inputs must be finite")
    if (np.any(np.diff(calibration_x) <= 0.0) or np.any(calibration_x <= 0.0)
            or np.any(calibration_x > 1.0)):
        raise ValueError("calibration compositions must increase in (0,1]")
    if (np.any(sigma_grid <= 0.0) or np.any(np.diff(sigma_grid) <= 0.0)
            or np.any(length_candidates <= 0.0) or np.any(np.diff(length_candidates) <= 0.0)
            or noise <= 0.0):
        raise ValueError("hyperparameter grids and noise sigma must be positive")
    x = certificate[:, 0]
    if np.any(x <= 0.0) or np.any(x >= 1.0) or np.any(np.diff(x) <= 0.0):
        raise ValueError("certificate compositions must increase strictly in (0,1)")

    best = None
    distances = np.abs(calibration_x[:, None] - calibration_x[None, :])
    identity = np.eye(calibration_x.size)
    for sigma in sigma_grid:
        for length in length_candidates:
            covariance = sigma * sigma * np.exp(-distances / length) + noise * noise * identity
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0.0:
                continue
            objective = float(logdet + residuals @ np.linalg.solve(covariance, residuals))
            candidate = (objective, float(sigma), float(length))
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("no positive-definite finite-size candidate")
    _, selected_sigma, selected_length = best
    points = np.concatenate((x, np.array([1.0])))
    kernel = selected_sigma ** 2 * np.exp(
        -np.abs(points[:, None] - points[None, :]) / selected_length
    )
    added = (kernel[:-1, :-1]
             - x[:, None] * kernel[-1, :-1][None, :]
             - kernel[:-1, -1][:, None] * x[None, :]
             + np.outer(x, x) * kernel[-1, -1])
    augmented = certificate.copy()
    augmented[:, 2:] = certificate[:, 2:] + added
    return np.column_stack((
        augmented,
        np.full(x.size, selected_sigma),
        np.full(x.size, selected_length),
    )).astype(float)

def fit_redlich_kister(augmented_certificate, temperature_k,
                               bracket=(0.05, 0.49), grid_points=257):
    """Fit and screen Redlich-Kister orders with GLS, AICc, and spinodal feasibility."""
    import math
    import numpy as np
    from numpy.polynomial import Polynomial
    certificate = np.asarray(augmented_certificate, dtype=float)
    temperature = float(temperature_k)
    left, right = map(float, bracket)
    grid_points = int(grid_points)
    if certificate.ndim != 2 or certificate.shape[0] < 6 or certificate.shape[1] != certificate.shape[0] + 4:
        raise ValueError("augmented_certificate must have shape (N,N+4) with N>=6")
    if not np.all(np.isfinite(certificate)) or not math.isfinite(temperature):
        raise ValueError("inputs must be finite")
    if temperature <= 0.0 or not (0.0 < left < right < 1.0) or grid_points < 33 or grid_points % 2 == 0:
        raise ValueError("invalid temperature, bracket, or odd grid_points")
    n = certificate.shape[0]
    x = certificate[:, 0]
    response = certificate[:, 1]
    covariance = certificate[:, 2:2 + n]
    if np.any(x <= 0.0) or np.any(x >= 1.0) or np.any(np.diff(x) <= 0.0):
        raise ValueError("certificate compositions must increase strictly in (0,1)")
    if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
        raise ValueError("response covariance must be positive definite")
    inverse = np.linalg.inv(covariance)
    k_b_ev_per_k = 8.617333262145e-5
    base = Polynomial([0.0, 1.0, -1.0])
    zeta_poly = Polynomial([-1.0, 2.0])
    basis_polynomials = [base * zeta_poly ** order for order in range(4)]
    grid = np.linspace(left, right, grid_points)
    rows = []
    for count in (2, 3, 4):
        design = np.column_stack([poly(x) for poly in basis_polynomials[:count]])
        normal_inverse = np.linalg.inv(design.T @ inverse @ design)
        coefficients = normal_inverse @ design.T @ inverse @ response
        residual = response - design @ coefficients
        chi_square = float(residual @ inverse @ residual)
        denominator = n - count - 1
        if denominator <= 0:
            raise ValueError("too few compositions for AICc candidates")
        aicc = chi_square + 2.0 * count + 2.0 * count * (count + 1.0) / denominator

        second = k_b_ev_per_k * temperature * (1.0 / grid + 1.0 / (1.0 - grid))
        for coefficient, poly in zip(coefficients, basis_polynomials[:count]):
            second = second + coefficient * poly.deriv(2)(grid)
        crossings = np.flatnonzero(second[:-1] * second[1:] < 0.0)
        feasible = int(crossings.size == 1)
        root = 0.0
        if feasible:
            lo = float(grid[crossings[0]])
            hi = float(grid[crossings[0] + 1])
            f_lo = float(second[crossings[0]])
            for _ in range(100):
                midpoint = 0.5 * (lo + hi)
                f_mid = k_b_ev_per_k * temperature * (1.0 / midpoint + 1.0 / (1.0 - midpoint))
                f_mid += sum(float(c * poly.deriv(2)(midpoint))
                             for c, poly in zip(coefficients, basis_polynomials[:count]))
                if hi - lo <= 1.0e-12:
                    break
                if f_lo * f_mid <= 0.0:
                    hi = midpoint
                else:
                    lo = midpoint
                    f_lo = f_mid
            root = 0.5 * (lo + hi)
        padded_coefficients = np.zeros(4)
        padded_coefficients[:count] = coefficients
        padded_covariance = np.zeros((4, 4))
        padded_covariance[:count, :count] = normal_inverse
        rows.append(np.concatenate((
            np.array([float(count), aicc, float(feasible), root]),
            padded_coefficients,
            padded_covariance.ravel(),
        )))
    return np.asarray(rows, dtype=float)

def common_tangent_screen(rk_ensemble, temperature_k,
                                  left_interval=(0.01, 0.49),
                                  right_interval=(0.51, 0.99),
                                  starts_per_axis=9, tolerance=1.0e-10,
                                  sigma_radius=1.0):
    """Append nominal and covariance-robust common-tangent certificates."""
    import math
    import numpy as np
    from numpy.polynomial import Polynomial
    ensemble = np.asarray(rk_ensemble, dtype=float)
    temperature = float(temperature_k)
    la, lb = map(float, left_interval)
    ra, rb = map(float, right_interval)
    starts = int(starts_per_axis)
    tol = float(tolerance)
    radius = float(sigma_radius)
    if ensemble.shape != (3, 24) or not np.all(np.isfinite(ensemble)):
        raise ValueError("rk_ensemble must be a finite (3,24) array")
    if temperature <= 0.0 or not (0.0 < la < lb < ra < rb < 1.0):
        raise ValueError("invalid temperature or disjoint composition intervals")
    if starts < 3 or tol <= 0.0 or not math.isfinite(radius) or radius < 0.0:
        raise ValueError("starts_per_axis>=3, positive tolerance, and nonnegative sigma_radius are required")
    kb = 8.617333262145e-5
    base = Polynomial([0.0, 1.0, -1.0])
    zeta = Polynomial([-1.0, 2.0])
    basis = [base * zeta ** order for order in range(4)]
    output = []
    for row in ensemble:
        count = int(row[0])
        coeff = row[4:4 + count]
        covariance = row[8:24].reshape(4, 4)[:count, :count]
        polys = basis[:count]

        def values(x, active_coeff):
            g = kb * temperature * (x * np.log(x) + (1.0 - x) * np.log(1.0 - x))
            gp = kb * temperature * np.log(x / (1.0 - x))
            gpp = kb * temperature * (1.0 / x + 1.0 / (1.0 - x))
            for c, p in zip(active_coeff, polys):
                g += c * p(x); gp += c * p.deriv(1)(x); gpp += c * p.deriv(2)(x)
            return float(g), float(gp), float(gpp)

        def curvature_root(active_coeff):
            grid = np.linspace(0.05, 0.49, 257)
            second = kb * temperature * (1.0 / grid + 1.0 / (1.0 - grid))
            for c, p in zip(active_coeff, polys):
                second = second + c * p.deriv(2)(grid)
            crossings = np.flatnonzero(second[:-1] * second[1:] < 0.0)
            if crossings.size != 1:
                return None
            lo, hi = float(grid[crossings[0]]), float(grid[crossings[0] + 1])
            f_lo = float(second[crossings[0]])
            for _ in range(100):
                midpoint = 0.5 * (lo + hi)
                f_mid = values(midpoint, active_coeff)[2]
                if hi - lo <= 1.0e-12:
                    break
                if f_lo * f_mid <= 0.0:
                    hi = midpoint
                else:
                    lo, f_lo = midpoint, f_mid
            return 0.5 * (lo + hi)

        def newton(active_coeff, x0, y0):
            x, y = float(x0), float(y0)
            for _ in range(80):
                gx, gpx, gppx = values(x, active_coeff)
                gy, gpy, gppy = values(y, active_coeff)
                sec = (gy - gx) / (y - x)
                f = np.array([gpx - sec, gpy - sec])
                if np.max(np.abs(f)) <= tol:
                    break
                dsx = (sec - gpx) / (y - x)
                dsy = (gpy - sec) / (y - x)
                jac = np.array([[gppx - dsx, -dsy], [-dsx, gppy - dsy]])
                try:
                    step = np.linalg.solve(jac, f)
                except np.linalg.LinAlgError:
                    break
                accepted = False
                for power in range(14):
                    scale = 0.5 ** power
                    xn, yn = x - scale * step[0], y - scale * step[1]
                    if la <= xn <= lb and ra <= yn <= rb and xn < yn:
                        x, y = float(xn), float(yn); accepted = True; break
                if not accepted:
                    break
            gx, gpx, _ = values(x, active_coeff)
            gy, gpy, _ = values(y, active_coeff)
            sec = (gy - gx) / (y - x)
            residual = max(abs(gpx - sec), abs(gpy - sec))
            if residual <= 10.0 * tol:
                return (x, y, residual, sec)
            return None

        solutions = []
        if int(round(row[2])) == 1:
            for x0 in np.linspace(la, lb, starts):
                for y0 in np.linspace(ra, rb, starts):
                    solved = newton(coeff, x0, y0)
                    if solved is not None:
                        x_sol, y_sol, residual_sol, slope_sol = solved
                        solutions.append((x_sol, -y_sol, residual_sol, slope_sol))
        feasible = int(bool(solutions))
        xa = xb = slope = residual = width = 0.0
        if feasible:
            xa, neg_xb, residual, slope = min(solutions)
            xb = -neg_xb
            width = xb - xa
            if not (xa < row[3] < 0.5 < xb):
                feasible = 0
        robust = 0
        robust_alpha = robust_beta = robust_residual = robust_margin = max_root_shift = 0.0
        if feasible and np.min(np.linalg.eigvalsh(covariance)) > 0.0:
            cholesky = np.linalg.cholesky(covariance)
            sigma_points = [coeff.copy()]
            for column in range(count):
                sigma_points.append(coeff + radius * cholesky[:, column])
                sigma_points.append(coeff - radius * cholesky[:, column])
            certificates = [(float(row[3]), xa, xb, residual)]
            continuation = (xa, xb)
            robust = 1
            for perturbed in sigma_points[1:]:
                perturbed_root = curvature_root(perturbed)
                solved = None if perturbed_root is None else newton(
                    perturbed, continuation[0], continuation[1]
                )
                if solved is None:
                    robust = 0
                    break
                perturbed_alpha, perturbed_beta, perturbed_residual, _ = solved
                if not (perturbed_alpha < perturbed_root < 0.5 < perturbed_beta):
                    robust = 0
                    break
                certificates.append((perturbed_root, perturbed_alpha,
                                     perturbed_beta, perturbed_residual))
                continuation = (perturbed_alpha, perturbed_beta)
            if robust:
                roots = np.array([item[0] for item in certificates])
                alphas = np.array([item[1] for item in certificates])
                betas = np.array([item[2] for item in certificates])
                robust_alpha = float(np.max(alphas))
                robust_beta = float(np.min(betas))
                robust_residual = float(np.max([item[3] for item in certificates]))
                robust_margin = float(min(np.min(roots) - robust_alpha,
                                          robust_beta - np.max(roots)))
                max_root_shift = float(np.max(np.abs(roots - row[3])))
                if not (robust_margin > 0.0 and robust_alpha < 0.5 < robust_beta):
                    robust = 0
        if not robust:
            robust_alpha = robust_beta = robust_residual = robust_margin = max_root_shift = 0.0
        output.append(np.concatenate((
            row, np.array([feasible, xa, xb, slope, residual, width,
                           robust, robust_alpha, robust_beta, robust_residual,
                           robust_margin, max_root_shift])
        )))
    return np.asarray(output, dtype=float)

def spinodal_lower_confidence(rk_ensemble, temperature_k,
                                      z_score=1.645, delta_aicc=4.0):
    """Choose the most conservative uncertainty bound across supported feasible RK orders."""
    import math
    import numpy as np
    from numpy.polynomial import Polynomial
    ensemble = np.asarray(rk_ensemble, dtype=float)
    temperature = float(temperature_k)
    z_value = float(z_score)
    delta_limit = float(delta_aicc)
    if ensemble.shape != (3, 36):
        raise ValueError("rk_ensemble must have shape (3,36)")
    if not np.all(np.isfinite(ensemble[:, :3])) or not math.isfinite(temperature + z_value + delta_limit):
        raise ValueError("inputs must be finite")
    if temperature <= 0.0 or z_value < 0.0 or delta_limit < 0.0:
        raise ValueError("invalid temperature, z score, or AICc support width")
    counts = ensemble[:, 0].astype(int)
    if not np.array_equal(counts, np.array([2, 3, 4])) or np.any(ensemble[:, 2] != np.round(ensemble[:, 2])):
        raise ValueError("candidate rows must be ordered counts 2,3,4 with binary feasibility")
    feasible = (ensemble[:, 2].astype(bool) & ensemble[:, 24].astype(bool)
                & ensemble[:, 30].astype(bool))
    if not np.any(feasible):
        raise ValueError("no physically feasible Redlich-Kister candidate")
    best_aicc = float(np.min(ensemble[feasible, 1]))
    supported = feasible & (ensemble[:, 1] - best_aicc <= delta_limit + 1.0e-12)
    k_b_ev_per_k = 8.617333262145e-5
    base = Polynomial([0.0, 1.0, -1.0])
    zeta_poly = Polynomial([-1.0, 2.0])
    basis = [base * zeta_poly ** order for order in range(4)]
    candidates = []
    for row_index in np.flatnonzero(supported):
        row = ensemble[row_index]
        count = int(row[0])
        root = float(row[3])
        coefficients = row[4:4 + count]
        covariance = row[8:24].reshape(4, 4)[:count, :count]
        if not math.isfinite(root) or np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
            raise ValueError("supported candidate has invalid root or covariance")
        second_basis = np.array([poly.deriv(2)(root) for poly in basis[:count]])
        third = k_b_ev_per_k * temperature * (
            -1.0 / root ** 2 + 1.0 / (1.0 - root) ** 2
        ) + sum(float(c * poly.deriv(3)(root))
                for c, poly in zip(coefficients, basis[:count]))
        standard_error = float(np.sqrt(second_basis @ covariance @ second_basis) / abs(third))
        if z_value == 0.0:
            lower_bound = root
        else:
            grid = np.linspace(0.05, root, 513)
            profile = []
            for point in grid:
                b2 = np.array([poly.deriv(2)(point) for poly in basis[:count]])
                mean_curvature = k_b_ev_per_k * temperature * (
                    1.0 / point + 1.0 / (1.0 - point)
                ) + float(coefficients @ b2)
                profile.append(mean_curvature - z_value * np.sqrt(b2 @ covariance @ b2))
            profile = np.asarray(profile)
            crossings = np.flatnonzero(profile[:-1] * profile[1:] < 0.0)
            if crossings.size != 1:
                continue
            lo, hi = float(grid[crossings[0]]), float(grid[crossings[0] + 1])
            f_lo = float(profile[crossings[0]])
            for _ in range(100):
                midpoint = 0.5 * (lo + hi)
                b2 = np.array([poly.deriv(2)(midpoint) for poly in basis[:count]])
                mean_curvature = k_b_ev_per_k * temperature * (
                    1.0 / midpoint + 1.0 / (1.0 - midpoint)
                ) + float(coefficients @ b2)
                f_mid = float(mean_curvature - z_value * np.sqrt(b2 @ covariance @ b2))
                if hi - lo <= 1.0e-12:
                    break
                if f_lo * f_mid <= 0.0:
                    hi = midpoint
                else:
                    lo, f_lo = midpoint, f_mid
            lower_bound = 0.5 * (lo + hi)
        if not (0.0 < lower_bound <= root):
            continue
        profile_half_width = root - lower_bound
        candidates.append((lower_bound, count, root, profile_half_width,
                           float(row[1] - best_aicc), float(third), standard_error))
    if not candidates:
        raise ValueError("no supported candidate has a physical profile bound")
    chosen = min(candidates, key=lambda value: (value[0], value[1]))
    return np.array([chosen[0], chosen[2], chosen[3], float(chosen[1]),
                     chosen[4], chosen[5], chosen[6]], dtype=float)

def ati_spinodal_pipeline(gap_blocks, lambdas, compositions,
                                  solution_volumes_a3, temperature_k,
                                  mass_reference_u, mass_solute_u,
                                  volume_reference_a3, pressure_target_gpa,
                                  bulk_modulus_gpa, bulk_derivative,
                                  volume_zero_a3, calibration_compositions,
                                  calibration_residuals_ev, sigma_grid_ev,
                                  length_grid, noise_sigma_ev,
                                  monitor_index=3, z_score=1.645,
                                  delta_aicc=4.0, coexistence_sigma_radius=1.0):
    """Compose all stages and stress-test the bound by synchronized block deletion."""
    import numpy as np
    blocks = np.asarray(gap_blocks, dtype=float)
    x = np.asarray(compositions, dtype=float)
    volumes = np.asarray(solution_volumes_a3, dtype=float)
    monitor = int(monitor_index)
    if blocks.ndim != 3 or blocks.shape[0] != x.size or blocks.shape[2] != np.asarray(lambdas).size:
        raise ValueError("gap_blocks must have shape (C,B,L)")
    if volumes.shape != x.shape or not (0 <= monitor < x.size):
        raise ValueError("volume vector and monitor_index must match compositions")
    if blocks.shape[1] < np.asarray(lambdas).size + 2:
        raise ValueError("need at least L+2 synchronized blocks for leave-one-block-out stability")

    def run_once(active_blocks):
        certificates = []
        for index in range(x.size):
            statistics = gap_block_statistics(active_blocks[index])
            ti = quadratic_gls_ti(lambdas, statistics)
            mass = alchemical_mass_correction(
                x[index], temperature_k, mass_reference_u, mass_solute_u
            )
            pv = birch_murnaghan_pv(
                volumes[index], volume_reference_a3, pressure_target_gpa,
                bulk_modulus_gpa, bulk_derivative, volume_zero_a3
            )
            certificates.append(
                np.array([ti[0] + mass[0] + pv[0], ti[1], ti[0], mass[0], pv[0]], dtype=float)
            )
        mixing = mixing_curve_covariance(
            x, np.asarray(certificates), temperature_k
        )
        augmented = augment_finite_size_covariance(
            mixing, calibration_compositions, calibration_residuals_ev,
            sigma_grid_ev, length_grid, noise_sigma_ev
        )
        rk = fit_redlich_kister(augmented, temperature_k)
        screened = common_tangent_screen(
            rk, temperature_k, sigma_radius=coexistence_sigma_radius
        )
        spinodal = spinodal_lower_confidence(
            screened, temperature_k, z_score=z_score, delta_aicc=delta_aicc
        )
        chosen_row = screened[np.flatnonzero(screened[:, 0] == spinodal[3])[0]]
        return spinodal, chosen_row

    full_spinodal, _ = run_once(blocks)
    deletion_candidates = []
    for omitted in range(blocks.shape[1]):
        fold_spinodal, fold_row = run_once(np.delete(blocks, omitted, axis=1))
        deletion_candidates.append((float(fold_spinodal[0]), omitted,
                                    fold_spinodal, fold_row))
    first_lower, first_omitted, first_spinodal, _ = min(
        deletion_candidates, key=lambda item: (item[0], item[1])
    )
    _, runner_up_omitted, runner_up_spinodal, _ = sorted(
        deletion_candidates, key=lambda item: (item[0], item[1])
    )[1]
    second_candidates = []
    runner_up_conditional_feasible = 0.0
    for second_omitted in range(blocks.shape[1]):
        if second_omitted == first_omitted:
            continue
        try:
            pair_spinodal, pair_row = run_once(
                np.delete(blocks, [first_omitted, second_omitted], axis=1)
            )
        except ValueError:
            continue
        if second_omitted == runner_up_omitted:
            runner_up_conditional_feasible = 1.0
        second_candidates.append((float(pair_spinodal[0]), second_omitted,
                                  pair_spinodal, pair_row))
    if not second_candidates:
        raise ValueError("no covariance-feasible conditional second deletion")
    robust_lower, second_omitted, chosen_spinodal, chosen_row = min(
        second_candidates, key=lambda item: (item[0], item[1])
    )
    return np.array([
        robust_lower, full_spinodal[0], first_lower, float(first_omitted + 1),
        float(second_omitted + 1), float(len(second_candidates)),
        chosen_spinodal[1], chosen_spinodal[2], chosen_spinodal[3],
        chosen_row[31], chosen_row[32], chosen_spinodal[6],
        first_lower - robust_lower, full_spinodal[0] - robust_lower,
        float(blocks.shape[1]), float(runner_up_omitted + 1),
        float(runner_up_spinodal[0]), runner_up_conditional_feasible,
        float(chosen_spinodal[1] - z_score * chosen_spinodal[6])
    ], dtype=float)
SCICODE_GOLD_EOF
