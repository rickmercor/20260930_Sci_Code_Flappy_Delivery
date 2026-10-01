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

def observation_plan() -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed, finite observing and injection records."""
    import numpy as np

    exposures = np.array([
        [0.00,  4.54e-5,  0.00e-5,  1.10e-5,  0.00015,  0.00425,  0.00020, 0.76,  92.0, 0.0045, 1],
        [0.31,  0.72e-5,  4.48e-5, -1.25e-5, -0.00420,  0.00060, -0.00020, 0.93, 118.0, 0.0052, 1],
        [0.69, -4.22e-5,  1.45e-5,  0.95e-5, -0.00120, -0.00410,  0.00015, 1.12,  86.0, 0.0048, 0],
        [1.08, -1.85e-5, -4.17e-5, -0.82e-5,  0.00390, -0.00170, -0.00020, 0.84, 141.0, 0.0054, 0],
        [1.56,  4.10e-5, -1.72e-5,  1.31e-5,  0.00170,  0.00400,  0.00025, 1.04, 104.0, 0.0046, 0],
        [2.03,  1.13e-5,  4.39e-5, -1.04e-5, -0.00410,  0.00100, -0.00010, 0.88, 127.0, 0.0050, 1],
    ], dtype=float)
    injections = np.array([
        [26.80,  101, -1.60, -1.10, -0.045,  0.030, 0.30, 0.15, 1.20],
        [26.99,  211,  1.35, -1.45,  0.310, -0.240, 0.35, 1.93, 0.85],
        [27.12,  307, -0.55,  1.72,  0.060,  0.020, 0.45, 3.42, 1.05],
        [27.20,  419,  1.78,  0.54,  2.900, -2.200, 0.30, 4.75, 1.60],
        [27.31,  523, -1.88,  0.36, -0.030,  0.055, 0.90, 5.62, 0.62],
        [27.44,  631,  0.70, -1.84, -0.520, -0.180, 0.15, 2.53, 1.80],
        [27.55,  743, -0.18,  0.84, -2.600,  2.500, 0.40, 0.87, 0.95],
        [27.66,  463,  1.07,  1.18,  0.420, -0.310, 0.55, 3.88, 0.75],
        [27.79,  967, -1.22, -0.31, -0.061,  0.011, 0.85, 5.10, 2.30],
        [27.90,  189,  0.31, -0.68,  0.015, -0.063, 0.25, 1.32, 0.55],
        [28.02, 1193,  1.63, -0.14,  0.570, -0.270, 0.35, 1.9634954085, 1.45],
        [28.08, 1301, -0.91,  1.39,  3.100,  1.900, 0.35, 4.4505895926, 1.75],
        [28.13, 1427,  0.92,  0.25, -1.450,  0.920, 0.70, 0.48, 0.65],
        [28.18, 1543, -1.47, -1.52,  0.048, -0.035, 0.20, 2.18, 2.30],
        [28.33, 1657,  0.08,  1.51, -0.370,  0.520, 0.45, 3.57, 0.90],
        [28.50, 1777,  1.47, -0.83, -1.180,  0.740, 0.30, 5.91, 1.10],
        [28.60, 1889, -0.73,  0.05,  1.620,  1.050, 0.60, 1.05, 2.20],
        [28.71, 1999,  0.53, -1.18, -0.026,  0.044, 0.20, 4.94, 0.80],
        [28.83, 2111, -1.31,  0.97,  0.510, -0.310, 0.95, 2.71, 1.70],
        [28.96, 2237,  1.16, -0.42,  0.019,  0.058, 0.40, 0.62, 0.60],
        [29.10, 2351, -0.34, -1.66, -1.100, -0.750, 0.25, 3.15, 1.30],
        [28.26, 2477,  1.72,  0.68,  0.028, -0.022, 0.15, 5.38, 2.40],
        [29.25, 2593, -1.05,  1.24,  0.310,  0.500, 0.75, 1.47, 2.00],
        [29.41, 2707,  0.44, -0.95, -0.017, -0.045, 0.30, 4.13, 0.70],
    ], dtype=float)
    return exposures, injections

import numpy as np

def spherical_orbit_state(phi: float, theta: float, distance_au: float, radial_speed_au_day: float, tangential_speed_au_day: float, inclination_rad: float, kappa: int) -> np.ndarray:
    """Construct the state using the spherical A--D tangent basis."""
    import numpy as np

    values = np.array([phi, theta, distance_au, radial_speed_au_day, tangential_speed_au_day, inclination_rad], dtype=float)
    if not np.all(np.isfinite(values)) or distance_au <= 0 or tangential_speed_au_day <= 0 or kappa not in (-1, 1):
        raise ValueError("invalid spherical-orbit parameters")
    ratio = np.cos(inclination_rad) / np.cos(theta)
    if abs(ratio) > 1.0 + 1e-12:
        raise ValueError("inclination is incompatible with latitude")
    psi = float(kappa * np.arccos(np.clip(ratio, -1.0, 1.0)))
    rhat = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)])
    ahat = np.array([-np.sin(phi), np.cos(phi), 0.0])
    dhat = np.cross(rhat, ahat)
    position = distance_au * rhat
    velocity = radial_speed_au_day * rhat + tangential_speed_au_day * (np.cos(psi) * ahat + np.sin(psi) * dhat)
    return np.concatenate([position, velocity, [psi]])

import numpy as np

def apparent_registration_path(state: np.ndarray, exposures: np.ndarray, pixel_scale_arcsec: float, wcs_rate_px_day: np.ndarray) -> np.ndarray:
    """Solve retarded midpoint directions and finite-exposure trajectories."""
    import numpy as np

    def acceleration(position: np.ndarray) -> np.ndarray:
        radius = float(np.linalg.norm(position))
        return -2.959122082855911e-4 * position / radius**3

    def propagate_rk4(position: np.ndarray, velocity: np.ndarray, elapsed_day: float) -> np.ndarray:
        count = max(1, int(np.ceil(abs(elapsed_day) / 0.01)))
        h = elapsed_day / count
        r, v = position.astype(float).copy(), velocity.astype(float).copy()
        for _ in range(count):
            k1r, k1v = v, acceleration(r)
            k2r, k2v = v + 0.5 * h * k1v, acceleration(r + 0.5 * h * k1r)
            k3r, k3v = v + 0.5 * h * k2v, acceleration(r + 0.5 * h * k2r)
            k4r, k4v = v + h * k3v, acceleration(r + h * k3r)
            r += h * (k1r + 2 * k2r + 2 * k3r + k4r) / 6
            v += h * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
        return r

    state = np.asarray(state, dtype=float)
    exposures = np.asarray(exposures, dtype=float)
    wcs_rate_px_day = np.asarray(wcs_rate_px_day, dtype=float)
    if (state.shape != (7,) or exposures.ndim != 2 or exposures.shape[1] != 11
            or not np.all(np.isfinite(state)) or not np.all(np.isfinite(exposures))
            or not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0
            or wcs_rate_px_day.shape != (2,) or not np.all(np.isfinite(wcs_rate_px_day))):
        raise ValueError("invalid state, exposure table, or pixel scale")
    r0, v0 = state[:3], state[3:6]
    u0 = r0 / np.linalg.norm(r0)
    east = np.array([-u0[1], u0[0], 0.0])
    east /= np.linalg.norm(east)
    north = np.cross(u0, east)
    def apparent_at(time: float, observer: np.ndarray) -> np.ndarray:
        emitted = time - np.linalg.norm(r0 - observer) / 173.144632674240
        for _ in range(5):
            position = propagate_rk4(r0, v0, emitted)
            emitted = time - np.linalg.norm(position - observer) / 173.144632674240
        sightline = position - observer
        direction = sightline / np.linalg.norm(sightline)
        denominator = float(np.dot(direction, u0))
        return np.array([206264.806247 * np.dot(direction, east) / denominator / pixel_scale_arcsec,
                         206264.806247 * np.dot(direction, north) / denominator / pixel_scale_arcsec])

    midpoint = []
    start = []
    stop = []
    for row in exposures:
        time, observer, velocity, duration = float(row[0]), row[1:4], row[4:7], float(row[9])
        half = 0.5 * duration
        midpoint.append(apparent_at(time, observer))
        start.append(apparent_at(time - half, observer - half * velocity))
        stop.append(apparent_at(time + half, observer + half * velocity))
    midpoint, start, stop = np.asarray(midpoint), np.asarray(start), np.asarray(stop)
    def residual(coordinates: np.ndarray, times: np.ndarray) -> np.ndarray:
        line = midpoint[0] + (times - exposures[0, 0])[:, None] * wcs_rate_px_day
        return coordinates - line
    middle_residual = residual(midpoint, exposures[:, 0])
    trail = stop - start - exposures[:, 9, None] * wcs_rate_px_day
    return np.round(np.column_stack([middle_residual, trail]), 6)

import numpy as np

def matched_filter_statistics(path: np.ndarray, exposures: np.ndarray, injections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate difference stamps and correlate each with its own PSF."""
    import numpy as np

    fractions = np.linspace(-0.5, 0.5, 7)

    def relative_response(amplitude_mag: float, phase_rad: float, red_blue_flux_ratio: float, band_code: float, sample_times_day: np.ndarray) -> np.ndarray:
        if red_blue_flux_ratio <= 0 or band_code not in (0.0, 1.0):
            raise ValueError("invalid source color or band code")
        relative_flux = 10.0 ** (-0.4 * amplitude_mag * np.sin(2.0 * np.pi * sample_times_day / 0.73 + phase_rad))
        return relative_flux * (red_blue_flux_ratio if band_code == 1.0 else 1.0)

    def trailed_psf(sigma: float, trail: np.ndarray, relative_flux: np.ndarray) -> np.ndarray:
        axis = np.arange(-5, 6, dtype=float)
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        template = np.zeros_like(xx)
        for fraction, weight in zip(fractions, relative_flux):
            template += weight * np.exp(-((xx - fraction * trail[0]) ** 2 + (yy - fraction * trail[1]) ** 2) / (2 * sigma**2))
        return template / template.sum()

    def source_profile(xx: np.ndarray, yy: np.ndarray, x0: float, y0: float, sigma: float, trail: np.ndarray, relative_flux: np.ndarray) -> np.ndarray:
        profile = np.zeros_like(xx)
        for fraction, weight in zip(fractions, relative_flux):
            profile += weight * np.exp(-((xx - x0 - fraction * trail[0]) ** 2 + (yy - y0 - fraction * trail[1]) ** 2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        return profile / 7.0

    def correlate_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        pad = kernel.shape[0] // 2
        padded = np.pad(image, pad, mode="constant")
        output = np.empty_like(image, dtype=float)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                output[y, x] = float(np.sum(padded[y:y + kernel.shape[0], x:x + kernel.shape[1]] * kernel))
        return output

    path = np.asarray(path, dtype=float)
    exposures = np.asarray(exposures, dtype=float)
    injections = np.asarray(injections, dtype=float)
    if path.shape != (len(exposures), 4) or exposures.ndim != 2 or exposures.shape[1] != 11 or injections.ndim != 2 or injections.shape[1] != 9:
        raise ValueError("incompatible path, exposure, or injection arrays")
    if not np.all(np.isfinite(path)) or not np.all(np.isfinite(exposures)) or not np.all(np.isfinite(injections)):
        raise ValueError("non-finite matched-filter input")
    if np.any(exposures[:, 7] <= 0.0) or np.any(exposures[:, 8] <= 0.0):
        raise ValueError("seeing widths and variances must be positive")
    grid = 25
    center = (grid - 1) / 2
    yy, xx = np.meshgrid(np.arange(grid, dtype=float), np.arange(grid, dtype=float), indexing="ij")
    xi = np.empty((len(injections), len(exposures), grid, grid), dtype=float)
    zeta = np.empty_like(xi)
    for source_index, source in enumerate(injections):
        magnitude, seed, base_x, base_y, drift_x, drift_y, amplitude_mag, phase_rad, red_blue_flux_ratio = source
        flux = 455.0 * 10.0 ** (-0.4 * (magnitude - 27.0))
        for exposure_index, exposure in enumerate(exposures):
            time, sigma, variance, duration, band_code = exposure[0], exposure[7], exposure[8], exposure[9], exposure[10]
            source_x = center + base_x + drift_x * time + path[exposure_index, 0]
            source_y = center + base_y + drift_y * time + path[exposure_index, 1]
            trail = path[exposure_index, 2:4] + duration * np.array([drift_x, drift_y])
            response = relative_response(float(amplitude_mag), float(phase_rad), float(red_blue_flux_ratio), float(band_code), time + fractions * duration)
            psf = trailed_psf(float(sigma), trail, response)
            model = flux * source_profile(xx, yy, source_x, source_y, sigma, trail, response)
            rng = np.random.default_rng(int(seed + 1009 * exposure_index))
            image = model + rng.normal(0.0, np.sqrt(variance), size=(grid, grid))
            xi[source_index, exposure_index] = correlate_same(image / variance, psf)
            zeta[source_index, exposure_index] = correlate_same(np.full((grid, grid), 1.0 / variance), psf**2)
    return xi, zeta

import numpy as np

def registered_coadd(xi: np.ndarray, zeta: np.ndarray, path: np.ndarray, exposures: np.ndarray, injections: np.ndarray) -> np.ndarray:
    """Resample ξ and ζ separately, with source-response likelihood weights."""
    import numpy as np

    def bilinear(image: np.ndarray, y: float, x: float) -> float:
        if y < 0 or x < 0 or y > image.shape[0] - 1 or x > image.shape[1] - 1:
            return 0.0
        y0, x0 = int(np.floor(y)), int(np.floor(x))
        y1, x1 = min(y0 + 1, image.shape[0] - 1), min(x0 + 1, image.shape[1] - 1)
        fy, fx = y - y0, x - x0
        return float((1 - fy) * ((1 - fx) * image[y0, x0] + fx * image[y0, x1]) + fy * ((1 - fx) * image[y1, x0] + fx * image[y1, x1]))

    def mean_relative_response(amplitude_mag: float, phase_rad: float, red_blue_flux_ratio: float, midpoint_day: float, duration_day: float, band_code: float) -> float:
        if red_blue_flux_ratio <= 0 or band_code not in (0.0, 1.0):
            raise ValueError("invalid source color or band code")
        fractions = np.linspace(-0.5, 0.5, 7)
        samples = 10.0 ** (-0.4 * amplitude_mag * np.sin(2.0 * np.pi * (midpoint_day + fractions * duration_day) / 0.73 + phase_rad))
        return float(np.mean(samples) * (red_blue_flux_ratio if band_code == 1.0 else 1.0))

    xi, zeta, path = np.asarray(xi, dtype=float), np.asarray(zeta, dtype=float), np.asarray(path, dtype=float)
    exposures, injections = np.asarray(exposures, dtype=float), np.asarray(injections, dtype=float)
    if xi.ndim != 4 or zeta.shape != xi.shape or path.shape != (xi.shape[1], 4) or exposures.shape != (xi.shape[1], 11) or injections.shape != (xi.shape[0], 9):
        raise ValueError("incompatible coadd arrays")
    if not all(np.all(np.isfinite(value)) for value in (xi, zeta, path, exposures, injections)):
        raise ValueError("non-finite coadd input")
    count, _, height, width = xi.shape
    result = np.empty((count, height, width), dtype=float)
    for source in range(count):
        numerator = np.zeros((height, width), dtype=float)
        precision = np.zeros((height, width), dtype=float)
        for exposure in range(xi.shape[1]):
            dx, dy = path[exposure, :2]
            weight = mean_relative_response(injections[source, 6], injections[source, 7], injections[source, 8], exposures[exposure, 0], exposures[exposure, 9], exposures[exposure, 10])
            for row in range(height):
                for col in range(width):
                    numerator[row, col] += weight * bilinear(xi[source, exposure], row + dy, col + dx)
                    precision[row, col] += weight**2 * bilinear(zeta[source, exposure], row + dy, col + dx)
        result[source] = numerator / np.sqrt(np.maximum(precision, 1e-300))
    return result

import numpy as np

def deduplicated_candidates(significance_maps: np.ndarray) -> np.ndarray:
    """Keep the highest peak after deterministic one-pixel local suppression."""
    import numpy as np

    maps = np.asarray(significance_maps, dtype=float)
    if maps.ndim != 3 or maps.shape[1] < 3 or maps.shape[2] < 3 or not np.all(np.isfinite(maps)):
        raise ValueError("invalid significance-map cube")
    selected = np.empty((maps.shape[0], 3), dtype=float)
    for source, image in enumerate(maps):
        interior = image[1:-1, 1:-1]
        row, col = np.unravel_index(np.argmax(interior), interior.shape)
        row, col = row + 1, col + 1
        selected[source] = [float(col), float(row), float(image[row, col])]
    return selected

import numpy as np

def injection_recoveries(candidates: np.ndarray, xi: np.ndarray, zeta: np.ndarray, path: np.ndarray, exposures: np.ndarray, injections: np.ndarray, threshold: float) -> np.ndarray:
    """Use ten-pass 3-sigma clipping that never empties the retained set."""
    import numpy as np

    def bilinear(image: np.ndarray, y: float, x: float) -> float:
        if y < 0 or x < 0 or y > image.shape[0] - 1 or x > image.shape[1] - 1:
            return 0.0
        y0, x0 = int(np.floor(y)), int(np.floor(x))
        y1, x1 = min(y0 + 1, image.shape[0] - 1), min(x0 + 1, image.shape[1] - 1)
        fy, fx = y - y0, x - x0
        return float((1 - fy) * ((1 - fx) * image[y0, x0] + fx * image[y0, x1]) + fy * ((1 - fx) * image[y1, x0] + fx * image[y1, x1]))

    def mean_relative_response(amplitude_mag: float, phase_rad: float, red_blue_flux_ratio: float, midpoint_day: float, duration_day: float, band_code: float) -> float:
        if red_blue_flux_ratio <= 0 or band_code not in (0.0, 1.0):
            raise ValueError("invalid source color or band code")
        fractions = np.linspace(-0.5, 0.5, 7)
        rotation = 10.0 ** (-0.4 * amplitude_mag * np.sin(2.0 * np.pi * (midpoint_day + fractions * duration_day) / 0.73 + phase_rad))
        return float(np.mean(rotation) * (red_blue_flux_ratio if band_code == 1.0 else 1.0))

    candidates, xi, zeta = np.asarray(candidates, dtype=float), np.asarray(xi, dtype=float), np.asarray(zeta, dtype=float)
    path, exposures, injections = np.asarray(path, dtype=float), np.asarray(exposures, dtype=float), np.asarray(injections, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or xi.ndim != 4 or zeta.shape != xi.shape:
        raise ValueError("invalid candidate or matched-statistic input")
    if (len(candidates) != xi.shape[0] or path.shape != (xi.shape[1], 4)
            or exposures.shape != (xi.shape[1], 11) or injections.shape != (xi.shape[0], 9)
            or not np.isfinite(threshold) or threshold <= 0):
        raise ValueError("incompatible forced-photometry arrays")
    if not all(np.all(np.isfinite(value)) for value in (candidates, xi, zeta, path, exposures, injections)):
        raise ValueError("non-finite forced-photometry input")
    recovered = np.empty(len(injections), dtype=float)
    for source in range(len(injections)):
        values, errors = [], []
        for exposure in range(xi.shape[1]):
            row = candidates[source, 1] + path[exposure, 1]
            col = candidates[source, 0] + path[exposure, 0]
            numerator = bilinear(xi[source, exposure], row, col)
            precision = bilinear(zeta[source, exposure], row, col)
            if precision <= 0.0:
                continue
            response = mean_relative_response(injections[source, 6], injections[source, 7], injections[source, 8], exposures[exposure, 0], exposures[exposure, 9], exposures[exposure, 10])
            values.append(numerator / (response * precision))
            errors.append(1.0 / (response * np.sqrt(precision)))
        values, errors = np.asarray(values, dtype=float), np.asarray(errors, dtype=float)
        if len(values) == 0:
            raise ValueError("candidate has no positive-precision forced measurement")
        keep = np.ones(len(values), dtype=bool)
        for _ in range(10):
            mean_flux = float(np.mean(values[keep]))
            updated = np.abs(values - mean_flux) <= 3.0 * errors
            if np.array_equal(updated, keep) or not updated.any():
                break
            keep = updated
        mean_flux = float(np.mean(values[keep]))
        mean_error = float(np.sqrt(np.sum(errors[keep] ** 2)) / np.count_nonzero(keep))
        recovered[source] = float(mean_flux / mean_error >= threshold)
    return np.column_stack([injections[:, 0], recovered])

import numpy as np

def logistic_completeness_fit(recoveries: np.ndarray) -> float:
    """Use damped Newton iterations for p(m)=expit(beta0+beta1*m)."""
    import numpy as np

    records = np.asarray(recoveries, dtype=float)
    if records.ndim != 2 or records.shape[1] != 2 or len(records) < 6 or not np.all(np.isfinite(records)):
        raise ValueError("invalid recovery table")
    magnitude, flag = records[:, 0], records[:, 1]
    if np.any((flag != 0.0) & (flag != 1.0)) or np.all(flag == flag[0]):
        raise ValueError("recovery flags must contain both binary outcomes")
    design = np.column_stack([np.ones_like(magnitude), magnitude - 28.7])
    beta = np.array([0.0, -2.2], dtype=float)
    for _ in range(80):
        eta = np.clip(design @ beta, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        weight = np.maximum(probability * (1.0 - probability), 1e-8)
        hessian = design.T @ (weight[:, None] * design) + 1e-8 * np.eye(2)
        step = np.linalg.solve(hessian, design.T @ (flag - probability))
        scale = 1.0
        base = float(np.sum(np.logaddexp(0.0, eta) - flag * eta))
        while scale > 2.0**-20:
            trial = beta + scale * step
            trial_eta = np.clip(design @ trial, -40.0, 40.0)
            trial_loss = float(np.sum(np.logaddexp(0.0, trial_eta) - flag * trial_eta))
            if trial_loss <= base:
                beta = trial
                break
            scale *= 0.5
        if np.linalg.norm(scale * step) < 1e-11:
            break
    if beta[1] >= -1e-6:
        raise ValueError("efficiency fit is not decreasing with magnitude")
    return float(28.7 - beta[0] / beta[1])

def completeness_magnitude(inclination_rad: float = 0.0614, branch_sign: int = -1) -> float:
    """Compose the eight preceding numerical stages using only their oracles."""
    import numpy as np

    inclination = float(inclination_rad)
    if not np.isfinite(inclination) or branch_sign not in (-1, 1):
        raise ValueError("invalid search configuration")
    exposures, injections = observation_plan()
    state = spherical_orbit_state(0.640, 0.061, 42.60, -0.00014, 0.00418, inclination, branch_sign)
    path = apparent_registration_path(state, exposures, 0.040, np.array([502.65520879, -2.27302087]))
    xi, zeta = matched_filter_statistics(path, exposures, injections)
    significance = registered_coadd(xi, zeta, path, exposures, injections)
    candidates = deduplicated_candidates(significance)
    recoveries = injection_recoveries(candidates, xi, zeta, path, exposures, injections, 8.80)
    return logistic_completeness_fit(recoveries)
SCICODE_GOLD_EOF
