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


def rs_operator(
    source_xy: "np.ndarray",
    target_xy: "np.ndarray",
    wavelength: float,
    z: float,
    source_area: float,
) -> "np.ndarray":
    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    if (
        source_xy.ndim != 2
        or target_xy.ndim != 2
        or source_xy.shape[1:] != (2,)
        or target_xy.shape[1:] != (2,)
        or min(len(source_xy), len(target_xy)) == 0
    ):
        raise ValueError(
            "Coordinates must be nonempty (N,2) and (M,2) arrays."
        )
    if (
        not all(
            np.isfinite(x).all()
            for x in [source_xy, target_xy, wavelength, z, source_area]
        )
        or min(wavelength, z, source_area) <= 0
    ):
        raise ValueError(
            "Finite coordinates and positive optical scales are required."
        )
    r = np.sqrt(
        np.sum((target_xy[:, None, :] - source_xy[None, :, :]) ** 2, axis=-1)
        + z * z
    )
    k = 2 * np.pi / wavelength
    return (
        source_area
        * z
        * (1 - 1j * k * r)
        * np.exp(1j * k * r)
        / (2 * np.pi * r**3)
    )

import numpy as np


def image_metrics(
    intensity: "np.ndarray",
    target: "np.ndarray",
    bright: "np.ndarray",
    pixel_area: float,
    incident_power: float,
) -> "np.ndarray":
    intensity = np.asarray(intensity, dtype=float)
    target = np.asarray(target, dtype=float)
    bright = np.asarray(bright, dtype=bool)
    if (
        intensity.ndim != 1
        or len(intensity) == 0
        or target.shape != intensity.shape
        or bright.shape != intensity.shape
        or not bright.any()
    ):
        raise ValueError(
            "Matching nonempty vectors and a nonempty bright region are"
            " required."
        )
    if (
        not all(
            np.isfinite(x).all()
            for x in [intensity, target, pixel_area, incident_power]
        )
        or np.any(intensity < 0)
        or min(pixel_area, incident_power) <= 0
    ):
        raise ValueError(
            "Nonnegative intensity and positive finite area/power are"
            " required."
        )
    n = len(intensity)
    b = bright.astype(float) / bright.sum()
    mu = b @ intensity
    if mu <= 0:
        raise ValueError("Bright-region mean must be positive.")
    q = intensity / mu - target
    rmse = np.sqrt(q @ q / n)
    d = intensity - mu
    sd = np.sqrt(d @ d / n)
    eta = pixel_area * intensity[bright].sum() / incident_power
    dr = (
        (q / mu - b * (q @ intensity) / mu**2) / (n * rmse)
        if rmse > 0
        else np.zeros(n)
    )
    ds = (d - b * d.sum()) / (n * sd) if sd > 0 else np.zeros(n)
    de = pixel_area / incident_power * bright
    return np.column_stack(([rmse, sd, eta], np.array([dr, ds, de])))

import numpy as np


def aps_gradient(
    phase: "np.ndarray",
    operator: "np.ndarray",
    amplitude: "np.ndarray",
    target: "np.ndarray",
    bright: "np.ndarray",
    pixel_area: float,
    incident_power: float,
    iteration: int,
    weights: "np.ndarray",
    transition: float,
    width: float,
) -> "np.ndarray":
    phase = np.asarray(phase, dtype=float)
    operator = np.asarray(operator, dtype=complex)
    amplitude = np.asarray(amplitude, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (
        phase.ndim != 1
        or amplitude.shape != phase.shape
        or operator.shape != (len(target), len(phase))
        or weights.shape != (3,)
    ):
        raise ValueError("Incompatible optical arrays or weights.")
    if (
        not all(
            np.isfinite(x).all()
            for x in [
                phase,
                operator,
                amplitude,
                weights,
                iteration,
                transition,
                width,
            ]
        )
        or np.any(amplitude < 0)
        or np.any(weights < 0)
        or iteration < 1
        or int(iteration) != iteration
        or width <= 0
    ):
        raise ValueError("Invalid phase, schedule, amplitude or weights.")
    p = amplitude * np.exp(1j * phase)
    field = operator @ p
    met = image_metrics(
        np.abs(field) ** 2, target, bright, pixel_area, incident_power
    )
    s = 0.5 * (1 + np.tanh((iteration - transition) / width))
    coef = np.array(
        [s * weights[0], (1 - s) * weights[1], -(1 - s) * weights[2]]
    )
    di = coef @ met[:, 1:]
    gradient = 2 * np.imag(np.conj(p) * (operator.conj().T @ (di * field)))
    return np.r_[coef @ met[:, 0] + (1 - s) * weights[2], gradient]

import numpy as np


def optimize_mask(
    initial_phase: "np.ndarray",
    operator: "np.ndarray",
    amplitude: "np.ndarray",
    target: "np.ndarray",
    bright: "np.ndarray",
    pixel_area: float,
    incident_power: float,
    weights: "np.ndarray",
    iterations: int,
    transition: float,
    width: float,
    learning_rate: float,
) -> "np.ndarray":
    phase = np.asarray(initial_phase, dtype=float).copy()
    if (
        phase.ndim != 1
        or phase.size == 0
        or not np.isfinite(phase).all()
        or not np.isfinite(
            [iterations, transition, width, learning_rate]
        ).all()
        or int(iterations) != iterations
        or iterations < 0
        or min(width, learning_rate) <= 0
    ):
        raise ValueError("Invalid phase or optimization controls.")
    m = np.zeros_like(phase)
    v = np.zeros_like(phase)
    for i in range(1, int(iterations) + 1):
        g = aps_gradient(
            phase,
            operator,
            amplitude,
            target,
            bright,
            pixel_area,
            incident_power,
            i,
            weights,
            transition,
            width,
        )[1:]
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        phase -= (
            learning_rate
            * (m / (1 - 0.9**i))
            / (np.sqrt(v / (1 - 0.999**i)) + 1e-8)
        )
    return phase

import numpy as np


def fabrication_moments(
    phase: "np.ndarray", levels: int, rotation_sigma: "np.ndarray"
) -> "np.ndarray":
    phase = np.asarray(phase, dtype=float)
    rotation_sigma = np.asarray(rotation_sigma, dtype=float)
    if (
        phase.ndim != 1
        or phase.size == 0
        or rotation_sigma.shape != phase.shape
        or not np.isfinite(phase).all()
        or not np.isfinite(rotation_sigma).all()
        or np.any(rotation_sigma < 0)
        or not np.isfinite(levels)
        or int(levels) != levels
        or levels < 2
    ):
        raise ValueError("Invalid phase, phase levels or angular errors.")
    step = 2 * np.pi / levels
    scaled = np.mod(phase, 2 * np.pi) / step
    half = np.floor(scaled) + 0.5
    scaled = np.where(
        np.abs(scaled - half) <= 4 * np.spacing(half), half, scaled
    )
    quantized = step * (np.floor(scaled + 0.5) % levels)
    sig = 2 * rotation_sigma
    return np.array(
        [np.exp(1j * k * quantized - 0.5 * k * k * sig**2) for k in [1, 2]]
    )

import numpy as np


def field_statistics(
    operator: "np.ndarray", amplitude: "np.ndarray", moments: "np.ndarray"
) -> "np.ndarray":
    operator = np.asarray(operator, dtype=complex)
    amplitude = np.asarray(amplitude, dtype=float)
    moments = np.asarray(moments, dtype=complex)
    if (
        operator.ndim != 2
        or min(operator.shape) == 0
        or amplitude.shape != (operator.shape[1],)
        or moments.shape != (2, operator.shape[1])
    ):
        raise ValueError("Incompatible field dimensions.")
    if not all(
        np.isfinite(x).all() for x in [operator, amplitude, moments]
    ) or np.any(amplitude < 0):
        raise ValueError("Invalid optical inputs.")
    d = 1 - np.abs(moments[0]) ** 2
    p = moments[1] - moments[0] ** 2
    if np.any(d < -1e-12) or np.any(np.abs(p) > d + 1e-12):
        raise ValueError("Moments do not describe a unit-modulus phasor.")
    # Repair roundoff outside d >= |p|, but preserve every positive d.
    # A small central moment can become significant after optical scaling.
    d = np.maximum(d, 0.0)
    magnitude = np.abs(p)
    outside = magnitude > d
    p[outside] *= d[outside] / magnitude[outside]
    h = operator * amplitude
    mu = h @ moments[0]
    c = np.abs(h) ** 2 @ d
    pseudo = h * h @ p
    return np.array(
        [
            mu.real,
            mu.imag,
            (c + pseudo.real) / 2,
            pseudo.imag / 2,
            (c - pseudo.real) / 2,
        ]
    ).T

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.optimize import brentq


def _intensity_cdf(threshold, row):
    if threshold < 0:
        return 0.0
    covariance = np.array([[row[2], row[3]], [row[3], row[4]]])
    eigenvalues, basis = np.linalg.eigh(covariance)
    mean = basis.T @ np.asarray(row[:2])
    eigenvalues = np.maximum(eigenvalues, 0.0)
    if eigenvalues[1] == 0:
        return float(mean @ mean <= threshold)
    # Only genuinely degenerate laws use the rank-one limit. An eigenvalue
    # ratio cannot bound the effect of a narrow direction on a displaced tail.
    if eigenvalues[0] == 0:
        remaining = threshold - mean[0] ** 2
        if remaining <= 0:
            return 0.0
        radius = np.sqrt(remaining)
        scale = np.sqrt(eigenvalues[1])
        return float(
            ndtr((radius - mean[1]) / scale)
            - ndtr((-radius - mean[1]) / scale)
        )
    scales = np.sqrt(eigenvalues)
    # Condition on the most strongly displaced standardized coordinate.
    # This resolves narrow displaced tails over an O(1) normal integration
    # range.
    axis = int(np.argmax(np.abs(mean) / scales))
    other = 1 - axis
    center, scale = mean[axis], scales[axis]
    radius = np.sqrt(threshold)
    difference = threshold - center**2
    upper = (
        difference / (scale * (radius + center))
        if center >= 0 and radius + center
        else (radius - center) / scale
    )
    lower = (
        -difference / (scale * (radius - center))
        if center < 0
        else (-radius - center) / scale
    )
    lower, upper = max(-11.0, lower), min(11.0, upper)
    if lower >= upper:
        return 0.0

    def _density(z):
        # Expand the square to avoid subtracting two nearly equal full squares.
        remaining = max(
            difference - 2 * center * scale * z - scale**2 * z**2, 0.0
        )
        half_width = np.sqrt(remaining)
        conditional = ndtr((half_width - mean[other]) / scales[other]) - ndtr(
            (-half_width - mean[other]) / scales[other]
        )
        return np.exp(-z * z / 2) / np.sqrt(2 * np.pi) * conditional

    points = [
        z
        for z in [-8.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 8.0]
        if lower < z < upper
    ]
    return float(
        quad(
            _density,
            lower,
            upper,
            points=points,
            epsabs=2e-11,
            epsrel=2e-11,
            limit=250,
        )[0]
    )


def intensity_quantiles(
    statistics: "np.ndarray", probabilities: "np.ndarray"
) -> "np.ndarray":
    statistics = np.asarray(statistics, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        statistics.ndim != 2
        or statistics.shape[1] != 5
        or statistics.shape[0] == 0
        or probabilities.ndim != 1
        or probabilities.size == 0
        or not np.isfinite(statistics).all()
        or not np.isfinite(probabilities).all()
        or np.any((probabilities <= 0) | (probabilities >= 1))
    ):
        raise ValueError("Invalid Gaussian statistics or probabilities.")
    result = np.zeros((len(statistics), len(probabilities)))
    for j, row in enumerate(statistics):
        sigma = np.array([[row[2], row[3]], [row[3], row[4]]])
        ev = np.linalg.eigvalsh(sigma)
        if ev[0] < -1e-12 * max(1.0, ev[1]):
            raise ValueError("Covariance must be positive semidefinite.")
        mean = row[0] ** 2 + row[1] ** 2 + row[2] + row[4]
        if ev[1] <= 0:
            result[j] = max(mean, 0.0)
            continue
        # Scale to a mean-one intensity law, retaining the noncircular
        # covariance.
        scale = max(mean, np.finfo(float).tiny)
        normalized = np.array(
            [
                row[0] / np.sqrt(scale),
                row[1] / np.sqrt(scale),
                row[2] / scale,
                row[3] / scale,
                row[4] / scale,
            ]
        )
        for k, prob in enumerate(probabilities):
            hi = 2.0
            while _intensity_cdf(hi, normalized) < prob:
                hi *= 2
            result[j, k] = scale * brentq(
                lambda q: _intensity_cdf(q, normalized) - prob,
                0.0,
                hi,
                xtol=2e-14,
                rtol=2e-14,
            )
    return result

import numpy as np


def exposure_window(
    statistics: "np.ndarray",
    quantiles: "np.ndarray",
    bright: "np.ndarray",
    pixel_area: float,
    incident_power: float,
    dose_cap: float,
) -> "np.ndarray":
    statistics = np.asarray(statistics, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    bright = np.asarray(bright, dtype=bool)
    if (
        statistics.ndim != 3
        or statistics.shape[2] != 5
        or quantiles.shape != statistics.shape[:2] + (2,)
        or bright.shape != (statistics.shape[1],)
        or not bright.any()
        or bright.all()
    ):
        raise ValueError(
            "Incompatible plane statistics, quantiles or bright/dark mask."
        )
    if (
        not all(
            np.isfinite(x).all()
            for x in [
                statistics,
                quantiles,
                pixel_area,
                incident_power,
                dose_cap,
            ]
        )
        or min(pixel_area, incident_power, dose_cap) <= 0
        or np.any(quantiles < 0)
        or np.any(quantiles[:, :, 0] > quantiles[:, :, 1])
    ):
        raise ValueError("Invalid physical scales or ordered quantiles.")
    qb = quantiles[:, bright, 0].min()
    qd = quantiles[:, ~bright, 1].max()
    lower = 1 / qb if qb > 0 else np.inf
    upper = min(dose_cap, 1 / qd) if qd > 0 else dose_cap
    means = (
        np.sum(statistics[:, :, :2] ** 2, axis=2)
        + statistics[:, :, 2]
        + statistics[:, :, 4]
    )
    eta = np.min(
        pixel_area * np.sum(means[:, bright], axis=1) / incident_power
    )
    return np.array([lower, upper, eta])

import numpy as np


def certify_hologram(
    source_xy: "np.ndarray",
    target_xy: "np.ndarray",
    amplitude: "np.ndarray",
    initial_phase: "np.ndarray",
    target: "np.ndarray",
    bright: "np.ndarray",
    wavelength: float,
    design_z: float,
    planes: "np.ndarray",
    source_area: float,
    pixel_area: float,
    weights: "np.ndarray",
    rotation_sigma: "np.ndarray",
    levels: int,
    iterations: int,
    eta_min: float,
    dose_cap: float,
    tail_probability: float = 0.05,
    transition: float = 175.0,
    width: float = 40.0,
    learning_rate: float = 0.005,
) -> float:
    amplitude = np.asarray(amplitude, dtype=float)
    weights = np.asarray(weights, dtype=float)
    planes = np.asarray(planes, dtype=float)
    if (
        weights.ndim != 2
        or weights.shape[1] != 3
        or len(weights) == 0
        or planes.ndim != 1
        or len(planes) == 0
        or not np.isfinite([eta_min, tail_probability]).all()
        or eta_min < 0
        or not 0 < tail_probability < 0.5
    ):
        raise ValueError(
            "Invalid candidate set, efficiency floor or tail probability."
        )
    pin = source_area * np.sum(amplitude**2)
    H = rs_operator(
        source_xy, target_xy, wavelength, design_z, source_area
    )
    propagation = [
        rs_operator(source_xy, target_xy, wavelength, z, source_area)
        for z in planes
    ]
    best = -1.0
    for w in weights:
        phi = optimize_mask(
            initial_phase,
            H,
            amplitude,
            target,
            bright,
            pixel_area,
            pin,
            w,
            iterations,
            transition,
            width,
            learning_rate,
        )
        moments = fabrication_moments(phi, levels, rotation_sigma)
        stats = np.array(
            [
                field_statistics(op, amplitude, moments)
                for op in propagation
            ]
        )
        qs = np.array(
            [
                intensity_quantiles(
                    s, np.array([tail_probability, 1 - tail_probability])
                )
                for s in stats
            ]
        )
        lo, hi, eta = exposure_window(
            stats, qs, bright, pixel_area, pin, dose_cap
        )
        if eta >= eta_min and hi > lo:
            score = float(np.log(hi / lo))
            if score > best:
                best = score
    return float(best)
SCICODE_GOLD_EOF
