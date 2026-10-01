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


def compute_modulation_fourier_coefficients(sigma_mean: float,
                                                    sigma_amplitudes: "np.ndarray",
                                                    sigma_phases: "np.ndarray",
                                                    capacity_mean: float,
                                                    capacity_amplitudes: "np.ndarray",
                                                    capacity_phases: "np.ndarray",
                                                    order: int) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(order, (int, np.integer)) and not isinstance(order, bool)
            and int(order) >= 1):
        raise ValueError("order must be an integer >= 1")
    order = int(order)
    grid = np.linspace(0.0, 2.0 * np.pi, 4097)[:-1]
    modes = np.zeros((2, 2 * order + 1), dtype=complex)

    for row, (mean, amplitudes, phases) in enumerate(
            ((sigma_mean, sigma_amplitudes, sigma_phases),
             (capacity_mean, capacity_amplitudes, capacity_phases))):
        if not (isinstance(mean, (int, float, np.floating, np.integer))
                and np.isfinite(mean) and float(mean) > 0.0):
            raise ValueError("each profile mean must be a finite number > 0")
        amp = np.atleast_1d(np.asarray(amplitudes, dtype=float)).ravel()
        pha = np.atleast_1d(np.asarray(phases, dtype=float)).ravel()
        if amp.shape != pha.shape:
            raise ValueError("amplitudes and phases must have the same length")
        if not (np.all(np.isfinite(amp)) and np.all(np.isfinite(pha))):
            raise ValueError("amplitudes and phases must be finite")
        if amp.size > order:
            raise ValueError("order must be at least the highest harmonic present")

        profile = np.full(grid.shape, float(mean))
        for harmonic, (a, p) in enumerate(zip(amp, pha), start=1):
            profile = profile + float(mean) * a * np.cos(harmonic * grid + p)
            # Realness of the profile pairs opposite indices as conjugates.
            modes[row, order + harmonic] = 0.5 * float(mean) * a * np.exp(1j * p)
            modes[row, order - harmonic] = 0.5 * float(mean) * a * np.exp(-1j * p)
        if np.min(profile) <= 0.0:
            raise ValueError("conductivity and capacity profiles must stay positive")
        modes[row, order] = float(mean)

    return modes

import numpy as np


def assemble_secular_matrix(alpha: complex,
                                    sigma_modes: "np.ndarray",
                                    capacity_modes: "np.ndarray",
                                    beta: float,
                                    modulation_speed: float) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    sigma = np.asarray(sigma_modes, dtype=complex)
    capacity = np.asarray(capacity_modes, dtype=complex)
    if sigma.ndim != 1 or sigma.size < 3 or sigma.size % 2 == 0:
        raise ValueError("sigma_modes must be a 1D array of odd length >= 3")
    if capacity.shape != sigma.shape:
        raise ValueError("capacity_modes must have the same shape as sigma_modes")
    if not (np.all(np.isfinite(sigma)) and np.all(np.isfinite(capacity))):
        raise ValueError("sigma_modes and capacity_modes must be finite")
    if not (isinstance(beta, (int, float, np.floating, np.integer))
            and np.isfinite(beta) and float(beta) != 0.0):
        raise ValueError("beta must be a finite nonzero number")
    if not (isinstance(modulation_speed, (int, float, np.floating, np.integer))
            and np.isfinite(modulation_speed)):
        raise ValueError("modulation_speed must be a finite number")
    alpha = complex(alpha)
    if not np.isfinite(alpha.real) or not np.isfinite(alpha.imag):
        raise ValueError("alpha must be finite")

    order = (sigma.size - 1) // 2
    indices = np.arange(-order, order + 1)
    rows = indices[:, None]
    cols = indices[None, :]

    # Toeplitz pickers: entry (n, m) reads the coefficient of harmonic n - m.
    difference = rows - cols
    inside = np.abs(difference) <= order
    sigma_toeplitz = np.where(inside, sigma[np.clip(difference + order, 0, 2 * order)], 0.0)
    capacity_toeplitz = np.where(inside, capacity[np.clip(difference + order, 0, 2 * order)], 0.0)

    conduction = (alpha + 1j * float(beta) * rows) * (alpha + 1j * float(beta) * cols) * sigma_toeplitz
    storage = 1j * float(beta) * rows * float(modulation_speed) * capacity_toeplitz
    return conduction + storage

import numpy as np


def solve_secular_roots(sigma_modes: "np.ndarray",
                                capacity_modes: "np.ndarray",
                                beta: float,
                                modulation_speed: float) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    sigma = np.asarray(sigma_modes, dtype=complex)
    capacity = np.asarray(capacity_modes, dtype=complex)
    if sigma.ndim != 1 or sigma.size < 3 or sigma.size % 2 == 0:
        raise ValueError("sigma_modes must be a 1D array of odd length >= 3")
    if capacity.shape != sigma.shape:
        raise ValueError("capacity_modes must have the same shape as sigma_modes")
    if not (np.all(np.isfinite(sigma)) and np.all(np.isfinite(capacity))):
        raise ValueError("sigma_modes and capacity_modes must be finite")
    if not (isinstance(beta, (int, float, np.floating, np.integer))
            and np.isfinite(beta) and float(beta) != 0.0):
        raise ValueError("beta must be a finite nonzero number")
    if not (isinstance(modulation_speed, (int, float, np.floating, np.integer))
            and np.isfinite(modulation_speed)):
        raise ValueError("modulation_speed must be a finite number")

    order = (sigma.size - 1) // 2
    size = 2 * order + 1
    indices = np.arange(-order, order + 1)
    rows = indices[:, None]
    cols = indices[None, :]
    difference = rows - cols
    inside = np.abs(difference) <= order
    sigma_toeplitz = np.where(inside, sigma[np.clip(difference + order, 0, 2 * order)], 0.0)
    capacity_toeplitz = np.where(inside, capacity[np.clip(difference + order, 0, 2 * order)], 0.0)

    quadratic = sigma_toeplitz
    linear = 1j * float(beta) * (rows + cols) * sigma_toeplitz
    constant = (-(float(beta) ** 2) * rows * cols * sigma_toeplitz
                + 1j * float(beta) * rows * float(modulation_speed) * capacity_toeplitz)
    try:
        reduced = np.linalg.solve(quadratic, np.hstack([constant, linear]))
    except np.linalg.LinAlgError as exc:
        raise ValueError("the conductivity Toeplitz matrix must be invertible") from exc

    companion = np.vstack([np.hstack([np.zeros((size, size), dtype=complex), np.eye(size, dtype=complex)]),
                           -reduced])
    alphas = np.linalg.eigvals(companion)
    keys = np.lexsort((alphas.imag, np.round(alphas.real, 6)))
    return alphas[keys]

import numpy as np


def compute_bloch_fourier_components(alpha: complex,
                                             sigma_modes: "np.ndarray",
                                             capacity_modes: "np.ndarray",
                                             beta: float,
                                             modulation_speed: float) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.  The coupling matrix itself comes from the
    # earlier step's own reference implementation, which the executing namespace
    # supplies, so the two steps cannot drift apart and every input check that
    # step declares is applied here too.
    import numpy as np

    matrix = np.asarray(assemble_secular_matrix(
        alpha, sigma_modes, capacity_modes, beta, modulation_speed))
    order = (matrix.shape[0] - 1) // 2

    # Deleting the zeroth row and column leaves the system that the
    # normalisation of the zeroth coefficient to one makes inhomogeneous.
    keep = [i for i in range(2 * order + 1) if i != order]
    envelope = np.zeros(2 * order + 1, dtype=complex)
    envelope[order] = 1.0
    try:
        envelope[keep] = -np.linalg.solve(matrix[np.ix_(keep, keep)], matrix[keep, order])
    except np.linalg.LinAlgError as exc:
        raise ValueError("the reduced coupling matrix must be invertible") from exc
    return envelope

import numpy as np


def compute_boundary_coefficients(alphas: "np.ndarray",
                                          envelopes: "np.ndarray",
                                          position_high: float,
                                          position_low: float) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    roots = np.asarray(alphas, dtype=complex)
    modes = np.asarray(envelopes, dtype=complex)
    if roots.ndim != 1:
        raise ValueError("alphas must be a one-dimensional array")
    if modes.ndim != 2 or modes.shape[0] < 3 or modes.shape[0] % 2 == 0:
        raise ValueError("envelopes must be 2D with an odd number of rows >= 3")
    order = (modes.shape[0] - 1) // 2
    if roots.size != 4 * order + 2 or modes.shape[1] != 4 * order + 2:
        raise ValueError("alphas and envelopes must supply 4 * order + 2 states")
    if not (np.all(np.isfinite(roots)) and np.all(np.isfinite(modes))):
        raise ValueError("alphas and envelopes must be finite")
    for name, value in (("position_high", position_high), ("position_low", position_low)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
    if float(position_high) == float(position_low):
        raise ValueError("the two terminals must sit at different positions")

    # Exact column equilibration: rescaling a column rescales the amplitude of
    # that state, so the factor is put back when the inverse is read off.
    shift = np.maximum(roots.real * float(position_high), roots.real * float(position_low))
    scale = np.max(np.abs(modes), axis=0)
    if np.any(scale == 0.0):
        raise ValueError("every envelope must have a nonzero coefficient")
    factor = np.exp(-shift) / scale
    matrix = np.vstack([np.exp(roots * float(position_high) - shift) * modes / scale,
                        np.exp(roots * float(position_low) - shift) * modes / scale])
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the boundary matrix must be invertible") from exc

    star = int(np.argmin(np.abs(roots)))
    return factor[star] * np.array([inverse[star, order], inverse[star, 3 * order + 1]],
                                   dtype=complex)

import numpy as np


def compute_time_averaged_flux(sigma_modes: "np.ndarray",
                                       star_envelope: "np.ndarray",
                                       beta: float,
                                       boundary_coefficients: "np.ndarray",
                                       phi_high: float,
                                       phi_low: float) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    sigma = np.asarray(sigma_modes, dtype=complex)
    envelope = np.asarray(star_envelope, dtype=complex)
    coefficients = np.asarray(boundary_coefficients, dtype=complex)
    if sigma.ndim != 1 or sigma.size < 3 or sigma.size % 2 == 0:
        raise ValueError("sigma_modes must be a 1D array of odd length >= 3")
    if envelope.shape != sigma.shape:
        raise ValueError("star_envelope must have the same shape as sigma_modes")
    if coefficients.shape != (2,):
        raise ValueError("boundary_coefficients must hold exactly two entries")
    if not (np.all(np.isfinite(sigma)) and np.all(np.isfinite(envelope))
            and np.all(np.isfinite(coefficients))):
        raise ValueError("all coefficient arrays must be finite")
    if not (isinstance(beta, (int, float, np.floating, np.integer))
            and np.isfinite(beta) and float(beta) != 0.0):
        raise ValueError("beta must be a finite nonzero number")
    for name, value in (("phi_high", phi_high), ("phi_low", phi_low)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")

    order = (sigma.size - 1) // 2
    harmonics = np.arange(-order, order + 1)
    # Cell average of the conductivity acting on the derivative of the envelope.
    kernel = np.sum(1j * float(beta) * harmonics * sigma[order - harmonics] * envelope)
    amplitude = coefficients[0] * float(phi_high) + coefficients[1] * float(phi_low)
    return float(np.real(-amplitude * kernel))

import numpy as np


def compute_effective_parameters(sigma_modes: "np.ndarray",
                                         star_envelope: "np.ndarray",
                                         beta: float,
                                         boundary_coefficients: "np.ndarray",
                                         length: float) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    sigma = np.asarray(sigma_modes, dtype=complex)
    envelope = np.asarray(star_envelope, dtype=complex)
    coefficients = np.asarray(boundary_coefficients, dtype=complex)
    if sigma.ndim != 1 or sigma.size < 3 or sigma.size % 2 == 0:
        raise ValueError("sigma_modes must be a 1D array of odd length >= 3")
    if envelope.shape != sigma.shape:
        raise ValueError("star_envelope must have the same shape as sigma_modes")
    if coefficients.shape != (2,):
        raise ValueError("boundary_coefficients must hold exactly two entries")
    if not (np.all(np.isfinite(sigma)) and np.all(np.isfinite(envelope))
            and np.all(np.isfinite(coefficients))):
        raise ValueError("all coefficient arrays must be finite")
    if not (isinstance(beta, (int, float, np.floating, np.integer))
            and np.isfinite(beta) and float(beta) != 0.0):
        raise ValueError("beta must be a finite nonzero number")
    if not (isinstance(length, (int, float, np.floating, np.integer))
            and np.isfinite(length) and float(length) > 0.0):
        raise ValueError("length must be a finite number > 0")
    if coefficients[1] == 0.0:
        raise ValueError("the second boundary coefficient must be nonzero")

    quotient = -coefficients[0] / coefficients[1]
    imaginary_tolerance = 1.0e-10 * max(1.0, abs(float(quotient.real)))
    if quotient.real <= 0.0 or abs(float(quotient.imag)) > imaginary_tolerance:
        raise ValueError("the boundary coefficients must give a positive real "
                         "exponential factor")
    quotient = float(quotient.real)

    order = (sigma.size - 1) // 2
    harmonics = np.arange(-order, order + 1)
    kernel = np.sum(1j * float(beta) * harmonics * sigma[order - harmonics] * envelope)

    logarithm = np.log(quotient)
    if logarithm == 0.0:
        raise ValueError("boundary coefficients that cancel leave the effective "
                         "conductivity undetermined by this matching")
    advection_ratio = float(np.real(logarithm) / float(length))
    advective = float(np.real(-(coefficients[0] + coefficients[1]) * kernel))
    conductivity = float(np.real(-float(length) * (coefficients[0] + coefficients[1])
                                 * kernel / logarithm))
    return np.array([conductivity, advective, advection_ratio], dtype=float)

import numpy as np


def compute_rectification_ratio(advection_ratio: float,
                                        length: float,
                                        phi_high: float,
                                        phi_low: float) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    for name, value in (("advection_ratio", advection_ratio), ("phi_high", phi_high),
                        ("phi_low", phi_low)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
    if not (isinstance(length, (int, float, np.floating, np.integer))
            and np.isfinite(length) and float(length) > 0.0):
        raise ValueError("length must be a finite number > 0")

    exponent = float(advection_ratio) * float(length)
    terminal_scale = max(abs(float(phi_high)), abs(float(phi_low)))
    if terminal_scale == 0.0 or exponent == 0.0:
        return 0.0
    # Normalising the terminal potentials and using tanh(x/2) evaluates the
    # algebraically equivalent (exp(x)-1)/(exp(x)+1) without overflow.
    high = float(phi_high) / terminal_scale
    low = float(phi_low) / terminal_scale
    total = high + low
    if total == 0.0:
        return 0.0
    difference = high - low
    scaled_balance = abs(total) * abs(float(np.tanh(0.5 * exponent)))
    if scaled_balance == 0.0:
        return 0.0
    direction = float(np.sign(total) * np.sign(exponent))
    return float(2.0 * direction * scaled_balance
                 / (scaled_balance + abs(difference)))

import numpy as np


def compute_homogenized_limit(sigma_modes: "np.ndarray",
                                      capacity_modes: "np.ndarray",
                                      modulation_speed: float) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    sigma = np.asarray(sigma_modes, dtype=complex)
    capacity = np.asarray(capacity_modes, dtype=complex)
    if sigma.ndim != 1 or sigma.size < 3 or sigma.size % 2 == 0:
        raise ValueError("sigma_modes must be a 1D array of odd length >= 3")
    if capacity.shape != sigma.shape:
        raise ValueError("capacity_modes must have the same shape as sigma_modes")
    if not (np.all(np.isfinite(sigma)) and np.all(np.isfinite(capacity))):
        raise ValueError("sigma_modes and capacity_modes must be finite")
    if not (isinstance(modulation_speed, (int, float, np.floating, np.integer))
            and np.isfinite(modulation_speed)):
        raise ValueError("modulation_speed must be a finite number")

    order = (sigma.size - 1) // 2
    nonzero = np.array([n for n in range(-order, order + 1) if n != 0])
    rows = nonzero[:, None]
    cols = nonzero[None, :]
    difference = rows - cols
    inside = np.abs(difference) <= order
    picker = np.where(inside, sigma[np.clip(difference + order, 0, 2 * order)], 0.0)
    # Row index is the projected harmonic, column index the summed harmonic.
    coupling = rows * cols * picker
    try:
        inverse = np.linalg.inv(coupling)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the reduced conductivity coupling must be invertible") from exc

    left = nonzero * sigma[order - nonzero]
    right_sigma = nonzero * sigma[order + nonzero]
    right_capacity = nonzero * capacity[order + nonzero]
    conductivity = float(np.real(sigma[order] - left @ inverse @ right_sigma))
    advective = float(np.real(float(modulation_speed) * (left @ inverse @ right_capacity)))
    if conductivity <= 0.0:
        raise ValueError("the limiting effective conductivity must be positive")
    return np.array([conductivity, advective, advective / conductivity], dtype=float)

import numpy as np


def run_diffusive_circulator_pipeline(
        n_cells: int = 5, modulation_speed: float = 0.06, phi_high: float = 30.0,
        phi_low: float = 10.0, length: float = 1.0, sigma_mean: float = 1.0,
        capacity_mean: float = 100.0, sigma_amplitudes: tuple = (0.7, 0.2),
        sigma_phases: tuple = (0.0, 1.0471975511965976),
        capacity_amplitudes: tuple = (0.5,), capacity_phases: tuple = (1.2566370614359172,),
        order: int = 16) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.  Every earlier step is called through its own
    #  twin, which the executing namespace supplies, so the reference
    # chain never runs a submitted implementation.
    import numpy as np

    def _number(value, positive=False, nonzero=False):
        return (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(value) and (float(value) > 0.0 or not positive)
                and (float(value) != 0.0 or not nonzero))

    if not (isinstance(n_cells, (int, np.integer)) and not isinstance(n_cells, bool)
            and int(n_cells) >= 1):
        raise ValueError("n_cells must be an integer >= 1")
    if not _number(length, positive=True):
        raise ValueError("length must be a finite number > 0")
    if not _number(modulation_speed, nonzero=True):
        raise ValueError("modulation_speed must be a finite nonzero number")

    beta = 2.0 * np.pi * int(n_cells) / float(length)
    modes = compute_modulation_fourier_coefficients(
        sigma_mean, np.asarray(sigma_amplitudes, dtype=float),
        np.asarray(sigma_phases, dtype=float), capacity_mean,
        np.asarray(capacity_amplitudes, dtype=float),
        np.asarray(capacity_phases, dtype=float), order)
    sigma_modes, capacity_modes = modes[0], modes[1]
    middle = (np.asarray(sigma_modes).size - 1) // 2
    alphas = solve_secular_roots(sigma_modes, capacity_modes, beta, modulation_speed)
    envelopes = np.column_stack([compute_bloch_fourier_components(
        a, sigma_modes, capacity_modes, beta, modulation_speed) for a in alphas])
    star = int(np.argmin(np.abs(alphas)))
    star_envelope = envelopes[:, star]

    # Each envelope must be annihilated by the coupling matrix at its own
    # decay constant - tightly for the non-decaying state, loosely for the
    # fastest decaying one - and the matrix must carry the conductivity
    # quadratically far from every root.
    for index, tolerance in ((star, 1.0e-12), (int(np.argmax(np.abs(alphas))), 1.0e-6)):
        coupling = np.asarray(assemble_secular_matrix(
            alphas[index], sigma_modes, capacity_modes, beta, modulation_speed))
        column = envelopes[:, index]
        if np.linalg.norm(coupling @ column) > tolerance * np.linalg.norm(
                coupling) * np.linalg.norm(column):
            raise ValueError("an envelope does not annihilate the coupling matrix")
    far = np.asarray(assemble_secular_matrix(
        1.0e6, sigma_modes, capacity_modes, beta, modulation_speed))
    if abs(far[middle, middle] / 1.0e12 - sigma_modes[middle]) > 1.0e-6 * abs(sigma_modes[middle]):
        raise ValueError("the coupling matrix does not carry the conductivity at leading order")

    coefficients = compute_boundary_coefficients(alphas, envelopes, 0.0, float(length))
    forward = compute_time_averaged_flux(sigma_modes, star_envelope, beta, coefficients, phi_high, phi_low)
    backward = compute_time_averaged_flux(sigma_modes, star_envelope, beta, coefficients, phi_low, phi_high)
    largest = max(abs(forward), abs(backward))
    if largest == 0.0:
        raise ValueError("the branch passes no flux in either direction")
    branch = (forward + backward) / largest
    parameters = compute_effective_parameters(sigma_modes, star_envelope, beta, coefficients, length)
    closed_form = compute_rectification_ratio(float(parameters[2]), length, phi_high, phi_low)
    if abs(closed_form - branch) > 1.0e-6 * max(1.0, abs(branch)):
        raise ValueError("the flux and equivalence routes disagree on the rectification")

    limit = compute_homogenized_limit(sigma_modes, capacity_modes, modulation_speed)
    limiting = compute_rectification_ratio(float(limit[2]), length, phi_high, phi_low)
    if limiting == 0.0:
        raise ValueError("the homogenised description shows no rectification to compare with")
    return float(branch / limiting)
SCICODE_GOLD_EOF
