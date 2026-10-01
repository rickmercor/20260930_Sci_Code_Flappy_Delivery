#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import math

ELEMENTARY_CHARGE = 1.602176634e-19


def _positive(value, label):
    """Return an argument as a float once it is known to be finite and above zero."""
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("%s must be finite" % label)
    if out <= 0.0:
        raise ValueError("%s must be above zero" % label)
    return out


def resolve_phase_conductivities(
    donor_density_matrix: float,
    mobility_matrix: float,
    donor_density_inclusion: float,
    mobility_inclusion: float,
    area_fraction: float,
) -> dict:
    """Reference implementation."""
    nd1 = _positive(donor_density_matrix, "donor_density_matrix")
    mu1 = _positive(mobility_matrix, "mobility_matrix")
    nd2 = _positive(donor_density_inclusion, "donor_density_inclusion")
    mu2 = _positive(mobility_inclusion, "mobility_inclusion")
    f = float(area_fraction)
    if not math.isfinite(f):
        raise ValueError("area_fraction must be finite")
    if not 0.0 < f < 1.0:
        raise ValueError("area_fraction must lie strictly between zero and one")

    c1 = ELEMENTARY_CHARGE * nd1 * mu1
    c2 = ELEMENTARY_CHARGE * nd2 * mu2
    if c2 <= c1:
        raise ValueError("the embedded phase must be the more conducting of the two")

    wiener_upper = f * c2 + (1.0 - f) * c1
    wiener_lower = 1.0 / (f / c2 + (1.0 - f) / c1)
    # two dimensions, so the denominator carries 2 c_m rather than 3 c_m
    hs_lower = c1 + f / (1.0 / (c2 - c1) + (1.0 - f) / (2.0 * c1))
    hs_upper = c2 + (1.0 - f) / (1.0 / (c1 - c2) + f / (2.0 * c2))
    return {
        "c_matrix": c1,
        "c_inclusion": c2,
        "contrast": c2 / c1,
        "wiener_lower": wiener_lower,
        "wiener_upper": wiener_upper,
        "hs_lower": hs_lower,
        "hs_upper": hs_upper,
    }

import math

import numpy as np


def _even_positive(value, label):
    """Return an argument as an int once it is known to be a positive even integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be an integer" % label)
    out = int(value)
    if out <= 0 or out % 2 != 0:
        raise ValueError("%s must be a positive even integer" % label)
    return out


def _positive_float(value, label):
    """Return an argument as a float once it is known to be finite and above zero."""
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError("%s must be finite and above zero" % label)
    return out


def _green_from_symbol(symbol, reference):
    """Assemble conj(beta) tensor beta divided by reference times conj(beta) . beta."""
    n = symbol.shape[1]
    weight = np.real(np.sum(np.conj(symbol) * symbol, axis=0))
    green = np.zeros((2, 2, n, n), complex)
    live = weight > 0.0
    for a in range(2):
        for b in range(2):
            numerator = np.conj(symbol[a]) * symbol[b]
            green[a, b][live] = numerator[live] / (reference * weight[live])
    return green


def _projector_defects(green, reference, retained):
    """Return the departures of reference times green from idempotence and from unit trace."""
    scaled = reference * green
    square = np.zeros_like(scaled)
    for a in range(2):
        for b in range(2):
            square[a, b] = scaled[a, 0] * scaled[0, b] + scaled[a, 1] * scaled[1, b]
    idem = 0.0
    for a in range(2):
        for b in range(2):
            idem = max(idem, float(np.max(np.abs((square[a, b] - scaled[a, b])[retained]))))
    trace = float(np.max(np.abs((scaled[0, 0] + scaled[1, 1])[retained] - 1.0)))
    return idem, trace


def spectral_lattice(
    n_grid: int,
    n_modes: int,
    period: float,
    reference: float,
) -> dict:
    """Reference implementation."""
    big_n = _even_positive(n_grid, "n_grid")
    small_m = _even_positive(n_modes, "n_modes")
    if small_m > big_n:
        raise ValueError("n_modes must not exceed n_grid")
    length = _positive_float(period, "period")
    c0 = _positive_float(reference, "reference")

    n = big_n + 1
    index = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    wave = 2.0 * np.pi * index / length
    wavevector = np.zeros((2, n, n))
    wavevector[0] = wave[:, None]
    wavevector[1] = wave[None, :]
    symbol = wavevector.astype(complex)

    keep = np.abs(index) <= small_m // 2
    retained = keep[:, None] & keep[None, :]
    retained[0, 0] = False

    green = _green_from_symbol(symbol, c0)
    idem, trace = _projector_defects(green, c0, retained)
    return {
        "frequency_index": index,
        "wavevector": wavevector,
        "symbol": symbol,
        "green": green,
        "retained": retained,
        "n_points": n,
        "n_retained": int(retained.sum()),
        "spacing": length / n,
        "projector_defect": idem,
        "trace_defect": trace,
    }

import math

import numpy as np


def _even_positive(value, label):
    """Return an argument as an int once it is known to be a positive even integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be an integer" % label)
    out = int(value)
    if out <= 0 or out % 2 != 0:
        raise ValueError("%s must be a positive even integer" % label)
    return out


def _positive_float(value, label):
    """Return an argument as a float once it is known to be finite and above zero."""
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError("%s must be finite and above zero" % label)
    return out


def checkerboard_conductivity(
    n_grid: int,
    c_matrix: float,
    c_inclusion: float,
    period: float,
) -> dict:
    """Reference implementation."""
    big_n = _even_positive(n_grid, "n_grid")
    c1 = _positive_float(c_matrix, "c_matrix")
    c2 = _positive_float(c_inclusion, "c_inclusion")
    length = _positive_float(period, "period")

    if c2 <= c1:
        raise ValueError("c_inclusion must exceed c_matrix")

    n = big_n + 1
    coordinate = np.arange(n) * (length / n)
    inside = coordinate < 0.5 * length
    conductivity = np.where(inside[:, None] & inside[None, :], c2, c1)
    count = int(inside.sum())
    sampled = float(count) ** 2 / float(n) ** 2
    return {
        "conductivity": conductivity,
        "coordinate": coordinate,
        "inclusion_points": count,
        "sampled_area_fraction": sampled,
        "exact_area_fraction": 0.25,
        "fraction_defect": sampled - 0.25,
    }

import math

import numpy as np


def _positive_float(value, label):
    """Return an argument as a float once it is known to be finite and above zero."""
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError("%s must be finite and above zero" % label)
    return out


def moulinec_suquet_field(
    conductivity,
    symbol,
    green,
    retained,
    reference: float,
    mean_field,
    period: float,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Reference implementation."""
    c = np.asarray(conductivity, dtype=float)
    beta = np.asarray(symbol)
    gamma = np.asarray(green)
    keep = np.asarray(retained, dtype=bool)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("conductivity must be a square two-dimensional array")
    n = c.shape[0]
    if beta.shape != (2, n, n):
        raise ValueError("symbol must have shape (2, n, n)")
    if gamma.shape != (2, 2, n, n):
        raise ValueError("green must have shape (2, 2, n, n)")
    if keep.shape != (n, n):
        raise ValueError("retained must have shape (n, n)")
    if not np.all(np.isfinite(c)) or np.any(c <= 0.0):
        raise ValueError("conductivity must be finite and everywhere above zero")
    if not keep.any():
        raise ValueError("at least one mode must be retained")

    c0 = _positive_float(reference, "reference")
    length = _positive_float(period, "period")
    tol = _positive_float(tolerance, "tolerance")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)):
        raise ValueError("max_iterations must be an integer")
    cap = int(max_iterations)
    if cap <= 0:
        raise ValueError("max_iterations must be above zero")

    mean = np.asarray(mean_field, dtype=float).ravel()
    if mean.size != 2 or not np.all(np.isfinite(mean)):
        raise ValueError("mean_field must be two finite components")
    mean_norm_sq = float(mean @ mean)
    if mean_norm_sq <= 0.0:
        raise ValueError("mean_field must not vanish")

    scale = length / (2.0 * np.pi)
    field = np.zeros((2, n, n))
    field[0] = mean[0]
    field[1] = mean[1]
    transform = None
    residual = float("inf")
    taken = 0
    for step in range(1, cap + 1):
        taken = step
        current = c * field
        current_hat = np.fft.fft2(current, axes=(1, 2))
        divergence = beta[0] * current_hat[0] + beta[1] * current_hat[1]
        mean_current = np.array([current_hat[0, 0, 0].real, current_hat[1, 0, 0].real])
        residual = scale * float(np.sqrt(np.sum(np.abs(divergence[keep]) ** 2))
                                 / np.linalg.norm(mean_current))
        if residual < tol:
            break
        if step == cap:
            # the residual has been measured and failed; a further update would be discarded
            break
        polarisation = current_hat - c0 * np.fft.fft2(field, axes=(1, 2))
        transform = np.zeros((2, n, n), complex)
        for a in range(2):
            transform[a] = -(gamma[a, 0] * polarisation[0] + gamma[a, 1] * polarisation[1])
        transform[0][~keep] = 0.0
        transform[1][~keep] = 0.0
        transform[0, 0, 0] = mean[0] * n * n
        transform[1, 0, 0] = mean[1] * n * n
        field = np.real(np.fft.ifft2(transform, axes=(1, 2)))
    if residual >= tol:
        raise ValueError("the residual did not fall below the tolerance within max_iterations sweeps")
    if transform is None:
        # the starting uniform field already satisfies equilibrium on the retained modes
        transform = np.fft.fft2(field, axes=(1, 2))

    mean_current = np.array([current_hat[0, 0, 0].real, current_hat[1, 0, 0].real]) / n ** 2
    c_num = float(mean_current @ mean) / mean_norm_sq
    energy = float(np.mean(c * (field[0] ** 2 + field[1] ** 2)))
    defect = abs(energy - c_num * mean_norm_sq) / (c_num * mean_norm_sq)
    return {
        "electric_field": field,
        "field_transform": transform,
        "iterations": taken,
        "residual": residual,
        "macroscopic_conductivity": c_num,
        "hill_mandel_defect": defect,
    }

import numpy as np


def _even_positive(value, label):
    """Return an argument as an int once it is known to be a positive even integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be an integer" % label)
    out = int(value)
    if out <= 0 or out % 2 != 0:
        raise ValueError("%s must be a positive even integer" % label)
    return out


def spectral_reconstruction(
    field_transform,
    n_grid: int,
    n_modes: int,
    refinement: int,
) -> dict:
    """Reference implementation."""
    big_n = _even_positive(n_grid, "n_grid")
    small_m = _even_positive(n_modes, "n_modes")
    if small_m > big_n:
        raise ValueError("n_modes must not exceed n_grid")
    if isinstance(refinement, bool) or not isinstance(refinement, (int, np.integer)):
        raise ValueError("refinement must be an integer")
    factor = int(refinement)
    if factor <= 0:
        raise ValueError("refinement must be above zero")

    transform = np.asarray(field_transform)
    n = big_n + 1
    if transform.shape != (2, n, n):
        raise ValueError("field_transform must have shape (2, n_grid + 1, n_grid + 1)")

    n_fine = factor * n
    index = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    # the Fourier series coefficients, the transform carrying the trapezoidal factor
    series = transform / float(n) ** 2
    chosen = np.where(np.abs(index) <= small_m // 2)[0]
    # position by frequency index modulo the fine length, not by array position
    rows = index[chosen] % n_fine
    padded = np.zeros((2, n_fine, n_fine), complex)
    padded[np.ix_([0, 1], rows, rows)] = series[np.ix_([0, 1], chosen, chosen)]
    fine = np.real(np.fft.ifft2(padded * float(n_fine) ** 2, axes=(1, 2)))
    coarse = np.real(np.fft.ifft2(transform, axes=(1, 2)))

    scale = float(np.max(np.abs(coarse)))
    if scale == 0.0:
        scale = 1.0
    shared = fine[:, ::factor, ::factor]
    node_defect = float(np.max(np.abs(shared - coarse))) / scale
    fine_peak = float(np.max(np.abs(fine[0])))
    coarse_peak = float(np.max(np.abs(coarse[0])))
    ratio = fine_peak / coarse_peak if coarse_peak > 0.0 else float("inf")
    return {
        "fine_field": fine,
        "coarse_field": coarse,
        "n_fine": n_fine,
        "node_defect": node_defect,
        "peak_ratio": ratio,
        "fine_peak": fine_peak,
        "coarse_peak": coarse_peak,
    }

import math

import numpy as np


def _even_positive(value, label):
    """Return an argument as an int once it is known to be a positive even integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be an integer" % label)
    out = int(value)
    if out <= 0 or out % 2 != 0:
        raise ValueError("%s must be a positive even integer" % label)
    return out


def _positive_float(value, label):
    """Return an argument as a float once it is known to be finite and above zero."""
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError("%s must be finite and above zero" % label)
    return out


def _half_cell_weights(n_fine):
    """Return the exact cell-averaged integral of each exponential over the lower half period."""
    index = np.fft.fftfreq(n_fine, d=1.0 / n_fine).astype(int)
    weights = np.zeros(n_fine, complex)
    weights[index == 0] = 0.5
    odd = (index % 2 != 0)
    weights[odd] = 1j / (np.pi * index[odd])
    return weights


def energy_norm_error(
    fine_field,
    n_modes: int,
    c_matrix: float,
    c_inclusion: float,
    mean_field,
) -> dict:
    """Reference implementation."""
    small_m = _even_positive(n_modes, "n_modes")
    c1 = _positive_float(c_matrix, "c_matrix")
    c2 = _positive_float(c_inclusion, "c_inclusion")

    field = np.asarray(fine_field, dtype=float)
    if field.ndim != 3 or field.shape[0] != 2 or field.shape[1] != field.shape[2]:
        raise ValueError("fine_field must have shape (2, nf, nf)")
    n_fine = field.shape[1]
    # the squared magnitude reaches index M in each direction, so every index from -M to M
    # must be separately representable by a transform of this length
    reach = np.fft.fftfreq(n_fine, d=1.0 / n_fine).astype(int)
    if int(reach.min()) > -small_m or int(reach.max()) < small_m:
        raise ValueError("fine_field must carry enough points to represent every index from "
                         "minus n_modes to n_modes")
    if not np.all(np.isfinite(field)):
        raise ValueError("fine_field must be finite")

    mean = np.asarray(mean_field, dtype=float).ravel()
    if mean.size != 2 or not np.all(np.isfinite(mean)):
        raise ValueError("mean_field must be two finite components")
    mean_norm_sq = float(mean @ mean)
    if mean_norm_sq <= 0.0:
        raise ValueError("mean_field must not vanish")

    square = field[0] ** 2 + field[1] ** 2
    # exact because the square of the magnitude is a trigonometric polynomial of order 2M
    coefficients = np.fft.fft2(square) / float(n_fine) ** 2
    weights = _half_cell_weights(n_fine)
    inclusion = float(np.real(np.sum(coefficients * weights[:, None] * weights[None, :])))
    mean_square = float(np.real(coefficients[0, 0]))

    energy = c1 * mean_square + (c2 - c1) * inclusion
    exact = c1 * math.sqrt((c1 + 3.0 * c2) / (3.0 * c1 + c2))
    apparent = energy / mean_norm_sq
    return {
        "cell_energy": energy,
        "mean_square_field": mean_square,
        "inclusion_integral": inclusion,
        "exact_macroscopic": exact,
        "apparent_conductivity": apparent,
        "energy_norm_error": apparent / exact - 1.0,
    }

import math

import numpy as np


def _positive_float(value, label):
    """Return an argument as a float once it is known to be finite and above zero."""
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError("%s must be finite and above zero" % label)
    return out


def _green_from_symbol(symbol, reference):
    """Assemble conj(beta) tensor beta divided by reference times conj(beta) . beta."""
    n = symbol.shape[1]
    weight = np.real(np.sum(np.conj(symbol) * symbol, axis=0))
    green = np.zeros((2, 2, n, n), complex)
    live = weight > 0.0
    for a in range(2):
        for b in range(2):
            numerator = np.conj(symbol[a]) * symbol[b]
            green[a, b][live] = numerator[live] / (reference * weight[live])
    return green


def _projector_defects(green, reference, retained):
    """Return the departures of reference times green from idempotence and from unit trace."""
    scaled = reference * green
    square = np.zeros_like(scaled)
    for a in range(2):
        for b in range(2):
            square[a, b] = scaled[a, 0] * scaled[0, b] + scaled[a, 1] * scaled[1, b]
    idem = 0.0
    for a in range(2):
        for b in range(2):
            idem = max(idem, float(np.max(np.abs((square[a, b] - scaled[a, b])[retained]))))
    trace = float(np.max(np.abs((scaled[0, 0] + scaled[1, 1])[retained] - 1.0)))
    return idem, trace


def generalised_green_operator(
    wavevector,
    spacing: float,
    reference: float,
    retained,
) -> dict:
    """Reference implementation."""
    wave = np.asarray(wavevector, dtype=float)
    if wave.ndim != 3 or wave.shape[0] != 2 or wave.shape[1] != wave.shape[2]:
        raise ValueError("wavevector must have shape (2, n, n)")
    if not np.all(np.isfinite(wave)):
        raise ValueError("wavevector must be finite")
    n = wave.shape[1]
    keep = np.asarray(retained, dtype=bool)
    if keep.shape != (n, n):
        raise ValueError("retained must have shape (n, n)")
    if not keep.any():
        raise ValueError("at least one mode must be retained")
    h = _positive_float(spacing, "spacing")
    c0 = _positive_float(reference, "reference")

    symbol = (1.0 - np.exp(-1j * wave * h)) / (1j * h)
    alpha = 2.0 * np.sin(wave * h / 2.0) / h
    green = _green_from_symbol(symbol, c0)

    scale = float(np.max(np.abs(wave)))
    if scale == 0.0:
        scale = 1.0
    symbol_defect = float(np.max(np.abs(np.conj(symbol) * symbol - alpha ** 2))) / scale ** 2
    spectral_defect = float(np.max(np.abs(symbol - wave))) / scale
    idem, trace = _projector_defects(green, c0, keep)
    return {
        "symbol": symbol,
        "alpha": alpha,
        "green": green,
        "symbol_defect": symbol_defect,
        "spectral_defect": spectral_defect,
        "projector_defect": idem,
        "trace_defect": trace,
    }

import math

import numpy as np


def _even_positive(value, label):
    """Return an argument as an int once it is known to be a positive even integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be an integer" % label)
    out = int(value)
    if out <= 0 or out % 2 != 0:
        raise ValueError("%s must be a positive even integer" % label)
    return out


def _positive_float(value, label):
    """Return an argument as a float once it is known to be finite and above zero."""
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError("%s must be finite and above zero" % label)
    return out


def optimal_truncation_ratio(
    donor_density_matrix: float,
    mobility_matrix: float,
    donor_density_inclusion: float,
    mobility_inclusion: float,
    period: float,
    mean_field,
    n_grid: int,
    mode_step: int,
    refinement: int,
    spacing_ratio: float,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Reference implementation: reruns the whole chain once per candidate truncation."""
    big_n = _even_positive(n_grid, "n_grid")
    step = _even_positive(mode_step, "mode_step")
    if big_n % step != 0:
        raise ValueError("mode_step must divide n_grid")
    if isinstance(refinement, bool) or not isinstance(refinement, (int, np.integer)):
        raise ValueError("refinement must be an integer")
    factor = int(refinement)
    if factor < 2:
        raise ValueError("refinement must be at least two")
    length = _positive_float(period, "period")
    ratio_h = _positive_float(spacing_ratio, "spacing_ratio")
    tol = _positive_float(tolerance, "tolerance")

    phases = resolve_phase_conductivities(                      # noqa: F821
        donor_density_matrix, mobility_matrix,
        donor_density_inclusion, mobility_inclusion, 0.25,
    )
    c1 = phases["c_matrix"]
    c2 = phases["c_inclusion"]
    c0 = 0.5 * (c1 + c2)

    board = checkerboard_conductivity(big_n, c1, c2, length)    # noqa: F821
    conductivity = board["conductivity"]
    spacing = length / (big_n + 1)

    candidates = tuple(range(step, big_n + 1, step))
    errors = []
    macroscopic = []
    peaks = []
    iterations = []
    generalised_excess = []
    for modes in candidates:
        lattice = spectral_lattice(big_n, modes, length, c0)    # noqa: F821
        solved = moulinec_suquet_field(                        # noqa: F821
            conductivity, lattice["symbol"], lattice["green"], lattice["retained"],
            c0, mean_field, length, tol, max_iterations,
        )
        rebuilt = spectral_reconstruction(                     # noqa: F821
            solved["field_transform"], big_n, modes, factor,
        )
        scored = energy_norm_error(                            # noqa: F821
            rebuilt["fine_field"], modes, c1, c2, mean_field,
        )
        errors.append(scored["energy_norm_error"])
        macroscopic.append(solved["macroscopic_conductivity"])
        peaks.append(rebuilt["peak_ratio"])
        iterations.append(solved["iterations"])

        modified = generalised_green_operator(                  # noqa: F821
            lattice["wavevector"], ratio_h * spacing, c0, lattice["retained"],
        )
        solved_g = moulinec_suquet_field(                       # noqa: F821
            conductivity, modified["symbol"], modified["green"], lattice["retained"],
            c0, mean_field, length, tol, max_iterations,
        )
        rebuilt_g = spectral_reconstruction(                    # noqa: F821
            solved_g["field_transform"], big_n, modes, factor,
        )
        scored_g = energy_norm_error(                           # noqa: F821
            rebuilt_g["fine_field"], modes, c1, c2, mean_field,
        )
        # not an energy-norm error: the difference rule breaks the compatibility the
        # identity needs, so this is an apparent-conductivity excess
        generalised_excess.append(scored_g["energy_norm_error"])

    best = int(np.argmin(errors))
    best_g = int(np.argmin(generalised_excess))
    exact = energy_norm_error(                                  # noqa: F821
        np.ones((2, 4 * step + 1, 4 * step + 1)), step, c1, c2, (1.0, 0.0),
    )["exact_macroscopic"]
    full = len(candidates) - 1
    return {
        "optimal_ratio": candidates[best] / float(big_n),
        "optimal_modes": int(candidates[best]),
        "optimal_error": errors[best],
        "error_at_full": errors[full],
        "error_ratio": errors[full] / errors[best],
        "macroscopic_at_optimum": macroscopic[best],
        "macroscopic_at_full": macroscopic[full],
        "exact_macroscopic": exact,
        "macro_error_at_optimum": abs(macroscopic[best] - exact) / exact,
        "macro_error_at_full": abs(macroscopic[full] - exact) / exact,
        "c_matrix": c1,
        "c_inclusion": c2,
        "contrast": phases["contrast"],
        "sampled_area_fraction": board["sampled_area_fraction"],
        "peak_ratio_at_optimum": peaks[best],
        "peak_ratio_at_full": peaks[full],
        "iterations_at_optimum": int(iterations[best]),
        "generalised_optimal_ratio": candidates[best_g] / float(big_n),
        "generalised_optimal_modes": int(candidates[best_g]),
        "generalised_optimal_excess": generalised_excess[best_g],
        "candidates": candidates,
        "errors": tuple(errors),
    }
SCICODE_GOLD_EOF
