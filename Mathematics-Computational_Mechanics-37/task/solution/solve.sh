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


def _hexagonal_keys():
    """Keys of the five free hexagonal entries, in the order this file consumes them."""
    return ("c11", "c33", "c12", "c13", "c44")


def _voigt_map():
    """Voigt slot carried by every index pair, in the ordering (11, 22, 33, 23, 13, 12)."""
    return np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])


def _voigt_axes():
    """First and second tensor index of each of the six Voigt slots."""
    return np.array([0, 1, 2, 1, 0, 0]), np.array([0, 1, 2, 2, 2, 1])


def _hexagonal_matrix(values):
    """Voigt 6 by 6 stiffness built from the five free entries c11, c33, c12, c13 and c44."""
    c11, c33, c12, c13, c44 = (float(entry) for entry in values)
    c66 = 0.5 * (c11 - c12)
    return np.array([
        [c11, c12, c13, 0.0, 0.0, 0.0],
        [c12, c11, c13, 0.0, 0.0, 0.0],
        [c13, c13, c33, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c44, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, c44, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, c66],
    ])


def _spin_about_c(angle):
    """Right-handed rotation about the third axis, angle in radians."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _tilt_about_a(angle):
    """Right-handed rotation about the first axis, angle in radians."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])


def _seated_lattice(matrix, angles_in_degrees):
    """Rotated compliance of one grain, in Voigt form, for the given orientation triple."""
    slots = _voigt_map()
    tensor = matrix[slots[:, :, None, None], slots[None, None, :, :]]
    first, tilt, second = np.radians(angles_in_degrees)
    rotation = _spin_about_c(first) @ _tilt_about_a(tilt) @ _spin_about_c(second)
    turned = np.einsum("pa,qb,rc,sd,abcd->pqrs", rotation, rotation, rotation, rotation, tensor)
    rows, columns = _voigt_axes()
    return np.linalg.inv(turned[rows[:, None], columns[:, None], rows, columns])


def reduce_oriented_grains(
    euler_angles: np.ndarray,
    crystal_constants: dict,
    grain_voxels: np.ndarray,
    specimen_density: float,
    bar_length: float,
    n_modes: int,
) -> dict:
    """Reference implementation."""
    angles = np.asarray(euler_angles, dtype=float)
    if angles.ndim != 2 or angles.shape[1] != 3 or angles.shape[0] < 1:
        raise ValueError("euler_angles wants shape (n_grains, 3) and at least one grain")
    if not np.isfinite(angles).all():
        raise ValueError("no orientation angle may be infinite or undefined")
    absent = [key for key in _hexagonal_keys() if key not in crystal_constants]
    if absent:
        raise ValueError("crystal_constants lacks the entry %s" % absent[0])
    entries = np.array([float(crystal_constants[key]) for key in _hexagonal_keys()])
    if not np.isfinite(entries).all() or entries.min() <= 0.0:
        raise ValueError("each hexagonal constant wants a finite value above zero")
    reference = _hexagonal_matrix(entries)
    if np.linalg.eigvalsh(reference).min() <= 0.0:
        raise ValueError("the five entries do not give a positive definite crystal stiffness")
    counts = np.asarray(grain_voxels)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer) or counts.shape[0] != angles.shape[0]:
        raise ValueError("grain_voxels wants one integer entry per orientation row")
    if counts.min() < 1:
        raise ValueError("a grain of zero voxels cannot be laid out")
    mass_density = float(specimen_density)
    span = float(bar_length)
    if not np.isfinite(mass_density) or not np.isfinite(span) or mass_density <= 0.0 or span <= 0.0:
        raise ValueError("both specimen_density and bar_length want a finite value above zero")
    if not isinstance(n_modes, (int, np.integer)) or int(n_modes) < 1:
        raise ValueError("n_modes wants an integer of one or more")

    n_grains = angles.shape[0]
    axial_compliance = np.zeros(n_grains)
    lateral_compliance = np.zeros(n_grains)
    for grain, triple in enumerate(angles):
        compliance = _seated_lattice(reference, triple)
        axial_compliance[grain] = compliance[0, 0]
        lateral_compliance[grain] = compliance[0, 1]
    young = 1.0 / axial_compliance
    poisson = -lateral_compliance / axial_compliance
    if poisson.min() <= -1.0 or poisson.max() >= 0.5:
        raise ValueError("a surrogate Poisson ratio left the open interval (-1, 1/2)")

    weights = counts / counts.sum()
    young_hom = 1.0 / np.sum(weights / young)
    poisson_hom = float(np.sum(weights * poisson))
    constrained_hom = young_hom * (1.0 - poisson_hom) / ((1.0 + poisson_hom) * (1.0 - 2.0 * poisson_hom))
    shear_hom = young_hom / (2.0 * (1.0 + poisson_hom))
    fast = np.sqrt(constrained_hom / mass_density)
    slow = np.sqrt(shear_hom / mass_density)
    rungs = np.arange(1, int(n_modes) + 1)
    return {
        "young": young,
        "poisson": poisson,
        "longitudinal_modulus": young * (1.0 - poisson) / ((1.0 + poisson) * (1.0 - 2.0 * poisson)),
        "shear_modulus": young / (2.0 * (1.0 + poisson)),
        "young_hom": float(young_hom),
        "poisson_hom": poisson_hom,
        "longitudinal_modulus_hom": float(constrained_hom),
        "shear_modulus_hom": float(shear_hom),
        "longitudinal_speed": float(fast),
        "transverse_speed": float(slow),
        "analytic_longitudinal": rungs * fast / (2.0 * span),
    }

import numpy as np


def _counted(value, label):
    """Return a count as a native int once it is known to be an integer of one or more."""
    if not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError("%s wants an integer of one or more" % label)
    return int(value)


def _checked_factor(value, label):
    """Return a padding factor as a float once it is known to be finite and non-negative."""
    factor = float(value)
    if not np.isfinite(factor):
        raise ValueError("the %s factor must be finite" % label)
    if factor < 0.0:
        raise ValueError("the %s factor must not be negative" % label)
    return factor


def _link_moduli(voxel_row, n_specimen):
    """Modulus of every link, harmonic inside the specimen and prescribed wherever padding is involved."""
    left = voxel_row
    right = np.roll(voxel_row, -1)
    total = left + right
    series = 2.0 * left * right / np.where(total > 0.0, total, 1.0)
    on_left = np.arange(voxel_row.size) < n_specimen
    on_right = np.roll(on_left, -1)
    inside = on_left & on_right
    straddling = on_left ^ on_right
    links = np.where(inside, series, left)
    return np.where(straddling, np.where(on_left, left, right), links)


def assemble_face_stiffness_cell(
    grain_voxels: np.ndarray,
    longitudinal_modulus: np.ndarray,
    shear_modulus: np.ndarray,
    specimen_density: float,
    n_pad: int,
    pad_stiffness_factor: float,
    pad_density_factor: float,
) -> dict:
    """Reference implementation."""
    counts = np.asarray(grain_voxels)
    if counts.ndim != 1 or counts.size < 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("grain_voxels wants a one-dimensional array of integers")
    if counts.min() < 1:
        raise ValueError("a grain of zero voxels cannot be laid out")
    axial = np.asarray(longitudinal_modulus, dtype=float)
    lateral = np.asarray(shear_modulus, dtype=float)
    if axial.shape != counts.shape or lateral.shape != counts.shape:
        raise ValueError("one longitudinal and one shear modulus are wanted per grain")
    if not (np.isfinite(axial).all() and np.isfinite(lateral).all()):
        raise ValueError("no grain modulus may be infinite or undefined")
    if min(axial.min(), lateral.min()) <= 0.0:
        raise ValueError("every grain modulus must sit above zero")
    specimen_density = float(specimen_density)
    if not np.isfinite(specimen_density) or specimen_density <= 0.0:
        raise ValueError("density wants a finite value above zero")
    pad = _counted(n_pad, "n_pad")
    stiffness_ratio = _checked_factor(pad_stiffness_factor, "padding stiffness")
    density_ratio = _checked_factor(pad_density_factor, "padding density")

    n_specimen = int(counts.sum())
    n_total = n_specimen + pad
    if n_total % 2 == 0:
        raise ValueError("this configuration lays out an odd cell, and %d is even" % n_total)

    axial_voxels = np.repeat(axial, counts)
    lateral_voxels = np.repeat(lateral, counts)
    axial_row = np.concatenate([axial_voxels, np.full(pad, stiffness_ratio * axial_voxels.mean())])
    lateral_row = np.concatenate([lateral_voxels, np.full(pad, stiffness_ratio * lateral_voxels.mean())])
    voxel_modulus = np.stack([axial_row, lateral_row, lateral_row])
    voxel_density = np.concatenate([
        np.full(n_specimen, specimen_density), np.full(pad, density_ratio * specimen_density)])
    return {
        "n_specimen": n_specimen,
        "n_total": n_total,
        "density": voxel_density,
        "voxel_modulus": voxel_modulus,
        "face_modulus": np.stack([_link_moduli(row, n_specimen) for row in voxel_modulus]),
    }

import numpy as np
from scipy.linalg import eigh


def _link_difference_symbol(n_total, voxel_edge):
    """Transform multiplier that shifts a field by one voxel and subtracts, over an odd cell."""
    if n_total % 2 == 0:
        raise ValueError("an even voxel count is outside this configuration")
    modes = np.fft.fftfreq(n_total, d=1.0 / n_total)
    return (np.exp(2j * np.pi * modes / n_total) - 1.0) / voxel_edge


def _stiffness_action(field, face_row, symbol):
    """Extend every link, weight it by its own modulus, and gather the forces back onto the voxels."""
    extension = np.fft.ifft(symbol * np.fft.fft(field, axis=-1), axis=-1).real
    force = np.fft.fft(face_row * extension, axis=-1)
    return np.fft.ifft(np.conj(symbol) * force, axis=-1).real


def _dense_operator(face_row, symbol):
    """Stiffness operator of one family, read off its action on the whole identity at once."""
    images = _stiffness_action(np.eye(face_row.size), face_row, symbol)
    return 0.5 * (images + images.T)


def _family_spectrum(face_row, mass, symbol, n_specimen, n_modes):
    """Natural frequencies of one family once the massless padding has been condensed away."""
    operator = _dense_operator(face_row, symbol)
    kept = operator[:n_specimen, :n_specimen]
    coupling = operator[:n_specimen, n_specimen:]
    padding = operator[n_specimen:, n_specimen:]
    reduced = kept - coupling @ np.linalg.solve(padding, coupling.T)
    reduced = 0.5 * (reduced + reduced.T)
    squared = eigh(reduced, np.diag(mass[:n_specimen]), eigvals_only=True)
    return np.sqrt(np.maximum(squared, 0.0))[1:n_modes + 1] / (2.0 * np.pi)


def solve_padded_eigenproblem(
    face_modulus: np.ndarray,
    voxel_density: np.ndarray,
    voxel_edge: float,
    n_specimen: int,
    n_modes: int,
) -> dict:
    """Reference implementation."""
    links = np.asarray(face_modulus, dtype=float)
    mass = np.asarray(voxel_density, dtype=float)
    if links.ndim != 2 or links.shape[0] != 3:
        raise ValueError("face_modulus wants three rows, one per displacement component")
    n_total = links.shape[1]
    if mass.shape != (n_total,):
        raise ValueError("density wants exactly one entry per voxel")
    if n_total % 2 == 0:
        raise ValueError("an even voxel count is outside this configuration")
    edge = float(voxel_edge)
    if edge <= 0.0:
        raise ValueError("voxel_size wants a value above zero")
    if not isinstance(n_specimen, (int, np.integer)) or not 1 <= int(n_specimen) <= n_total - 1:
        raise ValueError("n_specimen wants an integer between one and n_total minus one")
    kept = int(n_specimen)
    if mass[:kept].min() <= 0.0 or np.any(mass[kept:] != 0.0):
        raise ValueError("mass belongs on the specimen voxels and nowhere else")
    if links[:, kept:].min() <= 0.0:
        raise ValueError("every link beyond the specimen wants a modulus above zero")
    if not isinstance(n_modes, (int, np.integer)) or not 1 <= int(n_modes) <= kept - 1:
        raise ValueError("n_modes wants an integer between one and n_specimen minus one")
    wanted = int(n_modes)

    symbol = _link_difference_symbol(n_total, edge)
    return {
        "longitudinal": _family_spectrum(links[0], mass, symbol, kept, wanted),
        "transverse": _family_spectrum(links[1], mass, symbol, kept, wanted),
    }

import numpy as np


def _counted(value, label):
    """Return a count as a native int once it is known to be an integer of one or more."""
    if not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError("%s wants an integer of one or more" % label)
    return int(value)


def _link_difference_symbol(n_total, voxel_edge):
    """Transform multiplier that shifts a field by one voxel and subtracts, over an odd cell."""
    if n_total % 2 == 0:
        raise ValueError("an even voxel count is outside this configuration")
    modes = np.fft.fftfreq(n_total, d=1.0 / n_total)
    return (np.exp(2j * np.pi * modes / n_total) - 1.0) / voxel_edge


def _stiffness_action(field, face_row, symbol):
    """Extend every link, weight it by its own modulus, and gather the forces back onto the voxels."""
    extension = np.fft.ifft(symbol * np.fft.fft(field, axis=-1), axis=-1).real
    force = np.fft.fft(face_row * extension, axis=-1)
    return np.fft.ifft(np.conj(symbol) * force, axis=-1).real


def _reference_inverse(face_modulus, voxel_density, symbol, coefficient):
    """Per-component multiplier that inverts the operator of the volume-averaged reference medium."""
    row_average = face_modulus.mean(axis=1, keepdims=True)
    return 1.0 / (voxel_density.mean() + coefficient * row_average * np.abs(symbol) ** 2)


def invert_implicit_step(
    rhs: np.ndarray,
    face_modulus: np.ndarray,
    voxel_density: np.ndarray,
    voxel_edge: float,
    newmark_coefficient: float,
    residual_tolerance: float,
    iteration_ceiling: int,
) -> dict:
    """Reference implementation."""
    load = np.asarray(rhs, dtype=float)
    links = np.asarray(face_modulus, dtype=float)
    mass = np.asarray(voxel_density, dtype=float)
    if load.ndim != 2 or load.shape[0] != 3:
        raise ValueError("rhs wants three rows, one per displacement component")
    if links.shape != load.shape:
        raise ValueError("face_modulus wants the same shape as rhs")
    n_total = load.shape[1]
    if mass.shape != (n_total,) or mass.min() <= 0.0:
        raise ValueError("density wants one entry per voxel, every one of them above zero")
    if n_total % 2 == 0:
        raise ValueError("an even voxel count is outside this configuration")
    edge = float(voxel_edge)
    target = float(residual_tolerance)
    if edge <= 0.0 or target <= 0.0:
        raise ValueError("voxel_size and residual_tolerance both want values above zero")
    coefficient = float(newmark_coefficient)
    if coefficient < 0.0:
        raise ValueError("beta_dt2 must not be negative")
    cap = _counted(iteration_ceiling, "max_iterations")

    symbol = _link_difference_symbol(n_total, edge)
    multiplier = _reference_inverse(links, mass, symbol, coefficient)

    def _smooth(vector):
        return np.fft.ifft(multiplier * np.fft.fft(vector, axis=-1), axis=-1).real

    def _forward(vector):
        return mass * vector + coefficient * _stiffness_action(vector, links, symbol)

    scale = np.linalg.norm(load)
    if scale == 0.0:
        blank = np.zeros_like(load)
        return {"solution": blank, "operator_image": blank.copy(), "iterations": 0, "residual": 0.0}

    estimate = _smooth(load)
    image = _forward(estimate)
    defect = load - image
    smoothed = _smooth(defect)
    search = smoothed.copy()
    energy = float(np.sum(defect * smoothed))
    iterations = 0
    for _ in range(cap):
        if np.linalg.norm(defect) <= target * scale:
            break
        mapped = _forward(search)
        alpha = energy / float(np.sum(search * mapped))
        estimate = estimate + alpha * search
        image = image + alpha * mapped
        defect = defect - alpha * mapped
        smoothed = _smooth(defect)
        refreshed = float(np.sum(defect * smoothed))
        search = smoothed + (refreshed / energy) * search
        energy = refreshed
        iterations += 1

    achieved = float(np.linalg.norm(defect) / scale)
    if achieved > target:
        raise RuntimeError(
            "conjugate gradients exhausted the iteration ceiling at a relative residual of %g" % achieved
        )
    return {"solution": estimate, "operator_image": image, "iterations": iterations, "residual": achieved}

import numpy as np


def _counted(value, label):
    """Return a count as a native int once it is known to be an integer of one or more."""
    if not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError("%s wants an integer of one or more" % label)
    return int(value)


def _gaussian_doublet(time, amplitude, width, centre):
    """Two Gaussians of one width, their centres one width apart, the later subtracted from the earlier."""
    leading = np.exp(-(time - centre) ** 2 / (2.0 * width ** 2))
    trailing = np.exp(-(time - centre - width) ** 2 / (2.0 * width ** 2))
    return amplitude * (leading - trailing)


def _voxel_index(value, label, n_total):
    """Return an integer voxel index once it is known to address a voxel of the cell."""
    if not isinstance(value, (int, np.integer)):
        raise ValueError("%s wants an integer" % label)
    index = int(value)
    if not 0 <= index < n_total:
        raise ValueError("%s falls outside the cell" % label)
    return index


def march_pulsed_bar(
    face_modulus: np.ndarray,
    voxel_density: np.ndarray,
    voxel_edge: float,
    time_step: float,
    n_steps: int,
    newmark_beta: float,
    newmark_gamma: float,
    traction_amplitude: float,
    gaussian_width: float,
    gaussian_centre: float,
    traction_axis: np.ndarray,
    struck_voxel: int,
    receptor_voxel: int,
    residual_tolerance: float,
    iteration_ceiling: int,
) -> dict:
    """Reference implementation."""
    links = np.asarray(face_modulus, dtype=float)
    mass = np.asarray(voxel_density, dtype=float)
    if links.ndim != 2 or links.shape[0] != 3:
        raise ValueError("face_modulus wants three rows, one per displacement component")
    n_total = links.shape[1]
    if mass.shape != (n_total,) or mass.min() <= 0.0:
        raise ValueError("density wants one entry per voxel, every one of them above zero")
    if n_total % 2 == 0:
        raise ValueError("an even voxel count is outside this configuration")
    edge = float(voxel_edge)
    increment = float(time_step)
    width = float(gaussian_width)
    if min(edge, increment, width) <= 0.0:
        raise ValueError("voxel_size, time_step and gaussian_width all want values above zero")
    steps = _counted(n_steps, "n_steps")
    beta = float(newmark_beta)
    gamma = float(newmark_gamma)
    if not 0.0 < beta <= 1.0 or not 0.0 < gamma <= 1.0:
        raise ValueError("both Newmark parameters want a value in (0, 1]")
    aim = np.asarray(traction_axis, dtype=float)
    if aim.shape != (3,) or np.linalg.norm(aim) == 0.0:
        raise ValueError("pulse_direction wants a non-zero vector of three entries")
    source = _voxel_index(struck_voxel, "loaded_voxel", n_total)
    probe = _voxel_index(receptor_voxel, "receptor_voxel", n_total)

    unit = aim / np.linalg.norm(aim)
    coefficient = beta * increment * increment
    lag = increment * increment * (0.5 - beta)
    amplitude = float(traction_amplitude)
    centre = float(gaussian_centre)
    target = float(residual_tolerance)

    displacement = np.zeros((3, n_total))
    velocity = np.zeros((3, n_total))
    acceleration = np.zeros((3, n_total))
    body_force = np.zeros((3, n_total))
    history = np.zeros((steps + 1, 3))
    spent = 0
    for step in range(1, steps + 1):
        forecast = displacement + increment * velocity + lag * acceleration
        traction = _gaussian_doublet(step * increment, amplitude, width, centre)
        body_force[:, source] = traction * unit / edge
        outcome = invert_implicit_step(  # noqa: F821
            coefficient * body_force + mass * forecast, links, mass, edge, coefficient,
            target, iteration_ceiling,
        )
        if float(outcome["residual"]) > target:
            raise RuntimeError("step %d left the implicit system unconverged" % step)
        spent += outcome["iterations"]
        advanced = outcome["solution"]
        rate = (advanced - forecast) / coefficient
        velocity = velocity + increment * ((1.0 - gamma) * acceleration + gamma * rate)
        displacement = advanced
        acceleration = rate
        history[step] = displacement[:, probe]
    return {
        "record": history,
        "displacement": displacement,
        "velocity": velocity,
        "acceleration": acceleration,
        "mean_iterations": spent / steps,
    }

import numpy as np


def _log_vertex(low, middle, high):
    """Vertex position in bins, and the discrete curvature, of the parabola through three log amplitudes."""
    left, centre, right = np.log(low), np.log(middle), np.log(high)
    curvature = left - 2.0 * centre + right
    return float(0.5 * (left - right) / curvature), float(curvature)


def fit_subbin_resonance_lines(
    receptor_record: np.ndarray,
    time_step: float,
    target_frequencies: np.ndarray,
    search_fraction: float,
) -> dict:
    """Reference implementation."""
    samples = np.asarray(receptor_record, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 3 or samples.shape[0] < 2:
        raise ValueError("record wants shape (n_samples, 3) and two samples at the very least")
    if not np.isfinite(samples).all():
        raise ValueError("no recorded displacement may be infinite or undefined")
    interval = float(time_step)
    if interval <= 0.0:
        raise ValueError("dt wants a value above zero")
    targets = np.asarray(target_frequencies, dtype=float)
    if targets.ndim != 1 or targets.size < 1:
        raise ValueError("target_frequencies wants a one-dimensional array of at least one entry")
    if not np.isfinite(targets).all() or targets.min() <= 0.0:
        raise ValueError("every target frequency wants a finite value above zero")
    fraction = float(search_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("search_fraction wants a value inside the open interval (0, 1)")

    n_samples = samples.shape[0]
    frequencies = np.fft.rfftfreq(n_samples, d=interval)
    intensity = np.sqrt(np.sum(np.abs(np.fft.rfft(samples, axis=0)) ** 2, axis=1))
    spacing = float(frequencies[1] - frequencies[0])

    index = np.zeros(targets.size, dtype=np.int64)
    offset = np.zeros(targets.size)
    curvature = np.zeros(targets.size)
    crown = np.zeros(targets.size)
    sharpened = np.zeros(targets.size)
    for slot, target in enumerate(targets):
        inside = (frequencies >= target * (1.0 - fraction)) & (frequencies <= target * (1.0 + fraction))
        if not inside.any():
            raise ValueError("the window around %g hertz admits no frequency bin" % target)
        winner = int(np.argmax(np.where(inside, intensity, -np.inf)))
        if not 1 <= winner <= intensity.size - 2:
            raise ValueError("a winning bin sits at an end of the spectrum and cannot be sharpened")
        neighbourhood = intensity[winner - 1:winner + 2]
        if neighbourhood.min() <= 0.0:
            raise ValueError("a winning bin and both neighbours want strictly positive amplitude")
        low, middle, high = (float(entry) for entry in neighbourhood)
        vertex, bend = _log_vertex(low, middle, high)
        if bend >= 0.0:
            raise ValueError("the three log amplitudes give a parabola that does not open downwards")
        index[slot] = winner
        offset[slot] = vertex
        curvature[slot] = bend
        crown[slot] = middle
        sharpened[slot] = (winner + vertex) * spacing
    return {
        "frequencies": frequencies,
        "intensity": intensity,
        "bin_spacing": spacing,
        "line_index": index,
        "line_offset": offset,
        "line_curvature": curvature,
        "line_intensity": crown,
        "refined_frequency": sharpened,
    }

import numpy as np


def _newmark_pair():
    """The average-acceleration parameters used throughout this configuration."""
    return 0.25, 0.5


def _padding_factors():
    """Stiffness and specimen_density factors of the harmonic padding, then of the padding used by the march."""
    return (1.0e-7, 0.0), (0.0, 1.0)


def report_boundary_inertia_gap(
    euler_angles: np.ndarray,
    crystal_constants: dict,
    grain_voxels: np.ndarray,
    specimen_density: float,
    bar_length: float,
    n_pad: int,
    time_step: float,
    n_steps: int,
    traction_amplitude: float,
    gaussian_width: float,
    gaussian_centre: float,
    traction_axis: np.ndarray,
    ceiling: float,
    search_fraction: float,
    residual_tolerance: float,
    iteration_ceiling: int,
) -> dict:
    """Reference implementation chaining every earlier stage."""
    cap_frequency = float(ceiling)
    if not np.isfinite(cap_frequency) or cap_frequency <= 0.0:
        raise ValueError("ceiling wants a finite value above zero")
    newmark_beta, newmark_gamma = _newmark_pair()
    harmonic_padding, marching_padding = _padding_factors()

    material = reduce_oriented_grains(  # noqa: F821
        euler_angles, crystal_constants, grain_voxels, specimen_density, bar_length, 3
    )
    axial, lateral = material["longitudinal_modulus"], material["shear_modulus"]
    harmonic_cell = assemble_face_stiffness_cell(  # noqa: F821
        grain_voxels, axial, lateral, specimen_density, n_pad, harmonic_padding[0], harmonic_padding[1],
    )
    n_specimen = harmonic_cell["n_specimen"]
    edge = float(bar_length) / n_specimen

    # the baseline is established before the bar is struck, because the graded quantity is a
    # departure from it rather than a reading in its own right
    wanted = min(n_specimen - 1, 24)
    baseline = solve_padded_eigenproblem(  # noqa: F821
        harmonic_cell["face_modulus"], harmonic_cell["density"], edge, n_specimen, wanted
    )
    targets = baseline["longitudinal"][baseline["longitudinal"] < cap_frequency]
    if targets.size < 1:
        raise ValueError("no axial resonance of the specimen falls below the ceiling")
    both = np.concatenate([baseline["longitudinal"], baseline["transverse"]])

    marching_cell = assemble_face_stiffness_cell(  # noqa: F821
        grain_voxels, axial, lateral, specimen_density, n_pad, marching_padding[0], marching_padding[1],
    )
    march = march_pulsed_bar(  # noqa: F821
        marching_cell["face_modulus"], marching_cell["density"], edge, time_step, n_steps,
        newmark_beta, newmark_gamma, traction_amplitude, gaussian_width, gaussian_centre,
        traction_axis, 0, n_specimen - 1, residual_tolerance, iteration_ceiling,
    )
    lines = fit_subbin_resonance_lines(  # noqa: F821
        march["record"], time_step, targets, search_fraction
    )

    gap = lines["refined_frequency"] - targets
    mean_gap = float(gap.mean())
    relative = mean_gap / float(targets.mean())
    voxel_fraction = 2.0 / n_specimen
    return {
        "mean_gap": mean_gap,
        "mode_count": int(targets.size),
        "resonance_count": int((both < cap_frequency).sum()),
        "eigen_longitudinal": targets,
        "refined_line": lines["refined_frequency"],
        "gap": gap,
        "mean_relative_shift": relative,
        "voxel_mass_fraction": voxel_fraction,
        "face_mass_ratio": -relative / voxel_fraction,
        "first_offset": float(lines["line_offset"][0]),
        "first_curvature": float(lines["line_curvature"][0]),
        "bin_spacing": lines["bin_spacing"],
        "first_transverse": float(baseline["transverse"][0]),
        "analytic_first_longitudinal": float(material["analytic_longitudinal"][0]),
        "young_hom": material["young_hom"],
        "poisson_hom": material["poisson_hom"],
        "longitudinal_speed": material["longitudinal_speed"],
        "mean_cg_iterations": march["mean_iterations"],
    }
SCICODE_GOLD_EOF
