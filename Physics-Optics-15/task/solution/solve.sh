#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

"""Interpolate the first-principles dielectric-tensor field."""

import numpy as np

from scipy.interpolate import PchipInterpolator

def material_fields(
    mu_nodes: np.ndarray,
    tensors: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    nodes = np.asarray(mu_nodes, float)
    values = np.asarray(tensors, float)
    points = np.asarray(queries, float)
    if (nodes.ndim != 1 or points.ndim != 1 or values.ndim != 3
            or values.shape[0] != len(nodes) or values.shape[2] != 6):
        raise ValueError("expected nodes (J,), tensors (J,E,6), queries (Q,)")
    if (len(nodes) < 3 or not np.all(np.isfinite(nodes))
            or not np.all(np.isfinite(values)) or not np.all(np.isfinite(points))
            or np.any(np.diff(nodes) <= 0)):
        raise ValueError("tensor interpolation inputs must be finite and ordered")
    if np.any(points < nodes[0]) or np.any(points > nodes[-1]):
        raise ValueError("queries lie outside the interpolation domain")
    return PchipInterpolator(nodes, values, axis=0, extrapolate=False)(points)

def _material_fixture(nodes, energies):
    m = nodes[:, None]
    w = energies[None, :]
    return np.stack([
        260 * (w - .18 - .07 * m) + 3 * np.sin(11 * m),
        1.7 + 2.2 * w + .3 * m * m,
        240 * (w - .176 - .065 * m) + 2 * np.cos(9 * m),
        1.9 + 1.8 * w + .2 * m * m,
        .15 * np.cos(7 * m) + .02 * w,
        1.1 + .35 * np.sin(8 * m) + .4 * w,
    ], axis=-1)

"""Solve the semi-infinite anisotropic Voigt boundary problem."""

import numpy as np

def _outgoing_root(argument):
    root = np.sqrt(argument)
    reverse = (np.imag(root) < 0) | ((np.imag(root) == 0) & (np.real(root) < 0))
    return np.where(reverse, -root, root)

def _reflection(eps_x, eps_z, eps_xz, q, c):
    kz = _outgoing_root(eps_x + eps_xz**2 / eps_z - eps_x * q**2 / eps_z)
    denominator = eps_z - q**2
    admittance = (eps_z * kz - q * eps_xz) / denominator
    return (1 - c * admittance) / (1 + c * admittance)

def voigt_surface_responses(
    tensors: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    values = np.asarray(tensors, float)
    theta = np.asarray(angles, float)
    if values.ndim < 2 or values.shape[-1] != 6 or theta.ndim != 1:
        raise ValueError("expected tensor shape (...,E,6) and angle shape (A,)")
    if (not np.all(np.isfinite(values)) or not np.all(np.isfinite(theta))
            or np.any(theta <= 0) or np.any(theta >= 90)):
        raise ValueError("finite angles must lie strictly between 0 and 90 degrees")
    eps_x = values[..., 0] + 1j * values[..., 1]
    eps_z = values[..., 2] + 1j * values[..., 3]
    eps_xz = values[..., 4] + 1j * values[..., 5]
    q = np.sin(np.deg2rad(theta)).reshape((1,) * (eps_x.ndim - 1) + (len(theta), 1))
    c = np.cos(np.deg2rad(theta)).reshape((1,) * (eps_x.ndim - 1) + (len(theta), 1))
    ex = np.expand_dims(eps_x, -2)
    ez = np.expand_dims(eps_z, -2)
    exz = np.expand_dims(eps_xz, -2)
    if np.any(np.abs(ez) < 1e-14) or np.any(np.abs(ez - q**2) < 1e-14):
        raise ValueError("singular Voigt boundary problem")
    positive = _reflection(ex, ez, exz, q, c)
    negative = _reflection(ex, ez, exz, -q, c)
    result = np.abs(positive)**2 - np.abs(negative)**2
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite surface response")
    return np.asarray(result, float)

def _voigt_fixture(shape, count, phase):
    lead = int(np.prod(shape))
    u = np.linspace(-1, 1, lead * count).reshape(shape + (count,))
    ex = -1.4 + .8 * u + 1j * (1.1 + .15 * np.cos((phase + 2) * u))
    ez = -1.1 + .6 * u + 1j * (1.3 + .12 * np.sin((phase + 3) * u))
    exz = .04 * np.sin(3 * u) + 1j * (.28 + .05 * np.cos(5 * u))
    return np.stack((ex.real, ex.imag, ez.real, ez.imag, exz.real, exz.imag), axis=-1)

"""Extract signed Maxwell strength, support, and lobe separation."""

import numpy as np

from scipy.interpolate import PchipInterpolator

def _surface_single_metric(energies, row):
    sampled = np.asarray(row, float)
    scale = float(np.max(np.abs(sampled)))
    sign_floor = 64 * np.finfo(float).eps * max(1.0, scale)
    if np.max(sampled) <= sign_floor or np.min(sampled) >= -sign_floor:
        raise ValueError("each spectrum must contain positive and negative lobes")

    interpolant = PchipInterpolator(energies, row)
    stationary = interpolant.derivative().roots(extrapolate=False)
    points = np.unique(np.concatenate(([energies[0]], stationary, [energies[-1]])))
    points = points[(points >= energies[0]) & (points <= energies[-1])]
    values = np.asarray(interpolant(points), float)
    knot_indices = np.searchsorted(energies, points)
    at_knots = (knot_indices < len(energies))
    at_knots[at_knots] &= energies[knot_indices[at_knots]] == points[at_knots]
    values[at_knots] = sampled[knot_indices[at_knots]]

    positive = float(np.max(values))
    negative = float(np.min(values))
    if positive <= sign_floor or negative >= -sign_floor:
        raise ValueError("each spectrum must contain positive and negative lobes")
    positive_energy = float(np.min(points[values == positive]))
    negative_energy = float(np.min(points[values == negative]))
    pair_width = abs(positive_energy - negative_energy)
    if pair_width <= 0:
        raise ValueError("signed lobe extrema must occur at distinct energies")

    delta = max(positive, -negative)
    tied = points[np.abs(values) == delta]
    center = float(tied.min())
    peak = float(values[np.flatnonzero(points == center)[0]])
    sign = 1.0 if peak >= 0 else -1.0
    threshold = PchipInterpolator(energies, sign * np.asarray(row) - delta / 2)
    roots = threshold.roots(extrapolate=False)
    bounds = np.unique(np.concatenate(([energies[0]], roots, [energies[-1]])))
    bounds = bounds[(bounds >= energies[0]) & (bounds <= energies[-1])]
    good = np.asarray([
        threshold((left + right) / 2) >= 0
        for left, right in zip(bounds[:-1], bounds[1:])
    ])
    seeds = [
        index for index, (left, right) in enumerate(zip(bounds[:-1], bounds[1:]))
        if good[index] and left - 1e-14 <= center <= right + 1e-14
    ]
    if not seeds:
        raise ValueError("selected peak has no half-maximum support component")
    first = min(seeds)
    last = max(seeds)
    while first > 0 and good[first - 1]:
        first -= 1
    while last + 1 < len(good) and good[last + 1]:
        last += 1
    left = float(bounds[first])
    right = float(bounds[last + 1])
    bandwidth = right - left
    return np.array((
        delta, center, left, right, bandwidth, delta * bandwidth, pair_width
    ))

def full_maxwell_metrics(
    energies: np.ndarray,
    responses: np.ndarray,
) -> np.ndarray:
    x = np.asarray(energies, float)
    values = np.asarray(responses, float)
    if (x.ndim != 1 or len(x) < 5 or np.any(np.diff(x) <= 0)
            or not np.all(np.isfinite(x))):
        raise ValueError("energies must be finite, increasing, and length at least five")
    if (values.ndim < 2 or any(size == 0 for size in values.shape[:-1])
            or values.shape[-1] != len(x)
            or not np.all(np.isfinite(values))):
        raise ValueError("the finite final response axis must match energies")
    flat = values.reshape((-1, len(x)))
    result = np.asarray([_surface_single_metric(x, row) for row in flat])
    return result.reshape(values.shape[:-1] + (7,))

"""Recover the anchor paper's local ENZ design coordinates."""

import numpy as np

from scipy.interpolate import PchipInterpolator

def _single_root(interpolant, x, y):
    roots = np.asarray(interpolant.roots(extrapolate=False), float)
    tolerance = 64 * np.finfo(float).eps * max(1.0, abs(x[0]), abs(x[-1]))
    roots = roots[np.isfinite(roots)]
    roots = roots[(roots >= x[0] - tolerance) & (roots <= x[-1] + tolerance)]
    roots = np.where(np.abs(roots - x[0]) <= tolerance, x[0], roots)
    roots = np.where(np.abs(roots - x[-1]) <= tolerance, x[-1], roots)
    if y[0] == 0:
        roots = np.append(roots, x[0])
    if y[-1] == 0:
        roots = np.append(roots, x[-1])
    roots = np.sort(roots)
    unique = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > tolerance:
            unique.append(float(root))
    if len(unique) != 1:
        raise ValueError("Re xx must have exactly one ENZ crossing")
    return unique[0]

def enz_design_coordinates(spectra: list) -> np.ndarray:
    try:
        tables = list(spectra)
    except TypeError as error:
        raise ValueError("spectra must be a nonempty iterable") from error
    if not tables:
        raise ValueError("spectra must be a nonempty iterable")
    output = []
    for source in tables:
        table = np.asarray(source, float)
        if table.ndim != 2 or table.shape[0] < 5 or table.shape[1] != 7:
            raise ValueError("each spectrum must be a (K,7) table with K>=5")
        energy = table[:, 0]
        if np.any(np.diff(energy) <= 0) or not np.all(np.isfinite(table)):
            raise ValueError("spectra must be finite with increasing energies")
        columns = [PchipInterpolator(energy, table[:, j], extrapolate=False) for j in range(1, 7)]
        e0x = _single_root(columns[0], energy, table[:, 1])
        beta = abs(float(columns[0].derivative()(e0x)))
        loss = float(columns[1](e0x))
        real_off = float(columns[4](e0x))
        g = abs(float(columns[5](e0x)))
        primitives = np.asarray((beta, loss, real_off, g))
        if (not np.all(np.isfinite(primitives)) or beta <= 0
                or loss <= 0 or g <= 0):
            raise ValueError("beta, loss and gyrotropy must be positive")
        x = g / loss
        width_factor = 2 / np.sqrt(3) * np.sqrt(
            x*x - 1 + 2*np.sqrt(x**4 + x*x + 1)
        )
        derived = np.asarray((beta, loss, g, x, width_factor, abs(real_off) / g))
        if not np.all(np.isfinite(derived)):
            raise ValueError("derived ENZ coordinates must be finite")
        output.append(tuple(derived))
    return np.asarray(output)

"""Evaluate the anchor paper's universal strength and bandwidth laws."""

import numpy as np

def universal_law_metrics(
    coordinates: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    values = np.asarray(coordinates, float)
    theta = np.asarray(angles, float)
    if values.ndim != 2 or values.shape[1] != 6 or len(values) == 0:
        raise ValueError("coordinates must have finite nonempty shape (M,6)")
    if (not np.all(np.isfinite(values)) or np.any(values[:, :5] <= 0)
            or np.any(values[:, 5] < 0)):
        raise ValueError("coordinates must be finite and physical")
    if (theta.ndim != 1 or len(theta) == 0 or not np.all(np.isfinite(theta))
            or np.any(theta <= 0) or np.any(theta >= 90)):
        raise ValueError("angles must be finite and strictly between 0 and 90 degrees")
    beta, loss, g, x, width_factor, _ = values.T
    expected_x = g / loss
    expected_width = 2 / np.sqrt(3) * np.sqrt(
        x*x - 1 + 2*np.sqrt(x**4 + x*x + 1)
    )
    if (not np.allclose(x, expected_x, rtol=1e-10, atol=1e-12)
            or not np.allclose(width_factor, expected_width, rtol=1e-10, atol=1e-12)):
        raise ValueError("x and W must be consistent with beta, loss, and g")

    radians = np.deg2rad(theta)[None, :]
    sine = np.sin(radians)
    cosine = np.cos(radians)
    dimensionless_denominator = (
        ((width_factor / 2)**2 + 1 - x*x)**2 + 4*x*x
    )
    inverse_kappa = np.sqrt(loss * (1 + x*x) / 2)[:, None]
    scaled_angular = (
        sine * cosine * (1 + x[:, None]*x[:, None]) / 2
        / (1 + cosine * inverse_kappa)**2
    )
    dimensionless_kernel = np.abs(x * width_factor) / dimensionless_denominator
    delta = 8 * dimensionless_kernel[:, None] * scaled_angular
    bandwidth = np.broadcast_to((loss * width_factor / beta)[:, None], delta.shape)
    output = np.stack((delta, bandwidth, delta * bandwidth), axis=2)
    if not np.all(np.isfinite(output)):
        raise ValueError("universal-law metrics must be finite")
    return output

def _law_fixture():
    beta = np.array([120., 480., 900.])
    loss = np.array([.7, 3.2, 11.])
    g = np.array([.14, 2.4, 14.3])
    x = g / loss
    width = 2 / np.sqrt(3) * np.sqrt(x*x - 1 + 2*np.sqrt(x**4 + x*x + 1))
    coordinates = np.column_stack((beta, loss, g, x, width, [.03, .08, .12]))
    return coordinates

def _value_error_code(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    except Exception:
        return -1.0
    return 0.0

def _law_contract_args(coordinates, angles, kind):
    if kind == "shape":
        return np.ones((2, 5)), angles
    if kind == "empty_coordinates":
        return np.empty((0, 6)), angles
    if kind == "empty_angles":
        return coordinates, np.array([])
    if kind == "zero_angle":
        return coordinates, np.array([0.0, 30.0])
    if kind == "grazing_angle":
        return coordinates, np.array([20.0, 90.0])
    if kind == "nan":
        bad = coordinates.copy(); bad[0, 2] = np.nan
        return bad, angles
    if kind.startswith("column_"):
        column, value = {
            "column_0": (0, 0.0), "column_1": (1, 0.0),
            "column_2": (2, 0.0), "column_3": (3, 0.0),
            "column_4": (4, 0.0), "column_5": (5, -0.1),
        }[kind]
        bad = coordinates.copy(); bad[0, column] = value
        return bad, angles
    if kind == "x_identity":
        bad = coordinates.copy(); bad[0, 3] *= 1.01
        return bad, angles
    if kind == "width_identity":
        bad = coordinates.copy(); bad[0, 4] *= 0.99
        return bad, angles
    raise ValueError("unknown law contract fixture")

"""Apply universal-law validity and guarantee one Maxwell passband."""

import numpy as np

def _window_radius(grid, drift):
    spacing = np.diff(grid)
    if len(spacing) == 0 or not np.allclose(spacing, spacing[0], rtol=0, atol=1e-12):
        raise ValueError("design grids must be uniform")
    radius = int(round(float(drift) / spacing[0]))
    if radius < 0 or abs(radius * spacing[0] - drift) > 1e-10:
        raise ValueError("each drift must be a nonnegative integer grid multiple")
    return radius

def _first_window_location(mask):
    return tuple(np.argwhere(mask)[0])

def window_guarantees(
    mu: np.ndarray,
    angles: np.ndarray,
    metrics: np.ndarray,
    law_metrics: np.ndarray,
    validity: np.ndarray,
    drifts: np.ndarray,
    validity_limit: float,
    agreement_limits: np.ndarray,
    spectrum_valid: np.ndarray = None,
) -> np.ndarray:
    x = np.asarray(mu, float)
    theta = np.asarray(angles, float)
    values = np.asarray(metrics, float)
    laws = np.asarray(law_metrics, float)
    screens = np.asarray(validity, float)
    drift_values = np.asarray(drifts, float)
    agreement = np.asarray(agreement_limits, float)
    if (x.ndim != 1 or theta.ndim != 1 or len(x) < 2 or len(theta) < 2
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(theta))
            or np.any(np.diff(x) <= 0) or np.any(np.diff(theta) <= 0)):
        raise ValueError("design grids must be finite increasing one-dimensional arrays")
    if (drift_values.shape != (2,) or not np.all(np.isfinite(drift_values))
            or np.any(drift_values < 0)):
        raise ValueError("drifts must be a finite nonnegative length-2 vector")
    if values.shape != (len(x), len(theta), 7):
        raise ValueError("Maxwell metric shape mismatch")
    if laws.shape != (len(x), len(theta), 3) or screens.shape != (len(x),):
        raise ValueError("law metric or validity shape mismatch")
    if spectrum_valid is None:
        valid = np.ones((len(x), len(theta)), dtype=bool)
    else:
        valid_input = np.asarray(spectrum_valid)
        if valid_input.shape != (len(x), len(theta)) or valid_input.dtype != np.bool_:
            raise ValueError("spectrum_valid must be a Boolean grid-shaped mask")
        valid = valid_input
    arrays = (laws, screens, agreement)
    if any(not np.all(np.isfinite(item)) for item in arrays) or not np.isfinite(validity_limit):
        raise ValueError("all inputs and limits must be finite")
    if (agreement.shape != (2,) or np.any(agreement <= 0) or np.any(agreement >= 1)
            or validity_limit <= 0 or np.any(screens < 0)):
        raise ValueError("validity and agreement limits must be physical")
    valid_values = values[valid]
    if (not np.all(np.isfinite(valid_values))
            or np.any(valid_values[:, [0, 4, 5, 6]] <= 0) or np.any(laws <= 0)):
        raise ValueError("Maxwell and universal-law metrics must be positive")
    if (np.any(valid_values[:, 2] > valid_values[:, 1])
            or np.any(valid_values[:, 1] > valid_values[:, 3])
            or not np.allclose(valid_values[:, 3] - valid_values[:, 2], valid_values[:, 4], rtol=0, atol=1e-10)
            or not np.allclose(valid_values[:, 0] * valid_values[:, 4], valid_values[:, 5], rtol=0, atol=1e-10)
            or not np.allclose(laws[:, :, 0] * laws[:, :, 1], laws[:, :, 2], rtol=0, atol=1e-10)):
        raise ValueError("metric identities or half-maximum supports are invalid")

    rx = _window_radius(x, drift_values[0])
    rt = _window_radius(theta, drift_values[1])
    rows = []
    for i in range(rx, len(x) - rx):
        if np.any(screens[i-rx:i+rx+1] > validity_limit):
            continue
        for j in range(rt, len(theta) - rt):
            valid_block = valid[i-rx:i+rx+1, j-rt:j+rt+1]
            if not np.all(valid_block):
                continue
            strength_error = abs(laws[i, j, 0] - values[i, j, 0]) / values[i, j, 0]
            spacing_error = abs(laws[i, j, 1] - values[i, j, 6]) / values[i, j, 6]
            if strength_error > agreement[0] or spacing_error > agreement[1]:
                continue
            block = values[i-rx:i+rx+1, j-rt:j+rt+1]
            magnitudes = block[:, :, 0]
            dmin = float(np.min(magnitudes))
            common_left = float(np.max(block[:, :, 2]))
            common_right = float(np.min(block[:, :, 3]))
            if common_right <= common_left:
                continue
            di, dj = _first_window_location(magnitudes == dmin)
            li, lj = _first_window_location(block[:, :, 2] == common_left)
            ri, rj = _first_window_location(block[:, :, 3] == common_right)
            rows.append((
                x[i], theta[j], dmin, common_right - common_left,
                x[i-rx+di], theta[j-rt+dj], common_left, common_right,
                x[i-rx+li], theta[j-rt+lj], x[i-rx+ri], theta[j-rt+rj],
                strength_error, spacing_error, laws[i, j, 0], laws[i, j, 1],
            ))
    if not rows:
        raise ValueError("no admissible nominal node")
    return np.asarray(rows)

def _window_fixture(phase, n=9, a=7):
    x = np.linspace(-.4, .4, n)
    theta = np.linspace(5., 65., a)
    ii, jj = np.indices((n, a))
    peak = .42 + .017*ii + .011*jj + .003*phase
    center = .14 + .0017*ii - .0011*jj
    left = center - (.026 + .0008*((ii+2*jj+phase) % 4))
    right = center + (.024 + .0007*((2*ii+jj+phase) % 5))
    width = right - left
    pair = .06 + .001*ii + .0005*jj
    metrics = np.stack((peak, center, left, right, width, peak*width, pair), axis=2)
    law_peak = peak * (1 + .08*np.sin(.8*ii+.3*jj+phase))
    law_pair = pair * (1 + .06*np.cos(.4*ii+.7*jj+phase))
    laws = np.stack((law_peak, law_pair, law_peak*law_pair), axis=2)
    validity = .07 + .004*(np.arange(n)-3)**2
    return x, theta, metrics, laws, validity

def _value_error_code(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    except Exception:
        return -1.0
    return 0.0

def _window_contract_args(x, theta, metrics, laws, validity, drifts,
                          validity_limit, agreement, kind):
    args = [x, theta, metrics, laws, validity, drifts, validity_limit, agreement]
    if kind == "nonfinite_metric":
        bad = metrics.copy(); bad[3, 2, 0] = np.nan; args[2] = bad
    elif kind == "zero_pair":
        bad = metrics.copy(); bad[3, 2, 6] = 0.0; args[2] = bad
    elif kind == "width_identity":
        bad = metrics.copy(); bad[3, 2, 4] += 0.01; args[2] = bad
    elif kind == "law_product":
        bad = laws.copy(); bad[3, 2, 2] += 0.01; args[3] = bad
    elif kind == "nonfinite_validity":
        bad = validity.copy(); bad[2] = np.inf; args[4] = bad
    elif kind == "nonuniform_mu":
        bad = x.copy(); bad[4] += 0.01; args[0] = bad
    elif kind == "off_grid_drift":
        args[5] = (0.15, 10.0)
    elif kind == "zero_agreement":
        args[7] = (0.0, 0.1)
    elif kind == "unit_agreement":
        args[7] = (0.1, 1.0)
    elif kind == "strength_screen":
        bad = laws.copy(); bad[4, 3, 0] *= 1.5
        bad[4, 3, 2] = bad[4, 3, 0] * bad[4, 3, 1]; args[3] = bad
    elif kind == "spacing_screen":
        bad = laws.copy(); bad[4, 3, 1] *= 1.5
        bad[4, 3, 2] = bad[4, 3, 0] * bad[4, 3, 1]; args[3] = bad
    elif kind == "mu_shape":
        args[0] = x.reshape(1, -1)
    elif kind == "nonfinite_drift":
        args[5] = (0.1, np.nan)
    elif kind == "drift_shape":
        args[5] = (0.1,)
    else:
        raise ValueError("unknown window contract fixture")
    return tuple(args)

"""Select the strongest law-screened fixed-geometry common passband."""

import numpy as np

def select_operating_point(guarantees: np.ndarray) -> np.ndarray:
    values = np.asarray(guarantees, float)
    if values.ndim != 2 or values.shape[1] != 16 or len(values) == 0:
        raise ValueError("guarantee shape mismatch")
    if (not np.all(np.isfinite(values)) or np.any(values[:, 2:4] <= 0)
            or np.any(values[:, 12:14] < 0) or np.any(values[:, 14:16] <= 0)):
        raise ValueError("guarantees must be finite and physical")
    products = values[:, 2] * values[:, 3]
    best = float(np.max(products))
    indices = np.flatnonzero(products == best)
    index = min(indices, key=lambda k: (values[k, 0], values[k, 1]))
    row = values[index]
    return np.asarray((
        row[0], row[1], row[4], row[5], row[8], row[9], row[10], row[11],
        row[2], row[6], row[7], row[3], products[index], row[12], row[13],
        row[14], row[15],
    ))

def _value_error_code(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    except Exception:
        return -1.0
    return 0.0

def _selection_contract_table(guarantees, kind):
    if kind == "shape":
        return np.ones((2, 15))
    if kind == "empty":
        return np.empty((0, 16))
    bad = guarantees.copy()
    column, value = {
        "nan": (2, np.nan),
        "zero_dmin": (2, 0.0),
        "negative_error": (12, -0.1),
        "zero_law": (14, 0.0),
    }[kind]
    bad[0, column] = value
    return bad

"""Compose the law-screened target and robust full-Maxwell benchmark."""

import numpy as np

def _benchmark_common_guarantee(benchmark, angles, angle_drift):
    table = np.asarray(benchmark, float)
    theta = np.asarray(angles, float)
    if (table.ndim != 2 or table.shape[1] != 7 or len(table) < 5
            or not np.all(np.isfinite(table)) or not np.all(np.diff(table[:, 0]) > 0)):
        raise ValueError("benchmark must be an increasing finite (E,7) table")
    spacing = np.diff(theta)
    if (len(spacing) == 0
            or not np.allclose(spacing, spacing[0], rtol=0, atol=1e-12)):
        raise ValueError("angle grid must be uniform")
    radius = int(round(float(angle_drift) / spacing[0]))
    if (radius < 0
            or abs(radius * spacing[0] - angle_drift) > 1e-10):
        raise ValueError("angle drift must be a nonnegative integer grid multiple")
    responses = voigt_surface_responses(table[:, 1:], theta)
    metrics = {}
    for j in range(len(theta)):
        try:
            metrics[j] = full_maxwell_metrics(
                table[:, 0], responses[j:j+1]
            )[0]
        except ValueError:
            continue

    best = None
    for j in range(radius, len(theta) - radius):
        indices = range(j-radius, j+radius+1)
        if any(index not in metrics for index in indices):
            continue
        block = np.asarray([metrics[index] for index in indices])
        dmin = float(np.min(block[:, 0]))
        common_left = float(np.max(block[:, 2]))
        common_right = float(np.min(block[:, 3]))
        if common_right <= common_left:
            continue
        dloc = int(np.flatnonzero(block[:, 0] == dmin)[0])
        lloc = int(np.flatnonzero(block[:, 2] == common_left)[0])
        rloc = int(np.flatnonzero(block[:, 3] == common_right)[0])
        width = common_right - common_left
        row = np.asarray((
            theta[j], dmin, common_left, common_right, width, dmin*width,
            theta[j-radius+dloc], theta[j-radius+lloc], theta[j-radius+rloc],
        ))
        if best is None or row[5] > best[5]:
            best = row
    if best is None:
        raise ValueError("benchmark has no valid common alignment passband")
    return best

def robust_surface_advantage(
    mu_nodes: np.ndarray,
    energies: np.ndarray,
    tensors: np.ndarray,
    mu_grid: np.ndarray,
    angle_grid: np.ndarray,
    benchmark: np.ndarray,
    enz_window: np.ndarray,
    drifts: np.ndarray,
    validity_limit: float,
    agreement_limits: np.ndarray,
) -> float:
    energy = np.asarray(energies, float)
    window = np.asarray(enz_window, float)
    if (energy.ndim != 1 or len(energy) < 5 or not np.all(np.isfinite(energy))
            or np.any(np.diff(energy) <= 0)):
        raise ValueError("energies must be a finite increasing one-dimensional grid")
    if (window.shape != (2,) or not np.all(np.isfinite(window))
            or window[0] >= window[1]):
        raise ValueError("enz_window must contain finite increasing bounds")
    lower = np.nextafter(window[0], -np.inf)
    upper = np.nextafter(window[1], np.inf)
    selected = (energy >= lower) & (energy <= upper)
    if np.count_nonzero(selected) < 5:
        raise ValueError("enz_window must retain at least five energy nodes")

    fields = material_fields(mu_nodes, tensors, mu_grid)
    if fields.ndim != 3 or fields.shape[1] != len(energy):
        raise ValueError("tensor energy axis must match energies")
    responses = voigt_surface_responses(fields, angle_grid)
    metric_shape = responses.shape[:-1]
    full = np.full(metric_shape + (7,), np.nan)
    spectrum_valid = np.ones(metric_shape, dtype=bool)
    for index in np.ndindex(metric_shape):
        try:
            full[index] = full_maxwell_metrics(
                energy[selected], responses[index][selected][None, :]
            )[0]
        except ValueError:
            spectrum_valid[index] = False
    energy_column = np.broadcast_to(
        energy[None, selected, None], (len(mu_grid), selected.sum(), 1)
    )
    target_spectra = np.concatenate((energy_column, fields[:, selected, :]), axis=2)
    coordinates = enz_design_coordinates(list(target_spectra))
    target_laws = universal_law_metrics(coordinates, angle_grid)
    drift_values = np.asarray(drifts, float)
    design_mu = np.asarray(mu_grid, float)
    if (drift_values.shape != (2,) or not np.all(np.isfinite(drift_values))
            or np.any(drift_values < 0) or design_mu.ndim != 1
            or len(design_mu) < 2 or not np.all(np.isfinite(design_mu))
            or np.any(np.diff(design_mu) <= 0)):
        raise ValueError("drifts and chemical-potential grid must be finite and valid")
    mu_spacing = np.diff(design_mu)
    if not np.allclose(mu_spacing, mu_spacing[0], rtol=0, atol=1e-12):
        raise ValueError("chemical-potential grid must be uniform")
    mu_ratio = drift_values[0] / mu_spacing[0]
    nearest_mu_radius = int(round(mu_ratio))
    if abs(mu_ratio - nearest_mu_radius) <= 1e-10:
        mu_radius = nearest_mu_radius
    else:
        mu_radius = int(np.floor(np.nextafter(mu_ratio, np.inf)))
    effective_drifts = (mu_radius * mu_spacing[0], drift_values[1])
    guarantees = window_guarantees(
        mu_grid,
        angle_grid,
        full,
        target_laws,
        coordinates[:, 5],
        effective_drifts,
        validity_limit,
        agreement_limits,
        spectrum_valid,
    )
    optimum = select_operating_point(guarantees)
    benchmark_result = _benchmark_common_guarantee(
        benchmark, angle_grid, drifts[1]
    )
    return float(optimum[12] / benchmark_result[5])
SCICODE_GOLD_EOF
