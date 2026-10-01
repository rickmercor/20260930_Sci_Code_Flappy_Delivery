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


def _validate_background(sigma, d, ell, s):
    grid = np.asarray(sigma, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("sigma must be a one-dimensional array with at least two points")
    if not np.all(np.isfinite(grid)):
        raise ValueError("sigma must be finite")
    if np.any(grid < 0.0) or np.any(grid > 1.0):
        raise ValueError("sigma must lie in the closed interval from zero to one")
    if not isinstance(d, (int, np.integer)) or isinstance(d, bool) or int(d) < 4:
        raise ValueError("d must be an integer of at least four")
    if not isinstance(ell, (int, np.integer)) or isinstance(ell, bool) or int(ell) < 0:
        raise ValueError("ell must be a non-negative integer")
    spin = float(s)
    if not np.isfinite(spin) or spin < 0.0:
        raise ValueError("s must be finite and non-negative")
    return grid, int(d), int(ell), spin


def _reduced_potential_parts(d, ell, s):
    """Constant and sigma^(d - 3) coefficients of the reduced potential q."""
    constant = float(ell) * (ell + d - 3) + (d - 2) * (d - 4) / 4.0
    varying = (1.0 - s ** 2) * (d - 2) ** 2 / 4.0
    return constant, varying


def compactified_coefficients(
    sigma: np.ndarray,
    d: int,
    ell: int,
    s: float,
) -> dict:
    """Reference implementation."""
    grid, d, ell, spin = _validate_background(sigma, d, ell, s)
    m = d - 3
    weight = grid ** 2 - grid ** (m + 2)
    weight_derivative = 2.0 * grid - (m + 2) * grid ** (m + 1)
    constant, varying = _reduced_potential_parts(d, ell, spin)
    reduced = constant + varying * grid ** m
    return {
        "weight": weight,
        "weight_derivative": weight_derivative,
        "reduced_potential": reduced,
        "potential_at_horizon": float(constant + varying),
        "potential_at_infinity": float(constant),
    }

import numpy as np


def _validate_resolution(resolution):
    if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)):
        raise ValueError("resolution must be an integer")
    if int(resolution) < 4:
        raise ValueError("resolution must be at least four")
    return int(resolution)


def chebyshev_lobatto_operators(resolution: int) -> dict:
    """Reference implementation."""
    n = _validate_resolution(resolution)
    index = np.arange(n + 1)
    chebyshev = np.cos(np.pi * index / n)
    weights = np.where((index == 0) | (index == n), 2.0, 1.0) * (-1.0) ** index
    separation = chebyshev[:, None] - chebyshev[None, :]
    matrix = np.outer(weights, 1.0 / weights) / (separation + np.eye(n + 1))
    matrix = matrix - np.diag(matrix.sum(axis=1))
    nodes = (1.0 - chebyshev) / 2.0
    derivative = -2.0 * matrix
    residual = float(np.max(np.abs(derivative @ np.ones(n + 1))))
    return {"nodes": nodes, "derivative": derivative, "constant_residual": residual}

import numpy as np


def transmission_pencil(
    resolution: int,
    d: int,
    ell: int,
    s: float,
) -> dict:
    """Reference implementation."""
    grid = chebyshev_lobatto_operators(resolution)  # noqa: F821
    nodes = grid["nodes"]
    derivative = grid["derivative"]
    background = compactified_coefficients(nodes, d, ell, s)  # noqa: F821
    weight = background["weight"]
    weight_derivative = background["weight_derivative"]
    reduced = background["reduced_potential"]
    second = derivative @ derivative
    operator_a = (weight[:, None] * second
                  + weight_derivative[:, None] * derivative
                  - np.diag(reduced))
    operator_b = 2.0 * derivative
    smallest = float(np.linalg.svd(operator_b, compute_uv=False)[-1])
    return {
        "operator_a": operator_a,
        "operator_b": operator_b,
        "nodes": nodes,
        "derivative": derivative,
        "operator_b_smallest_singular_value": smallest,
    }

import numpy as np


def _validate_scan(operator_a, operator_b, lower, upper, samples):
    a = np.asarray(operator_a, dtype=complex)
    b = np.asarray(operator_b, dtype=complex)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 5:
        raise ValueError("operator_a must be a square array of side at least five")
    if b.shape != a.shape:
        raise ValueError("operator_b must have the same shape as operator_a")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("the pencil matrices must be finite")
    low, high = float(lower), float(upper)
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or high <= low:
        raise ValueError("the sweep interval must be finite, strictly positive and increasing")
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)):
        raise ValueError("samples must be an integer")
    if int(samples) < 5:
        raise ValueError("samples must be at least five")
    return a, b, low, high, int(samples)


def imaginary_axis_scan(
    operator_a: np.ndarray,
    operator_b: np.ndarray,
    lower: float,
    upper: float,
    samples: int,
) -> dict:
    """Reference implementation."""
    a, b, low, high, count = _validate_scan(operator_a, operator_b, lower, upper, samples)
    frequencies = np.linspace(low, high, count)
    values = np.empty(count)
    for index, height in enumerate(frequencies):
        shifted = a - (1j * (1j * height)) * b
        values[index] = np.linalg.svd(shifted, compute_uv=False)[-1]
    interior = [
        i for i in range(1, count - 1)
        if values[i] < values[i - 1] and values[i] < values[i + 1]
    ]
    minima = frequencies[interior] if interior else np.empty(0)
    if interior:
        location = float(frequencies[min(interior, key=lambda i: values[i])])
    else:
        location = float("nan")
    return {
        "frequencies": frequencies,
        "smallest_singular_values": values,
        "minima": minima,
        "location": location,
    }

import warnings

import numpy as np
import scipy.linalg as sla


def _validate_pencil(operator_a, operator_b, frequency_guess, tolerance, max_iterations):
    a = np.asarray(operator_a, dtype=complex)
    b = np.asarray(operator_b, dtype=complex)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 5:
        raise ValueError("operator_a must be a square array of side at least five")
    if b.shape != a.shape:
        raise ValueError("operator_b must have the same shape as operator_a")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("the pencil matrices must be finite")
    guess = complex(frequency_guess)
    if not np.isfinite(guess.real) or not np.isfinite(guess.imag):
        raise ValueError("frequency_guess must be finite")
    tol = float(tolerance)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance must be finite and strictly positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)):
        raise ValueError("max_iterations must be an integer")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be at least one")
    return a, b, guess, tol, int(max_iterations)


def _fix_phase(vector):
    """Unit Euclidean norm with the entry of largest modulus made real and positive."""
    scaled = vector / np.linalg.norm(vector)
    lead = scaled[int(np.argmax(np.abs(scaled)))]
    return scaled * (np.conj(lead) / abs(lead))


def regular_mode(
    operator_a: np.ndarray,
    operator_b: np.ndarray,
    frequency_guess: complex,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Reference implementation."""
    a, b, guess, tol, budget = _validate_pencil(
        operator_a, operator_b, frequency_guess, tolerance, max_iterations
    )
    size = a.shape[0]
    start = (1.0 / (1.0 + np.arange(size))).astype(complex)
    right = start / np.linalg.norm(start)
    left = right.copy()
    eigenvalue = 1j * guess
    correction = float("inf")
    taken = 0
    for _ in range(budget):
        shifted = a - eigenvalue * b
        try:
            with warnings.catch_warnings():
                # once the eigenvalue is converged the shifted pencil is singular to working
                # precision, which is the stopping signal rather than a fault
                warnings.simplefilter("ignore", sla.LinAlgWarning)
                factors = sla.lu_factor(shifted)
                trial_right = sla.lu_solve(factors, right)
                trial_left = sla.lu_solve(factors, left, trans=2)
        except Exception:
            break
        if not (np.all(np.isfinite(trial_right)) and np.all(np.isfinite(trial_left))):
            break
        trial_right = trial_right / np.linalg.norm(trial_right)
        trial_left = trial_left / np.linalg.norm(trial_left)
        pairing = np.conj(trial_left) @ (b @ trial_right)
        if pairing == 0.0 or not np.isfinite(pairing):
            break
        step = (np.conj(trial_left) @ (shifted @ trial_right)) / pairing
        if not np.isfinite(step):
            break
        right, left = trial_right, trial_left
        eigenvalue = eigenvalue + step
        correction = float(abs(step))
        taken += 1
        if correction < tol * max(abs(eigenvalue), 1.0):
            break
    frequency = -1j * eigenvalue
    return {
        "frequency_real": float(frequency.real),
        "frequency_imag": float(frequency.imag),
        "right_vector": _fix_phase(right),
        "left_vector": _fix_phase(left),
        "correction": correction,
        "iterations": taken,
    }

import numpy as np


def _chebyshev_coefficients(values):
    """Gauss-Lobatto Chebyshev transform of nodal values on a grid of N + 1 points."""
    samples = np.asarray(values)
    count = samples.size - 1
    index = np.arange(count + 1)
    scale = np.where((index == 0) | (index == count), 2.0, 1.0)
    basis = np.cos(np.pi * np.outer(index, index) / count)
    return (2.0 / count) * (basis @ (samples / scale)) / scale


def mode_certification(
    resolution: int,
    refinement: int,
    d: int,
    ell: int,
    s: float,
    frequency_guess: complex,
    tolerance: float,
) -> dict:
    """Reference implementation."""
    if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)):
        raise ValueError("resolution must be an integer")
    if int(resolution) < 8:
        raise ValueError("resolution must be at least eight")
    if isinstance(refinement, bool) or not isinstance(refinement, (int, np.integer)):
        raise ValueError("refinement must be an integer")
    if int(refinement) < 1:
        raise ValueError("refinement must be at least one")
    tol = float(tolerance)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance must be finite and strictly positive")

    coarse_size = int(resolution)
    fine_size = coarse_size + int(refinement)
    coarse = transmission_pencil(coarse_size, d, ell, s)  # noqa: F821
    fine = transmission_pencil(fine_size, d, ell, s)  # noqa: F821
    coarse_mode = regular_mode(  # noqa: F821
        coarse["operator_a"], coarse["operator_b"], frequency_guess, 1e-13, 200
    )
    fine_mode = regular_mode(  # noqa: F821
        fine["operator_a"], fine["operator_b"], frequency_guess, 1e-13, 200
    )

    coarse_frequency = complex(coarse_mode["frequency_real"], coarse_mode["frequency_imag"])
    fine_frequency = complex(fine_mode["frequency_real"], fine_mode["frequency_imag"])
    drift = float(abs(fine_frequency - coarse_frequency))

    vector = coarse_mode["right_vector"]
    coefficients = np.abs(_chebyshev_coefficients(vector))
    largest = float(np.max(coefficients))
    top = coefficients[(3 * coarse_size) // 4:]
    tail = float(np.max(top) / largest) if largest > 0.0 else float("inf")

    shifted = coarse["operator_a"] - (1j * coarse_frequency) * coarse["operator_b"]
    residual = float(np.max(np.abs(shifted @ vector)))

    return {
        "frequency_real": float(fine_frequency.real),
        "frequency_imag": float(fine_frequency.imag),
        "frequency_drift": drift,
        "spectral_tail": tail,
        "residual": residual,
        "certified": int(drift < tol and tail < tol),
    }

import numpy as np


def _cardinal_values(nodes, points):
    """Barycentric evaluation of the Chebyshev-Lobatto cardinal polynomials at arbitrary points."""
    count = nodes.size
    index = np.arange(count)
    bary = np.where((index == 0) | (index == count - 1), 0.5, 1.0) * (-1.0) ** index
    gap = points[:, None] - nodes[None, :]
    hit = np.isclose(gap, 0.0, rtol=0.0, atol=0.0)
    safe = np.where(hit, 1.0, gap)
    terms = bary[None, :] / safe
    values = terms / terms.sum(axis=1, keepdims=True)
    if np.any(hit):
        rows = np.any(hit, axis=1)
        values[rows] = hit[rows].astype(float)
    return values


def energy_gram_matrix(
    resolution: int,
    d: int,
    ell: int,
    s: float,
    quadrature_points: int,
) -> dict:
    """Reference implementation."""
    grid = chebyshev_lobatto_operators(resolution)  # noqa: F821
    nodes = grid["nodes"]
    derivative = grid["derivative"]
    compactified_coefficients(nodes, d, ell, s)  # noqa: F821  validates d, ell and s
    if isinstance(quadrature_points, bool) or not isinstance(quadrature_points, (int, np.integer)):
        raise ValueError("quadrature_points must be an integer")
    required = int(resolution) + (int(d) + 1) // 2
    if int(quadrature_points) < required:
        raise ValueError("quadrature_points must be at least resolution plus (d + 1) // 2")

    legendre_nodes, legendre_weights = np.polynomial.legendre.leggauss(int(quadrature_points))
    points = (1.0 - legendre_nodes) / 2.0
    weights = legendre_weights / 2.0
    cardinal = _cardinal_values(nodes, points)
    cardinal_derivative = cardinal @ derivative

    background = compactified_coefficients(points, d, ell, s)  # noqa: F821
    weight = background["weight"]
    reduced = background["reduced_potential"]

    energy = 0.5 * (
        cardinal_derivative.T @ ((weights * weight)[:, None] * cardinal_derivative)
        + cardinal.T @ ((weights * reduced)[:, None] * cardinal)
    )
    energy = 0.5 * (energy + energy.T)
    lebesgue = 0.5 * (cardinal.T @ (weights[:, None] * cardinal))
    lebesgue = 0.5 * (lebesgue + lebesgue.T)
    smallest = float(np.linalg.eigvalsh(energy)[0])
    return {
        "energy_gram": energy,
        "lebesgue_gram": lebesgue,
        "smallest_energy_eigenvalue": smallest,
    }

import numpy as np
import scipy.linalg as sla


def _validate_norm_inputs(gram, operator_b, right_vector, left_vector):
    G = np.asarray(gram, dtype=float)
    if G.ndim != 2 or G.shape[0] != G.shape[1] or G.shape[0] < 2:
        raise ValueError("gram must be a square array of side at least two")
    if not np.all(np.isfinite(G)):
        raise ValueError("gram must be finite")
    if not np.allclose(G, G.T, rtol=0.0, atol=1e-10 * max(1.0, float(np.max(np.abs(G))))):
        raise ValueError("gram must be symmetric")
    B = np.asarray(operator_b, dtype=complex)
    if B.shape != G.shape or not np.all(np.isfinite(B)):
        raise ValueError("operator_b must be a finite array of the same shape as gram")
    x = np.asarray(right_vector, dtype=complex)
    z = np.asarray(left_vector, dtype=complex)
    for name, v in (("right_vector", x), ("left_vector", z)):
        if v.ndim != 1 or v.size != G.shape[0]:
            raise ValueError(name + " must be a one-dimensional array of matching length")
        if not np.all(np.isfinite(v)) or np.linalg.norm(v) == 0.0:
            raise ValueError(name + " must be finite and non-zero")
    return G, B, x, z


def condition_number_in_norm(
    gram: np.ndarray,
    operator_b: np.ndarray,
    right_vector: np.ndarray,
    left_vector: np.ndarray,
) -> dict:
    """Reference implementation."""
    G, B, x, z = _validate_norm_inputs(gram, operator_b, right_vector, left_vector)
    try:
        factor = sla.cho_factor(G, lower=False)
    except sla.LinAlgError:
        raise ValueError("gram must be positive definite")
    adjoint = sla.cho_solve(factor, z)
    adjoint_norm_squared = float(np.real(np.conj(z) @ adjoint))
    right_norm_squared = float(np.real(np.conj(x) @ (G @ x)))
    if adjoint_norm_squared <= 0.0 or right_norm_squared <= 0.0:
        raise ValueError("gram must be positive definite")
    pairing = complex(np.conj(z) @ (B @ x))
    pairing_modulus = float(abs(pairing))
    if pairing_modulus == 0.0:
        raise ValueError("the pairing of the left vector with the first-order operator vanishes")
    adjoint_norm = float(np.sqrt(adjoint_norm_squared))
    right_norm = float(np.sqrt(right_norm_squared))
    return {
        "adjoint_vector": adjoint,
        "adjoint_norm": adjoint_norm,
        "right_norm": right_norm,
        "pairing_modulus": pairing_modulus,
        "condition_number": float(adjoint_norm * right_norm / pairing_modulus),
    }

import math


def _checked(value, name):
    """Reject anything that is not a finite, strictly positive real number."""
    if isinstance(value, bool) or isinstance(value, complex):
        raise ValueError(name + " must be a real number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(name + " must be a real number")
    if not math.isfinite(number):
        raise ValueError(name + " must be finite")
    if number <= 0.0:
        raise ValueError(name + " must be strictly positive")
    return number


def _significant(value, digits):
    """Round to the requested number of significant figures."""
    if value == 0.0:
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    return float(round(value, digits - 1 - exponent))


def reported_values(
    condition_number_energy: float,
    condition_number_lebesgue: float,
    frequency_imag: float,
    comparison_condition_number_energy: float,
    comparison_condition_number_lebesgue: float,
    comparison_frequency_imag: float,
) -> dict:
    """Reference implementation."""
    energy = _checked(condition_number_energy, "condition_number_energy")
    lebesgue = _checked(condition_number_lebesgue, "condition_number_lebesgue")
    frequency = _checked(frequency_imag, "frequency_imag")
    other_energy = _checked(
        comparison_condition_number_energy, "comparison_condition_number_energy"
    )
    other_lebesgue = _checked(
        comparison_condition_number_lebesgue, "comparison_condition_number_lebesgue"
    )
    other_frequency = _checked(comparison_frequency_imag, "comparison_frequency_imag")
    return {
        "graded_answer": float(round(energy, 6)),
        "reported_lebesgue": _significant(lebesgue, 4),
        "reported_frequency": _significant(frequency, 8),
        "reported_comparison_energy": _significant(other_energy, 4),
        "reported_comparison_lebesgue": _significant(other_lebesgue, 4),
        "reported_comparison_frequency": _significant(other_frequency, 8),
    }

import numpy as np


def _quadrature_count(resolution, dimensions):
    return int(resolution) + (max(int(v) for v in dimensions) + 1) // 2 + 4


def _condition_pair(resolution, d, ell, s, guess, quadrature):
    """Frequency and the two condition numbers of the mode nearest the guess."""
    pencil = transmission_pencil(resolution, d, ell, s)  # noqa: F821
    mode = regular_mode(  # noqa: F821
        pencil["operator_a"], pencil["operator_b"], guess, 1e-13, 200
    )
    gram = energy_gram_matrix(resolution, d, ell, s, quadrature)  # noqa: F821
    energy = condition_number_in_norm(  # noqa: F821
        gram["energy_gram"], pencil["operator_b"], mode["right_vector"], mode["left_vector"]
    )
    lebesgue = condition_number_in_norm(  # noqa: F821
        gram["lebesgue_gram"], pencil["operator_b"], mode["right_vector"], mode["left_vector"]
    )
    return {
        "frequency": complex(mode["frequency_real"], mode["frequency_imag"]),
        "energy": float(energy["condition_number"]),
        "lebesgue": float(lebesgue["condition_number"]),
        "smallest": float(gram["smallest_energy_eigenvalue"]),
    }


def transmission_mode_conditioning_report(
    resolution: int,
    refinement: int,
    d: int,
    ell: int,
    s: float,
    comparison_dimension: int,
    scan_resolution: int,
    scan_lower: float,
    scan_upper: float,
    scan_samples: int,
) -> dict:
    """Reference implementation."""
    if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)):
        raise ValueError("resolution must be an integer")
    if int(resolution) < 8:
        raise ValueError("resolution must be at least eight")
    if isinstance(refinement, bool) or not isinstance(refinement, (int, np.integer)):
        raise ValueError("refinement must be an integer")
    if int(refinement) < 1:
        raise ValueError("refinement must be at least one")
    if isinstance(comparison_dimension, bool) or not isinstance(comparison_dimension, (int, np.integer)):
        raise ValueError("comparison_dimension must be an integer")
    if int(comparison_dimension) < 4:
        raise ValueError("comparison_dimension must be at least four")

    scan_pencil = transmission_pencil(scan_resolution, d, ell, s)  # noqa: F821
    sweep = imaginary_axis_scan(  # noqa: F821
        scan_pencil["operator_a"], scan_pencil["operator_b"],
        scan_lower, scan_upper, scan_samples,
    )
    location = float(sweep["location"])
    if not np.isfinite(location):
        raise ValueError("the sweep found no mode on the requested interval")
    guess = complex(0.0, location)

    quadrature = _quadrature_count(resolution, (d, comparison_dimension))
    graded = _condition_pair(int(resolution), d, ell, s, guess, quadrature)

    fine_resolution = int(resolution) + int(refinement)
    fine_quadrature = _quadrature_count(fine_resolution, (d, comparison_dimension))
    refined = _condition_pair(fine_resolution, d, ell, s, guess, fine_quadrature)

    comparison_scan_pencil = transmission_pencil(  # noqa: F821
        scan_resolution, comparison_dimension, ell, s
    )
    comparison_sweep = imaginary_axis_scan(  # noqa: F821
        comparison_scan_pencil["operator_a"], comparison_scan_pencil["operator_b"],
        scan_lower, scan_upper, scan_samples,
    )
    comparison_location = float(comparison_sweep["location"])
    if not np.isfinite(comparison_location):
        raise ValueError("the sweep found no mode for the comparison dimension")
    comparison = _condition_pair(
        int(resolution), comparison_dimension, ell, s, complex(0.0, comparison_location), quadrature
    )

    certification = mode_certification(  # noqa: F821
        int(resolution), int(refinement), d, ell, s, guess, 1e-10
    )

    rounded = reported_values(  # noqa: F821
        graded["energy"], graded["lebesgue"], float(graded["frequency"].imag),
        comparison["energy"], comparison["lebesgue"], float(comparison["frequency"].imag),
    )

    return {
        "graded_answer": rounded["graded_answer"],
        "reported_lebesgue": rounded["reported_lebesgue"],
        "reported_frequency": rounded["reported_frequency"],
        "reported_comparison_energy": rounded["reported_comparison_energy"],
        "reported_comparison_lebesgue": rounded["reported_comparison_lebesgue"],
        "reported_comparison_frequency": rounded["reported_comparison_frequency"],
        "frequency_real": float(graded["frequency"].real),
        "frequency_imag": float(graded["frequency"].imag),
        "condition_number_energy": graded["energy"],
        "condition_number_lebesgue": graded["lebesgue"],
        "condition_number_comparison": comparison["energy"],
        "condition_number_lebesgue_comparison": comparison["lebesgue"],
        "frequency_imag_comparison": float(comparison["frequency"].imag),
        "condition_number_refined": refined["energy"],
        "refinement_drift": float(abs(refined["energy"] - graded["energy"])),
        "smallest_energy_eigenvalue": graded["smallest"],
        "scan_location": location,
        "certified": int(certification["certified"]),
    }
SCICODE_GOLD_EOF
