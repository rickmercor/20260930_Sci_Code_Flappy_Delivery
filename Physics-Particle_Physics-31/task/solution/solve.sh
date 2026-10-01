#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

from decimal import Decimal, localcontext
import numpy as np


def compute_log_derivative_coefficients(b: "np.ndarray") -> "np.ndarray":
    """Compute c_k in Q(s)=A'(s)/A(s) from the EFT coefficients b_j."""
    raw = np.asarray(b)
    if np.iscomplexobj(raw) and np.any(np.imag(raw) != 0.0):
        raise ValueError("b must be real-valued")
    try:
        coeff = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("b must contain real numeric values") from exc
    if coeff.ndim != 1 or coeff.size < 2 or not np.all(np.isfinite(coeff)):
        raise ValueError("b must be a finite one-dimensional array of length at least two")
    if coeff[0] == 0.0:
        raise ValueError("b[0] must be nonzero")

    # Carry the logarithmic-derivative recurrence at high precision so that
    # small roundoff errors do not move the final benchmark's tenth decimal.
    with localcontext() as ctx:
        ctx.prec = 60
        b_dec = [Decimal(str(float(value))) for value in coeff]
        c_dec = []

        for n in range(coeff.size - 1):
            previous = sum(
                (c_dec[k] * b_dec[n - k] for k in range(n)),
                Decimal(0),
            )
            current = (
                Decimal(n + 1) * b_dec[n + 1] - previous
            ) / b_dec[0]
            c_dec.append(current)

    return np.asarray([float(value) for value in c_dec], dtype=float)

import numpy as np


def choose_source_configuration(c: "np.ndarray") -> "tuple[int, int]":
    """Return the smallest positive odd r and largest supported probe size."""
    raw = np.asarray(c)
    if np.iscomplexobj(raw) and np.any(np.imag(raw) != 0.0):
        raise ValueError("c must be real-valued")
    try:
        coeff = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("c must contain real numeric values") from exc
    if coeff.ndim != 1 or coeff.size < 2 or not np.all(np.isfinite(coeff)):
        raise ValueError("c must be a finite one-dimensional array of length at least two")

    r = 1
    max_index = coeff.size - 1
    probe_size = 1 + (max_index - r) // 2
    if probe_size < 1:
        raise ValueError("c is too short for the smallest positive odd source parameter")
    return int(r), int(probe_size)

import numpy as np


def build_hankel_probe(c: "np.ndarray", r: int, size: int) -> "np.ndarray":
    """Build C_r^(size) with entries c[r+i+j]."""
    raw = np.asarray(c)
    if np.iscomplexobj(raw) and np.any(np.imag(raw) != 0.0):
        raise ValueError("c must be real-valued")
    try:
        coeff = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("c must contain real numeric values") from exc
    if coeff.ndim != 1 or coeff.size == 0 or not np.all(np.isfinite(coeff)):
        raise ValueError("c must be a nonempty finite one-dimensional array")
    if not isinstance(r, (int, np.integer)) or int(r) < 0:
        raise ValueError("r must be a nonnegative integer")
    if not isinstance(size, (int, np.integer)) or int(size) < 1:
        raise ValueError("size must be a positive integer")
    r = int(r)
    size = int(size)
    required = r + 2 * (size - 1)
    if required >= coeff.size:
        raise ValueError("c is too short for the requested Hankel block")

    idx = r + np.add.outer(np.arange(size), np.arange(size))
    return coeff[idx]

import numpy as np


def infer_finite_spectrum_size(hankel_probe: "np.ndarray", rtol: float) -> int:
    """Return the numerical Hankel rank using sigma_i > rtol*sigma_max."""
    raw = np.asarray(hankel_probe)
    if np.iscomplexobj(raw) and np.any(np.imag(raw) != 0.0):
        raise ValueError("hankel_probe must be real-valued")
    try:
        matrix = np.asarray(raw, dtype=float)
        tol = float(rtol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("hankel_probe and rtol must be real numeric values") from exc
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] == 0
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("hankel_probe must be a finite nonempty square matrix")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("rtol must be strictly positive and finite")

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values[0] == 0.0:
        return 0
    return int(np.count_nonzero(singular_values > tol * singular_values[0]))

import numpy as np


def build_shifted_hankel_pencil(
    c: "np.ndarray", r: int, d: int
) -> "tuple[np.ndarray, np.ndarray]":
    """Return the ordered pair (C_r^(d), C_{r+1}^(d))."""
    if not isinstance(d, (int, np.integer)) or int(d) < 1:
        raise ValueError("d must be a positive integer")
    d = int(d)
    current = build_hankel_probe(c, r, d)
    shifted = build_hankel_probe(c, r + 1, d)
    return current, shifted

import numpy as np
from scipy.linalg import eigvals


def recover_spectral_locations(
    hankel_current: "np.ndarray",
    hankel_shifted: "np.ndarray",
    imag_tol: float,
) -> "np.ndarray":
    """Solve the shifted Hankel pencil and return [lambda, 1/lambda] pairs."""
    try:
        current = np.asarray(hankel_current, dtype=float)
        shifted = np.asarray(hankel_shifted, dtype=float)
        tol = float(imag_tol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("matrices and imag_tol must be real numeric values") from exc
    if (
        current.ndim != 2
        or current.shape[0] != current.shape[1]
        or current.shape[0] == 0
        or shifted.shape != current.shape
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(shifted))
    ):
        raise ValueError("the Hankel matrices must be finite nonempty square arrays of equal shape")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("imag_tol must be nonnegative and finite")

    values = eigvals(shifted, current)
    if np.any(~np.isfinite(values)):
        raise ValueError("the generalized eigenproblem returned a nonfinite value")
    if np.any(np.abs(values.imag) > tol):
        raise ValueError("a generalized eigenvalue is not real within imag_tol")

    lambdas = values.real
    scale = max(1.0, float(np.max(np.abs(lambdas))))
    if np.any(np.abs(lambdas) <= np.finfo(float).eps * scale):
        raise ValueError("a generalized eigenvalue is numerically zero")
    locations = 1.0 / lambdas
    order = np.argsort(locations)
    return np.column_stack((lambdas[order], locations[order]))

import itertools
import numpy as np
from scipy.optimize import root_scalar


def recover_missing_eft_coefficient(
    b_incomplete: "np.ndarray", rank_rtol: float, imag_tol: float
) -> float:
    """Recover the unique missing EFT coefficient consistent with a real finite spectrum."""
    raw = np.asarray(b_incomplete)
    if np.iscomplexobj(raw) and np.any(np.imag(raw[np.isfinite(raw)]) != 0.0):
        raise ValueError("b_incomplete must be real-valued")
    try:
        coeff = np.asarray(raw, dtype=float)
        rank_tol = float(rank_rtol)
        eig_tol = float(imag_tol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("inputs must be real numeric values") from exc

    if coeff.ndim != 1 or coeff.size < 4:
        raise ValueError("b_incomplete must be a one-dimensional array of length at least four")
    missing = np.flatnonzero(np.isnan(coeff))
    if missing.size != 1:
        raise ValueError("b_incomplete must contain exactly one NaN marking the omitted coefficient")
    if np.any(np.isinf(coeff)):
        raise ValueError("b_incomplete must not contain infinite values")
    missing_index = int(missing[0])
    if missing_index == 0 or not np.isfinite(coeff[0]) or coeff[0] == 0.0:
        raise ValueError("b[0] must be known, finite, and nonzero")
    if not np.isfinite(rank_tol) or rank_tol <= 0.0:
        raise ValueError("rank_rtol must be strictly positive and finite")
    if not np.isfinite(eig_tol) or eig_tol < 0.0:
        raise ValueError("imag_tol must be nonnegative and finite")

    # Polynomial helpers use ascending powers of the unknown coefficient x.
    def poly_add(a, b):
        n = max(len(a), len(b))
        out = np.zeros(n, dtype=float)
        out[: len(a)] += a
        out[: len(b)] += b
        while out.size > 1 and abs(out[-1]) <= 1.0e-14 * max(1.0, float(np.max(np.abs(out)))):
            out = out[:-1]
        return out

    def poly_sub(a, b):
        return poly_add(a, -np.asarray(b, dtype=float))

    def poly_mul(a, b):
        out = np.polynomial.polynomial.polymul(a, b)
        while out.size > 1 and abs(out[-1]) <= 1.0e-14 * max(1.0, float(np.max(np.abs(out)))):
            out = out[:-1]
        return out

    b_poly = []
    for j, value in enumerate(coeff):
        if j == missing_index:
            b_poly.append(np.array([0.0, 1.0], dtype=float))
        else:
            b_poly.append(np.array([float(value)], dtype=float))

    # Construct c_k(x) directly from A Q = A'. Division is only by known b_0.
    c_poly = []
    for n in range(coeff.size - 1):
        rhs = (n + 1) * b_poly[n + 1]
        for k in range(n):
            rhs = poly_sub(rhs, poly_mul(c_poly[k], b_poly[n - k]))
        c_poly.append(rhs / coeff[0])

    # Configuration depends on the available transformed sequence length, not on x.
    dummy_c = np.zeros(coeff.size - 1, dtype=float)
    r, probe_size = choose_source_configuration(dummy_c)
    required = r + 2 * (probe_size - 1)
    if required >= len(c_poly):
        raise ValueError("insufficient transformed data for the maximal probe")

    # Determinant polynomial of C_r^(probe_size)(x), via the Leibniz formula.
    matrix = [
        [c_poly[r + i + j] for j in range(probe_size)]
        for i in range(probe_size)
    ]
    determinant = np.array([0.0], dtype=float)
    for perm in itertools.permutations(range(probe_size)):
        inversions = sum(
            perm[i] > perm[j]
            for i in range(probe_size)
            for j in range(i + 1, probe_size)
        )
        term = np.array([1.0], dtype=float)
        for i, j in enumerate(perm):
            term = poly_mul(term, matrix[i][j])
        determinant = poly_add(determinant, -term if inversions % 2 else term)

    scale = max(1.0, float(np.max(np.abs(determinant))))
    while determinant.size > 1 and abs(determinant[-1]) <= 1.0e-12 * scale:
        determinant = determinant[:-1]
    if determinant.size <= 1 or np.all(np.abs(determinant) <= 1.0e-14 * scale):
        raise ValueError("the finite-spectrum determinant condition does not isolate the missing coefficient")

    roots = np.polynomial.polynomial.polyroots(determinant)
    root_imag_tol = max(1.0e-8, 100.0 * eig_tol)
    real_seeds = sorted(
        float(z.real)
        for z in roots
        if np.isfinite(z.real) and np.isfinite(z.imag) and abs(z.imag) <= root_imag_tol
    )
    if not real_seeds:
        raise ValueError("the determinant condition has no real candidate completion")

    def determinant_at(x):
        completed = coeff.copy()
        completed[missing_index] = float(x)
        c = compute_log_derivative_coefficients(completed)
        rr, size = choose_source_configuration(c)
        probe = build_hankel_probe(c, rr, size)
        return float(np.linalg.det(probe))

    refined = []
    for seed in real_seeds:
        delta = 1.0e-6 * max(1.0, abs(seed))
        try:
            sol = root_scalar(
                determinant_at,
                x0=seed,
                x1=seed + delta,
                method="secant",
                xtol=1.0e-14,
                rtol=1.0e-14,
                maxiter=100,
            )
            candidate = float(sol.root) if sol.converged else seed
        except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
            candidate = seed
        if not any(abs(candidate - old) <= 1.0e-8 * max(1.0, abs(candidate), abs(old)) for old in refined):
            refined.append(candidate)

    admissible = []
    for candidate in refined:
        completed = coeff.copy()
        completed[missing_index] = candidate
        try:
            c = compute_log_derivative_coefficients(completed)
            rr, size = choose_source_configuration(c)
            probe = build_hankel_probe(c, rr, size)
            d = infer_finite_spectrum_size(probe, rank_tol)
            if d < 1 or d >= size:
                continue
            current, shifted = build_shifted_hankel_pencil(c, rr, d)
            spectrum = recover_spectral_locations(current, shifted, eig_tol)
        except (ValueError, np.linalg.LinAlgError):
            continue

        locations = spectrum[:, 1]
        if locations.size > 1:
            gaps = np.diff(locations)
            loc_scale = max(1.0, float(np.max(np.abs(locations))))
            if np.any(np.abs(gaps) <= 1.0e-8 * loc_scale):
                continue
        admissible.append(candidate)

    # Merge any numerically duplicated admissible roots.
    unique = []
    for candidate in sorted(admissible):
        if not unique or abs(candidate - unique[-1]) > 1.0e-8 * max(1.0, abs(candidate), abs(unique[-1])):
            unique.append(candidate)
    if len(unique) != 1:
        raise ValueError("the stated finite-spectrum conditions do not select a unique real completion")
    return float(unique[0])

import numpy as np


def classify_spectral_locations(
    hankel_current: "np.ndarray", r: int, spectral_data: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray]":
    """Return classified locations and ordered diagonal classification weights."""
    try:
        current = np.asarray(hankel_current, dtype=float)
        spectrum = np.asarray(spectral_data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("hankel_current and spectral_data must be real numeric arrays") from exc
    if not isinstance(r, (int, np.integer)) or int(r) <= 0 or int(r) % 2 != 1:
        raise ValueError("r must be a positive odd integer")
    if (
        current.ndim != 2
        or current.shape[0] != current.shape[1]
        or current.shape[0] == 0
        or spectrum.shape != (current.shape[0], 2)
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(spectrum))
    ):
        raise ValueError("inputs have incompatible shapes or contain nonfinite values")

    lambdas = spectrum[:, 0]
    locations = spectrum[:, 1]
    d = current.shape[0]
    vandermonde = np.vstack([lambdas ** i for i in range(d)])
    try:
        left = np.linalg.solve(vandermonde, current)
        d_r = np.linalg.solve(vandermonde, left.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("the Vandermonde matrix is singular") from exc
    d_r = 0.5 * (d_r + d_r.T)
    diagonal = np.diag(d_r).copy()
    scale = max(1.0, float(np.max(np.abs(diagonal))))
    if np.any(np.abs(diagonal) <= 1.0e-12 * scale):
        raise ValueError("a reconstructed diagonal weight is numerically zero")

    signs = np.where(diagonal > 0.0, 1.0, -1.0)
    classified = np.column_stack((locations, signs))
    return classified, diagonal

import numpy as np


def reconstruct_rational_amplitude(
    b: "np.ndarray", classified_locations: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray, float]":
    """Return numerator, denominator, and EFT re-expansion residual."""
    try:
        coeff = np.asarray(b, dtype=float)
        classified = np.asarray(classified_locations, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("inputs must be real numeric arrays") from exc
    if (
        coeff.ndim != 1
        or coeff.size < 1
        or not np.all(np.isfinite(coeff))
        or coeff[0] == 0.0
    ):
        raise ValueError("b must be a finite one-dimensional array with nonzero b[0]")
    if (
        classified.ndim != 2
        or classified.shape[1] != 2
        or classified.shape[0] < 1
        or not np.all(np.isfinite(classified))
        or np.any(classified[:, 0] == 0.0)
        or np.any(~np.isin(classified[:, 1], (-1.0, 1.0)))
    ):
        raise ValueError("classified_locations must contain nonzero locations and +/-1 signs")

    numerator = np.array([coeff[0]], dtype=float)
    denominator = np.array([1.0], dtype=float)
    for location, sign in classified:
        factor = np.array([1.0, -1.0 / location], dtype=float)
        if sign < 0.0:
            numerator = np.polynomial.polynomial.polymul(numerator, factor)
        else:
            denominator = np.polynomial.polynomial.polymul(denominator, factor)

    reconstructed = np.zeros_like(coeff)
    for n in range(coeff.size):
        rhs = numerator[n] if n < numerator.size else 0.0
        for k in range(1, min(n, denominator.size - 1) + 1):
            rhs -= denominator[k] * reconstructed[n - k]
        reconstructed[n] = rhs / denominator[0]

    residual = float(np.max(np.abs(reconstructed - coeff)))
    return numerator, denominator, residual

import numpy as np


def evaluate_inverse_eft_reconstruction(
    b: "np.ndarray",
    rank_rtol: float,
    imag_tol: float,
    residual_tol: float,
    s_eval: float,
) -> float:
    """Recover any single omitted coefficient, reconstruct A, and return A_rec(s_eval)."""
    raw = np.asarray(b)
    if np.iscomplexobj(raw) and np.any(np.imag(raw[np.isfinite(raw)]) != 0.0):
        raise ValueError("b must be real-valued")
    try:
        coeff = np.asarray(raw, dtype=float).copy()
        residual_limit = float(residual_tol)
        point = float(s_eval)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("inputs must be real numeric values") from exc
    if coeff.ndim != 1 or coeff.size < 2 or np.any(np.isinf(coeff)):
        raise ValueError("b must be a one-dimensional array with no infinite entries")
    if np.isnan(coeff[0]) or coeff[0] == 0.0:
        raise ValueError("b[0] must be known and nonzero")
    missing = np.flatnonzero(np.isnan(coeff))
    if missing.size > 1:
        raise ValueError("at most one EFT coefficient may be omitted")
    if not np.isfinite(residual_limit) or residual_limit <= 0.0:
        raise ValueError("residual_tol must be strictly positive and finite")
    if not np.isfinite(point):
        raise ValueError("s_eval must be finite")

    if missing.size == 1:
        recovered = recover_missing_eft_coefficient(coeff, rank_rtol, imag_tol)
        coeff[int(missing[0])] = recovered

    c = compute_log_derivative_coefficients(coeff)
    r, probe_size = choose_source_configuration(c)
    probe = build_hankel_probe(c, r, probe_size)
    d = infer_finite_spectrum_size(probe, rank_rtol)
    if d < 1:
        raise ValueError("the inferred finite spectrum is empty")
    current, shifted = build_shifted_hankel_pencil(c, r, d)
    spectral_data = recover_spectral_locations(current, shifted, imag_tol)
    classified, _weights = classify_spectral_locations(current, r, spectral_data)
    numerator, denominator, residual = reconstruct_rational_amplitude(coeff, classified)
    if residual >= residual_limit:
        raise ValueError("the EFT reconstruction residual exceeds residual_tol")

    denominator_value = float(np.polynomial.polynomial.polyval(point, denominator))
    if abs(denominator_value) <= 1.0e-12:
        raise ValueError("s_eval is a reconstructed pole")
    numerator_value = float(np.polynomial.polynomial.polyval(point, numerator))
    return float(np.round(numerator_value / denominator_value, 10))
SCICODE_GOLD_EOF
