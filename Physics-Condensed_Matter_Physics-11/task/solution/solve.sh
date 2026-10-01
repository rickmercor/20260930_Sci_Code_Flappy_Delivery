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


def modulation_coupling_matrix(alpha_m: float, E0: float, n_order: int) -> dict:
    alpha_m = float(alpha_m)
    E0 = float(E0)
    if not np.isfinite(alpha_m) or abs(alpha_m) >= 1.0:
        raise ValueError("alpha_m must be finite and of magnitude below one")
    if not np.isfinite(E0) or E0 <= 0.0:
        raise ValueError("E0 must be finite and above zero")
    if int(n_order) != n_order or int(n_order) < 1:
        raise ValueError("n_order must be an integer of one or more")
    n_order = int(n_order)

    coefficients = np.array([E0 * alpha_m / 2.0, E0, E0 * alpha_m / 2.0], dtype=np.float64)
    orders = np.arange(-n_order, n_order + 1)
    offset = orders[:, None] - orders[None, :]
    coupling = np.zeros((2 * n_order + 1, 2 * n_order + 1), dtype=np.float64)
    for p_index, p in enumerate((-1, 0, 1)):
        coupling[offset == p] = coefficients[p_index]
    return {"coefficients": coefficients, "coupling": coupling}

import numpy as np


def floquet_quadratic_operator(
    omega: float,
    coupling: np.ndarray,
    rho0: float,
    kappa_m: float,
    omega_m: float,
) -> dict:
    coupling = np.asarray(coupling, dtype=np.float64)
    if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
        raise ValueError("coupling must be a square matrix")
    if coupling.shape[0] % 2 != 1:
        raise ValueError("coupling must have an odd side length")
    if not np.all(np.isfinite(coupling)):
        raise ValueError("coupling must be finite")
    omega = float(omega)
    rho0 = float(rho0)
    kappa_m = float(kappa_m)
    omega_m = float(omega_m)
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and above zero")
    if not np.isfinite(rho0) or rho0 <= 0.0:
        raise ValueError("rho0 must be finite and above zero")
    if not np.isfinite(kappa_m) or kappa_m == 0.0:
        raise ValueError("kappa_m must be finite and not zero")
    if not np.isfinite(omega_m) or omega_m == 0.0:
        raise ValueError("omega_m must be finite and not zero")

    n_order = (coupling.shape[0] - 1) // 2
    orders = np.arange(-n_order, n_order + 1, dtype=np.float64)
    shift = np.diag(orders * kappa_m)
    inertia = np.diag((omega + orders * omega_m) ** 2)

    a2 = coupling.copy()
    a1 = shift @ coupling + coupling @ shift
    a0 = shift @ coupling @ shift - rho0 * inertia
    return {"a2": a2, "a1": a1, "a0": a0}

import numpy as np
import scipy.linalg as sla


def floquet_wavenumber_spectrum(a2: np.ndarray, a1: np.ndarray, a0: np.ndarray) -> dict:
    mats = [np.asarray(m, dtype=np.float64) for m in (a2, a1, a0)]
    for m in mats:
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("each coefficient matrix must be square")
        if not np.all(np.isfinite(m)):
            raise ValueError("each coefficient matrix must be finite")
    if mats[0].shape != mats[1].shape or mats[1].shape != mats[2].shape:
        raise ValueError("the coefficient matrices must share one shape")
    side = mats[0].shape[0]
    if side % 2 != 1:
        raise ValueError("the coefficient matrices must have odd side length")
    a2m, a1m, a0m = mats

    zero = np.zeros((side, side), dtype=np.float64)
    identity = np.eye(side, dtype=np.float64)
    left = np.block([[zero, identity], [-a0m, -a1m]])
    right = np.block([[identity, zero], [zero, a2m]])
    values, vectors = sla.eig(left, right)

    keep = np.isfinite(values)
    values = values[keep]
    vectors = vectors[:, keep]
    if values.size != 2 * side:
        raise ValueError("the companion pencil did not return twice the side length in finite eigenvalues")

    shapes = np.asarray(vectors[:side, :], dtype=np.complex128)
    norms = np.linalg.norm(shapes, axis=0)
    if np.any(norms <= 0.0):
        raise ValueError("the pencil returned a null mode shape")
    shapes = shapes / norms

    order = np.lexsort((values.imag, values.real))
    values = np.asarray(values[order], dtype=np.complex128)
    shapes = np.ascontiguousarray(shapes[:, order])
    return {
        "wavenumbers": values,
        "mode_shapes": shapes,
        "max_abs_imag": float(np.abs(values.imag).max()),
    }

import numpy as np


def directional_mode_families(
    wavenumbers: np.ndarray,
    mode_shapes: np.ndarray,
    coupling: np.ndarray,
    omega: float,
    rho0: float,
    kappa_m: float,
    omega_m: float,
) -> dict:
    wavenumbers = np.asarray(wavenumbers, dtype=np.complex128)
    mode_shapes = np.asarray(mode_shapes, dtype=np.complex128)
    coupling = np.asarray(coupling, dtype=np.float64)
    omega = float(omega)
    rho0 = float(rho0)
    kappa_m = float(kappa_m)
    omega_m = float(omega_m)

    if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1] or coupling.shape[0] % 2 != 1:
        raise ValueError("coupling must be square with odd side length")
    side = coupling.shape[0]
    n_order = (side - 1) // 2
    if wavenumbers.ndim != 1 or wavenumbers.size != 2 * side:
        raise ValueError("wavenumbers must hold twice the side length of coupling")
    if mode_shapes.shape != (side, 2 * side):
        raise ValueError("mode_shapes must have shape (side, twice side)")
    if not (np.all(np.isfinite(wavenumbers)) and np.all(np.isfinite(mode_shapes))
            and np.all(np.isfinite(coupling))):
        raise ValueError("wavenumbers, mode_shapes and coupling must be finite")
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and above zero")
    if not np.isfinite(rho0) or rho0 <= 0.0:
        raise ValueError("rho0 must be finite and above zero")
    if not np.isfinite(kappa_m) or kappa_m == 0.0:
        raise ValueError("kappa_m must be finite and not zero")
    if not np.isfinite(omega_m) or omega_m == 0.0:
        raise ValueError("omega_m must be finite and not zero")

    orders = np.arange(-n_order, n_order + 1, dtype=np.float64)
    frequencies = omega + orders * omega_m
    group = np.zeros(2 * side, dtype=np.complex128)
    for j in range(2 * side):
        mode = mode_shapes[:, j]
        shifted = wavenumbers[j] + orders * kappa_m
        numerator = mode @ (shifted * (coupling @ mode))
        denominator = rho0 * (mode @ (frequencies * mode))
        if denominator == 0.0:
            raise ValueError("a mode has vanishing inertia projection; the group velocity is undefined")
        group[j] = numerator / denominator

    forward = np.flatnonzero(group.real > 0.0)
    backward = np.flatnonzero(group.real < 0.0)
    if forward.size != side or backward.size != side:
        raise ValueError("the group velocity did not split the spectrum into two equal families")

    forward = forward[np.argsort(wavenumbers[forward].real, kind="stable")]
    backward = backward[np.argsort(-wavenumbers[backward].real, kind="stable")]
    return {
        "forward_wavenumbers": np.ascontiguousarray(wavenumbers[forward]),
        "backward_wavenumbers": np.ascontiguousarray(wavenumbers[backward]),
        "forward_modes": np.ascontiguousarray(mode_shapes[:, forward]),
        "backward_modes": np.ascontiguousarray(mode_shapes[:, backward]),
        "group_velocities": group,
    }

import numpy as np


def interface_coupling_system(
    forward_wavenumbers: np.ndarray,
    forward_modes: np.ndarray,
    backward_wavenumbers: np.ndarray,
    backward_modes: np.ndarray,
    coefficients: np.ndarray,
    omega: float,
    rho0: float,
    kappa_m: float,
    omega_m: float,
    segment_length: float,
) -> dict:
    kf = np.asarray(forward_wavenumbers, dtype=np.complex128)
    kb = np.asarray(backward_wavenumbers, dtype=np.complex128)
    uf = np.asarray(forward_modes, dtype=np.complex128)
    ub = np.asarray(backward_modes, dtype=np.complex128)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    omega = float(omega)
    rho0 = float(rho0)
    kappa_m = float(kappa_m)
    omega_m = float(omega_m)
    segment_length = float(segment_length)

    if coefficients.ndim != 1 or coefficients.size % 2 != 1:
        raise ValueError("coefficients must be a one-dimensional array of odd length")
    p_order = (coefficients.size - 1) // 2
    if kf.ndim != 1 or kb.shape != kf.shape:
        raise ValueError("the two wavenumber families must be one-dimensional and of equal size")
    side = kf.size
    if side % 2 != 1:
        raise ValueError("each family must hold an odd number of basic modes")
    if uf.shape != (side, side) or ub.shape != (side, side):
        raise ValueError("each mode-shape array must be square and match its family")
    for arr in (kf, kb, uf, ub, coefficients):
        if not np.all(np.isfinite(arr)):
            raise ValueError("every input array must be finite")
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and above zero")
    if not np.isfinite(rho0) or rho0 <= 0.0:
        raise ValueError("rho0 must be finite and above zero")
    if not np.isfinite(segment_length) or segment_length <= 0.0:
        raise ValueError("segment_length must be finite and above zero")
    if not np.isfinite(kappa_m) or kappa_m == 0.0:
        raise ValueError("kappa_m must be finite and not zero")
    if not np.isfinite(omega_m) or omega_m == 0.0:
        raise ValueError("omega_m must be finite and not zero")

    n_order = (side - 1) // 2
    j_order = n_order - p_order
    if j_order < 0:
        raise ValueError("the truncation leaves no matchable scattering order")

    e0 = float(coefficients[p_order])
    speed = np.sqrt(e0 / rho0)
    width = 2 * j_order + 1
    size = 4 * width
    matrix = np.zeros((size, size), dtype=np.complex128)
    rhs = np.zeros(size, dtype=np.complex128)

    def _amp_forward(s, n):
        return uf[n + n_order, s + n_order]

    def _amp_backward(s, n):
        return ub[n + n_order, s + n_order]

    def _shifted_forward(s, n):
        return kf[s + n_order] + n * kappa_m

    def _shifted_backward(s, n):
        return kb[s + n_order] + n * kappa_m

    def _exterior(n, sign):
        return sign * (omega + n * omega_m) / speed

    col_b = 0
    col_c = width
    col_r = 2 * width
    col_t = 3 * width
    incident_wavenumber = omega / speed

    row = 0
    for j in range(-j_order, j_order + 1):
        for s in range(-j_order, j_order + 1):
            matrix[row, col_b + s + j_order] -= _amp_forward(s, j)
            matrix[row, col_c + s + j_order] -= _amp_backward(s, j)
        matrix[row, col_r + j + j_order] += 1.0
        rhs[row] = -1.0 if j == 0 else 0.0
        row += 1

    for j in range(-j_order, j_order + 1):
        for s in range(-j_order, j_order + 1):
            for p in range(-p_order, p_order + 1):
                m = j - p
                if -n_order <= m <= n_order:
                    weight = coefficients[p + p_order]
                    matrix[row, col_b + s + j_order] -= weight * _shifted_forward(s, m) * _amp_forward(s, m)
                    matrix[row, col_c + s + j_order] -= weight * _shifted_backward(s, m) * _amp_backward(s, m)
        matrix[row, col_r + j + j_order] += e0 * _exterior(j, -1.0)
        rhs[row] = -e0 * incident_wavenumber if j == 0 else 0.0
        row += 1

    for j in range(-j_order, j_order + 1):
        for s in range(-j_order, j_order + 1):
            matrix[row, col_b + s + j_order] += _amp_forward(s, j) * np.exp(-1j * _shifted_forward(s, j) * segment_length)
            matrix[row, col_c + s + j_order] += _amp_backward(s, j) * np.exp(-1j * _shifted_backward(s, j) * segment_length)
        matrix[row, col_t + j + j_order] -= np.exp(-1j * _exterior(j, 1.0) * segment_length)
        row += 1

    for j in range(-j_order, j_order + 1):
        for s in range(-j_order, j_order + 1):
            phase_f = np.exp(-1j * _shifted_forward(s, j) * segment_length)
            phase_b = np.exp(-1j * _shifted_backward(s, j) * segment_length)
            for p in range(-p_order, p_order + 1):
                m = j - p
                if -n_order <= m <= n_order:
                    weight = coefficients[p + p_order]
                    matrix[row, col_b + s + j_order] += weight * _shifted_forward(s, m) * _amp_forward(s, m) * phase_f
                    matrix[row, col_c + s + j_order] += weight * _shifted_backward(s, m) * _amp_backward(s, m) * phase_b
        matrix[row, col_t + j + j_order] -= e0 * _exterior(j, 1.0) * np.exp(-1j * _exterior(j, 1.0) * segment_length)
        row += 1

    return {"matrix": matrix, "rhs": rhs, "j_order": int(j_order)}

import numpy as np


def harmonic_scattering_coefficients(matrix: np.ndarray, rhs: np.ndarray, j_order: int) -> dict:
    matrix = np.asarray(matrix, dtype=np.complex128)
    rhs = np.asarray(rhs, dtype=np.complex128)
    if int(j_order) != j_order or int(j_order) < 0:
        raise ValueError("j_order must be a non-negative integer")
    j_order = int(j_order)
    width = 2 * j_order + 1
    size = 4 * width
    if matrix.ndim != 2 or matrix.shape != (size, size):
        raise ValueError("matrix must be square of side four times two j_order plus one")
    if rhs.ndim != 1 or rhs.size != size:
        raise ValueError("rhs must match the side of matrix")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
        raise ValueError("matrix and rhs must be finite")

    condition_number = float(np.linalg.cond(matrix))
    if not np.isfinite(condition_number):
        raise ValueError("the interface system is singular")
    try:
        solution = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the interface system is singular") from exc

    reflection = np.abs(solution[2 * width:3 * width]).astype(np.float64)
    transmission = np.abs(solution[3 * width:4 * width]).astype(np.float64)
    return {
        "transmission": transmission,
        "reflection": reflection,
        "condition_number": condition_number,
    }

import numpy as np


def harmonic_power_budget(
    transmission: np.ndarray,
    reflection: np.ndarray,
    omega: float,
    omega_m: float,
) -> dict:
    transmission = np.asarray(transmission, dtype=np.float64)
    reflection = np.asarray(reflection, dtype=np.float64)
    omega = float(omega)
    omega_m = float(omega_m)
    if transmission.ndim != 1 or reflection.ndim != 1:
        raise ValueError("transmission and reflection must be one-dimensional")
    if transmission.shape != reflection.shape:
        raise ValueError("transmission and reflection must have the same shape")
    if transmission.size % 2 != 1:
        raise ValueError("transmission and reflection must have odd length")
    if not np.all(np.isfinite(transmission)) or not np.all(np.isfinite(reflection)):
        raise ValueError("transmission and reflection must be finite")
    if np.any(transmission < 0.0) or np.any(reflection < 0.0):
        raise ValueError("transmission and reflection are magnitudes and cannot be negative")
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and above zero")
    if not np.isfinite(omega_m) or omega_m == 0.0:
        raise ValueError("omega_m must be finite and not zero")

    j_order = (transmission.size - 1) // 2
    orders = np.arange(-j_order, j_order + 1, dtype=np.float64)
    weight = ((omega + orders * omega_m) / omega) ** 2
    transmitted_gain = float(np.sum(weight * transmission ** 2))
    reflected_gain = float(np.sum(weight * reflection ** 2))
    return {
        "transmitted_gain": transmitted_gain,
        "reflected_gain": reflected_gain,
        "total_gain": transmitted_gain + reflected_gain,
    }

import numpy as np


def nonreciprocal_gain_contrast(
    alpha_m: float,
    E0: float,
    rho0: float,
    kappa_m: float,
    omega_m: float,
    omega: float,
    segment_length: float,
    n_order: int,
) -> dict:
    if int(n_order) != n_order or int(n_order) < 2:
        raise ValueError("n_order must be an integer of two or more")
    n_order = int(n_order)

    built = modulation_coupling_matrix(alpha_m, E0, n_order)  # noqa: F821
    coefficients = built["coefficients"]
    coupling = built["coupling"]

    results = {}
    for label, sign in (("positive", 1.0), ("negative", -1.0)):
        travelling = sign * float(kappa_m)
        operator = floquet_quadratic_operator(  # noqa: F821
            omega, coupling, rho0, travelling, omega_m
        )
        spectrum = floquet_wavenumber_spectrum(  # noqa: F821
            operator["a2"], operator["a1"], operator["a0"]
        )
        families = directional_mode_families(  # noqa: F821
            spectrum["wavenumbers"], spectrum["mode_shapes"], coupling,
            omega, rho0, travelling, omega_m,
        )
        system = interface_coupling_system(  # noqa: F821
            families["forward_wavenumbers"], families["forward_modes"],
            families["backward_wavenumbers"], families["backward_modes"],
            coefficients, omega, rho0, travelling, omega_m, segment_length,
        )
        scattered = harmonic_scattering_coefficients(  # noqa: F821
            system["matrix"], system["rhs"], system["j_order"]
        )
        budget = harmonic_power_budget(  # noqa: F821
            scattered["transmission"], scattered["reflection"], omega, omega_m
        )
        results[label] = (budget, scattered, system["j_order"])

    positive_gain = float(results["positive"][0]["total_gain"])
    negative_gain = float(results["negative"][0]["total_gain"])
    if negative_gain == 0.0:
        raise ValueError("the negative-incidence run returned a gain of zero")

    scattered_positive, j_order = results["positive"][1], results["positive"][2]
    return {
        "positive_gain": positive_gain,
        "negative_gain": negative_gain,
        "contrast": positive_gain / negative_gain,
        "positive_transmission_zero": float(scattered_positive["transmission"][j_order]),
        "positive_reflection_minus_one": float(scattered_positive["reflection"][j_order - 1]),
    }
SCICODE_GOLD_EOF
