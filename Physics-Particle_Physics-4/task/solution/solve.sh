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

def build_maximal_tree_reduction_atlas(side_length):
    import numpy as np

    if isinstance(side_length, (bool, np.bool_)) or not isinstance(
        side_length, (int, np.integer)
    ):
        raise ValueError("side_length must be an ordinary integer")
    L = int(side_length)
    if L < 2 or L > 5:
        raise ValueError("invalid side length")

    V = L ** 3
    atlas = np.zeros((V - 1, 6 + 2 * V), dtype=float)
    z_count = L * L * (L - 1)
    y_start = z_count
    x_start = z_count + L * (L - 1)

    def flat(x, y, z):
        return x + L * (y + L * z)

    def z_row(x, y, z):
        return z * L * L + y * L + x

    def y_row(x, y):
        return y_start + y * L + x

    def x_row(x):
        return x_start + x

    def incoming(x, y, z):
        if z > 0:
            return z_row(x, y, z - 1)
        if y > 0:
            return y_row(x, y - 1)
        if x > 0:
            return x_row(x - 1)
        return -1

    def fill(row, family, tail_xyz, head_xyz, descendants):
        tail = flat(*tail_xyz)
        head = flat(*head_xyz)
        parent = incoming(*tail_xyz)
        atlas[row, :6] = (
            family,
            tail,
            head,
            parent,
            sum(head_xyz),
            len(descendants),
        )
        atlas[row, 6 + tail] = -1.0
        atlas[row, 6 + head] = 1.0
        atlas[row, 6 + V + np.asarray(descendants, dtype=int)] = 1.0

    for z in range(L - 1):
        for y in range(L):
            for x in range(L):
                descendants = [flat(x, y, zz) for zz in range(z + 1, L)]
                fill(z_row(x, y, z), 3, (x, y, z), (x, y, z + 1), descendants)

    for y in range(L - 1):
        for x in range(L):
            descendants = [
                flat(x, yy, zz)
                for zz in range(L)
                for yy in range(y + 1, L)
            ]
            fill(y_row(x, y), 2, (x, y, 0), (x, y + 1, 0), descendants)

    for x in range(L - 1):
        descendants = [
            flat(xx, yy, zz)
            for zz in range(L)
            for yy in range(L)
            for xx in range(x + 1, L)
        ]
        fill(x_row(x), 1, (x, 0, 0), (x + 1, 0, 0), descendants)

    return atlas

import numpy as np

def assemble_qsvt_field_amplitude_objects(
    side_length,
    lattice_spacing,
    polynomial_degree,
    n_qubits,
    a_max,
    tree_atlas,
):
    import numpy as np

    def require_int(value, name):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError("invalid " + name)
        return int(value)

    L = require_int(side_length, "side_length")
    degree = require_int(polynomial_degree, "polynomial_degree")
    K = require_int(n_qubits, "n_qubits")
    try:
        a = float(lattice_spacing)
        cutoff = float(a_max)
    except Exception as exc:
        raise ValueError("invalid scale") from exc
    if (
        L < 2 or L > 5
        or degree < 3 or degree > 16
        or K < 3 or K > 6
        or not np.isfinite(a) or a <= 0.0
        or not np.isfinite(cutoff) or cutoff <= 0.0
    ):
        raise ValueError("invalid lattice or digitization")
    V = L ** 3
    try:
        raw = np.asarray(tree_atlas)
        if np.iscomplexobj(raw):
            if not np.all(np.isfinite(raw)) or np.any(raw.imag != 0.0):
                raise ValueError("invalid tree atlas")
            raw = raw.real
        atlas = np.asarray(raw, dtype=float)
    except Exception as exc:
        raise ValueError("invalid tree atlas") from exc
    if atlas.shape != (V - 1, 6 + 2 * V) or not np.all(np.isfinite(atlas)):
        raise ValueError("invalid tree atlas")

    B = atlas[:, 6 : 6 + V]
    families = atlas[:, 0]
    kernel = np.zeros((V, V), dtype=float)
    for code in (3.0, 2.0, 1.0):
        block = B[np.isclose(families, code)]
        kernel = kernel + block.T @ block / (a * a)

    W = np.zeros((V, V - 1), dtype=float)
    for j in range(V - 1):
        scale = np.sqrt((j + 1.0) * (j + 2.0))
        W[: j + 1, j] = 1.0 / scale
        W[j + 1, j] = -(j + 1.0) / scale
    reduced = W.T @ kernel @ W
    spectrum = np.linalg.eigvalsh(reduced)
    alpha = float(spectrum[0])
    beta = float(spectrum[-1])
    if not (alpha > 0.0 and beta > alpha):
        raise ValueError("invalid singlet interval")
    n_nodes = 2 * (degree + 1) + 1
    theta = np.pi * (np.arange(n_nodes) + 0.5) / n_nodes
    y = np.cos(theta)
    center = 0.5 * (alpha + beta)
    radius = 0.5 * (beta - alpha)
    coeffs = np.empty(degree + 1, dtype=float)
    for k in range(degree + 1):
        coeffs[k] = (2.0 / n_nodes) * np.sum(
            np.cos(k * theta) / (center + radius * y)
        )
    coeffs[0] *= 0.5
    mapped = (2.0 * reduced - (alpha + beta) * np.eye(V - 1)) / (beta - alpha)
    t_prev = np.eye(V - 1, dtype=float)
    t_curr = mapped.copy()
    green_red = coeffs[0] * t_prev + coeffs[1] * t_curr
    for k in range(2, degree + 1):
        t_next = 2.0 * mapped @ t_curr - t_prev
        green_red = green_red + coeffs[k] * t_next
        t_prev, t_curr = t_curr, t_next
    green = W @ green_red @ W.T
    green = 0.5 * (green + green.T)

    n_states = 2 ** K
    labels = np.arange(n_states)
    delta_a = 2.0 * cutoff / (n_states - 1)
    a_vals = -cutoff + labels * delta_a
    a_op = np.diag(a_vals)
    z_weights = np.empty(K, dtype=float)
    reconstructed = np.zeros((n_states, n_states), dtype=float)
    for bit in range(K):
        pauli_z = np.diag(1.0 - 2.0 * ((labels >> bit) & 1).astype(float))
        z_weights[bit] = -delta_a * (2.0 ** (bit - 1))
        reconstructed = reconstructed + z_weights[bit] * pauli_z
    fourier = np.exp(
        2.0j * np.pi * np.outer(labels, labels) / n_states
    ) / np.sqrt(n_states)
    delta_pi = 1.0 / cutoff
    pi_op = (delta_pi / delta_a) * (fourier.conj().T @ a_op @ fourier)
    operators = np.stack((a_op, np.real(pi_op), np.imag(pi_op)))
    if not (
        np.allclose(a_op, reconstructed, rtol=1e-12, atol=1e-12)
        and np.all(np.isfinite(green))
        and np.all(np.isfinite(z_weights))
        and np.all(np.isfinite(operators))
    ):
        raise ValueError("nonfinite digitization or Green")
    return green, z_weights, operators

import numpy as np

def build_reduced_nonabelian_gauss_response(
    side_length,
    lattice_spacing,
    frame_index,
    tree_atlas,
):
    import numpy as np

    if isinstance(side_length, (bool, np.bool_)) or not isinstance(
        side_length, (int, np.integer)
    ):
        raise ValueError("side_length must be an ordinary integer")
    if isinstance(frame_index, (bool, np.bool_)) or not isinstance(
        frame_index, (int, np.integer)
    ):
        raise ValueError("frame_index must be an ordinary integer")
    L = int(side_length)
    frame = int(frame_index)
    try:
        a = float(lattice_spacing)
    except Exception as exc:
        raise ValueError("invalid lattice spacing") from exc
    if L < 2 or L > 5 or not np.isfinite(a) or a <= 0.0:
        raise ValueError("invalid lattice")
    if frame not in (0, 1, 2):
        raise ValueError("invalid residual frame")

    V = L ** 3
    try:
        raw_atlas = np.asarray(tree_atlas)
        if np.iscomplexobj(raw_atlas):
            if not np.all(np.isfinite(raw_atlas)) or np.any(raw_atlas.imag != 0.0):
                raise ValueError("tree atlas must be finite and real")
            raw_atlas = raw_atlas.real
        atlas = np.asarray(raw_atlas, dtype=float)
    except Exception as exc:
        raise ValueError("invalid tree atlas") from exc
    if atlas.shape != (V - 1, 6 + 2 * V) or not np.all(np.isfinite(atlas)):
        raise ValueError("invalid tree atlas")

    def flat(x, y, z):
        return x + L * (y + L * z)

    expected = []
    for z in range(L - 1):
        for y in range(L):
            for x in range(L):
                expected.append((3, flat(x, y, z), flat(x, y, z + 1)))
    for y in range(L - 1):
        for x in range(L):
            expected.append((2, flat(x, y, 0), flat(x, y + 1, 0)))
    for x in range(L - 1):
        expected.append((1, flat(x, 0, 0), flat(x + 1, 0, 0)))
    expected = np.asarray(expected, dtype=float)
    if not np.array_equal(atlas[:, :3], expected):
        raise ValueError("atlas endpoints or family order are inconsistent")

    B = atlas[:, 6 : 6 + V]
    wanted_B = np.zeros_like(B)
    row_numbers = np.arange(V - 1)
    wanted_B[row_numbers, expected[:, 1].astype(int)] = -1.0
    wanted_B[row_numbers, expected[:, 2].astype(int)] = 1.0
    if not np.array_equal(B, wanted_B):
        raise ValueError("atlas incidence block is inconsistent")

    lam = np.zeros((8, 3, 3), dtype=complex)
    lam[0, 0, 1] = lam[0, 1, 0] = 1.0
    lam[1, 0, 1] = -1.0j
    lam[1, 1, 0] = 1.0j
    lam[2] = np.diag([1.0, -1.0, 0.0])
    lam[3, 0, 2] = lam[3, 2, 0] = 1.0
    lam[4, 0, 2] = -1.0j
    lam[4, 2, 0] = 1.0j
    lam[5, 1, 2] = lam[5, 2, 1] = 1.0
    lam[6, 1, 2] = -1.0j
    lam[6, 2, 1] = 1.0j
    lam[7] = np.diag([1.0, 1.0, -2.0]) / np.sqrt(3.0)

    structure = np.zeros((8, 8, 8), dtype=float)
    for r in range(8):
        for s in range(8):
            commutator = lam[r] @ lam[s] - lam[s] @ lam[r]
            for t in range(8):
                structure[r, s, t] = float(
                    np.real(np.trace(commutator @ lam[t]) / (4.0j))
                )

    theta = np.roll(
        np.array([0.29, -0.17, 0.23, 0.11, -0.19, 0.13, -0.07, 0.31]),
        2 * frame,
    )
    theta[1::2] *= (-1.0) ** frame
    H = np.einsum("r,rij->ij", 0.5 * theta, lam)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    U = (eigenvectors * np.exp(1.0j * eigenvalues)) @ eigenvectors.conj().T
    rotation = np.empty((8, 8), dtype=float)
    for colour_out in range(8):
        for colour_in in range(8):
            rotation[colour_out, colour_in] = 0.5 * float(
                np.real(
                    np.trace(
                        lam[colour_out]
                        @ U
                        @ lam[colour_in]
                        @ U.conj().T
                    )
                )
            )

    tree_edges = {
        (int(round(row[1])), int(round(row[2])))
        for row in atlas
    }
    links = []
    for direction in (0, 1):
        for z in range(L):
            for y in range(L):
                for x in range(L):
                    coordinate = (x, y)[direction]
                    if coordinate >= L - 1:
                        continue
                    head_xyz = [x, y, z]
                    head_xyz[direction] += 1
                    tail = flat(x, y, z)
                    head = flat(*head_xyz)
                    if (tail, head) not in tree_edges:
                        links.append((direction, x, y, z, tail, head))

    Nphys = (L - 1) ** 2 * (2 * L + 1)
    if len(links) != Nphys:
        raise ValueError("atlas leaves an inconsistent physical-link count")
    response = np.zeros((2, 8 * V, 8 * Nphys), dtype=float)
    momenta = np.zeros(8 * Nphys, dtype=float)
    colours = np.arange(8, dtype=float)

    for link_index, (direction, x, y, z, tail, head) in enumerate(links):
        X = (x + 0.5) / L
        Y = (y + 0.5) / L
        Z = (z + 0.5) / L
        gauge = (
            0.41 * np.sin(
                np.pi * ((colours + 1.0) * X + 0.17 * (direction + 1) * Z)
            )
            + 0.23 * np.cos(
                np.pi
                * (
                    ((colours % 3.0) + 1.0) * Y
                    - 0.13 * (direction + 1) * X
                )
            )
        )
        momentum = (
            np.cos(
                np.pi
                * (
                    ((colours % 4.0) + 1.0) * Z
                    + 0.11 * (direction + 1) * Y
                )
            )
            + 0.37 * np.sin(
                np.pi
                * (
                    (colours + 2.0) * Y
                    - 0.19 * (direction + 1) * X
                )
            )
        )
        gauge = rotation @ gauge
        momentum = rotation @ momentum
        col0 = 8 * link_index
        momenta[col0 : col0 + 8] = momentum
        for b in range(8):
            response[0, b * V + tail, col0 + b] = 1.0
            response[0, b * V + head, col0 + b] = -1.0
            for r in range(8):
                response[1, b * V + tail, col0 + r] = np.dot(
                    structure[r, b], gauge
                )

    # Work at unit spacing, then restore the known powers of a. The physical
    # kernel or its inverse may overflow while the returned tuple is finite.
    K = B.T @ B
    kernel_values, kernel_vectors = np.linalg.eigh(K)
    tolerance = 1e-11 * float(kernel_values[-1])
    if (
        abs(float(kernel_values[0])) > tolerance
        or float(kernel_values[1]) <= tolerance
    ):
        raise ValueError("atlas incidence does not define one connected tree")
    inverse_values = np.zeros_like(kernel_values)
    inverse_values[1:] = 1.0 / kernel_values[1:]
    green = (kernel_vectors * inverse_values) @ kernel_vectors.T
    green -= np.mean(green, axis=0, keepdims=True)
    green -= np.mean(green, axis=1, keepdims=True)
    colour_green = np.kron(np.eye(8), green)
    M0, M1 = response
    coefficients = np.stack(
        (
            0.5 * (M0.T @ colour_green @ M0),
            0.5 * (M0.T @ colour_green @ M1 + M1.T @ colour_green @ M0),
            0.5 * (M1.T @ colour_green @ M1),
        )
    )
    coefficients = 0.5 * (coefficients + coefficients.transpose(0, 2, 1))
    coefficients[1] *= a
    coefficients[2] *= a
    coefficients[2] *= a
    response[0] /= a
    if not (
        np.all(np.isfinite(response))
        and np.all(np.isfinite(coefficients))
        and np.all(np.isfinite(momenta))
    ):
        raise ValueError("nonfinite reduced response")
    return response, coefficients, momenta

import numpy as np

def assemble_maximal_tree_gauss_terms(
    side_length, lattice_spacing, tree_atlas
):
    import numpy as np

    if isinstance(side_length, (bool, np.bool_)) or not isinstance(
        side_length, (int, np.integer)
    ):
        raise ValueError("side_length must be an ordinary integer")
    L = int(side_length)
    try:
        a = float(lattice_spacing)
    except Exception as exc:
        raise ValueError("invalid lattice spacing") from exc
    V = L ** 3
    try:
        raw_atlas = np.asarray(tree_atlas)
        if np.iscomplexobj(raw_atlas):
            if not np.all(np.isfinite(raw_atlas)) or np.any(raw_atlas.imag != 0.0):
                raise ValueError("tree atlas must be finite and real")
            raw_atlas = raw_atlas.real
        atlas = np.asarray(raw_atlas, dtype=float)
    except Exception as exc:
        raise ValueError("invalid tree atlas") from exc
    if L < 2 or L > 5 or not np.isfinite(a) or a <= 0.0:
        raise ValueError("invalid lattice")
    if atlas.shape != (V - 1, 6 + 2 * V) or not np.all(np.isfinite(atlas)):
        raise ValueError("invalid tree atlas")

    incidence = atlas[:, 6 : 6 + V]
    terms = np.empty((3, V, V), dtype=float)
    for output_index, family in enumerate((3, 2, 1)):
        block = incidence[np.rint(atlas[:, 0]).astype(int) == family]
        derivative = block / a
        terms[output_index] = derivative.T @ derivative
    return terms

import numpy as np

def invert_gauss_kernel_on_singlet(kernel):
    import numpy as np

    if np.iscomplexobj(kernel):
        raise ValueError("kernel must be real")
    try:
        K = np.asarray(kernel, dtype=float)
    except Exception as exc:
        raise ValueError("invalid kernel") from exc
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] < 2:
        raise ValueError("kernel must be square")
    if not np.all(np.isfinite(K)):
        raise ValueError("kernel must be finite")
    N = K.shape[0]
    matrix_scale = float(np.max(np.abs(K)))
    if matrix_scale == 0.0:
        raise ValueError("kernel is not positive on the singlet subspace")
    Kn = K / matrix_scale
    scale = float(np.linalg.norm(Kn, ord=2))
    if np.max(np.abs(Kn-Kn.T)) > 1e-11*scale:
        raise ValueError("kernel must be symmetric")
    if np.linalg.norm(Kn @ np.ones(N)) > 1e-10 * scale * np.sqrt(N):
        raise ValueError("constant vector is not the zero mode")

    U = np.zeros((N, N - 1), dtype=float)
    for j in range(1, N):
        s = np.sqrt(j * (j + 1.0))
        U[:j, j - 1] = 1.0 / s
        U[j, j - 1] = -j / s
    reduced = U.T @ ((Kn+Kn.T)/2) @ U
    reduced = (reduced+reduced.T)/2
    lam = np.linalg.eigvalsh(reduced)
    if lam[0] <= 1e-11 * float(lam[-1]):
        raise ValueError("kernel is not positive on the singlet subspace")
    G = U @ np.linalg.solve(reduced, U.T)
    return (0.5 * (G + G.T)) / matrix_scale

import numpy as np

def build_ordered_tree_charge_flow_map(
    side_length, lattice_spacing, tree_atlas
):
    import numpy as np

    if isinstance(side_length, (bool, np.bool_)) or not isinstance(
        side_length, (int, np.integer)
    ):
        raise ValueError("side_length must be an ordinary integer")
    L = int(side_length)
    try:
        a = float(lattice_spacing)
    except Exception as exc:
        raise ValueError("invalid lattice spacing") from exc
    V = L ** 3
    try:
        raw_atlas = np.asarray(tree_atlas)
        if np.iscomplexobj(raw_atlas):
            if not np.all(np.isfinite(raw_atlas)) or np.any(raw_atlas.imag != 0.0):
                raise ValueError("tree atlas must be finite and real")
            raw_atlas = raw_atlas.real
        atlas = np.asarray(raw_atlas, dtype=float)
    except Exception as exc:
        raise ValueError("invalid tree atlas") from exc
    if L < 2 or L > 5 or not np.isfinite(a) or a <= 0.0:
        raise ValueError("invalid lattice")
    if atlas.shape != (V - 1, 6 + 2 * V) or not np.all(np.isfinite(atlas)):
        raise ValueError("invalid tree atlas")
    return a * atlas[:, 6 + V : 6 + 2 * V]

import numpy as np

def solve_temporal_gauge_field(kernel, charges):
    import numpy as np

    if np.iscomplexobj(kernel) or np.iscomplexobj(charges):
        raise ValueError("inputs must be real")
    try:
        K = np.asarray(kernel, dtype=float)
        Q = np.asarray(charges, dtype=float)
    except Exception as exc:
        raise ValueError("invalid input") from exc
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] < 2:
        raise ValueError("kernel must be square")
    N = K.shape[0]
    if Q.ndim != 2 or Q.shape[1] != N or Q.shape[0] < 1:
        raise ValueError("charges must have shape (C,N)")
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(Q)):
        raise ValueError("inputs must be finite")
    matrix_scale = float(np.max(np.abs(K)))
    if matrix_scale == 0.0:
        raise ValueError("kernel is not positive on the singlet subspace")
    Kn = K / matrix_scale
    scale_k = float(np.linalg.norm(Kn, ord=2))
    if np.max(np.abs(Kn-Kn.T)) > 1e-11*scale_k:
        raise ValueError("kernel must be symmetric")
    if np.linalg.norm(Kn @ np.ones(N)) > 1e-10 * scale_k * np.sqrt(N):
        raise ValueError("constant vector is not the zero mode")
    row_scale = np.maximum(1.0, np.max(np.abs(Q), axis=1))
    scaled_q = Q / row_scale[:, None]
    norm_q = np.maximum(1.0 / row_scale, np.linalg.norm(scaled_q, axis=1))
    if np.any(np.abs(np.sum(scaled_q, axis=1)) > 1e-10 * norm_q * np.sqrt(N)):
        raise ValueError("charges must be globally neutral")

    U = np.zeros((N, N - 1), dtype=float)
    for j in range(1, N):
        s = np.sqrt(j * (j + 1.0))
        U[:j, j - 1] = 1.0 / s
        U[j, j - 1] = -j / s
    reduced = U.T @ ((Kn+Kn.T)/2) @ U
    reduced = (reduced+reduced.T)/2
    lam = np.linalg.eigvalsh(reduced)
    if lam[0] <= 1e-11 * float(lam[-1]):
        raise ValueError("kernel is not positive on the singlet subspace")
    solve_scale = np.max(np.abs(Q), axis=1)
    solve_scale = np.where(solve_scale == 0.0, 1.0, solve_scale)
    solve_q = Q / solve_scale[:, None]
    coeff = np.linalg.solve(reduced, (solve_q @ U).T)
    field = -(U @ coeff).T
    # Combine exponents so representable fields survive extreme input scales.
    field_m, field_e = np.frexp(field)
    charge_m, charge_e = np.frexp(solve_scale[:, None])
    kernel_m, kernel_e = np.frexp(matrix_scale)
    return np.ldexp(field_m * charge_m / kernel_m, field_e + charge_e - kernel_e)

import numpy as np

def maximal_tree_nonlocal_energy(side_length, lattice_spacing):
    import numpy as np

    if isinstance(side_length, (bool, np.bool_)) or not isinstance(
        side_length, (int, np.integer)
    ):
        raise ValueError("side_length must be an ordinary integer")
    L = int(side_length)
    try:
        a = float(lattice_spacing)
    except Exception as exc:
        raise ValueError("invalid lattice spacing") from exc
    if L < 2 or L > 5 or not np.isfinite(a) or a <= 0.0:
        raise ValueError("invalid lattice")

    V = L ** 3
    atlas = build_maximal_tree_reduction_atlas(L)
    qsvt_green, z_weights, operators = (
        assemble_qsvt_field_amplitude_objects(L, a, 5, 4, 1.0, atlas)
    )
    ones = np.ones(V, dtype=float)
    labels = np.arange(16)
    pauli_a = np.zeros((16, 16), dtype=float)
    for bit in range(4):
        pauli_z = np.diag(1.0 - 2.0 * ((labels >> bit) & 1).astype(float))
        pauli_a = pauli_a + z_weights[bit] * pauli_z
    if (
        not np.allclose(qsvt_green @ ones, 0.0, rtol=1e-5, atol=1e-5)
        or not np.allclose(operators[0, 0, 0], -1.0, rtol=1e-5, atol=1e-5)
        or not np.allclose(operators[0, 15, 15], 1.0, rtol=1e-5, atol=1e-5)
        or not np.allclose(operators[0], pauli_a, rtol=1e-5, atol=1e-5)
    ):
        raise RuntimeError("QSVT Green or field-amplitude digitization is inconsistent")
    incidence = atlas[:, 6 : 6 + V]
    D = incidence / a
    terms = assemble_maximal_tree_gauss_terms(L, a, atlas)
    K = np.sum(terms, axis=0)
    cuts = (0, L * L * (L - 1), L ** 3 - L)
    stops = (cuts[1], cuts[2], L ** 3 - 1)
    for term, lo, hi in zip(terms, cuts, stops):
        block = D[lo:hi]
        if not np.allclose(term, block.T @ block, rtol=1e-5, atol=1e-5):
            raise RuntimeError("tree-incidence family and Gauss-kernel term disagree")

    F = build_ordered_tree_charge_flow_map(L, a, atlas)
    G = invert_gauss_kernel_on_singlet(K)
    response, coefficients, momenta = (
        build_reduced_nonabelian_gauss_response(L, a, 2, atlas)
    )
    coupling = 0.91
    source_flat = (response[0] + coupling * response[1]) @ momenta
    Q = source_flat.reshape(8, V)
    Q -= np.mean(Q, axis=1, keepdims=True)
    source_amplitude = float(np.max(np.abs(Q)))
    source_norm = (
        source_amplitude * np.linalg.norm(Q / source_amplitude)
        if np.isfinite(source_amplitude) and source_amplitude > 0.0
        else source_amplitude
    )
    target_norm = np.linalg.norm(1.0 + 0.1 * np.arange(8, dtype=float))
    if not np.isfinite(source_norm) or source_norm == 0.0:
        raise RuntimeError("the reduced response produced a degenerate source")
    source_scale = target_norm / source_norm
    Q *= source_scale
    if not np.allclose(np.sum(Q, axis=1), 0.0, rtol=0.0, atol=1e-5):
        raise RuntimeError("source violates the residual global site singlet")
    flows = Q @ F.T
    if not np.allclose(flows @ D, Q, rtol=1e-5, atol=1e-5):
        raise RuntimeError("rooted tree flows do not reproduce neutral charge")

    A0 = solve_temporal_gauge_field(K, Q)
    if not np.allclose(A0, -Q @ G.T, rtol=1e-5, atol=1e-5):
        raise RuntimeError("independent temporal-field routes disagree")
    coefficient_matrix = (
        coefficients[0]
        + coupling * coefficients[1]
        + coupling * coupling * coefficients[2]
    )
    # Scaling both factors applies s**2 before a potentially overflowing contraction.
    scaled_momenta = source_scale * momenta
    energy = float(scaled_momenta @ coefficient_matrix @ scaled_momenta)
    green_energy = float(np.einsum("bi,ij,bj->", 0.5 * Q, G, Q))
    temporal_energy = float(np.sum((-0.5 * Q) * A0))
    if not np.allclose(
        energy, green_energy, rtol=1e-5, atol=1e-5
    ) or not np.allclose(
        energy, temporal_energy, rtol=1e-5, atol=1e-5
    ):
        raise RuntimeError("coefficient, Green, and temporal energies disagree")
    if not np.allclose(
        energy, np.sum((0.5 * flows) * flows), rtol=1e-5, atol=1e-5
    ):
        raise RuntimeError("Green and rooted-flow energies disagree")
    return energy
SCICODE_GOLD_EOF
