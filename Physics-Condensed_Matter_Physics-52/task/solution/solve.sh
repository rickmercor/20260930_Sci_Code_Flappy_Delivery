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


def build_noncommuting_impurity_model(
    impurity_energy: "np.ndarray",
    bath_energies: "np.ndarray",
    bath_couplings: "np.ndarray",
    interaction_u: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    impurity_energy = np.asarray(impurity_energy, dtype=np.complex128)
    bath_energies = np.asarray(bath_energies, dtype=np.complex128)
    bath_couplings = np.asarray(bath_couplings, dtype=np.complex128)
    if impurity_energy.shape != (2, 2):
        raise ValueError("impurity_energy must have shape (2,2)")
    if bath_energies.ndim != 3 or bath_energies.shape[1:] != (2, 2):
        raise ValueError("bath_energies must have shape (M,2,2)")
    if bath_couplings.shape != bath_energies.shape:
        raise ValueError("bath_couplings must match bath_energies")
    if any(not np.all(np.isfinite(x)) for x in
           (impurity_energy, bath_energies, bath_couplings)):
        raise ValueError("all one-body arrays must be finite")
    if not np.allclose(impurity_energy, impurity_energy.conj().T,
                       rtol=0.0, atol=2e-12):
        raise ValueError("impurity_energy must be Hermitian")
    if any(not np.allclose(block, block.conj().T, rtol=0.0, atol=2e-12)
           for block in bath_energies):
        raise ValueError("each bath-energy block must be Hermitian")
    if (not np.isscalar(interaction_u)
            or not np.isfinite(interaction_u)
            or abs(complex(interaction_u).imag) > 0.0):
        raise ValueError("interaction_u must be finite and real")

    site_count = int(bath_energies.shape[0])
    mode_count = 2 * (site_count + 1)
    dimension = 1 << mode_count
    one_body = np.zeros((mode_count, mode_count), dtype=np.complex128)
    one_body[:2, :2] = impurity_energy
    for site in range(site_count):
        block = slice(2 * (site + 1), 2 * (site + 2))
        one_body[block, block] = bath_energies[site]
        one_body[:2, block] = bath_couplings[site].conj().T
        one_body[block, :2] = bath_couplings[site]

    annihilators = np.zeros(
        (mode_count, dimension, dimension), dtype=np.complex128
    )
    for mode in range(mode_count):
        lower_mask = (1 << mode) - 1
        for state in range(dimension):
            if (state >> mode) & 1:
                target = state ^ (1 << mode)
                parity = (state & lower_mask).bit_count() & 1
                annihilators[mode, target, state] = -1.0 if parity else 1.0

    hamiltonian = np.zeros((dimension, dimension), dtype=np.complex128)
    for p in range(mode_count):
        creation = annihilators[p].conj().T
        for q in range(mode_count):
            if one_body[p, q] != 0.0:
                hamiltonian += one_body[p, q] * creation @ annihilators[q]
    number_0 = annihilators[0].conj().T @ annihilators[0]
    number_1 = annihilators[1].conj().T @ annihilators[1]
    interaction_hamiltonian = float(np.real(interaction_u)) * number_0 @ number_1
    hamiltonian += interaction_hamiltonian
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    occupations = np.asarray(
        [state.bit_count() for state in range(dimension)], dtype=np.int64
    )
    return hamiltonian, interaction_hamiltonian, annihilators, occupations

import numpy as np


def _aml_hermitian(matrix):
    return 0.5 * (matrix + matrix.conj().T)


def _aml_cluster_spectrum(poles, residues):
    poles = np.asarray(poles, dtype=np.float64)
    residues = np.asarray(residues, dtype=np.complex128).reshape((-1, 4, 4))
    if poles.size == 0:
        return poles, residues
    order = np.argsort(poles, kind="stable")
    poles, residues = poles[order], residues[order]
    out_poles, out_residues = [], []
    start = 0
    while start < poles.size:
        stop = start + 1
        while (stop < poles.size
               and poles[stop] - poles[start] <= 1e-10):
            stop += 1
        block = residues[start:stop]
        total_residue = _aml_hermitian(np.sum(block, axis=0))
        traces = np.real(np.trace(block, axis1=1, axis2=2))
        total = float(np.sum(traces))
        if total > 1e-12:
            out_poles.append(float(np.dot(traces, poles[start:stop]) / total))
            out_residues.append(total_residue)
        start = stop
    return (np.asarray(out_poles, dtype=np.float64),
            np.asarray(out_residues, dtype=np.complex128).reshape((-1, 4, 4)))


def compute_augmented_matrix_lehmann_spectrum(
    hamiltonian: "np.ndarray",
    interaction_hamiltonian: "np.ndarray",
    annihilators: "np.ndarray",
    occupations: "np.ndarray",
    particle_number: int,
) -> "tuple[float, np.ndarray, np.ndarray]":
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    interaction = np.asarray(interaction_hamiltonian, dtype=np.complex128)
    annihilators = np.asarray(annihilators, dtype=np.complex128)
    occupations = np.asarray(occupations)
    if (hamiltonian.ndim != 2 or hamiltonian.shape[0] == 0
            or hamiltonian.shape[0] != hamiltonian.shape[1]):
        raise ValueError("hamiltonian must be a nonempty square matrix")
    dimension = int(hamiltonian.shape[0])
    if dimension & (dimension - 1):
        raise ValueError("the Fock-space dimension must be a power of two")
    mode_count = dimension.bit_length() - 1
    if mode_count < 2:
        raise ValueError("at least two impurity modes are required")
    if interaction.shape != hamiltonian.shape:
        raise ValueError("interaction_hamiltonian has an incompatible shape")
    if annihilators.shape != (mode_count, dimension, dimension):
        raise ValueError("annihilators has an incompatible shape")
    if occupations.shape != (dimension,) or occupations.dtype.kind not in "iu":
        raise ValueError("occupations must be an integer vector")
    expected_occupations = np.asarray(
        [state.bit_count() for state in range(dimension)], dtype=np.int64
    )
    if not np.array_equal(occupations, expected_occupations):
        raise ValueError("occupations do not match the occupation-bit basis")
    if any(not np.all(np.isfinite(x)) for x in
           (hamiltonian, interaction, annihilators)):
        raise ValueError("all operator arrays must be finite")
    if (not np.allclose(hamiltonian, hamiltonian.conj().T,
                        rtol=0.0, atol=3e-11)
            or not np.allclose(interaction, interaction.conj().T,
                               rtol=0.0, atol=3e-11)):
        raise ValueError("Hamiltonians must be Hermitian")
    if not isinstance(particle_number, (int, np.integer)):
        raise ValueError("particle_number must be an integer")
    number = int(particle_number)
    if not 1 <= number <= mode_count - 1:
        raise ValueError("both adjacent particle sectors must exist")
    identity = np.eye(dimension, dtype=np.complex128)
    for mode in range(mode_count):
        anticommutator = (
            annihilators[mode] @ annihilators[mode].conj().T
            + annihilators[mode].conj().T @ annihilators[mode]
        )
        if not np.allclose(anticommutator, identity, rtol=0.0, atol=3e-11):
            raise ValueError("annihilators violate the canonical algebra")

    sectors = {}
    for sector in (number - 1, number, number + 1):
        indices = np.flatnonzero(occupations == sector)
        values, vectors = np.linalg.eigh(
            hamiltonian[np.ix_(indices, indices)]
        )
        sectors[sector] = indices, values, vectors
    indices, ground_values, ground_vectors = sectors[number]
    energy_scale = max(1.0, float(np.max(np.abs(ground_values), initial=0.0)))
    if (ground_values.size > 1
            and ground_values[1] - ground_values[0] <= 1e-10 * energy_scale):
        raise ValueError("the canonical reference state must be unique")
    ground_energy = float(ground_values[0])
    ground = np.zeros(dimension, dtype=np.complex128)
    ground[indices] = ground_vectors[:, 0]

    physical = [annihilators[0], annihilators[1]]
    auxiliary = [
        operator @ interaction - interaction @ operator
        for operator in physical
    ]
    operators = physical + auxiliary
    raw_poles, raw_residues = [], []

    indices, values, vectors = sectors[number + 1]
    for energy, column in zip(values, vectors.T):
        state = np.zeros(dimension, dtype=np.complex128)
        state[indices] = column
        row = np.asarray([
            np.vdot(state, operator.conj().T @ ground)
            for operator in operators
        ])
        raw_poles.append(float(energy - ground_energy))
        raw_residues.append(np.outer(row.conj(), row))

    indices, values, vectors = sectors[number - 1]
    for energy, column in zip(values, vectors.T):
        state = np.zeros(dimension, dtype=np.complex128)
        state[indices] = column
        amplitudes = np.asarray([
            np.vdot(state, operator @ ground) for operator in operators
        ])
        row = amplitudes.conj()
        raw_poles.append(float(ground_energy - energy))
        raw_residues.append(np.outer(row.conj(), row))

    poles, residues = _aml_cluster_spectrum(raw_poles, raw_residues)
    return ground_energy, poles, residues

import numpy as np
import scipy.linalg


def _mse_h(matrix):
    matrix = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (matrix + matrix.conj().T)


def _mse_power(matrix, exponent, *, require_positive=False):
    matrix = _mse_h(matrix)
    values, vectors = np.linalg.eigh(matrix)
    scale = max(1.0, float(np.max(np.abs(values), initial=0.0)))
    threshold = 2e-11 * scale
    if float(np.min(values, initial=0.0)) < -threshold:
        raise ValueError("a required spectral matrix is not positive semidefinite")
    kept = np.where(values > threshold, values, 0.0)
    if require_positive and np.any(kept == 0.0):
        raise ValueError("a required spectral metric is singular")
    if exponent < 0.0 and np.any(kept == 0.0):
        raise ValueError("a required spectral metric is singular")
    powered = np.where(kept > 0.0, kept ** exponent, 0.0)
    return _mse_h((vectors * powered) @ vectors.conj().T)


def _mse_factor_spectrum(poles, residues):
    poles = np.asarray(poles, dtype=np.float64)
    residues = np.asarray(residues, dtype=np.complex128)
    rows, expanded_poles = [], []
    for pole, residue in zip(poles, residues):
        residue = _mse_h(residue)
        values, vectors = np.linalg.eigh(residue)
        scale = max(1.0, float(np.max(np.abs(values), initial=0.0)))
        threshold = 2e-11 * scale
        if float(np.min(values, initial=0.0)) < -threshold:
            raise ValueError("a residue is not positive semidefinite")
        for index in np.flatnonzero(values > threshold):
            rows.append(np.sqrt(values[index]) * vectors[:, index].conj())
            expanded_poles.append(float(pole))
    dimension = int(residues.shape[-1])
    return (np.asarray(expanded_poles, dtype=np.float64),
            np.asarray(rows, dtype=np.complex128).reshape((-1, dimension)))


def _mse_cluster_spectrum(poles, residues):
    poles = np.asarray(poles, dtype=np.float64)
    residues = np.asarray(residues, dtype=np.complex128)
    dimension = int(residues.shape[-1]) if residues.ndim == 3 else 0
    residues = residues.reshape((-1, dimension, dimension))
    if poles.size == 0:
        return poles, residues
    order = np.argsort(poles, kind="stable")
    poles, residues = poles[order], residues[order]
    out_poles, out_residues = [], []
    start = 0
    while start < poles.size:
        stop = start + 1
        while (stop < poles.size
               and poles[stop] - poles[start] <= 1e-10):
            stop += 1
        block = residues[start:stop]
        total_residue = _mse_h(np.sum(block, axis=0))
        traces = np.real(np.trace(block, axis1=1, axis2=2))
        total = float(np.sum(traces))
        if total > 2e-11:
            out_poles.append(float(np.dot(traces, poles[start:stop]) / total))
            out_residues.append(total_residue)
        start = stop
    return (np.asarray(out_poles, dtype=np.float64),
            np.asarray(out_residues, dtype=np.complex128).reshape(
                (-1, dimension, dimension)
            ))


def recover_causal_matrix_self_energy(
    augmented_poles: "np.ndarray",
    augmented_residues: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    poles = np.asarray(augmented_poles)
    residues = np.asarray(augmented_residues, dtype=np.complex128)
    if poles.ndim != 1 or poles.size == 0:
        raise ValueError("augmented_poles must be a nonempty vector")
    if (not np.all(np.isfinite(poles))
            or np.iscomplexobj(poles) and np.max(np.abs(np.imag(poles))) > 2e-12):
        raise ValueError("augmented poles must be finite and real")
    poles = np.real(poles).astype(np.float64)
    if (residues.ndim != 3 or residues.shape[0] != poles.size
            or residues.shape[1] != residues.shape[2]
            or residues.shape[1] < 2 or residues.shape[1] % 2):
        raise ValueError("augmented_residues must have shape (P,2*N,2*N)")
    if not np.all(np.isfinite(residues)):
        raise ValueError("augmented residues must be finite")
    doubled_dimension = int(residues.shape[1])
    physical_dimension = doubled_dimension // 2
    for residue in residues:
        if not np.allclose(residue, residue.conj().T, rtol=0.0, atol=3e-11):
            raise ValueError("augmented residues must be Hermitian")

    expanded_poles, amplitudes = _mse_factor_spectrum(poles, residues)
    if amplitudes.shape[0] < physical_dimension:
        raise ValueError("the retained spectrum is incomplete")
    norm = _mse_h(amplitudes.conj().T @ amplitudes)
    if not np.allclose(norm[:physical_dimension, :physical_dimension],
                       np.eye(physical_dimension), rtol=3e-9, atol=3e-10):
        raise ValueError("the physical zeroth moment is not normalized")
    static = _mse_h(norm[:physical_dimension, physical_dimension:])
    auxiliary_norm = norm[physical_dimension:, physical_dimension:]
    if np.linalg.norm(auxiliary_norm, ord=2) <= 2e-11:
        return (static, np.empty(0, dtype=np.float64),
                np.empty((0, physical_dimension, physical_dimension),
                         dtype=np.complex128))

    norm_inverse_sqrt = _mse_power(norm, -0.5, require_positive=True)
    isometry = amplitudes @ norm_inverse_sqrt
    gram = _mse_h(isometry.conj().T @ isometry)
    isometry = isometry @ _mse_power(gram, -0.5, require_positive=True)
    complement = scipy.linalg.null_space(
        isometry.conj().T, rcond=2e-11
    )
    omega_isometry = expanded_poles[:, None] * isometry
    omega_aa = _mse_h(isometry.conj().T @ omega_isometry)

    if complement.shape[1]:
        omega_complement = expanded_poles[:, None] * complement
        omega_bb = _mse_h(complement.conj().T @ omega_complement)
        omega_ba = complement.conj().T @ omega_isometry
        inverse_poles, inverse_vectors = np.linalg.eigh(omega_bb)
        inverse_rows = inverse_vectors.conj().T @ omega_ba
    else:
        inverse_poles = np.empty(0, dtype=np.float64)
        inverse_rows = np.empty(
            (0, doubled_dimension), dtype=np.complex128
        )

    norm_inverse = _mse_h(norm_inverse_sqrt @ norm_inverse_sqrt)
    dynamic_norm = _mse_power(
        norm_inverse[physical_dimension:, physical_dimension:],
        -1.0,
        require_positive=True,
    )
    dynamic_sqrt = _mse_power(dynamic_norm, 0.5, require_positive=True)
    transformed_static = _mse_h(
        norm_inverse_sqrt @ omega_aa @ norm_inverse_sqrt
    )
    projected_static = _mse_h(
        dynamic_sqrt
        @ transformed_static[physical_dimension:, physical_dimension:]
        @ dynamic_sqrt
    )
    if inverse_rows.shape[0]:
        projected_rows = (
            inverse_rows @ norm_inverse_sqrt
        )[:, physical_dimension:] @ dynamic_sqrt
        linearization = np.block([
            [projected_static, projected_rows.conj().T],
            [projected_rows, np.diag(inverse_poles)],
        ])
    else:
        linearization = projected_static
    raw_poles, vectors = np.linalg.eigh(_mse_h(linearization))
    raw_residues = []
    for column in vectors[:physical_dimension, :].T:
        vector = dynamic_sqrt @ column
        raw_residues.append(np.outer(vector, vector.conj()))
    self_energy_poles, self_energy_residues = _mse_cluster_spectrum(
        raw_poles, raw_residues
    )
    return static, self_energy_poles, self_energy_residues

import numpy as np


def update_exact_matrix_hybridization(
    bath_energies: "np.ndarray",
    bath_couplings: "np.ndarray",
    self_energy_static: "np.ndarray",
    self_energy_poles: "np.ndarray",
    self_energy_residues: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    bath_energies = np.asarray(bath_energies, dtype=np.complex128)
    bath_couplings = np.asarray(bath_couplings, dtype=np.complex128)
    static = np.asarray(self_energy_static, dtype=np.complex128)
    poles = np.asarray(self_energy_poles)
    residues = np.asarray(self_energy_residues, dtype=np.complex128)
    if (bath_energies.ndim != 3 or bath_energies.shape[1] == 0
            or bath_energies.shape[1] != bath_energies.shape[2]):
        raise ValueError("bath_energies must have shape (M,N,N)")
    if bath_couplings.shape != bath_energies.shape:
        raise ValueError("bath_couplings must match bath_energies")
    dimension = int(bath_energies.shape[1])
    if static.shape != (dimension, dimension):
        raise ValueError("self_energy_static has an incompatible shape")
    if poles.ndim != 1:
        raise ValueError("self_energy_poles must be a vector")
    if residues.shape != (poles.size, dimension, dimension):
        raise ValueError("self_energy_residues has an incompatible shape")
    if any(not np.all(np.isfinite(x)) for x in
           (bath_energies, bath_couplings, static, poles, residues)):
        raise ValueError("all inputs must be finite")
    if np.iscomplexobj(poles) and np.max(np.abs(np.imag(poles)), initial=0.0) > 2e-12:
        raise ValueError("self-energy poles must be real")
    poles = np.real(poles).astype(np.float64)
    if not np.allclose(static, static.conj().T, rtol=0.0, atol=3e-11):
        raise ValueError("self_energy_static must be Hermitian")
    if any(not np.allclose(block, block.conj().T, rtol=0.0, atol=3e-11)
           for block in bath_energies):
        raise ValueError("bath-energy blocks must be Hermitian")
    for residue in residues:
        if not np.allclose(residue, residue.conj().T, rtol=0.0, atol=3e-11):
            raise ValueError("self-energy residues must be Hermitian")

    expanded_poles, rows = _mse_factor_spectrum(poles, residues)
    raw_poles, raw_residues = [], []
    for energy, coupling in zip(bath_energies, bath_couplings):
        if expanded_poles.size:
            star = np.block([
                [energy + static, rows.conj().T],
                [rows, np.diag(expanded_poles)],
            ])
        else:
            star = energy + static
        values, vectors = np.linalg.eigh(_mse_h(star))
        for pole, column in zip(values, vectors.T):
            projected = coupling.conj().T @ column[:dimension]
            raw_poles.append(float(pole))
            raw_residues.append(np.outer(projected, projected.conj()))
    if not raw_poles:
        return (np.empty(0, dtype=np.float64),
                np.empty((0, dimension, dimension), dtype=np.complex128))
    return _mse_cluster_spectrum(raw_poles, raw_residues)

import numpy as np


def _efi_sector_basis(mode_count, particle_number):
    return np.asarray([
        state for state in range(1 << mode_count)
        if state.bit_count() == particle_number
    ], dtype=np.int64)


def _efi_sector_hamiltonian(one_body, interaction_u, particle_number):
    mode_count = int(one_body.shape[0])
    basis = _efi_sector_basis(mode_count, particle_number)
    lookup = {int(state): index for index, state in enumerate(basis)}
    hamiltonian = np.zeros((basis.size, basis.size), dtype=np.complex128)
    for column, state_value in enumerate(basis):
        state = int(state_value)
        if (state & 1) and (state & 2):
            hamiltonian[column, column] += interaction_u
        for q in range(mode_count):
            if not ((state >> q) & 1):
                continue
            sign_q = -1.0 if ((state & ((1 << q) - 1)).bit_count() & 1) else 1.0
            intermediate = state ^ (1 << q)
            for p in range(mode_count):
                if (intermediate >> p) & 1:
                    continue
                sign_p = -1.0 if ((intermediate & ((1 << p) - 1)).bit_count() & 1) else 1.0
                target = intermediate | (1 << p)
                hamiltonian[lookup[target], column] += (
                    one_body[p, q] * sign_p * sign_q
                )
    return basis, _mse_h(hamiltonian)


def _efi_apply_impurity(
    source_basis, target_lookup, vectors, mode, create, interaction_u, auxiliary
):
    output = np.zeros(
        (len(target_lookup), vectors.shape[1]), dtype=np.complex128
    )
    other_mode = 1 - mode
    for row, state_value in enumerate(source_basis):
        state = int(state_value)
        if auxiliary and not ((state >> other_mode) & 1):
            continue
        occupied = (state >> mode) & 1
        if (create and occupied) or ((not create) and not occupied):
            continue
        sign = -1.0 if ((state & ((1 << mode) - 1)).bit_count() & 1) else 1.0
        target = state | (1 << mode) if create else state ^ (1 << mode)
        coefficient = interaction_u if auxiliary else 1.0
        output[target_lookup[target]] += coefficient * sign * vectors[row]
    return output


def solve_exact_feedback_impurity(
    impurity_energy: "np.ndarray",
    hybridization_poles: "np.ndarray",
    hybridization_residues: "np.ndarray",
    interaction_u: float,
    particle_number: int,
) -> "tuple[float, np.ndarray, np.ndarray]":
    impurity = np.asarray(impurity_energy, dtype=np.complex128)
    poles = np.asarray(hybridization_poles)
    residues = np.asarray(hybridization_residues, dtype=np.complex128)
    if impurity.shape != (2, 2):
        raise ValueError("impurity_energy must have shape (2,2)")
    if poles.ndim != 1:
        raise ValueError("hybridization_poles must be a vector")
    if residues.shape != (poles.size, 2, 2):
        raise ValueError("hybridization_residues must have shape (R,2,2)")
    if any(not np.all(np.isfinite(x)) for x in (impurity, poles, residues)):
        raise ValueError("all spectral inputs must be finite")
    if np.iscomplexobj(poles) and np.max(np.abs(np.imag(poles)), initial=0.0) > 2e-12:
        raise ValueError("hybridization poles must be real")
    poles = np.real(poles).astype(np.float64)
    if not np.allclose(impurity, impurity.conj().T, rtol=0.0, atol=3e-11):
        raise ValueError("impurity_energy must be Hermitian")
    for residue in residues:
        if not np.allclose(residue, residue.conj().T, rtol=0.0, atol=3e-11):
            raise ValueError("hybridization residues must be Hermitian")
    if (not np.isscalar(interaction_u) or not np.isfinite(interaction_u)
            or abs(complex(interaction_u).imag) > 0.0):
        raise ValueError("interaction_u must be finite and real")

    bath_poles, bath_rows = _mse_factor_spectrum(poles, residues)
    mode_count = 2 + int(bath_poles.size)
    if not isinstance(particle_number, (int, np.integer)):
        raise ValueError("particle_number must be an integer")
    number = int(particle_number)
    if not 1 <= number <= mode_count - 1:
        raise ValueError("both adjacent particle sectors must exist")
    one_body = np.zeros((mode_count, mode_count), dtype=np.complex128)
    one_body[:2, :2] = impurity
    if bath_poles.size:
        one_body[2:, 2:] = np.diag(bath_poles)
        one_body[2:, :2] = bath_rows
        one_body[:2, 2:] = bath_rows.conj().T

    basis_minus, h_minus = _efi_sector_hamiltonian(
        one_body, float(np.real(interaction_u)), number - 1
    )
    basis_ground, h_ground = _efi_sector_hamiltonian(
        one_body, float(np.real(interaction_u)), number
    )
    basis_plus, h_plus = _efi_sector_hamiltonian(
        one_body, float(np.real(interaction_u)), number + 1
    )
    energies_minus, vectors_minus = np.linalg.eigh(h_minus)
    energies_ground, vectors_ground = np.linalg.eigh(h_ground)
    energies_plus, vectors_plus = np.linalg.eigh(h_plus)
    scale = max(1.0, float(np.max(np.abs(energies_ground), initial=0.0)))
    if (energies_ground.size > 1
            and energies_ground[1] - energies_ground[0] <= 1e-10 * scale):
        raise ValueError("the canonical reference state must be unique")
    ground_energy = float(energies_ground[0])
    ground = vectors_ground[:, [0]]
    lookup_minus = {int(state): i for i, state in enumerate(basis_minus)}
    lookup_plus = {int(state): i for i, state in enumerate(basis_plus)}
    additions, removals = [], []
    for auxiliary in (False, True):
        for mode in range(2):
            additions.append(_efi_apply_impurity(
                basis_ground, lookup_plus, ground, mode, True,
                float(np.real(interaction_u)), auxiliary
            )[:, 0])
            removals.append(_efi_apply_impurity(
                basis_ground, lookup_minus, ground, mode, False,
                float(np.real(interaction_u)), auxiliary
            )[:, 0])

    raw_poles, raw_residues = [], []
    for energy, state in zip(energies_plus, vectors_plus.T):
        row = np.asarray([np.vdot(state, vector) for vector in additions])
        raw_poles.append(float(energy - ground_energy))
        raw_residues.append(np.outer(row.conj(), row))
    for energy, state in zip(energies_minus, vectors_minus.T):
        amplitudes = np.asarray([
            np.vdot(state, vector) for vector in removals
        ])
        row = amplitudes.conj()
        raw_poles.append(float(ground_energy - energy))
        raw_residues.append(np.outer(row.conj(), row))
    output_poles, output_residues = _aml_cluster_spectrum(
        raw_poles, raw_residues
    )
    return ground_energy, output_poles, output_residues

import numpy as np


def certify_regularized_matrix_response(
    augmented_poles: "np.ndarray",
    augmented_residues: "np.ndarray",
    regulator: float,
) -> "tuple[np.ndarray, np.ndarray]":
    if (not np.isscalar(regulator) or not np.isfinite(regulator)
            or abs(complex(regulator).imag) > 0.0
            or float(np.real(regulator)) <= 0.0):
        raise ValueError("regulator must be positive, finite, and real")
    static, poles, residues = recover_causal_matrix_self_energy(
        augmented_poles, augmented_residues
    )
    dimension = int(static.shape[0])
    regularization = float(np.real(regulator))
    inverse_response = np.eye(dimension, dtype=np.complex128)
    for pole, residue in zip(poles, residues):
        inverse_response += residue / (pole * pole + regularization * regularization)
    response = _mse_h(np.linalg.inv(_mse_h(inverse_response)))
    if (not np.all(np.isfinite(response))
            or not np.allclose(response, response.conj().T,
                               rtol=0.0, atol=3e-11)):
        raise ValueError("the regularized response is not finite and Hermitian")
    response_eigenvalues = np.linalg.eigvalsh(response)
    if (response_eigenvalues[0] < -3e-10
            or response_eigenvalues[-1] > 1.0 + 3e-10):
        raise ValueError("the regularized response violates causal eigenvalue bounds")
    if residues.size:
        moment0 = np.sum(residues, axis=0)
        moment1 = np.sum(poles[:, None, None] * residues, axis=0)
        minimum_residue = min(
            float(np.min(np.linalg.eigvalsh(_mse_h(residue))))
            for residue in residues
        )
        minimum_residue = max(0.0, minimum_residue)
    else:
        moment0 = np.zeros((dimension, dimension), dtype=np.complex128)
        moment1 = np.zeros((dimension, dimension), dtype=np.complex128)
        minimum_residue = 0.0
    certificate = np.asarray([
        float(np.trace(moment0).real),
        float(np.trace(moment1).real),
        minimum_residue,
        float(response_eigenvalues[0]),
        float(response_eigenvalues[-1]),
    ])
    return response, certificate

import numpy as np


def bounded_matrix_two_solve_weight(
    impurity_energy: "np.ndarray",
    bath_energies: "np.ndarray",
    bath_couplings: "np.ndarray",
    interaction_u: float,
    first_particle_number: int,
    second_particle_number: int,
    regulator: float,
    probe: "np.ndarray",
) -> float:
    probe = np.asarray(probe, dtype=np.complex128)
    if probe.shape != (2,) or not np.all(np.isfinite(probe)):
        raise ValueError("probe must be a finite vector of shape (2,)")
    norm = float(np.vdot(probe, probe).real)
    if not np.isclose(norm, 1.0, rtol=0.0, atol=2e-12):
        raise ValueError("probe must be normalized")
    hamiltonian, interaction, annihilators, occupations = (
        build_noncommuting_impurity_model(
            impurity_energy, bath_energies, bath_couplings, interaction_u
        )
    )
    _, augmented_poles, augmented_residues = (
        compute_augmented_matrix_lehmann_spectrum(
            hamiltonian,
            interaction,
            annihilators,
            occupations,
            first_particle_number,
        )
    )
    static, self_energy_poles, self_energy_residues = (
        recover_causal_matrix_self_energy(
            augmented_poles, augmented_residues
        )
    )
    feedback_poles, feedback_residues = (
        update_exact_matrix_hybridization(
            bath_energies,
            bath_couplings,
            static,
            self_energy_poles,
            self_energy_residues,
        )
    )
    _, second_poles, second_residues = solve_exact_feedback_impurity(
        impurity_energy,
        feedback_poles,
        feedback_residues,
        interaction_u,
        second_particle_number,
    )
    response, _ = certify_regularized_matrix_response(
        second_poles, second_residues, regulator
    )
    projected = np.vdot(probe, response @ probe)
    if abs(float(np.imag(projected))) > 2e-10:
        raise ValueError("the projected response is not real")
    return float(np.round(float(np.real(projected)), 10))
SCICODE_GOLD_EOF
