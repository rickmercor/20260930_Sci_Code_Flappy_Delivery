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


def condense_volumetric_core(
    material_stiffness: np.ndarray,
    pressure_coupling: np.ndarray,
    pressure_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference factor-preserving pressure condensation."""
    stiffness = np.asarray(material_stiffness, dtype=float)
    coupling = np.asarray(pressure_coupling, dtype=float)
    block = np.asarray(pressure_block, dtype=float)
    if (
        stiffness.ndim != 2
        or stiffness.shape[0] != stiffness.shape[1]
        or stiffness.shape[0] == 0
    ):
        raise ValueError("material_stiffness must be a nonempty square matrix")
    n_dof = stiffness.shape[0]
    if coupling.ndim != 2 or coupling.shape[1] != n_dof:
        raise ValueError("pressure_coupling must have shape (m, n)")
    n_pressure = coupling.shape[0]
    if block.ndim != 2 or block.shape != (n_pressure, n_pressure):
        raise ValueError("pressure_block must have shape (m, m)")
    if not all(np.all(np.isfinite(x)) for x in (stiffness, coupling, block)):
        raise ValueError("all inputs must be finite")
    if not np.allclose(stiffness, stiffness.T, rtol=0.0, atol=1e-12):
        raise ValueError("material_stiffness must be symmetric")
    if not np.allclose(block, block.T, rtol=0.0, atol=1e-12):
        raise ValueError("pressure_block must be symmetric")
    try:
        factor = np.linalg.cholesky(block)
    except np.linalg.LinAlgError as exc:
        raise ValueError("pressure_block must be positive definite") from exc

    if n_pressure == 0:
        scaled_rows = np.zeros((0, n_dof), dtype=float)
        volumetric = np.zeros((n_dof, n_dof), dtype=float)
    else:
        scaled_rows = np.linalg.solve(factor, coupling)
        volumetric = scaled_rows.T @ scaled_rows
        volumetric = 0.5 * (volumetric + volumetric.T)
    core = stiffness + volumetric
    core = 0.5 * (core + core.T)
    try:
        np.linalg.cholesky(core)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the condensed core must be positive definite") from exc
    return scaled_rows, volumetric, core

import numpy as np


def assemble_contact_operators(
    condensed_core: np.ndarray,
    interaction_rows: np.ndarray,
    correction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference four-way contact assembly."""
    core = np.asarray(condensed_core, dtype=float)
    rows = np.asarray(interaction_rows, dtype=float)
    extra = np.asarray(correction, dtype=float)
    if core.ndim != 2 or core.shape[0] != core.shape[1] or core.shape[0] == 0:
        raise ValueError("condensed_core must be a nonempty square matrix")
    n_dof = core.shape[0]
    if rows.ndim != 2 or rows.shape[1] != n_dof:
        raise ValueError("interaction_rows must have shape (m, n)")
    if extra.ndim != 2 or extra.shape != (n_dof, n_dof):
        raise ValueError("correction must have shape (n, n)")
    if not all(np.all(np.isfinite(x)) for x in (core, rows, extra)):
        raise ValueError("all inputs must be finite")
    if not np.allclose(core, core.T, rtol=0.0, atol=1e-12):
        raise ValueError("condensed_core must be symmetric")
    try:
        np.linalg.cholesky(core)
    except np.linalg.LinAlgError as exc:
        raise ValueError("condensed_core must be positive definite") from exc

    interaction_energy = rows.T @ rows
    interaction_energy = 0.5 * (interaction_energy + interaction_energy.T)
    reference = core + interaction_energy
    reference = 0.5 * (reference + reference.T)
    full_operator = reference + extra
    symmetric_part = reference + 0.5 * (extra + extra.T)
    return interaction_energy, reference, full_operator, symmetric_part

import numpy as np


def build_response_gramian(
    condensed_core: np.ndarray,
    interaction_rows: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Reference metric response and variational dominant selection."""
    core = np.asarray(condensed_core, dtype=float)
    rows = np.asarray(interaction_rows, dtype=float)
    if core.ndim != 2 or core.shape[0] != core.shape[1] or core.shape[0] == 0:
        raise ValueError("condensed_core must be a nonempty square matrix")
    n_dof = core.shape[0]
    if rows.ndim != 2 or rows.shape[1] != n_dof:
        raise ValueError("interaction_rows must have shape (m, n)")
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)):
        raise ValueError("rank must be an integer")  # noqa: TRY004 - API contract
    rank = int(rank)
    n_rows = rows.shape[0]
    if rank < 0 or rank > n_rows:
        raise ValueError("rank must lie in [0, m]")
    if not np.all(np.isfinite(core)) or not np.all(np.isfinite(rows)):
        raise ValueError("all inputs must be finite")
    if not np.allclose(core, core.T, rtol=0.0, atol=1e-12):
        raise ValueError("condensed_core must be symmetric")
    try:
        np.linalg.cholesky(core)
    except np.linalg.LinAlgError as exc:
        raise ValueError("condensed_core must be positive definite") from exc
    if n_rows == 0:
        empty_square = np.zeros((0, 0), dtype=float)
        return empty_square, np.zeros(0), empty_square, 0.0

    response = rows @ np.linalg.solve(core, rows.T)
    response = 0.5 * (response + response.T)
    eigenvalues, eigenvectors = np.linalg.eigh(response)
    if eigenvalues[0] < -1e-10 * max(1.0, float(np.max(np.abs(eigenvalues)))):
        raise ValueError("the response must be positive semidefinite")
    order = np.argsort(eigenvalues)[::-1]
    levels = np.maximum(eigenvalues[order], 0.0)
    vectors = eigenvectors[:, order]
    scale = max(1.0, float(levels[0]))
    positive_tolerance = max(n_rows, n_dof, 1) * np.finfo(float).eps * scale
    numerical_rank = int(np.count_nonzero(levels > positive_tolerance))
    if rank > numerical_rank:
        raise ValueError("rank exceeds the numerical rank of the response")
    gap_tolerance = 1e-10 * scale
    if (
        0 < rank < n_rows
        and abs(float(levels[rank - 1] - levels[rank])) <= gap_tolerance
    ):
        raise ValueError("the maximizing projector is not unique")
    retained = vectors[:, :rank]
    projector = retained @ retained.T
    projector = 0.5 * (projector + projector.T)
    cutoff_ratio = float(levels[rank] / levels[rank - 1]) if 0 < rank < n_rows else 0.0
    return response, levels, projector, cutoff_ratio

import numpy as np


def certify_preconditioned_spectrum(
    condensed_core: np.ndarray,
    interaction_rows: np.ndarray,
    response_levels: np.ndarray,
    retained_projector: np.ndarray | None,
    retained_factor: np.ndarray | None = None,
    reference_core: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, np.ndarray]:
    """Reference theorem-level spectral certification."""
    core = np.asarray(condensed_core, dtype=float)
    rows = np.asarray(interaction_rows, dtype=float)
    levels = np.asarray(response_levels, dtype=float)
    reference = (
        core if reference_core is None else np.asarray(reference_core, dtype=float)
    )
    supplied_projector = retained_projector
    supplied_factor = retained_factor
    if core.ndim != 2 or core.shape[0] != core.shape[1] or core.shape[0] == 0:
        raise ValueError("condensed_core must be a nonempty square matrix")
    n_dof = core.shape[0]
    if rows.ndim != 2 or rows.shape[1] != n_dof:
        raise ValueError("interaction_rows must have shape (m, n)")
    n_rows = rows.shape[0]
    if levels.shape != (n_rows,):
        raise ValueError("response_levels must have shape (m,)")
    if reference.shape != (n_dof, n_dof):
        raise ValueError("reference_core must have shape (n, n)")
    if (supplied_projector is None) == (supplied_factor is None):
        raise ValueError("supply exactly one retained-space representation")
    if not all(np.all(np.isfinite(x)) for x in (core, rows, levels, reference)):
        raise ValueError("all inputs must be finite")
    if not np.allclose(core, core.T, rtol=0.0, atol=1e-12):
        raise ValueError("condensed_core must be symmetric")
    try:
        core_factor = np.linalg.cholesky(core)
    except np.linalg.LinAlgError as exc:
        raise ValueError("condensed_core must be positive definite") from exc
    if not np.allclose(reference, reference.T, rtol=0.0, atol=1e-12):
        raise ValueError("reference_core must be symmetric")
    try:
        np.linalg.cholesky(reference)
    except np.linalg.LinAlgError as exc:
        raise ValueError("reference_core must be positive definite") from exc
    if np.any(levels < -1e-12) or np.any(np.diff(levels) > 1e-12):
        raise ValueError("response_levels must be nonnegative and nonincreasing")
    if supplied_factor is None:
        projector = np.asarray(supplied_projector, dtype=float)
        if projector.shape != (n_rows, n_rows):
            raise ValueError("retained_projector must have shape (m, m)")
        if not np.all(np.isfinite(projector)):
            raise ValueError("all inputs must be finite")
        if not np.allclose(projector, projector.T, rtol=1e-10, atol=1e-12):
            raise ValueError("retained_projector must be symmetric")
        if not np.allclose(projector @ projector, projector, rtol=1e-9, atol=1e-11):
            raise ValueError("retained_projector must be idempotent")
    else:
        spanning = np.asarray(supplied_factor, dtype=float)
        if spanning.ndim != 2 or spanning.shape[0] != n_rows:
            raise ValueError("retained_factor must have shape (m, k)")
        if not np.all(np.isfinite(spanning)):
            raise ValueError("all inputs must be finite")
        if n_rows == 0 or spanning.shape[1] == 0:
            retained = np.zeros((n_rows, 0), dtype=float)
        else:
            left, singular_values, _ = np.linalg.svd(spanning, full_matrices=False)
            largest = float(singular_values[0]) if singular_values.size else 0.0
            rank_tolerance = (
                max(spanning.shape[0], spanning.shape[1], 1)
                * np.finfo(float).eps
                * largest
            )
            retained = left[:, singular_values > rank_tolerance]
        projector = retained @ retained.T

    response = rows @ np.linalg.solve(core, rows.T)
    response = 0.5 * (response + response.T)
    measured_levels = np.linalg.eigvalsh(response)[::-1]
    measured_levels = np.maximum(measured_levels, 0.0)
    if not np.allclose(levels, measured_levels, rtol=1e-9, atol=1e-11):
        raise ValueError("response_levels do not match the metric response")
    if n_rows > 0 and not np.allclose(
        response @ projector, projector @ response, rtol=1e-8, atol=1e-10
    ):
        raise ValueError("retained_projector is not response-invariant")

    # The surviving response is the one carried by the orthogonal complement of
    # the retained space, which is fixed by the projector rather than by the
    # position of a level in the sorted list.
    if n_rows > 0:
        weights, directions = np.linalg.eigh(projector)
        complement = directions[:, weights < 0.5]
        if complement.shape[1] > 0:
            omitted_all = np.linalg.eigvalsh(complement.T @ response @ complement)
            omitted_all = np.maximum(omitted_all[::-1], 0.0)
        else:
            omitted_all = np.zeros(0, dtype=float)
    else:
        omitted_all = np.zeros(0, dtype=float)

    preconditioner = core + rows.T @ projector @ rows
    surrogate_reference = core + rows.T @ rows
    preconditioner = 0.5 * (preconditioner + preconditioner.T)
    surrogate_reference = 0.5 * (surrogate_reference + surrogate_reference.T)
    try:
        factor = np.linalg.cholesky(preconditioner)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "the reduced preconditioner must be positive definite"
        ) from exc
    left = np.linalg.solve(factor, surrogate_reference)
    whitened = np.linalg.solve(factor, left.T).T
    whitened = 0.5 * (whitened + whitened.T)
    measured_spectrum = np.linalg.eigvalsh(whitened)

    scale = max(1.0, float(levels[0])) if n_rows else 1.0
    positive_tolerance = max(n_dof, n_rows, 1) * np.finfo(float).eps * scale
    omitted = omitted_all[omitted_all > positive_tolerance]
    if omitted.size > n_dof:
        raise ValueError("more omitted levels than displacement dimensions")
    predicted_spectrum = np.concatenate(
        [np.ones(n_dof - omitted.size, dtype=float), 1.0 + omitted]
    )
    predicted_spectrum.sort()
    if not np.allclose(measured_spectrum, predicted_spectrum, rtol=1e-8, atol=1e-10):
        raise ValueError("measured generalized spectrum violates the prediction")
    omitted_level = float(omitted[0]) if omitted.size else 0.0
    certificate = omitted_level / (1.0 + omitted_level)
    posterior_bound = 1.0 + omitted_level
    smallest = float(measured_spectrum[0])
    if smallest <= 0.0:
        raise ValueError("the preconditioned pencil must be positive definite")
    condition_number = float(measured_spectrum[-1] / smallest)
    if condition_number > posterior_bound + 1e-8:
        raise ValueError("the measured conditioning exceeds the posterior bound")

    # Appendix C: certify the unchanged reference operator when response
    # selection and Woodbury application use a fixed approximate SPD core.
    scaled_reference = np.linalg.solve(core_factor, reference)
    scaled_reference = np.linalg.solve(core_factor, scaled_reference.T).T
    scaled_reference = 0.5 * (scaled_reference + scaled_reference.T)
    equivalence = np.linalg.eigvalsh(scaled_reference)
    c1 = float(equivalence[0])
    c2 = float(equivalence[-1])

    complement_projector = np.eye(n_rows, dtype=float) - projector
    omitted_matrix = rows.T @ (complement_projector @ rows)
    omitted_matrix = 0.5 * (omitted_matrix + omitted_matrix.T)
    scaled_omitted = np.linalg.solve(core_factor, omitted_matrix)
    scaled_omitted = np.linalg.solve(core_factor, scaled_omitted.T).T
    scaled_omitted = 0.5 * (scaled_omitted + scaled_omitted.T)
    delta = max(0.0, float(np.linalg.eigvalsh(scaled_omitted)[-1]))
    robust_bound = float(max(c2 + delta, 1.0) / min(c1, 1.0))

    true_reference = reference + rows.T @ rows
    true_reference = 0.5 * (true_reference + true_reference.T)
    robust_left = np.linalg.solve(factor, true_reference)
    robust_whitened = np.linalg.solve(factor, robust_left.T).T
    robust_whitened = 0.5 * (robust_whitened + robust_whitened.T)
    robust_spectrum = np.linalg.eigvalsh(robust_whitened)
    if robust_spectrum[0] <= 0.0:
        raise ValueError("the true preconditioned pencil must be positive definite")
    robust_condition = float(robust_spectrum[-1] / robust_spectrum[0])
    if robust_condition > robust_bound + 1e-8:
        raise ValueError("the true conditioning exceeds the Appendix-C bound")
    robust_certificate = np.array(
        [c1, c2, delta, robust_bound, robust_condition], dtype=float
    )
    return (
        measured_spectrum,
        predicted_spectrum,
        float(certificate),
        condition_number,
        float(omitted_level),
        float(posterior_bound),
        robust_certificate,
    )

import numpy as np


def build_reduced_interaction_state(
    condensed_core: np.ndarray,
    interaction_rows: np.ndarray,
    retained_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference construction of the source's compact retained state."""
    core = np.asarray(condensed_core, dtype=float)
    rows = np.asarray(interaction_rows, dtype=float)
    basis = np.asarray(retained_basis, dtype=float)
    if core.ndim != 2 or core.shape[0] != core.shape[1] or core.shape[0] == 0:
        raise ValueError("condensed_core must be a nonempty square matrix")
    n_dof = core.shape[0]
    if rows.ndim != 2 or rows.shape[1] != n_dof:
        raise ValueError("interaction_rows must have shape (m, n)")
    n_rows = rows.shape[0]
    if basis.ndim != 2 or basis.shape[0] != n_rows or basis.shape[1] > n_rows:
        raise ValueError("retained_basis must have shape (m, r), 0 <= r <= m")
    if not all(np.all(np.isfinite(x)) for x in (core, rows, basis)):
        raise ValueError("all inputs must be finite")
    if not np.allclose(core, core.T, rtol=0.0, atol=1e-12):
        raise ValueError("condensed_core must be symmetric")
    try:
        np.linalg.cholesky(core)
    except np.linalg.LinAlgError as exc:
        raise ValueError("condensed_core must be positive definite") from exc

    rank = basis.shape[1]
    if rank and not np.allclose(basis.T @ basis, np.eye(rank), rtol=1e-10, atol=1e-12):
        raise ValueError("retained_basis must have orthonormal columns")

    retained_rows = basis.T @ rows
    if rank == 0:
        return (
            0.5 * (core + core.T),
            np.zeros((0, n_dof), dtype=float),
            np.zeros((n_dof, 0), dtype=float),
            np.zeros((0, 0), dtype=float),
        )

    core_responses = np.linalg.solve(core, retained_rows.T)
    reduced_matrix = np.eye(rank, dtype=float) + retained_rows @ core_responses
    reduced_matrix = 0.5 * (reduced_matrix + reduced_matrix.T)
    try:
        np.linalg.cholesky(reduced_matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the reduced matrix must be positive definite") from exc

    preconditioner = core + retained_rows.T @ retained_rows
    preconditioner = 0.5 * (preconditioner + preconditioner.T)
    try:
        np.linalg.cholesky(preconditioner)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the preconditioner must be positive definite") from exc
    return preconditioner, retained_rows, core_responses, reduced_matrix

import numpy as np


def apply_reduced_inverse(
    base_operator: np.ndarray,
    retained_rows: np.ndarray,
    core_responses: np.ndarray,
    reduced_matrix: np.ndarray,
    right_hand_sides: np.ndarray,
    transposed: bool,
) -> np.ndarray:
    """Reference compact forward or adjoint Woodbury action."""
    base = np.asarray(base_operator, dtype=float)
    rows = np.asarray(retained_rows, dtype=float)
    responses = np.asarray(core_responses, dtype=float)
    reduced = np.asarray(reduced_matrix, dtype=float)
    rhs = np.asarray(right_hand_sides, dtype=float)
    if not isinstance(transposed, (bool, np.bool_)):
        raise ValueError("transposed must be a bool")  # noqa: TRY004
    if base.ndim != 2 or base.shape[0] != base.shape[1] or base.shape[0] == 0:
        raise ValueError("base_operator must be a nonempty square matrix")
    n_dof = base.shape[0]
    if rows.ndim != 2 or rows.shape[1] != n_dof:
        raise ValueError("retained_rows must have shape (r, n)")
    rank = rows.shape[0]
    if responses.shape != (n_dof, rank):
        raise ValueError("core_responses must have shape (n, r)")
    if reduced.shape != (rank, rank):
        raise ValueError("reduced_matrix must have shape (r, r)")
    if rhs.ndim == 1:
        if rhs.shape != (n_dof,):
            raise ValueError("a vector right side must have shape (n,)")
    elif rhs.ndim == 2:
        if rhs.shape[0] != n_dof or rhs.shape[1] == 0:
            raise ValueError("a block right side must have shape (n, k), k >= 1")
    else:
        raise ValueError("right_hand_sides must be a vector or column block")
    if not all(np.all(np.isfinite(x)) for x in (base, rows, responses, reduced, rhs)):
        raise ValueError("all inputs must be finite")
    if not np.allclose(base @ responses, rows.T, rtol=1e-10, atol=1e-12):
        raise ValueError("core_responses do not solve the retained transpose loads")
    expected_reduced = np.eye(rank, dtype=float) + rows @ responses
    if not np.allclose(reduced, expected_reduced, rtol=1e-10, atol=1e-12):
        raise ValueError("reduced_matrix is inconsistent with the compact state")

    try:
        if bool(transposed):
            corrected = rhs - rows.T @ np.linalg.solve(reduced.T, responses.T @ rhs)
            result = np.linalg.solve(base.T, corrected)
        else:
            core_action = np.linalg.solve(base, rhs)
            result = core_action - responses @ np.linalg.solve(
                reduced, rows @ core_action
            )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "base_operator and reduced_matrix must be nonsingular"
        ) from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("the inverse action must be finite")
    return result

import numpy as np


def certify_residual_bounds(
    full_operator: np.ndarray,
    right_inverse: np.ndarray,
    right_hand_side: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Reference one-step field-of-values certificate."""
    operator = np.asarray(full_operator, dtype=float)
    inverse = np.asarray(right_inverse, dtype=float)
    rhs = np.asarray(right_hand_side, dtype=float)
    if (
        operator.ndim != 2
        or operator.shape[0] != operator.shape[1]
        or operator.shape[0] == 0
    ):
        raise ValueError("full_operator must be a nonempty square matrix")
    n_dof = operator.shape[0]
    if inverse.shape != (n_dof, n_dof):
        raise ValueError("right_inverse must have shape (n, n)")
    if rhs.shape != (n_dof,):
        raise ValueError("right_hand_side must have shape (n,)")
    if not all(np.all(np.isfinite(x)) for x in (operator, inverse, rhs)):
        raise ValueError("all inputs must be finite")
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        raise ValueError("right_hand_side must have nonzero norm")

    preconditioned = operator @ inverse
    symmetric = 0.5 * (preconditioned + preconditioned.T)
    symmetric_low = float(np.linalg.eigvalsh(symmetric)[0])
    spectral_norm = float(np.linalg.svd(preconditioned, compute_uv=False)[0])
    if spectral_norm <= 0.0:
        raise ValueError("the preconditioned operator must be nonzero")
    clamped = max(symmetric_low, 0.0)
    estimate = float(np.sqrt(max(0.0, 1.0 - (clamped / spectral_norm) ** 2)))

    image = preconditioned @ rhs
    image_norm = float(np.linalg.norm(image))
    zero_tolerance = np.finfo(float).eps * max(1.0, spectral_norm) * rhs_norm
    if image_norm <= zero_tolerance:
        coefficient = 0.0
        reduction = 1.0
    else:
        coefficient = float(rhs @ image) / float(image @ image)
        reduction = float(np.linalg.norm(rhs - coefficient * image) / rhs_norm)
    if not all(
        np.isfinite(x)
        for x in (reduction, coefficient, estimate, symmetric_low, spectral_norm)
    ):
        raise ValueError("the certificate quantities must be finite")
    if reduction > estimate + 1e-9:
        raise ValueError("the exact reduction violates the field-of-values estimate")
    return reduction, coefficient, estimate, symmetric_low, spectral_norm

import numpy as np


def compute_directional_residual(
    full_operator: np.ndarray,
    right_hand_side: np.ndarray,
    base_operator: np.ndarray,
    retained_row_sequence: tuple[np.ndarray, ...],
    core_response_sequence: tuple[np.ndarray, ...],
    reduced_matrix_sequence: tuple[np.ndarray, ...],
    max_iterations: int,
    cycle_lengths: np.ndarray | None = None,
    initial_guess: np.ndarray | None = None,
    lower_retained_rows: np.ndarray | None = None,
    lower_core_responses: np.ndarray | None = None,
    lower_reduced_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Reference restarted, optionally nested compact-state FGMRES."""
    operator = np.asarray(full_operator, dtype=float)
    rhs = np.asarray(right_hand_side, dtype=float)
    base = np.asarray(base_operator, dtype=float)
    if (
        operator.ndim != 2
        or operator.shape[0] != operator.shape[1]
        or operator.shape[0] == 0
    ):
        raise ValueError("full_operator must be a nonempty square matrix")
    n_dof = operator.shape[0]
    if base.shape != (n_dof, n_dof):
        raise ValueError("base_operator must have shape (n, n)")
    if rhs.shape != (n_dof,):
        raise ValueError("right_hand_side must have shape (n,)")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(
        max_iterations, (int, np.integer)
    ):
        raise ValueError("max_iterations must be an integer")  # noqa: TRY004
    max_iterations = int(max_iterations)
    if max_iterations < 1 or max_iterations > n_dof:
        raise ValueError("max_iterations must lie in [1, n]")
    if cycle_lengths is None:
        cycles = np.array([max_iterations], dtype=int)
    else:
        raw_cycles = np.asarray(cycle_lengths)
        if (
            raw_cycles.ndim != 1
            or raw_cycles.size == 0
            or raw_cycles.dtype.kind not in {"i", "u"}
            or raw_cycles.dtype.kind == "b"
        ):
            raise ValueError("cycle_lengths must be a positive-integer vector")
        cycles = np.asarray(raw_cycles, dtype=int)
        if np.any(cycles <= 0) or int(np.sum(cycles)) != max_iterations:
            raise ValueError("cycle_lengths must be positive and sum to k")
    if not all(
        isinstance(sequence, (tuple, list)) and len(sequence) == max_iterations
        for sequence in (
            retained_row_sequence,
            core_response_sequence,
            reduced_matrix_sequence,
        )
    ):
        raise ValueError("each compact-state sequence must have length k")
    if not all(np.all(np.isfinite(x)) for x in (operator, rhs, base)):
        raise ValueError("all inputs must be finite")
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        raise ValueError("right_hand_side must have nonzero norm")
    try:
        np.linalg.solve(base, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("base_operator must be nonsingular") from exc

    lower_flags = (
        lower_retained_rows is not None,
        lower_core_responses is not None,
        lower_reduced_matrix is not None,
    )
    if any(lower_flags) and not all(lower_flags):
        raise ValueError("supply all three lower compact arrays or none")
    if all(lower_flags):
        lower_rows = np.asarray(lower_retained_rows, dtype=float)
        lower_responses = np.asarray(lower_core_responses, dtype=float)
        lower_reduced = np.asarray(lower_reduced_matrix, dtype=float)
        if lower_rows.ndim != 2 or lower_rows.shape[1] != n_dof:
            raise ValueError("lower_retained_rows must have shape (p, n)")
        lower_rank = lower_rows.shape[0]
        if lower_responses.shape != (n_dof, lower_rank):
            raise ValueError("lower_core_responses must have shape (n, p)")
        if lower_reduced.shape != (lower_rank, lower_rank):
            raise ValueError("lower_reduced_matrix must have shape (p, p)")
        if not all(
            np.all(np.isfinite(x)) for x in (lower_rows, lower_responses, lower_reduced)
        ):
            raise ValueError("the lower compact state must be finite")
        if not np.allclose(
            base @ lower_responses,
            lower_rows.T,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("the lower core responses are inconsistent")
        expected_lower = np.eye(lower_rank) + lower_rows @ lower_responses
        if not np.allclose(lower_reduced, expected_lower, rtol=1e-10, atol=1e-12):
            raise ValueError("the lower reduced matrix is inconsistent")
        if lower_rank:
            try:
                np.linalg.solve(lower_reduced, np.ones(lower_rank))
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "the lower reduced matrix must be nonsingular"
                ) from exc
        effective_core = base + lower_rows.T @ lower_rows
    else:
        lower_rows = np.zeros((0, n_dof), dtype=float)
        lower_responses = np.zeros((n_dof, 0), dtype=float)
        lower_reduced = np.zeros((0, 0), dtype=float)
        effective_core = base

    def apply_effective_core_inverse(vector: np.ndarray) -> np.ndarray:
        """Apply the direct or nested lower-level inverse."""
        action = np.linalg.solve(base, vector)
        if lower_rows.shape[0]:
            action = action - lower_responses @ np.linalg.solve(
                lower_reduced, lower_rows @ action
            )
        return action

    compact_states = []
    for column in range(max_iterations):
        rows = np.asarray(retained_row_sequence[column], dtype=float)
        responses = np.asarray(core_response_sequence[column], dtype=float)
        reduced = np.asarray(reduced_matrix_sequence[column], dtype=float)
        if rows.ndim != 2 or rows.shape[1] != n_dof:
            raise ValueError("each retained row matrix must have shape (r_j, n)")
        rank = rows.shape[0]
        if responses.shape != (n_dof, rank):
            raise ValueError("each core response block must have shape (n, r_j)")
        if reduced.shape != (rank, rank):
            raise ValueError("each reduced matrix must have shape (r_j, r_j)")
        if not all(np.all(np.isfinite(x)) for x in (rows, responses, reduced)):
            raise ValueError("all compact states must be finite")
        if not np.allclose(effective_core @ responses, rows.T, rtol=1e-10, atol=1e-12):
            raise ValueError("a core response block is inconsistent")
        expected_reduced = np.eye(rank, dtype=float) + rows @ responses
        if not np.allclose(reduced, expected_reduced, rtol=1e-10, atol=1e-12):
            raise ValueError("a reduced matrix is inconsistent")
        if rank:
            try:
                np.linalg.solve(reduced, np.ones(rank, dtype=float))
            except np.linalg.LinAlgError as exc:
                raise ValueError("each reduced matrix must be nonsingular") from exc
        compact_states.append((rows, responses, reduced))

    if initial_guess is None:
        iterate = np.zeros(n_dof, dtype=float)
    else:
        iterate = np.asarray(initial_guess, dtype=float)
        if iterate.shape != (n_dof,) or not np.all(np.isfinite(iterate)):
            raise ValueError("initial_guess must be a finite vector of shape (n,)")
        iterate = iterate.copy()

    history = np.zeros(max_iterations + 1, dtype=float)
    coefficients = np.zeros(max_iterations, dtype=float)
    initial_residual = rhs - operator @ iterate
    history[0] = float(np.linalg.norm(initial_residual) / rhs_norm)
    used = 0
    terminate = False
    operator_norm = float(np.linalg.norm(operator, ord=2))

    for cycle_length in cycles:
        cycle_start = iterate.copy()
        cycle_residual = rhs - operator @ cycle_start
        cycle_norm = float(np.linalg.norm(cycle_residual))
        zero_scale = max(
            1.0,
            rhs_norm,
            operator_norm * float(np.linalg.norm(cycle_start)),
        )
        if cycle_norm <= 100.0 * np.finfo(float).eps * zero_scale:
            terminate = True
            break

        basis = np.zeros((n_dof, int(cycle_length) + 1), dtype=float)
        preconditioned = np.zeros((n_dof, int(cycle_length)), dtype=float)
        basis[:, 0] = cycle_residual / cycle_norm
        cycle_offset = used

        for column in range(int(cycle_length)):
            rows, responses, reduced = compact_states[used]
            core_action = apply_effective_core_inverse(basis[:, column])
            if rows.shape[0]:
                correction = responses @ np.linalg.solve(reduced, rows @ core_action)
            else:
                correction = np.zeros(n_dof, dtype=float)
            direction = core_action - correction
            direction_scale = max(
                1.0,
                float(np.linalg.norm(core_action)),
                float(np.linalg.norm(correction)),
            )
            if (
                float(np.linalg.norm(direction))
                <= np.finfo(float).eps * direction_scale
            ):
                raise ValueError("a compact inverse action produced a zero direction")
            preconditioned[:, column] = direction

            raw_image = operator @ direction
            work = raw_image.copy()
            active_basis = basis[:, : column + 1]
            for _ in range(2):
                work -= active_basis @ (active_basis.T @ work)
            next_norm = float(np.linalg.norm(work))
            breakdown = next_norm <= 100.0 * np.finfo(float).eps * max(
                1.0, float(np.linalg.norm(raw_image))
            )
            if not breakdown:
                basis[:, column + 1] = work / next_norm

            images = operator @ preconditioned[:, : column + 1]
            local_coefficients = np.linalg.lstsq(images, cycle_residual, rcond=None)[0]
            coefficients[cycle_offset : cycle_offset + column + 1] = local_coefficients
            iterate = cycle_start + preconditioned[:, : column + 1] @ local_coefficients
            residual = float(np.linalg.norm(rhs - operator @ iterate) / rhs_norm)
            if not np.isfinite(residual):
                raise ValueError("the fresh residual must be finite")
            used += 1
            history[used] = residual
            if breakdown:
                terminate = True
                break
        if terminate:
            break

    history[used + 1 :] = history[used]

    if not all(np.all(np.isfinite(x)) for x in (iterate, history, coefficients)):
        raise ValueError("the cycle outputs must be finite")
    return iterate, history, coefficients, used

import numpy as np


def compute_contact_residual(
    material_stiffness: np.ndarray,
    pressure_coupling: np.ndarray,
    pressure_block: np.ndarray,
    interaction_rows: np.ndarray,
    correction: np.ndarray,
    right_hand_side: np.ndarray,
    condition_budget: float | np.ndarray,
    pressure_condition_budget: float | None = None,
    cycle_lengths: np.ndarray | None = None,
    initial_guess: np.ndarray | None = None,
) -> float:
    """Chain all preceding oracle contracts and return the certified residual."""
    raw_budget = np.asarray(condition_budget)
    if raw_budget.dtype.kind not in {"i", "u", "f"} or raw_budget.ndim > 1:
        raise ValueError("condition_budget must be a scalar or real vector")
    try:
        if raw_budget.ndim == 0:
            budgets = np.array([float(raw_budget)], dtype=float)
        else:
            budgets = np.asarray(condition_budget, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("condition_budget must be real") from exc
    if budgets.size == 0 or not np.all(np.isfinite(budgets)):
        raise ValueError("condition_budget must be nonempty and finite")
    if np.any(budgets < 1.0):
        raise ValueError("every condition budget must be at least one")

    if pressure_condition_budget is None:
        pressure_budget = None
    else:
        if isinstance(pressure_condition_budget, (bool, np.bool_)):
            raise ValueError("pressure_condition_budget must be a real scalar")
        raw_pressure_budget = np.asarray(pressure_condition_budget)
        if raw_pressure_budget.ndim != 0 or raw_pressure_budget.dtype.kind not in {
            "i",
            "u",
            "f",
        }:
            raise ValueError("pressure_condition_budget must be a real scalar")
        pressure_budget = float(raw_pressure_budget)
        if not np.isfinite(pressure_budget) or pressure_budget < 1.0:
            raise ValueError(
                "pressure_condition_budget must be finite and at least one"
            )

    scaled_rows, volumetric, core = condense_volumetric_core(
        material_stiffness,
        pressure_coupling,
        pressure_block,
    )
    _, _, full_operator, symmetric_part = assemble_contact_operators(
        core,
        interaction_rows,
        correction,
    )
    n_dof = core.shape[0]
    if budgets.size > n_dof:
        raise ValueError("the budget schedule cannot exceed the displacement dimension")
    rows = np.asarray(interaction_rows, dtype=float)
    identity = np.eye(n_dof)

    def response_rank(levels: np.ndarray, n_rows: int) -> int:
        """Return the numerical response rank used by the selection contracts."""
        scale = max(1.0, float(levels[0])) if levels.size else 1.0
        tolerance = max(n_rows, n_dof, 1) * np.finfo(float).eps * scale
        return int(np.count_nonzero(levels > tolerance))

    def dominant_basis(response: np.ndarray, rank: int) -> np.ndarray:
        """Recover the dominant response basis without trusting level order."""
        eigenvalues, eigenvectors = np.linalg.eigh(response)
        order = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, order[:rank]]

    lower_rows = None
    lower_responses = None
    lower_reduced = None
    if pressure_budget is None:
        selection_core = core
    else:
        material = np.asarray(material_stiffness, dtype=float)
        pressure_response, pressure_levels, _, _ = build_response_gramian(
            material,
            scaled_rows,
            0,
        )
        pressure_rank = response_rank(pressure_levels, scaled_rows.shape[0])
        pressure_selected = None
        pressure_tolerance = 1e-12 * max(1.0, abs(pressure_budget))
        for candidate_rank in range(pressure_rank + 1):
            candidate = build_response_gramian(
                material,
                scaled_rows,
                candidate_rank,
            )
            candidate_basis = dominant_basis(
                candidate[0],
                candidate_rank,
            )
            candidate_certificate = certify_preconditioned_spectrum(
                material,
                scaled_rows,
                candidate[1],
                None,
                candidate_basis,
            )
            if candidate_certificate[5] <= pressure_budget + pressure_tolerance:
                pressure_selected = (*candidate, candidate_basis, candidate_certificate)
                break
        if pressure_selected is None:
            raise ValueError("no pressure response rank meets its condition budget")
        (
            pressure_response,
            pressure_levels,
            pressure_projector,
            pressure_cutoff,
            pressure_basis,
            pressure_certificate,
        ) = pressure_selected
        (
            selection_core,
            lower_rows,
            lower_responses,
            lower_reduced,
        ) = build_reduced_interaction_state(
            material,
            scaled_rows,
            pressure_basis,
        )
        lower_inverse = apply_reduced_inverse(
            material,
            lower_rows,
            lower_responses,
            lower_reduced,
            identity,
            False,
        )
        lower_adjoint = apply_reduced_inverse(
            material,
            lower_rows,
            lower_responses,
            lower_reduced,
            identity,
            True,
        )
        if not np.allclose(
            pressure_response @ pressure_projector,
            pressure_projector @ pressure_response,
            rtol=1e-9,
            atol=1e-11,
        ):
            raise ValueError("the pressure projector is not response-invariant")
        if not np.allclose(
            pressure_basis @ pressure_basis.T,
            pressure_projector,
            rtol=1e-9,
            atol=1e-11,
        ):
            raise ValueError("the pressure basis does not span its projector")
        if not 0.0 <= pressure_cutoff <= 1.0 + 1e-12:
            raise ValueError("the pressure cutoff ratio must lie in [0, 1]")
        if pressure_certificate[5] > pressure_budget + pressure_tolerance:
            raise ValueError("the pressure projector violates its condition budget")
        if not np.allclose(
            lower_adjoint,
            lower_inverse.T,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("the lower adjoint action is inconsistent")
        if not np.allclose(
            lower_inverse @ selection_core,
            identity,
            rtol=1e-9,
            atol=1e-11,
        ):
            raise ValueError("the lower compact action does not invert its core")
        expected_selection_core = material + scaled_rows.T @ (
            pressure_projector @ scaled_rows
        )
        if not np.allclose(
            selection_core,
            0.5 * (expected_selection_core + expected_selection_core.T),
            rtol=1e-11,
            atol=1e-13,
        ):
            raise ValueError("the pressure-surrogate core is inconsistent")

    _contact_response, contact_levels, _, _ = build_response_gramian(
        selection_core,
        interaction_rows,
        0,
    )
    numerical_rank = response_rank(contact_levels, rows.shape[0])
    retained_row_states = []
    core_response_states = []
    reduced_matrix_states = []
    first_reduction = None
    first_estimate = None

    rhs = np.asarray(right_hand_side, dtype=float)
    if initial_guess is None:
        certification_rhs = rhs
    else:
        guess = np.asarray(initial_guess, dtype=float)
        if guess.shape != (n_dof,) or not np.all(np.isfinite(guess)):
            raise ValueError("initial_guess must be a finite vector of shape (n,)")
        certification_rhs = rhs - full_operator @ guess
        certification_scale = max(
            1.0,
            float(np.linalg.norm(rhs)),
            float(np.linalg.norm(full_operator, ord=2) * np.linalg.norm(guess)),
        )
        if float(np.linalg.norm(certification_rhs)) <= (
            100.0 * np.finfo(float).eps * certification_scale
        ):
            certification_rhs = rhs

    for iteration, budget in enumerate(budgets):
        budget_tolerance = 1e-12 * max(1.0, abs(float(budget)))
        selected = None
        for candidate_rank in range(numerical_rank + 1):
            candidate = build_response_gramian(
                selection_core,
                interaction_rows,
                candidate_rank,
            )
            (
                _candidate_response,
                candidate_levels,
                candidate_projector,
                _candidate_ratio,
            ) = candidate
            candidate_basis = dominant_basis(candidate[0], candidate_rank)
            candidate_certificate = certify_preconditioned_spectrum(
                selection_core,
                interaction_rows,
                candidate_levels,
                None,
                candidate_basis,
                core,
            )
            if candidate_certificate[6][3] <= budget + budget_tolerance:
                selected = (*candidate, candidate_basis, *candidate_certificate)
                break
        if selected is None:
            raise ValueError("no admissible response rank meets a condition budget")
        (
            response,
            levels,
            projector,
            cutoff_ratio,
            retained_basis,
            measured,
            predicted,
            certificate,
            condition_number,
            omitted_interaction,
            posterior_bound,
            robust_certificate,
        ) = selected
        (
            preconditioner,
            retained_rows,
            core_responses,
            reduced_matrix,
        ) = build_reduced_interaction_state(
            selection_core,
            interaction_rows,
            retained_basis,
        )
        right_inverse = apply_reduced_inverse(
            selection_core,
            retained_rows,
            core_responses,
            reduced_matrix,
            identity,
            False,
        )
        adjoint_inverse = apply_reduced_inverse(
            selection_core,
            retained_rows,
            core_responses,
            reduced_matrix,
            identity,
            True,
        )
        reduction, coefficient, estimate, _, _ = certify_residual_bounds(
            full_operator,
            right_inverse,
            certification_rhs,
        )

        if not np.allclose(
            response @ projector, projector @ response, rtol=1e-9, atol=1e-11
        ):
            raise ValueError("the retained projector is not response-invariant")
        if not np.allclose(
            retained_basis @ retained_basis.T,
            projector,
            rtol=1e-9,
            atol=1e-11,
        ):
            raise ValueError("the retained basis does not span the selected projector")
        if not 0.0 <= cutoff_ratio <= 1.0 + 1e-12:
            raise ValueError("the cutoff ratio must lie in [0, 1]")
        if not np.allclose(measured, predicted, rtol=1e-8, atol=1e-10):
            raise ValueError("the spectral certificate is inconsistent")
        if not np.isfinite(certificate) or certificate < 0.0 or certificate >= 1.0:
            raise ValueError("the omitted-response certificate is invalid")
        if not np.isfinite(condition_number) or condition_number < 1.0 - 1e-12:
            raise ValueError("the preconditioned condition number is invalid")
        if not np.isfinite(omitted_interaction) or omitted_interaction < 0.0:
            raise ValueError("the worst omitted interaction is invalid")
        if not np.isclose(
            posterior_bound, 1.0 + omitted_interaction, rtol=1e-12, atol=1e-14
        ):
            raise ValueError(
                "the posterior bound does not follow the omitted interaction"
            )
        if condition_number > posterior_bound + 1e-8:
            raise ValueError("the measured conditioning exceeds the posterior bound")
        if robust_certificate.shape != (5,) or not np.all(
            np.isfinite(robust_certificate)
        ):
            raise ValueError("the approximate-core certificate is invalid")
        c1, c2, delta, robust_bound, robust_condition = robust_certificate
        if c1 <= 0.0 or c2 < c1 or delta < 0.0:
            raise ValueError("the approximate-core constants are invalid")
        expected_robust_bound = max(c2 + delta, 1.0) / min(c1, 1.0)
        if not np.isclose(
            robust_bound,
            expected_robust_bound,
            rtol=1e-12,
            atol=1e-14,
        ):
            raise ValueError("the Appendix-C ceiling is inconsistent")
        if robust_condition < 1.0 - 1e-12 or robust_condition > robust_bound + 1e-8:
            raise ValueError("the robust measured conditioning is invalid")
        if robust_bound > budget + budget_tolerance:
            raise ValueError("the selected projector violates its robust budget")
        expected_preconditioner = selection_core + rows.T @ (projector @ rows)
        if not np.allclose(
            preconditioner,
            0.5 * (expected_preconditioner + expected_preconditioner.T),
            rtol=1e-11,
            atol=1e-13,
        ):
            raise ValueError("the reduced preconditioner is inconsistent")
        if not np.allclose(
            selection_core @ core_responses,
            retained_rows.T,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("the compact core responses are inconsistent")
        if not np.allclose(
            reduced_matrix,
            np.eye(retained_rows.shape[0]) + retained_rows @ core_responses,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("the compact reduced matrix is inconsistent")
        if not np.allclose(adjoint_inverse, right_inverse.T, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "the adjoint action is not the transpose of the forward one"
            )
        if not np.allclose(
            right_inverse @ preconditioner, identity, rtol=1e-9, atol=1e-11
        ):
            raise ValueError("the reduced inverse does not invert the preconditioner")
        if not all(np.isfinite(value) for value in (reduction, coefficient, estimate)):
            raise ValueError("a residual certificate is invalid")
        if reduction > estimate + 1e-9:
            raise ValueError("a field-of-values certificate is invalid")

        retained_row_states.append(retained_rows)
        core_response_states.append(core_responses)
        reduced_matrix_states.append(reduced_matrix)
        if iteration == 0:
            first_reduction = float(reduction)
            first_estimate = float(estimate)

    iterate, history, coefficients, used = compute_directional_residual(
        full_operator,
        right_hand_side,
        np.asarray(material_stiffness, dtype=float)
        if pressure_budget is not None
        else selection_core,
        tuple(retained_row_states),
        tuple(core_response_states),
        tuple(reduced_matrix_states),
        int(budgets.size),
        cycle_lengths,
        initial_guess,
        lower_rows,
        lower_responses,
        lower_reduced,
    )

    if not np.allclose(scaled_rows.T @ scaled_rows, volumetric, rtol=1e-11, atol=1e-13):
        raise ValueError("the scaled pressure rows do not reproduce the Gram matrix")
    if not np.allclose(
        symmetric_part,
        0.5 * (full_operator + full_operator.T),
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError("the symmetric part is inconsistent with the full operator")
    if used < 0 or used > budgets.size:
        raise ValueError("the flexible iteration count is invalid")
    if coefficients.shape != budgets.shape or not np.all(np.isfinite(coefficients)):
        raise ValueError("the flexible coefficients are invalid")
    fresh = float(np.linalg.norm(rhs - full_operator @ iterate) / np.linalg.norm(rhs))
    rho = float(history[-1])
    if not np.allclose(rho, fresh, rtol=1e-11, atol=1e-13):
        raise ValueError("the residual history is not a fresh full residual")
    if used:
        expected_first_reduction = history[0] * first_reduction
        expected_first_estimate = history[0] * first_estimate
        if not np.allclose(
            history[1],
            expected_first_reduction,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("the first certified reduction disagrees with the cycle")
        if history[1] > expected_first_estimate + 1e-9:
            raise ValueError("the first field-of-values certificate is invalid")
    return rho
SCICODE_GOLD_EOF
