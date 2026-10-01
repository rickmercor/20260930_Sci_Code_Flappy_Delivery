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

def build_mass_weighted_dynamical_matrix(
    phi: "np.ndarray",
    masses: "np.ndarray",
) -> "np.ndarray":
    """Construct the mass-weighted dynamical matrix.

    Parameters
    ----------
    phi : np.ndarray
        Real symmetric force-constant matrix with shape $(N, N)$.
    masses : np.ndarray
        One-dimensional array of $N$ strictly positive coordinate masses.

    Returns
    -------
    mass_weighted_matrix : np.ndarray
        Real symmetric array with shape $(N, N)$.

    Raises
    ------
    ValueError
        If either input cannot be represented by a real numeric array; if phi
        is not a nonempty square matrix; if masses does not contain exactly one
        entry per coordinate; if either input contains a nonfinite value; if
        any mass is not strictly positive; or if phi is not symmetric with
        rtol=0 and atol=$10^{-12}$.
    """
    try:
        phi_raw = np.asarray(phi)
        masses_raw = np.asarray(masses)
    except (TypeError, ValueError) as exc:
        raise ValueError("phi and masses must be numeric arrays") from exc

    if np.iscomplexobj(phi_raw) and np.any(np.imag(phi_raw) != 0.0):
        raise ValueError("phi must be real-valued")
    if np.iscomplexobj(masses_raw) and np.any(np.imag(masses_raw) != 0.0):
        raise ValueError("masses must be real-valued")

    try:
        phi_array = np.asarray(phi_raw, dtype=float)
        masses_array = np.asarray(masses_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("phi and masses must contain real numeric values") from exc

    if (
        phi_array.ndim != 2
        or phi_array.shape[0] != phi_array.shape[1]
        or phi_array.shape[0] == 0
    ):
        raise ValueError("phi must be a nonempty square matrix")
    if masses_array.ndim != 1 or masses_array.size != phi_array.shape[0]:
        raise ValueError("masses must have one entry per coordinate")
    if not np.all(np.isfinite(phi_array)) or not np.all(np.isfinite(masses_array)):
        raise ValueError("phi and masses must contain only finite values")
    if np.any(masses_array <= 0.0):
        raise ValueError("all masses must be strictly positive")
    if not np.allclose(phi_array, phi_array.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("phi must be symmetric within atol=1e-12")

    root_masses = np.sqrt(masses_array)
    result = phi_array / np.outer(root_masses, root_masses)
    return 0.5 * (result + result.T)

import numpy as np

def compute_molecular_berry_curvature(
    eps_v: "np.ndarray",
    eps_c: "np.ndarray",
    g_cv: "np.ndarray",
) -> "np.ndarray":
    """Compute the real antisymmetric molecular Berry-curvature matrix.

    Parameters
    ----------
    eps_v : np.ndarray
        One-dimensional real valence-energy array with shape $(N_k,)$.
    eps_c : np.ndarray
        One-dimensional real conduction-energy array with shape $(N_k,)$.
    g_cv : np.ndarray
        Complex electron-phonon couplings with shape $(N_k, N)$.

    Returns
    -------
    curvature : np.ndarray
        Real antisymmetric molecular Berry-curvature matrix with shape $(N, N)$.

    Raises
    ------
    ValueError
        If the energy arrays are not nonempty, real, finite, one-dimensional,
        and equal in length; if g_cv is not a finite two-dimensional array with
        one row per k point and at least one coordinate; or if any conduction
        energy is not strictly greater than its corresponding valence energy.
    """
    try:
        eps_v_raw = np.asarray(eps_v)
        eps_c_raw = np.asarray(eps_c)
        coupling = np.asarray(g_cv, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError("energies and couplings must be numeric arrays") from exc

    if np.iscomplexobj(eps_v_raw) and np.any(np.imag(eps_v_raw) != 0.0):
        raise ValueError("eps_v must be real-valued")
    if np.iscomplexobj(eps_c_raw) and np.any(np.imag(eps_c_raw) != 0.0):
        raise ValueError("eps_c must be real-valued")
    try:
        valence = np.asarray(eps_v_raw, dtype=float)
        conduction = np.asarray(eps_c_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("energies must contain real numeric values") from exc

    if valence.ndim != 1 or conduction.ndim != 1 or valence.size == 0:
        raise ValueError("energy arrays must be nonempty and one-dimensional")
    if conduction.shape != valence.shape:
        raise ValueError("energy arrays must have equal length")
    if coupling.ndim != 2 or coupling.shape[0] != valence.size or coupling.shape[1] == 0:
        raise ValueError("g_cv must have shape (N_k, N) with N >= 1")
    if not np.all(np.isfinite(valence)) or not np.all(np.isfinite(conduction)):
        raise ValueError("energies must be finite")
    if not np.all(np.isfinite(coupling.real)) or not np.all(np.isfinite(coupling.imag)):
        raise ValueError("g_cv must be finite")

    gaps = conduction - valence
    if np.any(gaps <= 0.0):
        raise ValueError("every conduction energy must exceed its valence energy")

    scaled_coupling = coupling / gaps[:, None]
    gram = scaled_coupling.conj().T @ scaled_coupling
    raw_curvature = (1j / float(valence.size)) * (gram - gram.conj())
    real_curvature = np.asarray(raw_curvature.real, dtype=float)
    return 0.5 * (real_curvature - real_curvature.T)

import numpy as np

def analyze_unperturbed_subspace(
    k_tilde: "np.ndarray",
    omega0: float,
    degeneracy_tolerance: float = 1.0e-8,
) -> "np.ndarray":
    """Return the unperturbed spectrum and target-subspace projector.

    Parameters
    ----------
    k_tilde : np.ndarray
        Real symmetric positive-semidefinite matrix with shape $(N, N)$.
    omega0 : float
        Strictly positive finite target frequency.
    degeneracy_tolerance : float
        Strictly positive finite absolute tolerance for selecting frequencies
        equal to omega0.

    Returns
    -------
    analysis : np.ndarray
        Real array with shape $(N + 1, N)$. The first row contains the sorted
        unperturbed frequencies and the remaining rows contain the rank-two
        orthogonal projector onto the target degenerate subspace.

    Raises
    ------
    ValueError
        If k_tilde cannot be represented by a real numeric array; if it is not
        a nonempty finite square matrix symmetric with rtol=0 and atol=$10^{-12}$;
        if it has an eigenvalue below -$10^{-10}$; if omega0 or
        degeneracy_tolerance is not strictly positive and finite; or if
        exactly two frequencies are not within degeneracy_tolerance of omega0.
    """
    try:
        matrix_raw = np.asarray(k_tilde)
    except (TypeError, ValueError) as exc:
        raise ValueError("k_tilde must be a numeric array") from exc
    if np.iscomplexobj(matrix_raw) and np.any(np.imag(matrix_raw) != 0.0):
        raise ValueError("k_tilde must be real-valued")
    try:
        matrix = np.asarray(matrix_raw, dtype=float)
        target = float(omega0)
        tolerance = float(degeneracy_tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("matrix and scalar inputs must be real numeric values") from exc

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("k_tilde must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("k_tilde must be finite")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("k_tilde must be symmetric within atol=1e-12")
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("omega0 must be strictly positive and finite")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("degeneracy_tolerance must be strictly positive and finite")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if np.min(eigenvalues) < -1.0e-10:
        raise ValueError("k_tilde must be positive semidefinite")
    frequencies = np.sqrt(np.clip(eigenvalues, 0.0, None))
    target_indices = np.flatnonzero(np.abs(frequencies - target) <= tolerance)
    if target_indices.size != 2:
        raise ValueError("exactly two frequencies must belong to the target subspace")

    basis = eigenvectors[:, target_indices]
    projector = basis @ basis.T
    projector = 0.5 * (projector + projector.T)
    return np.vstack((frequencies, projector))

import numpy as np

def compute_first_order_frequencies(
    unperturbed_analysis: "np.ndarray",
    g_tilde: "np.ndarray",
    omega0: float,
) -> "np.ndarray":
    """Compute the two first-order MBC-split modes.

    Parameters
    ----------
    unperturbed_analysis : np.ndarray
        Real packed array from analyze_unperturbed_subspace with shape
        $(N + 1, N)$.
    g_tilde : np.ndarray
        Real antisymmetric curvature matrix with shape $(N, N)$.
    omega0 : float
        Strictly positive finite degenerate frequency.

    Returns
    -------
    first_order_modes : np.ndarray
        Complex array with shape $(N + 1, 2)$. The first row contains the two
        first-order frequencies omega0 + lam/(2*omega0) in ascending order, where
        lam are the eigenvalues of the projected first-order operator, and the
        remaining $N$ rows contain their associated orthonormal displacement
        vectors.

    Raises
    ------
    ValueError
        If unperturbed_analysis is not a finite real array with shape
        $(N + 1, N)$, a nonnegative sorted spectrum, and a symmetric idempotent
        rank-two projector within atol=$10^{-8}$; if g_tilde is not a finite real
        antisymmetric $(N, N)$ array within atol=$10^{-12}$; or if omega0 is not
        strictly positive and finite.
    """
    try:
        analysis_raw = np.asarray(unperturbed_analysis)
        curvature_raw = np.asarray(g_tilde)
    except (TypeError, ValueError) as exc:
        raise ValueError("analysis and curvature must be numeric arrays") from exc
    if np.iscomplexobj(analysis_raw) and np.any(np.imag(analysis_raw) != 0.0):
        raise ValueError("unperturbed_analysis must be real-valued")
    if np.iscomplexobj(curvature_raw) and np.any(np.imag(curvature_raw) != 0.0):
        raise ValueError("g_tilde must be real-valued")
    try:
        analysis = np.asarray(analysis_raw, dtype=float)
        curvature = np.asarray(curvature_raw, dtype=float)
        target = float(omega0)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("inputs must contain real numeric values") from exc

    if analysis.ndim != 2 or analysis.shape[0] != analysis.shape[1] + 1:
        raise ValueError("unperturbed_analysis must have shape (N+1, N)")
    n_coordinates = analysis.shape[1]
    if n_coordinates < 2 or not np.all(np.isfinite(analysis)):
        raise ValueError("unperturbed_analysis must be finite with N >= 2")
    spectrum = analysis[0]
    projector = analysis[1:]
    if np.any(spectrum < 0.0) or np.any(np.diff(spectrum) < -1.0e-12):
        raise ValueError("the packed spectrum must be nonnegative and sorted")
    if not np.allclose(projector, projector.T, rtol=0.0, atol=1.0e-8):
        raise ValueError("the packed projector must be symmetric")
    if not np.allclose(projector @ projector, projector, rtol=0.0, atol=1.0e-8):
        raise ValueError("the packed projector must be idempotent")
    if not np.isclose(np.trace(projector), 2.0, rtol=0.0, atol=1.0e-8):
        raise ValueError("the packed projector must have rank two")
    if curvature.shape != (n_coordinates, n_coordinates):
        raise ValueError("g_tilde must have shape (N, N)")
    if not np.all(np.isfinite(curvature)):
        raise ValueError("g_tilde must be finite")
    if not np.allclose(curvature, -curvature.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("g_tilde must be antisymmetric within atol=1e-12")
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("omega0 must be strictly positive and finite")

    projector_values, projector_vectors = np.linalg.eigh(projector)
    basis = projector_vectors[:, np.argsort(projector_values)[-2:]]
    restricted = basis.conj().T @ (2j * target * curvature) @ basis
    restricted = 0.5 * (restricted + restricted.conj().T)
    shifts_in_squared_frequency, restricted_vectors = np.linalg.eigh(restricted)
    frequencies = target + shifts_in_squared_frequency / (2.0 * target)
    order = np.argsort(frequencies.real, kind="stable")
    frequencies = np.asarray(frequencies.real[order], dtype=float)
    displacements = basis.astype(complex) @ restricted_vectors[:, order]
    return np.vstack((frequencies.astype(complex), displacements))

import numpy as np

def solve_full_mbc_eigenproblem(
    k_tilde: "np.ndarray",
    g_tilde: "np.ndarray",
    imaginary_tolerance: float = 1.0e-9,
) -> "np.ndarray":
    """Solve the companion linearization of the full MBC eigenproblem.

    Parameters
    ----------
    k_tilde : np.ndarray
        Real symmetric positive-definite matrix with shape $(N, N)$.
    g_tilde : np.ndarray
        Real antisymmetric matrix with shape $(N, N)$.
    imaginary_tolerance : float
        Strictly positive finite upper bound on the magnitude of an eligible
        companion root's imaginary part.

    Returns
    -------
    modes : np.ndarray
        Complex array with shape $(N + 1, N)$. The first row contains ascending
        positive frequencies and each remaining column contains the associated
        normalized displacement vector.

    Raises
    ------
    ValueError
        If k_tilde or g_tilde cannot be represented by real numeric arrays; if
        they are not equal-size, nonempty, finite square matrices; if k_tilde
        is not symmetric within atol=$10^{-12}$ or is not positive definite; if
        g_tilde is not antisymmetric within atol=$10^{-12}$; if
        imaginary_tolerance is not strictly positive and finite; if the
        companion spectrum does not contain exactly $N$ eligible positive roots;
        or if an eligible root has a zero or nonfinite displacement norm.
    """
    try:
        matrix_raw = np.asarray(k_tilde)
        curvature_raw = np.asarray(g_tilde)
    except (TypeError, ValueError) as exc:
        raise ValueError("k_tilde and g_tilde must be numeric arrays") from exc
    if np.iscomplexobj(matrix_raw) and np.any(np.imag(matrix_raw) != 0.0):
        raise ValueError("k_tilde must be real-valued")
    if np.iscomplexobj(curvature_raw) and np.any(np.imag(curvature_raw) != 0.0):
        raise ValueError("g_tilde must be real-valued")
    try:
        matrix = np.asarray(matrix_raw, dtype=float)
        curvature = np.asarray(curvature_raw, dtype=float)
        tolerance = float(imaginary_tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("inputs must contain real numeric values") from exc

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("k_tilde must be a nonempty square matrix")
    if curvature.shape != matrix.shape:
        raise ValueError("g_tilde must have the same square shape as k_tilde")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(curvature)):
        raise ValueError("k_tilde and g_tilde must be finite")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("k_tilde must be symmetric within atol=1e-12")
    if np.min(np.linalg.eigvalsh(matrix)) <= 0.0:
        raise ValueError("k_tilde must be positive definite")
    if not np.allclose(curvature, -curvature.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("g_tilde must be antisymmetric within atol=1e-12")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("imaginary_tolerance must be strictly positive and finite")

    n_coordinates = matrix.shape[0]
    identity = np.eye(n_coordinates, dtype=complex)
    zero = np.zeros((n_coordinates, n_coordinates), dtype=complex)
    quadratic_block = matrix + 2.0 * curvature.conj().T @ curvature
    companion = np.block(
        [
            [zero, identity],
            [quadratic_block.astype(complex), 2j * curvature],
        ]
    )
    roots, companion_vectors = np.linalg.eig(companion)
    eligible = np.flatnonzero(
        (roots.real > 0.0) & (np.abs(roots.imag) <= tolerance)
    )
    if eligible.size != n_coordinates:
        raise ValueError("the companion spectrum must contain exactly N eligible roots")

    eligible = eligible[np.argsort(roots[eligible].real, kind="stable")]
    frequencies = roots[eligible].real.astype(float)
    displacements = np.empty((n_coordinates, n_coordinates), dtype=complex)
    for output_column, eigenvector_index in enumerate(eligible):
        displacement = companion_vectors[:n_coordinates, eigenvector_index]
        norm = np.linalg.norm(displacement)
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("an eligible root has an invalid displacement vector")
        displacement = displacement / norm
        pivot = int(np.argmax(np.abs(displacement)))
        displacement = displacement * np.exp(-1j * np.angle(displacement[pivot]))
        if displacement[pivot].real < 0.0:
            displacement = -displacement
        displacements[:, output_column] = displacement

    return np.vstack((frequencies.astype(complex), displacements))

import numpy as np

def match_target_modes(
    first_order_modes: "np.ndarray",
    full_modes: "np.ndarray",
) -> "np.ndarray":
    """Select the two full modes with greatest target-subspace weight.

    Parameters
    ----------
    first_order_modes : np.ndarray
        Complex packed table from compute_first_order_frequencies with shape
        $(N + 1, 2)$. Its first row contains two ascending positive
        first-order frequencies, and its remaining rows contain the associated
        orthonormal displacement basis.
    full_modes : np.ndarray
        Complex modal table with shape $(N + 1, M)$, $M$ at least two. Its first
        row contains positive real frequencies and the remaining rows contain
        normalized displacement vectors.

    Returns
    -------
    matched_modes : np.ndarray
        Complex array with shape $(N + 2, 2)$. Row zero contains ascending target
        frequencies, row one contains their projection weights, and the final
        $N$ rows contain their normalized displacement vectors.

    Raises
    ------
    ValueError
        If first_order_modes is not a finite complex array with shape
        $(N + 1, 2)$; if its frequencies are not ascending, positive, and real
        within atol=$10^{-9}$; if its displacement columns are not orthonormal
        within atol=$10^{-8}$; if full_modes does not have shape $(N + 1, M)$ with $M$
        at least two; if its entries are nonfinite; if its frequencies are not
        positive and real within atol=$10^{-9}$; or if its displacement columns are
        not normalized within atol=$10^{-8}$.
    """
    try:
        first_order = np.asarray(first_order_modes, dtype=complex)
        modes = np.asarray(full_modes, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError("first-order and full modes must be numeric arrays") from exc

    if first_order.ndim != 2 or first_order.shape[1] != 2 or first_order.shape[0] < 3:
        raise ValueError("first_order_modes must have shape (N+1, 2) with N >= 2")
    if not np.all(np.isfinite(first_order.real)) or not np.all(np.isfinite(first_order.imag)):
        raise ValueError("first_order_modes must be finite")
    n_coordinates = first_order.shape[0] - 1
    first_order_frequencies = first_order[0]
    if (
        np.any(np.abs(first_order_frequencies.imag) > 1.0e-9)
        or np.any(first_order_frequencies.real <= 0.0)
        or np.any(np.diff(first_order_frequencies.real) < -1.0e-12)
    ):
        raise ValueError("first-order frequencies must be ascending, positive, and real")
    first_order_basis = first_order[1:]
    gram = first_order_basis.conj().T @ first_order_basis
    if not np.allclose(gram, np.eye(2), rtol=0.0, atol=1.0e-8):
        raise ValueError("first-order displacement columns must be orthonormal")
    projector = first_order_basis @ first_order_basis.conj().T
    if modes.ndim != 2 or modes.shape[0] != n_coordinates + 1 or modes.shape[1] < 2:
        raise ValueError("full_modes must have shape (N+1, M) with M >= 2")
    if not np.all(np.isfinite(modes.real)) or not np.all(np.isfinite(modes.imag)):
        raise ValueError("full_modes must be finite")

    frequencies = modes[0]
    if np.any(np.abs(frequencies.imag) > 1.0e-9) or np.any(frequencies.real <= 0.0):
        raise ValueError("full-mode frequencies must be positive and real")
    displacements = modes[1:]
    norms = np.linalg.norm(displacements, axis=0)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1.0e-8):
        raise ValueError("full-mode displacement columns must be normalized")

    weights = np.real(
        np.sum(displacements.conj() * (projector @ displacements), axis=0)
    )
    weights = np.clip(weights, 0.0, 1.0)
    selected = np.argsort(-weights, kind="stable")[:2]
    selected = selected[np.argsort(frequencies.real[selected], kind="stable")]
    return np.vstack(
        (
            frequencies.real[selected].astype(complex),
            weights[selected].astype(complex),
            displacements[:, selected],
        )
    )

import numpy as np

def label_modes_by_chirality(
    matched_modes: "np.ndarray",
    xy_pairs: "np.ndarray",
) -> "np.ndarray":
    """Compute chiralities and order the two modes as plus and minus branches.

    Parameters
    ----------
    matched_modes : np.ndarray
        Complex array with shape $(N + 2, 2)$ from match_target_modes. Row zero
        contains frequencies, row one contains projection weights, and the
        remaining $N$ rows contain normalized displacement vectors.
    xy_pairs : np.ndarray
        Integer array with shape $(N/2, 2)$ that partitions all coordinate
        indices into ordered Cartesian (x, y) pairs.

    Returns
    -------
    branch_data : np.ndarray
        Real array with shape $(2, 2)$. Row zero contains frequencies and row one
        contains chiralities; columns are ordered [plus, minus].

    Raises
    ------
    ValueError
        If matched_modes is not a finite numeric array with shape $(N + 2, 2)$,
        $N$ at least two; if its frequencies are not positive and real within
        atol=$10^{-9}$; if its weights are not real and in $[0,1]$ within atol=$10^{-9}$;
        if its displacement vectors are not normalized within atol=$10^{-8}$; if
        xy_pairs is not an integer array that covers every coordinate exactly
        once; if a chirality expectation has an imaginary component larger
        than $10^{-9}$; or if the two chiralities are not respectively separable
        into one strictly positive and one strictly negative value.
    """
    try:
        modes = np.asarray(matched_modes, dtype=complex)
        pair_raw = np.asarray(xy_pairs)
    except (TypeError, ValueError) as exc:
        raise ValueError("matched_modes and xy_pairs must be numeric arrays") from exc

    if modes.ndim != 2 or modes.shape[1] != 2 or modes.shape[0] < 4:
        raise ValueError("matched_modes must have shape (N+2, 2) with N >= 2")
    if not np.all(np.isfinite(modes.real)) or not np.all(np.isfinite(modes.imag)):
        raise ValueError("matched_modes must be finite")
    n_coordinates = modes.shape[0] - 2
    frequencies = modes[0]
    weights = modes[1]
    displacements = modes[2:]
    if np.any(np.abs(frequencies.imag) > 1.0e-9) or np.any(frequencies.real <= 0.0):
        raise ValueError("frequencies must be positive and real")
    if np.any(np.abs(weights.imag) > 1.0e-9):
        raise ValueError("projection weights must be real")
    if np.any(weights.real < -1.0e-9) or np.any(weights.real > 1.0 + 1.0e-9):
        raise ValueError("projection weights must lie in [0,1]")
    if not np.allclose(
        np.linalg.norm(displacements, axis=0),
        1.0,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError("displacement vectors must be normalized")

    if pair_raw.ndim != 2 or pair_raw.shape[1] != 2:
        raise ValueError("xy_pairs must have shape (N/2, 2)")
    try:
        pairs = np.asarray(pair_raw, dtype=int)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("xy_pairs must contain integer indices") from exc
    if not np.array_equal(pair_raw, pairs):
        raise ValueError("xy_pairs must contain integer indices")
    if pairs.size != n_coordinates:
        raise ValueError("xy_pairs must cover every coordinate exactly once")
    if not np.array_equal(np.sort(pairs.ravel()), np.arange(n_coordinates)):
        raise ValueError("xy_pairs must cover every coordinate exactly once")

    angular_momentum = np.zeros((n_coordinates, n_coordinates), dtype=complex)
    for x_index, y_index in pairs:
        angular_momentum[x_index, y_index] = -1j
        angular_momentum[y_index, x_index] = 1j

    chiralities = np.empty(2, dtype=float)
    for mode_index in range(2):
        displacement = displacements[:, mode_index]
        expectation = displacement.conj() @ angular_momentum @ displacement
        if abs(expectation.imag) > 1.0e-9:
            raise ValueError("chirality expectation must be real")
        chiralities[mode_index] = float(expectation.real)

    plus_index = int(np.argmax(chiralities))
    minus_index = int(np.argmin(chiralities))
    if chiralities[plus_index] <= 0.0 or chiralities[minus_index] >= 0.0:
        raise ValueError("one mode must have positive and one negative chirality")
    order = np.array([plus_index, minus_index], dtype=int)
    return np.vstack((frequencies.real[order], chiralities[order])).astype(float)

import numpy as np

def compute_corrected_peak_separation(
    phi: "np.ndarray",
    masses: "np.ndarray",
    eps_v: "np.ndarray",
    eps_c: "np.ndarray",
    g_cv: "np.ndarray",
    omega0: float,
    xy_pairs: "np.ndarray",
    gamma: "np.ndarray",
    inv_q: "np.ndarray",
    conversion: float,
) -> float:
    r"""Return the chirality-resolved, Fano-corrected peak separation.

    Parameters
    ----------
    phi : np.ndarray
        Real symmetric force-constant matrix with shape $(N, N)$.
    masses : np.ndarray
        One-dimensional array of $N$ strictly positive coordinate masses.
    eps_v : np.ndarray
        One-dimensional valence-energy array with shape $(N_k,)$.
    eps_c : np.ndarray
        One-dimensional conduction-energy array with shape $(N_k,)$.
    g_cv : np.ndarray
        Complex coupling array with shape $(N_k, N)$.
    omega0 : float
        Strictly positive finite target frequency.
    xy_pairs : np.ndarray
        Integer Cartesian-pair array covering all $N$ coordinates exactly once.
    gamma : np.ndarray
        Two finite nonnegative linewidths ordered [plus, minus].
    inv_q : np.ndarray
        Two finite inverse Fano parameters ordered [plus, minus].
    conversion : float
        Strictly positive finite conversion factor from meV to $\mathrm{cm}^{-1}$.

    Returns
    -------
    separation : float
        Absolute corrected peak separation in $\mathrm{cm}^{-1}$, rounded at the final step
        to 11 decimal places.

    Raises
    ------
    ValueError
        Under any invalid-input condition declared by the preceding seven
        steps; if gamma or inv_q is not a finite one-dimensional array of
        length two; if either linewidth is negative; or if conversion is not
        strictly positive and finite.
    """
    try:
        linewidths = np.asarray(gamma, dtype=float)
        inverse_q = np.asarray(inv_q, dtype=float)
        unit_conversion = float(conversion)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("gamma, inv_q, and conversion must be real numeric values") from exc
    if linewidths.ndim != 1 or linewidths.shape != (2,):
        raise ValueError("gamma must be a one-dimensional array of length two")
    if inverse_q.ndim != 1 or inverse_q.shape != (2,):
        raise ValueError("inv_q must be a one-dimensional array of length two")
    if not np.all(np.isfinite(linewidths)) or not np.all(np.isfinite(inverse_q)):
        raise ValueError("gamma and inv_q must be finite")
    if np.any(linewidths < 0.0):
        raise ValueError("linewidths must be nonnegative")
    if not np.isfinite(unit_conversion) or unit_conversion <= 0.0:
        raise ValueError("conversion must be strictly positive and finite")

    mass_weighted = build_mass_weighted_dynamical_matrix(phi, masses)
    curvature = compute_molecular_berry_curvature(eps_v, eps_c, g_cv)
    unperturbed = analyze_unperturbed_subspace(mass_weighted, omega0)
    first_order_modes = compute_first_order_frequencies(
        unperturbed,
        curvature,
        omega0,
    )
    full_modes = solve_full_mbc_eigenproblem(mass_weighted, curvature)
    matched = match_target_modes(first_order_modes, full_modes)
    branch_data = label_modes_by_chirality(matched, xy_pairs)

    corrected_peaks = branch_data[0] + 0.5 * linewidths * inverse_q
    separation = abs(float(corrected_peaks[1] - corrected_peaks[0]))
    return float(np.round(separation * unit_conversion, 11))
SCICODE_GOLD_EOF
