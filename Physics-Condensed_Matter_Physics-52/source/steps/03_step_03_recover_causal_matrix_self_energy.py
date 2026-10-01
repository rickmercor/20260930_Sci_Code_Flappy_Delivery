"""
Recover a causal matrix-valued discrete self-energy from an augmented Lehmann spectrum.

Causal response functions have real poles and positive-semidefinite matrix residues.  Preserving that structure is especially important for noncommuting orbital channels, where elementwise scalar manipulations do not respect the shared matrix metric.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray], the static self-energy matrix, dynamical self-energy poles, and dynamical matrix residues
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def recover_causal_matrix_self_energy(
    augmented_poles: "np.ndarray",
    augmented_residues: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Return the static matrix and causal dynamical self-energy spectrum.

    Parameters
    ----------
    augmented_poles : np.ndarray
        Finite real poles of shape ``(P,)``.
    augmented_residues : np.ndarray
        Hermitian positive-semidefinite residues of shape ``(P, 2*N, 2*N)``
        for a complete normalized doubled propagator, ordered with physical
        channels first and matching auxiliary channels second.  The physical
        zeroth moment is the ``N``-dimensional identity.  Either the complete
        doubled norm is positive definite or the auxiliary sector is identically
        zero.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The Hermitian static contribution of shape ``(N, N)``, real dynamical
        poles of shape ``(Q,)``, and Hermitian positive-semidefinite dynamical
        residues of shape ``(Q, N, N)``.  Ordering and equivalent splitting of
        exactly degenerate residues do not change the represented self-energy.
        An identically zero auxiliary sector returns empty dynamical arrays.

    Raises
    ------
    ValueError
        If shapes, finiteness, reality, positivity, normalization, parity of the
        doubled dimension, or the required spectral metric violate the contract.

    Notes
    -----
    Input arrays are not required to be preserved by implementations.
    For each positive-semidefinite metric or residue with eigenvalues ``r``,
    numerical rank retains ``r[k] > 2e-11 * max(1, max(abs(r)))``.  A
    reconstructed pole group whose summed-residue trace is at most ``2e-11``
    is numerically null and omitted.
    """
    return self_energy_static, self_energy_poles, self_energy_residues

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_recover_causal_matrix_self_energy(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return interacting, noninteracting, larger-bath, and invalid cases."""
    evaluator = """import numpy as np
def evaluate(builder, lehmann, recover, impurity, bath, coupling, interaction, number):
    h, hint, annihilators, occupations = builder(
        impurity.copy(), bath.copy(), coupling.copy(), interaction
    )
    _, augmented_poles, augmented_residues = lehmann(
        h, hint, annihilators, occupations, number
    )
    static, poles, residues = recover(augmented_poles, augmented_residues)
    static = np.asarray(static, complex)
    poles = np.asarray(poles, float)
    residues = np.asarray(residues, complex)
    if static.shape != (2,2) or poles.ndim != 1 or residues.shape != (poles.size,2,2):
        raise AssertionError("self-energy spectrum has the wrong shape")
    if residues.size:
        minimum = min(float(np.min(np.linalg.eigvalsh((w+w.conj().T)/2))) for w in residues)
        if minimum < -2e-8:
            raise AssertionError("self-energy spectrum is not causal")
    probes = (0.17+0.53j, -1.49+0.39j, 2.31+0.77j)
    values = [static + sum((w/(z-p) for p,w in zip(poles,residues)), np.zeros((2,2),complex))
              for z in probes]
    moment0 = np.sum(residues,axis=0) if poles.size else np.zeros((2,2),complex)
    moment1 = np.sum(poles[:,None,None]*residues,axis=0) if poles.size else np.zeros((2,2),complex)
    return np.concatenate((static.ravel(), *(x.ravel() for x in values), moment0.ravel(), moment1.ravel()))
"""
    return [
        {
            "setup": evaluator + """
impurity=np.array([[-0.973779,0.112032-0.097339j],[0.112032+0.097339j,-1.300553]],complex)
bath=np.array([[[-1.084493,-0.148375-0.048758j],[-0.148375+0.048758j,1.029384]]],complex)
coupling=np.array([[[-0.049439-0.310296j,-0.118014+0.213365j],[0.009953+0.122430j,-0.019628+0.285367j]]],complex)
interaction=2.367874
number=2
""",
            "call": "evaluate(build_noncommuting_impurity_model, compute_augmented_matrix_lehmann_spectrum, recover_causal_matrix_self_energy, impurity, bath, coupling, interaction, number)",
            "gold_call": "evaluate(_oracle_build_noncommuting_impurity_model, _oracle_compute_augmented_matrix_lehmann_spectrum, _oracle_recover_causal_matrix_self_energy, impurity, bath, coupling, interaction, number)",
            "tol": 6e-9,
        },
        {
            "setup": evaluator + """
impurity=np.array([[-0.41,0.09-0.04j],[0.09+0.04j,-0.18]],complex)
bath=np.array([[[-0.83,0.05+0.02j],[0.05-0.02j,0.62]]],complex)
coupling=np.array([[[0.2+0.03j,-0.08+0.04j],[0.06-0.07j,0.17+0.02j]]],complex)
interaction=0.0
number=2
""",
            "call": "evaluate(build_noncommuting_impurity_model, compute_augmented_matrix_lehmann_spectrum, recover_causal_matrix_self_energy, impurity, bath, coupling, interaction, number)",
            "gold_call": "evaluate(_oracle_build_noncommuting_impurity_model, _oracle_compute_augmented_matrix_lehmann_spectrum, _oracle_recover_causal_matrix_self_energy, impurity, bath, coupling, interaction, number)",
            "tol": 5e-10,
        },
        {
            "setup": evaluator + """
impurity=np.array([[-0.72,0.16+0.09j],[0.16-0.09j,-0.31]],complex)
bath=np.array([[[-1.1,0.12-0.03j],[0.12+0.03j,0.45]],
               [[-0.25,-0.08+0.11j],[-0.08-0.11j,1.2]]],complex)
coupling=np.array([[[0.21+0.04j,-0.07+0.13j],[0.09-0.02j,0.18+0.06j]],
                   [[-0.11+0.08j,0.15-0.05j],[0.04+0.12j,-0.19+0.03j]]],complex)
interaction=1.35
number=3
""",
            "call": "evaluate(build_noncommuting_impurity_model, compute_augmented_matrix_lehmann_spectrum, recover_causal_matrix_self_energy, impurity, bath, coupling, interaction, number)",
            "gold_call": "evaluate(_oracle_build_noncommuting_impurity_model, _oracle_compute_augmented_matrix_lehmann_spectrum, _oracle_recover_causal_matrix_self_energy, impurity, bath, coupling, interaction, number)",
            "tol": 8e-9,
        },
        {
            "setup": """import numpy as np
poles=np.array([0.2])
residues=np.zeros((1,4,4),complex)
residues[0]=np.diag([1.0,1.0,-0.1,0.2])
def candidate():
    try:
        recover_causal_matrix_self_energy(poles.copy(),residues.copy())
        return np.array([0.0])
    except ValueError:
        return np.array([1.0])
def reference():
    try:
        _oracle_recover_causal_matrix_self_energy(poles.copy(),residues.copy())
        return np.array([0.0])
    except ValueError:
        return np.array([1.0])
""",
            "call": "candidate()",
            "gold_call": "reference()",
        },
    ]
