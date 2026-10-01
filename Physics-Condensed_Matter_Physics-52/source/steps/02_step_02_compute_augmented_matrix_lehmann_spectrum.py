"""
Compute the complete zero-temperature discrete spectrum of the doubled impurity propagator.

Lehmann spectra encode addition and removal excitations of a finite fermionic Hamiltonian.  Matrix-valued residues retain orbital coherences and phases, while canonical-sector resolution avoids mixing states with different conserved particle numbers.

Returns
-------
tuple[float, np.ndarray, np.ndarray], the canonical ground energy, augmented real poles, and augmented matrix residues
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_augmented_matrix_lehmann_spectrum(
    hamiltonian: "np.ndarray",
    interaction_hamiltonian: "np.ndarray",
    annihilators: "np.ndarray",
    occupations: "np.ndarray",
    particle_number: int,
) -> "tuple[float, np.ndarray, np.ndarray]":
    """Return the canonical ground energy and augmented Lehmann spectrum.

    Parameters
    ----------
    hamiltonian : np.ndarray
        Finite Hermitian many-body Hamiltonian of shape ``(D, D)``.
    interaction_hamiltonian : np.ndarray
        Matching finite Hermitian interaction Hamiltonian.
    annihilators : np.ndarray
        Mode-resolved annihilation matrices of shape ``(L, D, D)`` in the
        occupation-bit convention of the preceding model construction.
    occupations : np.ndarray
        Integer particle-number labels of shape ``(D,)``.
    particle_number : int
        Canonical reference sector with a unique lowest state and both adjacent
        particle sectors available.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        The ground energy, real poles of shape ``(P,)``, and Hermitian
        positive-semidefinite residues of shape ``(P, 4, 4)`` for the complete
        zero-temperature doubled propagator.  Component order is the two
        physical impurity channels followed by their two auxiliary channels.
        Pole order and equivalent decompositions within an exactly degenerate
        pole subspace do not change the represented propagator.

    Raises
    ------
    ValueError
        If dimensions, finiteness, Hermiticity, occupation labels, particle
        sector, canonical algebra, or reference-state uniqueness violate the
        contract.

    Notes
    -----
    Input arrays are not required to be preserved by implementations.
    Pole groups with span at most ``1e-10`` are coalesced, and a group whose
    summed-residue trace is at most ``1e-12`` is numerically null and omitted.
    """
    return ground_energy, poles, residues

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_compute_augmented_matrix_lehmann_spectrum(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and nonunique-reference cases."""
    evaluator = """import numpy as np
def evaluate(builder, spectral, impurity, bath, coupling, interaction, number):
    h, hint, annihilators, occupations = builder(
        impurity.copy(), bath.copy(), coupling.copy(), interaction
    )
    ground, poles, residues = spectral(
        h, hint, annihilators, occupations, number
    )
    poles = np.asarray(poles, float)
    residues = np.asarray(residues, complex)
    if poles.ndim != 1 or residues.shape != (poles.size, 4, 4):
        raise AssertionError("augmented spectrum has the wrong shape")
    probes = (0.23+0.61j, -1.37+0.47j, 2.11+0.83j)
    values = [sum((w/(z-p) for p, w in zip(poles, residues)), np.zeros((4,4), complex))
              for z in probes]
    moment0 = np.sum(residues, axis=0) if poles.size else np.zeros((4,4), complex)
    moment1 = np.sum(poles[:,None,None]*residues, axis=0) if poles.size else np.zeros((4,4), complex)
    return np.concatenate((np.array([ground], complex), *(x.ravel() for x in values), moment0.ravel(), moment1.ravel()))
"""
    return [
        {
            "setup": evaluator + """
impurity = np.array([[-0.973779, 0.112032-0.097339j],
                     [0.112032+0.097339j, -1.300553]], complex)
bath = np.array([[[-1.084493, -0.148375-0.048758j],
                  [-0.148375+0.048758j, 1.029384]]], complex)
coupling = np.array([[[-0.049439-0.310296j, -0.118014+0.213365j],
                      [0.009953+0.122430j, -0.019628+0.285367j]]], complex)
interaction = 2.367874
number = 2
""",
            "call": "evaluate(build_noncommuting_impurity_model, compute_augmented_matrix_lehmann_spectrum, impurity, bath, coupling, interaction, number)",
            "gold_call": "evaluate(_oracle_build_noncommuting_impurity_model, _oracle_compute_augmented_matrix_lehmann_spectrum, impurity, bath, coupling, interaction, number)",
            "tol": 5e-10,
        },
        {
            "setup": evaluator + """
impurity = np.array([[-0.41, 0.09-0.04j], [0.09+0.04j, -0.18]], complex)
bath = np.array([[[-0.83, 0.05+0.02j], [0.05-0.02j, 0.62]]], complex)
coupling = np.array([[[0.2+0.03j, -0.08+0.04j], [0.06-0.07j, 0.17+0.02j]]], complex)
interaction = 0.0
number = 2
""",
            "call": "evaluate(build_noncommuting_impurity_model, compute_augmented_matrix_lehmann_spectrum, impurity, bath, coupling, interaction, number)",
            "gold_call": "evaluate(_oracle_build_noncommuting_impurity_model, _oracle_compute_augmented_matrix_lehmann_spectrum, impurity, bath, coupling, interaction, number)",
            "tol": 4e-10,
        },
        {
            "setup": evaluator + """
impurity = np.array([[-0.72, 0.16+0.09j], [0.16-0.09j, -0.31]], complex)
bath = np.array([[[-1.1, 0.12-0.03j], [0.12+0.03j, 0.45]],
                 [[-0.25, -0.08+0.11j], [-0.08-0.11j, 1.2]]], complex)
coupling = np.array([[[0.21+0.04j, -0.07+0.13j], [0.09-0.02j, 0.18+0.06j]],
                     [[-0.11+0.08j, 0.15-0.05j], [0.04+0.12j, -0.19+0.03j]]], complex)
interaction = 1.35
number = 3
""",
            "call": "evaluate(build_noncommuting_impurity_model, compute_augmented_matrix_lehmann_spectrum, impurity, bath, coupling, interaction, number)",
            "gold_call": "evaluate(_oracle_build_noncommuting_impurity_model, _oracle_compute_augmented_matrix_lehmann_spectrum, impurity, bath, coupling, interaction, number)",
            "tol": 8e-10,
        },
        {
            "setup": """import numpy as np
impurity = np.zeros((2,2), complex)
bath = np.empty((0,2,2), complex)
coupling = np.empty((0,2,2), complex)
def candidate():
    h, hint, annihilators, occupations = build_noncommuting_impurity_model(
        impurity.copy(), bath.copy(), coupling.copy(), 0.0
    )
    try:
        compute_augmented_matrix_lehmann_spectrum(h, hint, annihilators, occupations, 1)
        return np.array([0.0])
    except ValueError:
        return np.array([1.0])
def reference():
    h, hint, annihilators, occupations = _oracle_build_noncommuting_impurity_model(
        impurity.copy(), bath.copy(), coupling.copy(), 0.0
    )
    try:
        _oracle_compute_augmented_matrix_lehmann_spectrum(h, hint, annihilators, occupations, 1)
        return np.array([0.0])
    except ValueError:
        return np.array([1.0])
""",
            "call": "candidate()",
            "gold_call": "reference()",
        },
    ]
