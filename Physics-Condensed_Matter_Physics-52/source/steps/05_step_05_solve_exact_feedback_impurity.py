"""
Realize the feedback hybridization exactly and solve the resulting interacting impurity in a fixed particle sector.

A finite causal matrix spectrum can be represented by auxiliary bath orbitals without diagonalizing the impurity channels.  Canonical-sector exact diagonalization then provides a controlled second interacting solution while retaining the full complex orbital coherence.

Returns
-------
tuple[float, np.ndarray, np.ndarray], the feedback model's canonical ground energy, augmented real poles, and augmented matrix residues
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_exact_feedback_impurity(
    impurity_energy: "np.ndarray",
    hybridization_poles: "np.ndarray",
    hybridization_residues: "np.ndarray",
    interaction_u: float,
    particle_number: int,
) -> "tuple[float, np.ndarray, np.ndarray]":
    """Return the second ground energy and augmented Lehmann spectrum.

    Parameters
    ----------
    impurity_energy : np.ndarray
        Finite Hermitian array of shape ``(2, 2)``.
    hybridization_poles : np.ndarray
        Finite real poles of shape ``(R,)`` for an unreduced causal
        hybridization.
    hybridization_residues : np.ndarray
        Hermitian positive-semidefinite residues of shape ``(R, 2, 2)``.
        Equivalent rank decompositions at a degenerate pole describe the same
        bath.
    interaction_u : float
        Finite real coefficient of the impurity interaction ``n_0 n_1``.
    particle_number : int
        Canonical reference sector of the realized finite model, with a unique
        lowest state and both adjacent sectors available.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        The second canonical ground energy, real poles of shape ``(P,)``, and
        Hermitian positive-semidefinite residues of shape ``(P, 4, 4)`` for the
        complete doubled impurity propagator.  Components are ordered as the
        two physical channels followed by their two auxiliary channels.

    Raises
    ------
    ValueError
        If dimensions, finiteness, Hermiticity, positivity, reality, particle
        sector, or reference-state uniqueness violate the contract.

    Notes
    -----
    Input arrays are not required to be preserved by implementations.
    For each positive-semidefinite hybridization residue with eigenvalues
    ``r``, numerical rank retains
    ``r[k] > 2e-11 * max(1, max(abs(r)))``.  Output Lehmann pole groups with
    span at most ``1e-10`` are coalesced, and a group whose summed-residue
    trace is at most ``1e-12`` is numerically null and omitted.
    """
    return ground_energy, poles, residues

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_solve_exact_feedback_impurity(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return full-feedback, noninteracting, synthetic-rank, and invalid cases."""
    evaluator = """import numpy as np
def summarize(result):
    ground,poles,residues=result
    poles=np.asarray(poles,float); residues=np.asarray(residues,complex)
    if poles.ndim!=1 or residues.shape!=(poles.size,4,4):
        raise AssertionError("second augmented spectrum has the wrong shape")
    probes=(0.19+0.49j,-1.31+0.41j,1.97+0.87j)
    values=[sum((w/(z-p) for p,w in zip(poles,residues)),np.zeros((4,4),complex)) for z in probes]
    m0=np.sum(residues,axis=0) if poles.size else np.zeros((4,4),complex)
    m1=np.sum(poles[:,None,None]*residues,axis=0) if poles.size else np.zeros((4,4),complex)
    return np.concatenate((np.array([ground],complex),*(x.ravel() for x in values),m0.ravel(),m1.ravel()))
def pipeline(builder,lehmann,recover,update,solve,impurity,bath,coupling,interaction,first_number,second_number):
    h,hint,a,occ=builder(impurity.copy(),bath.copy(),coupling.copy(),interaction)
    _,ap,aw=lehmann(h,hint,a,occ,first_number)
    static,sp,sw=recover(ap,aw)
    dp,dw=update(bath.copy(),coupling.copy(),static,sp,sw)
    return summarize(solve(impurity.copy(),dp,dw,interaction,second_number))
def direct(solve,impurity,poles,residues,interaction,number):
    return summarize(solve(impurity.copy(),poles.copy(),residues.copy(),interaction,number))
"""
    return [
        {
            "setup": evaluator + """
impurity=np.array([[-0.973779,0.112032-0.097339j],[0.112032+0.097339j,-1.300553]],complex)
bath=np.array([[[-1.084493,-0.148375-0.048758j],[-0.148375+0.048758j,1.029384]]],complex)
coupling=np.array([[[-0.049439-0.310296j,-0.118014+0.213365j],[0.009953+0.122430j,-0.019628+0.285367j]]],complex)
interaction=2.367874
first_number=2
second_number=4
""",
            "call": "pipeline(build_noncommuting_impurity_model,compute_augmented_matrix_lehmann_spectrum,recover_causal_matrix_self_energy,update_exact_matrix_hybridization,solve_exact_feedback_impurity,impurity,bath,coupling,interaction,first_number,second_number)",
            "gold_call": "pipeline(_oracle_build_noncommuting_impurity_model,_oracle_compute_augmented_matrix_lehmann_spectrum,_oracle_recover_causal_matrix_self_energy,_oracle_update_exact_matrix_hybridization,_oracle_solve_exact_feedback_impurity,impurity,bath,coupling,interaction,first_number,second_number)",
            "tol": 9e-9,
        },
        {
            "setup": evaluator + """
impurity=np.array([[-0.4,0.13-0.08j],[0.13+0.08j,-0.2]],complex)
bath=np.array([[[-0.9,0.11+0.03j],[0.11-0.03j,0.65]]],complex)
coupling=np.array([[[0.22+0.04j,-0.07+0.1j],[0.05-0.09j,0.19-0.02j]]],complex)
interaction=0.0
first_number=2
second_number=2
""",
            "call": "pipeline(build_noncommuting_impurity_model,compute_augmented_matrix_lehmann_spectrum,recover_causal_matrix_self_energy,update_exact_matrix_hybridization,solve_exact_feedback_impurity,impurity,bath,coupling,interaction,first_number,second_number)",
            "gold_call": "pipeline(_oracle_build_noncommuting_impurity_model,_oracle_compute_augmented_matrix_lehmann_spectrum,_oracle_recover_causal_matrix_self_energy,_oracle_update_exact_matrix_hybridization,_oracle_solve_exact_feedback_impurity,impurity,bath,coupling,interaction,first_number,second_number)",
            "tol": 8e-10,
        },
        {
            "setup": evaluator + """
impurity=np.array([[-0.63,0.14+0.06j],[0.14-0.06j,-0.27]],complex)
poles=np.array([-0.8,0.15,0.95])
r0=np.array([0.23+0.02j,-0.09+0.11j]); r1=np.array([0.08-0.17j,0.19+0.05j])
w2=np.array([[0.07,0.018-0.009j],[0.018+0.009j,0.05]],complex)
residues=np.array([np.outer(r0,r0.conj()),np.outer(r1,r1.conj()),w2])
interaction=1.1
number=2
""",
            "call": "direct(solve_exact_feedback_impurity,impurity,poles,residues,interaction,number)",
            "gold_call": "direct(_oracle_solve_exact_feedback_impurity,impurity,poles,residues,interaction,number)",
            "tol": 3e-9,
        },
        {
            "setup": """import numpy as np
impurity=np.diag([-0.4,-0.2]).astype(complex)
poles=np.array([-0.7,0.8]); residues=np.array([0.1*np.eye(2),0.2*np.eye(2)],complex)
def candidate():
    try:
        solve_exact_feedback_impurity(impurity.copy(),poles.copy(),residues.copy(),1.0,7)
        return np.array([0.0])
    except ValueError:
        return np.array([1.0])
def reference():
    try:
        _oracle_solve_exact_feedback_impurity(impurity.copy(),poles.copy(),residues.copy(),1.0,7)
        return np.array([0.0])
    except ValueError:
        return np.array([1.0])
""",
            "call": "candidate()",
            "gold_call": "reference()",
        },
    ]
