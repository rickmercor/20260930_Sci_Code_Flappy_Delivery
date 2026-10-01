"""
Compose the bounded two-solve matrix feedback calculation into one deterministic projected quasiparticle weight.

For coherently mixed impurity channels, the low-energy response is a Hermitian matrix rather than a collection of independent scalar weights.  A normalized complex probe selects a reproducible scalar while retaining sensitivity to both diagonal response and phase-dependent orbital coherence.

Returns
-------
float, the probe-projected second-solve regularized response rounded once to ten digits after the decimal point
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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
    """Return the rounded probe projection after one exact feedback update.

    Parameters
    ----------
    impurity_energy : np.ndarray
        Finite Hermitian impurity matrix of shape ``(2, 2)``.
    bath_energies : np.ndarray
        Finite Hermitian bath blocks of shape ``(M, 2, 2)``.
    bath_couplings : np.ndarray
        Finite coupling blocks of matching shape.
    interaction_u : float
        Finite real coefficient of the impurity interaction ``n_0 n_1``.
    first_particle_number : int
        Canonical particle number for the initial interacting solve.
    second_particle_number : int
        Canonical particle number for the exact feedback solve.
    regulator : float
        Positive finite energy regulator for the second low-energy response.
    probe : np.ndarray
        Finite normalized complex vector of shape ``(2,)``.

    Returns
    -------
    float
        The real scalar ``probe^H Z probe`` from the second regularized response
        matrix, rounded once to ten digits after completing the two-solve
        calculation.

    Raises
    ------
    ValueError
        If the probe is not finite and normalized, or if an upstream contract
        required by the supplied inputs is violated.

    Notes
    -----
    Input arrays are not required to be preserved by implementations.
    """
    return projected_weight

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bounded_matrix_two_solve_weight(
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
        _oracle_build_noncommuting_impurity_model(
            impurity_energy, bath_energies, bath_couplings, interaction_u
        )
    )
    _, augmented_poles, augmented_residues = (
        _oracle_compute_augmented_matrix_lehmann_spectrum(
            hamiltonian,
            interaction,
            annihilators,
            occupations,
            first_particle_number,
        )
    )
    static, self_energy_poles, self_energy_residues = (
        _oracle_recover_causal_matrix_self_energy(
            augmented_poles, augmented_residues
        )
    )
    feedback_poles, feedback_residues = (
        _oracle_update_exact_matrix_hybridization(
            bath_energies,
            bath_couplings,
            static,
            self_energy_poles,
            self_energy_residues,
        )
    )
    _, second_poles, second_residues = _oracle_solve_exact_feedback_impurity(
        impurity_energy,
        feedback_poles,
        feedback_residues,
        interaction_u,
        second_particle_number,
    )
    response, _ = _oracle_certify_regularized_matrix_response(
        second_poles, second_residues, regulator
    )
    projected = np.vdot(probe, response @ probe)
    if abs(float(np.imag(projected))) > 2e-10:
        raise ValueError("the projected response is not real")
    return float(np.round(float(np.real(projected)), 10))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return benchmark, boundary, alternate, and invalid-probe cases."""
    return [
        {
            "setup": """import numpy as np
impurity=np.array([[-0.973779,0.112032-0.097339j],[0.112032+0.097339j,-1.300553]],complex)
bath=np.array([[[-1.084493,-0.148375-0.048758j],[-0.148375+0.048758j,1.029384]]],complex)
coupling=np.array([[[-0.049439-0.310296j,-0.118014+0.213365j],[0.009953+0.122430j,-0.019628+0.285367j]]],complex)
probe=np.array([np.sqrt(0.6),np.exp(0.73j)*np.sqrt(0.4)])
""",
            "call": "bounded_matrix_two_solve_weight(impurity.copy(),bath.copy(),coupling.copy(),2.367874,2,4,0.04,probe.copy())",
            "gold_call": "_oracle_bounded_matrix_two_solve_weight(impurity.copy(),bath.copy(),coupling.copy(),2.367874,2,4,0.04,probe.copy())",
            "tol": 5e-11,
        },
        {
            "setup": """import numpy as np
impurity=np.array([[-0.4,0.13-0.08j],[0.13+0.08j,-0.2]],complex)
bath=np.array([[[-0.9,0.11+0.03j],[0.11-0.03j,0.65]]],complex)
coupling=np.array([[[0.22+0.04j,-0.07+0.1j],[0.05-0.09j,0.19-0.02j]]],complex)
probe=np.array([np.sqrt(0.5),1j*np.sqrt(0.5)])
""",
            "call": "bounded_matrix_two_solve_weight(impurity.copy(),bath.copy(),coupling.copy(),0.0,2,2,0.04,probe.copy())",
            "gold_call": "_oracle_bounded_matrix_two_solve_weight(impurity.copy(),bath.copy(),coupling.copy(),0.0,2,2,0.04,probe.copy())",
            "tol": 5e-12,
        },
        {
            "setup": """import numpy as np
impurity=np.array([[-0.4,0.25+0.13j],[0.25-0.13j,-1.0]],complex)
bath=np.array([[[-0.8,0.2-0.15j],[0.2+0.15j,0.9]]],complex)
coupling=np.array([[[0.35+0.12j,-0.21+0.18j],[0.14-0.1j,0.32+0.08j]]],complex)
probe=np.array([np.sqrt(0.45),np.exp(0.9j)*np.sqrt(0.55)])
""",
            "call": "bounded_matrix_two_solve_weight(impurity.copy(),bath.copy(),coupling.copy(),3.2,2,4,0.05,probe.copy())",
            "gold_call": "_oracle_bounded_matrix_two_solve_weight(impurity.copy(),bath.copy(),coupling.copy(),3.2,2,4,0.05,probe.copy())",
            "tol": 5e-11,
        },
        {
            "setup": """import numpy as np
impurity=np.diag([-0.4,-0.2]).astype(complex)
bath=np.array([np.diag([-0.8,0.7])],complex)
coupling=np.array([0.2*np.eye(2)],complex)
probe=np.array([1.0,1.0],complex)
def candidate():
    try:
        bounded_matrix_two_solve_weight(impurity.copy(),bath.copy(),coupling.copy(),1.0,2,2,0.04,probe.copy())
        return np.array([0.0])
    except ValueError:
        return np.array([1.0])
def reference():
    try:
        _oracle_bounded_matrix_two_solve_weight(impurity.copy(),bath.copy(),coupling.copy(),1.0,2,2,0.04,probe.copy())
        return np.array([0.0])
    except ValueError:
        return np.array([1.0])
""",
            "call": "candidate()",
            "gold_call": "reference()",
        },
    ]
