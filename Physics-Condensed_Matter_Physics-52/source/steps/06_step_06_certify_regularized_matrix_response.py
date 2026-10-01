"""
Recover and certify the regularized low-energy matrix response of the second interacting solution.

Low-energy quasiparticle response is matrix valued when orbital channels mix.  A finite regulator controls the influence of poles close to the Fermi level, while spectral moments and eigenvalue bounds provide basis-independent checks of causality and physical scale.

Returns
-------
tuple[np.ndarray, np.ndarray], the regularized response matrix and five-entry spectral certificate
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def certify_regularized_matrix_response(
    augmented_poles: "np.ndarray",
    augmented_residues: "np.ndarray",
    regulator: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Return the regularized response matrix and its spectral certificate.

    Parameters
    ----------
    augmented_poles : np.ndarray
        Finite real poles of shape ``(P,)`` for the second complete doubled
        propagator.
    augmented_residues : np.ndarray
        Hermitian positive-semidefinite residues of shape ``(P, 2*N, 2*N)``,
        ordered with physical channels first and matching auxiliary channels
        second.
    regulator : float
        Positive finite energy regulator used by the source-defined
        low-energy response.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The Hermitian regularized response matrix of shape ``(N, N)`` and a
        five-entry real certificate containing, in order, the traces of the
        zeroth and first dynamical self-energy moments, the minimum residue
        eigenvalue, and the minimum and maximum response eigenvalues.  The
        minimum residue eigenvalue is zero for an empty dynamical spectrum.

    Raises
    ------
    ValueError
        If ``regulator`` is not positive and finite, if the augmented spectrum
        violates the preceding self-energy contract, or if the recovered
        response is nonfinite or non-Hermitian.

    Notes
    -----
    Input arrays are not required to be preserved by implementations.
    """
    return response_matrix, certificate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_certify_regularized_matrix_response(
    augmented_poles: "np.ndarray",
    augmented_residues: "np.ndarray",
    regulator: float,
) -> "tuple[np.ndarray, np.ndarray]":
    if (not np.isscalar(regulator) or not np.isfinite(regulator)
            or abs(complex(regulator).imag) > 0.0
            or float(np.real(regulator)) <= 0.0):
        raise ValueError("regulator must be positive, finite, and real")
    static, poles, residues = _oracle_recover_causal_matrix_self_energy(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return interacting, noninteracting, alternate, and invalid-regulator cases."""
    evaluator = """import numpy as np
def pack(result):
    response,certificate=result
    response=np.asarray(response,complex); certificate=np.asarray(certificate,float)
    if response.shape!=(2,2) or certificate.shape!=(5,):
        raise AssertionError("regularized response output has the wrong shape")
    return np.concatenate((response.ravel(),certificate.astype(complex)))
def pipeline(builder,lehmann,recover,update,solve,certify,impurity,bath,coupling,interaction,first_number,second_number,regulator):
    h,hint,a,occ=builder(impurity.copy(),bath.copy(),coupling.copy(),interaction)
    _,ap,aw=lehmann(h,hint,a,occ,first_number)
    static,sp,sw=recover(ap,aw)
    dp,dw=update(bath.copy(),coupling.copy(),static,sp,sw)
    _,ap2,aw2=solve(impurity.copy(),dp,dw,interaction,second_number)
    return pack(certify(ap2,aw2,regulator))
"""
    return [
        {
            "setup": evaluator + """
impurity=np.array([[-0.973779,0.112032-0.097339j],[0.112032+0.097339j,-1.300553]],complex)
bath=np.array([[[-1.084493,-0.148375-0.048758j],[-0.148375+0.048758j,1.029384]]],complex)
coupling=np.array([[[-0.049439-0.310296j,-0.118014+0.213365j],[0.009953+0.122430j,-0.019628+0.285367j]]],complex)
interaction=2.367874; first_number=2; second_number=4; regulator=0.04
""",
            "call": "pipeline(build_noncommuting_impurity_model,compute_augmented_matrix_lehmann_spectrum,recover_causal_matrix_self_energy,update_exact_matrix_hybridization,solve_exact_feedback_impurity,certify_regularized_matrix_response,impurity,bath,coupling,interaction,first_number,second_number,regulator)",
            "gold_call": "pipeline(_oracle_build_noncommuting_impurity_model,_oracle_compute_augmented_matrix_lehmann_spectrum,_oracle_recover_causal_matrix_self_energy,_oracle_update_exact_matrix_hybridization,_oracle_solve_exact_feedback_impurity,_oracle_certify_regularized_matrix_response,impurity,bath,coupling,interaction,first_number,second_number,regulator)",
            "tol": 9e-9,
        },
        {
            "setup": evaluator + """
impurity=np.array([[-0.4,0.13-0.08j],[0.13+0.08j,-0.2]],complex)
bath=np.array([[[-0.9,0.11+0.03j],[0.11-0.03j,0.65]]],complex)
coupling=np.array([[[0.22+0.04j,-0.07+0.1j],[0.05-0.09j,0.19-0.02j]]],complex)
interaction=0.0; first_number=2; second_number=2; regulator=0.04
""",
            "call": "pipeline(build_noncommuting_impurity_model,compute_augmented_matrix_lehmann_spectrum,recover_causal_matrix_self_energy,update_exact_matrix_hybridization,solve_exact_feedback_impurity,certify_regularized_matrix_response,impurity,bath,coupling,interaction,first_number,second_number,regulator)",
            "gold_call": "pipeline(_oracle_build_noncommuting_impurity_model,_oracle_compute_augmented_matrix_lehmann_spectrum,_oracle_recover_causal_matrix_self_energy,_oracle_update_exact_matrix_hybridization,_oracle_solve_exact_feedback_impurity,_oracle_certify_regularized_matrix_response,impurity,bath,coupling,interaction,first_number,second_number,regulator)",
            "tol": 8e-10,
        },
        {
            "setup": evaluator + """
impurity=np.array([[-0.4,0.25+0.13j],[0.25-0.13j,-1.0]],complex)
bath=np.array([[[-0.8,0.2-0.15j],[0.2+0.15j,0.9]]],complex)
coupling=np.array([[[0.35+0.12j,-0.21+0.18j],[0.14-0.1j,0.32+0.08j]]],complex)
interaction=3.2; first_number=2; second_number=4; regulator=0.05
""",
            "call": "pipeline(build_noncommuting_impurity_model,compute_augmented_matrix_lehmann_spectrum,recover_causal_matrix_self_energy,update_exact_matrix_hybridization,solve_exact_feedback_impurity,certify_regularized_matrix_response,impurity,bath,coupling,interaction,first_number,second_number,regulator)",
            "gold_call": "pipeline(_oracle_build_noncommuting_impurity_model,_oracle_compute_augmented_matrix_lehmann_spectrum,_oracle_recover_causal_matrix_self_energy,_oracle_update_exact_matrix_hybridization,_oracle_solve_exact_feedback_impurity,_oracle_certify_regularized_matrix_response,impurity,bath,coupling,interaction,first_number,second_number,regulator)",
            "tol": 1e-8,
        },
        {
            "setup": """import numpy as np
poles=np.array([-0.5,0.6]); residues=np.zeros((2,4,4),complex)
residues[:,0,0]=0.5; residues[:,1,1]=0.5
def candidate():
    try:
        certify_regularized_matrix_response(poles.copy(),residues.copy(),0.0)
        return np.array([0.0])
    except ValueError:
        return np.array([1.0])
def reference():
    try:
        _oracle_certify_regularized_matrix_response(poles.copy(),residues.copy(),0.0)
        return np.array([0.0])
    except ValueError:
        return np.array([1.0])
""",
            "call": "candidate()",
            "gold_call": "reference()",
        },
    ]
