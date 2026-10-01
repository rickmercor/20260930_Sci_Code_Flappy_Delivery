"""
Apply the exact discrete matrix feedback map to a bath and a causal self-energy spectrum.

In a matrix-valued impurity problem, bath levels and self-energy residues generally do not share an eigenbasis.  An exact feedback spectrum must therefore retain the coupled orbital structure and its positive matrix weights rather than update separate diagonal channels.

Returns
-------
tuple[np.ndarray, np.ndarray], the exact feedback hybridization poles and matrix residues
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def update_exact_matrix_hybridization(
    bath_energies: "np.ndarray",
    bath_couplings: "np.ndarray",
    self_energy_static: "np.ndarray",
    self_energy_poles: "np.ndarray",
    self_energy_residues: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    """Return the unreduced causal spectrum of the feedback hybridization.

    Parameters
    ----------
    bath_energies : np.ndarray
        Finite Hermitian blocks of shape ``(M, N, N)``.
    bath_couplings : np.ndarray
        Finite coupling blocks of matching shape in the convention where the
        original hybridization is represented by these bath blocks.
    self_energy_static : np.ndarray
        Finite Hermitian static matrix of shape ``(N, N)`` in the same fixed
        orbital basis as the bath blocks.
    self_energy_poles : np.ndarray
        Finite real dynamical poles of shape ``(Q,)``.
    self_energy_residues : np.ndarray
        Hermitian positive-semidefinite residues of shape ``(Q, N, N)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Real poles of shape ``(R,)`` and Hermitian positive-semidefinite
        residues of shape ``(R, N, N)`` for the exact discrete feedback
        hybridization.  No approximate pole reduction is applied.  Pole order
        and equivalent decompositions inside exactly degenerate subspaces do
        not change the represented function.

    Raises
    ------
    ValueError
        If dimensions, finiteness, reality, Hermiticity, or positivity violate
        the contract.

    Notes
    -----
    Input arrays are not required to be preserved by implementations.
    For each positive-semidefinite residue with eigenvalues ``r``, numerical
    rank retains ``r[k] > 2e-11 * max(1, max(abs(r)))``.  A resulting pole
    group whose summed-residue trace is at most ``2e-11`` is numerically null
    and omitted.
    """
    return hybridization_poles, hybridization_residues

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_update_exact_matrix_hybridization(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return interacting, noninteracting, noncommuting, empty, and invalid cases."""
    evaluator = """import numpy as np
def summarize(poles, residues):
    poles=np.asarray(poles,float); residues=np.asarray(residues,complex)
    n=residues.shape[1] if residues.ndim==3 else 0
    if poles.ndim!=1 or residues.shape!=(poles.size,n,n) or n<1:
        raise AssertionError("hybridization spectrum has the wrong shape")
    if residues.size and min(float(np.min(np.linalg.eigvalsh((w+w.conj().T)/2))) for w in residues)<-2e-8:
        raise AssertionError("hybridization spectrum is not causal")
    probes=(0.31+0.57j,-1.23+0.43j,2.07+0.91j)
    values=[sum((w/(z-p) for p,w in zip(poles,residues)),np.zeros((n,n),complex)) for z in probes]
    m0=np.sum(residues,axis=0) if poles.size else np.zeros((n,n),complex)
    m1=np.sum(poles[:,None,None]*residues,axis=0) if poles.size else np.zeros((n,n),complex)
    return np.concatenate((*(x.ravel() for x in values),m0.ravel(),m1.ravel()))
def pipeline(builder,lehmann,recover,update,impurity,bath,coupling,interaction,number):
    h,hint,a,occ=builder(impurity.copy(),bath.copy(),coupling.copy(),interaction)
    _,ap,aw=lehmann(h,hint,a,occ,number)
    static,sp,sw=recover(ap,aw)
    return summarize(*update(bath.copy(),coupling.copy(),static,sp,sw))
def direct(update,bath,coupling,static,poles,residues):
    return summarize(*update(bath.copy(),coupling.copy(),static.copy(),poles.copy(),residues.copy()))
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
            "call": "pipeline(build_noncommuting_impurity_model, compute_augmented_matrix_lehmann_spectrum, recover_causal_matrix_self_energy, update_exact_matrix_hybridization, impurity, bath, coupling, interaction, number)",
            "gold_call": "pipeline(_oracle_build_noncommuting_impurity_model, _oracle_compute_augmented_matrix_lehmann_spectrum, _oracle_recover_causal_matrix_self_energy, _oracle_update_exact_matrix_hybridization, impurity, bath, coupling, interaction, number)",
            "tol": 7e-9,
        },
        {
            "setup": evaluator + """
impurity=np.array([[-0.41,0.09-0.04j],[0.09+0.04j,-0.18]],complex)
bath=np.array([[[-0.83,0.05+0.02j],[0.05-0.02j,0.62]]],complex)
coupling=np.array([[[0.2+0.03j,-0.08+0.04j],[0.06-0.07j,0.17+0.02j]]],complex)
interaction=0.0
number=2
""",
            "call": "pipeline(build_noncommuting_impurity_model, compute_augmented_matrix_lehmann_spectrum, recover_causal_matrix_self_energy, update_exact_matrix_hybridization, impurity, bath, coupling, interaction, number)",
            "gold_call": "pipeline(_oracle_build_noncommuting_impurity_model, _oracle_compute_augmented_matrix_lehmann_spectrum, _oracle_recover_causal_matrix_self_energy, _oracle_update_exact_matrix_hybridization, impurity, bath, coupling, interaction, number)",
            "tol": 8e-10,
        },
        {
            "setup": evaluator + """
bath=np.array([[[-1.0,0.17-0.09j],[0.17+0.09j,0.65]],
               [[-0.22,-0.11+0.04j],[-0.11-0.04j,1.08]]],complex)
coupling=np.array([[[0.24+0.03j,-0.08+0.12j],[0.05-0.06j,0.19+0.01j]],
                   [[-0.13+0.07j,0.16-0.02j],[0.09+0.08j,-0.15+0.05j]]],complex)
static=np.array([[0.38,0.09-0.14j],[0.09+0.14j,0.21]],complex)
poles=np.array([-0.7,0.9])
r0=np.array([0.22+0.03j,-0.11+0.17j]); r1=np.array([0.08-0.19j,0.2+0.04j])
residues=np.array([np.outer(r0,r0.conj()),np.outer(r1,r1.conj())])
""",
            "call": "direct(update_exact_matrix_hybridization,bath,coupling,static,poles,residues)",
            "gold_call": "direct(_oracle_update_exact_matrix_hybridization,bath,coupling,static,poles,residues)",
            "tol": 3e-10,
        },
        {
            "setup": evaluator + """
bath=np.empty((0,2,2),complex); coupling=np.empty((0,2,2),complex)
static=np.array([[0.3,0.07-0.02j],[0.07+0.02j,0.1]],complex)
poles=np.array([-0.5,0.8])
residues=np.array([0.04*np.eye(2),0.03*np.eye(2)],complex)
""",
            "call": "direct(update_exact_matrix_hybridization,bath,coupling,static,poles,residues)",
            "gold_call": "direct(_oracle_update_exact_matrix_hybridization,bath,coupling,static,poles,residues)",
            "tol": 2e-13,
        },
        {
            "setup": """import numpy as np
bath=np.array([np.eye(2)],complex); coupling=np.array([np.eye(2)],complex)
static=np.zeros((2,2),complex); poles=np.array([0.2])
residues=np.array([[[0.1,0.0],[0.0,-0.05]]],complex)
def candidate():
    try:
        update_exact_matrix_hybridization(bath.copy(),coupling.copy(),static.copy(),poles.copy(),residues.copy())
        return np.array([0.0])
    except ValueError:
        return np.array([1.0])
def reference():
    try:
        _oracle_update_exact_matrix_hybridization(bath.copy(),coupling.copy(),static.copy(),poles.copy(),residues.copy())
        return np.array([0.0])
    except ValueError:
        return np.array([1.0])
""",
            "call": "candidate()",
            "gold_call": "reference()",
        },
    ]
