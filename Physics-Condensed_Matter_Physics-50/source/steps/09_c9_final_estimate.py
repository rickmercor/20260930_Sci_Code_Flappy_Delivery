"""
Compose the complete six-site stochastic-projector pipeline and report the real part of the direct paired-trajectory estimate.

The computation canonicalizes the supplied MPS, constructs the chronological operator-Schmidt factors and deterministic reference projector bank, propagates the fixed independent ket and bra subspace histories without normalization, and forms the direct complex cross-trajectory mean.  The real part is rounded once, at the end, to ten decimal places.

Returns
-------
Return one Python `float`: the real part of the direct complex mean rounded once to exactly 10 decimal places.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_final_estimate(
    site_tensors: "tuple[np.ndarray, ...]",
    bond_indices: "np.ndarray",
    pauli_axes: "np.ndarray",
    angles: "np.ndarray",
    ket_histories: "np.ndarray",
    bra_histories: "np.ndarray",
    observable: "np.ndarray",
) -> float:
    """Return the frozen dimensionless direct estimate rounded to 10 decimals.

    Histories must have matching shape ``(M,L,2)`` and correspond to the ``L``
    chronological gate events.  The retained subspace dimension is fixed at two.
    """
    return final_estimate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_final_estimate(
    site_tensors: "tuple[np.ndarray, ...]",
    bond_indices: "np.ndarray",
    pauli_axes: "np.ndarray",
    angles: "np.ndarray",
    ket_histories: "np.ndarray",
    bra_histories: "np.ndarray",
    observable: "np.ndarray",
) -> float:
    ket_choices = np.asarray(ket_histories)
    bra_choices = np.asarray(bra_histories)
    if ket_choices.shape != bra_choices.shape or ket_choices.ndim != 3 or ket_choices.shape[0] == 0:
        raise ValueError("ket and bra histories must have matching nonempty shape (M,L,2)")
    canonical, spectra = _oracle_canonicalize_initial_mps(site_tensors)
    gate_left, gate_right = _oracle_build_operator_schmidt_factors(pauli_axes, angles)
    ql_bank, qr_bank, _, inclusion = _oracle_build_reference_projector_bank(
        canonical, spectra, bond_indices, gate_left, gate_right, 2
    )
    ket_states = _oracle_propagate_weighted_histories(
        canonical, bond_indices, gate_left, gate_right, ql_bank, qr_bank, inclusion, ket_choices
    )
    bra_states = _oracle_propagate_weighted_histories(
        canonical, bond_indices, gate_left, gate_right, ql_bank, qr_bank, inclusion, bra_choices
    )
    _, direct_mean = _oracle_compute_direct_cross_estimator(bra_states, ket_states, observable)
    return float(np.round(direct_mean.real, 10))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base = '''import numpy as np
dims=(1,2,2,2,2,2,1); raw=[]
for n in range(6):
    t=np.empty((2,dims[n],dims[n+1]),dtype=np.complex128)
    for sig in range(2):
        for a in range(dims[n]):
            for b in range(dims[n+1]):
                z=11*(n+1)+7*(sig+1)+5*(a+1)+3*(b+1)+(n+1)*(sig+1)*(a+b+2)
                t[sig,a,b]=np.sin(.37*z)+.31*np.cos(.19*(z*z+3*a+5*b))+1j*(np.cos(.29*(z+2*a*b))+.23*np.sin(.41*(z*z+sig+b)))
    raw.append(t)
site_tensors=tuple(raw)
bond_indices=np.array([2,4,3,2,4,3],dtype=np.int64)
pauli_axes=np.array([0,1,2,1,2,0],dtype=np.int64)
angles=np.array([.31,.27,.23,.41,.35,.29])
ket_histories=np.array([[[0,1],[0,2],[1,2],[0,1],[0,3],[1,3]],[[0,2],[1,2],[0,1],[0,3],[1,3],[0,1]],[[1,2],[0,1],[0,3],[1,3],[0,1],[0,2]],[[0,1],[0,3],[1,3],[0,1],[0,2],[1,2]],[[0,3],[1,3],[0,1],[0,2],[1,2],[0,1]],[[1,3],[0,1],[0,2],[1,2],[0,1],[0,3]]],dtype=np.int64)
bra_histories=np.array([[[0,1],[1,2],[1,3],[0,1],[1,2],[1,3]],[[0,3],[0,1],[0,2],[0,3],[0,1],[0,2]],[[1,2],[1,3],[0,1],[1,2],[1,3],[0,1]],[[0,1],[0,2],[0,3],[0,1],[0,2],[0,3]],[[1,3],[0,1],[1,2],[1,3],[0,1],[1,2]],[[0,2],[0,3],[0,1],[0,2],[0,3],[0,1]]],dtype=np.int64)
alphas=np.array([.41,.73,1.02,.58,.91,.36]); phis=np.array([.17,-.29,.43,.61,-.37,.52])
eta=np.array([1.+0j])
for a,p in zip(alphas,phis): eta=np.kron(eta,np.array([np.cos(a/2),np.exp(1j*p)*np.sin(a/2)]))
X=np.array([[0,1],[1,0]],dtype=np.complex128); Y=np.array([[0,-1j],[1j,0]],dtype=np.complex128); Z=np.diag([1,-1]).astype(np.complex128); I=np.eye(2,dtype=np.complex128)
def kron6(ops):
    out=np.array([1.+0j])
    for op in ops: out=np.kron(out,op)
    return out
observable=.70*np.outer(eta,eta.conj())+.20*kron6((X,I,Y,Z,I,X))-.15*kron6((I,Y,X,I,Z,Y))
'''
    args = "site_tensors,bond_indices,pauli_axes,angles,ket_histories,bra_histories,observable"
    return [
        {"setup": base, "call": f"compute_final_estimate({args})", "gold_call": f"_oracle_compute_final_estimate({args})", "tol": 5e-10},
        {"setup": base + "angles=np.array([.19,.46,.32,.28,.51,.22])", "call": f"compute_final_estimate({args})", "gold_call": f"_oracle_compute_final_estimate({args})", "tol": 5e-10},
        {"setup": base + "ket_histories=bra_histories.copy(); bra_histories=np.array(ket_histories[::-1],copy=True)", "call": f"compute_final_estimate({args})", "gold_call": f"_oracle_compute_final_estimate({args})", "tol": 5e-10},
    ]
