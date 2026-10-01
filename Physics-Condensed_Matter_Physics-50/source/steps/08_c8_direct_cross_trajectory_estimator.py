"""
Evaluate independently paired bra/ket trajectory observables and their direct unreweighted finite-ensemble mean.

For each sample index (m), the complex cross value is \(z_m=\langle\psi^{\mathrm{bra}}_m|O|\psi^{\mathrm{ket}}_m\rangle\).  Bra and ket trajectories remain independently selected and unnormalized.  The direct estimator is the arithmetic mean \(M^{-1}\sum_m z_m\), with no norm ratio or Markov-chain reweighting.

Returns
-------
Return `(cross_samples, direct_mean)`, where the samples are complex128 shape `(M,)` and the mean is one complex scalar formed without reweighting.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_direct_cross_estimator(
    bra_states: "np.ndarray",
    ket_states: "np.ndarray",
    observable: "np.ndarray",
) -> "tuple[np.ndarray, complex]":
    """Return the complex sample vector and its direct complex mean.

    The state arrays have matching shape ``(M,64)`` and the observable is a
    finite Hermitian ``(64,64)`` matrix.  The returned shapes are ``(M,)`` and
    scalar complex, respectively.  Raise ``ValueError`` for mismatched,
    nonfinite, empty, or non-Hermitian inputs.
    """
    return cross_samples, direct_mean

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_direct_cross_estimator(
    bra_states: "np.ndarray",
    ket_states: "np.ndarray",
    observable: "np.ndarray",
) -> "tuple[np.ndarray, complex]":
    bra = np.asarray(bra_states, dtype=np.complex128)
    ket = np.asarray(ket_states, dtype=np.complex128)
    operator = np.asarray(observable, dtype=np.complex128)
    if bra.ndim != 2 or bra.shape[1] != 64 or ket.shape != bra.shape or bra.shape[0] == 0:
        raise ValueError("bra and ket ensembles must have matching nonempty shape (M,64)")
    if operator.shape != (64, 64):
        raise ValueError("observable must have shape (64,64)")
    if any(not np.all(np.isfinite(value)) for value in (bra, ket, operator)):
        raise ValueError("all inputs must be finite")
    if np.linalg.norm(operator - operator.conj().T) > 1.0e-12:
        raise ValueError("observable must be Hermitian")
    samples = np.einsum("mi,ij,mj->m", bra.conj(), operator, ket)
    mean = complex(np.mean(samples, dtype=np.complex128))
    return np.asarray(samples, dtype=np.complex128), mean

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    pack = "import numpy as np\ndef _pack(value):\n    parts=[]\n    for item in value:\n        arr=np.asarray(item); parts.append(np.asarray([arr.ndim,*arr.shape],dtype=np.complex128)); parts.append(arr.astype(np.complex128).ravel())\n    return np.concatenate(parts)\n"
    return [
        {"setup": pack + "bra_states=(np.arange(192).reshape(3,64)/31+1j*np.arange(192,0,-1).reshape(3,64)/47).astype(np.complex128)\nket_states=(np.arange(192,384).reshape(3,64)/53-1j*np.arange(64,256).reshape(3,64)/59).astype(np.complex128)\nobservable=np.diag(np.linspace(-.7,1.1,64)).astype(np.complex128)\nobservable[3,41]=.2+.17j; observable[41,3]=.2-.17j", "call": "_pack(compute_direct_cross_estimator(bra_states,ket_states,observable))", "gold_call": "_pack(_oracle_compute_direct_cross_estimator(bra_states,ket_states,observable))", "tol": 1e-10},
        {"setup": pack + "bra_states=np.zeros((1,64),dtype=np.complex128); ket_states=np.zeros((1,64),dtype=np.complex128)\nbra_states[0,5]=1+2j; ket_states[0,11]=-3+.5j\nobservable=np.zeros((64,64),dtype=np.complex128); observable[5,11]=.4-.7j; observable[11,5]=.4+.7j", "call": "_pack(compute_direct_cross_estimator(bra_states,ket_states,observable))", "gold_call": "_pack(_oracle_compute_direct_cross_estimator(bra_states,ket_states,observable))", "tol": 1e-12},
        {"setup": pack + "bra_states=np.eye(4,64,dtype=np.complex128); ket_states=np.roll(bra_states,7,axis=1)*(1+1j)\nobservable=np.eye(64,dtype=np.complex128); observable[0,7]=.3+.8j; observable[7,0]=.3-.8j; observable[1,8]=-.2+.4j; observable[8,1]=-.2-.4j", "call": "_pack(compute_direct_cross_estimator(bra_states,ket_states,observable))", "gold_call": "_pack(_oracle_compute_direct_cross_estimator(bra_states,ket_states,observable))", "tol": 1e-12},
        {"setup": "import numpy as np\nbra_states=np.ones((2,64),dtype=np.complex128); ket_states=bra_states.copy(); observable=np.eye(64,dtype=np.complex128); observable[0,1]=1j\ndef candidate_result():\n    try:\n        compute_direct_cross_estimator(bra_states,ket_states,observable)\n        return 0\n    except ValueError:\n        return 1\ndef reference_result():\n    try:\n        _oracle_compute_direct_cross_estimator(bra_states,ket_states,observable)\n        return 0\n    except ValueError:\n        return 1", "call": "candidate_result()", "gold_call": "reference_result()"},
    ]
