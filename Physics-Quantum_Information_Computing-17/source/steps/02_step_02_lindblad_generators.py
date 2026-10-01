"""
Return separate coherent and common dissipative superoperators for the supplied Markovian pulse family.

Use row-major vectorization in the given basis. The global master equation defines the physical model. The two returned parts act on vectorized density matrices and retain the supplied chronological pulse order; they are generators rather than finite-duration channels.

Returns
-------
tuple    Coherent stack (P,d*d,d*d) and dissipator (d*d,d*d).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lindblad_generators(hamiltonians: "np.ndarray", jumps: "np.ndarray", rates: "np.ndarray") -> tuple:
    """Build row-major coherent generators and their shared dissipator.

    Parameters
    ----------
    hamiltonians : array_like
        Hermitian stack (P,d,d) of finite bounded Hamiltonians.
    jumps : array_like
        Finite bounded jump stack (J,d,d), including an empty stack.
    rates : array_like
        Real nonnegative rates (J,) not exceeding 100.

    Returns
    -------
    tuple
        Coherent stack (P,d*d,d*d) and dissipator (d*d,d*d).

    Raises
    ------
    ValueError
        If dimensions fail (P>=1, J>=0, d>=2), arrays are not finite numeric
        with magnitudes<=100, rates are not real nonnegative or a Hamiltonian
        fails absolute Hermiticity tolerance 1e-10. An empty jump stack is
        (0,d,d). Nonfinite arithmetic also raises ValueError.
    """
    return None


# EXPECTED RETURN LINE
# tuple of complex arrays, coherent pulse generators and the common dissipator

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_lindblad_generators(hamiltonians: "np.ndarray", jumps: "np.ndarray", rates: "np.ndarray") -> tuple:
    arrays = []
    for value in (hamiltonians, jumps, rates):
        a = np.asarray(value)
        if a.dtype.kind not in 'iufc' or not np.all(np.isfinite(a)) or np.any(np.abs(a)>100):
            raise ValueError('invalid numeric array')
        arrays.append(a)
    h, c, g = arrays
    if h.ndim != 3 or h.shape[0]<1 or h.shape[1]<2 or h.shape[1]!=h.shape[2]:
        raise ValueError('invalid Hamiltonian shape')
    d = h.shape[1]
    if c.ndim!=3 or c.shape[1:]!=(d,d) or g.shape!=(len(c),) or g.dtype.kind not in 'iuf' or np.any(g<0):
        raise ValueError('invalid jump/rate shape')
    if not np.allclose(h,h.conj().transpose(0,2,1),atol=1e-10,rtol=0):
        raise ValueError('Hamiltonian is not Hermitian')
    eye = np.eye(d)
    coherent = np.array([-1j*(np.kron(x,eye)-np.kron(eye,x.T)) for x in h])
    dissipative = np.zeros((d*d,d*d),complex)
    for jump, rate in zip(c,g):
        gram = jump.conj().T@jump
        dissipative += rate*(np.kron(jump,jump.conj())-.5*np.kron(gram,eye)-.5*np.kron(eye,gram.T))
    if not np.all(np.isfinite(coherent)) or not np.all(np.isfinite(dissipative)):
        raise ValueError('non-finite generator')
    return coherent, dissipative

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base = 'h=np.array([[[.2,.3j],[-.3j,-.1]],[[0.,.4],[.4,0.]]]);c=np.array([[[0.,1.],[0.,0.]]]);g=np.array([.17])'
    cases: list[dict] = [
        {'setup':base,'call':'lindblad_generators(h.copy(),c.copy(),g.copy())','gold_call':'_oracle_lindblad_generators(h.copy(),c.copy(),g.copy())'},
        {'setup':'h=np.zeros((1,2,2));c=np.zeros((0,2,2));g=np.zeros(0)','call':'lindblad_generators(h.copy(),c.copy(),g.copy())','gold_call':'_oracle_lindblad_generators(h.copy(),c.copy(),g.copy())'},
        {'setup':'h=np.array([np.diag([.2,-.1,.7])]);c=np.array([[[0,1,0],[0,0,1j],[0,0,0]],np.diag([1.,-1.,0.])]);g=np.array([.4,.3])','call':'lindblad_generators(h.copy(),c.copy(),g.copy())','gold_call':'_oracle_lindblad_generators(h.copy(),c.copy(),g.copy())'},
    ]
    for change in ('g[0]=-1','h[0,0,1]=1','g[0]=np.nan'):
        setup=base+'\n'+change+'\n'
        for name,fn in (('run_model','lindblad_generators'),('run_gold','_oracle_lindblad_generators')):
            setup+=f'def {name}():\n    try:\n        {fn}(h.copy(),c.copy(),g.copy())\n    except ValueError:\n        return 1\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    packing = 'def pack(values):\n    arrays=[np.asarray(v,dtype=complex) for v in values]\n    shape=np.array([x for a in arrays for x in (a.ndim,)+a.shape],dtype=complex)\n    return np.concatenate([shape]+[a.ravel() for a in arrays])\n'
    for case in cases:
        if case['call'].startswith('lindblad_generators('):
            case['setup'] = packing + case['setup']
            case['call'] = 'pack(' + case['call'] + ')'
            case['gold_call'] = 'pack(' + case['gold_call'] + ')'
        case['setup'] = 'import numpy as np\n' + case['setup']
        case['tol'] = 1e-11
    return cases
