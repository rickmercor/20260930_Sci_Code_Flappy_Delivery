"""
Resolve the affine dependence of both tabletop reverse coefficients on the reset Bloch vector.

Definition 3 and Appendix C.1: one time-independent ancilla must supply both reverse Taylor coefficients.

Returns
-------
complex ndarray, shape (2, 4, d*d, d*d). Entry [k,a] is the coefficient of t**(k+1) in the reverse map with ancilla operator sigma_a/2, with a ordered I,X,Y,Z. A reset with Bloch vector s uses coefficients result[:,0] + sum_a s[a]*result[:,a+1]. Rows and columns use column-major vectorization; units are inverse-time powers.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def reverse_coefficients(hamiltonian: np.ndarray) -> np.ndarray:
    """hamiltonian: Hermitian complex (2*d,2*d) array, system-then-qubit-ancilla order, hbar=1.
    The reverse map is T_t,eta(A)=Tr_E[e^(itH)(A tensor eta)e^(-itH)]. Sigma_I is identity and sigma_X,Y,Z are the standard Pauli matrices. Coefficients are ordinary Taylor coefficients.
    Every numerical input must be finite. Hermiticity, unit trace, semidefinite positivity, and physical Bloch norm are checked with absolute tolerance 1e-10; positive definiteness and scalar sign bounds are strict.

    Raises
    ------
    ValueError if an input has an invalid shape, a nonfinite value, or violates any documented matrix, state, scalar, or Bloch-vector domain. Real inputs must have zero imaginary part.

    Returns
    -------
    complex ndarray, shape (2, 4, d*d, d*d). Entry [k,a] is the coefficient of t**(k+1) in the reverse map with ancilla operator sigma_a/2, with a ordered I,X,Y,Z. A reset with Bloch vector s uses coefficients result[:,0] + sum_a s[a]*result[:,a+1]. Rows and columns use column-major vectorization; units are inverse-time powers.
    """
    return np.zeros((2,4,(len(hamiltonian)//2)**2,(len(hamiltonian)//2)**2), dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm, solve_sylvester
from scipy.optimize import brentq

def _finite_array(value, name, shape=None, real=False):
    try:
        a = np.asarray(value, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a numerical array') from exc
    if not np.all(np.isfinite(a)):
        raise ValueError(name + ' must contain finite values')
    if shape is not None and a.shape != shape:
        raise ValueError(name + ' has an invalid shape')
    if real:
        if np.any(a.imag != 0):
            raise ValueError(name + ' must be real')
        a = a.real
    return a

def _hamiltonian(value, dimension=None):
    a = _hermitian(value, 'hamiltonian')
    if a.shape[0] % 2 or (dimension is not None and a.shape != (2*dimension, 2*dimension)):
        raise ValueError('hamiltonian must act on system then qubit ancilla')
    return a

def _hermitian(value, name):
    a = _finite_array(value, name)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[0] != a.shape[1]:
        raise ValueError(name + ' must be a nonempty square matrix')
    if not np.allclose(a, a.conj().T, atol=1e-10, rtol=0):
        raise ValueError(name + ' must be Hermitian')
    return a

def _paulis():
    return np.array([np.eye(2), [[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]],complex)

def _ptr(a,d):
    return np.trace(a.reshape(d,2,d,2),axis1=1,axis2=3)

def _super(action,d):
    out=np.empty((d*d,d*d),complex)
    for j in range(d):
        for i in range(d):
            e=np.zeros((d,d),complex);e[i,j]=1
            out[:,i+d*j]=action(e).reshape(-1,order='F')
    return out

def _oracle_reverse_coefficients(hamiltonian):
    try:
        h=_hamiltonian(hamiltonian);d=len(h)//2;basis=_paulis()/2
        out=np.empty((2,4,d*d,d*d),complex)
        for a,b in enumerate(basis):
            def f(e,k):
                x=np.kron(e,b);c=h@x-x@h
                return _ptr(1j*c if k==1 else -(h@c-c@h)/2,d)
            out[0,a]=_super(lambda e:f(e,1),d)
            out[1,a]=_super(lambda e:f(e,2),d)
        return out
    except ValueError as exc:
        raise ValueError('reverse_coefficients: ' + str(exc)) from exc

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nh = np.array([[0j, 0j, 0j, 0j], [0j, 0j, 0j, 0j], [0j, 0j, 0j, 0j], [0j, 0j, 0j, 0j]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.63+0j), 0j], [0j, (0.37+0j)]], dtype=complex)\ntime = 0.17', 'call': 'reverse_coefficients(h)', 'gold_call': '_oracle_reverse_coefficients(h)'}, {'setup': 'import numpy as np\nh = np.array([[(0.3+0j), 0j, (0.7+0j), 0j], [0j, (0.3+0j), 0j, (0.7+0j)], [(0.7+0j), 0j, (-0.3+0j), 0j], [0j, (0.7+0j), 0j, (-0.3+0j)]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.63+0j), 0j], [0j, (0.37+0j)]], dtype=complex)\ntime = 0.17', 'call': 'reverse_coefficients(h)', 'gold_call': '_oracle_reverse_coefficients(h)'}, {'setup': 'import numpy as np\nh = np.array([[(0.2+0j), -0.6j, 0j, 0j], [0.6j, (-0.2+0j), 0j, 0j], [0j, 0j, (0.2+0j), -0.6j], [0j, 0j, 0.6j, (-0.2+0j)]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.63+0j), 0j], [0j, (0.37+0j)]], dtype=complex)\ntime = 0.17', 'call': 'reverse_coefficients(h)', 'gold_call': '_oracle_reverse_coefficients(h)'}, {'setup': 'import numpy as np\nh = np.array([[0j, 0j, 0j, (0.83+0j)], [0j, 0j, (0.83+0j), 0j], [0j, (0.83+0j), 0j, 0j], [(0.83+0j), 0j, 0j, 0j]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.5+0j), (0.205+0j)], [(0.205+0j), (0.5+0j)]], dtype=complex)\ntime = 0.17', 'call': 'reverse_coefficients(h)', 'gold_call': '_oracle_reverse_coefficients(h)'}, {'setup': 'import numpy as np\nh = np.array([[0j, 0j, 0j, 0j], [0j, 0j, (1+0j), 0j], [0j, (1+0j), 0j, 0j], [0j, 0j, 0j, 0j]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\ntime = 0.17', 'call': 'reverse_coefficients(h)', 'gold_call': '_oracle_reverse_coefficients(h)'}, {'setup': 'import numpy as np\nh = np.array([[(0.24+0j), 0j, 0.31j, -0.8j], [0j, (0.24+0j), 0.8j, -0.31j], [-0.31j, -0.8j, (-0.24+0j), 0j], [0.8j, 0.31j, 0j, (-0.24+0j)]], dtype=complex)\nbath = np.array([[(0.7+0j), (0.105+0.14j)], [(0.105-0.14j), (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.4+0j), (0.115-0.18j)], [(0.115+0.18j), (0.6+0j)]], dtype=complex)\ntime = 0.17', 'call': 'reverse_coefficients(h)', 'gold_call': '_oracle_reverse_coefficients(h)'}, {'setup': 'import numpy as np\nh = np.array([[(0.2+0j), 0j, 0j, (0.43+0j), 0j, (-0.61+0j)], [0j, (0.2+0j), (0.43+0j), 0j, (0.61+0j), 0j], [0j, (0.43+0j), 0j, 0j, 0j, 0j], [(0.43+0j), 0j, 0j, 0j, 0j, 0j], [0j, (0.61+0j), 0j, 0j, (-0.2+0j), 0j], [(-0.61+0j), 0j, 0j, 0j, 0j, (-0.2+0j)]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.5+0j), 0j, 0j], [0j, (0.3+0j), 0j], [0j, 0j, (0.2+0j)]], dtype=complex)\ntime = 0.17', 'call': 'reverse_coefficients(h)', 'gold_call': '_oracle_reverse_coefficients(h)'}, {'setup': 'import numpy as np\nh = np.array([[0j, 0j, 0j, (0.77+0j)], [0j, 0j, (0.77+0j), 0j], [0j, (0.77+0j), 0j, 0j], [(0.77+0j), 0j, 0j, 0j]], dtype=complex)\nbath = np.array([[(0.42+0j), (0.165-0.125j)], [(0.165+0.125j), (0.58+0j)]], dtype=complex)\nprior = np.array([[(0.5+0j), (-0.14+0j)], [(-0.14+0j), (0.5+0j)]], dtype=complex)\ntime = 0.43', 'call': 'reverse_coefficients(h)', 'gold_call': '_oracle_reverse_coefficients(h)'}, {'setup': 'import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0', 'call': '_raises(lambda: reverse_coefficients(np.eye(3)))', 'gold_call': '_raises(lambda: _oracle_reverse_coefficients(np.eye(3)))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nh[0,1]=.2j', 'call': '_raises(lambda: reverse_coefficients(h))', 'gold_call': '_raises(lambda: _oracle_reverse_coefficients(h))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nh[0,0]=np.nan', 'call': '_raises(lambda: reverse_coefficients(h))', 'gold_call': '_raises(lambda: _oracle_reverse_coefficients(h))'}]
