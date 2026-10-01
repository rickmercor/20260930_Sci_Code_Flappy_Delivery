"""
Resolve the first two finite-collision coefficients after tracing the ancilla.

Appendices B and C: finite collision coefficients retain bath-induced nonunitality.

Returns
-------
complex ndarray, shape (2, d*d, d*d), ordered as the coefficients [C1, C2] of t and t**2 in the forward channel; columns and rows use column-major vectorization. Units are inverse time and inverse time squared.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def collision_coefficients(hamiltonian: np.ndarray, bath: np.ndarray) -> np.ndarray:
    """hamiltonian: Hermitian complex (2*d,2*d) array in system-then-qubit-ancilla order, hbar=1.
    bath: positive semidefinite trace-one complex (2,2) array.
    The forward channel is N_t(A)=Tr_E[e^(-itH)(A tensor bath)e^(itH)]. Coefficients are ordinary Taylor coefficients, with factorials absorbed.
    Every numerical input must be finite. Hermiticity, unit trace, semidefinite positivity, and physical Bloch norm are checked with absolute tolerance 1e-10; positive definiteness and scalar sign bounds are strict.

    Raises
    ------
    ValueError if an input has an invalid shape, a nonfinite value, or violates any documented matrix, state, scalar, or Bloch-vector domain. Real inputs must have zero imaginary part.

    Returns
    -------
    complex ndarray, shape (2, d*d, d*d), ordered as the coefficients [C1, C2] of t and t**2 in the forward channel; columns and rows use column-major vectorization. Units are inverse time and inverse time squared.
    """
    return np.zeros((2,(len(hamiltonian)//2)**2,(len(hamiltonian)//2)**2), dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm, solve_sylvester
from scipy.optimize import brentq

def _density(value, name, positive_definite=False, dimension=None):
    a = _hermitian(value, name)
    if dimension is not None and a.shape != (dimension, dimension):
        raise ValueError(name + ' has an invalid dimension')
    if abs(np.trace(a) - 1) > 1e-10:
        raise ValueError(name + ' must have unit trace')
    low = np.linalg.eigvalsh(a)[0]
    if (positive_definite and low <= 0) or (not positive_definite and low < -1e-10):
        raise ValueError(name + ' violates its positivity domain')
    return a

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

def _ptr(a,d):
    return np.trace(a.reshape(d,2,d,2),axis1=1,axis2=3)

def _super(action,d):
    out=np.empty((d*d,d*d),complex)
    for j in range(d):
        for i in range(d):
            e=np.zeros((d,d),complex);e[i,j]=1
            out[:,i+d*j]=action(e).reshape(-1,order='F')
    return out

def _oracle_collision_coefficients(hamiltonian,bath):
    try:
        h=_hamiltonian(hamiltonian);xi=_density(bath,'bath',dimension=2);d=h.shape[0]//2
        def coefficient(e,k):
            b=np.kron(e,xi);c=h@b-b@h
            return _ptr(-1j*c if k==1 else -(h@c-c@h)/2,d)
        return np.stack([_super(lambda e:coefficient(e,1),d),_super(lambda e:coefficient(e,2),d)])
    except ValueError as exc:
        raise ValueError('collision_coefficients: ' + str(exc)) from exc

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nh = np.array([[0j, 0j, 0j, 0j], [0j, 0j, 0j, 0j], [0j, 0j, 0j, 0j], [0j, 0j, 0j, 0j]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.63+0j), 0j], [0j, (0.37+0j)]], dtype=complex)\ntime = 0.17', 'call': 'collision_coefficients(h,bath)', 'gold_call': '_oracle_collision_coefficients(h,bath)'}, {'setup': 'import numpy as np\nh = np.array([[(0.3+0j), 0j, (0.7+0j), 0j], [0j, (0.3+0j), 0j, (0.7+0j)], [(0.7+0j), 0j, (-0.3+0j), 0j], [0j, (0.7+0j), 0j, (-0.3+0j)]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.63+0j), 0j], [0j, (0.37+0j)]], dtype=complex)\ntime = 0.17', 'call': 'collision_coefficients(h,bath)', 'gold_call': '_oracle_collision_coefficients(h,bath)'}, {'setup': 'import numpy as np\nh = np.array([[(0.2+0j), -0.6j, 0j, 0j], [0.6j, (-0.2+0j), 0j, 0j], [0j, 0j, (0.2+0j), -0.6j], [0j, 0j, 0.6j, (-0.2+0j)]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.63+0j), 0j], [0j, (0.37+0j)]], dtype=complex)\ntime = 0.17', 'call': 'collision_coefficients(h,bath)', 'gold_call': '_oracle_collision_coefficients(h,bath)'}, {'setup': 'import numpy as np\nh = np.array([[0j, 0j, 0j, (0.83+0j)], [0j, 0j, (0.83+0j), 0j], [0j, (0.83+0j), 0j, 0j], [(0.83+0j), 0j, 0j, 0j]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.5+0j), (0.205+0j)], [(0.205+0j), (0.5+0j)]], dtype=complex)\ntime = 0.17', 'call': 'collision_coefficients(h,bath)', 'gold_call': '_oracle_collision_coefficients(h,bath)'}, {'setup': 'import numpy as np\nh = np.array([[0j, 0j, 0j, 0j], [0j, 0j, (1+0j), 0j], [0j, (1+0j), 0j, 0j], [0j, 0j, 0j, 0j]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\ntime = 0.17', 'call': 'collision_coefficients(h,bath)', 'gold_call': '_oracle_collision_coefficients(h,bath)'}, {'setup': 'import numpy as np\nh = np.array([[0j, 0j, 0j, 0j], [0j, 0j, (1+0j), 0j], [0j, (1+0j), 0j, 0j], [0j, 0j, 0j, 0j]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.5+0j), 0j], [0j, (0.5+0j)]], dtype=complex)\ntime = 0.17', 'call': 'collision_coefficients(h,bath)', 'gold_call': '_oracle_collision_coefficients(h,bath)'}, {'setup': 'import numpy as np\nh = np.array([[(0.24+0j), 0j, 0.31j, -0.8j], [0j, (0.24+0j), 0.8j, -0.31j], [-0.31j, -0.8j, (-0.24+0j), 0j], [0.8j, 0.31j, 0j, (-0.24+0j)]], dtype=complex)\nbath = np.array([[(0.7+0j), (0.105+0.14j)], [(0.105-0.14j), (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.4+0j), (0.115-0.18j)], [(0.115+0.18j), (0.6+0j)]], dtype=complex)\ntime = 0.17', 'call': 'collision_coefficients(h,bath)', 'gold_call': '_oracle_collision_coefficients(h,bath)'}, {'setup': 'import numpy as np\nh = np.array([[(0.2+0j), 0j, 0j, (0.43+0j), 0j, (-0.61+0j)], [0j, (0.2+0j), (0.43+0j), 0j, (0.61+0j), 0j], [0j, (0.43+0j), 0j, 0j, 0j, 0j], [(0.43+0j), 0j, 0j, 0j, 0j, 0j], [0j, (0.61+0j), 0j, 0j, (-0.2+0j), 0j], [(-0.61+0j), 0j, 0j, 0j, 0j, (-0.2+0j)]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.5+0j), 0j, 0j], [0j, (0.3+0j), 0j], [0j, 0j, (0.2+0j)]], dtype=complex)\ntime = 0.17', 'call': 'collision_coefficients(h,bath)', 'gold_call': '_oracle_collision_coefficients(h,bath)'}, {'setup': 'import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0', 'call': '_raises(lambda: collision_coefficients(np.eye(3),np.eye(2)/2))', 'gold_call': '_raises(lambda: _oracle_collision_coefficients(np.eye(3),np.eye(2)/2))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nh[0,1]=.2j', 'call': '_raises(lambda: collision_coefficients(h, bath))', 'gold_call': '_raises(lambda: _oracle_collision_coefficients(h, bath))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nh[0,0]=np.nan', 'call': '_raises(lambda: collision_coefficients(h, bath))', 'gold_call': '_raises(lambda: _oracle_collision_coefficients(h, bath))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nbath=2*bath', 'call': '_raises(lambda: collision_coefficients(h, bath))', 'gold_call': '_raises(lambda: _oracle_collision_coefficients(h, bath))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nbath=np.diag([1.2,-.2])', 'call': '_raises(lambda: collision_coefficients(h, bath))', 'gold_call': '_raises(lambda: _oracle_collision_coefficients(h, bath))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nbath=np.array([[.5,.2j],[0,.5]])', 'call': '_raises(lambda: collision_coefficients(h, bath))', 'gold_call': '_raises(lambda: _oracle_collision_coefficients(h, bath))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nbath=np.full((2,2),np.inf)', 'call': '_raises(lambda: collision_coefficients(h, bath))', 'gold_call': '_raises(lambda: _oracle_collision_coefficients(h, bath))'}]
