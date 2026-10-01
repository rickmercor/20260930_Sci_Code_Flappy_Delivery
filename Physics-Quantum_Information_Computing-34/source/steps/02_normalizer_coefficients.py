"""
Resolve the noncommuting inverse-square-root coefficients of the evolving prior.

Appendix C.2: the prior normalization requires noncommuting square-root perturbation terms.

Returns
-------
complex ndarray, shape (3, d, d), ordered as [W0, W1, W2] in (N_t(prior))**(-1/2) = W0 + t*W1 + t**2*W2 + O(t**3); units are respectively dimensionless, inverse time, inverse time squared.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def normalizer_coefficients(prior: np.ndarray, collision: np.ndarray) -> np.ndarray:
    """prior: positive-definite trace-one complex (d,d) array.
    collision: complex (2,d*d,d*d) array [C1,C2] from collision_coefficients.
    The exact forward channel is evaluated with this prior fixed while t varies. Matrix powers mean positive spectral powers.
    Every numerical input must be finite. Hermiticity, unit trace, semidefinite positivity, and physical Bloch norm are checked with absolute tolerance 1e-10; positive definiteness and scalar sign bounds are strict.

    Raises
    ------
    ValueError if an input has an invalid shape, a nonfinite value, or violates any documented matrix, state, scalar, or Bloch-vector domain. Real inputs must have zero imaginary part.

    Returns
    -------
    complex ndarray, shape (3, d, d), ordered as [W0, W1, W2] in (N_t(prior))**(-1/2) = W0 + t*W1 + t**2*W2 + O(t**3); units are respectively dimensionless, inverse time, inverse time squared.
    """
    return np.zeros((3,len(prior),len(prior)), dtype=complex)

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

def _hermitian(value, name):
    a = _finite_array(value, name)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[0] != a.shape[1]:
        raise ValueError(name + ' must be a nonempty square matrix')
    if not np.allclose(a, a.conj().T, atol=1e-10, rtol=0):
        raise ValueError(name + ' must be Hermitian')
    return a

def _pow(a,p):
    v,u=np.linalg.eigh((a+a.conj().T)/2)
    return (u*v**p)@u.conj().T

def _oracle_normalizer_coefficients(prior,collision):
    try:
        g=_density(prior,'prior',positive_definite=True);d=len(g);c=_finite_array(collision,'collision',(2,d*d,d*d))
        g1=(c[0]@g.reshape(-1,order='F')).reshape(d,d,order='F')
        g2=(c[1]@g.reshape(-1,order='F')).reshape(d,d,order='F')
        root=_pow(g,.5);w0=_pow(g,-.5)
        s1=solve_sylvester(root,root,g1)
        s2=solve_sylvester(root,root,g2-s1@s1)
        w1=-w0@s1@w0
        w2=w0@s1@w0@s1@w0-w0@s2@w0
        return np.stack([w0,w1,w2])
    except ValueError as exc:
        raise ValueError('normalizer_coefficients: ' + str(exc)) from exc

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nh = np.array([[0j, 0j, 0j, 0j], [0j, 0j, 0j, 0j], [0j, 0j, 0j, 0j], [0j, 0j, 0j, 0j]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.63+0j), 0j], [0j, (0.37+0j)]], dtype=complex)\ntime = 0.17\ncollision = collision_coefficients(h,bath)', 'call': 'normalizer_coefficients(prior,collision)', 'gold_call': '_oracle_normalizer_coefficients(prior,collision)'}, {'setup': 'import numpy as np\nh = np.array([[(0.3+0j), 0j, (0.7+0j), 0j], [0j, (0.3+0j), 0j, (0.7+0j)], [(0.7+0j), 0j, (-0.3+0j), 0j], [0j, (0.7+0j), 0j, (-0.3+0j)]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.63+0j), 0j], [0j, (0.37+0j)]], dtype=complex)\ntime = 0.17\ncollision = collision_coefficients(h,bath)', 'call': 'normalizer_coefficients(prior,collision)', 'gold_call': '_oracle_normalizer_coefficients(prior,collision)'}, {'setup': 'import numpy as np\nh = np.array([[0j, 0j, 0j, (0.83+0j)], [0j, 0j, (0.83+0j), 0j], [0j, (0.83+0j), 0j, 0j], [(0.83+0j), 0j, 0j, 0j]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.5+0j), (0.205+0j)], [(0.205+0j), (0.5+0j)]], dtype=complex)\ntime = 0.17\ncollision = collision_coefficients(h,bath)', 'call': 'normalizer_coefficients(prior,collision)', 'gold_call': '_oracle_normalizer_coefficients(prior,collision)'}, {'setup': 'import numpy as np\nh = np.array([[0j, 0j, 0j, 0j], [0j, 0j, (1+0j), 0j], [0j, (1+0j), 0j, 0j], [0j, 0j, 0j, 0j]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\ntime = 0.17\ncollision = collision_coefficients(h,bath)', 'call': 'normalizer_coefficients(prior,collision)', 'gold_call': '_oracle_normalizer_coefficients(prior,collision)'}, {'setup': 'import numpy as np\nh = np.array([[0j, 0j, 0j, 0j], [0j, 0j, (1+0j), 0j], [0j, (1+0j), 0j, 0j], [0j, 0j, 0j, 0j]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.5+0j), 0j], [0j, (0.5+0j)]], dtype=complex)\ntime = 0.17\ncollision = collision_coefficients(h,bath)', 'call': 'normalizer_coefficients(prior,collision)', 'gold_call': '_oracle_normalizer_coefficients(prior,collision)'}, {'setup': 'import numpy as np\nh = np.array([[(0.24+0j), 0j, 0.31j, -0.8j], [0j, (0.24+0j), 0.8j, -0.31j], [-0.31j, -0.8j, (-0.24+0j), 0j], [0.8j, 0.31j, 0j, (-0.24+0j)]], dtype=complex)\nbath = np.array([[(0.7+0j), (0.105+0.14j)], [(0.105-0.14j), (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.4+0j), (0.115-0.18j)], [(0.115+0.18j), (0.6+0j)]], dtype=complex)\ntime = 0.17\ncollision = collision_coefficients(h,bath)', 'call': 'normalizer_coefficients(prior,collision)', 'gold_call': '_oracle_normalizer_coefficients(prior,collision)'}, {'setup': 'import numpy as np\nh = np.array([[0j, -0.37j, 0j, (0.43+0j)], [0.37j, 0j, (0.43+0j), 0j], [0j, (0.43+0j), 0j, 0.37j], [(0.43+0j), 0j, -0.37j, 0j]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[(0.998+0j), 0j], [0j, (0.002+0j)]], dtype=complex)\ntime = 0.06\ncollision = collision_coefficients(h,bath)', 'call': 'normalizer_coefficients(prior,collision)', 'gold_call': '_oracle_normalizer_coefficients(prior,collision)'}, {'setup': 'import numpy as np\nh = np.array([[0j, -0.7j, 0j, 0j, 0.2j, (0.4+0j), 0j, 0j], [0.7j, 0j, 0j, 0j, (0.4+0j), -0.2j, 0j, 0j], [0j, 0j, 0j, 0.7j, 0j, 0j, -0.2j, (0.4+0j)], [0j, 0j, -0.7j, 0j, 0j, 0j, (0.4+0j), 0.2j], [-0.2j, (0.4+0j), 0j, 0j, 0j, -0.7j, 0j, 0j], [(0.4+0j), 0.2j, 0j, 0j, 0.7j, 0j, 0j, 0j], [0j, 0j, 0.2j, (0.4+0j), 0j, 0j, 0j, 0.7j], [0j, 0j, (0.4+0j), -0.2j, 0j, 0j, -0.7j, 0j]], dtype=complex)\nbath = np.array([[(0.7+0j), 0j], [0j, (0.3+0j)]], dtype=complex)\nprior = np.array([[0.25, 0.0, 0.0, 0.0], [0.0, 0.25, 0.0, 0.0], [0.0, 0.0, 0.25, 0.0], [0.0, 0.0, 0.0, 0.25]], dtype=float)\ntime = 0.17\ncollision = collision_coefficients(h,bath)', 'call': 'normalizer_coefficients(prior,collision)', 'gold_call': '_oracle_normalizer_coefficients(prior,collision)'}, {'setup': 'import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0', 'call': '_raises(lambda: normalizer_coefficients(np.eye(2)/2,np.zeros((1,4,4))))', 'gold_call': '_raises(lambda: _oracle_normalizer_coefficients(np.eye(2)/2,np.zeros((1,4,4))))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nprior=np.diag([1.,0.])', 'call': '_raises(lambda: normalizer_coefficients(prior, collision))', 'gold_call': '_raises(lambda: _oracle_normalizer_coefficients(prior, collision))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nprior=np.diag([1.1,-.1])', 'call': '_raises(lambda: normalizer_coefficients(prior, collision))', 'gold_call': '_raises(lambda: _oracle_normalizer_coefficients(prior, collision))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nprior=2*prior', 'call': '_raises(lambda: normalizer_coefficients(prior, collision))', 'gold_call': '_raises(lambda: _oracle_normalizer_coefficients(prior, collision))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nprior=np.array([[.6,.1j],[0,.4]])', 'call': '_raises(lambda: normalizer_coefficients(prior, collision))', 'gold_call': '_raises(lambda: _oracle_normalizer_coefficients(prior, collision))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nprior=np.full((2,2),np.nan)', 'call': '_raises(lambda: normalizer_coefficients(prior, collision))', 'gold_call': '_raises(lambda: _oracle_normalizer_coefficients(prior, collision))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\ncollision[0,0,0]=np.nan', 'call': '_raises(lambda: normalizer_coefficients(prior, collision))', 'gold_call': '_raises(lambda: _oracle_normalizer_coefficients(prior, collision))'}]
