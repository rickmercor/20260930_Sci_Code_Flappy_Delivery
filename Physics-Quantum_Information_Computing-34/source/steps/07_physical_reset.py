"""
Find the global physical reset under a Bloch-radius bound and longitudinal-polarization floor.

Task-specific laboratory preparation constraint, using standard convex quadratic optimization.

Returns
-------
float ndarray, shape (3,), the unique minimizing Bloch vector ordered [s_x,s_y,s_z], dimensionless.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def physical_reset(objective: np.ndarray, radius: float, z_floor: float) -> np.ndarray:
    """objective: real symmetric (4,4) packed matrix from shared_reset_objective, with strictly positive-definite Q.
    radius: finite real scalar satisfying 0<radius<=1.
    z_floor: finite real scalar satisfying -radius<=z_floor<=radius.
    Minimize J(s)=s.T@Q@s-2*b.T@s+c over ||s||_2<=radius and s_z>=z_floor.
    Every numerical input must be finite. Hermiticity, unit trace, semidefinite positivity, and physical Bloch norm are checked with absolute tolerance 1e-10; positive definiteness and scalar sign bounds are strict.

    Raises
    ------
    ValueError if an input has an invalid shape, a nonfinite value, or violates any documented matrix, state, scalar, or Bloch-vector domain. Real inputs must have zero imaginary part.

    Returns
    -------
    float ndarray, shape (3,), the unique minimizing Bloch vector ordered [s_x,s_y,s_z], dimensionless.
    """
    return np.zeros((3,), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm, solve_sylvester
from scipy.optimize import brentq

def _bounds(radius, z_floor):
    r = _real_scalar(radius, 'radius', positive=True)
    z = _real_scalar(z_floor, 'z_floor')
    if r > 1 or not -r <= z <= r:
        raise ValueError('Require 0 < radius <= 1 and -radius <= z_floor <= radius')
    return r, z

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

def _real_scalar(value, name, positive=False, nonnegative=False):
    a = _finite_array(value, name, (), real=True)
    x = float(a)
    if (positive and x <= 0) or (nonnegative and x < 0):
        raise ValueError(name + ' violates its sign domain')
    return x

def _oracle_physical_reset(objective,radius,z_floor):
    try:
        m=_finite_array(objective,'objective',(4,4),real=True)
        _hermitian(m,'objective')
        if np.linalg.eigvalsh(m[:3,:3])[0]<=0:raise ValueError('Q must be positive definite')
        r,z=_bounds(radius,z_floor)
        q=m[:3,:3];b=m[:3,3]
        def ball(q,b,r):
            if r==0:return np.zeros(len(b))
            s=np.linalg.solve(q,b)
            if np.linalg.norm(s)<=r:return s
            def f(mu):return np.linalg.norm(np.linalg.solve(q+mu*np.eye(len(b)),b))-r
            hi=1.
            while f(hi)>0:hi*=2
            mu=brentq(f,0,hi,xtol=5e-15,rtol=5e-15)
            return np.linalg.solve(q+mu*np.eye(len(b)),b)
        s=ball(q,b,r)
        if s[2]<z:
            xy=ball(q[:2,:2],b[:2]-q[:2,2]*z,np.sqrt(max(0,r*r-z*z)))
            s=np.r_[xy,z]
        return s
    except ValueError as exc:
        raise ValueError('physical_reset: ' + str(exc)) from exc

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nobjective=np.array([[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, -0.2], [0.0, 0.0, 1.0, 0.15], [0.1, -0.2, 0.15, 4.0]], dtype=float)\nradius=0.8\nz_floor=-0.5', 'call': 'physical_reset(objective,radius,z_floor)', 'gold_call': '_oracle_physical_reset(objective,radius,z_floor)'}, {'setup': 'import numpy as np\nobjective=np.array([[1.0, 0.0, 0.0, 1.2], [0.0, 2.0, 0.0, -1.0], [0.0, 0.0, 3.0, 0.3], [1.2, -1.0, 0.3, 4.0]], dtype=float)\nradius=0.6\nz_floor=-0.5', 'call': 'physical_reset(objective,radius,z_floor)', 'gold_call': '_oracle_physical_reset(objective,radius,z_floor)'}, {'setup': 'import numpy as np\nobjective=np.array([[2.0, 0.0, 0.0, 0.2], [0.0, 1.0, 0.0, 0.1], [0.0, 0.0, 3.0, -1.0], [0.2, 0.1, -1.0, 4.0]], dtype=float)\nradius=0.8\nz_floor=0.3', 'call': 'physical_reset(objective,radius,z_floor)', 'gold_call': '_oracle_physical_reset(objective,radius,z_floor)'}, {'setup': 'import numpy as np\nobjective=np.array([[2.0, 0.4, 0.2, 1.0], [0.4, 1.2, -0.1, -0.8], [0.2, -0.1, 2.4, -0.3], [1.0, -0.8, -0.3, 4.0]], dtype=float)\nradius=0.6\nz_floor=0.4', 'call': 'physical_reset(objective,radius,z_floor)', 'gold_call': '_oracle_physical_reset(objective,radius,z_floor)'}, {'setup': 'import numpy as np\nobjective=np.array([[1.0, 0.0, 0.0, 0.4], [0.0, 1.0, 0.0, 0.1], [0.0, 0.0, 1.0, -0.2], [0.4, 0.1, -0.2, 4.0]], dtype=float)\nradius=0.5\nz_floor=0.5', 'call': 'physical_reset(objective,radius,z_floor)', 'gold_call': '_oracle_physical_reset(objective,radius,z_floor)'}, {'setup': 'import numpy as np\nobjective=np.array([[1.0, 0.0, 0.0, 0.1], [0.0, 1.5, 0.0, -0.2], [0.0, 0.0, 2.0, -3.0], [0.1, -0.2, -3.0, 4.0]], dtype=float)\nradius=0.5\nz_floor=-0.5', 'call': 'physical_reset(objective,radius,z_floor)', 'gold_call': '_oracle_physical_reset(objective,radius,z_floor)'}, {'setup': 'import numpy as np\nobjective=np.array([[2.0, 0.7, -0.4, 0.8], [0.7, 3.0, 0.6, 0.9], [-0.4, 0.6, 1.7, -0.4], [0.8, 0.9, -0.4, 4.0]], dtype=float)\nradius=0.65\nz_floor=0.22', 'call': 'physical_reset(objective,radius,z_floor)', 'gold_call': '_oracle_physical_reset(objective,radius,z_floor)'}, {'setup': 'import numpy as np\nobjective=np.array([[0.3, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 4.0]], dtype=float)\nradius=1.0\nz_floor=-0.7', 'call': 'physical_reset(objective,radius,z_floor)', 'gold_call': '_oracle_physical_reset(objective,radius,z_floor)'}, {'setup': 'import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0', 'call': '_raises(lambda: physical_reset(np.eye(4),0.,0.))', 'gold_call': '_raises(lambda: _oracle_physical_reset(np.eye(4),0.,0.))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nobjective[0,0]=np.nan', 'call': '_raises(lambda: physical_reset(objective, radius, z_floor))', 'gold_call': '_raises(lambda: _oracle_physical_reset(objective, radius, z_floor))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nobjective[0,1]=.2', 'call': '_raises(lambda: physical_reset(objective, radius, z_floor))', 'gold_call': '_raises(lambda: _oracle_physical_reset(objective, radius, z_floor))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nobjective[0,0]=-1.', 'call': '_raises(lambda: physical_reset(objective, radius, z_floor))', 'gold_call': '_raises(lambda: _oracle_physical_reset(objective, radius, z_floor))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nobjective[0,0]=0.', 'call': '_raises(lambda: physical_reset(objective, radius, z_floor))', 'gold_call': '_raises(lambda: _oracle_physical_reset(objective, radius, z_floor))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nobjective=objective.astype(complex);objective[0,0]+=1j', 'call': '_raises(lambda: physical_reset(objective, radius, z_floor))', 'gold_call': '_raises(lambda: _oracle_physical_reset(objective, radius, z_floor))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nradius=np.inf', 'call': '_raises(lambda: physical_reset(objective, radius, z_floor))', 'gold_call': '_raises(lambda: _oracle_physical_reset(objective, radius, z_floor))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nz_floor=np.nan', 'call': '_raises(lambda: physical_reset(objective, radius, z_floor))', 'gold_call': '_raises(lambda: _oracle_physical_reset(objective, radius, z_floor))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nradius=1.2', 'call': '_raises(lambda: physical_reset(objective, radius, z_floor))', 'gold_call': '_raises(lambda: _oracle_physical_reset(objective, radius, z_floor))'}, {'setup': 'import numpy as np\nh=np.eye(4,dtype=complex)\nbath=np.eye(2)/2\nprior=np.diag([.6,.4])\nnext_prior=prior.copy()\ncollision=np.zeros((2,4,4),complex)\nnormalizer=np.stack([np.diag(1/np.sqrt([.6,.4])),np.zeros((2,2)),np.zeros((2,2))])\nhamiltonians=h[None,:,:]\ninitial_prior=prior.copy()\ntimes=np.array([.1])\npetz=np.zeros((1,2,4,4),complex)\nreverse=np.zeros((1,2,4,4,4),complex)\npenalty=.1\nnominal=np.zeros(3)\nobjective=np.eye(4)\nradius=.5\nz_floor=.1\ntime=.1\nreset=np.zeros(3)\nmaps=np.tile(np.eye(4),(1,3,1,1))\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nz_floor=.6', 'call': '_raises(lambda: physical_reset(objective, radius, z_floor))', 'gold_call': '_raises(lambda: _oracle_physical_reset(objective, radius, z_floor))'}]
