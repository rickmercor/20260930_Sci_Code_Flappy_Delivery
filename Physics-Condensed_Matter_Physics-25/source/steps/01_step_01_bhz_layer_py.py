"""
Assemble the single-layer angular tight-binding operator.

Main paper Eqs. (18)–(19), generalized to supplied on-site masses and scalar potentials. Each edge occurs once; theta points from its first vertex to its second. Off-diagonal blocks are t1*sigma_z + (i*delta/2)*(cos(theta)*sigma_x+sin(theta)*sigma_y). Reverse blocks are Hermitian conjugates.

Returns
-------
return result  # complex ndarray (2N,2N), Hermitian A in vertex-major, orbital-minor order; units t.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def bhz_layer(coords: ArrayLike, edges: ArrayLike, mass: ArrayLike, potential: ArrayLike, t1: float, delta: float) -> np.ndarray:
    'Assemble the single-layer angular tight-binding operator.\n\nParameters\n----------\ncoords : real (N,2), distinct vertex coordinates in a.\nedges : integer (B,2), unique undirected non-self edges, in any orientation.\nmass, potential : real scalars or (N,), on-site mass and scalar potential in t.\nt1, delta : real scalars, bond coefficients in t.\n\nRaises\n------\nValueError for incompatible shapes, numeric types, nonfinite values, or the stated domain violations. Numeric array-like inputs are accepted; real inputs have real numeric type, complex input is allowed only for layer, and booleans, strings and object arrays are invalid. Integer parameters and index arrays have integer type; integer entries packed into real factor/crossing records are integer-valued. N is positive; coordinates are distinct. Edge indices are in [0,N), with no self-edges or repeated undirected edges. Empty edges have shape (0,2). On-site arrays are either scalars or exactly (N,).\n\nReturns\n-------\ncomplex ndarray (2N,2N), Hermitian A in vertex-major, orbital-minor order; units t.'
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def _oracle_bhz_layer(coords: ArrayLike, edges: ArrayLike, mass: ArrayLike, potential: ArrayLike, t1: float, delta: float) -> np.ndarray:
    def _checked_coords(coords, distinct=False):
        r = _checked_numeric(coords, 'coords')
        if r.ndim != 2 or r.shape[1] != 2 or len(r) == 0:
            raise ValueError('coords must have shape (N,2), N>0')
        if distinct and len(np.unique(r, axis=0)) != len(r):
            raise ValueError('vertex coordinates must be distinct')
        return r

    def _checked_edges(edges, n):
        try:
            e = np.asarray(edges)
        except (TypeError, ValueError) as exc:
            raise ValueError('edges must be an integer array of shape (B,2)') from exc
        if e.dtype.kind not in 'iu' or e.ndim != 2 or e.shape[1] != 2:
            raise ValueError('edges must be an integer array of shape (B,2), including B=0')
        if np.any(e < 0) or np.any(e >= n) or np.any(e[:, 0] == e[:, 1]):
            raise ValueError('edge endpoints must be distinct valid vertex indices')
        if len(np.unique(np.sort(e, axis=1), axis=0)) != len(e):
            raise ValueError('each undirected edge must occur once')
        return e.astype(int)

    def _checked_numeric(value, name, real=True):
        try:
            a = np.asarray(value)
            allowed = 'iuf' if real else 'iufc'
            if a.dtype.kind not in allowed or not np.isfinite(a).all():
                raise ValueError(name + ' must contain finite numeric values of the declared type')
            with np.errstate(over='ignore', invalid='ignore'):
                a = a.astype(float if real else complex)
            if not np.isfinite(a).all():
                raise ValueError(name + ' exceeds the supported floating-point range')
            return a
        except (TypeError, OverflowError) as exc:
            raise ValueError(name + ' must be a numeric scalar or array') from exc

    def _checked_scalar(value, name, minimum=None, strict=False, integer=False):
        a = _checked_numeric(value, name)
        if a.shape != ():
            raise ValueError(name + ' must be a scalar')
        x = float(a)
        if integer and (np.asarray(value).dtype.kind not in 'iu' or x != np.floor(x)):
            raise ValueError(name + ' must have integer type')
        if minimum is not None and (x < minimum or (strict and x == minimum)):
            raise ValueError(name + ' is outside its stated domain')
        return int(x) if integer else x

    def _checked_vector(value, n, name, scalar=False):
        a = _checked_numeric(value, name)
        if scalar and a.shape == ():
            return np.full(n, float(a))
        if a.shape != (n,):
            raise ValueError(name + ' has an incompatible shape')
        return a

    r=_checked_coords(coords, distinct=True);n=len(r);edges=_checked_edges(edges,n)
    mass=_checked_vector(mass,n,'mass',scalar=True);potential=_checked_vector(potential,n,'potential',scalar=True)
    t1=_checked_scalar(t1,'t1');delta=_checked_scalar(delta,'delta')
    sx=np.array([[0,1],[1,0]],complex);sy=np.array([[0,-1j],[1j,0]]);sz=np.diag([1.,-1.])
    h=np.zeros((2*n,2*n),complex)
    for j in range(n):h[2*j:2*j+2,2*j:2*j+2]=mass[j]*sz+potential[j]*np.eye(2)
    for j,k in edges:
        d=r[k]-r[j]; d=d/np.linalg.norm(d)
        b=t1*sz+.5j*delta*(d[0]*sx+d[1]*sy)
        h[2*j:2*j+2,2*k:2*k+2]=b;h[2*k:2*k+2,2*j:2*j+2]=b.conj().T
    return h

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nr=np.array([[0, 0]],dtype=float)\ne=np.empty((0, 2),dtype=int)\nu=np.array([1.2],dtype=float)\nv=np.array([0.0],dtype=float)', 'call': 'bhz_layer(r,e,u,v,0.5,0.5)', 'gold_call': '_oracle_bhz_layer(r,e,u,v,0.5,0.5)'}, {'setup': 'import numpy as np\nr=np.array([[0, 0]],dtype=float)\ne=np.empty((0, 2),dtype=int)\nu=np.array([0.0],dtype=float)\nv=np.array([0.7],dtype=float)', 'call': 'bhz_layer(r,e,u,v,0.0,0.0)', 'gold_call': '_oracle_bhz_layer(r,e,u,v,0.0,0.0)'}, {'setup': 'import numpy as np\nr=np.array([[0, 0], [1, 0]],dtype=float)\ne=np.array([[0, 1]],dtype=int)\nu=np.array([-0.6, 0.8],dtype=float)\nv=np.array([0.0, 0.0],dtype=float)', 'call': 'bhz_layer(r,e,u,v,0.5,0.5)', 'gold_call': '_oracle_bhz_layer(r,e,u,v,0.5,0.5)'}, {'setup': 'import numpy as np\nr=np.array([[0, 0], [0, 1]],dtype=float)\ne=np.array([[0, 1]],dtype=int)\nu=np.array([-0.6, -0.6],dtype=float)\nv=np.array([0.0, 0.0],dtype=float)', 'call': 'bhz_layer(r,e,u,v,0.0,0.5)', 'gold_call': '_oracle_bhz_layer(r,e,u,v,0.0,0.5)'}, {'setup': 'import numpy as np\nr=np.array([[0.0, 0.0], [0.8, -0.6]],dtype=float)\ne=np.array([[1, 0]],dtype=int)\nu=np.array([-0.6, -0.3],dtype=float)\nv=np.array([0.1, -0.2],dtype=float)', 'call': 'bhz_layer(r,e,u,v,0.5,0.8)', 'gold_call': '_oracle_bhz_layer(r,e,u,v,0.5,0.8)'}, {'setup': 'import numpy as np\nr=np.array([[0, 0], [1, 0], [4, 3]],dtype=float)\ne=np.array([[0, 1]],dtype=int)\nu=np.array([-1.0, 0.2, 3.0],dtype=float)\nv=np.array([0.0, 0.4, -0.4],dtype=float)', 'call': 'bhz_layer(r,e,u,v,-0.3,0.5)', 'gold_call': '_oracle_bhz_layer(r,e,u,v,-0.3,0.5)'}, {'setup': 'import numpy as np\nr=np.array([[0.0, 0.0], [1.0, 0.1], [1.2, 1.0], [0.1, 0.9]],dtype=float)\ne=np.array([[0, 1], [1, 2], [2, 3], [3, 0]],dtype=int)\nu=np.array([-0.6, -0.6, -0.6, -0.6],dtype=float)\nv=np.array([0.1, 0.2, -0.1, -0.2],dtype=float)', 'call': 'bhz_layer(r,e,u,v,0.5,0.5)', 'gold_call': '_oracle_bhz_layer(r,e,u,v,0.5,0.5)'}, {'setup': 'import numpy as np\nr=np.array([[1, 0], [0, 0], [0, 1]],dtype=float)\ne=np.array([[1, 0], [1, 2]],dtype=int)\nu=np.array([0.4, -0.6, 0.8],dtype=float)\nv=np.array([0.3, -0.1, 0.2],dtype=float)', 'call': 'bhz_layer(r,e,u,v,0.5,0.0)', 'gold_call': '_oracle_bhz_layer(r,e,u,v,0.5,0.0)'}]
