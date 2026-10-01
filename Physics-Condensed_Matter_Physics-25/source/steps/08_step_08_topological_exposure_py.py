"""
Compute the integrated protection of the complete class-AII lattice.

Final orchestrator. Invoke and combine all seven earlier public functions. Assemble the physical layer and pencil; obtain orientation by applying the same reduction and factor certificate to atomic A=(abs(E)+1)I and zero mixing; determine the closures and combine phase integrals. The private reference chains the matching _oracle_ twins. This is an author-defined finite coupling exposure built entirely from the main paper’s local parity and protection.

Returns
-------
return result  # float, sum of gap integrals over all nontrivial intervals between 0 and cmax, in the supplied dimensionless t units (physical dimension t squared).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def topological_exposure(coords: ArrayLike, edges: ArrayLike, mass: ArrayLike, potential: ArrayLike, profile: ArrayLike, t1: float, delta: float, probe: ArrayLike, kappa: float, cmax: float, order: int = 64) -> float:
    'Compute the integrated protection of the complete class-AII lattice.\n\nParameters\n----------\ncoords : real (N,2), coordinates in a.\nedges : integer (B,2), unique non-self undirected edges.\nmass, potential : real scalars or (N,), onsite coefficients in t.\nprofile : real (N,), dimensionless interlayer texture.\nt1, delta : real bond coefficients in t.\nprobe : real (3,), ordered x/a,y/a,E/t.\nkappa : nonnegative float in t/a.\ncmax : positive upper coupling in t.\norder : positive Gauss–Legendre order per interval, default 64. Inputs satisfy the regular-pencil and root-separation domain of pencil_crossings.\n\nRaises\n------\nValueError for incompatible shapes, numeric types, nonfinite values, or the stated domain violations. Numeric array-like inputs are accepted; real inputs have real numeric type, complex input is allowed only for layer, and booleans, strings and object arrays are invalid. Integer parameters and index arrays have integer type; integer entries packed into real factor/crossing records are integer-valued. All input contracts of bhz_layer, aii_pencil and pencil_crossings apply, including distinct coordinates, exact edge shape, finite arrays, nonnegative kappa and a regular pencil. cmax and the integer quadrature order are positive.\n\nReturns\n-------\nfloat, sum of gap integrals over all nontrivial intervals between 0 and cmax, in the supplied dimensionless t units (physical dimension t squared).'
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def _oracle_topological_exposure(coords: ArrayLike, edges: ArrayLike, mass: ArrayLike, potential: ArrayLike, profile: ArrayLike, t1: float, delta: float, probe: ArrayLike, kappa: float, cmax: float, order: int = 64) -> float:
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

    cmax=_checked_scalar(cmax,'cmax',minimum=0,strict=True)
    order=_checked_scalar(order,'order',minimum=1,integer=True)
    h=_oracle_bhz_layer(coords,edges,mass,potential,t1,delta)
    p=_oracle_aii_pencil(h,coords,profile,probe,kappa)
    atom=(abs(probe[2])+1)*np.eye(len(h))
    ref=_oracle_aii_pencil(atom,coords,np.zeros(len(coords)),probe,kappa)[0]
    perm=_oracle_paired_ordering(ref);fac=_oracle_skew_factor(ref,perm)
    orientation=int(_oracle_pfaffian_certificate(ref,fac)[0])
    crossing=_oracle_pencil_crossings(p,0.,cmax)
    rows=_oracle_phase_integrals(p,crossing,0.,cmax,orientation,order)
    return float(np.sum(rows[:,3]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nn=5\nr=np.array([(i-(n-1)/2+0.08*np.sin(2*j+i),j-(n-1)/2+0.06*np.cos(i-3*j)) for j in range(n) for i in range(n)])\ne=np.array([(j*n+i,j*n+i+1) for j in range(n) for i in range(n-1)]+[(j*n+i,(j+1)*n+i) for j in range(n-1) for i in range(n)],dtype=int).reshape(-1,2)\nu=np.array([-.6+.09*np.cos(j) for j in range(n*n)])\nv=np.array([.07*np.sin(1.3*j) for j in range(n*n)])\ng=1+.14*np.cos(np.arange(n*n)*.7)\nz=np.array([.13,-.17,.05])\n', 'call': 'topological_exposure(r,e,u,v,g,0.5,0.5,z,0.08,0.6,64)', 'gold_call': '_oracle_topological_exposure(r,e,u,v,g,0.5,0.5,z,0.08,0.6,64)'}, {'setup': 'import numpy as np\nn=1\nr=np.array([(i-(n-1)/2+0.08*np.sin(2*j+i),j-(n-1)/2+0.06*np.cos(i-3*j)) for j in range(n) for i in range(n)])\ne=np.array([(j*n+i,j*n+i+1) for j in range(n) for i in range(n-1)]+[(j*n+i,(j+1)*n+i) for j in range(n-1) for i in range(n)],dtype=int).reshape(-1,2)\nu=np.array([-.6+.09*np.cos(j) for j in range(n*n)])\nv=np.array([.07*np.sin(1.3*j) for j in range(n*n)])\ng=1+.14*np.cos(np.arange(n*n)*.7)\nz=np.array([.13,-.17,.05])\n', 'call': 'topological_exposure(r,e,u,v,g,0.5,0.5,z,0.08,0.6,16)', 'gold_call': '_oracle_topological_exposure(r,e,u,v,g,0.5,0.5,z,0.08,0.6,16)'}, {'setup': 'import numpy as np\nn=3\nr=np.array([(i-(n-1)/2+0.08*np.sin(2*j+i),j-(n-1)/2+0.06*np.cos(i-3*j)) for j in range(n) for i in range(n)])\ne=np.array([(j*n+i,j*n+i+1) for j in range(n) for i in range(n-1)]+[(j*n+i,(j+1)*n+i) for j in range(n-1) for i in range(n)],dtype=int).reshape(-1,2)\nu=np.array([-.6+.09*np.cos(j) for j in range(n*n)])\nv=np.array([.07*np.sin(1.3*j) for j in range(n*n)])\ng=1+.14*np.cos(np.arange(n*n)*.7)\nz=np.array([.13,-.17,.05])\n', 'call': 'topological_exposure(r,e,u,v,g,0.5,0.5,z,0.25,0.7,24)', 'gold_call': '_oracle_topological_exposure(r,e,u,v,g,0.5,0.5,z,0.25,0.7,24)'}, {'setup': 'import numpy as np\nn=4\nr=np.array([(i-(n-1)/2+0.08*np.sin(2*j+i),j-(n-1)/2+0.06*np.cos(i-3*j)) for j in range(n) for i in range(n)])\ne=np.array([(j*n+i,j*n+i+1) for j in range(n) for i in range(n-1)]+[(j*n+i,(j+1)*n+i) for j in range(n-1) for i in range(n)],dtype=int).reshape(-1,2)\nu=np.array([-.6+.09*np.cos(j) for j in range(n*n)])\nv=np.array([.07*np.sin(1.3*j) for j in range(n*n)])\ng=1+.14*np.cos(np.arange(n*n)*.7)\nz=np.array([.13,-.17,.05])\ng[:]=0', 'call': 'topological_exposure(r,e,u,v,g,0.5,0.5,z,0.08,0.5,24)', 'gold_call': '_oracle_topological_exposure(r,e,u,v,g,0.5,0.5,z,0.08,0.5,24)'}, {'setup': 'import numpy as np\nn=3\nr=np.array([(i-(n-1)/2+0.08*np.sin(2*j+i),j-(n-1)/2+0.06*np.cos(i-3*j)) for j in range(n) for i in range(n)])\ne=np.array([(j*n+i,j*n+i+1) for j in range(n) for i in range(n-1)]+[(j*n+i,(j+1)*n+i) for j in range(n-1) for i in range(n)],dtype=int).reshape(-1,2)\nu=np.array([-.6+.09*np.cos(j) for j in range(n*n)])\nv=np.array([.07*np.sin(1.3*j) for j in range(n*n)])\ng=1+.14*np.cos(np.arange(n*n)*.7)\nz=np.array([.13,-.17,.05])\nu[:]=3.', 'call': 'topological_exposure(r,e,u,v,g,0.5,0.5,z,0.12,0.6,24)', 'gold_call': '_oracle_topological_exposure(r,e,u,v,g,0.5,0.5,z,0.12,0.6,24)'}, {'setup': 'import numpy as np\nn=4\nr=np.array([(i-(n-1)/2+0.08*np.sin(2*j+i),j-(n-1)/2+0.06*np.cos(i-3*j)) for j in range(n) for i in range(n)])\ne=np.array([(j*n+i,j*n+i+1) for j in range(n) for i in range(n-1)]+[(j*n+i,(j+1)*n+i) for j in range(n-1) for i in range(n)],dtype=int).reshape(-1,2)\nu=np.array([-.6+.09*np.cos(j) for j in range(n*n)])\nv=np.array([.07*np.sin(1.3*j) for j in range(n*n)])\ng=1+.14*np.cos(np.arange(n*n)*.7)\nz=np.array([.13,-.17,.05])\ng[::2]*=-1', 'call': 'topological_exposure(r,e,u,v,g,0.5,0.5,z,0.1,0.6,24)', 'gold_call': '_oracle_topological_exposure(r,e,u,v,g,0.5,0.5,z,0.1,0.6,24)'}, {'setup': 'import numpy as np\nn=3\nr=np.array([(i-(n-1)/2+0.08*np.sin(2*j+i),j-(n-1)/2+0.06*np.cos(i-3*j)) for j in range(n) for i in range(n)])\ne=np.array([(j*n+i,j*n+i+1) for j in range(n) for i in range(n-1)]+[(j*n+i,(j+1)*n+i) for j in range(n-1) for i in range(n)],dtype=int).reshape(-1,2)\nu=np.array([-.6+.09*np.cos(j) for j in range(n*n)])\nv=np.array([.07*np.sin(1.3*j) for j in range(n*n)])\ng=1+.14*np.cos(np.arange(n*n)*.7)\nz=np.array([.13,-.17,.05])\nv[:]=0; u[:]=-.6; r=np.round(r)', 'call': 'topological_exposure(r,e,u,v,g,0.5,0.5,z,0.1,0.8,24)', 'gold_call': '_oracle_topological_exposure(r,e,u,v,g,0.5,0.5,z,0.1,0.8,24)'}, {'setup': 'import numpy as np\nn=4\nr=np.array([(i-(n-1)/2+0.08*np.sin(2*j+i),j-(n-1)/2+0.06*np.cos(i-3*j)) for j in range(n) for i in range(n)])\ne=np.array([(j*n+i,j*n+i+1) for j in range(n) for i in range(n-1)]+[(j*n+i,(j+1)*n+i) for j in range(n-1) for i in range(n)],dtype=int).reshape(-1,2)\nu=np.array([-.6+.09*np.cos(j) for j in range(n*n)])\nv=np.array([.07*np.sin(1.3*j) for j in range(n*n)])\ng=1+.14*np.cos(np.arange(n*n)*.7)\nz=np.array([.13,-.17,.05])\nu[r[:,0]>0]=1.2; v[r[:,0]>0]=.2', 'call': 'topological_exposure(r,e,u,v,g,0.5,0.5,z,0.07,0.7,24)', 'gold_call': '_oracle_topological_exposure(r,e,u,v,g,0.5,0.5,z,0.07,0.7,24)'}]
