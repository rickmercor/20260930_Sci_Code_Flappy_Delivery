"""
Splits the reconstructed potential into a short-range reference part and a tail part at the position of its minimum.

The reference part isolates the excluded-volume core that dominates the structure; the tail is the remainder and is treated perturbatively.

Returns
-------
A float64 array of shape (2, 1024).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stf_wca_split(bphi: "np.ndarray", g_target: "np.ndarray") -> "np.ndarray":
    r"""bphi: $(1024,)$ array, reduced effective pair potential on the fixed grid.
    g_target: $(1024,)$ array, the radial distribution function used to locate the core.

    Returns a numpy float64 array of shape $(2, 1024)$: row 0 the reduced reference (core)
    potential and row 1 the reduced tail potential, split at the position of the minimum of
    bphi outside the core, following the decomposition the source adopts. The two rows sum to
    bphi at every grid point. What each row takes inside the split radius follows the source.

    Raises:
        ValueError: on arrays not of length 1024 or not finite.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.fft import dst as _dst

def _grid():
    n=1024; dr=0.02; r=(np.arange(n)+1)*dr; k=np.pi*(np.arange(n)+1)/((n+1)*dr)
    return r,k,dr
def _tz(y,x):
    y=np.asarray(y,dtype=np.float64); x=np.asarray(x,dtype=np.float64)
    return float(np.sum((y[1:]+y[:-1])*(x[1:]-x[:-1]))*0.5)
def _ft3(f):
    r,k,dr=_grid(); return 4.0*np.pi*_dst(r*f,type=1)*dr/(2.0*k)
def _ift3(F):
    r,k,dr=_grid(); return _dst(k*F,type=1)*(k[0]/(2.0*np.pi**2))/(2.0*r)
def _oz(bphi, rho, closure, n_iter, mix=0.15):
    b=np.clip(np.asarray(bphi,dtype=np.float64),-60.0,60.0); e=np.exp(-b); gam=np.zeros(b.size)
    def _cf(g): return (1.0+g)*(e-1.0) if closure=="PY" else np.exp(-b+g)-g-1.0
    for _ in range(int(n_iter)):
        c=_cf(gam); ck=_ft3(c); hk=ck/(1.0-rho*ck); gam=(1.0-mix)*gam+mix*_ift3(hk-ck)
    return 1.0+gam+_cf(gam)
def _hs_ck(k, sigma, rho):
    eta=np.pi*rho*sigma**3/6.0; q=k*sigma
    a=(1.0+2.0*eta)**2/(1.0-eta)**4; b=-6.0*eta*(1.0+0.5*eta)**2/(1.0-eta)**4; c=0.5*eta*a
    sq=np.sin(q); cq=np.cos(q)
    t=a*q**3*(sq-q*cq)+b*q**2*(2.0*q*sq+(2.0-q*q)*cq-2.0)+c*((4.0*q**3-24.0*q)*sq-(q**4-12.0*q*q+24.0)*cq+24.0)
    return -4.0*np.pi*sigma**3*t/q**6
def _hs_rdf(sigma, rho):
    r,k,dr=_grid(); ck=_hs_ck(k,sigma,rho); hk=ck/(1.0-rho*ck)
    return 1.0+_ift3(hk)
def _pos(x,name):
    v=float(x)
    if not np.isfinite(v) or v<=0.0: raise ValueError("bad "+name)
    return v
def _arr(x,name):
    a=np.asarray(x,dtype=np.float64).ravel()
    if a.size!=1024 or not np.all(np.isfinite(a)): raise ValueError("bad "+name)
    return a

def _oracle_stf_wca_split(bphi: "np.ndarray", g_target: "np.ndarray") -> "np.ndarray":
    if np.asarray(bphi).size!=1024 or np.asarray(g_target).size!=1024: raise ValueError("arrays must have length 1024")
    b=_arr(bphi,"bphi"); g=_arr(g_target,"g_target"); r,_,_=_grid()
    core=g<1e-6; iR=int(np.argmin(np.where(core,np.inf,b))); Rm=r[iR]; bR=b[iR]
    bref=np.where(r<Rm,b-bR,0.0); btail=np.where(r<Rm,bR,b)
    return np.vstack([bref,btail]).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ng=_oracle_stf_target_rdf(0.4,0.5,3.5,400)\nb=_oracle_stf_ibi_potential(g,0.4,0.2,60)\n', 'call': 'stf_wca_split(b,g)', 'gold_call': '_oracle_stf_wca_split(b,g)'},
        {'setup': 'import numpy as np\ng=_oracle_stf_target_rdf(0.3,0.5,3.5,300)\nb=_oracle_stf_ibi_potential(g,0.3,0.25,40)\n', 'call': 'stf_wca_split(b,g)', 'gold_call': '_oracle_stf_wca_split(b,g)'},
        {'setup': 'import numpy as np\ng=_oracle_stf_target_rdf(0.5,0.6,3.0,300)\nb=_oracle_stf_ibi_potential(g,0.5,0.15,50)\n', 'call': 'stf_wca_split(b,g)', 'gold_call': '_oracle_stf_wca_split(b,g)'},
        {'setup': 'import numpy as np\n# boundary: the low-density inversion itself, one update only\ng=_oracle_stf_target_rdf(0.05,0.5,3.5,200)\nb=_oracle_stf_ibi_potential(g,0.05,0.2,1)\n', 'call': 'stf_wca_split(b,g)', 'gold_call': '_oracle_stf_wca_split(b,g)'},
        {'setup': 'import numpy as np\n# invalid input: a potential of the wrong length must raise ValueError\ng=_oracle_stf_target_rdf(0.4,0.5,3.5,400)\ndef _exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_exception_code(stf_wca_split, np.zeros(10), g)', 'gold_call': '_exception_code(_oracle_stf_wca_split, np.zeros(10), g)'},
    ]
