"""
Generates the structural input, the radial distribution function of a cut-and-shifted 12-6 fluid, by a fixed-count integral-equation solve on the fixed radial grid.

Everything downstream is inferred from this one function alone, as if it had been measured; the potential used to generate it is treated as unknown from here on.

Returns
-------
A float64 array of shape (1024,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stf_target_rdf(rho: float, beta_eps: float, rc: float, n_iter: int) -> "np.ndarray":
    r"""rho: positive float, number density in units of sigma^-3.
    beta_eps: positive float, well depth of the 12-6 Mie potential in units of k_B T.
    rc: positive float, cut-and-shift radius in units of sigma.
    n_iter: positive integer, number of fixed-point iterations of the integral-equation solve.

    Returns a numpy float64 array of shape $(1024,)$: the radial distribution function of the
    cut-and-shifted 12-6 fluid on the fixed radial grid $r_i = 0.02\,i$, $i = 1..1024$, obtained by
    solving the Ornstein-Zernike relation with the closure the source prescribes for soft
    interactions, by a Picard iteration of exactly n_iter steps with mixing 0.15 and no tolerance
    exit. The iteration mixes the indirect correlation function $h - c$ from a zero start, and
    every three-dimensional Fourier transform on the grid is the type-1 discrete sine transform of
    $r f(r)$ with wavenumbers $k_j = \pi j / (1025 \times 0.02)$, $j = 1..1024$. Negative values
    are clipped to zero. This array is the structural input to every later step; the potential
    that generated it must not be used downstream.

    Raises:
        ValueError: on non-finite or non-positive rho, beta_eps or rc, or non-integral or
            non-positive n_iter.
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

def _oracle_stf_target_rdf(rho: float, beta_eps: float, rc: float, n_iter: int) -> "np.ndarray":
    p=_pos(rho,"rho"); be=_pos(beta_eps,"beta_eps"); c=_pos(rc,"rc")
    ni=float(n_iter)
    if not np.isfinite(ni) or ni<=0 or abs(ni-round(ni))>0: raise ValueError("bad n_iter")
    r,_,_=_grid()
    u=4.0*be*((1.0/r)**12-(1.0/r)**6); uc=4.0*be*((1.0/c)**12-(1.0/c)**6)
    bphi=np.where(r<c,u-uc,0.0)
    return np.maximum(_oz(bphi,p,"HNC",int(round(ni))),0.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nP=(0.4,0.5,3.5,400)\n","call":"stf_target_rdf(*P)","gold_call":"_oracle_stf_target_rdf(*P)"},
        {"setup":"import numpy as np\nP=(0.3,0.5,3.5,300)\n","call":"stf_target_rdf(*P)","gold_call":"_oracle_stf_target_rdf(*P)"},
        {"setup":"import numpy as np\nP=(0.5,0.6,3.0,300)\n","call":"stf_target_rdf(*P)","gold_call":"_oracle_stf_target_rdf(*P)"},
        {"setup":"import numpy as np\n# boundary: dilute state whose structure has a single peak\nP=(0.05,0.5,3.5,200)\n","call":"stf_target_rdf(*P)","gold_call":"_oracle_stf_target_rdf(*P)"},
        {"setup":"import numpy as np\n# invalid input: zero iterations must raise ValueError\ndef _probe():\n    try:\n        stf_target_rdf(0.4,0.5,3.5,0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _expected_probe():\n    try:\n        _oracle_stf_target_rdf(0.4,0.5,3.5,0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe()","gold_call":"_expected_probe()"},
    ]
