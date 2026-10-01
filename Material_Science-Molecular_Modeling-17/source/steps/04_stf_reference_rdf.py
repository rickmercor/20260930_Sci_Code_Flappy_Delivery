"""
Solves for the radial distribution function of the reference (core-only) system by a fixed-count integral-equation solve.

The reference structure is what the effective hard-sphere size is fitted against; the comparison structures come from the closed-form hard-sphere solution of the same closure.

Returns
-------
A float64 array of shape (1024,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stf_reference_rdf(bphi_ref: "np.ndarray", rho: float, n_iter: int) -> "np.ndarray":
    r"""bphi_ref: $(1024,)$ array, the reduced reference potential on the fixed grid.
    rho: positive float, number density.
    n_iter: positive integer, number of fixed-point iterations.

    Returns a numpy float64 array of shape $(1024,)$: the radial distribution function of the
    reference system, from the Ornstein-Zernike relation with the closure the source prescribes
    for interactions that carry a hard core, by a Picard iteration of exactly n_iter steps with
    mixing 0.15 and no tolerance exit.

    Raises:
        ValueError: on a bphi_ref not of length 1024 or not finite, non-finite or non-positive
            rho, or non-integral or non-positive n_iter.
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

def _oracle_stf_reference_rdf(bphi_ref: "np.ndarray", rho: float, n_iter: int) -> "np.ndarray":
    b=_arr(bphi_ref,"bphi_ref"); p=_pos(rho,"rho"); ni=float(n_iter)
    if not np.isfinite(ni) or ni<=0 or abs(ni-round(ni))>0: raise ValueError("bad n_iter")
    return _oz(b,p,"PY",int(round(ni)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\ng=_oracle_stf_target_rdf(0.4,0.5,3.5,400)\nb=_oracle_stf_ibi_potential(g,0.4,0.2,60)\ns=_oracle_stf_wca_split(b,g)\n","call":"stf_reference_rdf(s[0],0.4,300)","gold_call":"_oracle_stf_reference_rdf(s[0],0.4,300)"},
        {"setup":"import numpy as np\nr=(np.arange(1024)+1)*0.02\nb=np.where(r<0.95,60.0,0.0)\n","call":"stf_reference_rdf(b,0.35,300)","gold_call":"_oracle_stf_reference_rdf(b,0.35,300)"},
        {"setup":"import numpy as np\nr=(np.arange(1024)+1)*0.02\nb=np.where(r<1.0,60.0,0.0)\n","call":"stf_reference_rdf(b,0.5,250)","gold_call":"_oracle_stf_reference_rdf(b,0.5,250)"},
        {"setup":"import numpy as np\n# boundary: a vanishing potential is the ideal gas, g equal to one everywhere\nb=np.zeros(1024)\n","call":"stf_reference_rdf(b,0.3,100)","gold_call":"_oracle_stf_reference_rdf(b,0.3,100)"},
        {"setup":"import numpy as np\n# invalid input: a negative iteration count must raise ValueError\nb=np.zeros(1024)\ndef _probe():\n    try:\n        stf_reference_rdf(b,0.3,-5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _expected_probe():\n    try:\n        _oracle_stf_reference_rdf(b,0.3,-5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe()","gold_call":"_expected_probe()"},
    ]
