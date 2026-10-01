"""
Assembles the reduced excess free energy per particle from the hard-sphere reference at the assigned diameter and the integrated tail strength.

The bulk excess free energy is the reference contribution plus a term quadratic in density carrying the integrated tail strength.

Returns
-------
A float64 array of shape (3,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stf_free_energy(rho: float, sigma: float, a_strength: float) -> "np.ndarray":
    r"""rho: positive float, number density.
    sigma: positive float, the effective hard-sphere diameter.
    a_strength: finite float, the reduced integrated tail strength.

    Returns a numpy float64 array of shape $(3,)$: the reduced excess free energy per particle
    $\beta F^{\rm exc}/N$ of the fluid, the reduced excess free energy per particle of the
    hard-sphere reference alone from the equation of state the source recommends, and that
    reference's reduced compressibility factor $\beta P/\rho$.

    Raises:
        ValueError: on non-finite or non-positive rho or sigma, non-finite a_strength, or a
            packing fraction at or above one.
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

def _oracle_stf_free_energy(rho: float, sigma: float, a_strength: float) -> "np.ndarray":
    p=_pos(rho,"rho"); s=_pos(sigma,"sigma"); a=float(a_strength)
    if not np.isfinite(a): raise ValueError("bad a_strength")
    eta=np.pi*p*s**3/6.0
    if eta>=1.0: raise ValueError("packing fraction out of range")
    f_cs=eta*(4.0-3.0*eta)/(1.0-eta)**2
    z_cs=(1.0+eta+eta**2-eta**3)/(1.0-eta)**3
    return np.array([f_cs+0.5*p*a,f_cs,z_cs],dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nP=(0.4,0.98,-7.26)\n","call":"stf_free_energy(*P)","gold_call":"_oracle_stf_free_energy(*P)"},
        {"setup":"import numpy as np\nP=(0.3,1.0,-5.1)\n","call":"stf_free_energy(*P)","gold_call":"_oracle_stf_free_energy(*P)"},
        {"setup":"import numpy as np\nP=(0.55,0.95,-8.4)\n","call":"stf_free_energy(*P)","gold_call":"_oracle_stf_free_energy(*P)"},
        {"setup":"import numpy as np\n# boundary: zero tail strength, the reference alone, near close packing\nP=(1.8,1.0,0.0)\n","call":"stf_free_energy(*P)","gold_call":"_oracle_stf_free_energy(*P)"},
        {"setup":"import numpy as np\n# invalid input: a packing fraction above one must raise ValueError\ndef _probe():\n    try:\n        stf_free_energy(2.0,1.0,-1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _expected_probe():\n    try:\n        _oracle_stf_free_energy(2.0,1.0,-1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe()","gold_call":"_expected_probe()"},
    ]
