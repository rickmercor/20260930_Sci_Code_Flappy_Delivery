"""
Computes the integrated strength of the tail interaction by the thermodynamic-integration route and by the second-virial route.

The tail enters the bulk free energy through a single integrated strength. Which correlation function weights the tail inside that integral is the source's choice, and the virial route offers a dilute-limit check on it.

Returns
-------
A float64 array of shape (2,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stf_integrated_strength(g_target: "np.ndarray", bphi_ref: "np.ndarray", bphi_tail: "np.ndarray") -> "np.ndarray":
    r"""g_target: $(1024,)$ array, the measured radial distribution function.
    bphi_ref, bphi_tail: $(1024,)$ arrays, the reduced reference and tail potentials.

    Returns a numpy float64 array of shape $(2,)$: the reduced integrated interaction strength
    of the tail by the thermodynamic-integration route with the approximation the source adopts
    for the correlation weight, and the reduced integrated strength by the second-virial route the
    source gives as its dilute-limit alternative. Both are in units of $k_B T\,\sigma^3$.

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

def _oracle_stf_integrated_strength(g_target: "np.ndarray", bphi_ref: "np.ndarray", bphi_tail: "np.ndarray") -> "np.ndarray":
    if np.asarray(g_target).size!=1024: raise ValueError("g_target must have length 1024")
    g=_arr(g_target,"g_target"); br=_arr(bphi_ref,"bphi_ref"); bt=_arr(bphi_tail,"bphi_tail")
    r,_,_=_grid()
    a_ti=4.0*np.pi*_tz(r*r*g*bt,r)
    ef=np.exp(-np.clip(br+bt,-60.0,60.0)); er=np.exp(-np.clip(br,-60.0,60.0))
    b2=-2.0*np.pi*_tz((ef-1.0)*r*r,r); b2r=-2.0*np.pi*_tz((er-1.0)*r*r,r)
    return np.array([a_ti,2.0*(b2-b2r)],dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\ng=_oracle_stf_target_rdf(0.4,0.5,3.5,400)\nb=_oracle_stf_ibi_potential(g,0.4,0.2,60)\ns=_oracle_stf_wca_split(b,g)\n","call":"stf_integrated_strength(g,s[0],s[1])","gold_call":"_oracle_stf_integrated_strength(g,s[0],s[1])"},
        {"setup":"import numpy as np\ng=_oracle_stf_target_rdf(0.3,0.5,3.5,300)\nb=_oracle_stf_ibi_potential(g,0.3,0.25,40)\ns=_oracle_stf_wca_split(b,g)\n","call":"stf_integrated_strength(g,s[0],s[1])","gold_call":"_oracle_stf_integrated_strength(g,s[0],s[1])"},
        {"setup":"import numpy as np\ng=_oracle_stf_target_rdf(0.5,0.6,3.0,300)\nb=_oracle_stf_ibi_potential(g,0.5,0.15,50)\ns=_oracle_stf_wca_split(b,g)\n","call":"stf_integrated_strength(g,s[0],s[1])","gold_call":"_oracle_stf_integrated_strength(g,s[0],s[1])"},
        {"setup":"import numpy as np\n# boundary: a vanishing tail gives zero strength by both routes\ng=_oracle_stf_target_rdf(0.4,0.5,3.5,400)\nb=_oracle_stf_ibi_potential(g,0.4,0.2,60)\ns=_oracle_stf_wca_split(b,g)\n","call":"stf_integrated_strength(g,s[0],np.zeros(1024))","gold_call":"_oracle_stf_integrated_strength(g,s[0],np.zeros(1024))"},
        {"setup":"import numpy as np\n# invalid input: a structure of the wrong length must raise ValueError\ndef _probe():\n    try:\n        stf_integrated_strength(np.ones(5),np.zeros(1024),np.zeros(1024))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _expected_probe():\n    try:\n        _oracle_stf_integrated_strength(np.ones(5),np.zeros(1024),np.zeros(1024))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe()","gold_call":"_expected_probe()"},
    ]
