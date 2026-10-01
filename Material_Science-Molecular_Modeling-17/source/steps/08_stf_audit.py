"""
Runs the whole study from the generated structure to the excess free energy, reporting every intermediate the source's construction depends on.

Comparing the two diameters and the two integrated strengths side by side shows how much of the final free energy rides on the choices the source makes at each stage.

Returns
-------
A float64 array of shape (9,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stf_audit(rho: float, beta_eps: float, rc: float, gamma: float, n_ibi: int) -> "np.ndarray":
    r"""rho, beta_eps, rc: as in the first step. gamma, n_ibi: as in the second step.

    The orchestrator. It must call the earlier functions rather than reimplementing them, using
    400 iterations to generate the structural input and 300 for every reference-system solve.
    Returns a numpy float64 array of shape $(9,)$: the split radius, the comparison cutoff, the
    effective hard-sphere diameter by the source's procedure, the diameter by the conventional
    prescription, the integrated strength by the thermodynamic-integration route, the integrated
    strength by the second-virial route, the hard-sphere reference free energy per particle at the
    source's diameter, the reduced excess free energy per particle using the thermodynamic-
    integration strength, and the same using the second-virial strength.

    Raises:
        ValueError: whenever any of the functions it calls would raise.
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

def _oracle_stf_audit(rho: float, beta_eps: float, rc: float, gamma: float, n_ibi: int) -> "np.ndarray":
    for v,nm in ((rho,"rho"),(beta_eps,"beta_eps"),(rc,"rc"),(gamma,"gamma")):
        if not np.isfinite(float(v)) or float(v)<=0.0: raise ValueError("bad "+nm)
    r,_,_=_grid()
    g=_oracle_stf_target_rdf(rho,beta_eps,rc,400)
    b=_oracle_stf_ibi_potential(g,rho,gamma,n_ibi)
    sp=_oracle_stf_wca_split(b,g)
    Rm=float(r[int(np.argmax(sp[0]==0.0))])
    gr=_oracle_stf_reference_rdf(sp[0],rho,300)
    d=_oracle_stf_hs_diameter(gr,g,sp[0],rho)
    A=_oracle_stf_integrated_strength(g,sp[0],sp[1])
    f_ti=_oracle_stf_free_energy(rho,d[0],A[0]); f_b2=_oracle_stf_free_energy(rho,d[0],A[1])
    return np.array([Rm,d[1],d[0],d[2],A[0],A[1],f_ti[1],f_ti[0],f_b2[0]],dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nP=(0.4,0.5,3.5,0.2,800)\n","call":"stf_audit(*P)","gold_call":"_oracle_stf_audit(*P)"},
        {"setup":"import numpy as np\nP=(0.3,0.5,3.5,0.2,300)\n","call":"stf_audit(*P)","gold_call":"_oracle_stf_audit(*P)"},
        {"setup":"import numpy as np\nP=(0.5,0.6,3.0,0.2,300)\n","call":"stf_audit(*P)","gold_call":"_oracle_stf_audit(*P)"},
        {"setup":"import numpy as np\n# boundary: dilute single-peak state\nP=(0.05,0.5,3.5,0.2,200)\n","call":"stf_audit(*P)","gold_call":"_oracle_stf_audit(*P)"},
        {"setup":"import numpy as np\n# invalid input: zero inversion steps must raise ValueError\ndef _probe():\n    try:\n        stf_audit(0.4,0.5,3.5,0.2,0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _expected_probe():\n    try:\n        _oracle_stf_audit(0.4,0.5,3.5,0.2,0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe()","gold_call":"_expected_probe()"},
    ]
