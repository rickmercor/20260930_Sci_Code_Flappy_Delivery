"""
Reconstructs the reduced effective pair potential from the radial distribution function by a fixed number of iterative Boltzmann inversion steps.

At finite density the low-density inversion neglects many-body correlations, so the potential is refined iteratively against the target structure. The reconstruction is only an intermediate: its purpose is to expose the repulsive core so that an effective particle size can be assigned.

Returns
-------
A float64 array of shape (1024,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stf_ibi_potential(g_target: "np.ndarray", rho: float, gamma: float, n_ibi: int) -> "np.ndarray":
    r"""g_target: $(1024,)$ array, the radial distribution function on the fixed grid.
    rho: positive float, number density.
    gamma: positive float, the mixing parameter of the inversion update.
    n_ibi: positive integer, number of inversion iterations.

    Returns a numpy float64 array of shape $(1024,)$: the reduced effective pair potential
    $\beta\phi(r)$ reconstructed from g_target by iterative Boltzmann inversion, started from the
    low-density inversion and updated exactly n_ibi times with the closure the source prescribes
    for interactions that carry a hard core. Inside the core, where g_target is below 1e-6, the
    potential is held at 60. The inner integral-equation solve is warm-started between inversion
    steps and takes exactly 80 fixed-point iterations with mixing 0.15 per step.

    Raises:
        ValueError: on a g_target that is not of length 1024 or not finite, on non-finite or
            non-positive rho or gamma, or non-integral or non-positive n_ibi.
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

def _oracle_stf_ibi_potential(g_target: "np.ndarray", rho: float, gamma: float, n_ibi: int) -> "np.ndarray":
    g=_arr(g_target,"g_target"); p=_pos(rho,"rho"); gm=_pos(gamma,"gamma")
    ni=float(n_ibi)
    if not np.isfinite(ni) or ni<=0 or abs(ni-round(ni))>0: raise ValueError("bad n_ibi")
    gt=np.maximum(g,1e-10); core=g<1e-6
    bphi=np.where(core,60.0,-np.log(gt)); gam=np.zeros(g.size)
    for _ in range(int(round(ni))):
        e=np.exp(-np.clip(bphi,-60.0,60.0))
        for _ in range(80):
            c=(1.0+gam)*(e-1.0); ck=_ft3(c); hk=ck/(1.0-p*ck); gam=0.85*gam+0.15*_ift3(hk-ck)
        gn=np.maximum(1.0+gam+(1.0+gam)*(e-1.0),1e-10)
        bphi=bphi+gm*np.log(gn/gt); bphi=np.where(core,60.0,bphi)
    return bphi

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ng=_oracle_stf_target_rdf(0.4,0.5,3.5,400)\n', 'call': 'stf_ibi_potential(g,0.4,0.2,60)', 'gold_call': '_oracle_stf_ibi_potential(g,0.4,0.2,60)'},
        {'setup': 'import numpy as np\ng=_oracle_stf_target_rdf(0.3,0.5,3.5,300)\n', 'call': 'stf_ibi_potential(g,0.3,0.25,40)', 'gold_call': '_oracle_stf_ibi_potential(g,0.3,0.25,40)'},
        {'setup': 'import numpy as np\ng=_oracle_stf_target_rdf(0.5,0.6,3.0,300)\n', 'call': 'stf_ibi_potential(g,0.5,0.15,50)', 'gold_call': '_oracle_stf_ibi_potential(g,0.5,0.15,50)'},
        {'setup': 'import numpy as np\n# boundary: a single inversion step from the low-density start\ng=_oracle_stf_target_rdf(0.05,0.5,3.5,200)\n', 'call': 'stf_ibi_potential(g,0.05,0.2,1)', 'gold_call': '_oracle_stf_ibi_potential(g,0.05,0.2,1)'},
        {'setup': 'import numpy as np\n# invalid input: zero mixing must raise ValueError\ng=_oracle_stf_target_rdf(0.4,0.5,3.5,400)\ndef _exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_exception_code(stf_ibi_potential, g, 0.4, 0.0, 10)', 'gold_call': '_exception_code(_oracle_stf_ibi_potential, g, 0.4, 0.0, 10)'},
    ]
