"""
Assigns the effective hard-sphere diameter by the source's structure-matching procedure, reports the comparison cutoff it uses, and the diameter the conventional prescription would give.

Mapping the repulsive core onto a hard-sphere fluid lets a known equation of state supply the reference free energy. The source claims the way it chooses that diameter as a novelty of its approach.

Returns
-------
A float64 array of shape (3,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stf_hs_diameter(g_ref: "np.ndarray", g_target: "np.ndarray", bphi_ref: "np.ndarray", rho: float) -> "np.ndarray":
    r"""g_ref: $(1024,)$ array, the reference-system radial distribution function.
    g_target: $(1024,)$ array, the measured radial distribution function.
    bphi_ref: $(1024,)$ array, the reduced reference potential.
    rho: positive float, number density.

    Returns a numpy float64 array of shape $(3,)$: the effective hard-sphere diameter the
    source's procedure assigns to this structure, the cutoff distance below which that procedure
    compares structures, and for comparison the diameter the conventional prescription would give
    from bphi_ref. The source's procedure scans diameters from 0.86 to 1.04 on 19 equally spaced
    points, then refines the best bracket by exactly 12 golden-section steps and returns the
    bracket midpoint; hard-sphere structures for the comparison come from the closed-form
    solution of the hard-core closure for hard spheres in $k$ space (Wertheim), carried through
    the Ornstein-Zernike relation and transformed onto the grid. Peaks of the measured function
    are strict local maxima that exceed 1.02; the measured function is used as it stands at every
    distance and the cutoff serves only as the upper limit of the comparison. The conventional
    diameter is the integral from zero of one minus the Boltzmann factor of bphi_ref, with the
    integrand taken as one at the origin. Integrals are trapezoid sums on the grid.

    Raises:
        ValueError: on arrays not of length 1024 or not finite, or non-finite or non-positive rho.
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

def _oracle_stf_hs_diameter(g_ref: "np.ndarray", g_target: "np.ndarray", bphi_ref: "np.ndarray", rho: float) -> "np.ndarray":
    if not np.isfinite(float(rho)) or float(rho)<=0.0: raise ValueError("bad rho")
    gr=_arr(g_ref,"g_ref"); gt=_arr(g_target,"g_target"); br=_arr(bphi_ref,"bphi_ref"); p=_pos(rho,"rho")
    r,_,_=_grid(); n=r.size
    pk=[i for i in range(2,n-2) if gt[i]>gt[i-1] and gt[i]>gt[i+1] and gt[i]>1.02]
    if len(pk)>1:
        rp=float(np.mean(np.diff(r[pk]))); rc=float(r[pk[-1]]+rp)
    else:
        rc=float(2.0*r[pk[0]])
    m=r<=rc
    def _D(s):
        gh=_hs_rdf(s,p)
        return _tz((gr[m]-gh[m])**2,r[m])
    grid=np.linspace(0.86,1.04,19); vals=[_D(s) for s in grid]; i=int(np.argmin(vals))
    a=grid[max(i-1,0)]; b=grid[min(i+1,18)]; phi=(np.sqrt(5.0)-1.0)/2.0
    x1=b-phi*(b-a); x2=a+phi*(b-a); f1=_D(x1); f2=_D(x2)
    for _ in range(12):
        if f1<f2: b,x2,f2=x2,x1,f1; x1=b-phi*(b-a); f1=_D(x1)
        else:     a,x1,f1=x1,x2,f2; x2=a+phi*(b-a); f2=_D(x2)
    s_opt=float((a+b)/2.0)
    f=1.0-np.exp(-np.clip(br,-60.0,60.0))
    s_bh=_tz(np.concatenate(([1.0],f)),np.concatenate(([0.0],r)))
    return np.array([s_opt,rc,s_bh],dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ng=_oracle_stf_target_rdf(0.4,0.5,3.5,400)\nb=_oracle_stf_ibi_potential(g,0.4,0.2,60)\ns=_oracle_stf_wca_split(b,g)\ngr=_oracle_stf_reference_rdf(s[0],0.4,300)\n', 'call': 'stf_hs_diameter(gr,g,s[0],0.4)', 'gold_call': '_oracle_stf_hs_diameter(gr,g,s[0],0.4)'},
        {'setup': 'import numpy as np\ng=_oracle_stf_target_rdf(0.3,0.5,3.5,300)\nb=_oracle_stf_ibi_potential(g,0.3,0.25,40)\ns=_oracle_stf_wca_split(b,g)\ngr=_oracle_stf_reference_rdf(s[0],0.3,300)\n', 'call': 'stf_hs_diameter(gr,g,s[0],0.3)', 'gold_call': '_oracle_stf_hs_diameter(gr,g,s[0],0.3)'},
        {'setup': 'import numpy as np\ng=_oracle_stf_target_rdf(0.5,0.6,3.0,300)\nb=_oracle_stf_ibi_potential(g,0.5,0.15,50)\ns=_oracle_stf_wca_split(b,g)\ngr=_oracle_stf_reference_rdf(s[0],0.5,300)\n', 'call': 'stf_hs_diameter(gr,g,s[0],0.5)', 'gold_call': '_oracle_stf_hs_diameter(gr,g,s[0],0.5)'},
        {'setup': 'import numpy as np\n# boundary: a single-peak structure, where the cutoff rule takes its other branch\ng=_oracle_stf_target_rdf(0.05,0.5,3.5,200)\nb=_oracle_stf_ibi_potential(g,0.05,0.2,60)\ns=_oracle_stf_wca_split(b,g)\ngr=_oracle_stf_reference_rdf(s[0],0.05,300)\n', 'call': 'stf_hs_diameter(gr,g,s[0],0.05)', 'gold_call': '_oracle_stf_hs_diameter(gr,g,s[0],0.05)'},
        {'setup': 'import numpy as np\n# invalid input: a zero density must raise ValueError\ng=_oracle_stf_target_rdf(0.4,0.5,3.5,400)\nb=_oracle_stf_ibi_potential(g,0.4,0.2,60)\ns=_oracle_stf_wca_split(b,g)\ngr=_oracle_stf_reference_rdf(s[0],0.4,300)\ndef _exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_exception_code(stf_hs_diameter, gr, g, s[0], 0.0)', 'gold_call': '_exception_code(_oracle_stf_hs_diameter, gr, g, s[0], 0.0)'},
    ]
