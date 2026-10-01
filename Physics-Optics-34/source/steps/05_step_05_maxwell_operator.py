"""
Reduce the constitutive Fourier operator to the dimensionless longitudinal Maxwell matrix. Input P has shape (6N,6N) in (component,n,m) order, with N=(2mx+1)(2my+1), m fast, mx and my integers in [0,4] and N<=25. Its entry magnitudes are at most 1e6. The positive real period ratio a2/a1 is in [0.25,4], positive k0*a1 is in [0.25,20] and real k1/k0 and k2/k0 lie in [-2,2]. Define X=diag(k1/k0+2 pi m/(k0*a1)), Y=diag(k2/k0+2 pi n/((a2/a1)(k0*a1))) and tangential order t=(E1,E2,H1,H2). Normal order is z=(E3,H3). Let C_z=[[0,0,Y,-X],[-Y,X,0,0]] and Z=Pzz^{-1}(C_z-Pzt); its row halves are Z_E and Z_H. Let U=Ptt+Ptz Z and split its rows as U_D1,U_D2,U_B1,U_B2. Return rows [U_B2+X Z_E;-U_B1+Y Z_E;-U_D2+X Z_H;U_D1+Y Z_H]. No condition-number cutoff applies. All inputs and intermediate/results must be finite; a singular Pzz or invalid input raises ValueError.

The longitudinal components of the curl equations contain no longitudinal propagation eigenvalue and can be eliminated exactly at fixed Fourier truncation. For exp(-i omega t) the equations are k cross E=k0 B and k cross H=-k0 D. Their remaining tangential rows define an ordinary eigenproblem with eigenvalues k3/k0; imposing a physical-mode filter would be a separate operation and is not done here.

Returns
-------
complex ndarray, dimensionless tangential Maxwell matrix
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def maxwell_operator(p: "np.ndarray | list | tuple", mx: int, my: int, period_ratio: float, frequency: float, qx: float, qy: float) -> "np.ndarray":
    """Return the dimensionless tangential propagation matrix.

    Parameters
    ----------
    p : array_like
        Finite complex (6N,6N) constitutive operator, magnitude at most 1e6.
    mx, my : int
        Retained orders 0 through 4, excluding booleans; N<=25.
    period_ratio : float
        a2/a1 in [0.25,4].
    frequency : float
        k0*a1 in [0.25,20].
    qx, qy : float
        k1/k0 and k2/k0 in [-2,2].

    Returns
    -------
    ndarray
        Complex (4N,4N) matrix with eigenvalues k3/k0.

    Raises
    ------
    ValueError
        For invalid inputs, singular longitudinal block or nonfinite arithmetic.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_maxwell_operator(p: "np.ndarray | list | tuple", mx: int, my: int, period_ratio: float, frequency: float, qx: float, qy: float) -> "np.ndarray":
    if any(isinstance(a,(bool,np.bool_)) or not isinstance(a,(int,np.integer)) or not 0<=a<=4 for a in (mx,my)):raise ValueError('order')
    ng=(2*mx+1)*(2*my+1)
    if ng>25:raise ValueError('size')
    try:
        p=np.asarray(p,dtype=complex)
        vals=[]
        for a in (period_ratio,frequency,qx,qy):
            x=np.asarray(a)
            if x.ndim or np.iscomplexobj(x):raise ValueError('scalar')
            vals.append(float(x))
        r,f,qx,qy=vals
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError('input') from exc
    if not all(np.isfinite(vals)) or not .25<=r<=4 or not .25<=f<=20 or not -2<=qx<=2 or not -2<=qy<=2:raise ValueError('range')
    if p.shape!=(6*ng,6*ng) or not np.all(np.isfinite(p)) or np.max(np.abs(p))>1e6:raise ValueError('matrix')
    ids=np.arange(6*ng).reshape(6,ng);normal=ids[[2,5]].ravel();tang=ids[[0,1,3,4]].ravel()
    ms=np.tile(np.arange(-mx,mx+1),2*my+1);ns=np.repeat(np.arange(-my,my+1),2*mx+1)
    x=np.diag(qx+2*np.pi*ms/f);y=np.diag(qy+2*np.pi*ns/(r*f));zero=np.zeros((ng,ng))
    cz=np.block([[zero,zero,y,-x],[-y,x,zero,zero]])
    try:
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            z=np.linalg.solve(p[np.ix_(normal,normal)],cz-p[np.ix_(normal,tang)])
            u=p[np.ix_(tang,tang)]+p[np.ix_(tang,normal)]@z
            dx,dy,bx,by=np.split(u,4,axis=0);ez,hz=np.split(z,2,axis=0)
            out=np.vstack([by+x@ez,-bx+y@ez,-dy+x@hz,dx+y@hz])
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(out)):raise ValueError('nonfinite')
    except np.linalg.LinAlgError as exc:raise ValueError('singular') from exc
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base='import numpy as np\nrng=np.random.default_rng(833)\nz=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6));a=z@z.conj().T/6+np.eye(6)+.02j*np.eye(6)\n'
    cases=[
        {'setup':base+'p=np.kron(a,np.eye(9))','call':'maxwell_operator(p.copy(),1,1,1.3,3.3,.21,-.17)','gold_call':'_oracle_maxwell_operator(p,1,1,1.3,3.3,.21,-.17)'},
        {'setup':base+'p=np.kron(a,np.eye(1))','call':'maxwell_operator(p.copy(),0,0,1,1,0,0)','gold_call':'_oracle_maxwell_operator(p,0,0,1,1,0,0)'},
        {'setup':base+'p=np.kron(a,np.eye(15))','call':'maxwell_operator(p.copy(),2,1,4,.25,-2,2)','gold_call':'_oracle_maxwell_operator(p,2,1,4,.25,-2,2)'}]
    for args in ('np.eye(6),True,0,1,1,0,0','np.eye(6),0,0,1,0,0,0','np.zeros((6,6)),0,0,1,1,0,0','np.eye(6),0,0,1,1,float("nan"),0','np.eye(5),0,0,1,1,0,0'):
        setup='import numpy as np\n'
        for name,fn in [('run_model','maxwell_operator'),('run_gold','_oracle_maxwell_operator')]:
            setup+=f'def {name}():\n    try:\n        {fn}({args})\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    return cases
