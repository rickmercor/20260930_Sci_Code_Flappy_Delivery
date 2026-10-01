"""
Compose the directional constitutive Fourier construction for a rectangular inclusion. First apply directional_factorization to inclusion P1 and host P0 with the x1 indicator and axis 0. Outside the x2 interval the first-stage host is kron(P0,I_k1), with k1 the x1 indicator size. Apply directional_factorization to these first-stage operators with the x2 indicator and axis 1. Return the final matrix in (component,n,m) order, where m is the old x1 harmonic and n the new x2 harmonic. Inputs P1 and P0 are finite complex 6 by 6 matrices with magnitudes at most 1e6; both indicator matrices satisfy the directional step's Hermitian/contractive tolerance 1e-10, have odd sizes from 1 to 9 and their size product is at most 25. Required inverses must exist and results must be finite. Invalid inputs raise ValueError.

The finite directional constructions do not commute. This operation represents x1-then-x2 factorization, without averaging it with the reverse order. At each x2 interface the normal block includes all retained x1 harmonics, so entrywise treatment of that block would change the finite constitutive operator.

Returns
-------
complex ndarray, finite rectangular constitutive operator
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rectangle_operator(p_in: "np.ndarray | list | tuple", p_out: "np.ndarray | list | tuple", tx: "np.ndarray | list | tuple", ty: "np.ndarray | list | tuple") -> "np.ndarray":
    """Compose the x1-then-x2 rectangular Fourier operator.

    Parameters
    ----------
    p_in, p_out : array_like
        Finite complex 6 by 6 material matrices, entry magnitudes at most 1e6.
    tx, ty : array_like
        Odd-sized Hermitian contractive indicators, size 1 through 9 each,
        with size product at most 25 and tolerance 1e-10.

    Returns
    -------
    ndarray
        Complex matrix ordered by component, x2 harmonic then x1 harmonic.

    Raises
    ------
    ValueError
        For invalid shapes or any violated directional factorization contract.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_rectangle_operator(p_in: "np.ndarray | list | tuple", p_out: "np.ndarray | list | tuple", tx: "np.ndarray | list | tuple", ty: "np.ndarray | list | tuple") -> "np.ndarray":
    try:pi,po,x,y=[np.asarray(a,dtype=complex) for a in (p_in,p_out,tx,ty)]
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError('array') from exc
    if pi.shape!=(6,6) or po.shape!=(6,6):raise ValueError('material shape')
    if any(a.ndim!=2 or a.shape[0]!=a.shape[1] or len(a)%2!=1 or not 1<=len(a)<=9 for a in (x,y)):raise ValueError('indicator shape')
    if len(x)*len(y)>25:raise ValueError('harmonic count')
    first=_oracle_directional_factorization(pi,po,x,0)
    return _oracle_directional_factorization(first,np.kron(po,np.eye(len(x))),y,1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base='import numpy as np\nrng=np.random.default_rng(212)\nz=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6));p=z@z.conj().T/6+np.eye(6)\nb=np.diag([2.,3.,4.,1.,1.2,1.5]).astype(complex)\n'
    cases=[
        {'setup':base+'x=.4*np.eye(3)+.03*np.ones((3,3));y=.3*np.eye(3)+.04*np.ones((3,3))','call':'rectangle_operator(p.copy(),b.copy(),x.copy(),y.copy())','gold_call':'_oracle_rectangle_operator(p,b,x,y)'},
        {'setup':base+'x=.4*np.eye(1)+.03*np.ones((1,1));y=.3*np.eye(1)+.04*np.ones((1,1))','call':'rectangle_operator(p.copy(),b.copy(),x.copy(),y.copy())','gold_call':'_oracle_rectangle_operator(p,b,x,y)'},
        {'setup':base+'x=.4*np.eye(5)+.03*np.ones((5,5));y=.3*np.eye(3)+.04*np.ones((3,3))','call':'rectangle_operator(p.copy(),b.copy(),x.copy(),y.copy())','gold_call':'_oracle_rectangle_operator(p,b,x,y)'}]
    for a in ('np.eye(5),np.eye(6),np.eye(1),np.eye(1)','np.eye(6),np.eye(6),np.eye(2),np.eye(1)','np.eye(6),np.eye(6),np.eye(9),np.eye(9)','np.eye(6),np.eye(6),2*np.eye(1),np.eye(1)'):
        setup='import numpy as np\n'
        for name,fn in [('run_model','rectangle_operator'),('run_gold','_oracle_rectangle_operator')]:
            setup+=f'def {name}():\n    try:\n        {fn}({a})\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    return cases
