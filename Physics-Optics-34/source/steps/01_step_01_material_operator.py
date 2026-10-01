"""
Assemble the relative constitutive operator P=[[epsilon,xi],[xi^dagger,mu]]+i eta I_6 in component order (E1,E2,E3,H1,H2,H3) for inputs and (D1,D2,D3,B1,B2,B3) for outputs. All three tensors are finite complex 3 by 3 arrays with entry magnitudes at most 1e6. eta is a finite real scalar in [0,1]. Return the complex 6 by 6 matrix; invalid inputs raise ValueError. The dagger denotes conjugate transpose, not elementwise conjugation.

Electric-magnetic coupling mixes the fields in both constitutive equations. For Hermitian positive input diagonal blocks and sufficiently small coupling, the Hermitian part represents a stable material at the specified frequency, while a positive eta supplies uniform passive loss. This routine assembles the supplied tensors without imposing an additional passivity test.

Returns
-------
complex ndarray of shape (6,6), relative constitutive operator
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def material_operator(epsilon: "np.ndarray | list | tuple", mu: "np.ndarray | list | tuple", xi: "np.ndarray | list | tuple", eta: float) -> "np.ndarray":
    """Assemble one electric-magnetic material matrix.

    Parameters
    ----------
    epsilon, mu, xi : array_like
        Complex 3 by 3 tensors with finite entries of magnitude at most 1e6.
    eta : float
        Real finite loss coefficient in [0,1].

    Returns
    -------
    ndarray
        Complex 6 by 6 constitutive matrix in the stated component order.

    Raises
    ------
    ValueError
        For invalid shape, magnitude, finiteness or loss coefficient.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_material_operator(epsilon: "np.ndarray | list | tuple", mu: "np.ndarray | list | tuple", xi: "np.ndarray | list | tuple", eta: float) -> "np.ndarray":
    try:
        arrays=[np.asarray(x,dtype=complex) for x in (epsilon,mu,xi)]
        loss=np.asarray(eta)
        if loss.ndim or np.iscomplexobj(loss):raise ValueError('loss')
        loss=float(loss)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError('invalid input') from exc
    if not np.isfinite(loss) or not 0<=loss<=1:raise ValueError('loss')
    if any(x.shape!=(3,3) or not np.all(np.isfinite(x)) or np.max(np.abs(x))>1e6 for x in arrays):raise ValueError('tensor')
    e,m,x=arrays
    return np.block([[e,x],[x.conj().T,m]])+1j*loss*np.eye(6)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    cases=[
        {'setup':'import numpy as np\ne=np.diag([2.,3.,4.]);m=np.eye(3);x=np.array([[.1j,.2,0],[.3j,0,.4],[0,.2j,.1]])','call':'material_operator(e.copy(),m.copy(),x.copy(),.025)','gold_call':'_oracle_material_operator(e,m,x,.025)'},
        {'setup':'import numpy as np\ne=np.eye(3);m=2*np.eye(3);x=np.zeros((3,3))','call':'material_operator(e.copy(),m.copy(),x.copy(),0)','gold_call':'_oracle_material_operator(e,m,x,0)'},
        {'setup':'import numpy as np\ne=1e6*np.eye(3);m=np.eye(3);x=1j*np.eye(3)','call':'material_operator(e.copy(),m.copy(),x.copy(),1)','gold_call':'_oracle_material_operator(e,m,x,1)'}]
    for args in ('np.eye(2),np.eye(3),np.eye(3),0','np.eye(3),np.eye(3),np.eye(3),float("inf")','np.eye(3),np.eye(3),np.eye(3),1j','np.full((3,3),np.nan),np.eye(3),np.eye(3),0'):
        setup='import numpy as np\n'
        for name,fn in [('run_model','material_operator'),('run_gold','_oracle_material_operator')]:
            setup+=f'def {name}():\n    try:\n        {fn}({args})\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    return cases
