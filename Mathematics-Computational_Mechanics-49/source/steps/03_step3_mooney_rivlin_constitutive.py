"""
Evaluates the first Piola-Kirchhoff stress and exact consistent tangent actions.

For a hyperelastic material, the first Piola-Kirchhoff stress and material tangent must come from the same strain-energy density. The tangent action combines derivatives of the invariants, volumetric terms, and $F^{-T}$, so using the state from Step 2 avoids finite-difference approximations while maintaining the exact Newton linearization.

Returns
-------
tuple[np.ndarray, np.ndarray], stress (3,3) and tangent actions (2,3,3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def mooney_rivlin_constitutive(F: np.ndarray, directions: np.ndarray,
                                     state: np.ndarray, differential_state: np.ndarray,
                                     c1: float, c2: float, kappa: float):
    """Evaluate the first Piola-Kirchhoff stress and exact consistent tangent actions.

        Parameters
        ----------
        state : np.ndarray
            Constitutive state vector with shape (5,).
        differential_state : np.ndarray
            Directional constitutive-state variations with shape (2, 5).
        c1 : float
            First positive Mooney-Rivlin material coefficient.
        c2 : float
            Second positive Mooney-Rivlin material coefficient.
        kappa : float
            Positive bulk-modulus parameter.

        Returns
        -------
        P : np.ndarray
            First Piola-Kirchhoff stress with shape (3, 3).
        tangent_actions : np.ndarray
            Exact consistent tangent actions with shape (2, 3, 3).
            
        Raises
        -------
        ValueError : If the deformation data, constitutive-state data, or material parameters
            have incompatible dimensions, contain nonfinite values, or are mutually inconsistent
            with the admissible contitutive state.
    """
    return P,dP

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_mooney_rivlin_constitutive(F,directions,state,differential_state,c1,c2,kappa):
    F=np.asarray(F,float); directions=np.asarray(directions,float); state=np.asarray(state,float); differential_state=np.asarray(differential_state,float)
    vals=np.array([c1,c2,kappa],float)
    if F.shape!=(3,3): raise ValueError('F must have shape (3,3)')
    if directions.shape!=(2,3,3): raise ValueError('directions must have shape (2,3,3)')
    if state.shape!=(5,) or differential_state.shape!=(2,5): raise ValueError('invalid constitutive state shape')
    if any(not np.all(np.isfinite(x)) for x in (F,directions,state,differential_state,vals)): raise ValueError('all inputs must be finite')
    if np.any(vals<=0): raise ValueError('material parameters must be positive')
    J,I1,I2,s1,s2=state
    if J<=0: raise ValueError('J must be positive')
    # Guard against an inconsistent caller-provided state without recomputing the full state pipeline.
    if not np.isclose(J,np.linalg.det(F),rtol=1e-10,atol=1e-12): raise ValueError('state is inconsistent with F')
    invF=np.linalg.inv(F); invFT=invF.T; C=F.T@F; G=I1*F-F@C
    beta=-(2/3)*c1*s1*I1-(4/3)*c2*s2*I2+.5*kappa*(J*J-1)
    P=2*c1*s1*F+2*c2*s2*G+beta*invFT
    dP=np.empty((2,3,3))
    for q,H in enumerate(directions):
        tr,dJ,dI1,dI2,dlogJ=differential_state[q]
        ds1=-(2/3)*s1*dlogJ; ds2=-(4/3)*s2*dlogJ
        dC=H.T@F+F.T@H; dG=dI1*F+I1*H-H@C-F@dC
        dbeta=-(2/3)*c1*(ds1*I1+s1*dI1)-(4/3)*c2*(ds2*I2+s2*dI2)+kappa*J*dJ
        dinvFT=-invFT@H.T@invFT
        dP[q]=2*c1*(ds1*F+s1*H)+2*c2*(ds2*G+s2*dG)+dbeta*invFT+beta*dinvFT
    return P,dP

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            # Case 1: nominal state produced from the same deformation and directions
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
F=np.array([[1.02,.01,0],[.01,.99,.005],[0,.005,1.01]])
directions=np.stack([np.eye(3)*.2,np.diag([.1,-.1,.05])])
def make_state(F,directions):
    J=float(np.linalg.det(F)); invF=np.linalg.inv(F); C=F.T@F; I1=float(np.trace(C)); I2=.5*(I1*I1-float(np.trace(C@C)))
    state=np.array([J,I1,I2,J**(-2/3),J**(-4/3)])
    ds=[]
    for H in directions:
        tr=float(np.trace(invF@H)); dJ=J*tr; dC=H.T@F+F.T@H; dI1=float(np.trace(dC)); dI2=I1*dI1-float(np.trace(C@dC)); ds.append([tr,dJ,dI1,dI2,tr])
    return state,np.array(ds)
state,differential_state=make_state(F,directions)
c1,c2,kappa=1.3,.7,250.""",
            'call':'_scalarize_test_result(mooney_rivlin_constitutive(F,directions,state,differential_state,c1,c2,kappa))','gold_call':'_scalarize_test_result(_oracle_mooney_rivlin_constitutive(F,directions,state,differential_state,c1,c2,kappa))'
        },
        {
            # Case 2: undeformed state
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
F=np.eye(3); directions=np.stack([np.eye(3)*.1,np.diag([.2,-.1,-.1])]); J=1.; I1=3.; I2=3.
state=np.array([J,I1,I2,1.,1.]); invF=np.eye(3); ds=[]
for H in directions:
    tr=float(np.trace(H)); dJ=tr; dC=H.T+H; dI1=float(np.trace(dC)); C=np.eye(3); dI2=I1*dI1-float(np.trace(C@dC)); ds.append([tr,dJ,dI1,dI2,tr])
differential_state=np.array(ds); c1,c2,kappa=.8,.4,100.""",
            'call':'_scalarize_test_result(mooney_rivlin_constitutive(F,directions,state,differential_state,c1,c2,kappa))','gold_call':'_scalarize_test_result(_oracle_mooney_rivlin_constitutive(F,directions,state,differential_state,c1,c2,kappa))'
        },
        {
            # Case 3: diagonal deformation with consistent supplied state
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
F=np.diag([1.15,.92,1.04]); directions=np.stack([np.diag([.3,.1,-.2]),np.array([[0,.1,0],[.1,0,.05],[0,.05,0]])])
J=float(np.linalg.det(F)); invF=np.linalg.inv(F); C=F.T@F; I1=float(np.trace(C)); I2=.5*(I1*I1-float(np.trace(C@C))); state=np.array([J,I1,I2,J**(-2/3),J**(-4/3)])
ds=[]
for H in directions:
    tr=float(np.trace(invF@H)); dJ=J*tr; dC=H.T@F+F.T@H; dI1=float(np.trace(dC)); dI2=I1*dI1-float(np.trace(C@dC)); ds.append([tr,dJ,dI1,dI2,tr])
differential_state=np.array(ds); c1,c2,kappa=1.1,.3,180.""",
            'call':'_scalarize_test_result(mooney_rivlin_constitutive(F,directions,state,differential_state,c1,c2,kappa))','gold_call':'_scalarize_test_result(_oracle_mooney_rivlin_constitutive(F,directions,state,differential_state,c1,c2,kappa))'
        },
        {
            # Case 4: inconsistent supplied J
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
F=np.eye(3); directions=np.zeros((2,3,3)); state=np.array([1.1,3.,3.,1.,1.]); differential_state=np.zeros((2,5)); c1,c2,kappa=1.,1.,10.
def run_model():
    try: mooney_rivlin_constitutive(F,directions,state,differential_state,c1,c2,kappa); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try: _oracle_mooney_rivlin_constitutive(F,directions,state,differential_state,c1,c2,kappa); return 0
    except ValueError: return 1
    except Exception: return 2""",
            'call':'run_model()','gold_call':'run_oracle()'
        },
        {
            # Case 5: invalid constitutive-state shape
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
F=np.eye(3); directions=np.zeros((2,3,3)); state=np.ones(4); differential_state=np.zeros((2,5)); c1,c2,kappa=1.,1.,10.
def run_model():
    try: mooney_rivlin_constitutive(F,directions,state,differential_state,c1,c2,kappa); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try: _oracle_mooney_rivlin_constitutive(F,directions,state,differential_state,c1,c2,kappa); return 0
    except ValueError: return 1
    except Exception: return 2""",
            'call':'run_model()','gold_call':'run_oracle()'
        },
        {
            # Case 6: invalid material parameter
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
F=np.eye(3); directions=np.zeros((2,3,3)); state=np.array([1.,3.,3.,1.,1.]); differential_state=np.zeros((2,5)); c1,c2,kappa=0.,1.,10.
def run_model():
    try: mooney_rivlin_constitutive(F,directions,state,differential_state,c1,c2,kappa); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try: _oracle_mooney_rivlin_constitutive(F,directions,state,differential_state,c1,c2,kappa); return 0
    except ValueError: return 1
    except Exception: return 2""",
            'call':'run_model()','gold_call':'run_oracle()'
        }
    ]
