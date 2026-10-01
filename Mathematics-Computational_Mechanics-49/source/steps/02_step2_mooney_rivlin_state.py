"""
Computes the Mooney-Rivlin constitutive state quantities and their directional variations.

Compressible Mooney-Rivlin elasticity depends on the deformation ratio $J$, the invariants $C=F^TF$, and the isochoric powers of $J$. Consistent linearization requires directional changes in these coupled quantities for each reduced direction, making the constitutive state a nonlinear tensor-calculus subproblem rather than an isolated scalar calculation.

Returns
-------
tuple[np.ndarray, np.ndarray], state vector [J,I1,I2,J^(-2/3),J^(-4/3)] and a (2,5) matrix [tr(F^-1 H),dJ,dI1,dI2,d(log J)]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def mooney_rivlin_state(F: np.ndarray, directions: np.ndarray):
    """Compute Mooney-Rivlin constitutive state quantities and their directional variations.

        Parameters
        ----------
        F : np.ndarray
            Deformation gradient with shape (3, 3).
        directions : np.ndarray
            Two deformation-gradient perturbation directions with shape (2, 3, 3).

        Returns
        -------
        state : np.ndarray
            Constitutive state vector with shape (5,).
        differential_state : np.ndarray
            Directional variations of the constitutive state with shape (2, 5).

        Raises
        -------
        ValueError : If the deformation gradient or perturbation directions
            have incompatible dimensions, contain nonfinite values, or
            place the deformation outside the admissible hyperelastic domain.
    """
    return state, differential_state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_mooney_rivlin_state(F,directions):
    F=np.asarray(F,float); directions=np.asarray(directions,float)
    if F.shape!=(3,3): raise ValueError('F must have shape (3,3)')
    if directions.shape!=(2,3,3): raise ValueError('directions must have shape (2,3,3)')
    if not np.all(np.isfinite(F)) or not np.all(np.isfinite(directions)): raise ValueError('all inputs must be finite')
    J=float(np.linalg.det(F))
    if J<=0: raise ValueError('det(F) must be positive')
    invF=np.linalg.inv(F); C=F.T@F
    I1=float(np.trace(C)); I2=.5*(I1*I1-float(np.trace(C@C)))
    s1=J**(-2/3); s2=J**(-4/3)
    state=np.array([J,I1,I2,s1,s2],float)
    diff=np.empty((2,5),float)
    for q,H in enumerate(directions):
        tr=float(np.trace(invF@H)); dJ=J*tr
        dC=H.T@F+F.T@H; dI1=float(np.trace(dC)); dI2=I1*dI1-float(np.trace(C@dC))
        diff[q]=[tr,dJ,dI1,dI2,tr]
    return state,diff

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            # Case 1: general finite deformation with two independent directions
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
F=np.array([[1.02,.01,0],[.01,.99,.005],[0,.005,1.01]])
directions=np.stack([np.array([[.6,.15,0],[.15,-.2,.05],[0,.05,.1]]),np.array([[-.1,.2,.04],[.2,.5,0],[.04,0,-.3]])])""",
            'call':'_scalarize_test_result(mooney_rivlin_state(F,directions))','gold_call':'_scalarize_test_result(_oracle_mooney_rivlin_state(F,directions))'
        },
        {
            # Case 2: undeformed state with volumetric and isochoric directions
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
F=np.eye(3)
directions=np.stack([np.eye(3)*.1,np.diag([.2,-.1,-.1])])""",
            'call':'_scalarize_test_result(mooney_rivlin_state(F,directions))','gold_call':'_scalarize_test_result(_oracle_mooney_rivlin_state(F,directions))'
        },
        {
            # Case 3: diagonal finite deformation with shear direction
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
F=np.diag([1.15,.92,1.04])
directions=np.stack([np.diag([.3,.1,-.2]),np.array([[0,.1,0],[.1,0,.05],[0,.05,0]])])""",
            'call':'_scalarize_test_result(mooney_rivlin_state(F,directions))','gold_call':'_scalarize_test_result(_oracle_mooney_rivlin_state(F,directions))'
        },
        {
            # Case 4: invalid negative determinant
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
F=np.diag([1.,1.,-1.])
directions=np.zeros((2,3,3))
def run_model():
    try: mooney_rivlin_state(F,directions); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try: _oracle_mooney_rivlin_state(F,directions); return 0
    except ValueError: return 1
    except Exception: return 2""",
            'call':'run_model()','gold_call':'run_oracle()'
        },
        {
            # Case 5: invalid direction shape
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
F=np.eye(3)
directions=np.zeros((3,3,3))
def run_model():
    try: mooney_rivlin_state(F,directions); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try: _oracle_mooney_rivlin_state(F,directions); return 0
    except ValueError: return 1
    except Exception: return 2""",
            'call':'run_model()','gold_call':'run_oracle()'
        },
        {
            # Case 6: invalid nonfinite deformation
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
F=np.eye(3); F[0,0]=np.inf
directions=np.zeros((2,3,3))
def run_model():
    try: mooney_rivlin_state(F,directions); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try: _oracle_mooney_rivlin_state(F,directions); return 0
    except ValueError: return 1
    except Exception: return 2""",
            'call':'run_model()','gold_call':'run_oracle()'
        }
    ]
