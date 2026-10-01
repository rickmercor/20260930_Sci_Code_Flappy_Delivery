"""
Creates the deformation gradients and reduced strain-mode matrices required by the constitutive stages.

Component-wise model reduction can represent each substructure in local reduced coordinates while maintaining a shared global reduced state. In the strain-space formulation, local coordinates and boundary-consistent lifting fields reconstruct deformation gradients at selected integration points, and the associated local reduced directions define how constitutive quantities are projected.

Returns
-------
tuple[np.ndarray, np.ndarray], deformation gradients (m,3,3) and reduced mode matrices (m,9,2)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def component_point_kinematics(C: np.ndarray, y: np.ndarray, d: np.ndarray,
                                     points: np.ndarray, B1: np.ndarray, B2: np.ndarray,
                                     L1: np.ndarray, L2: np.ndarray):
    """Construct selected-point deformation gradients and reduced mode matrices for one component.

        Parameters
        ----------
        C : np.ndarray
            Component coordinate map with shape (2, 2).
        y : np.ndarray
            Global reduced state with shape (2,).
        d : np.ndarray
            Boundary parameters with shape (2,).
        points : np.ndarray
            Selected-point data with shape (m, 5), where each row is
            [a, b, ell1, ell2, weight].
        B1 : np.ndarray
            First reduced deformation-mode matrix with shape (3, 3).
        B2 : np.ndarray
            Second reduced deformation-mode matrix with shape (3, 3).
        L1 : np.ndarray
            First boundary-lifting matrix with shape (3, 3).
        L2 : np.ndarray
            Second boundary-lifting matrix with shape (3, 3).

        Returns
        -------
        F_points : np.ndarray
            Selected-point deformation gradients with shape (m, 3, 3).
        phis : np.ndarray
            Selected-point reduced mode matrices with shape (m, 9, 2).
        
        Raises
        -------
        ValueError : If the component map, reduced-state data, point data, or mode
            have incompatible dimensions, contain nonfinite values, or violate the
            positivity conditions.
    """
    return F_points, phis

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_component_point_kinematics(C,y,d,points,B1,B2,L1,L2):
    C=np.asarray(C,float); y=np.asarray(y,float); d=np.asarray(d,float); points=np.asarray(points,float)
    mats=[np.asarray(M,float) for M in (B1,B2,L1,L2)]
    if C.shape!=(2,2): raise ValueError('C must have shape (2,2)')
    if y.shape!=(2,) or d.shape!=(2,): raise ValueError('y and d must have shape (2,)')
    if points.ndim!=2 or points.shape[1]!=5 or points.shape[0]<1: raise ValueError('points must have shape (m,5), m>=1')
    if any(M.shape!=(3,3) for M in mats): raise ValueError('B1, B2, L1, L2 must have shape (3,3)')
    if any(not np.all(np.isfinite(x)) for x in [C,y,d,points]+mats): raise ValueError('all inputs must be finite')
    if np.any(points[:,4]<=0): raise ValueError('weights must be positive')
    z=C@y; B1,B2,L1,L2=mats; m=points.shape[0]
    F_points=np.empty((m,3,3)); phis=np.empty((m,9,2))
    for j,(a,b,e1,e2,w) in enumerate(points):
        F_points[j]=np.eye(3)+a*z[0]*B1+b*z[1]*B2+e1*d[0]*L1+e2*d[1]*L2
        phis[j]=np.column_stack((a*B1.ravel(order='C'),b*B2.ravel(order='C')))
    return F_points,phis

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            # Case 1: nominal two-point component
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
C=np.array([[1.,.35],[-.25,.9]])
y=np.array([.018,-.012])
d=np.array([.04,-.03])
points=np.array([[1.,.7,.8,-.2,.55],[-.6,1.1,.3,.9,.85]])
B1=np.array([[.6,.15,0],[.15,-.2,.05],[0,.05,.1]])
B2=np.array([[-.1,.2,.04],[.2,.5,0],[.04,0,-.3]])
L1=np.array([[.4,.1,0],[.1,-.1,.02],[0,.02,-.2]])
L2=np.array([[-.2,0,.05],[0,.3,.08],[.05,.08,.1]])""",
            'call':'_scalarize_test_result(component_point_kinematics(C,y,d,points,B1,B2,L1,L2))',
            'gold_call':'_scalarize_test_result(_oracle_component_point_kinematics(C,y,d,points,B1,B2,L1,L2))'
        },
        {
            # Case 2: single point with zero reduced state
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
C=np.array([[.9,.1],[-.2,1.1]])
y=np.zeros(2)
d=np.array([.03,.02])
points=np.array([[1.2,-.4,.5,.7,.6]])
B1=np.eye(3)*.2
B2=np.diag([.1,-.1,.2])
L1=np.eye(3)*.3
L2=np.diag([-.2,.1,.05])""",
            'call':'_scalarize_test_result(component_point_kinematics(C,y,d,points,B1,B2,L1,L2))',
            'gold_call':'_scalarize_test_result(_oracle_component_point_kinematics(C,y,d,points,B1,B2,L1,L2))'
        },
        {
            # Case 3: three selected points and nontrivial component map
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
C=np.array([[.75,-.25],[.4,1.2]])
y=np.array([-.01,.02])
d=np.array([-.015,.025])
points=np.array([[.8,.6,.4,-.3,.5],[-1.1,.2,.7,.9,.8],[.5,-.9,-.2,.6,.7]])
B1=np.array([[.4,.1,0],[.1,-.1,.03],[0,.03,.2]])
B2=np.array([[-.2,.05,.02],[.05,.3,0],[.02,0,-.1]])
L1=np.eye(3)*.15
L2=np.array([[.1,0,.02],[0,-.05,.01],[.02,.01,.08]])""",
            'call':'_scalarize_test_result(component_point_kinematics(C,y,d,points,B1,B2,L1,L2))',
            'gold_call':'_scalarize_test_result(_oracle_component_point_kinematics(C,y,d,points,B1,B2,L1,L2))'
        },
        {
            # Case 4: invalid component map shape
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
C=np.eye(3)
y=np.array([.01,.02])
d=np.array([.03,.04])
points=np.array([[1.,1.,1.,1.,1.]])
B1=B2=L1=L2=np.eye(3)
def run_model():
    try:
        component_point_kinematics(C,y,d,points,B1,B2,L1,L2); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try:
        _oracle_component_point_kinematics(C,y,d,points,B1,B2,L1,L2); return 0
    except ValueError: return 1
    except Exception: return 2""",
            'call':'run_model()','gold_call':'run_oracle()'
        },
        {
            # Case 5: invalid nonpositive hyperreduction weight
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
C=np.eye(2)
y=np.array([.01,.02])
d=np.array([.03,.04])
points=np.array([[1.,1.,1.,1.,0.]])
B1=B2=L1=L2=np.eye(3)
def run_model():
    try:
        component_point_kinematics(C,y,d,points,B1,B2,L1,L2); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try:
        _oracle_component_point_kinematics(C,y,d,points,B1,B2,L1,L2); return 0
    except ValueError: return 1
    except Exception: return 2""",
            'call':'run_model()','gold_call':'run_oracle()'
        },
        {
            # Case 6: invalid nonfinite kinematic matrix
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
C=np.eye(2)
y=np.array([.01,.02])
d=np.array([.03,.04])
points=np.array([[1.,1.,1.,1.,1.]])
B1=np.eye(3); B1[0,0]=np.nan
B2=L1=L2=np.eye(3)
def run_model():
    try:
        component_point_kinematics(C,y,d,points,B1,B2,L1,L2); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try:
        _oracle_component_point_kinematics(C,y,d,points,B1,B2,L1,L2); return 0
    except ValueError: return 1
    except Exception: return 2""",
            'call':'run_model()','gold_call':'run_oracle()'
        }
    ]
