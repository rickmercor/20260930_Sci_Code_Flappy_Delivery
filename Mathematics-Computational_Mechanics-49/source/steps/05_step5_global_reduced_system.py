"""
Assembles the coupled global reduced residual and Jacobian from component contributions.

A component-local reduced system contributes to the common coordinates through its coordinate map. Residuals transform through the transpose map and Jacobians from the congruence transformation. The component contributions and external load then form the coupled reduced equilibrium system.

Returns
-------
tuple[np.ndarray, np.ndarray], global residual (2,) and global Jacobian (2,2)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def global_reduced_system(component_maps: np.ndarray, local_residuals: np.ndarray, local_jacobians: np.ndarray, f_ext: np.ndarray):
    """Assemble the coupled global reduced residual and Jacobian from component contributions.

        Parameters
        ----------
        component_maps : np.ndarray
            Component coordinate maps with shape (n, 2, 2).
        local_residuals : np.ndarray
            Component-local reduced residuals with shape (n, 2).
        local_jacobians : np.ndarray
            Component-local reduced Jacobians with shape (n, 2, 2).
        f_ext : np.ndarray
            External reduced load with shape (2,).

        Returns
        -------
        r : np.ndarray
            Coupled global reduced residual with shape (2,).
        K : np.ndarray
            Coupled global reduced Jacobian with shape (2, 2).

        Raises
        -------
        ValueError : If the component maps, local residuals, local Jacobians, or external load
            have incompatible dimensions, contain nonfinite values, or inconsistent 
            component counts.
    """
    return r,K

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_global_reduced_system(component_maps,local_residuals,local_jacobians,f_ext):
    component_maps=np.asarray(component_maps,float); local_residuals=np.asarray(local_residuals,float); local_jacobians=np.asarray(local_jacobians,float); f_ext=np.asarray(f_ext,float)
    if component_maps.ndim!=3 or component_maps.shape[1:]!=(2,2) or component_maps.shape[0]<1: raise ValueError('component_maps must have shape (n,2,2), n>=1')
    n=component_maps.shape[0]
    if local_residuals.shape!=(n,2) or local_jacobians.shape!=(n,2,2) or f_ext.shape!=(2,): raise ValueError('incompatible global assembly shapes')
    if any(not np.all(np.isfinite(x)) for x in (component_maps,local_residuals,local_jacobians,f_ext)): raise ValueError('all inputs must be finite')
    r=-f_ext.copy(); K=np.zeros((2,2))
    for i,C in enumerate(component_maps):
        r+=C.T@local_residuals[i]; K+=C.T@local_jacobians[i]@C
    return r,K

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            # Case 1: two coupled components
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
component_maps=np.array([[[1.,.35],[-.25,.9]],[[.8,-.4],[.3,1.1]]])
local_residuals=np.array([[1.2,-.2],[1.5,-.1]])
local_jacobians=np.array([[[4.,.2],[.2,2.]],[[3.,-.1],[-.1,1.]]])
f_ext=np.array([.08,-.03])""",
            'call':'_scalarize_test_result(global_reduced_system(component_maps,local_residuals,local_jacobians,f_ext))','gold_call':'_scalarize_test_result(_oracle_global_reduced_system(component_maps,local_residuals,local_jacobians,f_ext))'
        },
        {
            # Case 2: single component
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
component_maps=np.array([np.eye(2)])
local_residuals=np.array([[.5,-.2]])
local_jacobians=np.array([[[2.,.1],[.1,3.]]])
f_ext=np.array([.1,.05])""",
            'call':'_scalarize_test_result(global_reduced_system(component_maps,local_residuals,local_jacobians,f_ext))','gold_call':'_scalarize_test_result(_oracle_global_reduced_system(component_maps,local_residuals,local_jacobians,f_ext))'
        },
        {
            # Case 3: three components
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
component_maps=np.array([np.eye(2),[[.8,.2],[-.1,1.1]],[[1.2,-.3],[.4,.7]]])
local_residuals=np.array([[.2,.1],[-.3,.4],[.5,-.2]])
local_jacobians=np.array([np.eye(2),[[2.,.3],[.3,1.5]],[[1.2,-.2],[-.2,2.4]]])
f_ext=np.array([-.1,.2])""",
            'call':'_scalarize_test_result(global_reduced_system(component_maps,local_residuals,local_jacobians,f_ext))','gold_call':'_scalarize_test_result(_oracle_global_reduced_system(component_maps,local_residuals,local_jacobians,f_ext))'
        },
        {
            # Case 4: invalid residual count
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
component_maps=np.array([np.eye(2),np.eye(2)]); local_residuals=np.zeros((1,2)); local_jacobians=np.array([np.eye(2),np.eye(2)]); f_ext=np.zeros(2)
def run_model():
    try: global_reduced_system(component_maps,local_residuals,local_jacobians,f_ext); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try: _oracle_global_reduced_system(component_maps,local_residuals,local_jacobians,f_ext); return 0
    except ValueError: return 1
    except Exception: return 2""",
            'call':'run_model()','gold_call':'run_oracle()'
        },
        {
            # Case 5: invalid map shape
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
component_maps=np.zeros((2,3,3)); local_residuals=np.zeros((2,2)); local_jacobians=np.zeros((2,2,2)); f_ext=np.zeros(2)
def run_model():
    try: global_reduced_system(component_maps,local_residuals,local_jacobians,f_ext); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try: _oracle_global_reduced_system(component_maps,local_residuals,local_jacobians,f_ext); return 0
    except ValueError: return 1
    except Exception: return 2""",
            'call':'run_model()','gold_call':'run_oracle()'
        },
        {
            # Case 6: nonfinite Jacobian
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
component_maps=np.array([np.eye(2)]); local_residuals=np.zeros((1,2)); local_jacobians=np.array([np.eye(2)]); local_jacobians[0,0,0]=np.nan; f_ext=np.zeros(2)
def run_model():
    try: global_reduced_system(component_maps,local_residuals,local_jacobians,f_ext); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try: _oracle_global_reduced_system(component_maps,local_residuals,local_jacobians,f_ext); return 0
    except ValueError: return 1
    except Exception: return 2""",
            'call':'run_model()','gold_call':'run_oracle()'
        }
    ]
