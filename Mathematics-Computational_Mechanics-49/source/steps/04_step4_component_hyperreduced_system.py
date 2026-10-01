"""
Assembles the component-local hyperreduced residual and Jacobian.

Hyperreduction approximates reduced internal-force and stiffness contributions by evaluating only selected integration points with effective weights. Stress and tangent contributions must be projected through the same reduced strain modes so that the local Jacobian remains consistent with the local residual.

Returns
-------
tuple[np.ndarray, np.ndarray], local residual (2,) and local Jacobian (2,2)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def component_hyperreduced_system(phis: np.ndarray, weights: np.ndarray, stresses: np.ndarray, tangent_actions: np.ndarray):
    """Assemble the component-local hyperreduced residual and Jacobian.

        Parameters
        ----------
        phis : np.ndarray
            Selected-point reduced mode matrices with shape (m, 9, 2).
        weights : np.ndarray
            Positive hyperreduction weights with shape (m,).
        stresses : np.ndarray
            Selected-point first Piola-Kirchhoff stresses with shape (m, 3, 3).
        tangent_actions : np.ndarray
            Selected-point tangent actions with shape (m, 2, 3, 3).

        Returns
        -------
        r_local : np.ndarray
            Component-local reduced residual with shape (2,).
        K_local : np.ndarray
            Component-local reduced Jacobian with shape (2, 2).

        Raises
        -------
        ValueError : If the selected-point mode, weight, stress, or tangent data
            have incompatible dimensions, contain nonfinite values, or invalid
            hyperreduction weights.
    """
    return r_local,K_local

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_component_hyperreduced_system(phis,weights,stresses,tangent_actions):
    phis=np.asarray(phis,float); weights=np.asarray(weights,float); stresses=np.asarray(stresses,float); tangent_actions=np.asarray(tangent_actions,float)
    if phis.ndim!=3 or phis.shape[1:]!=(9,2) or phis.shape[0]<1: raise ValueError('phis must have shape (m,9,2), m>=1')
    m=phis.shape[0]
    if weights.shape!=(m,) or stresses.shape!=(m,3,3) or tangent_actions.shape!=(m,2,3,3): raise ValueError('point arrays have incompatible shapes')
    if any(not np.all(np.isfinite(x)) for x in (phis,weights,stresses,tangent_actions)): raise ValueError('all inputs must be finite')
    if np.any(weights<=0): raise ValueError('weights must be positive')
    r=np.zeros(2); K=np.zeros((2,2))
    for c in range(m):
        r+=weights[c]*(phis[c].T@stresses[c].ravel(order='C'))
        for q in range(2): K[:,q]+=weights[c]*(phis[c].T@tangent_actions[c,q].ravel(order='C'))
    return r,K

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            # Case 1: two-point projected system
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
phis=np.zeros((2,9,2))
phis[0,:,0]=np.arange(1,10)
phis[0,:,1]=np.linspace(.5,1.3,9)
phis[1]=phis[0]*.7
weights=np.array([.55,.85])
stresses=np.stack([np.eye(3),np.eye(3)*2])
tangent_actions=np.arange(36,dtype=float).reshape(2,2,3,3)/20.""",
            'call':'_scalarize_test_result(component_hyperreduced_system(phis,weights,stresses,tangent_actions))','gold_call':'_scalarize_test_result(_oracle_component_hyperreduced_system(phis,weights,stresses,tangent_actions))'
        },
        {
            # Case 2: single selected point
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
phis=np.zeros((1,9,2)); phis[0,0,0]=1.; phis[0,4,1]=2.
weights=np.array([1.])
stresses=np.arange(9,dtype=float).reshape(1,3,3)
tangent_actions=np.arange(18,dtype=float).reshape(1,2,3,3)""",
            'call':'_scalarize_test_result(component_hyperreduced_system(phis,weights,stresses,tangent_actions))','gold_call':'_scalarize_test_result(_oracle_component_hyperreduced_system(phis,weights,stresses,tangent_actions))'
        },
        {
            # Case 3: three random deterministic points
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
rng=np.random.default_rng(4)
phis=rng.normal(size=(3,9,2)); weights=np.array([.2,.6,1.1]); stresses=rng.normal(size=(3,3,3)); tangent_actions=rng.normal(size=(3,2,3,3))""",
            'call':'_scalarize_test_result(component_hyperreduced_system(phis,weights,stresses,tangent_actions))','gold_call':'_scalarize_test_result(_oracle_component_hyperreduced_system(phis,weights,stresses,tangent_actions))'
        },
        {
            # Case 4: invalid zero weight
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
phis=np.ones((1,9,2)); weights=np.array([0.]); stresses=np.ones((1,3,3)); tangent_actions=np.ones((1,2,3,3))
def run_model():
    try: component_hyperreduced_system(phis,weights,stresses,tangent_actions); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try: _oracle_component_hyperreduced_system(phis,weights,stresses,tangent_actions); return 0
    except ValueError: return 1
    except Exception: return 2""",
            'call':'run_model()','gold_call':'run_oracle()'
        },
        {
            # Case 5: invalid tangent count
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
phis=np.ones((2,9,2)); weights=np.ones(2); stresses=np.ones((2,3,3)); tangent_actions=np.ones((1,2,3,3))
def run_model():
    try: component_hyperreduced_system(phis,weights,stresses,tangent_actions); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try: _oracle_component_hyperreduced_system(phis,weights,stresses,tangent_actions); return 0
    except ValueError: return 1
    except Exception: return 2""",
            'call':'run_model()','gold_call':'run_oracle()'
        },
        {
            # Case 6: invalid nonfinite stress
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
phis=np.ones((1,9,2)); weights=np.ones(1); stresses=np.ones((1,3,3)); stresses[0,0,0]=np.inf; tangent_actions=np.ones((1,2,3,3))
def run_model():
    try: component_hyperreduced_system(phis,weights,stresses,tangent_actions); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try: _oracle_component_hyperreduced_system(phis,weights,stresses,tangent_actions); return 0
    except ValueError: return 1
    except Exception: return 2""",
            'call':'run_model()','gold_call':'run_oracle()'
        }
    ]
