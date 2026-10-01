"""
Applies one undamped Newton correction and computes numerical diagnostics.

Newton's method uses the reduced residual and its Jacobian to determine a local correction. Nonsingularity and conditioning of the Jacobian govern whether this correction is numerically meaningful, while the residual and correction norms give compact diagnostics.

Returns
-------
tuple[np.ndarray, np.ndarray], updated state (2,) and diagnostics [||r||_2, cond_2(K), ||delta||_2]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def newton_update_diagnostics(y: np.ndarray, r: np.ndarray, K: np.ndarray):
    """Apply one undamped Newton correction and compute numerical diagnostics.

        Parameters
        ----------
        y : np.ndarray
            Current reduced state with shape (2,).
        r : np.ndarray
            Coupled reduced residual with shape (2,).
        K : np.ndarray
            Coupled reduced Jacobian with shape (2, 2).

        Returns
        -------
        y_new : np.ndarray
            Updated reduced state with shape (2,).
        diagnostics : np.ndarray
            Array containing the residual 2-norm, Jacobian 2-norm condition number,
            and Newton-correction 2-norm.

        Raises
        -------
        ValueError : If the reduced state, residual, or Jacobian
            have incompatible dimensions, contain nonfinite values, or 
            do not define a valid finite Newton update.
    """
    return y_new,diagnostics

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_newton_update_diagnostics(y,r,K):
    y=np.asarray(y,float); r=np.asarray(r,float); K=np.asarray(K,float)
    if y.shape!=(2,) or r.shape!=(2,) or K.shape!=(2,2): raise ValueError('y and r must have shape (2,), K must have shape (2,2)')
    if any(not np.all(np.isfinite(x)) for x in (y,r,K)): raise ValueError('all inputs must be finite')
    try: delta=np.linalg.solve(K,r)
    except np.linalg.LinAlgError as exc: raise ValueError('K must be nonsingular') from exc
    cond=float(np.linalg.cond(K,2))
    if not np.isfinite(cond): raise ValueError('condition number must be finite')
    return y-delta,np.array([np.linalg.norm(r),cond,np.linalg.norm(delta)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            # Case 1: nominal Newton system
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
y=np.array([.018,-.012])
r=np.array([2.58,-.51])
K=np.array([[140.8,-24.2],[-24.2,36.1]])""",
            'call':'_scalarize_test_result(newton_update_diagnostics(y,r,K))','gold_call':'_scalarize_test_result(_oracle_newton_update_diagnostics(y,r,K))'
        },
        {
            # Case 2: zero residual
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
y=np.array([.01,-.02])
r=np.zeros(2)
K=np.array([[2.,.3],[.3,1.5]])""",
            'call':'_scalarize_test_result(newton_update_diagnostics(y,r,K))','gold_call':'_scalarize_test_result(_oracle_newton_update_diagnostics(y,r,K))'
        },
        {
            # Case 3: nonsymmetric invertible Jacobian
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
y=np.array([-.1,.2])
r=np.array([.3,-.4])
K=np.array([[3.,1.],[-.2,2.]])""",
            'call':'_scalarize_test_result(newton_update_diagnostics(y,r,K))','gold_call':'_scalarize_test_result(_oracle_newton_update_diagnostics(y,r,K))'
        },
        {
            # Case 4: singular Jacobian
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
y=np.zeros(2); r=np.ones(2); K=np.array([[1.,2.],[2.,4.]])
def run_model():
    try: newton_update_diagnostics(y,r,K); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try: _oracle_newton_update_diagnostics(y,r,K); return 0
    except ValueError: return 1
    except Exception: return 2""",
            'call':'run_model()','gold_call':'run_oracle()'
        },
        {
            # Case 5: invalid residual shape
            'setup':"""import numpy as np
def _scalarize_test_result(result):
    if isinstance(result, tuple):
        flat=np.concatenate([np.asarray(item,dtype=float).ravel() for item in result])
    else:
        flat=np.asarray(result,dtype=float).ravel()
    weights=np.arange(1,flat.size+1,dtype=float)
    return float(np.dot(flat,weights))
y=np.zeros(2); r=np.zeros(3); K=np.eye(2)
def run_model():
    try: newton_update_diagnostics(y,r,K); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try: _oracle_newton_update_diagnostics(y,r,K); return 0
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
y=np.zeros(2); r=np.ones(2); K=np.eye(2); K[0,1]=np.inf
def run_model():
    try: newton_update_diagnostics(y,r,K); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try: _oracle_newton_update_diagnostics(y,r,K); return 0
    except ValueError: return 1
    except Exception: return 2""",
            'call':'run_model()','gold_call':'run_oracle()'
        }
    ]
