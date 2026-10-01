"""
Convert the projected controlled dynamics into a valid reduced quantum instrument by stimulus-wise polar completion.

Projection onto a retained memory support generally destroys trace preservation. For each stimulus separately, use the nearest-isometry polar repair on the support of the projected isometry Gram operator, with inverse square roots taken only on its supported spectrum. Eigenvalues greater than the supplied cutoff are retained. Complete any missing retained support by one Kraus operator equal to its orthogonal projector, with action $y=0$ and a new environment label; all other operators on that new label vanish. Keep the original action and environment labels, so later purified-history comparisons have a fixed dilation rather than an independently optimized gauge.

Returns
-------
np.ndarray, complex completed instrument of shape (X, Y, E+1, d, d).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real

import numpy as np

def repair_controlled_instrument(kraus: np.ndarray, projector: np.ndarray, support_tol: float = 1e-12) -> np.ndarray:
    """Repair the projected instrument in the original ambient memory basis.

    Parameters
    ----------
    kraus : np.ndarray
        Finite complex array (X,Y,E,d,d) defining a trace-preserving instrument
        for each stimulus, within absolute tolerance 1e-10.
    projector : np.ndarray
        Nonzero Hermitian orthogonal projector (d,d), checked at absolute
        tolerance 1e-10. Retained and discarded directions may be non-coordinate.
    support_tol : float
        Finite real number in (0,1), excluding booleans. Gram eigenvalues
        strictly greater than this absolute cutoff define the polar support.

    Returns
    -------
    repaired : np.ndarray
        Complex array (X,Y,E+1,d,d). Existing environment labels keep their
        order; the extra label E, action 0, contains the projector onto the
        missing retained support. Each stimulus is complete on projector.
        No input is mutated. Full retention returns the original operators
        with one zero-padded environment label.

    Raises
    ------
    ValueError
        If shapes or finiteness fail, the instrument is not trace preserving,
        projector is not a nonzero orthogonal projector, or support_tol is invalid.
    """
    return np.zeros((kraus.shape[0], kraus.shape[1], kraus.shape[2]+1, kraus.shape[-1], kraus.shape[-1]), dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_repair_controlled_instrument(kraus: "np.ndarray", projector: "np.ndarray", support_tol: float = 1e-12) -> "np.ndarray":
    import numpy as np
    from numbers import Real

    try:
        K = np.asarray(kraus, dtype=complex)
        P = np.asarray(projector, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("numeric arrays are required") from exc
    if K.ndim != 5 or min(K.shape) < 1 or K.shape[-2] != K.shape[-1] or not np.isfinite(K).all():
        raise ValueError("invalid Kraus array")
    X,Y,E,d,_ = K.shape
    if P.shape != (d,d) or not np.isfinite(P).all() or not np.allclose(P, P.conj().T, atol=1e-10, rtol=0):
        raise ValueError("invalid projector")
    if not np.allclose(P @ P, P, atol=1e-10, rtol=0) or np.trace(P).real < .5:
        raise ValueError("projector must be nonzero and idempotent")
    if isinstance(support_tol, (bool, np.bool_)) or not isinstance(support_tol, Real) or not np.isfinite(support_tol) or not 0 < support_tol < 1:
        raise ValueError("invalid support cutoff")
    for x in range(X):
        gram = sum((A.conj().T @ A for A in K[x].reshape(-1,d,d)), np.zeros((d,d), complex))
        if not np.allclose(gram, np.eye(d), atol=1e-10, rtol=0):
            raise ValueError("instrument must be trace preserving")
    result = np.zeros((X,Y,E+1,d,d), complex)
    if np.array_equal(P, np.eye(d)):
        result[:,:,:E] = K
        return result
    # Work inside the retained space so roundoff cannot introduce
    # discarded directions into the polar support.
    pvalues, pvectors = np.linalg.eigh((P + P.conj().T)/2)
    retained = pvectors[:, pvalues > 0.5]
    r = retained.shape[1]
    for x in range(X):
        projected = np.array([
            retained.conj().T @ A @ retained
            for A in K[x].reshape(-1,d,d)
        ])
        left, singular, right = np.linalg.svd(
            projected.reshape(-1,r), full_matrices=False
        )
        mask = singular**2 > support_tol
        polar = (left[:,mask] @ right[mask,:]).reshape(Y,E,r,r)
        result[x,:,:E] = retained @ polar @ retained.conj().T
        missing = np.eye(r) - right[mask,:].conj().T @ right[mask,:]
        result[x,0,E] = retained @ missing @ retained.conj().T
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic case specifications."""
    return [
        {
            "setup": """import numpy as np
K = np.eye(4).reshape(1,1,1,4,4)
P = np.array([[2,1-1j,0,1+1j], [1+1j,2,1-1j,0],
              [0,1+1j,2,1-1j], [1-1j,0,1+1j,2]]) / 4
""",
            "call": 'repair_controlled_instrument(K, P, 1e-16)',
            "gold_call": '_oracle_repair_controlled_instrument(K, P, 1e-16)',
        },
        {
            "setup": """import numpy as np
n = 4
s0 = np.ones(n) / np.sqrt(n)
K = np.zeros((2,2,n,n,n), dtype=complex)
K[0,0,0] = np.diag(np.ones(n-1), -1)
K[0,1,0] = np.outer(s0, np.eye(n)[-1])
for eta in range(n):
    K[1,0,eta] = np.outer(s0, np.eye(n)[eta])
R = np.array([[[.86,.08],[.01,.05]],[[.10,.51],[.19,.20]]])
U=np.exp(2j*np.pi*np.outer(np.arange(n),np.arange(n))/n)/np.sqrt(n)
P=U[:,:2]@U[:,:2].conj().T
""",
            "call": 'repair_controlled_instrument(K, P)',
            "gold_call": '_oracle_repair_controlled_instrument(K, P)',
        },
        {
            "setup": """import numpy as np
n = 4
s0 = np.ones(n) / np.sqrt(n)
K = np.zeros((2,2,n,n,n), dtype=complex)
K[0,0,0] = np.diag(np.ones(n-1), -1)
K[0,1,0] = np.outer(s0, np.eye(n)[-1])
for eta in range(n):
    K[1,0,eta] = np.outer(s0, np.eye(n)[eta])
R = np.array([[[.86,.08],[.01,.05]],[[.10,.51],[.19,.20]]])
P=np.eye(n)
""",
            "call": 'repair_controlled_instrument(K, P)',
            "gold_call": '_oracle_repair_controlled_instrument(K, P)',
        },
        {
            "setup": """import numpy as np
K=np.roll(np.eye(3),1,axis=0).reshape(1,1,1,3,3)
P=np.diag([1,1,0])
""",
            "call": 'repair_controlled_instrument(K, P)',
            "gold_call": '_oracle_repair_controlled_instrument(K, P)',
        },
        {
            "setup": """import numpy as np
K=np.array([[0,1],[1,0]]).reshape(1,1,1,2,2)
P=np.diag([1,0])
""",
            "call": 'repair_controlled_instrument(K, P)',
            "gold_call": '_oracle_repair_controlled_instrument(K, P)',
        },
        {
            "setup": """import numpy as np
K=np.array([[.5,-np.sqrt(.75)],[np.sqrt(.75),.5]]).reshape(1,1,1,2,2)
P=np.diag([1,0])
""",
            "call": 'repair_controlled_instrument(K, P, .25)',
            "gold_call": '_oracle_repair_controlled_instrument(K, P, .25)',
        },
        {
            "setup": """import numpy as np
n = 4
s0 = np.ones(n) / np.sqrt(n)
K = np.zeros((2,2,n,n,n), dtype=complex)
K[0,0,0] = np.diag(np.ones(n-1), -1)
K[0,1,0] = np.outer(s0, np.eye(n)[-1])
for eta in range(n):
    K[1,0,eta] = np.outer(s0, np.eye(n)[eta])
R = np.array([[[.86,.08],[.01,.05]],[[.10,.51],[.19,.20]]])
P=.8*np.eye(n)

def _run_model():
    try:
        repair_controlled_instrument(K, P)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_repair_controlled_instrument(K, P)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_run_model()',
            "gold_call": '_run_gold()',
        },
        {
            "setup": """import numpy as np
n = 4
s0 = np.ones(n) / np.sqrt(n)
K = np.zeros((2,2,n,n,n), dtype=complex)
K[0,0,0] = np.diag(np.ones(n-1), -1)
K[0,1,0] = np.outer(s0, np.eye(n)[-1])
for eta in range(n):
    K[1,0,eta] = np.outer(s0, np.eye(n)[eta])
R = np.array([[[.86,.08],[.01,.05]],[[.10,.51],[.19,.20]]])
P=np.eye(n)

def _run_model():
    try:
        repair_controlled_instrument(K, P, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_repair_controlled_instrument(K, P, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_run_model()',
            "gold_call": '_run_gold()',
        },
        {
            "setup": """import numpy as np
n = 4
s0 = np.ones(n) / np.sqrt(n)
K = np.zeros((2,2,n,n,n), dtype=complex)
K[0,0,0] = np.diag(np.ones(n-1), -1)
K[0,1,0] = np.outer(s0, np.eye(n)[-1])
for eta in range(n):
    K[1,0,eta] = np.outer(s0, np.eye(n)[eta])
R = np.array([[[.86,.08],[.01,.05]],[[.10,.51],[.19,.20]]])
P=np.eye(n)
K[0,0,0,0,0]=np.nan

def _run_model():
    try:
        repair_controlled_instrument(K, P)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_repair_controlled_instrument(K, P)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_run_model()',
            "gold_call": '_run_gold()',
        },
    ]
