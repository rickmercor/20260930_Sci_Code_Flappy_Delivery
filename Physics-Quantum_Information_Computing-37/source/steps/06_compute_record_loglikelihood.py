"""
Condition sequentially on a supplied stimulus/action record and accumulate its finite log probability.

An observed action does not reveal its environment outcome. The corresponding conditional memory update is therefore a completely positive sum over environment outcomes, not a coherent sum of Kraus amplitudes. Normalize the posterior after each observation and use it for the next observation. Stimuli in this calculation are externally forced: no probability from the reference input source enters the likelihood. Accumulate natural logarithms of one-step conditional probabilities so that long nonzero-probability records remain evaluable even when their full probability underflows.

Returns
-------
float, natural logarithm of the conditional record probability, in nats.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

import numpy as np

def compute_record_loglikelihood(kraus: np.ndarray, initial: np.ndarray, stimuli: np.ndarray, actions: np.ndarray) -> float:
    """Evaluate a forced-stimulus record by sequential quantum filtering.

    Parameters
    ----------
    kraus : np.ndarray
        Finite complex array (X,Y,E,d,d), complete on one common nonzero
        orthogonal support P. Each operator has B=P@B@P. Identity support is
        allowed. Completeness and support tolerance is 1e-10.
    initial : np.ndarray
        Finite Hermitian positive semidefinite (d,d) density matrix of trace
        one, supported on P. These checks use absolute tolerance 1e-10.
    stimuli : np.ndarray
        One-dimensional integer array, length T, with entries in [0,X).
        Booleans are excluded. Empty records are allowed.
    actions : np.ndarray
        One-dimensional integer array, length T, with entries in [0,Y).
        Booleans are excluded. Environment outcomes are unobserved.

    Returns
    -------
    log_probability : float
        Native finite Python float in nats. The empty-record value is zero.
        Posteriors are normalized sequentially; no input array is mutated.
        The result must remain finite for long records with positive
        one-step probabilities even when their product underflows.

    Raises
    ------
    ValueError
        If shapes, finiteness, instrument support, density-matrix conditions
        or integer record ranges fail, lengths differ, or any observed action
        has zero conditional probability.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_record_loglikelihood(kraus: "np.ndarray", initial: "np.ndarray", stimuli: "np.ndarray", actions: "np.ndarray") -> float:
    import math
    import numpy as np

    try:
        K = np.asarray(kraus, dtype=complex)
        state = np.asarray(initial, dtype=complex).copy()
        xs, ys = np.asarray(stimuli), np.asarray(actions)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("numeric arrays are required") from exc
    if K.ndim != 5 or min(K.shape) < 1 or K.shape[-2] != K.shape[-1] or not np.isfinite(K).all():
        raise ValueError("invalid instrument")
    X,Y,E,d,_ = K.shape
    if state.shape != (d,d) or not np.isfinite(state).all() or not np.allclose(state, state.conj().T, atol=1e-10, rtol=0):
        raise ValueError("invalid initial density matrix")
    if np.min(np.linalg.eigvalsh(state)) < -1e-10 or abs(np.trace(state)-1)>1e-10:
        raise ValueError("initial density matrix must be positive and normalized")
    if xs.ndim != 1 or ys.ndim != 1 or len(xs) != len(ys):
        raise ValueError("records must have matching one-dimensional shapes")
    if xs.size and (not np.issubdtype(xs.dtype, np.integer) or np.any(xs<0) or np.any(xs>=X)):
        raise ValueError("invalid stimulus record")
    if ys.size and (not np.issubdtype(ys.dtype, np.integer) or np.any(ys<0) or np.any(ys>=Y)):
        raise ValueError("invalid action record")
    if np.issubdtype(xs.dtype, np.bool_) or np.issubdtype(ys.dtype, np.bool_):
        raise ValueError("Boolean records are not integer records")
    P = sum((A.conj().T @ A for A in K[0].reshape(-1,d,d)), np.zeros((d,d),complex))
    if not np.allclose(P @ P, P, atol=1e-10, rtol=0) or np.trace(P).real<.5:
        raise ValueError("invalid common support")
    for x in range(X):
        gram = sum((A.conj().T @ A for A in K[x].reshape(-1,d,d)), np.zeros((d,d),complex))
        if not np.allclose(gram,P,atol=1e-10,rtol=0) or not np.allclose(K[x],P @ K[x] @ P,atol=1e-10,rtol=0):
            raise ValueError("instrument is not complete on its support")
    if not np.allclose(P @ state @ P,state,atol=1e-10,rtol=0):
        raise ValueError("initial state lies outside the instrument support")
    logs=[]
    for x,y in zip(xs,ys):
        operators = K[int(x),int(y)]
        updated = sum((A @ state @ A.conj().T for A in operators), np.zeros((d,d),complex))
        updated = (updated+updated.conj().T)/2
        probability = float(np.trace(updated).real)
        if not np.isfinite(probability) or probability <= 0 or probability > 1+1e-9:
            raise ValueError("observed action has invalid conditional probability")
        logs.append(math.log(probability))
        state = updated/probability
    return float(math.fsum(logs))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic case specifications."""
    return [
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
S=np.tril(np.ones((n,n)))/np.sqrt(np.arange(n,0,-1))[None,:]
rho=(S*np.array([.4,.3,.2,.1]))@S.T
x=np.array([0,0,1,0,0,0,0])
y=np.array([0,1,0,0,0,1,0])
""",
            "call": 'compute_record_loglikelihood(K, rho, x, y)',
            "gold_call": '_oracle_compute_record_loglikelihood(K, rho, x, y)',
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
S=np.tril(np.ones((n,n)))/np.sqrt(np.arange(n,0,-1))[None,:]
rho=(S*np.array([.4,.3,.2,.1]))@S.T
x=np.array([0,0,1,0,0,0,0])
y=np.array([0,1,0,0,0,1,0])
x=np.array([],dtype=int); y=x.copy()
""",
            "call": 'compute_record_loglikelihood(K, rho, x, y)',
            "gold_call": '_oracle_compute_record_loglikelihood(K, rho, x, y)',
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
S=np.tril(np.ones((n,n)))/np.sqrt(np.arange(n,0,-1))[None,:]
rho=(S*np.array([.4,.3,.2,.1]))@S.T
x=np.array([0,0,1,0,0,0,0])
y=np.array([0,1,0,0,0,1,0])
x=np.zeros(1000,dtype=int); y=np.ones(1000,dtype=int)
""",
            "call": 'compute_record_loglikelihood(K, rho, x, y)',
            "gold_call": '_oracle_compute_record_loglikelihood(K, rho, x, y)',
        },
        {
            "setup": """import numpy as np
K=np.zeros((1,1,2,2,2),complex);K[0,0,0]=np.eye(2)/np.sqrt(2);K[0,0,1]=np.diag([1j,-1j])/np.sqrt(2)
rho=np.array([[.5,.2j],[-.2j,.5]])
x=np.zeros(5,dtype=int);y=x.copy()
""",
            "call": 'compute_record_loglikelihood(K, rho, x, y)',
            "gold_call": '_oracle_compute_record_loglikelihood(K, rho, x, y)',
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
S=np.tril(np.ones((n,n)))/np.sqrt(np.arange(n,0,-1))[None,:]
rho=(S*np.array([.4,.3,.2,.1]))@S.T
x=np.array([0,0,1,0,0,0,0])
y=np.array([0,1,0,0,0,1,0])
x=np.array([1]);y=np.array([1])

def _run_model():
    try:
        compute_record_loglikelihood(K, rho, x, y)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_compute_record_loglikelihood(K, rho, x, y)
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
S=np.tril(np.ones((n,n)))/np.sqrt(np.arange(n,0,-1))[None,:]
rho=(S*np.array([.4,.3,.2,.1]))@S.T
x=np.array([0,0,1,0,0,0,0])
y=np.array([0,1,0,0,0,1,0])
y=y[:-1]

def _run_model():
    try:
        compute_record_loglikelihood(K, rho, x, y)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_compute_record_loglikelihood(K, rho, x, y)
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
S=np.tril(np.ones((n,n)))/np.sqrt(np.arange(n,0,-1))[None,:]
rho=(S*np.array([.4,.3,.2,.1]))@S.T
x=np.array([0,0,1,0,0,0,0])
y=np.array([0,1,0,0,0,1,0])
rho *= 2

def _run_model():
    try:
        compute_record_loglikelihood(K, rho, x, y)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_compute_record_loglikelihood(K, rho, x, y)
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
S=np.tril(np.ones((n,n)))/np.sqrt(np.arange(n,0,-1))[None,:]
rho=(S*np.array([.4,.3,.2,.1]))@S.T
x=np.array([0,0,1,0,0,0,0])
y=np.array([0,1,0,0,0,1,0])
x=x.astype(float)

def _run_model():
    try:
        compute_record_loglikelihood(K, rho, x, y)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_compute_record_loglikelihood(K, rho, x, y)
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
