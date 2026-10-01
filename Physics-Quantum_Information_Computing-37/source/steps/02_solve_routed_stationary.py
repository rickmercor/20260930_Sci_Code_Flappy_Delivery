"""
Find the stationary quantum-memory blocks conditioned on the state of a correlated classical input generator.

For reference probabilities $R_{cx c^{\prime}}$, the source transition $c\to c^{\prime}$ and the selected stimulus $x$ belong to the same event. The joint stationary state is block diagonal in the classical source basis, but its memory blocks generally differ. Solve the routed channel fixed-point problem with one total trace constraint, retaining source–memory correlations rather than averaging the stimulus distribution first. The output blocks are subnormalized: only the sum of their traces equals one. A nonunique stationary state does not define the requested compression target.

Returns
-------
np.ndarray, complex stationary source-conditioned blocks of shape (C, d, d).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solve_routed_stationary(kraus: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Solve the unique stationary state of the classically routed channel.

    Parameters
    ----------
    kraus : np.ndarray
        Finite complex array (X, Y, E, d, d) with nonempty axes. For each
        stimulus, the sum of K.conj().T @ K over actions and environment
        outcomes equals the identity within absolute tolerance 1e-10.
    reference : np.ndarray
        Finite real array (C, X, C), C >= 1, indexed by old source state,
        stimulus, new source state. Entries are nonnegative; each old-state
        slice sums to one within absolute tolerance 1e-12.

    Returns
    -------
    blocks : np.ndarray
        Complex array (C, d, d) of positive semidefinite stationary blocks,
        jointly normalized to total trace one. Inputs are not mutated.

    Raises
    ------
    ValueError
        If shapes, finiteness, stochastic normalization or Kraus completeness
        fail, or the trace-one stationary solution is not unique. The augmented
        stationary linear system must have full column rank with relative singular-value cutoff 1e-12.
    """
    return np.zeros((reference.shape[0], kraus.shape[-1], kraus.shape[-1]), dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_routed_stationary(kraus: "np.ndarray", reference: "np.ndarray") -> "np.ndarray":
    import numpy as np

    try:
        K = np.asarray(kraus, dtype=complex)
        raw = np.asarray(reference)
        if np.iscomplexobj(raw) and np.any(raw.imag != 0):
            raise ValueError("reference must be real")
        R = np.asarray(raw.real, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("numeric arrays are required") from exc
    if K.ndim != 5 or min(K.shape) < 1 or K.shape[-2] != K.shape[-1] or not np.isfinite(K).all():
        raise ValueError("invalid Kraus array")
    X, Y, E, d, _ = K.shape
    if R.ndim != 3 or R.shape[0] < 1 or R.shape[1] != X or R.shape[2] != R.shape[0]:
        raise ValueError("invalid reference shape")
    if not np.isfinite(R).all() or np.any(R < 0) or not np.allclose(R.sum(axis=(1, 2)), 1, atol=1e-12, rtol=0):
        raise ValueError("reference must be stochastic")
    for x in range(X):
        gram = sum((A.conj().T @ A for A in K[x].reshape(-1, d, d)), np.zeros((d,d), complex))
        if not np.allclose(gram, np.eye(d), atol=1e-10, rtol=0):
            raise ValueError("instrument is not trace preserving")
    C = R.shape[0]
    m = C*d*d
    channel = np.zeros((m, m), dtype=complex)
    for c in range(C):
        for cp in range(C):
            for x in range(X):
                for A in K[x].reshape(-1, d, d):
                    channel[cp*d*d:(cp+1)*d*d, c*d*d:(c+1)*d*d] += R[c,x,cp]*np.kron(A, A.conj())
    trace = np.zeros(m)
    for c in range(C):
        trace[c*d*d + np.arange(d)*(d+1)] = 1
    lhs = np.vstack((np.eye(m) - channel, trace))
    rhs = np.zeros(m+1, complex)
    rhs[-1] = 1
    solution, _, rank, _ = np.linalg.lstsq(lhs, rhs, rcond=1e-12)
    if rank != m or np.max(np.abs(lhs @ solution - rhs)) > 1e-10:
        raise ValueError("stationary state is not uniquely determined")
    blocks = solution.reshape(C, d, d)
    blocks = (blocks + blocks.conj().transpose(0, 2, 1))/2
    blocks /= np.trace(blocks, axis1=1, axis2=2).real.sum()
    if min(np.linalg.eigvalsh(blocks).ravel()) < -1e-10:
        raise ValueError("stationary solution is not positive semidefinite")
    return blocks

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
""",
            "call": 'solve_routed_stationary(K, R)',
            "gold_call": '_oracle_solve_routed_stationary(K, R)',
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
R = np.array([[[0.0],[1.0]]])
""",
            "call": 'solve_routed_stationary(K, R)',
            "gold_call": '_oracle_solve_routed_stationary(K, R)',
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
R = np.array([[[1e-7, .4], [0, .6-1e-7]], [[.75, 0], [.25-1e-7, 1e-7]]])
""",
            "call": 'solve_routed_stationary(K, R)',
            "gold_call": '_oracle_solve_routed_stationary(K, R)',
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
K=U @ K @ U.conj().T
R=np.array([[[.4,.1,.1],[.1,.2,.1]],[[.05,.35,.1],[.15,.05,.3]],[[.1,.1,.4],[.2,.1,.1]]])
""",
            "call": 'solve_routed_stationary(K, R)',
            "gold_call": '_oracle_solve_routed_stationary(K, R)',
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
R[0,0,0] = -0.1

def _run_model():
    try:
        solve_routed_stationary(K, R)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_solve_routed_stationary(K, R)
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
K *= 0.8

def _run_model():
    try:
        solve_routed_stationary(K, R)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_solve_routed_stationary(K, R)
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
K=np.eye(2).reshape(1,1,1,2,2)
R=np.ones((1,1,1))

def _run_model():
    try:
        solve_routed_stationary(K, R)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_solve_routed_stationary(K, R)
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
R = np.ones((2,2,3))

def _run_model():
    try:
        solve_routed_stationary(K, R)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_solve_routed_stationary(K, R)
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
