"""
Evaluate the purified-history divergence rate from the common-label mixed transfer map under the correlated reference.

A repaired agent and its original have a definite overlap only after their inaccessible dilation labels have been aligned. Zero-pad the original instrument on the added environment label; do not optimize the environment gauge or replace the mixed transfer by an observed-output fidelity. Route both instruments through the same finite-memory source and obtain the asymptotic purified-history divergence rate from the spectral radius of the rectangular mixed transfer. The source is classical, so only its diagonal blocks contribute nonzero transfer eigenvalues. The answer is in bits per timestep; zero overlap is outside the finite-rate contract.

Returns
-------
float, finite purified-history divergence rate in bits per timestep.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_history_rate(kraus: np.ndarray, repaired: np.ndarray, reference: np.ndarray) -> float:
    """Compute the common-dilation purified-history divergence certificate.

    Parameters
    ----------
    kraus : np.ndarray
        Finite complex original instrument (X,Y,E,d,d), complete on identity.
    repaired : np.ndarray
        Finite complex instrument (X,Y,E+1,d,d), complete on one common
        nonzero orthogonal support P and satisfying B=P@B@P. Label E is
        compared with a zero operator in the original dilation. Completeness
        and support are checked at absolute tolerance 1e-10.
    reference : np.ndarray
        Real nonnegative array (C,X,C) of old-source/stimulus/new-source
        probabilities. Each old-state slice sums to one within 1e-12.

    Returns
    -------
    rate : float
        Native finite Python float, in bits per timestep, obtained from the
        common-label purified-history mixed transfer spectral radius. The
        original and reduced histories use identical source labels. A value
        of the spectral radius above one by at most 1e-9 is clipped to one.
        Identical zero-padded instruments have rate exactly zero.

    Raises
    ------
    ValueError
        If shapes, finiteness, normalization, completeness or support fail,
        or the mixed spectral radius is zero, nonfinite, or greater than
        1+1e-9. The supplied histories must have nonzero asymptotic overlap.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_history_rate(kraus: "np.ndarray", repaired: "np.ndarray", reference: "np.ndarray") -> float:
    import numpy as np

    try:
        K = np.asarray(kraus, dtype=complex)
        B = np.asarray(repaired, dtype=complex)
        raw = np.asarray(reference)
        if np.iscomplexobj(raw) and np.any(raw.imag != 0):
            raise ValueError("reference must be real")
        R = np.asarray(raw.real, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("numeric arrays are required") from exc
    if K.ndim != 5 or min(K.shape) < 1 or K.shape[-2] != K.shape[-1] or not np.isfinite(K).all():
        raise ValueError("invalid original instrument")
    X,Y,E,d,_ = K.shape
    if B.shape != (X,Y,E+1,d,d) or not np.isfinite(B).all():
        raise ValueError("invalid repaired instrument")
    if R.ndim != 3 or R.shape[0] < 1 or R.shape[1] != X or R.shape[2] != R.shape[0]:
        raise ValueError("invalid reference shape")
    if not np.isfinite(R).all() or np.any(R < 0) or not np.allclose(R.sum(axis=(1,2)), 1, atol=1e-12, rtol=0):
        raise ValueError("reference must be stochastic")
    P = sum((A.conj().T @ A for A in B[0].reshape(-1,d,d)), np.zeros((d,d),complex))
    if not np.allclose(P @ P, P, atol=1e-10, rtol=0) or np.trace(P).real < .5:
        raise ValueError("invalid repaired support")
    for x in range(X):
        orig_gram = sum((A.conj().T @ A for A in K[x].reshape(-1,d,d)), np.zeros((d,d),complex))
        gram = sum((A.conj().T @ A for A in B[x].reshape(-1,d,d)), np.zeros((d,d),complex))
        if not np.allclose(orig_gram, np.eye(d), atol=1e-10, rtol=0) or not np.allclose(gram, P, atol=1e-10, rtol=0):
            raise ValueError("instruments are not complete on their supports")
        if not np.allclose(B[x], P @ B[x] @ P, atol=1e-10, rtol=0):
            raise ValueError("repaired operators leave their support")
    if np.array_equal(B[:,:,:E], K) and not np.any(B[:,:,E]):
        return 0.0
    C = len(R)
    mixed = np.zeros((C*d*d,C*d*d),complex)
    for c in range(C):
        for cp in range(C):
            for x in range(X):
                for y in range(Y):
                    for eta in range(E):
                        mixed[cp*d*d:(cp+1)*d*d,c*d*d:(c+1)*d*d] += R[c,x,cp]*np.kron(B[x,y,eta], K[x,y,eta].conj())
    mu = float(np.max(np.abs(np.linalg.eigvals(mixed))))
    if not np.isfinite(mu) or mu <= 0 or mu > 1+1e-9:
        raise ValueError("finite nonzero-overlap certificate is undefined")
    return float(max(0.0, -0.5*np.log2(min(mu,1.0))))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic case specifications."""
    return [
        {
            "setup": """import numpy as np
K=np.zeros((2,2,1,2,2),complex)
B=np.zeros((2,2,2,2,2),complex)
for x in range(2):
    a=np.arange(1,9).reshape(4,2)
    t=np.sin(a*(.31+.1*x))+1j*np.cos(a*(.47+.03*x))
    q,_=np.linalg.qr(t)
    K[x,:,0]=q.reshape(2,2,2)
    t=np.cos(a*(.27+.07*x))+1j*np.sin(a*(.61+.04*x))
    q,_=np.linalg.qr(t)
    B[x,:,0]=q.reshape(2,2,2)
R=np.array([[[.86,.08],[.01,.05]],[[.10,.51],[.19,.20]]])
""",
            "call": 'compute_history_rate(K, B, R)',
            "gold_call": '_oracle_compute_history_rate(K, B, R)',
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
B=np.pad(K,((0,0),(0,0),(0,1),(0,0),(0,0)))
""",
            "call": 'compute_history_rate(K, B, R)',
            "gold_call": '_oracle_compute_history_rate(K, B, R)',
        },
        {
            "setup": """import numpy as np
K=np.sqrt(np.array([.2,.8])).reshape(1,2,1,1,1)
B=np.zeros((1,2,2,1,1));B[0,:,0,0,0]=np.sqrt([.7,.3])
R=np.ones((1,1,1))
""",
            "call": 'compute_history_rate(K, B, R)',
            "gold_call": '_oracle_compute_history_rate(K, B, R)',
        },
        {
            "setup": """import numpy as np
K=np.ones((1,1,1,1,1));B=np.array([.6,.8]).reshape(1,1,2,1,1)
R=np.ones((1,1,1))
""",
            "call": 'compute_history_rate(K, B, R)',
            "gold_call": '_oracle_compute_history_rate(K, B, R)',
        },
        {
            "setup": """import numpy as np
K=np.zeros((2,2,1,2,2),complex)
B=np.zeros((2,2,2,2,2),complex)
for x in range(2):
    a=np.arange(1,9).reshape(4,2)
    t=np.sin(a*(.31+.1*x))+1j*np.cos(a*(.47+.03*x))
    q,_=np.linalg.qr(t)
    K[x,:,0]=q.reshape(2,2,2)
    t=np.cos(a*(.27+.07*x))+1j*np.sin(a*(.61+.04*x))
    q,_=np.linalg.qr(t)
    B[x,:,0]=q.reshape(2,2,2)
R=np.array([[[.86,.08],[.01,.05]],[[.10,.51],[.19,.20]]])
B *= .9

def _run_model():
    try:
        compute_history_rate(K, B, R)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_compute_history_rate(K, B, R)
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
K=np.zeros((2,2,1,2,2),complex)
B=np.zeros((2,2,2,2,2),complex)
for x in range(2):
    a=np.arange(1,9).reshape(4,2)
    t=np.sin(a*(.31+.1*x))+1j*np.cos(a*(.47+.03*x))
    q,_=np.linalg.qr(t)
    K[x,:,0]=q.reshape(2,2,2)
    t=np.cos(a*(.27+.07*x))+1j*np.sin(a*(.61+.04*x))
    q,_=np.linalg.qr(t)
    B[x,:,0]=q.reshape(2,2,2)
R=np.array([[[.86,.08],[.01,.05]],[[.10,.51],[.19,.20]]])
R[0,0,0]=np.inf

def _run_model():
    try:
        compute_history_rate(K, B, R)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_compute_history_rate(K, B, R)
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
K=np.array([1.,0.]).reshape(1,2,1,1,1)
B=np.zeros((1,2,2,1,1));B[0,1,0,0,0]=1
R=np.ones((1,1,1))

def _run_model():
    try:
        compute_history_rate(K, B, R)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_compute_history_rate(K, B, R)
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
