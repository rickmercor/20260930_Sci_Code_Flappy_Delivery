#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def build_clock_instrument(n: int) -> "np.ndarray":
    import numpy as np
    from numbers import Integral

    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 2:
        raise ValueError("n must be an integer at least 2")
    n = int(n)
    kraus = np.zeros((2, 2, n, n, n), dtype=complex)
    reset = np.ones(n) / np.sqrt(n)
    kraus[0, 0, 0] = np.diag(np.ones(n - 1), -1)
    kraus[0, 1, 0, :, -1] = reset
    for eta in range(n):
        kraus[1, 0, eta, :, eta] = reset
    return kraus

def solve_routed_stationary(kraus: "np.ndarray", reference: "np.ndarray") -> "np.ndarray":
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

def select_memory_subspace(blocks: "np.ndarray", rank: int) -> "np.ndarray":
    import numpy as np
    from numbers import Integral

    try:
        blocks = np.asarray(blocks, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("numeric blocks are required") from exc
    if blocks.ndim != 3 or min(blocks.shape) < 1 or blocks.shape[1] != blocks.shape[2] or not np.isfinite(blocks).all():
        raise ValueError("invalid block array")
    d = blocks.shape[-1]
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, Integral) or not 1 <= rank <= d:
        raise ValueError("invalid retained dimension")
    if not np.allclose(blocks, blocks.conj().transpose(0,2,1), atol=1e-10, rtol=0):
        raise ValueError("blocks must be Hermitian")
    if np.min(np.linalg.eigvalsh(blocks)) < -1e-10 or abs(np.trace(blocks, axis1=1, axis2=2).sum()-1) > 1e-10:
        raise ValueError("blocks must be positive and jointly normalized")
    rho = blocks.sum(axis=0)
    if rank == d:
        return np.eye(d, dtype=complex)
    values, vectors = np.linalg.eigh((rho + rho.conj().T)/2)
    if values[d-rank] - values[d-rank-1] <= 1e-10:
        raise ValueError("spectral cut is not unique")
    retained = vectors[:, -int(rank):]
    return retained @ retained.conj().T

def repair_controlled_instrument(kraus: "np.ndarray", projector: "np.ndarray", support_tol: float = 1e-12) -> "np.ndarray":
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

def compute_history_rate(kraus: "np.ndarray", repaired: "np.ndarray", reference: "np.ndarray") -> float:
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

def compute_record_loglikelihood(kraus: "np.ndarray", initial: "np.ndarray", stimuli: "np.ndarray", actions: "np.ndarray") -> float:
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

def evaluate_compression_logratio(
    n: int = 6,
    reference: "np.ndarray" = (((.86,.08),(.01,.05)),((.10,.51),(.19,.20))),
    rate_limit: float = .007,
    stimuli: "np.ndarray" = (0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0),
    actions: "np.ndarray" = (0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1)
) -> float:
    import numpy as np
    from numbers import Real

    if isinstance(rate_limit, (bool, np.bool_)) or not isinstance(rate_limit, Real) or not np.isfinite(rate_limit) or rate_limit < 0:
        raise ValueError("rate_limit must be a nonnegative finite real")
    K = build_clock_instrument(n)
    blocks = solve_routed_stationary(K, reference)
    rho = blocks.sum(axis=0)
    selected_P = None
    selected_B = None
    for rank in range(2, int(n)+1):
        P = select_memory_subspace(blocks, rank)
        B = repair_controlled_instrument(K, P, 1e-12)
        rate = compute_history_rate(K, B, reference)
        if rate <= rate_limit:
            selected_P, selected_B = P, B
            break
    if selected_P is None:
        raise ValueError("no admissible retained rank")
    initial_reduced = selected_P @ rho @ selected_P
    initial_reduced /= np.trace(initial_reduced).real
    original_log = compute_record_loglikelihood(K, rho, stimuli, actions)
    reduced_log = compute_record_loglikelihood(selected_B, initial_reduced, stimuli, actions)
    return float(reduced_log-original_log)
SCICODE_GOLD_EOF
