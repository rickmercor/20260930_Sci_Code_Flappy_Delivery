#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def uncertain_system_matrices(A0: np.ndarray, B0: np.ndarray, A_dirs: list, B_dirs: list, theta: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    A0 = np.asarray(A0, dtype=float)
    B0 = np.asarray(B0, dtype=float)
    th = np.asarray(theta, dtype=float)
    if A0.ndim != 2 or A0.shape[0] != A0.shape[1] or A0.shape[0] < 1:
        raise ValueError("A0 must be a square 2-D array")
    if B0.ndim != 2 or B0.shape[0] != A0.shape[0]:
        raise ValueError("B0 must be 2-D with the same number of rows as A0")
    if th.ndim != 1:
        raise ValueError("theta must be one-dimensional")
    if len(A_dirs) != th.shape[0] or len(B_dirs) != th.shape[0]:
        raise ValueError("A_dirs and B_dirs must have the same length as theta")
    if not (np.all(np.isfinite(A0)) and np.all(np.isfinite(B0))
            and np.all(np.isfinite(th))):
        raise ValueError("inputs must be finite")
    A = A0.copy()
    B = B0.copy()
    for j in range(th.shape[0]):
        Aj = np.asarray(A_dirs[j], dtype=float)
        Bj = np.asarray(B_dirs[j], dtype=float)
        if Aj.shape != A0.shape:
            raise ValueError("each entry of A_dirs must have the shape of A0")
        if Bj.shape != B0.shape:
            raise ValueError("each entry of B_dirs must have the shape of B0")
        if not (np.all(np.isfinite(Aj)) and np.all(np.isfinite(Bj))):
            raise ValueError("inputs must be finite")
        A = A + float(th[j]) * Aj
        B = B + float(th[j]) * Bj
    return np.hstack([A, B])

def lyapunov_certificate_margin(A: np.ndarray, B: np.ndarray, K: np.ndarray, P: np.ndarray, alpha: float) -> float:
    """Reference implementation."""
    import numpy as np
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    K = np.asarray(K, dtype=float)
    P = np.asarray(P, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square 2-D array")
    n = A.shape[0]
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError("B must be 2-D with the same number of rows as A")
    m = B.shape[1]
    if K.shape != (m, n):
        raise ValueError("K must have shape (m, n)")
    if P.shape != (n, n):
        raise ValueError("P must have shape (n, n)")
    if not np.allclose(P, P.T, rtol=0.0, atol=1e-10):
        raise ValueError("P must be symmetric within atol 1e-10")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating, np.integer)):
        raise ValueError("alpha must be a real number in (0, 1]")
    a = float(alpha)
    if not np.isfinite(a) or a <= 0.0 or a > 1.0:
        raise ValueError("alpha must be a finite real number in (0, 1]")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))
            and np.all(np.isfinite(K)) and np.all(np.isfinite(P))):
        raise ValueError("inputs must be finite")
    cl = A + B @ K
    M = a * a * P - cl.T @ P @ cl
    M = 0.5 * (M + M.T)
    return float(np.min(np.linalg.eigvalsh(M)))

def select_active_index(x: np.ndarray, z: np.ndarray, P_list: list) -> int:
    """Reference implementation."""
    import numpy as np
    xv = np.asarray(x, dtype=float)
    zv = np.asarray(z, dtype=float)
    if xv.ndim != 1 or xv.shape[0] < 1:
        raise ValueError("x must be a one-dimensional array")
    if zv.ndim != 1 or zv.shape[0] != 2:
        raise ValueError("z must be a one-dimensional array of length 2")
    q = len(P_list)
    if q < 1:
        raise ValueError("P_list must be non-empty")
    n = xv.shape[0]
    Ps = []
    for Pj in P_list:
        Pa = np.asarray(Pj, dtype=float)
        if Pa.shape != (n, n):
            raise ValueError("each entry of P_list must have shape (n, n)")
        if not np.allclose(Pa, Pa.T, rtol=0.0, atol=1e-10):
            raise ValueError("each entry of P_list must be symmetric within atol 1e-10")
        if not np.all(np.isfinite(Pa)):
            raise ValueError("inputs must be finite")
        Ps.append(Pa)
    if not (np.all(np.isfinite(xv)) and np.all(np.isfinite(zv))):
        raise ValueError("inputs must be finite")
    z1 = float(zv[0])
    z2 = int(round(float(zv[1])))
    if 1 <= z2 <= q:
        if float(xv @ Ps[z2 - 1] @ xv) <= z1:
            return int(z2)
        return int((z2 % q) + 1)
    return 1

def controller_state_update(x: np.ndarray, index: int, P_list: list, alpha: float) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    xv = np.asarray(x, dtype=float)
    if xv.ndim != 1 or xv.shape[0] < 1:
        raise ValueError("x must be a one-dimensional array")
    q = len(P_list)
    if q < 1:
        raise ValueError("P_list must be non-empty")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("index must be an integer in 1..q")
    idx = int(index)
    if idx < 1 or idx > q:
        raise ValueError("index must be an integer in 1..q")
    n = xv.shape[0]
    Ps = []
    for Pj in P_list:
        Pa = np.asarray(Pj, dtype=float)
        if Pa.shape != (n, n):
            raise ValueError("each entry of P_list must have shape (n, n)")
        if not np.allclose(Pa, Pa.T, rtol=0.0, atol=1e-10):
            raise ValueError("each entry of P_list must be symmetric within atol 1e-10")
        if not np.all(np.isfinite(Pa)):
            raise ValueError("inputs must be finite")
        Ps.append(Pa)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating, np.integer)):
        raise ValueError("alpha must be a real number in (0, 1]")
    a = float(alpha)
    if not np.isfinite(a) or a <= 0.0 or a > 1.0:
        raise ValueError("alpha must be a finite real number in (0, 1]")
    if not np.all(np.isfinite(xv)):
        raise ValueError("inputs must be finite")
    level = a * a * float(xv @ Ps[idx - 1] @ xv) - float(xv @ xv)
    return np.array([level, float(idx)], dtype=float)

def closed_loop_step(A: np.ndarray, B: np.ndarray, x: np.ndarray, index: int, K_list: list) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    xv = np.asarray(x, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square 2-D array")
    n = A.shape[0]
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError("B must be 2-D with the same number of rows as A")
    m = B.shape[1]
    if xv.ndim != 1 or xv.shape[0] != n:
        raise ValueError("x must be one-dimensional of length n")
    q = len(K_list)
    if q < 1:
        raise ValueError("K_list must be non-empty")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("index must be an integer in 1..q")
    idx = int(index)
    if idx < 1 or idx > q:
        raise ValueError("index must be an integer in 1..q")
    Ks = []
    for Kj in K_list:
        Ka = np.asarray(Kj, dtype=float)
        if Ka.shape != (m, n):
            raise ValueError("each entry of K_list must have shape (m, n)")
        if not np.all(np.isfinite(Ka)):
            raise ValueError("inputs must be finite")
        Ks.append(Ka)
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))
            and np.all(np.isfinite(xv))):
        raise ValueError("inputs must be finite")
    u = Ks[idx - 1] @ xv
    return A @ xv + B @ u

def switched_trajectory(A: np.ndarray, B: np.ndarray, x0: np.ndarray, K_list: list, P_list: list, alpha: float, n_steps: int) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    xv = np.asarray(x0, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square 2-D array")
    n = A.shape[0]
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError("B must be 2-D with the same number of rows as A")
    if xv.ndim != 1 or xv.shape[0] != n:
        raise ValueError("x0 must be one-dimensional of length n")
    if len(K_list) < 1 or len(P_list) < 1:
        raise ValueError("K_list and P_list must be non-empty")
    if len(K_list) != len(P_list):
        raise ValueError("K_list and P_list must have the same length")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)):
        raise ValueError("n_steps must be an integer >= 1")
    T = int(n_steps)
    if T < 1:
        raise ValueError("n_steps must be an integer >= 1")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating, np.integer)):
        raise ValueError("alpha must be a real number in (0, 1]")
    a = float(alpha)
    if not np.isfinite(a) or a <= 0.0 or a > 1.0:
        raise ValueError("alpha must be a finite real number in (0, 1]")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))
            and np.all(np.isfinite(xv))):
        raise ValueError("inputs must be finite")
    margins = [lyapunov_certificate_margin(A, B, K_list[j], P_list[j], a)
               for j in range(len(K_list))]
    if not any(m > 0.0 for m in margins):
        raise ValueError("no candidate certificate is valid for the realized system")
    x = xv.copy()
    z = np.array([0.0, 0.0], dtype=float)
    states = np.empty((T, n), dtype=float)
    indices = np.empty(T, dtype=int)
    for t in range(T):
        idx = select_active_index(x, z, P_list)
        z_next = controller_state_update(x, idx, P_list, a)
        x = closed_loop_step(A, B, x, idx, K_list)
        z = z_next
        states[t, :] = x
        indices[t] = idx
    return np.column_stack([states, indices.astype(float)])

def terminal_state_norm(A0: np.ndarray, B0: np.ndarray, A_dirs: list, B_dirs: list, theta: np.ndarray, x0: np.ndarray, K_list: list, P_list: list, alpha: float, n_steps: int) -> float:
    """Reference implementation."""
    import numpy as np
    AB = uncertain_system_matrices(A0, B0, A_dirs, B_dirs, theta)
    n = A0.shape[0]
    A, B = AB[:, :n], AB[:, n:]
    trace = switched_trajectory(A, B, x0, K_list, P_list, alpha,
                                        n_steps)
    return float(np.linalg.norm(trace[-1, :-1]))
SCICODE_GOLD_EOF
