"""
Calculate finite-community feasibility and stability certificates.

For each plan vector d and climate matrix B, first form K = B/d[None,:]. Let m be the column means of K and C = K-m[None,:]. For the finite feasibility boundary, build an orthonormal Krylov basis from v=ones by repeatedly orthogonalizing v against the existing basis and then replacing v by -C@q; stop when its norm is at most 1e-11 or after q steps. Keep the positive real eigenvalues of Q.T@(-C)@Q whose imaginary magnitude is at most 1e-8. Also find every positive sign-changing root of 1-m@solve(a*I+C,ones) on the 300-point geometric grid from max(1e-8, largest retained pole + 1e-7) to 100, using 80 bisections per bracket. The feasibility value is the largest retained pole or denominator root, including zero. The stability boundary is max(0,-lambda_min((K+K.T)/2)). Return the three feasibility values, the three stability values, and max(minimum_effort, buffer times the largest of those six values).

Returns
-------
return a float64 matrix of shape (plans, 7): three feasibility poles, three symmetric-part stability boundaries, and the buffered lower effort.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def plan_certificates(competition: "np.ndarray", designs: "np.ndarray", buffer: float, minimum_effort: float) -> "np.ndarray":
    """Calculate feasibility and stability certificates.
 
    Returns
    -------
    A float64 matrix with three feasibility thresholds, three stability boundaries, and the lower effort per plan.
 
    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
 
def _canonical(k):
    m = k.mean(axis=0)
    return np.vstack([k - m[None, :], m])
 
 
def _feasibility(p):
    c, m = p[:-1], p[-1]
    n = c.shape[0]
    # Finite poles reached from uniform forcing only.
    v = np.ones(n, dtype=np.float64)
    krylov = []
    for _ in range(n):
        for q in krylov:
            v = v - q * np.dot(q, v)
        norm = np.linalg.norm(v)
        if norm <= 1e-11:
            break
        q = v / norm
        krylov.append(q)
        v = -c @ q
    Q = np.column_stack(krylov)
    A = Q.T @ (-c) @ Q
    eig = np.linalg.eigvals(A)
    poles = eig.real[np.abs(eig.imag) <= 1e-8]
    bounds = [0.0]
    for pole in poles:
        if pole > 0.0:
            bounds.append(float(pole))
    grid = np.unique(np.array(bounds, dtype=np.float64))
    # Locate the last sign boundary of 1 - (aI+C)^-1 1 dot m.
    def _den(a):
        return float(1.0 - m @ np.linalg.solve(a * np.eye(n) + c, np.ones(n)))
    candidates = list(grid)
    test = np.geomspace(max(1e-8, grid.max(initial=0.0) + 1e-7), 100.0, 300)
    prev_a, prev_v = test[0], _den(test[0])
    for a in test[1:]:
        val = _den(a)
        if np.isfinite(val) and np.isfinite(prev_v) and val * prev_v < 0.0:
            lo, hi = prev_a, a
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if _den(lo) * _den(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            candidates.append(0.5 * (lo + hi))
        prev_a, prev_v = a, val
    return float(max(candidates))
 
 
def _oracle_plan_certificates(competition: "np.ndarray", designs: "np.ndarray", buffer: float, minimum_effort: float) -> "np.ndarray":
    B = np.asarray(competition, dtype=np.float64)
    D = np.asarray(designs, dtype=np.float64)
    if B.ndim != 3 or D.ndim != 2 or D.shape[1] != B.shape[1] or buffer <= 1.0 or minimum_effort <= 0.0:
        raise ValueError("invalid plan-certificate inputs")
    rows = []
    for d in D:
        f, g = [], []
        for b in B:
            k = b / d[None, :]
            f.append(_feasibility(_canonical(k)))
            g.append(max(0.0, -float(np.linalg.eigvalsh((k + k.T) / 2.0)[0])))
        lo = max(float(minimum_effort), float(buffer) * max(max(f), max(g)))
        rows.append(f + g + [lo])
    return np.asarray(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = """import numpy as np
B=np.array([[[0.0,0.16,0.10],[0.11,0.0,0.14],[0.09,0.13,0.0]],[[0.0,0.18,0.08],[0.10,0.0,0.16],[0.12,0.11,0.0]],[[0.0,0.14,0.12],[0.13,0.0,0.10],[0.08,0.17,0.0]]])
D=np.array([[0.8,1.0,1.2],[1.1,0.9,1.0]])"""
    return [
        {"setup": common, "call": "plan_certificates(B,D,1.2,0.2)", "gold_call": "_oracle_plan_certificates(B,D,1.2,0.2)", "tol": 1e-9},
        {"setup": common, "call": "plan_certificates(B,D,1.01,0.2)", "gold_call": "_oracle_plan_certificates(B,D,1.01,0.2)", "tol": 1e-9},
        {"setup": common, "call": "plan_certificates(B,D,1.2,2.0)", "gold_call": "_oracle_plan_certificates(B,D,1.2,2.0)", "tol": 1e-9},
        {"setup": common + "\ndef caught(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    except Exception:\n        return np.array([-1.0])\n    return np.array([0.0])", "call": "caught(lambda: plan_certificates(B,D,1.0,0.2))", "gold_call": "caught(lambda: _oracle_plan_certificates(B,D,1.0,0.2))", "tol": 0.0},
    ]
