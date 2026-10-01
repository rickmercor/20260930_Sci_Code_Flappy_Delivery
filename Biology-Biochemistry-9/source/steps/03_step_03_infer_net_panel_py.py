"""
Infer the transcriptome specific net flux panel.

The required flux state combines reaction-specific transcriptomic evidence with the source treatment of biological directionality. It is the phenotype-specific stationary state for each supplied panel entry.

Returns
-------
A finite ndarray of shape (C,P,N), in the input reaction orientation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def infer_net_panel(
    balance: "np.ndarray", weights: "np.ndarray", irreversible: "np.ndarray"
) -> "np.ndarray":
    """
    balance is a finite (r,N+1) independent-row system; its last column is demand.
    weights is positive finite (C,P,N). irreversible is a unique integer vector of zero-
    based reaction indices, of length at most six. These indices carry forward-only
    biological evidence; all remaining net directions are unspecified. Use the source-
    matched transcriptomic maximum-entropy treatment: keep a strictly positive forward
    and reverse component f_j and r_j for every reaction, whose difference f_j - r_j is
    that reaction's net flux, and maximise -sum_j [f_j ln(f_j / g_j) + r_j ln(r_j / g_j)]
    with g_j the supplied reaction weight, subject to the balance rows and to
    f_j - r_j >= 0 on the reactions named by irreversible. Return finite signed net fluxes,
    shape (C,P,N). Raise ValueError for invalid inputs or an infeasible state. Inputs
    use the common numerical units declared in the main problem. Numerical acceptance
    uses absolute tolerance 2e-6.

    An unresolved stationary solver raises RuntimeError."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_infer_net_panel(
    balance: "np.ndarray", weights: "np.ndarray", irreversible: "np.ndarray"
) -> "np.ndarray":
    import numpy as np
    from itertools import combinations
    from scipy.linalg import null_space

    a = np.asarray(balance, dtype=float)
    g = np.asarray(weights, dtype=float)
    ids = np.asarray(irreversible)
    if (
        g.ndim != 3
        or min(g.shape) == 0
        or a.ndim != 2
        or (a.shape[1] != g.shape[-1] + 1)
        or (not np.isfinite(a).all())
        or (not np.isfinite(g).all())
        or np.any(g <= 0)
        or (ids.ndim != 1)
        or (not np.issubdtype(ids.dtype, np.integer))
        or (len(ids) > 6)
        or (len(set(ids.tolist())) != len(ids))
        or np.any(ids < 0)
        or np.any(ids >= g.shape[-1])
    ):
        raise ValueError("Invalid panel, balance, or direction indices")
    n = g.shape[-1]
    A = a[:, :n]
    b = a[:, n]
    if np.linalg.matrix_rank(A, tol=1e-10) != len(A):
        raise ValueError("Independent balance rows required")
    faces = []
    for k in range(len(ids) + 1):
        for active in combinations(ids.tolist(), k):
            C = np.vstack([A, np.eye(n)[list(active)]])
            d = np.r_[b, np.zeros(k)]
            x = np.linalg.lstsq(C, d, rcond=1e-12)[0]
            if np.max(np.abs(C @ x - d), initial=0) > 1e-09:
                continue
            N = null_space(C, rcond=1e-12)
            faces.append((x, N))
    output = np.empty_like(g)
    for index in np.ndindex(g.shape[:-1]):
        scale = 2 * g[index] / np.e
        best = None
        best_f = np.inf

        def _cost(v):
            return float(np.sum(v * np.arcsinh(v / scale) - np.hypot(v, scale)))

        for origin, N in faces:
            v = origin.copy()
            for iteration in range(100):
                grad = N.T @ np.arcsinh(v / scale)
                if grad.size == 0 or np.max(np.abs(grad)) < 2e-12:
                    break
                hess = N.T / np.hypot(v, scale) @ N
                step = -N @ np.linalg.solve(hess, grad)
                decrement = float(grad @ np.linalg.solve(hess, grad))
                alpha = 1.0
                old = _cost(v)
                while alpha > 2 ** (-40):
                    trial = v + alpha * step
                    if _cost(trial) <= old - 0.0001 * alpha * decrement + 2e-13:
                        break
                    alpha *= 0.5
                v = trial
            else:
                raise RuntimeError("Stationary solve did not converge")
            if np.any(v[ids] < -2e-09):
                continue
            obj = _cost(v)
            if obj < best_f:
                best_f = obj
                best = v.copy()
        if best is None:
            raise ValueError("No state satisfies direction evidence")
        best[np.abs(best) < 2e-11] = 0.0
        output[index] = best
    return output

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\n"
            "a = np.array([[1.0, 1.0, -1.0, 4.0]])\n"
            "g = np.array([[[1.0, 3.0, 5.0], [2.0, 1.0, 9.0]]])\n"
            "i = np.array([2])",
            "call": "infer_net_panel(a, g, i)",
            "gold_call": "_oracle_infer_net_panel(a, g, i)",
            "tol": 2e-06,
        },
        {
            "setup": "import numpy as np\n"
            "a = np.c_[np.eye(3), [0.0, 2.0, -1.0]]\n"
            "g = np.array([[[2.0, 3.0, 4.0]]])\n"
            "i = np.array([0, 1])",
            "call": "infer_net_panel(a, g, i)",
            "gold_call": "_oracle_infer_net_panel(a, g, i)",
            "tol": 2e-06,
        },
        {
            "setup": "import numpy as np\n"
            "a = np.array([[1.0, -1.0, 0.0, 2.0], [0.0, 1.0, 1.0, 1.0]])\n"
            "g = np.array([[[0.4, 9.0, 1.0]], [[2.0, 1.0, 7.0]]])\n"
            "i = np.array([0])",
            "call": "infer_net_panel(a, g, i)",
            "gold_call": "_oracle_infer_net_panel(a, g, i)",
            "tol": 2e-06,
        },
        {
            "setup": "import numpy as np\n"
            "a = np.array([[1.0, -2.0]])\n"
            "g = np.ones((1, 1, 1))\n"
            "i = np.array([0])\n"
            "\n"
            "def _error_check(fn, args):\n"
            "    try:\n"
            "        fn(*args)\n"
            "    except ValueError:\n"
            "        return 1.0\n"
            "    return 0.0",
            "call": "_error_check(infer_net_panel, (a,g,i,))",
            "gold_call": "_error_check(_oracle_infer_net_panel, (a,g,i,))",
            "tol": 0.0,
        },
    ]
