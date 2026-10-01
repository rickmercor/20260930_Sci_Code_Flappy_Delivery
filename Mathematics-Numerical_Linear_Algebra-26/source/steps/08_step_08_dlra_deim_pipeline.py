"""
Implement dlra_deim_pipeline as the final end-to-end orchestrator.

Build the discrete nonlinear Schrödinger initial state and coupling matrix, use the earlier full-order integrator to obtain the pre-propagated and reference states, initialise the rank-r factors, certify the Kronecker sampling construction, and call the earlier low-rank integrator through the final time. Return a dictionary containing the relative Frobenius error, its integer accuracy score floor(-log10(rel_error)), and the number of low-rank time steps.

The final pipeline compares an interpolatory dynamical low-rank trajectory with a full-order reference for the discrete nonlinear Schrödinger system. The relative Frobenius error measures the end-to-end approximation, and the reported integer is the number of resolved decimal orders obtained from that error.

Returns
-------
Dict[str, Any] as specified by the function Returns section.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from typing import Dict, Any, Tuple

def dlra_deim_pipeline(n: int, alpha: float, r: int, h_ref: float, h_dlra: float, t0_pre: float, T: float) -> Dict[str, Any]:
    """Run the complete DLRA-DEIM pipeline for the Schrodinger equation.

Parameters
----------
n : int
    Spatial grid size (A in C^{n x n}).
alpha : float
    Nonlinearity parameter.
r : int
    Approximation rank.
h_ref : float
    Step size for the RK4 reference integration.
h_dlra : float
    Step size for PRK2-QDEIM integration.
t0_pre : float
    Pre-propagation time (RK4 from 0 to t0_pre).
T : float
    Final time.

Returns
-------
result : dict
    Dictionary with keys:
    - 'rel_error': float, relative Frobenius error
    - 'answer': int, floor(-log10(rel_error))
    - 'n_steps': int, number of PRK2-QDEIM steps taken

Implementation requirement
--------------------------
Call the previously defined public functions ``prk2_qdeim_integrate``, ``rk4_integrate``, ``schrodinger_rhs`` rather than reproducing their algorithms locally.

Implementation requirement
--------------------------
Call ``kron_qdeim_indices`` after the initialization SVD to certify the source Algorithm 2 two-stage Kronecker sampler before advancing the matrix trajectory."""
    result = {}
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from typing import Dict, Any, Tuple

def _oracle_dlra_deim_pipeline(n: int, alpha: float, r: int, h_ref: float, h_dlra: float, t0_pre: float, T: float) -> Dict[str, Any]:
    """Reference implementation."""
    sigma = 0.15 * n
    mu1, mu2 = (0.7 * n, 0.4 * n)
    nu1, nu2 = (0.6 * n, 0.3 * n)
    j = np.arange(1, n + 1, dtype=float)
    k = np.arange(1, n + 1, dtype=float)
    A0 = (
        np.exp(-((j[:, None] - mu1) ** 2 + (k[None, :] - nu1) ** 2) / sigma ** 2)
        + np.exp(-((j[:, None] - mu2) ** 2 + (k[None, :] - nu2) ** 2) / sigma ** 2)
    ).astype(complex)
    B = np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    rhs_fn = lambda A: _oracle_schrodinger_rhs(A, B, alpha)
    A_t0 = _oracle_rk4_integrate(A0, rhs_fn, 0.0, t0_pre, h_ref)
    A_ref = _oracle_rk4_integrate(A_t0, rhs_fn, t0_pre, T, h_ref)
    U0, s0, Vh0 = np.linalg.svd(A_t0, full_matrices=False)
    _tensor_core = np.eye(int(r) * int(r), int(r), dtype=float)
    _tensor_indices = _oracle_kron_qdeim_indices(U0[:, :int(r)], Vh0[:int(r), :].T, _tensor_core)
    if _tensor_indices.shape != (int(r),) or np.unique(_tensor_indices).size != int(r):
        raise RuntimeError('the two-stage Kronecker QDEIM certificate failed')
    U0, s0, Vh0 = (U0[:, :r], s0[:r], Vh0[:r, :])
    U_N, s_N, Vh_N = _oracle_prk2_qdeim_integrate(U0, s0, Vh0, rhs_fn, t0_pre, T, h_dlra, r)
    Y_N = U_N @ np.diag(s_N) @ Vh_N
    n_steps = int(round((T - t0_pre) / h_dlra))
    rel_error = float(np.linalg.norm(A_ref - Y_N, 'fro') / np.linalg.norm(A_ref, 'fro'))
    answer = int(np.floor(-np.log10(rel_error)))
    return {'rel_error': rel_error, 'answer': answer, 'n_steps': n_steps}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Case 1
        {
            "setup": """import numpy as np
n, alpha, r = 8, 0.5, 2
h_ref, h_dlra = 1e-3, 1e-2
t0_pre, T = 0.01, 0.05
expected_keys = {'rel_error', 'answer', 'n_steps'}
def has_keys(d):
    return set(d.keys()) == expected_keys
""",
            "call": "has_keys(dlra_deim_pipeline(n, alpha, r, h_ref, h_dlra, t0_pre, T))",
            "gold_call": "has_keys(_oracle_dlra_deim_pipeline(n, alpha, r, h_ref, h_dlra, t0_pre, T))",
        },
        # Case 2
        {
            "setup": """import numpy as np
n, alpha, r = 8, 0.5, 2
h_ref, h_dlra = 1e-3, 1e-2
t0_pre, T = 0.01, 0.05
def answer_is_int(d):
    return isinstance(d['answer'], int) and d['answer'] >= 0
""",
            "call": "answer_is_int(dlra_deim_pipeline(n, alpha, r, h_ref, h_dlra, t0_pre, T))",
            "gold_call": "answer_is_int(_oracle_dlra_deim_pipeline(n, alpha, r, h_ref, h_dlra, t0_pre, T))",
        },
        # Case 3
        {
            "setup": """import numpy as np
n, alpha, r = 8, 0.5, 3
h_ref, h_dlra = 1e-3, 5e-3
t0_pre, T = 0.01, 0.06
expected_steps = int(round((T - t0_pre) / h_dlra))
def check_steps(d):
    return d['n_steps'] == expected_steps
""",
            "call": "check_steps(dlra_deim_pipeline(n, alpha, r, h_ref, h_dlra, t0_pre, T))",
            "gold_call": "check_steps(_oracle_dlra_deim_pipeline(n, alpha, r, h_ref, h_dlra, t0_pre, T))",
        },
        # Case 4
        {
            "setup": """import numpy as np
n, alpha, r = 8, 0.5, 3
h_ref, h_dlra = 1e-3, 5e-3
t0_pre, T = 0.01, 0.06
""",
            "call": "dlra_deim_pipeline(n, alpha, r, h_ref, h_dlra, t0_pre, T)['answer']",
            "gold_call": "_oracle_dlra_deim_pipeline(n, alpha, r, h_ref, h_dlra, t0_pre, T)['answer']",
        },
        # Case 5
        {
            "setup": """import numpy as np
n, alpha, r = 8, 0.5, 3
h_ref, h_dlra = 1e-3, 5e-3
t0_pre, T = 0.01, 0.06
def rel_err(d):
    return round(d['rel_error'], 6)
""",
            "call": "rel_err(dlra_deim_pipeline(n, alpha, r, h_ref, h_dlra, t0_pre, T))",
            "gold_call": "rel_err(_oracle_dlra_deim_pipeline(n, alpha, r, h_ref, h_dlra, t0_pre, T))",
        },
        # Case 6
        {
            "setup": """import numpy as np
n, alpha = 10, 0.5
h_ref, h_dlra = 1e-3, 5e-3
t0_pre, T = 0.01, 0.06
def compare_ranks(pipeline_fn):
    e2 = pipeline_fn(n, alpha, 2, h_ref, h_dlra, t0_pre, T)['rel_error']
    e4 = pipeline_fn(n, alpha, 4, h_ref, h_dlra, t0_pre, T)['rel_error']
    return e4 <= e2
""",
            "call": "compare_ranks(dlra_deim_pipeline)",
            "gold_call": "compare_ranks(_oracle_dlra_deim_pipeline)",
        },
        # Case 7
        {
            "setup": """import numpy as np
n, alpha, r = 8, 0.5, 2
h_ref, h_dlra = 1e-3, 1e-2
t0_pre, T = 0.01, 0.05
def det_check(fn):
    r1 = fn(n, alpha, r, h_ref, h_dlra, t0_pre, T)
    r2 = fn(n, alpha, r, h_ref, h_dlra, t0_pre, T)
    return abs(r1['rel_error'] - r2['rel_error']) < 1e-15
""",
            "call": "det_check(dlra_deim_pipeline)",
            "gold_call": "det_check(_oracle_dlra_deim_pipeline)",
        },
        # Case 8
        {
            "setup": """import numpy as np
n, alpha, r = 10, 0.5, 3
h_ref, h_dlra = 1e-3, 5e-3
t0_pre, T = 0.01, 0.06
""",
            "call": "round(dlra_deim_pipeline(n, alpha, r, h_ref, h_dlra, t0_pre, T)['rel_error'], 6)",
            "gold_call": "round(_oracle_dlra_deim_pipeline(n, alpha, r, h_ref, h_dlra, t0_pre, T)['rel_error'], 6)",
        },
        # Case 9
        {
            "setup": """import numpy as np
n, alpha, r = 32, 0.5, 4
h_ref, h_dlra = 1e-4, 1e-3
t0_pre, T = 0.01, 0.51
""",
            "call": "round(dlra_deim_pipeline(n, alpha, r, h_ref, h_dlra, t0_pre, T)['rel_error'], 6)",
            "gold_call": "round(_oracle_dlra_deim_pipeline(n, alpha, r, h_ref, h_dlra, t0_pre, T)['rel_error'], 6)",
        },
    ]
