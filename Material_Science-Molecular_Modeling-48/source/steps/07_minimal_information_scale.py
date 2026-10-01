"""
Given a pooled data information matrix and the Jacobian of the target quantities of interest with their target uncertainties, return the ****smallest common weight scale $t \\ge 0$ for which the source paper's information-matching condition holds****. ****Neither the target's information matrix nor the matching condition is given: derive both**** - the first from how a scalar prediction with a prescribed uncertainty translates into an information requirement, the second from the paper. Then reduce the condition to something computable: with a single free scale it is not a semidefinite program but a symmetric-definite generalised eigenvalue problem, and the answer is a single eigenvalue. Raise a `ValueError` if the pooled matrix is singular, because then no finite common weight can satisfy the condition in the deficient directions.

Comparing two information matrices through the positive-semidefinite ordering, rather than through a scalar summary such as a determinant or a trace, is what makes the condition respect the **geometry** of both: it demands enough information in every parameter direction the target actually uses, and none in directions it does not. When the free weights collapse to one overall scale, the condition $t\\,\\mathcal I \\succeq \\mathcal J$ can be whitened: writing $\\mathcal I = LL^{\\mathsf T}$ for a Cholesky factor $L$, it becomes $t\\,\\mathcal E \\succeq L^{-1}\\mathcal J L^{-\\mathsf T}$, whose smallest solution is the largest eigenvalue of the whitened target - equivalently the largest eigenvalue of the pencil $(\\mathcal J, \\mathcal I)$. The target matrix may be rank deficient, which is harmless; the pooled matrix may not be, and if the target has any component in the pooled matrix's null space the problem is infeasible, which is the paper's own diagnosis when a set of surrogate properties fails to cover the parameter directions a target needs.

****--- Formulas ---****

With $H$ the $Q\\times P$ QoI Jacobian, $\\boldsymbol\\delta$ the $Q$ target uncertainties and $D = \\mathrm{diag}(\\boldsymbol\\delta)$,

$$\\mathcal J \\;=\\; \\sum_{n=1}^{Q}\\frac{1}{\\delta_n^{2}}\\,\\boldsymbol h_n\\boldsymbol h_n^{\\mathsf T} \\;=\\; \\big(D^{-1}H\\big)^{\\mathsf T}\\big(D^{-1}H\\big), \\qquad \\boldsymbol h_n^{\\mathsf T} = \\text{row } n \\text{ of } H,$$

$$t^{\\star} \\;=\\; \\min\\big\\{\\,t \\ge 0 \\;:\\; t\\,\\mathcal I_{\\mathrm{pool}} - \\mathcal J \\succeq 0\\,\\big\\} \\;=\\; \\max\\Big(0,\\; \\lambda_{\\max}\\big(L^{-1}\\mathcal J L^{-\\mathsf T}\\big)\\Big), \\qquad \\mathcal I_{\\mathrm{pool}} = LL^{\\mathsf T}.$$

Returns
-------
`float`, non-negative and finite: the smallest common weight scale. It is $0$ exactly when the target Jacobian is identically zero, and it scales as $c^{-2}$ when every target uncertainty is multiplied by $c$, and as $s^{-1}$ when `I_pool` is multiplied by $s$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def minimal_information_scale(I_pool, qoi_jacobian, delta):
    """I_pool: symmetric positive definite (P, P) pooled data information matrix.
    qoi_jacobian: (Q, P) Jacobian of the Q quantities of interest with respect to the
    same P parameters.
    delta: length-Q sequence of target uncertainties, in the units of the QoIs.
    Return the smallest common weight scale t >= 0 satisfying the matching condition,
    a float. Raise ValueError if I_pool is singular."""
    # Implement per the principle above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_minimal_information_scale(I_pool, qoi_jacobian, delta):
    A = np.asarray(I_pool, float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("I_pool must be a square matrix")
    if not np.allclose(A, A.T, rtol=0.0, atol=1e-8 * max(1.0, np.abs(A).max())):
        raise ValueError("I_pool must be symmetric")
    H = np.atleast_2d(np.asarray(qoi_jacobian, float))
    if H.shape[1] != A.shape[0]:
        raise ValueError("qoi_jacobian and I_pool disagree in the parameter dimension")
    d = np.atleast_1d(np.asarray(delta, float)).ravel()
    if d.size != H.shape[0]:
        raise ValueError("delta must supply one target uncertainty per QoI")
    if np.any(d <= 0.0) or not np.all(np.isfinite(d)):
        raise ValueError("every target uncertainty must be finite and positive")
    J = (H / d[:, None]).T @ (H / d[:, None])
    ev = np.linalg.eigvalsh(A)
    if ev.min() <= 1e-12 * max(ev.max(), 1.0):
        raise ValueError("I_pool is singular: the candidate data cannot match any "
                         "target with a finite common weight")
    Li = np.linalg.inv(np.linalg.cholesky(A))
    return float(max(np.linalg.eigvalsh(Li @ J @ Li.T).max(), 0.0))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    TA = ("TA = np.array([2.860082, 3.086341, 33.787168, 33.787168, 8.489528, 4.527748,\n"
          "              0.611679, 1.032101, 0.176977, 0.353954,\n"
          "              -5.103845, -0.405524, 1.112997, -3.585325,\n"
          "              -5.14, 0.0, 1.640098, 0.221375, 0.848843, -5.141526])\n")
    CFG = ("def _cfg(m, n_cell=2):\n"
           "    a = 3.05 + 0.06 * (m % 8); L = a * n_cell\n"
           "    amp = 0.08 + 0.04 * (m % 4)\n"
           "    c = np.arange(n_cell, dtype=float)\n"
           "    I, J, K = np.meshgrid(c, c, c, indexing='ij')\n"
           "    b = np.stack([I, J, K], -1).reshape(-1, 3)\n"
           "    R = np.concatenate([b, b + 0.5]) * a\n"
           "    i = np.arange(R.shape[0], dtype=float)\n"
           "    u = amp * np.stack([np.sin(1.7*i + 0.9*m + 0.3),\n"
           "                        np.sin(2.3*i + 1.4*m + 1.1),\n"
           "                        np.sin(3.1*i + 2.2*m + 1.9)], axis=1)\n"
           "    return (R + u) % L, np.full(3, L)\n")
    return [
        # normal: a hand-checkable 3x3 case. J = diag(1, 1/4, 1/4) + off-diagonal from
        # the second row; whitening by diag(1, sqrt2, 2) gives t* = 1 exactly.
        {"setup": "import numpy as np\nI = np.diag([1.0, 2.0, 4.0])\n"
                  "H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])\nd = np.array([1.0, 2.0])\n",
         "call": "minimal_information_scale(I, H, d)",
         "gold_call": "_oracle_minimal_information_scale(I, H, d)"},
        # normal: the real 7x7 instance on a four-configuration pool, with the five
        # target uncertainties of the source paper.
        {"setup": "import numpy as np\n" + TA + CFG +
                  "C = [_cfg(m) for m in (0, 4, 7, 10)]\n"
                  "IE, IF = _oracle_data_fisher_matrices([c[0] for c in C], "
                  "[c[1] for c in C], TA)\nI = 0.01 * IE + IF\n"
                  "H = _oracle_log_parameter_jacobian("
                  "lambda q: _oracle_bcc_indicator_properties(q), TA)\n"
                  "d = np.array([0.0075, 1.0869, 9.6143, 5.9992, 6.4602])\n",
         "call": "minimal_information_scale(I, H, d)",
         "gold_call": "_oracle_minimal_information_scale(I, H, d)"},
        # boundary: a single QoI against a non-diagonal, strongly anisotropic I_pool, so
        # the whitening cannot be replaced by a division by the diagonal.
        {"setup": "import numpy as np\nM = np.array([[3.0, 1.0, 0.5], [1.0, 2.0, -0.4], "
                  "[0.5, -0.4, 1.5]])\nI = M @ M.T\nH = np.array([[2.0, -1.0, 0.7]])\n"
                  "d = np.array([0.3])\n",
         "call": "minimal_information_scale(I, H, d)",
         "gold_call": "_oracle_minimal_information_scale(I, H, d)"},
        # boundary: the same case with delta doubled, which must divide t* by exactly
        # four; an implementation using 1/delta instead of 1/delta^2 halves it instead.
        {"setup": "import numpy as np\nM = np.array([[3.0, 1.0, 0.5], [1.0, 2.0, -0.4], "
                  "[0.5, -0.4, 1.5]])\nI = M @ M.T\nH = np.array([[2.0, -1.0, 0.7]])\n"
                  "d = np.array([0.6])\n",
         "call": "minimal_information_scale(I, H, d)",
         "gold_call": "_oracle_minimal_information_scale(I, H, d)"},
        # edge: a RANK-DEFICIENT target in a 4-dimensional parameter space - two QoIs
        # that touch only two of the four parameters. The target FIM is singular and
        # that is harmless; only I_pool must be invertible.
        {"setup": "import numpy as np\nI = np.diag([5.0, 5.0, 5.0, 5.0])\n"
                  "H = np.zeros((2, 4))\nH[0, 0] = 1.0\nH[1, 3] = 2.0\nd = np.array([1.0, 1.0])\n",
         "call": "minimal_information_scale(I, H, d)",
         "gold_call": "_oracle_minimal_information_scale(I, H, d)"},
        # edge: a rotated I_pool with a soft direction of eigenvalue 0.1 and two QoIs
        # with unequal targets, so the binding direction is a nontrivial combination and
        # no scalar surrogate for the matrix inequality reproduces t*.
        {"setup": "import numpy as np\nrt = np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], "
                  "[0.0, 0.0, 1.0]])\nI = rt @ np.diag([10.0, 1.0, 0.1]) @ rt.T\n"
                  "H = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]])\nd = np.array([0.5, 2.0])\n",
         "call": "minimal_information_scale(I, H, d)",
         "gold_call": "_oracle_minimal_information_scale(I, H, d)"},
    ]
