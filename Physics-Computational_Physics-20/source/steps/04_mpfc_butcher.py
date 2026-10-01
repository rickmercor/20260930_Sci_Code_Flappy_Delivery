"""
Build the pair of Butcher tableaux this scheme runs on from the source's ****general coefficient family****, and certify it. The source displays one explicit pair of tableaux and then, separately, the family of pairs its stability and accuracy proofs actually permit, parameterised by five free real constants - in the source's own notation $\\tilde a_{11},\\tilde a_{32},\\tilde a_{33},a_{31},a_{43}$, of which three must be nonzero. Given those five constants, return the concatenated real array of shape `(49,)` holding, in this order: the $4\\times4$ implicit tableau $\\tilde A$ flattened row by row (16 entries), the $4\\times4$ explicit tableau $A$ flattened row by row (16 entries), the weight vector $b$ (4 entries), the residuals of the source's ****nine**** order conditions for third-order temporal accuracy, each written as left-hand side minus right-hand side and taken in the source's own order (9 entries), and the four eigenvalues of the energy-stability matrix $P=D\\tilde A+\\tilde A^{T}D-bb^{T}$ with $D=\\operatorname{diag}(b)$, sorted ascending (4 entries). Raise `ValueError` if any of the three constants the family requires to be nonzero is zero.

A Runge-Kutta method for a problem that has been split into a linear part and a nonlinear part can carry a different tableau for each part. Here the tableau of the linear part is lower triangular **including** its diagonal, so each stage is obtained from a constant-coefficient implicit solve, which is what makes the method stable at large step sizes; the tableau of the nonlinear part is strictly lower triangular, so the nonlinear terms are evaluated on stage values already computed and no nonlinear iteration is ever needed. The two tableaux share one weight vector. Third-order accuracy is not a property of one tableau but a set of polynomial identities coupling the two - nine of them here, more than the four a single classical third-order tableau needs, because the elementary differentials of a split problem are more numerous. Energy stability is a separate and much simpler condition: one symmetric matrix built from the implicit tableau and the weights must be positive semi-definite and the weights must be non-negative. The two conditions are independent, which is exactly why a whole family of coefficient sets survives both, and why picking a different member changes the numerical trajectory while changing neither the order nor the stability.

**--- Formulas ---**

Both tableaux are $4\\times4$ and every entry of both, and of $b$, is fixed by the five constants as follows; any entry not named below is zero. The implicit tableau $\\tilde A$ is lower triangular with a nonvanishing diagonal: $\\tilde a_{11}$ free, $\\tilde a_{22}=\\tfrac23$, third row $(\\tilde a_{31},\\tilde a_{32},\\tilde a_{33},0)$ with $\\tilde a_{32}$ and $\\tilde a_{33}$ free and the first entry ****determined****, $\\tilde a_{31}=\\dfrac{2-6\\tilde a_{11}}{3a_{43}}+\\tilde a_{11}-\\tilde a_{32}-\\tilde a_{33}$, and fourth row $(0,-1,0,1)$. The explicit tableau $A$ is strictly lower triangular: $a_{21}=\\tfrac23$, $a_{31}$ free, the next entry ****determined****, $a_{32}=\\dfrac{2}{3a_{43}}-a_{31}$, and $a_{41}=-a_{43}$ with $a_{43}$ free. The weights are $b=(0,\\tfrac34,0,\\tfrac14)$ and do not depend on the five constants at all. Write $\\tilde s_l=\\sum_{m=1}^{l}\\tilde a_{lm}$, $s_l=\\sum_{m=1}^{l-1}a_{lm}$, $\\tilde c_l=\\sum_{m=l}^{4}b_m\\tilde a_{ml}$ and $c_l=\\sum_{m=l+1}^{4}b_m a_{ml}$. The nine order conditions for third-order temporal accuracy are, in this order, with every sum taken over $l=1,\\dots,4$: (1) $\\sum_l b_l=1$; (2) $\\sum_l b_l\\tilde s_l=\\tfrac12$; (3) $\\sum_l \\tilde c_l\\tilde s_l=\\tfrac16$; (4) $\\sum_l b_l s_l=\\tfrac12$; (5) $\\sum_l \\tilde c_l s_l=\\tfrac16$; (6) $\\sum_l b_l\\tilde s_l s_l=\\tfrac13$; (7) $\\sum_l b_l s_l^{2}=\\tfrac13$; (8) $\\sum_l c_l\\tilde s_l=\\tfrac16$; (9) $\\sum_l c_l s_l=\\tfrac16$. Each residual is that condition's left-hand side minus its right-hand side. Finally $P=D\\tilde A+\\tilde A^{T}D-bb^{T}$ with $D=\\operatorname{diag}(b)$, and its eigenvalues are returned sorted ascending.

Returns
-------
`np.ndarray` of shape `(49,)`, real and finite. For every admissible choice of the five constants the nine residuals are zero to round-off and the four eigenvalues of $P$ are $0,0,0$ and $7/8$ - the family was constructed so that neither depends on the constants. Entries 32 to 35 are $b=(0,\\tfrac34,0,\\tfrac14)$ always.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mpfc_butcher(a11t: "float | None" = None, a32t: "float | None" = None,
        a33t: "float | None" = None, a31: "float | None" = None,
        a43: "float | None" = None) -> "np.ndarray":
    """a11t, a32t, a33t, a31, a43: the five free real constants of the
       source's general coefficient family; None means this task's
       prescribed value for that constant.
    Return the real array of shape (49,) holding, in order: Atilde
    flattened (16), A flattened (16), b (4), the nine order-condition
    residuals (9) and the four eigenvalues of P sorted ascending (4).
    Raise ValueError if any constant is not finite, or if any of
    a11t, a33t and a43 - the three the family requires to be nonzero -
    is zero.  a32t and a31 may be zero."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_mpfc_butcher(a11t: "float | None" = None, a32t: "float | None" = None,
        a33t: "float | None" = None, a31: "float | None" = None,
        a43: "float | None" = None) -> "np.ndarray":
    given = dict(a11t=a11t, a32t=a32t, a33t=a33t, a31=a31, a43=a43)
    c = _check_coef({k: v for k, v in given.items() if v is not None})
    At = np.zeros((4, 4))
    A = np.zeros((4, 4))
    b = np.array([0.0, 0.75, 0.0, 0.25])
    At[0, 0] = c["a11t"]
    At[1, 1] = 2.0 / 3.0
    At[2, 0] = (2.0 - 6.0 * c["a11t"]) / (3.0 * c["a43"]) + c["a11t"] - c["a32t"] - c["a33t"]
    At[2, 1] = c["a32t"]
    At[2, 2] = c["a33t"]
    At[3, 1] = -1.0
    At[3, 3] = 1.0
    A[1, 0] = 2.0 / 3.0
    A[2, 0] = c["a31"]
    A[2, 1] = 2.0 / (3.0 * c["a43"]) - c["a31"]
    A[3, 0] = -c["a43"]
    A[3, 2] = c["a43"]
    st = [sum(At[l, m] for m in range(l + 1)) for l in range(4)]
    s = [sum(A[l, m] for m in range(l)) for l in range(4)]
    ct = [sum(b[m] * At[m, l] for m in range(l, 4)) for l in range(4)]
    ca = [sum(b[m] * A[m, l] for m in range(l + 1, 4)) for l in range(4)]
    res = np.array([
        sum(b) - 1.0,
        sum(b[l] * st[l] for l in range(4)) - 0.5,
        sum(ct[l] * st[l] for l in range(4)) - 1.0 / 6.0,
        sum(b[l] * s[l] for l in range(1, 4)) - 0.5,
        sum(ct[l] * s[l] for l in range(1, 4)) - 1.0 / 6.0,
        sum(b[l] * st[l] * s[l] for l in range(1, 4)) - 1.0 / 3.0,
        sum(b[l] * s[l] ** 2 for l in range(1, 4)) - 1.0 / 3.0,
        sum(ca[l] * st[l] for l in range(0, 3)) - 1.0 / 6.0,
        sum(ca[l] * s[l] for l in range(1, 3)) - 1.0 / 6.0,
    ])
    P = np.diag(b) @ At + At.T @ np.diag(b) - np.outer(b, b)
    eig = np.sort(np.linalg.eigvalsh(0.5 * (P + P.T)))
    return np.concatenate([At.ravel(), A.ravel(), b, res, eig])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: the five free constants this task prescribes.
        {"setup": "import numpy as np\n",
         "call": "mpfc_butcher(0.5, -0.25, 1.5, 1.0 / 3.0, 2.0 / 3.0)",
         "gold_call": "_oracle_mpfc_butcher(0.5, -0.25, 1.5, 1.0 / 3.0, 2.0 / 3.0)"},
        # normal: the constants that reproduce the displayed tableau of the
        # source, which is the family member every transcription lands on.
        {"setup": "import numpy as np\n",
         "call": "mpfc_butcher(1.0, 1.0, 1.0, 1.0, 1.0)",
         "gold_call": "_oracle_mpfc_butcher(1.0, 1.0, 1.0, 1.0, 1.0)"},
        # boundary: a negative last constant, which flips the sign of two
        # entries of the explicit tableau and of one entry of the implicit one.
        {"setup": "import numpy as np\n",
         "call": "mpfc_butcher(-0.75, 0.4, -2.0, -1.5, -0.5)",
         "gold_call": "_oracle_mpfc_butcher(-0.75, 0.4, -2.0, -1.5, -0.5)"},
        # boundary: the defaults, taken when no constant is supplied.
        {"setup": "import numpy as np\n",
         "call": "mpfc_butcher()",
         "gold_call": "_oracle_mpfc_butcher()"},
        # edge: a nearly explicit first stage, where the first diagonal entry
        # is small but nonzero and the first column of the implicit tableau is
        # correspondingly large.
        {"setup": "import numpy as np\n",
         "call": "mpfc_butcher(1.0e-3, 0.0, 5.0, 0.0, 3.0)",
         "gold_call": "_oracle_mpfc_butcher(1.0e-3, 0.0, 5.0, 0.0, 3.0)"},
        # edge: a vanishing constant that the family forbids, which must raise
        # ValueError rather than divide by zero.
        {"setup": "import numpy as np\n"
                  "def _guard(f, *a):\n"
                  "    try:\n"
                  "        f(*a)\n"
                  "    except ValueError:\n"
                  "        return 1.0\n"
                  "    return 0.0\n",
         "call": "_guard(mpfc_butcher, 1.0, 1.0, 1.0, 1.0, 0.0)",
         "gold_call": "_guard(_oracle_mpfc_butcher, 1.0, 1.0, 1.0, 1.0, 0.0)"},
        # edge: negative constants, where the sign of the determined entry
        # (2 - 6*a11t)/(3*a43) + a11t - a32t - a33t and of 2/(3*a43) - a31 both
        # flip, and a transcription that drops a sign still passes at all-ones.
        {"setup": "import numpy as np\n",
         "call": "mpfc_butcher(-1.5, 2.25, -0.5, -2.0, -0.75)",
         "gold_call": "_oracle_mpfc_butcher(-1.5, 2.25, -0.5, -2.0, -0.75)"},
    ]
