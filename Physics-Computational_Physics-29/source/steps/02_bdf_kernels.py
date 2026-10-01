"""
Return the two multistep kernels of order $q\\in\\{3,4,5\\}$ that the time discretisation needs, stacked as a `(2, q)` array. Row $0$ is the BDF$q$ **convolution kernel** $\\beta^{(q)}$, the weights that act on FIRST DIFFERENCES in $\\mathcal{B}_qv^n=\\tfrac1\\tau\\sum_{j\\ge0}\\beta_j^{(q)}\\delta_\\tau v^{n-j}$ where $\\delta_\\tau v^n=v^n-v^{n-1}$. Row $1$ is the extrapolation kernel $\\alpha^{(q)}$ appearing in $\\hat v_q^{\\,n}=v^n-\\delta_\\tau^qv^n=v^n-\\sum_{j\\ge0}\\alpha_j^{(q)}\\delta_\\tau v^{n-j}$, where $\\delta_\\tau^m$ denotes the $m$-fold first difference. Both kernels vanish for $j\\ge q$, so only the first $q$ entries are returned. **Row $0$ is not the direct BDF$q$ coefficient vector** acting on $v^n,\\dots,v^{n-q}$: the two vectors share only their leading entry, and the conversion between them belongs to the step that uses them, not here.

A $q$-step backward differentiation formula can be written in two equivalent ways: directly, as a weighted combination $\\tfrac1\\tau\\sum_{j=0}^qc_jv^{n-j}$ of the solution levels, or as a convolution $\\tfrac1\\tau\\sum_{j=0}^{q-1}\\beta_j\\delta_\\tau v^{n-j}$ over first differences. The second form is the one the energy analysis needs, because a quadratic decomposition of the kind $v^n\\sum_j\\beta_jv^{n-j}=(\\vec v^{\\,n})^T\\mathbf G\\vec v^{\\,n}-(\\vec v^{\\,n-1})^T\\mathbf G\\vec v^{\\,n-1}+(\\vec v^{\\,n})^T\\mathbf R\\vec v^{\\,n}+\\tfrac{\\kappa}{2}|v^n|^2$ - which is what makes a multistep method telescope and therefore dissipate - exists for the difference kernel with a positive semidefinite $\\mathbf G$ and a positive $\\kappa$, and only up to $q=5$. The two forms are related by a summation identity that is deliberately not written out here: deriving it, and with it the direct coefficients $c_j$ implied by a kernel $\\beta$, belongs to the step that consumes these kernels. That a correct derivation reproduces the textbook BDF3, BDF4 and BDF5 coefficient vectors exactly is the check that the kernel has been read correctly. The kernel $\\beta^{(q)}$ itself is not tabulated here either. It is pinned by one requirement - that $\\tfrac1\\tau\\sum_{j=0}^{q-1}\\beta_j\\delta_\\tau v^{n-j}$ be the BDF$q$ approximation of $\\partial_tv(t_n)$ - and reading that requirement off level by level relates $\\beta$ to the direct coefficients in a way that determines each from the other. The extrapolation is built from the same difference calculus, through $\\delta_\\tau^qv^n=\\delta_\\tau^{q-1}(\\delta_\\tau v^n)$; expanding the $(q-1)$-fold first difference over the levels gives $\\alpha^{(q)}$, and that expansion is left to be done rather than quoted. Subtracting the $q$-fold difference from $v^n$ annihilates the $v^n$ term exactly and leaves the unique polynomial extrapolation of degree $q-1$ through $v^{n-1},\\dots,v^{n-q}$. That the coefficient of $v^n$ comes out identically zero is the check on this kernel, just as reproducing the textbook BDF3, BDF4 and BDF5 coefficient vectors is the check on the other. The stability analysis of this scheme runs on a third kernel, the discrete orthogonal convolution (DOC) kernel $\\gamma^{(q)}$, which is the convolution inverse of $\\beta^{(q)}$: it is defined by $\\sum_{i=0}^{m}\\gamma_i^{(q)}\\beta_{m-i}^{(q)}=\\delta_{m0}$ for every $m\\ge0$, with $\\beta_j^{(q)}=0$ for $j\\ge q$. Reading that identity at $m=0$ and then at successive $m$ determines every entry in turn from the ones before it, so the kernel is unique once $\\beta^{(q)}$ is known. It is not a truncation, a reversal or a reciprocal of $\\beta^{(q)}$, and none of those three reproduces it. The DOC kernel decays geometrically, and the published bound $|\\gamma_j^{(q)}|\\le\\tfrac{r_q}{4}\\big(\\tfrac q7\\big)^j$ with $r_3=10/3$, $r_4=6$ and $r_5=96/5$ is the check on it; it is a bound, not the value, so it confirms a derivation and cannot replace one.

$$\\tfrac1\\tau\\sum_{j=0}^{q-1}\\beta_j^{(q)}\\delta_\\tau v^{n-j}\\;=\\;\\partial_tv(t_n)+O(\\tau^q),\\qquad \\hat v_q^{\\,n}=v^n-\\delta_\\tau^qv^n=v^n-\\sum_{j=0}^{q-1}\\alpha_j^{(q)}\\delta_\\tau v^{n-j}.$$



$$\\sum_{i=0}^{m}\\gamma_i^{(q)}\\beta_{m-i}^{(q)}=\\delta_{m0}\\quad(m\\ge0),\\qquad \\beta_j^{(q)}=0\\ \\text{for } j\\ge q.$$



These three identities determine all three rows for $q=3,4,5$, and no numerical values for any row are given in this task. Return `np.stack([beta, alpha, gamma])`, of shape `(3, q)`, in that order: row $0$ the convolution kernel, row $1$ the extrapolation coefficients, row $2$ the leading $q$ entries $\\gamma_0^{(q)},\\dots,\\gamma_{q-1}^{(q)}$ of the DOC kernel, all indexed by $j=0,\\dots,q-1$ in the level ordering above. Do not convert any row to direct coefficients here; that conversion belongs to the step that uses them.

Returns
-------
`np.ndarray` of shape `(3, q)`, real and finite. Row $0$ sums to exactly $1$ for every $q$, a consequence of the two consistency conditions $\\sum_ic_i=0$ and $\\sum_i ic_i=-1$ satisfied by the direct coefficients of the underlying BDF$q$ formula. Row $1$ sums to $0$ for $q\\ge2$ and is exactly integer-valued. Row $2$ begins at $\\gamma_0^{(q)}=1/\\beta_0^{(q)}$, which is strictly between $0$ and $1$ for $q=3,4,5$, and every entry obeys the published decay bound $|\\gamma_j^{(q)}|\\le\\tfrac{r_q}{4}(q/7)^j$. Row $2$ is not a rearrangement of row $0$: convolving the two rows and extending with zeros returns the discrete delta.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sqpfc_bdf_kernels(q: int) -> "np.ndarray":
    """q: BDF order, one of 3, 4, 5.
    Return the real array [beta^{(q)}, alpha^{(q)}, gamma^{(q)}] of shape
    (3, q): beta the BDFq convolution kernel acting on first differences,
    alpha the extrapolation kernel, and gamma the leading q entries of the
    discrete orthogonal convolution (DOC) kernel, the convolution inverse
    of beta.
    Raise ValueError if q is not one of 3, 4, 5."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sqpfc_bdf_kernels(q: int) -> "np.ndarray":
    """BDFq convolution kernel and the extrapolation kernel."""
    if q != int(q) or int(q) not in (3, 4, 5):
        raise ValueError("q must be 3, 4 or 5")
    q = int(q)
    beta = {3: [11.0 / 6.0, -7.0 / 6.0, 1.0 / 3.0],
            4: [25.0 / 12.0, -23.0 / 12.0, 13.0 / 12.0, -1.0 / 4.0],
            5: [137.0 / 60.0, -163.0 / 60.0, 137.0 / 60.0,
                -21.0 / 20.0, 1.0 / 5.0]}[q]
    a = [1.0]
    for j in range(1, q):
        a.append(-a[-1] * (q - j) / j)             # (-1)^j C(q-1, j)
    g = [1.0 / beta[0]]                            # DOC kernel: conv inverse
    for m in range(1, q):
        s = sum(g[i] * beta[m - i] for i in range(m) if m - i < q)
        g.append(-s / beta[0])
    return np.stack([np.array(beta, float), np.array(a, float),
                     np.array(g, float)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        {"setup": "", "call": "sqpfc_bdf_kernels(3)",
         "gold_call": "_oracle_sqpfc_bdf_kernels(3)"},
        {"setup": "", "call": "sqpfc_bdf_kernels(4)",
         "gold_call": "_oracle_sqpfc_bdf_kernels(4)"},
        {"setup": "", "call": "sqpfc_bdf_kernels(5)",
         "gold_call": "_oracle_sqpfc_bdf_kernels(5)"},
        # The direct BDF coefficients implied by row 0 must be the textbook
        # ones.  The reference is carried inside the transformation, which is
        # applied to both sides, so a kernel that misses it is poisoned to NaN
        # and the case fails rather than agreeing with itself.
        {"setup": ("def direct(k, ref):\n"
                   "    q = k.shape[1]\n"
                   "    c = np.zeros(q + 1)\n"
                   "    for j in range(q):\n"
                   "        c[j] += k[0, j]\n"
                   "        c[j + 1] -= k[0, j]\n"
                   "    if not np.allclose(c, ref, rtol=0.0, atol=1e-12):\n"
                   "        return np.full(c.shape, np.nan)\n"
                   "    return c\n"
                   "R3 = np.array([11/6, -3.0, 1.5, -1/3])\n"),
         "call": "direct(sqpfc_bdf_kernels(3), R3)",
         "gold_call": "direct(_oracle_sqpfc_bdf_kernels(3), R3)"},
        {"setup": ("def direct(k, ref):\n"
                   "    q = k.shape[1]\n"
                   "    c = np.zeros(q + 1)\n"
                   "    for j in range(q):\n"
                   "        c[j] += k[0, j]\n"
                   "        c[j + 1] -= k[0, j]\n"
                   "    if not np.allclose(c, ref, rtol=0.0, atol=1e-12):\n"
                   "        return np.full(c.shape, np.nan)\n"
                   "    return c\n"
                   "R5 = np.array([137/60, -5.0, 5.0, -10/3, 1.25, -0.2])\n"),
         "call": "direct(sqpfc_bdf_kernels(5), R5)",
         "gold_call": "direct(_oracle_sqpfc_bdf_kernels(5), R5)"},
        # the extrapolation must annihilate v^n exactly
        {"setup": ("def head(k, ref):\n"
                   "    q = k.shape[1]\n"
                   "    e = np.zeros(q + 1)\n"
                   "    e[0] += 1.0\n"
                   "    for j in range(q):\n"
                   "        e[j] -= k[1, j]\n"
                   "        e[j + 1] += k[1, j]\n"
                   "    if not np.allclose(e, ref, rtol=0.0, atol=1e-12):\n"
                   "        return np.full(e.shape, np.nan)\n"
                   "    return e\n"
                   "R4 = np.array([0.0, 4.0, -6.0, 4.0, -1.0])\n"),
         "call": "head(sqpfc_bdf_kernels(4), R4)",
         "gold_call": "head(_oracle_sqpfc_bdf_kernels(4), R4)"},
        # The DOC row must invert the convolution row exactly: convolving
        # rows 2 and 0 has to return the discrete delta.  A truncation, a
        # reversal or a reciprocal of row 0 all fail this.
        {"setup": ("def conv(k):\n"
                   "    q = k.shape[1]\n"
                   "    d = np.zeros(q)\n"
                   "    for m in range(q):\n"
                   "        d[m] = sum(k[2, i] * k[0, m - i]\n"
                   "                   for i in range(m + 1))\n"
                   "    ref = np.zeros(q)\n"
                   "    ref[0] = 1.0\n"
                   "    if not np.allclose(d, ref, rtol=0.0, atol=1e-12):\n"
                   "        return np.full(q, np.nan)\n"
                   "    return d\n"),
         "call": "conv(sqpfc_bdf_kernels(5))",
         "gold_call": "conv(_oracle_sqpfc_bdf_kernels(5))"},
        # the published geometric decay bound on the DOC kernel
        {"setup": ("def decay(k, r):\n"
                   "    q = k.shape[1]\n"
                   "    g = k[2]\n"
                   "    b = (r / 4.0) * (q / 7.0) ** np.arange(q)\n"
                   "    if np.any(np.abs(g) > b + 1e-12):\n"
                   "        return np.full(q, np.nan)\n"
                   "    return g\n"),
         "call": "decay(sqpfc_bdf_kernels(3), 10.0 / 3.0)",
         "gold_call": "decay(_oracle_sqpfc_bdf_kernels(3), 10.0 / 3.0)"},
        # contract: the decomposition data, and this task, exist only for
        # q = 3, 4, 5, so q = 2 must raise rather than return a BDF2 row.
        {"setup": ("def trap(f):\n"
                  "    try:\n"
                  "        return float(np.asarray(f()).ravel()[0])\n"
                  "    except ValueError:\n"
                  "        return -12345.0\n"),
         "call": "trap(lambda: sqpfc_bdf_kernels(2))",
         "gold_call": "trap(lambda: _oracle_sqpfc_bdf_kernels(2))"},
    ]
