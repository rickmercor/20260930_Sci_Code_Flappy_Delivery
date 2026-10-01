"""
Advance the fully discrete BDF$q$ convex-splitting scheme by one time step and return the new field $\\phi^n$. The input `hist` is the array $[\\phi^{n-q},\\phi^{n-q+1},\\dots,\\phi^{n-1}]$ of shape `(q, Mx, My)`, oldest first, and $q$ is read from its leading axis. The scheme is $\\mathcal{B}_q\\phi^n=\\Delta_h\\mu^n$ with $\\mu^n=N(\\phi^n)-\\varepsilon\\hat\\phi_q^{\\,n}+(1+\\Delta_h)^2\\phi^n-S\\tau^q\\Delta_h\\mathcal{B}_q\\phi^n$, where $N$ is the quartic divergence of the previous step. Eliminate $\\mu^n$, move every term linear in $\\phi^n$ to the left, and invert the resulting constant-coefficient operator in Fourier space; the remaining nonlinearity is resolved by the Picard iteration this rearrangement defines, started from $\\phi^{(0)}=\\hat\\phi_q^{\\,n}$ and stopped when $\\max_{ij}|\\phi^{(s+1)}-\\phi^{(s)}|<10^{-13}$, with a hard cap of $200$ iterations.

Convex splitting treats the convex part of the energy implicitly and the concave part explicitly. Here the convex part supplies $N(\\phi^n)+(1+\\Delta_h)^2\\phi^n$, both at the new level, and the concave part $-\\tfrac{\\varepsilon}{2}\\|\\phi\\|^2$ supplies $-\\varepsilon\\hat\\phi_q^{\\,n}$, extrapolated. At first order this alone dissipates the energy for every step size, because a convex functional lies above its tangent plane and a concave one below; above first order the extrapolation can inject energy, and the stabilization term is what dominates that injection. Building the stabilization out of the same operator $\\mathcal{B}_q$ that discretises the time derivative, scaled by $\\tau^q$, keeps it consistent to the order of the scheme and gives it the same quadratic decomposition the time-derivative term already has. Two structural points govern the implementation. The BDF$q$ operator is affine in the unknown, so the single contribution it makes at the new level separates from everything the history determines; its leading kernel coefficient $\\beta_0^{(q)}$ is positive for $q=3,4,5$. And once $\\mu^n$ has been eliminated, every operator multiplying $\\phi^n$ is a constant-coefficient polynomial in $\\Delta_h$, hence diagonal in Fourier space, so the linear solve is one elementwise division; only the quartic term, whose coefficient depends on $\\nabla_h\\phi^n$, is not, and it is the sole reason an iteration is needed. How many powers of $\\Delta_h$ the stabilization term carries once that elimination has been done, and which power of $\\tau$ is left standing next to $S$, follow from the elimination itself and are not stated here; either one taken on faith rather than derived changes the answer. The resulting symbol is strictly positive at every wavenumber - at the zero mode, and at the wavenumber the model selects, where one of its terms vanishes - so no regularisation is ever required; that is something to verify on the symbol you derive, not to assume. Mass is conserved exactly by construction, through the zero mode of the solve, and identifying the mechanism is part of the task.

$\\mathcal{B}_q$ and $\\hat\\phi_q^{\\,n}$ are the convolution and the extrapolation of the earlier sub-problem. Both are written over the FIRST DIFFERENCES $\\delta_\\tau\\phi^{j}=\\phi^{j}-\\phi^{j-1}$, so both have to be expanded over the levels of `hist` before anything can be rearranged.



Every transform is the 2-D DFT in the conventions of the first sub-problem, and the real part of each inverse transform is taken. Only the nonlinear term changes between sweeps, so the symbol of the linear operator and the terms that do not depend on the current iterate are formed once, outside the loop.

Returns
-------
`np.ndarray` of shape `hist.shape[1:]`, real and finite. Its discrete mean equals that of every layer of a mass-conservative history, to round-off. If the history is a single constant $c$ repeated $q$ times, the step returns that same constant field exactly, because $g^n=0$, $\\hat\\phi_q^{\\,n}=c$, $N\\equiv0$ and the only surviving Fourier mode is $\\lambda=0$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sqpfc_convex_split_step(hist: "np.ndarray", cell: "float | Sequence[float]",
                            eps: float, S: float, tau: float,
                            tol: float = 1e-13,
                            max_iter: int = 200) -> "np.ndarray":
    """hist: array [phi^{n-q}, ..., phi^{n-1}] of shape (q, Mx, My), oldest
    first; q is read from the leading axis and must be 3, 4 or 5.
    cell: domain edge lengths, a scalar or a length-2 sequence (Lx, Ly).
    eps: the parameter epsilon.  S: stabilization parameter.  tau: time step.
    tol: increment at which the Picard iteration is stopped.
    max_iter: hard cap on the number of Picard sweeps.
    Return the real array phi^n of shape (Mx, My), the fixed point of the
    Picard iteration, stopped when the max-norm increment falls below tol.
    Raise ValueError if hist does not have shape (q, Mx, My) with q in
    {3, 4, 5}, if tau or tol is not positive, if S is negative, or if
    max_iter is below 1.  Raise RuntimeError if the increment has not
    fallen below tol within max_iter sweeps: the last iterate is not the
    step and must not be returned in its place."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sqpfc_convex_split_step(hist: "np.ndarray", cell: "float | Sequence[float]",
                                    eps: float, S: float, tau: float,
                                    tol: float = 1e-13,
                                    max_iter: int = 200) -> "np.ndarray":
    """One BDFq convex-splitting step: kernels from step 2, symbol from step 1,
    nonlinear term from step 5."""
    hist = np.asarray(hist, float)
    if hist.ndim != 3:
        raise ValueError("hist must have shape (q, Mx, My)")
    if hist.shape[0] not in (3, 4, 5):
        raise ValueError("the leading axis of hist must be q = 3, 4 or 5")
    if not float(tau) > 0.0:
        raise ValueError("tau must be positive")
    if float(S) < 0.0:
        raise ValueError("S must be non-negative")
    if not float(tol) > 0.0 or int(max_iter) < 1:
        raise ValueError("tol must be positive and max_iter at least 1")
    q = hist.shape[0]
    eps = float(eps)
    S = float(S)
    tau = float(tau)
    _k = _oracle_sqpfc_bdf_kernels(q)
    beta, alph = _k[0], _k[1]
    lam = _oracle_sqpfc_laplacian_symbol(hist.shape[1:], cell)

    ex = np.zeros(q + 1)
    ex[0] += 1.0
    for j in range(q):
        ex[j] -= alph[j]
        ex[j + 1] += alph[j]
    phi_hat = sum(ex[j] * hist[q - j] for j in range(1, q + 1))

    dc = np.zeros(q + 1)
    for j in range(q):
        dc[j] += beta[j]
        dc[j + 1] -= beta[j]
    g = sum(dc[j] * hist[q - j] for j in range(1, q + 1)) / tau

    P = 1.0 + S * tau ** q * lam ** 2
    A = (beta[0] / tau) * P + lam * (1.0 - lam) ** 2
    rhs = eps * lam * np.fft.fft2(phi_hat) - P * np.fft.fft2(g)

    tol = float(tol)
    phi = phi_hat.copy()
    for _ in range(int(max_iter)):
        nl = _oracle_sqpfc_quartic_divergence(phi, cell)
        new = np.fft.ifft2((-lam * np.fft.fft2(nl) + rhs) / A).real
        d = float(np.max(np.abs(new - phi)))
        phi = new
        if d < tol:
            return phi
    raise RuntimeError("the Picard iteration did not meet tol within max_iter "
                       "sweeps; the last iterate is not the step")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    P1 = ("Lv = (8.0 * np.pi, 8.0 * np.pi)\n"
          "Mv = (32, 32)\n"
          "X, Y = np.meshgrid(np.arange(Mv[0]) * Lv[0] / Mv[0],\n"
          "                   np.arange(Mv[1]) * Lv[1] / Mv[1], indexing='ij')\n"
          "u = 0.07 + 0.60 * (np.cos(X) * np.cos(Y)\n"
          "                   + 0.40 * np.sin(2.0 * X) * np.cos(Y)\n"
          "                   + 0.30 * np.cos(X - 2.0 * Y))\n"
          "H3 = np.stack([u, u, u])\n")
    P4 = ("Lv = (32.0, 20.0)\n"
          "Mv = (48, 24)\n"
          "X, Y = np.meshgrid(np.arange(Mv[0]) * Lv[0] / Mv[0],\n"
          "                   np.arange(Mv[1]) * Lv[1] / Mv[1], indexing='ij')\n"
          "r = np.sqrt((X - 0.5 * Lv[0])**2 + (Y - 0.5 * Lv[1])**2)\n"
          "u = 2.5 * (1.0 - np.tanh(0.5 * (r - 2.0)))\n"
          "H5 = np.stack([u] * 5)\n")
    P3 = ("Lv = (6.0 * np.pi, 10.0 * np.pi)\n"
          "Mv = (24, 40)\n"
          "X, Y = np.meshgrid(np.arange(Mv[0]) * Lv[0] / Mv[0],\n"
          "                   np.arange(Mv[1]) * Lv[1] / Mv[1], indexing='ij')\n"
          "u = 0.07 + 0.60 * (np.cos(X) * np.cos(Y)\n"
          "                   + 0.40 * np.sin(2.0 * X) * np.cos(Y)\n"
          "                   + 0.30 * np.cos(X - 2.0 * Y))\n"
          "H4 = np.stack([u] * 4)\n"
          "H4b = np.stack([u, 1.01 * u, 0.99 * u, u])\n")
    return [
        # q = 3, square box, constant start-up
        {"setup": P1,
         "call": "sqpfc_convex_split_step(H3.copy(), Lv, 0.25, 5.0, 0.05)",
         "gold_call": "_oracle_sqpfc_convex_split_step(H3.copy(), Lv, 0.25, 5.0, 0.05)"},
        # q = 5, rectangular box with hx != hy, nucleus field
        {"setup": P4,
         "call": "sqpfc_convex_split_step(H5.copy(), Lv, 0.30, 1.0, 0.08)",
         "gold_call": "_oracle_sqpfc_convex_split_step(H5.copy(), Lv, 0.30, 1.0, 0.08)"},
        # q = 4, rectangular box, non-constant history
        {"setup": P3,
         "call": "sqpfc_convex_split_step(H4b.copy(), Lv, 0.40, 10.0, 0.04)",
         "gold_call": "_oracle_sqpfc_convex_split_step(H4b.copy(), Lv, 0.40, 10.0, 0.04)"},
        # S = 0 removes the stabilization entirely
        {"setup": P3,
         "call": "sqpfc_convex_split_step(H4.copy(), Lv, 0.40, 0.0, 0.04)",
         "gold_call": "_oracle_sqpfc_convex_split_step(H4.copy(), Lv, 0.40, 0.0, 0.04)"},
        # a constant history is a fixed point of the step
        {"setup": ("Lv = (7.0, 11.0)\n"
                   "c = np.full((16, 20), 0.13)\n"
                   "Hc = np.stack([c, c, c])\n"
                   "def pin(w, ref):\n"
                   "    if not np.allclose(w, ref, rtol=0.0, atol=1e-10):\n"
                   "        return np.full(np.shape(w), np.nan)\n"
                   "    return w\n"
                   "REF = np.full((16, 20), 0.13)\n"),
         "call": ("pin(sqpfc_convex_split_step("
                  "Hc.copy(), Lv, 0.35, 4.0, 0.07), REF)"),
         "gold_call": ("pin(_oracle_sqpfc_convex_split_step("
                       "Hc.copy(), Lv, 0.35, 4.0, 0.07), REF)")},
        # mass is conserved exactly
        {"setup": (P1 + "a = (Lv[0] / Mv[0]) * (Lv[1] / Mv[1])\n"
                   "REF = a * float(np.sum(u))\n"
                   "def mass(w):\n"
                   "    m = a * float(np.sum(w))\n"
                   "    if abs(m - REF) > 1e-9 * max(1.0, abs(REF)):\n"
                   "        return float('nan')\n"
                   "    return m\n"),
         "call": ("mass(sqpfc_convex_split_step("
                  "H3.copy(), Lv, 0.25, 5.0, 0.05))"),
         "gold_call": ("mass(_oracle_sqpfc_convex_split_step("
                       "H3.copy(), Lv, 0.25, 5.0, 0.05))")},
        # contract: BDF2 is outside the scheme, so a two-level history must
        # raise rather than be stepped with a kernel that does not exist here.
        {"setup": ("def trap(f):\n"
                  "    try:\n"
                  "        return float(np.asarray(f()).ravel()[0])\n"
                  "    except ValueError:\n"
                  "        return -12345.0\n"),
         "call": "trap(lambda: sqpfc_convex_split_step(np.zeros((2, 8, 8)), 4.0, 0.25, 1.0, 0.05))",
         "gold_call": ("trap(lambda: _oracle_sqpfc_convex_split_step("
                       "np.zeros((2, 8, 8)), 4.0, 0.25, 1.0, 0.05))")},
        # contract: one sweep cannot reach 1e-13 from the extrapolated state, so
        # the cap is exhausted and the call must raise RuntimeError. The trap
        # catches RuntimeError ONLY: returning the unconverged last iterate,
        # which is what an implementation that breaks out of the loop would do,
        # fails the case.
        {"setup": (P1 + "def rtrap(f):\n"
                  "    try:\n"
                  "        return float(np.asarray(f()).ravel()[0])\n"
                  "    except RuntimeError:\n"
                  "        return -12345.0\n"),
         "call": "rtrap(lambda: sqpfc_convex_split_step(H3.copy(), Lv, 0.25, 5.0, 0.05, 1e-13, 1))",
         "gold_call": ("rtrap(lambda: _oracle_sqpfc_convex_split_step("
                       "H3.copy(), Lv, 0.25, 5.0, 0.05, 1e-13, 1))")},
    ]
