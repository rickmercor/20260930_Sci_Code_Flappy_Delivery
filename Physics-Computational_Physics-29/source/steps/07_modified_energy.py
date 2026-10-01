"""
Evaluate the modified discrete energy $\\hat E^n$, the quantity whose decay the scheme's stability theorem establishes - for data band-limited strictly below Nyquist, a hypothesis this task's convention does not make automatic and which the background states in full. The input `hist` is $[\\phi^{n-q},\\dots,\\phi^{n}]$ of shape $(q+1, Mx, My)$, oldest first, so that the $q$ first differences $\\delta_\\tau\\vec\\phi^{\\,n}=(\\delta_\\tau\\phi^{n-q+1},\\dots,\\delta_\\tau\\phi^{n})$ can be formed; $q$ is read from the shape of the supplied matrix $\\mathbf G$. The quantity to evaluate is the modified discrete energy defined in the problem statement, with $\\mathbf G$ and $\\mathbf J$ supplied as arguments and $\\langle\\vec a,\\vec b\\rangle$ the sum of the componentwise discrete inner products over the $q$ slots; how each of its terms ends up scaling in $\\tau$ is part of what is graded. where $E^n$ is the original discrete energy of the LAST slot, $\\langle u,w\\rangle_{-1}=\\langle(-\\Delta_h)^{-1}u,w\\rangle$ with the zero Fourier mode of $(-\\Delta_h)^{-1}$ set to zero, and the middle term contracts the two components of each gradient as well as the $q$ slots. **Note the factor $\\varepsilon$ on the $\\mathbf J$ term.** Return a single float.

A multistep method does not telescope on its own, so the quantity it decreases is not the original energy but a modification of it. The device is a quadratic decomposition: one seeks a symmetric positive-semidefinite $q\\times q$ matrix $\\mathbf G$ and a constant $\\kappa>0$ such that $v^n\\sum_j\\beta_jv^{n-j}$ equals a difference of two identical quadratic forms evaluated one step apart, plus two sign-definite remainders. The difference telescopes, so adding the form $\\langle\\mathbf G\\delta_\\tau\\vec\\phi,\\delta_\\tau\\vec\\phi\\rangle$ to the energy converts the multistep estimate into a genuine one-step decay. The same construction applied to the extrapolation kernel yields a matrix $\\mathbf J$ and a constant $\\eta$, but with the opposite sign on the remainder, so the extrapolation consumes dissipation rather than supplying it - which is exactly the deficit the stabilization must cover. Three of the four terms therefore come in matched pairs: the time-derivative term is measured in the $H^{-1}$ inner product, because the flow is an $H^{-1}$ gradient flow and the test function is $(-\\Delta_h)^{-1}\\delta_\\tau\\phi^n$; the stabilization term inherits the same $\\mathbf G$ but is measured in the gradient, because its operator carries an extra $\\Delta_h$; and the extrapolation term carries $\\mathbf J$ and, crucially, a factor $\\varepsilon$, since the estimate it telescopes is the one for $-\\varepsilon\\langle\\hat\\phi_q^{\\,n},\\phi^n-\\phi^{n-1}\\rangle$ and $\\varepsilon$ multiplies every piece of it. The $H^{-1}$ inner product is defined only on mean-zero fields; mass conservation is what puts every first difference there, and the zero Fourier mode of the inverse Laplacian is simply set to zero. Because all three corrections are quadratic in the first differences, they vanish identically whenever consecutive levels coincide - at a steady state, and also at any start-up whose layers are all equal, which makes $\\hat E=E$ there an exact and independent check on both energy expressions. One boundary is worth stating plainly, because the decay is easy to over-read. The proof of $\\hat E^n\\le\\hat E^{n-1}$ passes through $\\|\\nabla_hv\\|^2=\\|v\\|^2-\\langle(1+\\Delta_h)v,v\\rangle$ and through $\\|v\\|^2\\le\\|\\nabla_hv\\|\\,\\|v\\|_{-1}$, and each of those needs $\\nabla_h$ and $\\Delta_h$ to be exact adjoints. Under the Nyquist convention fixed for this task they are not adjoint on the Nyquist rows: the even-order symbol $\\lambda$ keeps that entry and the odd-order multipliers of $D_x$ and $D_y$ drop it. For a field carried by the Nyquist mode $\\nabla_hv$ vanishes identically while $v$ does not, so the second inequality fails as completely as an inequality can, and the conclusion goes with it. The caveat is not vacuous. On an $8\\times8$ grid over $(2\\pi)^2$ with $\\varepsilon=0.25$, $\\tau=1$ and $S=200\\,S_{\\min}$ - so that $C_3=5.4708$ comfortably exceeds $C_{\\mathrm{req}}=0.3868$ and the stabilization condition holds - the seed $\\phi_0=(-1)^i$ produces $\\hat E=4436.39,\\,491.578,\\,5.70347,\\,10.3598,\\,0.09156,\\,0.72268,\\dots$, which rises twice, while on the same grid, the same $\\tau$ and the same $S$ the band-limited seed $\\cos(2\\pi i/8)\\cos(2\\pi j/8)$ falls at every step. So $\\hat E^n\\le\\hat E^{n-1}$ is a statement about fields band-limited strictly below Nyquist; on anything else it is a measurement rather than a theorem. On the five configurations this task runs, the adjointness gap $\\nu/\\|\\nabla_h\\phi\\|^2$ is zero to round-off on the three trigonometric seeds and about $4.7\\times10^{-4}$ on the two nucleus seeds, and $\\hat E$ is measured - not assumed - to decrease at every step of every one of them.

With $d^{(i)}=\\phi^{n-q+1+i}-\\phi^{n-q+i}$ for $i=0,\\dots,q-1$ the $q$ first differences, each of the three corrections is the contraction of the supplied matrix with those differences in the slot ordering $(d^{(0)},\\dots,d^{(q-1)})$: the $\\mathbf G$ term of the first bracket in the $\\langle\\cdot,\\cdot\\rangle_{-1}$ inner product, the $\\mathbf G$ term of the second bracket in the plain inner product of the GRADIENTS, contracting the two components as well as the $q$ slots, and the $\\mathbf J$ term in the plain inner product, every $\\langle\\cdot,\\cdot\\rangle$ carrying the cell area $h_xh_y$. The inverse Laplacian is applied by dividing by $\\lambda_{k,m}$ in Fourier space and setting the $k=m=0$ entry of the result to zero. Both $\\mathbf G$ and $\\mathbf J$ are symmetric here, so the order of the two indices does not matter, but keep it as written.

Returns
-------
A Python `float`, finite. When all $q+1$ layers are equal every first difference vanishes and the result is exactly the original discrete energy of that field. Because $\\mathbf G$ and $\\mathbf J$ are positive semidefinite and $S,\\varepsilon,\\tau\\ge0$, the three correction terms are non-negative, so $\\hat E^n\\ge E^n$ always. The value is not monotone in $n$ for every sequence of layers: its decay is guaranteed only where $\\nabla_h$ and $\\Delta_h$ are exact adjoints, which is what the background makes precise.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sqpfc_modified_energy(hist: "np.ndarray", cell: "float | Sequence[float]",
                          eps: float, S: float, tau: float,
                          G: "np.ndarray", J: "np.ndarray") -> float:
    """hist: array [phi^{n-q}, ..., phi^n] of shape (q+1, Mx, My), oldest first.
    cell: domain edge lengths, a scalar or a length-2 sequence (Lx, Ly).
    eps, S, tau: the parameter epsilon, the stabilization parameter, the step.
    G, J: the (q, q) quadratic-decomposition matrices; q = G.shape[0].
    Return the float Ehat^n defined above, with the factor eps on the J term.
    Raise ValueError if G and J are not square matrices of the same shape,
    if hist does not have shape (q+1, Mx, My) with q = G.shape[0], if tau
    is not positive, or if S is negative."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _pair(v):
    a = np.atleast_1d(np.asarray(v, float))
    return (float(a.flat[0]), float(a.flat[-1]))


def _area(phi, L):
    Lx, Ly = _pair(L)
    return (Lx / phi.shape[0]) * (Ly / phi.shape[1])


def _lapf(phi, L):
    """Delta_h phi, through the symbol of step 1."""
    lam = _oracle_sqpfc_laplacian_symbol(phi.shape, L)
    return np.fft.ifft2(-lam * np.fft.fft2(phi)).real


def _ipm1(u, w, L):
    """<u, w>_{-1}, zero mode excluded; the symbol comes from step 1."""
    lam = _oracle_sqpfc_laplacian_symbol(u.shape, L)
    inv = np.zeros_like(lam)
    nz = lam > 0
    inv[nz] = 1.0 / lam[nz]
    iu = np.fft.ifft2(inv * np.fft.fft2(u)).real
    return _area(u, L) * float(np.sum(iu * w))


def _oracle_sqpfc_modified_energy(hist: "np.ndarray", cell: "float | Sequence[float]",
                                  eps: float, S: float, tau: float,
                                  G: "np.ndarray", J: "np.ndarray") -> float:
    """Modified discrete energy; E from step 4, the gradients from step 3."""
    hist = np.asarray(hist, float)
    G = np.asarray(G, float)
    J = np.asarray(J, float)
    if G.ndim != 2 or G.shape[0] != G.shape[1] or J.shape != G.shape:
        raise ValueError("G and J must be square matrices of the same shape")
    if hist.ndim != 3 or hist.shape[0] != G.shape[0] + 1:
        raise ValueError("hist must have shape (q+1, Mx, My) with q = G.shape[0]")
    if not float(tau) > 0.0:
        raise ValueError("tau must be positive")
    if float(S) < 0.0:
        raise ValueError("S must be non-negative")
    q = G.shape[0]
    eps = float(eps)
    a = _area(hist[0], cell)
    d = hist[1:] - hist[:-1]
    gr = np.stack([_oracle_sqpfc_spectral_gradient(z, cell) for z in d])

    t1 = t2 = t3 = 0.0
    for i in range(q):
        for j in range(q):
            if G[i, j] != 0.0:
                t1 += G[i, j] * _ipm1(d[j], d[i], cell)
                t2 += G[i, j] * a * float(np.sum(gr[j] * gr[i]))
            if J[i, j] != 0.0:
                t3 += J[i, j] * a * float(np.sum(d[j] * d[i]))
    return (_oracle_sqpfc_discrete_energy(hist[-1], cell, eps)
            + t1 / float(tau) + float(S) * float(tau) ** (q - 1) * t2 + eps * t3)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    P1 = ("G3 = np.array([[0.0, 0.0, 0.0],\n"
          "               [0.0, 1.0 / 6.0, -7.0 / 24.0],\n"
          "               [0.0, -7.0 / 24.0, 65.0 / 96.0]])\n"
          "J3 = np.array([[0.0, 0.0, 0.0],\n"
          "               [0.0, 0.5, -0.5],\n"
          "               [0.0, -0.5, 1.0]])\n"
          "Lv = (8.0 * np.pi, 8.0 * np.pi)\n"
          "Mv = (32, 32)\n"
          "X, Y = np.meshgrid(np.arange(Mv[0]) * Lv[0] / Mv[0],\n"
          "                   np.arange(Mv[1]) * Lv[1] / Mv[1], indexing='ij')\n"
          "u = 0.07 + 0.60 * (np.cos(X) * np.cos(Y)\n"
          "                   + 0.40 * np.sin(2.0 * X) * np.cos(Y)\n"
          "                   + 0.30 * np.cos(X - 2.0 * Y))\n"
          # One plain assignment per level: packaging freezes each bound value,
          # but cannot freeze oracle output appended to a list in a loop.
          "h1 = _oracle_sqpfc_convex_split_step(np.stack([u, u, u]), Lv, 0.25, 5.0, 0.05)\n"
          "h2 = _oracle_sqpfc_convex_split_step(np.stack([u, u, h1]), Lv, 0.25, 5.0, 0.05)\n"
          "h3 = _oracle_sqpfc_convex_split_step(np.stack([u, h1, h2]), Lv, 0.25, 5.0, 0.05)\n"
          "h4 = _oracle_sqpfc_convex_split_step(np.stack([h1, h2, h3]), Lv, 0.25, 5.0, 0.05)\n"
          "W = np.stack([h1, h2, h3, h4])\n"
          "W0 = np.stack([u, u, u, u])\n")
    P2 = ("G3 = np.array([[0.0, 0.0, 0.0],\n"
          "               [0.0, 1.0 / 6.0, -7.0 / 24.0],\n"
          "               [0.0, -7.0 / 24.0, 65.0 / 96.0]])\n"
          "J3 = np.array([[0.0, 0.0, 0.0],\n"
          "               [0.0, 0.5, -0.5],\n"
          "               [0.0, -0.5, 1.0]])\n"
          "Lv = (25.0, 25.0)\n"
          "Mv = (32, 32)\n"
          "X, Y = np.meshgrid(np.arange(Mv[0]) * Lv[0] / Mv[0],\n"
          "                   np.arange(Mv[1]) * Lv[1] / Mv[1], indexing='ij')\n"
          "r = np.sqrt((X - 0.5 * Lv[0])**2 + (Y - 0.5 * Lv[1])**2)\n"
          "v = 2.5 * (1.0 - np.tanh(0.5 * (r - 2.0)))\n"
          "k1 = _oracle_sqpfc_convex_split_step(np.stack([v, v, v]), Lv, 0.50, 5.0, 0.10)\n"
          "k2 = _oracle_sqpfc_convex_split_step(np.stack([v, v, k1]), Lv, 0.50, 5.0, 0.10)\n"
          "k3 = _oracle_sqpfc_convex_split_step(np.stack([v, k1, k2]), Lv, 0.50, 5.0, 0.10)\n"
          "V = np.stack([v, k1, k2, k3])\n")
    return [
        {"setup": P1,
         "call": "sqpfc_modified_energy(W.copy(), Lv, 0.25, 5.0, 0.05, G3.copy(), J3.copy())",
         "gold_call": "_oracle_sqpfc_modified_energy(W.copy(), Lv, 0.25, 5.0, 0.05, G3.copy(), J3.copy())"},
        {"setup": P2,
         "call": "sqpfc_modified_energy(V.copy(), Lv, 0.50, 5.0, 0.10, G3.copy(), J3.copy())",
         "gold_call": "_oracle_sqpfc_modified_energy(V.copy(), Lv, 0.50, 5.0, 0.10, G3.copy(), J3.copy())"},
        # a constant-layer history collapses Ehat onto E exactly
        {"setup": (P1 +
                   "def pin(e, ref):\n"
                   "    if abs(e - ref) > 1e-9 * max(1.0, abs(ref)):\n"
                   "        return float('nan')\n"
                   "    return e\n"
                   "REF = _oracle_sqpfc_discrete_energy(u.copy(), Lv, 0.25)\n"),
         "call": ("pin(sqpfc_modified_energy(W0.copy(), Lv, 0.25, 5.0, 0.05,"
                  " G3.copy(), J3.copy()), REF)"),
         "gold_call": ("pin(_oracle_sqpfc_modified_energy(W0.copy(), Lv, 0.25,"
                       " 5.0, 0.05, G3.copy(), J3.copy()), REF)")},
        # S = 0 removes only the gradient correction
        {"setup": P1,
         "call": "sqpfc_modified_energy(W.copy(), Lv, 0.25, 0.0, 0.05, G3.copy(), J3.copy())",
         "gold_call": "_oracle_sqpfc_modified_energy(W.copy(), Lv, 0.25, 0.0, 0.05, G3.copy(), J3.copy())"},
        # zeroing J removes exactly the eps-weighted extrapolation correction
        {"setup": P1,
         "call": ("sqpfc_modified_energy(W.copy(), Lv, 0.25, 5.0, 0.05, G3.copy(),"
                  " np.zeros((3, 3)))"),
         "gold_call": ("_oracle_sqpfc_modified_energy(W.copy(), Lv, 0.25, 5.0, 0.05, G3.copy(),"
                       " np.zeros((3, 3)))")},
        # with G and J both zero, Ehat is just E of the last slot
        {"setup": (P2 +
                   "def pin(e, ref):\n"
                   "    if abs(e - ref) > 1e-9 * max(1.0, abs(ref)):\n"
                   "        return float('nan')\n"
                   "    return e\n"
                   "REF = _oracle_sqpfc_discrete_energy(V[-1].copy(), Lv, 0.50)\n"),
         "call": ("pin(sqpfc_modified_energy(V.copy(), Lv, 0.50, 5.0, 0.10,"
                  " np.zeros((3, 3)), np.zeros((3, 3))), REF)"),
         "gold_call": ("pin(_oracle_sqpfc_modified_energy(V.copy(), Lv, 0.50,"
                       " 5.0, 0.10, np.zeros((3, 3)), np.zeros((3, 3))), REF)")},
        # contract: G is 3x3, so the history must carry q+1 = 4 levels; three
        # levels must raise rather than be contracted against a wrong slot count.
        {"setup": (P1 + "def trap(f):\n"
                  "    try:\n"
                  "        return float(np.asarray(f()).ravel()[0])\n"
                  "    except ValueError:\n"
                  "        return -12345.0\n"),
         "call": "trap(lambda: sqpfc_modified_energy(W[:3].copy(), Lv, 0.25, 5.0, 0.05, G3.copy(), J3.copy()))",
         "gold_call": ("trap(lambda: _oracle_sqpfc_modified_energy("
                       "W[:3].copy(), Lv, 0.25, 5.0, 0.05, G3.copy(), J3.copy()))")},
    ]
