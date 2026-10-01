"""
Evaluate the original discrete free energy of the square phase field crystal model, $E=\\tfrac14\\|\\nabla_h\\phi\\|_4^4-\\tfrac{\\varepsilon}{2}\\|\\phi\\|^2+\\tfrac12\\|(1+\\Delta_h)\\phi\\|^2$, and return it as a single float. All three norms carry the discrete inner product $\\langle u,w\\rangle=h_xh_y\\sum_{ij}u_{ij}w_{ij}$, so every sum is weighted by the cell area $h_xh_y=(L_x/M_x)(L_y/M_y)$. The quartic is a **gradient** quartic and the modulus inside it is the pointwise Euclidean length of the vector $\\nabla_h\\phi$, so the summand is $\\big((D_x\\phi)^2+(D_y\\phi)^2\\big)^2$: it is neither $\\phi^4$ nor the square of the squared $L^2$ norm of the gradient. The concave term enters with a minus sign.

This is a Swift-Hohenberg energy with the nonlinearity moved into the gradient. The operator $(1+\\Delta)$ annihilates every mode of wavenumber $|\\boldsymbol k|=1$ exactly, so $\\tfrac12\\|(1+\\Delta_h)\\phi\\|^2$ costs nothing on the selected shell and grows quartically in $|\\boldsymbol k|$ away from it; the concave term $-\\tfrac{\\varepsilon}{2}\\|\\phi\\|^2$ destabilises the flat state and sets the amplitude scale; and the quartic bounds the whole functional below and decides which lattice on the shell wins. With the classical $\\tfrac14\\phi^4$ the cubic three-mode resonance among unit wavevectors is available and a hexagonal lattice is selected; with $\\tfrac14|\\nabla\\phi|^4$ that resonance is absent and the square lattice wins. Two structural facts are worth having. First the functional is bounded below: $\\tfrac14a^4\\ge a^2-1$ pointwise, from $(a^2-2)^2\\ge0$, gives $\\tfrac14\\|\\nabla_h\\phi\\|_4^4\\ge\\|\\nabla_h\\phi\\|^2-|\\Omega|$, while expanding the convex term gives $\\tfrac12\\|(1+\\Delta_h)\\phi\\|^2=\\tfrac12\\|\\phi\\|^2+\\langle\\phi,\\Delta_h\\phi\\rangle+\\tfrac12\\|\\Delta_h\\phi\\|^2$. Summation by parts turns $\\langle\\phi,\\Delta_h\\phi\\rangle$ into $-\\|\\nabla_h\\phi\\|^2$, the two gradient terms cancel, and what is left is $E\\ge\\tfrac{1-\\varepsilon}{2}\\|\\phi\\|^2+\\tfrac12\\|\\Delta_h\\phi\\|^2-|\\Omega|$; for $\\varepsilon\\le1$ this bounds $\\|\\Delta_h\\phi\\|$ by the energy, which is the estimate that converts energy stability into an $H^2_h$ bound and then, by a discrete Sobolev embedding, into a $W^{1,6}_h$ bound on the gradient. One discrete caveat is worth carrying, because it is a consequence of the Nyquist convention rather than of the analysis: since that convention keeps the Nyquist entry in $\\Delta_h$ but drops it from $\\nabla_h$, the two are not exact adjoints there, and the correct discrete statement is $\\langle\\phi,\\Delta_h\\phi\\rangle=-\\|\\nabla_h\\phi\\|^2-\\nu$ with $\\nu\\ge0$ the part of the gradient energy carried by the Nyquist rows. Summation by parts is therefore exact for any field band-limited strictly below Nyquist and approximate otherwise, and the lower bound holds up to that $\\nu$. It is a small effect on well-resolved data - measured on the fields used here, $\\nu/\\|\\nabla_h\\phi\\|^2$ is $0$ to round-off for the smooth trigonometric seeds and about $5\\times10^{-4}$ for the tanh-profiled ones - but it is not identically zero, and a response that claims the cancellation is exact on every grid function is overstating it. Second, the convex term has two equivalent discrete forms, $\\tfrac12\\|(1+\\Delta_h)\\phi\\|^2$ and $\\tfrac12\\langle\\phi,(1+\\Delta_h)^2\\phi\\rangle$, equal exactly because $\\Delta_h$ is self-adjoint for this inner product; checking one against the other is a cheap and sharp test of the operator implementation.

$$E \\;=\\; \\frac14\\,h_xh_y\\sum_{i,j}\\Big((D_x\\phi)_{ij}^2+(D_y\\phi)_{ij}^2\\Big)^2 \\;-\\;\\frac{\\varepsilon}{2}\\,h_xh_y\\sum_{i,j}\\phi_{ij}^2 \\;+\\;\\frac12\\,h_xh_y\\sum_{i,j}\\Big(\\phi_{ij}+(\\Delta_h\\phi)_{ij}\\Big)^2 .$$



Use the gradient of the previous step for $D_x\\phi$ and $D_y\\phi$, and the symbol $-\\lambda_{k,m}$ of the first step for $\\Delta_h\\phi$ - in $\\lambda$ the Nyquist entry is kept. Return a Python float.

Returns
-------
A Python `float`, finite. For a constant field $\\phi\\equiv c$ the gradient vanishes and $E=\\tfrac{1-\\varepsilon}{2}c^2|\\Omega|$ exactly, with $|\\Omega|=L_xL_y$. The value is bounded below on any fixed mesh - for $\\varepsilon\\le1$ it exceeds $\\tfrac{1-\\varepsilon}{2}\\|\\phi\\|^2+\\tfrac12\\|\\Delta_h\\phi\\|^2-|\\Omega|-\\nu$, with $\\nu\\ge0$ the Nyquist correction described in the background - and it may be negative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sqpfc_discrete_energy(phi: "np.ndarray", cell: "float | Sequence[float]",
                          eps: float) -> float:
    """phi: real periodic field of shape (Mx, My).
    cell: domain edge lengths, a scalar or a length-2 sequence (Lx, Ly).
    eps: the parameter epsilon of the model.
    Return the float E = (1/4)||grad_h phi||_4^4 - (eps/2)||phi||^2
    + (1/2)||(1 + Lap_h) phi||^2, every norm carrying the cell area hx*hy.
    Raise ValueError if phi is not two-dimensional."""
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


def _oracle_sqpfc_discrete_energy(phi: "np.ndarray", cell: "float | Sequence[float]",
                                  eps: float) -> float:
    """Original discrete free energy; gradient from step 3, symbol from step 1."""
    phi = np.asarray(phi, float)
    if phi.ndim != 2:
        raise ValueError("phi must be a two-dimensional array")
    eps = float(eps)
    a = _area(phi, cell)
    g = _oracle_sqpfc_spectral_gradient(phi, cell)
    g2 = g[0] ** 2 + g[1] ** 2
    w = phi + _lapf(phi, cell)
    return (0.25 * a * float(np.sum(g2 ** 2))
            - 0.5 * eps * a * float(np.sum(phi ** 2))
            + 0.5 * a * float(np.sum(w ** 2)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    TRIG = ("Lv = (8.0 * np.pi, 8.0 * np.pi)\n"
            "Mv = (32, 32)\n"
            "X, Y = np.meshgrid(np.arange(Mv[0]) * Lv[0] / Mv[0],\n"
            "                   np.arange(Mv[1]) * Lv[1] / Mv[1], indexing='ij')\n"
            "u = 0.07 + 0.60 * (np.cos(X) * np.cos(Y)\n"
            "                   + 0.40 * np.sin(2.0 * X) * np.cos(Y)\n"
            "                   + 0.30 * np.cos(X - 2.0 * Y))\n")
    RECT = ("Lv = (10.0 * np.pi, 4.0 * np.pi)\n"
            "Mv = (40, 24)\n"
            "X, Y = np.meshgrid(np.arange(Mv[0]) * Lv[0] / Mv[0],\n"
            "                   np.arange(Mv[1]) * Lv[1] / Mv[1], indexing='ij')\n"
            "u = 0.07 + 0.60 * (np.cos(X) * np.cos(Y)\n"
            "                   + 0.40 * np.sin(2.0 * X) * np.cos(Y)\n"
            "                   + 0.30 * np.cos(X - 2.0 * Y))\n")
    NUC = ("Lv = (25.0, 25.0)\n"
           "Mv = (32, 32)\n"
           "X, Y = np.meshgrid(np.arange(Mv[0]) * Lv[0] / Mv[0],\n"
           "                   np.arange(Mv[1]) * Lv[1] / Mv[1], indexing='ij')\n"
           "r = np.sqrt((X - 0.5 * Lv[0])**2 + (Y - 0.5 * Lv[1])**2)\n"
           "u = 2.5 * (1.0 - np.tanh(0.5 * (r - 2.0)))\n")
    return [
        {"setup": TRIG, "call": "sqpfc_discrete_energy(u.copy(), Lv, 0.25)",
         "gold_call": "_oracle_sqpfc_discrete_energy(u.copy(), Lv, 0.25)"},
        {"setup": RECT, "call": "sqpfc_discrete_energy(u.copy(), Lv, 0.20)",
         "gold_call": "_oracle_sqpfc_discrete_energy(u.copy(), Lv, 0.20)"},
        {"setup": NUC, "call": "sqpfc_discrete_energy(u.copy(), Lv, 0.50)",
         "gold_call": "_oracle_sqpfc_discrete_energy(u.copy(), Lv, 0.50)"},
        # on a constant field the energy is the algebraic expression below
        {"setup": ("u = np.full((12, 18), 0.4)\nLv = (5.0, 9.0)\n"
                   "def pin(e, ref):\n"
                   "    if abs(e - ref) > 1e-9 * max(1.0, abs(ref)):\n"
                   "        return float('nan')\n"
                   "    return e\n"
                   "REF = 0.5 * (1.0 - 0.3) * 0.4**2 * 45.0\n"),
         "call": "pin(sqpfc_discrete_energy(u.copy(), Lv, 0.3), REF)",
         "gold_call": "pin(_oracle_sqpfc_discrete_energy(u.copy(), Lv, 0.3), REF)"},
        {"setup": TRIG, "call": "sqpfc_discrete_energy(u.copy(), Lv, 0.0)",
         "gold_call": "_oracle_sqpfc_discrete_energy(u.copy(), Lv, 0.0)"},
        # the eps dependence is exactly linear with the stated slope
        {"setup": (TRIG + "a = (Lv[0] / Mv[0]) * (Lv[1] / Mv[1])\n"
                   "REF = 0.35 * a * float(np.sum(u**2))\n"
                   "def slope(f):\n"
                   "    d = f(0.0) - f(0.7)\n"
                   "    if abs(d - REF) > 1e-9 * max(1.0, abs(REF)):\n"
                   "        return float('nan')\n"
                   "    return d\n"),
         "call": "slope(lambda e: sqpfc_discrete_energy(u.copy(), Lv, e))",
         "gold_call": ("slope(lambda e: _oracle_sqpfc_discrete_energy("
                       "u.copy(), Lv, e))")},
        # contract: a stack of fields is not a field; the energy is defined on
        # one two-dimensional grid function at a time.
        {"setup": ("def trap(f):\n"
                  "    try:\n"
                  "        return float(np.asarray(f()).ravel()[0])\n"
                  "    except ValueError:\n"
                  "        return -12345.0\n"),
         "call": "trap(lambda: sqpfc_discrete_energy(np.zeros((2, 8, 8)), 4.0, 0.25))",
         "gold_call": ("trap(lambda: _oracle_sqpfc_discrete_energy("
                       "np.zeros((2, 8, 8)), 4.0, 0.25))")},
    ]
