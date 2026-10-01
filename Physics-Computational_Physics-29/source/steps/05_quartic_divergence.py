"""
Evaluate the nonlinear term of the chemical potential, $N(\\phi)=-\\nabla_h\\cdot\\big(|\\nabla_h\\phi|^2\\nabla_h\\phi\\big)$, and return it as a real array with the same shape as $\\phi$. Build it in the order the expression is written: form the vector $\\nabla_h\\phi=(D_x\\phi,D_y\\phi)$, form the scalar $|\\nabla_h\\phi|^2=(D_x\\phi)^2+(D_y\\phi)^2$ pointwise, multiply it into each component to get the flux $\\boldsymbol F=|\\nabla_h\\phi|^2\\nabla_h\\phi$, then take $-\\big(D_xF_x+D_yF_y\\big)$. Both the gradient and the divergence use the first-derivative operators of the earlier step, with their Nyquist entries dropped. **Do not expand by the chain rule**: $-\\nabla_h\\cdot(|\\nabla_h\\phi|^2\\nabla_h\\phi)$ is not $-3|\\nabla_h\\phi|^2\\Delta_h\\phi$, because $|\\nabla_h\\phi|^2$ is not constant, and the discrete product rule fails for spectral differences in any case.

This is the variational derivative of the gradient quartic: for $F[\\phi]=\\tfrac14\\int|\\nabla\\phi|^4$, perturbing $\\phi\\to\\phi+\\rho\\psi$ and differentiating at $\\rho=0$ gives $\\int|\\nabla\\phi|^2\\nabla\\phi\\cdot\\nabla\\psi$, and one integration by parts moves the divergence off $\\psi$ to give $-\\int\\nabla\\cdot(|\\nabla\\phi|^2\\nabla\\phi)\\,\\psi$. The operator is a 4-Laplacian, the $p=4$ member of the $p$-Laplacian family $-\\nabla\\cdot(|\\nabla\\phi|^{p-2}\\nabla\\phi)$, and it is quasilinear rather than semilinear: its coefficient depends on the solution's own gradient, so it degenerates where $\\nabla\\phi=0$ and stiffens sharply where the gradient is large. That is what makes the square phase field crystal model harder than the classical one, whose nonlinearity is the algebraic $\\phi^3$. Two consequences shape the implementation. Because the coefficient is not constant the operator is not diagonal in Fourier space, so it cannot be applied as a multiplier and must be kept on the explicit side of the time step - which is exactly why the scheme solves a fixed-point iteration rather than a single linear system. And because the flux is cubic in first derivatives, evaluating it pointwise on the grid aliases: the product of three fields band-limited to $M/2$ contains wavenumbers up to $3M/2$, which fold back onto the grid. Pseudo-spectral practice here is to accept that aliasing rather than to dealias, and the discretisation is fixed accordingly - no $3/2$ rule, no filter. Discretely the operator inherits the structure that matters: with the divergence and gradient built from the same skew-adjoint multipliers, $\\langle N(\\phi),\\psi\\rangle=\\langle|\\nabla_h\\phi|^2\\nabla_h\\phi,\\nabla_h\\psi\\rangle$, so $\\langle N(\\phi),\\phi\\rangle=\\|\\nabla_h\\phi\\|_4^4\\ge0$ and $N$ has exactly zero discrete mean - which is what lets the flow conserve mass.

$$\\boldsymbol F \\;=\\; \\big((D_x\\phi)^2+(D_y\\phi)^2\\big)\\,\\big(D_x\\phi,\\;D_y\\phi\\big),\\qquad N(\\phi) \\;=\\; -\\big(D_xF_x+D_yF_y\\big).$$



Four transforms in, four out: two to form $\\nabla_h\\phi$ and two to take the divergence of $\\boldsymbol F$. Use the same Nyquist-dropped multipliers in both places.

Returns
-------
`np.ndarray` of the same shape as `phi`, real and finite. Its discrete mean is exactly $0$ (to round-off), since it is a discrete divergence. It vanishes identically for a constant field, and it is odd under $\\phi\\to-\\phi$ and homogeneous of degree three, so $N(c\\phi)=c^3N(\\phi)$ for any scalar $c$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sqpfc_quartic_divergence(phi: "np.ndarray",
                             cell: "float | Sequence[float]") -> "np.ndarray":
    """phi: real periodic field of shape (Mx, My).
    cell: domain edge lengths, a scalar or a length-2 sequence (Lx, Ly).
    Return the real array N(phi) = -div_h(|grad_h phi|^2 grad_h phi) of the
    same shape as phi.
    Raise ValueError if phi is not two-dimensional."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sqpfc_quartic_divergence(phi: "np.ndarray",
                                     cell: "float | Sequence[float]") -> "np.ndarray":
    """N(phi) = -div_h(|grad_h phi|^2 grad_h phi); every derivative is step 3."""
    phi = np.asarray(phi, float)
    if phi.ndim != 2:
        raise ValueError("phi must be a two-dimensional array")
    g = _oracle_sqpfc_spectral_gradient(phi, cell)
    g2 = g[0] ** 2 + g[1] ** 2
    return -(_oracle_sqpfc_spectral_gradient(g2 * g[0], cell)[0]
             + _oracle_sqpfc_spectral_gradient(g2 * g[1], cell)[1])

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
    RECT = ("Lv = (6.0 * np.pi, 10.0 * np.pi)\n"
            "Mv = (24, 40)\n"
            "X, Y = np.meshgrid(np.arange(Mv[0]) * Lv[0] / Mv[0],\n"
            "                   np.arange(Mv[1]) * Lv[1] / Mv[1], indexing='ij')\n"
            "u = 0.07 + 0.60 * (np.cos(X) * np.cos(Y)\n"
            "                   + 0.40 * np.sin(2.0 * X) * np.cos(Y)\n"
            "                   + 0.30 * np.cos(X - 2.0 * Y))\n")
    NUC = ("Lv = (32.0, 20.0)\n"
           "Mv = (48, 24)\n"
           "X, Y = np.meshgrid(np.arange(Mv[0]) * Lv[0] / Mv[0],\n"
           "                   np.arange(Mv[1]) * Lv[1] / Mv[1], indexing='ij')\n"
           "r = np.sqrt((X - 0.5 * Lv[0])**2 + (Y - 0.5 * Lv[1])**2)\n"
           "u = 2.5 * (1.0 - np.tanh(0.5 * (r - 2.0)))\n")
    return [
        {"setup": TRIG, "call": "sqpfc_quartic_divergence(u.copy(), Lv)",
         "gold_call": "_oracle_sqpfc_quartic_divergence(u.copy(), Lv)"},
        {"setup": RECT, "call": "sqpfc_quartic_divergence(u.copy(), Lv)",
         "gold_call": "_oracle_sqpfc_quartic_divergence(u.copy(), Lv)"},
        {"setup": NUC, "call": "sqpfc_quartic_divergence(u.copy(), Lv)",
         "gold_call": "_oracle_sqpfc_quartic_divergence(u.copy(), Lv)"},
        # constant field -> identically zero
        {"setup": ("u = np.full((10, 16), 1.25)\nLv = (4.0, 6.0)\n"
                   "def pin(n, ref):\n"
                   "    if not np.allclose(n, ref, rtol=0.0, atol=1e-10):\n"
                   "        return np.full(np.shape(n), np.nan)\n"
                   "    return n\n"
                   "REF = np.zeros((10, 16))\n"),
         "call": "pin(sqpfc_quartic_divergence(u.copy(), Lv), REF)",
         "gold_call": "pin(_oracle_sqpfc_quartic_divergence(u.copy(), Lv), REF)"},
        # cubic homogeneity: N(3u) = 27 N(u)
        {"setup": TRIG, "call": "sqpfc_quartic_divergence(3.0 * u, Lv)",
         "gold_call": "27.0 * _oracle_sqpfc_quartic_divergence(u.copy(), Lv)"},
        # <N(u), u> = ||grad u||_4^4  (summation by parts)
        {"setup": (TRIG + "a = (Lv[0] / Mv[0]) * (Lv[1] / Mv[1])\n"
                   "g = _oracle_sqpfc_spectral_gradient(u.copy(), Lv)\n"
                   "q4 = a * float(np.sum((g[0]**2 + g[1]**2)**2))\n"
                   "def pair(f):\n"
                   "    v = a * float(np.sum(f(u.copy()) * u))\n"
                   "    if abs(v - q4) > 1e-8 * max(1.0, abs(q4)):\n"
                   "        return float('nan')\n"
                   "    return v\n"),
         "call": "pair(lambda w: sqpfc_quartic_divergence(w, Lv))",
         "gold_call": ("pair(lambda w: _oracle_sqpfc_quartic_divergence("
                       "w, Lv))")},
        # contract: a one-dimensional field has no divergence on this grid.
        {"setup": ("def trap(f):\n"
                  "    try:\n"
                  "        return float(np.asarray(f()).ravel()[0])\n"
                  "    except ValueError:\n"
                  "        return -12345.0\n"),
         "call": "trap(lambda: sqpfc_quartic_divergence(np.arange(8.0), 4.0))",
         "gold_call": ("trap(lambda: _oracle_sqpfc_quartic_divergence("
                       "np.arange(8.0), 4.0))")},
    ]
