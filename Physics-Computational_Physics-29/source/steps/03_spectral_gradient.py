"""
Apply the two first-derivative operators $D_x$ and $D_y$ of the Fourier pseudo-spectral discretisation to one real periodic field, and return them stacked as $\\nabla_hv=(D_xv,D_yv)^T$ with shape `(2,) + v.shape`. Each is applied by transforming, multiplying by that direction's symbol, transforming back and taking the real part. **The Nyquist entry is dropped**: the multiplier is set to zero at $k=-M_x/2$ for $D_x$ and at $m=-M_y/2$ for $D_y$ before the inverse transform. No dealiasing of any other kind is applied. Each axis carries its own edge length and its own point count.

For a real grid function the Fourier coefficients obey $\\tilde v_{-k,-m}=\\overline{\\tilde v_{k,m}}$, which pairs every mode with a conjugate partner and makes the inverse transform real. On an even grid the Nyquist index is its own negative, so that symmetry maps the whole Nyquist row $k=-M_x/2$ onto itself with $m\\mapsto-m$: the row is Hermitian in $m$ as a whole, and its entries are forced real only where $m$ is self-paired too, at $m=0$ and $m=-M_y/2$. In two dimensions they are otherwise complex - on an $8\\times8$ grid the real field $(-1)^i\\sin(2\\pi j/8)$ has row entries $-32i$ at $(k,m)=(-4,1)$ and $+32i$ at $(-4,-1)$ - so what has to be got right is the statement about the row, not a claim about each coefficient. An even-order multiplier such as $-4\\pi^2k^2/L_x^2$ is real and constant along the row, so it preserves that symmetry and the result is real. An odd-order multiplier such as $2\\pi ik/L_x$ is purely imaginary and likewise constant along the row, so it turns the row's combined contribution to the output into a purely imaginary one - a contribution that is spurious, because a Nyquist oscillation has no well-defined slope on the grid: the pure Nyquist mode $v_{ij}=(-1)^i$, constant in $y$, is equally $\\cos(\\pi i)$ and $\\cos(-\\pi i)$, whose derivatives differ by a sign, and the convention returns $D_xv=0$ for it outright. More than one prescription removes that spurious contribution - one acting on the multiplier before the inverse transform, one acting on the array after it - and whether they agree is not something to take on faith: deciding which prescriptions are admissible here, and checking on this grid that they coincide, is part of this step. What is fixed, and what the tests hold you to, is the outcome the convention must produce: the returned array is real, and the pure Nyquist oscillation is assigned zero slope. A prescription that leaves a complex array instead is wrong, and every norm computed from it is wrong with it. The convention matters here in practice because the nonlinear term of this model differentiates twice - once to build $\\nabla_h\\phi$, again to take the divergence of $|\\nabla_h\\phi|^2\\nabla_h\\phi$ - and the cubing in between moves amplitude into the highest modes, exactly where the ambiguity lives.

$$(D_xv)^\\sim_{k,m} \\;=\\; \\frac{2\\pi ik}{L_x}\\,\\tilde v_{k,m},\\qquad (D_yv)^\\sim_{k,m} \\;=\\; \\frac{2\\pi im}{L_y}\\,\\tilde v_{k,m},$$



with the signed indices of the previous step. How the Nyquist rows $k=-M_x/2$ and $m=-M_y/2$ are handled, and what each inverse transform needs so that the returned array is real, follow from the background and are deliberately not prescribed here. Return `np.stack([Dx_v, Dy_v])`. Note that $\\Delta_h$ is **not** obtained by composing these two operators: it carries its own second-order multiplier $-\\lambda_{k,m}$, in which the Nyquist entry is kept.

Returns
-------
`np.ndarray` of shape `(2,) + phi.shape`, real and finite. For a constant field both slices are exactly $0$. For a single resolved mode $\\cos(2\\pi(ax/L_x+by/L_y))$ with $|a|<M_x/2$ and $|b|<M_y/2$ the slices are the exact derivatives $-\\tfrac{2\\pi a}{L_x}\\sin(\\cdot)$ and $-\\tfrac{2\\pi b}{L_y}\\sin(\\cdot)$ to round-off. Both slices have exactly zero discrete mean.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sqpfc_spectral_gradient(phi: "np.ndarray",
                            cell: "float | Sequence[float]") -> "np.ndarray":
    """phi: real periodic field of shape (Mx, My).
    cell: domain edge lengths, a scalar or a length-2 sequence (Lx, Ly).
    Return the real array [D_x phi, D_y phi] of shape (2,) + phi.shape, with
    the Nyquist entry of each first-derivative multiplier set to zero.
    Raise ValueError if phi is not two-dimensional, or if cell is not a
    positive scalar or length-2 sequence."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _pair(v):
    a = np.atleast_1d(np.asarray(v, float))
    return (float(a.flat[0]), float(a.flat[-1]))


def _oracle_sqpfc_spectral_gradient(phi: "np.ndarray",
                                    cell: "float | Sequence[float]") -> "np.ndarray":
    """(D_x phi, D_y phi), Nyquist dropped from the odd-order multipliers."""
    phi = np.asarray(phi, float)
    if phi.ndim != 2:
        raise ValueError("phi must be a two-dimensional array")
    c = np.atleast_1d(np.asarray(cell, float))
    if c.size not in (1, 2) or np.any(c <= 0.0):
        raise ValueError("cell must be a positive scalar or length-2 sequence")
    Mx, My = phi.shape
    Lx, Ly = _pair(cell)
    kx = np.fft.fftfreq(Mx) * Mx
    ky = np.fft.fftfreq(My) * My
    kx = np.where(np.abs(kx) == Mx // 2, 0.0, kx)
    ky = np.where(np.abs(ky) == My // 2, 0.0, ky)
    f = np.fft.fft2(phi)
    pi = float(np.pi)
    dx = np.fft.ifft2((2.0 * pi / Lx * 1j * kx).reshape(Mx, 1) * f).real
    dy = np.fft.ifft2((2.0 * pi / Ly * 1j * ky).reshape(1, My) * f).real
    return np.stack([dx, dy])

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
    RECT = ("Lv = (32.0, 20.0)\n"
            "Mv = (48, 24)\n"
            "X, Y = np.meshgrid(np.arange(Mv[0]) * Lv[0] / Mv[0],\n"
            "                   np.arange(Mv[1]) * Lv[1] / Mv[1], indexing='ij')\n"
            "r = np.sqrt((X - 0.5 * Lv[0])**2 + (Y - 0.5 * Lv[1])**2)\n"
            "u = 2.5 * (1.0 - np.tanh(0.5 * (r - 2.0)))\n")
    NYQ = ("Lv = (4.0 * np.pi, 6.0 * np.pi)\n"
           "Mv = (16, 12)\n"
           "u = ((-1.0) ** np.arange(Mv[0])).reshape(Mv[0], 1) * np.ones((1, Mv[1]))\n"
           "u = u + 0.5 * np.cos(2 * np.pi * np.arange(Mv[0]).reshape(Mv[0], 1) / Mv[0])\n")
    MODE = ("Lv = (6.0, 10.0)\n"
            "Mv = (24, 20)\n"
            "X, Y = np.meshgrid(np.arange(Mv[0]) * Lv[0] / Mv[0],\n"
            "                   np.arange(Mv[1]) * Lv[1] / Mv[1], indexing='ij')\n"
            "th = 2 * np.pi * (3 * X / Lv[0] + 2 * Y / Lv[1])\n"
            "u = np.cos(th)\n")
    return [
        {"setup": TRIG,
         "call": "sqpfc_spectral_gradient(u.copy(), Lv)",
         "gold_call": "_oracle_sqpfc_spectral_gradient(u.copy(), Lv)"},
        {"setup": RECT,
         "call": "sqpfc_spectral_gradient(u.copy(), Lv)",
         "gold_call": "_oracle_sqpfc_spectral_gradient(u.copy(), Lv)"},
        {"setup": NYQ,
         "call": "sqpfc_spectral_gradient(u.copy(), Lv)",
         "gold_call": "_oracle_sqpfc_spectral_gradient(u.copy(), Lv)"},
        # a constant field has an identically zero gradient
        {"setup": ("u = np.full((10, 14), -0.375)\nLv = (3.0, 7.0)\n"
                   "def pin(g, ref):\n"
                   "    if not np.allclose(g, ref, rtol=0.0, atol=1e-10):\n"
                   "        return np.full(np.shape(g), np.nan)\n"
                   "    return g\n"
                   "REF = np.zeros((2, 10, 14))\n"),
         "call": "pin(sqpfc_spectral_gradient(u.copy(), Lv), REF)",
         "gold_call": "pin(_oracle_sqpfc_spectral_gradient(u.copy(), Lv), REF)"},
        # a single Fourier mode is differentiated exactly
        {"setup": (MODE +
                   "def pin(g, ref):\n"
                   "    if not np.allclose(g, ref, rtol=0.0, atol=1e-10):\n"
                   "        return np.full(np.shape(g), np.nan)\n"
                   "    return g\n"
                   "REF = np.stack([-(2 * np.pi * 3 / Lv[0]) * np.sin(th),\n"
                   "                -(2 * np.pi * 2 / Lv[1]) * np.sin(th)])\n"),
         "call": "pin(sqpfc_spectral_gradient(u.copy(), Lv), REF)",
         "gold_call": "pin(_oracle_sqpfc_spectral_gradient(u.copy(), Lv), REF)"},
        {"setup": ("Lv = (2.0 * np.pi, 2.0 * np.pi)\n"
                   "X, Y = np.meshgrid(np.arange(8) * Lv[0] / 8,\n"
                   "                   np.arange(8) * Lv[1] / 8, indexing='ij')\n"
                   "u = np.sin(X + 2 * Y) + 0.25\n"),
         "call": "sqpfc_spectral_gradient(u.copy(), Lv)",
         "gold_call": "_oracle_sqpfc_spectral_gradient(u.copy(), Lv)"},
        # contract: a one-dimensional field has no second axis to difference.
        {"setup": ("def trap(f):\n"
                  "    try:\n"
                  "        return float(np.asarray(f()).ravel()[0])\n"
                  "    except ValueError:\n"
                  "        return -12345.0\n"),
         "call": "trap(lambda: sqpfc_spectral_gradient(np.arange(8.0), 4.0))",
         "gold_call": ("trap(lambda: _oracle_sqpfc_spectral_gradient("
                       "np.arange(8.0), 4.0))")},
    ]
