"""
Build the Fourier symbol of the negative Laplacian $-\\Delta_h$ on a periodic rectangle discretised by the Fourier pseudo-spectral method. The rectangle is $(0,L_x)\\times(0,L_y)$ and carries $M_x\\times M_y$ equispaced points with $M_x$ and $M_y$ even; the admissible wavenumber indices are the signed set $-M_x/2\\le k\\le M_x/2-1$ and $-M_y/2\\le m\\le M_y/2-1$, stored in the order `numpy.fft.fftfreq` produces, so that the returned array can be multiplied elementwise against `numpy.fft.fft2` output without any reordering. Return the real array $\\lambda_{k,m}$ of shape `(Mx, My)`. The two axes are independent: each carries its own number of points and its own edge length, and neither the grid spacing nor a single scalar $L$ may be substituted for the pair.

The discrete Fourier transform diagonalises every constant-coefficient differential operator on a periodic box. Writing a grid function as $v_{ij}=\\sum_{k,m}\\tilde v_{k,m}e^{i2\\pi(kx_i/L_x+my_j/L_y)}$, the second derivative in $x$ multiplies the coefficient $\\tilde v_{k,m}$ by $(i2\\pi k/L_x)^2=-4\\pi^2k^2/L_x^2$, and likewise in $y$, so the Laplacian carries $-4\\pi^2(k^2/L_x^2+m^2/L_y^2)$ and $-\\Delta_h$ carries the non-negative quantity $\\lambda_{k,m}=4\\pi^2(k^2/L_x^2+m^2/L_y^2)$. This is a *spectral* symbol, exact for every representable mode; it is not the bounded finite-difference symbol $4h_x^{-2}\\sin^2(\\pi k/M_x)+4h_y^{-2}\\sin^2(\\pi m/M_y)$, which agrees with it only as $k/M_x\\to0$ and falls short by a factor of about $2.5$ at the corner mode on the meshes used here. Two features matter downstream. The symbol is unbounded in the mesh, growing like $\\pi^2(M_x^2/L_x^2+M_y^2/L_y^2)$ at the corner, which is what makes an implicit treatment of the sixth-order linear operator necessary. And the index set is signed rather than $0,\\dots,M-1$: for a real field, mode $k$ and mode $k-M_x$ are the same grid function, but only the signed representative gives the right derivative, so using unsigned indices assigns spuriously large symbols to every mode above Nyquist. The zero mode $k=m=0$ gives $\\lambda=0$, reflecting that $-\\Delta_h$ annihilates constants; that null direction is why the $H^{-1}$ inner product used later has to be restricted to mean-zero fields.

$$\\lambda_{k,m} \\;=\\; 4\\pi^2\\left(\\frac{k^2}{L_x^2}+\\frac{m^2}{L_y^2}\\right),\\qquad k\\in\\{-\\tfrac{M_x}{2},\\dots,\\tfrac{M_x}{2}-1\\},\\quad m\\in\\{-\\tfrac{M_y}{2},\\dots,\\tfrac{M_y}{2}-1\\}.$$



The signed index vectors are exactly `np.fft.fftfreq(Mx) * Mx` and `np.fft.fftfreq(My) * My`, which return $0,1,\\dots,M/2-1,-M/2,\\dots,-1$; keep that storage order. Broadcast $k$ along the first axis and $m$ along the second. Both `mesh` and `cell` may be given as a scalar or as a length-2 sequence; a scalar means the same value on both axes.

Returns
-------
`np.ndarray` of shape `(Mx, My)`, real, non-negative and finite. Entry `[0, 0]` is exactly $0$; the array is symmetric under $k\\to-k$ and $m\\to-m$; its maximum is attained at the corner index $(-M_x/2,-M_y/2)$ and equals $\\pi^2(M_x^2/L_x^2+M_y^2/L_y^2)$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sqpfc_laplacian_symbol(mesh: "int | Sequence[int]",
                           cell: "float | Sequence[float]") -> "np.ndarray":
    """mesh: number of grid points, a scalar or a length-2 sequence (Mx, My).
    cell: domain edge lengths, a scalar or a length-2 sequence (Lx, Ly).
    Return the real array lambda_{k,m} = 4 pi^2 (k^2/Lx^2 + m^2/Ly^2) of shape
    (Mx, My), with the signed wavenumber indices in numpy fft storage order.
    Raise ValueError if either argument is a sequence whose length is not
    2, if mesh is not a positive whole number in both directions, or if a
    cell edge length is not positive and finite."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _pair(v):
    a = np.atleast_1d(np.asarray(v, float))
    return (float(a.flat[0]), float(a.flat[-1]))


def _oracle_sqpfc_laplacian_symbol(mesh: "int | Sequence[int]",
                                   cell: "float | Sequence[float]") -> "np.ndarray":
    """Fourier symbol of -Delta_h on the signed index set."""
    n = np.atleast_1d(np.asarray(mesh, float))
    c = np.atleast_1d(np.asarray(cell, float))
    if n.size not in (1, 2) or c.size not in (1, 2):
        raise ValueError("mesh and cell must be scalars or length-2 sequences")
    if np.any(n < 1.0) or np.any(n != np.round(n)):
        raise ValueError("mesh must contain positive whole numbers")
    if np.any(c <= 0.0) or not np.all(np.isfinite(c)):
        raise ValueError("cell edge lengths must be positive and finite")
    Mx, My = (int(round(t)) for t in _pair(mesh))
    Lx, Ly = _pair(cell)
    kx = (np.fft.fftfreq(Mx) * Mx).reshape(Mx, 1)
    ky = (np.fft.fftfreq(My) * My).reshape(1, My)
    return (4.0 * float(np.pi) ** 2
            * ((kx / Lx) ** 2 + (ky / Ly) ** 2) * np.ones((Mx, My)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        {"setup": "",
         "call": "sqpfc_laplacian_symbol(32, 8.0 * np.pi)",
         "gold_call": "_oracle_sqpfc_laplacian_symbol(32, 8.0 * np.pi)"},
        {"setup": "",
         "call": "sqpfc_laplacian_symbol((24, 40), (6.0 * np.pi, 10.0 * np.pi))",
         "gold_call": "_oracle_sqpfc_laplacian_symbol((24, 40), (6.0 * np.pi, 10.0 * np.pi))"},
        {"setup": "",
         "call": "sqpfc_laplacian_symbol((48, 24), (32.0, 20.0))",
         "gold_call": "_oracle_sqpfc_laplacian_symbol((48, 24), (32.0, 20.0))"},
        {"setup": "",
         "call": "sqpfc_laplacian_symbol((40, 24), (10.0 * np.pi, 4.0 * np.pi))",
         "gold_call": "_oracle_sqpfc_laplacian_symbol((40, 24), (10.0 * np.pi, 4.0 * np.pi))"},
        # a strongly anisotropic cell: hx = 0.25, hy = 2.0
        {"setup": "",
         "call": "sqpfc_laplacian_symbol((16, 8), (4.0, 16.0))",
         "gold_call": "_oracle_sqpfc_laplacian_symbol((16, 8), (4.0, 16.0))"},
        {"setup": "",
         "call": "sqpfc_laplacian_symbol(2, (1.0, 3.0))",
         "gold_call": "_oracle_sqpfc_laplacian_symbol(2, (1.0, 3.0))"},
        # contract: a length-3 cell is not a rectangle, so the call must raise
        # ValueError rather than quietly use two of the three entries. The trap
        # catches ValueError ONLY, so returning an array fails the case.
        {"setup": ("def trap(f):\n"
                  "    try:\n"
                  "        return float(np.asarray(f()).ravel()[0])\n"
                  "    except ValueError:\n"
                  "        return -12345.0\n"),
         "call": "trap(lambda: sqpfc_laplacian_symbol((16, 16), (4.0, 8.0, 12.0)))",
         "gold_call": ("trap(lambda: _oracle_sqpfc_laplacian_symbol("
                       "(16, 16), (4.0, 8.0, 12.0)))")},
        # contract: a fractional mode count has no signed index set.
        {"setup": ("def trap(f):\n"
                  "    try:\n"
                  "        return float(np.asarray(f()).ravel()[0])\n"
                  "    except ValueError:\n"
                  "        return -12345.0\n"),
         "call": "trap(lambda: sqpfc_laplacian_symbol((16.5, 16), 4.0))",
         "gold_call": ("trap(lambda: _oracle_sqpfc_laplacian_symbol("
                       "(16.5, 16), 4.0))")},
    ]
