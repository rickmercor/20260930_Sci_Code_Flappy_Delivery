"""
Return the array of squared Fourier wave numbers $|\\boldsymbol{k}|^2$ on a periodic **rectangular** two-dimensional box, in `numpy` FFT mode order. Two things are general here and both are graded: $N$ is either an `int` or a length-2 sequence $(N_x, N_y)$, and $L$ is either a scalar or a length-2 sequence $(L_x, L_y)$, so direction $a$ carries its own mode count $N_a$ *and* its own box length $L_a$, and the two are not interchangeable. Every spectral operator in the rest of the pipeline - the Laplacian, the biharmonic and tri-Laplacian operators inside the implicit symbol, the inverse Laplacian of the $H^{-1}$ norm, and the two quadratic forms of the free energy - is built from this one array.

On a periodic box of length $L_a$ discretised by $N_a$ equally spaced points with spacing $h_a = L_a/N_a$, the discrete Fourier modes are $e^{\\mathrm{i}k_ax_a}$ with $k_a = 2\\pi p_a/L_a$ and $p_a$ an integer frequency index. In `numpy` that index runs $0, 1, \\dots, N_a/2-1, -N_a/2, \\dots, -1$, i.e. exactly `np.fft.fftfreq(N_a) * N_a`, so the wave numbers are `2*np.pi*np.fft.fftfreq(N_a, d=h_a)`. Differentiation becomes multiplication: $\\partial_a \\to \\mathrm{i}k_a$, $\\nabla^2 \\to -|\\boldsymbol{k}|^2$, $\\nabla^4 \\to |\\boldsymbol{k}|^4$, $\\nabla^6 \\to -|\\boldsymbol{k}|^6$. None of this survives pairing $N_a$ with the wrong $L_b$: the resulting array has the right shape and the wrong physics.




 **Formulas**



With $h_a = L_a/N_a$ and $p_a$ the `fftfreq` integer indices,




$$k_a[p_a] = \\frac{2\\pi p_a}{L_a} = 2\\pi\\,\\mathrm{fftfreq}(N_a, h_a), \\qquad |\\boldsymbol{k}|^2[p_x,p_y] = k_x[p_x]^2 + k_y[p_y]^2 ,$$




the two one-dimensional vectors being broadcast along their own axis only, so the result has shape $(N_x, N_y)$. The zero mode is exactly $0$ and every entry is $\\ge 0$.

Returns
-------
`np.ndarray` of shape `(Nx, Ny)`, real, all entries $\\ge 0$, with `ksq[0, 0]` exactly $0$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectral_wavenumbers(N: "int | Sequence[int]",
                         L: "float | Sequence[float]") -> "np.ndarray":
    """N: int, or length-2 sequence (Nx, Ny), the per-direction mode counts.
    L: float, or length-2 sequence (Lx, Ly), the per-direction box lengths.
    Return the real array of squared Fourier wave numbers |k|^2 in numpy FFT
    mode order, shape (Nx, Ny)."""
    # Implement per the formulas above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_spectral_wavenumbers(N: "int | Sequence[int]",
                                 L: "float | Sequence[float]") -> "np.ndarray":
    _N = np.atleast_1d(np.asarray(N))
    if _N.size not in (1, 2):
        raise ValueError("N must be an integer or a length-2 sequence (Nx, Ny)")
    if not all(float(v).is_integer() for v in _N.ravel()):
        raise ValueError("every grid size must be an integer")
    nv = np.broadcast_to(_N.astype(float), (2,)).astype(int)
    if np.any(nv < 2):
        raise ValueError("every grid size must be >= 2")
    _L = np.atleast_1d(np.asarray(L, float))
    if _L.size not in (1, 2):
        raise ValueError("L must be a scalar or a length-2 vector of box lengths")
    if not np.all(np.isfinite(_L)) or np.any(_L <= 0.0):
        raise ValueError("all box lengths must be finite and positive")
    Lv = np.broadcast_to(np.asarray(L, float), (2,))
    k = [2.0 * np.pi * np.fft.fftfreq(int(nv[a]), d=float(Lv[a]) / int(nv[a]))
         for a in range(2)]
    return k[0][:, None] ** 2 + k[1][None, :] ** 2


def _geom(phi, L):
    nv = phi.shape
    Lv = np.broadcast_to(np.asarray(L, float), (2,))
    hx, hy = float(Lv[0]) / nv[0], float(Lv[1]) / nv[1]
    return nv, Lv, hx, hy

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: the production grid and box of the benchmark instance.
        {"setup": "import numpy as np\nN = 64\nL = 32.0 * np.pi\n",
         "call": "spectral_wavenumbers(N, L)",
         "gold_call": "_oracle_spectral_wavenumbers(N, L)"},
        # normal: an irrational box length no hard-coded 2*pi/N reproduces.
        {"setup": "import numpy as np\nN = 32\nL = 12.0 * np.pi\n",
         "call": "spectral_wavenumbers(N, L)",
         "gold_call": "_oracle_spectral_wavenumbers(N, L)"},
        # boundary: a RECTANGULAR grid with anisotropic box lengths. Nx, Ny and
        # Lx, Ly all differ, so each axis carries its own mode count AND its own
        # length; an implementation that builds one vector and reuses it fails.
        {"setup": "import numpy as np\nNv = (24, 32)\n"
                  "Lv = (20.0 * np.pi, 32.0 * np.pi)\n",
         "call": "spectral_wavenumbers(Nv, Lv)",
         "gold_call": "_oracle_spectral_wavenumbers(Nv, Lv)"},
        # boundary: the same lengths with permuted mode counts. Pairing N_a with
        # the wrong L_a gives a same-shaped, wrong array.
        {"setup": "import numpy as np\nNv = (32, 24)\n"
                  "Lv = (20.0 * np.pi, 32.0 * np.pi)\n",
         "call": "spectral_wavenumbers(Nv, Lv)",
         "gold_call": "_oracle_spectral_wavenumbers(Nv, Lv)"},
        # boundary: scalar L with a rectangular N, which must broadcast the one
        # length to both directions while keeping the two mode counts distinct.
        {"setup": "import numpy as np\nNv = (10, 14)\n",
         "call": "spectral_wavenumbers(Nv, 15.0)",
         "gold_call": "_oracle_spectral_wavenumbers(Nv, 15.0)"},
        # edge: the smallest sensible periodic mesh, where every index is 0 or -1.
        {"setup": "import numpy as np\nNv = (2, 3)\nLv = (1.0, 4.0)\n",
         "call": "spectral_wavenumbers(Nv, Lv)",
         "gold_call": "_oracle_spectral_wavenumbers(Nv, Lv)"},
        # edge: odd mode counts, where the Nyquist mode does not exist and the
        # fftfreq index set is not symmetric in the usual way.
        {"setup": "import numpy as np\nNv = (7, 9)\nLv = (0.8, 3.3)\n",
         "call": "spectral_wavenumbers(Nv, Lv)",
         "gold_call": "_oracle_spectral_wavenumbers(Nv, Lv)"},
    ]
