"""
Return the squared $H^{-1}$ norm $\\|v\\|_{-1}^2=\\int_\\Omega v\\,(-\\Delta)^{-1}v$ of a mean-zero field on the periodic rectangular box, evaluated spectrally. The inverse Laplacian is defined only up to a constant and only on mean-zero data, so the zero Fourier mode must be **removed from the input and excluded from the sum**; the function must therefore return the same value for $v$ and for $v+c$. This norm carries the leading term of the modified discrete energy whose dissipation the scheme guarantees, and it is the norm in which the variable-step BDF2 difference operator is tested against the increment.

For $v$ with zero mean on a periodic domain, let $\\psi$ solve $-\\Delta\\psi=v$ with zero mean. Then $\\|v\\|_{-1}^2=(v,\\psi)=\\|\\nabla\\psi\\|^2$, and in Fourier space $\\hat\\psi=\\hat v/|\\boldsymbol{k}|^2$ for $\\boldsymbol{k}\\ne0$. The norm therefore weights each mode by $|\\boldsymbol{k}|^{-2}$, which is the sense in which it is weaker than $L^2$: it discounts short wavelengths. Conserved gradient flows are gradient flows *in this norm*, which is why the energy estimates for Cahn-Hilliard-type and PFC-type schemes are carried out in it rather than in $L^2$.




 **Formulas**



With $\\kappa=|\\boldsymbol{k}|^2$ from step 1, $\\tilde v=v-\\overline{v}$ its mean-zero part, $\\hat{\\tilde v}=\\mathrm{fft2}(\\tilde v)$ and $w=h_xh_y/(N_xN_y)$,




$$\\|v\\|_{-1}^2 \\;=\\; w\\sum_{\\boldsymbol{p}\\,:\\,\\kappa>0}\\frac{|\\hat{\\tilde v}[\\boldsymbol{p}]|^2}{\\kappa[\\boldsymbol{p}]},$$




the $\\boldsymbol{p}=\\boldsymbol{0}$ term being omitted rather than divided by zero. The result is real and non-negative, and is unchanged by adding any constant to $v$.

Returns
-------
A Python `float`, real and $\\ge 0$, invariant under $v \\mapsto v + c$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def h_minus1_norm_sq(v: "np.ndarray",
                     L: "float | Sequence[float]") -> float:
    """v: real 2-D array of shape (Nx, Ny).
    L: float, or length-2 sequence (Lx, Ly), the per-direction box lengths.
    Return the squared H^{-1} norm of v as a float, with the zero Fourier mode
    removed from v and excluded from the sum."""
    # Implement per the formulas above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_h_minus1_norm_sq(v: "np.ndarray", L: "float | Sequence[float]") -> float:
    v = np.asarray(v, float)
    if v.ndim != 2:
        raise ValueError("v must be a two-dimensional array")
    nv, Lv, hx, hy = _geom(v, L)
    ksq = _oracle_spectral_wavenumbers(nv, Lv)
    vh = np.fft.fft2(v - v.mean())
    acc = np.zeros_like(ksq)
    nz = ksq > 0.0
    acc[nz] = np.abs(vh[nz]) ** 2 / ksq[nz]
    return float(hx * hy / (nv[0] * nv[1]) * np.sum(acc))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: a single Fourier mode, whose H^{-1} norm is analytic:
        # for v = a cos(2 pi m x / Lx) the value is a^2 |Omega| / (2 k^2).
        {"setup": "import numpy as np\nN = 32\nL = 8.0\n"
                  "x = np.arange(N) * (L / N)\n"
                  "v = 0.7 * np.cos(2 * np.pi * 3 * x / L)[:, None] "
                  "* np.ones(N)[None, :]\n",
         "call": "h_minus1_norm_sq(v, L)",
         "gold_call": "_oracle_h_minus1_norm_sq(v, L)"},
        # normal: a multi-mode field on the production box.
        {"setup": "import numpy as np\nN = 64\nL = 32.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "v = (0.4 * np.cos(16 * u + 0.3)[:, None] * np.cos(9 * u)[None, :]\n"
                  "     + 0.2 * np.sin(5 * u)[:, None] * np.sin(11 * u + 0.9)[None, :])\n",
         "call": "h_minus1_norm_sq(v, L)",
         "gold_call": "_oracle_h_minus1_norm_sq(v, L)"},
        # boundary: the SAME field shifted by a large constant. The zero mode must
        # be projected out, so the answer must be identical to the previous case;
        # an implementation that forgets the projection returns a huge number or
        # divides by zero.
        {"setup": "import numpy as np\nN = 64\nL = 32.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "v = (0.4 * np.cos(16 * u + 0.3)[:, None] * np.cos(9 * u)[None, :]\n"
                  "     + 0.2 * np.sin(5 * u)[:, None] * np.sin(11 * u + 0.9)[None, :]\n"
                  "     - 12.5)\n",
         "call": "h_minus1_norm_sq(v, L)",
         "gold_call": "_oracle_h_minus1_norm_sq(v, L)"},
        # boundary: a RECTANGULAR grid on an anisotropic box, so the two wave-number
        # vectors and the volume element all differ.
        {"setup": "import numpy as np\nNv = (20, 28)\nLv = (5.0, 13.0)\n"
                  "a = np.arange(Nv[0]) * (2 * np.pi / Nv[0])\n"
                  "b = np.arange(Nv[1]) * (2 * np.pi / Nv[1])\n"
                  "v = np.cos(2 * a)[:, None] * np.sin(3 * b)[None, :]\n",
         "call": "h_minus1_norm_sq(v, Lv)",
         "gold_call": "_oracle_h_minus1_norm_sq(v, Lv)"},
        # boundary: a field concentrated at the highest resolvable wave numbers,
        # where the 1/k^2 weight is smallest.
        {"setup": "import numpy as np\nN = 16\ni = np.arange(N)\n"
                  "v = ((-1.0) ** i)[:, None] * ((-1.0) ** i)[None, :]\n",
         "call": "h_minus1_norm_sq(v, 4.0)",
         "gold_call": "_oracle_h_minus1_norm_sq(v, 4.0)"},
        # edge: an exactly constant field, whose mean-zero part vanishes, so the
        # answer is exactly zero rather than a division by zero.
        {"setup": "import numpy as np\nv = np.full((12, 12), -0.27)\n",
         "call": "h_minus1_norm_sq(v, 3.0)",
         "gold_call": "_oracle_h_minus1_norm_sq(v, 3.0)"},
        # edge: odd mode counts, where the fftfreq index set is asymmetric.
        {"setup": "import numpy as np\nNv = (7, 9)\nLv = (1.5, 6.0)\n"
                  "a = np.arange(Nv[0]) * (2 * np.pi / Nv[0])\n"
                  "b = np.arange(Nv[1]) * (2 * np.pi / Nv[1])\n"
                  "v = np.sin(a)[:, None] + np.cos(2 * b)[None, :]\n",
         "call": "h_minus1_norm_sq(v, Lv)",
         "gold_call": "_oracle_h_minus1_norm_sq(v, Lv)"},
    ]
