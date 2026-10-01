"""
Evaluate, by the Fourier spectral method on a uniform periodic grid, five spatial operators applied to one real periodic field $u$: the two first partial derivatives $\\partial_x u$ and $\\partial_y u$, the Laplacian $\\Delta u$, the bi-Laplacian $\\Delta^2 u$, and the ****shifted bi-Laplacian**** $(\\Delta+a)^2u$ for a real shift $a$. Return all five stacked along a new leading axis, in that order, so that the shape is `(5,) + u.shape`. Every operator is applied by transforming, multiplying by its symbol, transforming back and taking the real part; ****no dealiasing of any kind is applied**** and the Nyquist wavenumber is kept exactly as `numpy.fft.fftfreq` returns it. The shifted bi-Laplacian is graded separately from the bi-Laplacian because the whole interspecies coupling of the model rides on it, and because expanding it as $\\Delta^2+a^2$ - dropping its two cross terms - is the single most tempting simplification in the pipeline.

On a periodic box of edge lengths $(L_x,L_y)$ discretised by $N_x\\times N_y$ equispaced points, the discrete Fourier transform diagonalises every constant-coefficient differential operator: $\\partial_x\\mapsto i k_x$, $\\Delta\\mapsto -(k_x^2+k_y^2)$, and any polynomial in $\\Delta$ maps to the same polynomial in $-|k|^2$. The wavenumbers are $k_x = 2\\pi\\,\\texttt{fftfreq}(N_x, L_x/N_x)$ and likewise in $y$, so they run over $2\\pi m/L_x$ for $m$ in the signed range that `fftfreq` produces, with the Nyquist entry carried as the negative value. The sign carried at the Nyquist entry is in fact immaterial: for a real field the Nyquist Fourier coefficient is real, multiplying it by $\\pm i k_{\\mathrm{Nyq}}$ makes it imaginary, and taking the real part discards it either way, so **the first-derivative operator annihilates the Nyquist mode** while every even-order operator here keeps it. That asymmetry is harmless inside this step but has a consequence two steps later: on a field with Nyquist content the integral of the squared pointwise first derivatives is strictly smaller than $-(u,\\Delta u)$, so the two continuum-equivalent readings of a squared gradient norm are not equal on the grid. Because the model's quadratic energy is built from $(\\Delta+a)^2$, whose symbol $(a-|k|^2)^2$ vanishes on the circle $|k|^2=a$, every operator here is fourth order in the wavenumber and an implementation that confuses $\\Delta$ with $\\Delta^2$ is wrong by two powers of $|k|$ at every mode.

**--- Formulas ---**

Each operator is applied the same way: transform, multiply by that operator's symbol in the wavenumbers above, transform back, take the real part. Two points are conventions rather than derivations, and both are graded: the wavenumber grids are `2*np.pi*np.fft.fftfreq(N, d=L/N)` in each direction with the Nyquist entry left exactly as returned, and **no dealiasing of any kind is applied**. One point is a definition worth stating twice: the shifted bi-Laplacian is the square of the operator $\\Delta+a$, so it carries that square's two cross terms and is **not** $\\Delta^2u+a^2u$.

Returns
-------
`np.ndarray` of shape `(5,) + u.shape`, real and finite. For a constant field the first four entries are exactly $0$ and the fifth is $a^2u$; for a single Fourier mode each entry is that mode times the corresponding symbol.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def pfc_spectral_derivatives(u, cell, a=0.0):
    """u: real periodic field on a uniform grid, shape (Nx, Ny).
    cell: domain edge lengths, a scalar or a length-2 sequence (Lx, Ly).
    a: real shift of the shifted bi-Laplacian (Delta + a)^2.
    Return the real array [du/dx, du/dy, lap u, bilap u, (Delta+a)^2 u]
    of shape (5,) + u.shape."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

# ----------------------------------------------------------------------------
# shared fixtures
# ----------------------------------------------------------------------------

PFC_PARAMS = dict(eps=0.1, M_phi=1.0, M_m=1.0e-2, eta=10.0, gamma12=0.5,
                  omega0=1.0, theta=1.0e-9, theta1=1.0e-9, theta2=1.0e-9,
                  alpha=1.0, beta=10.0, S_phi=10.0, S_m=10.0,
                  a1=1.0, a2=1.0, a12=1.2, B=1.0e7,
                  gamma1=0.01, gamma2=0.01, eta1=-0.1, eta2=-0.1)

_KEYS = ("eps", "M_phi", "M_m", "eta", "gamma12", "omega0", "theta", "theta1",
         "theta2", "alpha", "beta", "S_phi", "S_m", "a1", "a2", "a12", "B",
         "gamma1", "gamma2", "eta1", "eta2")


def _check_params(params):
    if params is None:
        return dict(PFC_PARAMS)
    p = dict(PFC_PARAMS)
    for k in params:
        if k not in _KEYS:
            raise ValueError("unknown model parameter %r" % (k,))
    p.update({k: float(v) for k, v in params.items()})
    if not all(np.isfinite(v) for v in p.values()):
        raise ValueError("model parameters must be finite")
    if p["B"] <= 0.0:
        raise ValueError("B must be positive")
    if p["M_phi"] <= 0.0 or p["M_m"] <= 0.0:
        raise ValueError("the mobilities must be positive")
    if p["omega0"] < 0.0 or p["beta"] < 0.0:
        raise ValueError("omega0 and beta must be non-negative")
    return p


def _check_cell(cell, shape):
    L = np.atleast_1d(np.asarray(cell, float)).ravel()
    if L.size == 1:
        L = np.full(len(shape), float(L[0]))
    if L.size != len(shape):
        raise ValueError("cell must be a scalar or match the field rank")
    if not np.all(np.isfinite(L)) or np.any(L <= 0.0):
        raise ValueError("cell edge lengths must be finite and positive")
    return L


def _wavenumbers(shape, L):
    return [2.0 * np.pi * np.fft.fftfreq(n, d=Ln / n) for n, Ln in zip(shape, L)]


def _k2(shape, L):
    ks = _wavenumbers(shape, L)
    return sum(k.reshape([-1 if i == j else 1 for j in range(len(shape))]) ** 2
               for i, k in enumerate(ks))


def _ip(f, g, L):
    """Discrete L2 inner product on a uniform periodic grid."""
    dv = np.prod(L) / np.prod(f.shape[-len(L):])
    return float(np.sum(f * g) * dv)


def _oracle_pfc_spectral_derivatives(u, cell, a=0.0):
    u = np.asarray(u, float)
    if u.ndim != 2:
        raise ValueError("u must be a two-dimensional periodic field")
    if not np.all(np.isfinite(u)):
        raise ValueError("u must be finite")
    L = _check_cell(cell, u.shape)
    a = float(a)
    if not np.isfinite(a):
        raise ValueError("a must be finite")
    kx, ky = _wavenumbers(u.shape, L)
    KX, KY = kx[:, None], ky[None, :]
    k2 = KX ** 2 + KY ** 2
    uh = np.fft.fft2(u)
    ux = np.real(np.fft.ifft2(1j * KX * uh))
    uy = np.real(np.fft.ifft2(1j * KY * uh))
    lap = np.real(np.fft.ifft2(-k2 * uh))
    bih = np.real(np.fft.ifft2(k2 ** 2 * uh))
    shf = np.real(np.fft.ifft2((a - k2) ** 2 * uh))
    return np.stack([ux, uy, lap, bih, shf])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    FLD = ("N = 24\n"
           "Lc = (32.0, 32.0)\n"
           "x = np.arange(N) * (Lc[0] / N)\n"
           "y = np.arange(N) * (Lc[1] / N)\n"
           "X, Y = np.meshgrid(x, y, indexing='ij')\n"
           "k = 2.0 * np.pi / 32.0\n"
           "phi1 = 0.4 * np.cos(3 * k * X) * np.sin(2 * k * Y) + 0.10\n"
           "phi2 = 0.3 * np.cos(2 * k * X) * np.cos(3 * k * Y) - 0.05\n"
           "Mf = np.stack([0.5 * np.sin(k * X) * np.sin(2 * k * Y),\n"
           "               0.4 * np.cos(2 * k * X) * np.cos(k * Y)])\n"
           "Hf = np.stack([np.sin(2 * k * X) * np.cos(k * Y), np.cos(k * Y)])\n")

    STIFF = "PS = {'gamma1': 1.0, 'gamma2': 1.0, 'eta1': -100.0, 'eta2': -100.0}\n"
    return [
        # normal: a smooth two-mode field, every derivative exercised at once.
        {"setup": "import numpy as np\n" + FLD,
         "call": "pfc_spectral_derivatives(phi1, Lc)",
         "gold_call": "_oracle_pfc_spectral_derivatives(phi1, Lc)"},
        # normal: the shifted bi-Laplacian with the interspecies length scale.
        {"setup": "import numpy as np\n" + FLD,
         "call": "pfc_spectral_derivatives(phi2, Lc, 1.2)",
         "gold_call": "_oracle_pfc_spectral_derivatives(phi2, Lc, 1.2)"},
        # boundary: a pure single mode, where every operator has a closed form,
        # on a RECTANGULAR box so that kx and ky cannot be interchanged.
        {"setup": "import numpy as np\n"
                  "x = np.arange(16) * (8.0 / 16)\ny = np.arange(32) * (20.0 / 32)\n"
                  "X, Y = np.meshgrid(x, y, indexing='ij')\n"
                  "u = np.cos(2 * np.pi * X / 8.0) * np.sin(4 * np.pi * Y / 20.0)\n",
         "call": "pfc_spectral_derivatives(u, (8.0, 20.0), 0.7)",
         "gold_call": "_oracle_pfc_spectral_derivatives(u, (8.0, 20.0), 0.7)"},
        # boundary: a scalar cell, which must broadcast to both directions.
        {"setup": "import numpy as np\n" + FLD,
         "call": "pfc_spectral_derivatives(phi1 * phi2, 32.0, -0.4)",
         "gold_call": "_oracle_pfc_spectral_derivatives(phi1 * phi2, 32.0, -0.4)"},
        # edge: a constant field - every derivative is exactly zero except the
        # shifted bi-Laplacian, which returns a^2 times the constant.
        {"setup": "import numpy as np\nu = np.full((8, 8), 2.5)\n",
         "call": "pfc_spectral_derivatives(u, 10.0, 1.2)",
         "gold_call": "_oracle_pfc_spectral_derivatives(u, 10.0, 1.2)"},
        # edge: a field carrying the Nyquist mode in x, which the two first
        # derivatives annihilate and the three even-order operators keep.
        {"setup": "import numpy as np\n"
                  "x = np.arange(16) * (16.0 / 16)\ny = np.arange(16) * (16.0 / 16)\n"
                  "X, Y = np.meshgrid(x, y, indexing='ij')\n"
                  "u = np.cos(np.pi * X) + 0.5 * np.sin(2 * np.pi * Y / 16.0)\n",
         "call": "pfc_spectral_derivatives(u, 16.0)",
         "gold_call": "_oracle_pfc_spectral_derivatives(u, 16.0)"},
    ]
