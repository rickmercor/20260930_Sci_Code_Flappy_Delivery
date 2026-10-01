"""
Evaluate, by the Fourier spectral method on a uniform periodic grid, a fixed list of spatial operators applied to one real periodic field $u$. The field is $d$-dimensional for ****any**** $d\\ge1$, not two-dimensional only, and the list is: the $d$ first partial derivatives $\\partial_{x_0}u,\\dots,\\partial_{x_{d-1}}u$ in axis order, then the Laplacian $\\Delta u$, the bi-Laplacian $\\Delta^2u$, the ****shifted bi-Laplacian**** $(\\Delta+1)^2u$, and the ****mean-free inverse Laplacian**** $\\chi[u]$, defined as the unique periodic solution of $-\\Delta\\chi = u-\\bar u$ with $\\int_\\Omega\\chi = 0$, where $\\bar u$ is the spatial mean of $u$. Return them all stacked along a new leading axis, in that order, so that the shape is `(d + 4,) + u.shape` - six planes when $d=2$, five when $d=1$, seven when $d=3$. Each operator is applied by transforming, multiplying by its symbol, transforming back and taking the real part; ****no dealiasing of any kind is applied**** and the Nyquist wavenumber is kept exactly as `numpy.fft.fftfreq` returns it, which matters on every axis with an even number of points. The last two entries are graded separately from the derivatives because the whole model rides on them: the shifted bi-Laplacian is the operator whose symbol is degenerate on a sphere of wavenumbers, and the mean-free inverse Laplacian is the only operator here whose symbol has to be defined by hand at $k=0$.

On a periodic box of edge lengths $(L_x,L_y)$ discretised by $N_x\\times N_y$ equispaced points, the discrete Fourier transform diagonalises every constant-coefficient differential operator: $\\partial_x\\mapsto ik_x$, $\\Delta\\mapsto-(k_x^2+k_y^2)$, and any polynomial in $\\Delta$ maps to the same polynomial in $-|k|^2$. The wavenumbers are $k_x=2\\pi\\,\\texttt{fftfreq}(N_x,\\,L_x/N_x)$ and likewise in $y$, so they run over $2\\pi m/L_x$ for $m$ in the signed range `fftfreq` produces, with the Nyquist entry carried as the negative value. The sign at Nyquist is immaterial here: for a real field the Nyquist Fourier coefficient is real, multiplying it by $\\pm ik_{\\mathrm{Nyq}}$ makes it imaginary, and taking the real part discards it either way, so ****the first-derivative operator annihilates the Nyquist mode**** while every even-order operator keeps it. Two consequences matter downstream. First, the two continuum-equivalent readings of a squared gradient norm - the integral of the squared pointwise first derivatives, and minus the inner product with the Laplacian - are not equal on the grid once a field has Nyquist content. Second, the symbol $|k|^{-2}$ of the inverse Laplacian is singular at $k=0$; the equation $-\\Delta\\chi=u-\\bar u$ is solvable precisely because the right-hand side has zero mean, and the solution is fixed only up to a constant, so the zero mode of $\\hat\\chi$ must be ****set to zero****. Setting it to zero also performs the mean subtraction automatically, since the zero mode of $\\hat u$ is the only place $\\bar u$ lives.




****--- Formulas ---****

Each operator is applied the same way: transform, multiply by that operator's symbol in the wavenumbers above, transform back, take the real part. The symbols are $ik_0,\\dots,ik_{d-1}$ for the $d$ derivatives, in axis order, then $-|k|^2$, $|k|^4$, $(1-|k|^2)^2$ and, for the last entry, $|k|^{-2}$ at every $k\\neq0$ together with the value $0$ at $k=0$. Two points are conventions rather than derivations, and both are graded: the wavenumber grids are `2*np.pi*np.fft.fftfreq(N, d=L/N)` - one per axis, each built from that axis's own point count and edge length and broadcast against the others, so that $|k|^2$ is their sum of squares - with the Nyquist entry left exactly as returned, and ****no dealiasing of any kind is applied****. One point is a definition worth stating twice: the shifted bi-Laplacian is the square of the operator $\\Delta+1$, so it carries that square's two cross terms and its symbol is $(1-|k|^2)^2$, ****not**** $|k|^4+1$.

Returns
-------
`np.ndarray` of shape `(d + 4,) + u.shape` with $d=$ `u.ndim`, real and finite. For a constant field every entry except the shifted bi-Laplacian is exactly $0$ and that one is the field itself; for a single Fourier mode each entry is that mode times the corresponding symbol. The last entry always has exactly zero spatial mean.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mpfc_spectral_operators(u: "np.ndarray", cell: "float | tuple") -> "np.ndarray":
    """u: real periodic field on a uniform grid of ANY dimension d >= 1,
       shape (N_0, ..., N_{d-1}).
    cell: domain edge lengths, a scalar or a length-d sequence.
    Return the real array holding, in order, the d first partial
    derivatives of u in axis order, then lap u, bilap u, (Delta+1)^2 u
    and chi, of shape (d + 4,) + u.shape, where chi is the mean-free
    periodic solution of -lap chi = u - mean(u).
    Raise ValueError if u is not a finite real array of at least one
    dimension with at least two points along every axis, or if cell is
    not positive, or is a sequence whose length is not u.ndim."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

# ---------------------------------------------------------------------------
# shared fixtures (step 1 block)
# ---------------------------------------------------------------------------

def _check_params(params):
    """Default model parameters, kept inside the function so that field
    extraction cannot separate them from their only user."""
    p = dict(M=10.0, eps=0.4, alpha=0.25, C_sav=400.0)
    if params is not None and not isinstance(params, dict):
        raise ValueError("params must be a dict of overrides, or None")
    if params is not None:
        for k in params:
            if k not in p:
                raise ValueError("unknown model parameter %r" % (k,))
        p.update({k: float(v) for k, v in params.items()})
    if not all(np.isfinite(v) for v in p.values()):
        raise ValueError("model parameters must be finite")
    if p["M"] <= 0.0:
        raise ValueError("the mobility M must be positive")
    if p["alpha"] < 0.0:
        raise ValueError("alpha must be non-negative")
    return p


def _check_coef(coef):
    """Prescribed member of the source's coefficient family, kept inside the
    function for the same reason."""
    c = dict(a11t=0.5, a32t=-0.25, a33t=1.5, a31=1.0 / 3.0, a43=2.0 / 3.0)
    if coef is not None and not isinstance(coef, dict):
        raise ValueError("coef must be a dict of overrides, or None")
    if coef is not None:
        for k in coef:
            if k not in c:
                raise ValueError("unknown Runge-Kutta coefficient %r" % (k,))
        c.update({k: float(v) for k, v in coef.items()})
    if not all(np.isfinite(v) for v in c.values()):
        raise ValueError("Runge-Kutta coefficients must be finite")
    if c["a11t"] == 0.0 or c["a33t"] == 0.0 or c["a43"] == 0.0:
        raise ValueError("a11t, a33t and a43 must be nonzero")
    return c


def _check_cell(cell, shape):
    L = np.atleast_1d(np.asarray(cell, float)).ravel()
    if L.size == 1:
        L = np.full(len(shape), float(L[0]))
    if L.size != len(shape):
        raise ValueError("cell must be a scalar or match the field rank")
    if not np.all(np.isfinite(L)) or np.any(L <= 0.0):
        raise ValueError("cell edge lengths must be finite and positive")
    return L


def _check_field(u, name="u"):
    if np.asarray(u).dtype.kind == "c":
        raise ValueError("%s must be a real field, not a complex one" % name)
    u = np.asarray(u, float)
    if u.ndim < 1:
        raise ValueError("%s must be a periodic field of at least one dimension" % name)
    if min(u.shape) < 2:
        raise ValueError("%s must have at least two points along every axis" % name)
    if not np.all(np.isfinite(u)):
        raise ValueError("%s must be finite" % name)
    return u


def _wavenumbers(shape, L):
    return [2.0 * np.pi * np.fft.fftfreq(n, d=Ln / n) for n, Ln in zip(shape, L)]


def _kgrids(shape, L):
    """The d wavenumber axes, each shaped so that it broadcasts over the field."""
    ks = _wavenumbers(shape, L)
    d = len(shape)
    out = []
    for i, k in enumerate(ks):
        sh = [1] * d
        sh[i] = len(k)
        out.append(k.reshape(sh))
    return out


def _k2(shape, L):
    return sum(k ** 2 for k in _kgrids(shape, L))


def _ip(f, g, L):
    """Discrete L2 inner product: plain grid sum times the cell area."""
    dv = float(np.prod(L)) / float(np.prod(f.shape))
    return float(np.sum(f * g) * dv)


def _oracle_mpfc_spectral_operators(u: "np.ndarray", cell: "float | tuple") -> "np.ndarray":
    u = _check_field(u)
    L = _check_cell(cell, u.shape)
    ks = _kgrids(u.shape, L)
    k2 = sum(k ** 2 for k in ks)
    uh = np.fft.fftn(u)
    planes = [np.real(np.fft.ifftn(1j * k * uh)) for k in ks]
    planes.append(np.real(np.fft.ifftn(-k2 * uh)))
    planes.append(np.real(np.fft.ifftn(k2 ** 2 * uh)))
    planes.append(np.real(np.fft.ifftn((1.0 - k2) ** 2 * uh)))
    invh = np.zeros_like(uh)
    nz = np.broadcast_to(k2, u.shape) > 0.0
    invh[nz] = uh[nz] / np.broadcast_to(k2, u.shape)[nz]
    planes.append(np.real(np.fft.ifftn(invh)))
    return np.stack(planes)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    FLD = ('N1, N2 = 24, 32\n'
           'Lc = (18.0, 24.0)\n'
           'x = np.arange(N1) * (Lc[0] / N1)\n'
           'y = np.arange(N2) * (Lc[1] / N2)\n'
           "X, Y = np.meshgrid(x, y, indexing='ij')\n"
           'kx = 2.0 * np.pi / Lc[0]\n'
           'ky = 2.0 * np.pi / Lc[1]\n'
           'phi = 0.15 + 0.30 * np.cos(3 * kx * X) * np.cos(2 * ky * Y) + 0.20 * np.sin(kx * X) * np.sin(3 * ky * Y)\n'
           'psi = 0.05 - 0.25 * np.cos(2 * kx * X) * np.sin(2 * ky * Y)\n')
    FLD1 = ('N1 = 20\n'
            'L1 = 7.0\n'
            'x1 = np.arange(N1) * (L1 / N1)\n'
            'k1 = 2.0 * np.pi / L1\n'
            'u1 = 0.15 + 0.30 * np.cos(2 * k1 * x1) + 0.20 * np.sin(3 * k1 * x1)\n')
    FLD3 = ('N3 = (8, 10, 6)\n'
            'L3 = (5.0, 7.0, 4.0)\n'
            'g3 = [np.arange(n) * (l / n) for n, l in zip(N3, L3)]\n'
            "X3, Y3, Z3 = np.meshgrid(*g3, indexing='ij')\n"
            'a3, b3, c3 = [2.0 * np.pi / l for l in L3]\n'
            'u3 = 0.15 + 0.25 * np.cos(a3 * X3) * np.cos(2 * b3 * Y3) + 0.20 * np.sin(2 * c3 * Z3) * np.cos(b3 * Y3)\n')
    FLDODD = ('No = (9, 11)\n'
              'Lo = (5.0, 7.0)\n'
              'xo = np.arange(No[0]) * (Lo[0] / No[0])\n'
              'yo = np.arange(No[1]) * (Lo[1] / No[1])\n'
              "Xo, Yo = np.meshgrid(xo, yo, indexing='ij')\n"
              'ko, lo = 2.0 * np.pi / Lo[0], 2.0 * np.pi / Lo[1]\n'
              'uo = 0.15 + 0.30 * np.cos(ko * Xo) * np.cos(2 * lo * Yo)\n')
    return [
        # normal: a smooth two-mode field on a rectangular box, every operator
        # exercised at once.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD,
         "call": "mpfc_spectral_operators(_dup(phi), _dup(Lc))",
         "gold_call": "_oracle_mpfc_spectral_operators(_dup(phi), _dup(Lc))"},
        # normal: a pointwise product, which carries higher wavenumbers than
        # either factor and so probes the aliased tail of every symbol.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD,
         "call": "mpfc_spectral_operators(phi * psi, _dup(Lc))",
         "gold_call": "_oracle_mpfc_spectral_operators(phi * psi, _dup(Lc))"},
        # boundary: a single mode on a NON-SQUARE grid over a RECTANGULAR box,
        # where kx and ky can neither be interchanged nor shared.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n"
                  "x = np.arange(16) * (10.0 / 16)\ny = np.arange(40) * (25.0 / 40)\n"
                  "X, Y = np.meshgrid(x, y, indexing='ij')\n"
                  "u = np.cos(4 * np.pi * X / 10.0) * np.sin(6 * np.pi * Y / 25.0) - 0.3\n",
         "call": "mpfc_spectral_operators(_dup(u), (10.0, 25.0))",
         "gold_call": "_oracle_mpfc_spectral_operators(_dup(u), (10.0, 25.0))"},
        # boundary: a scalar cell, which must broadcast to both directions.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD,
         "call": "mpfc_spectral_operators(_dup(psi), 18.0)",
         "gold_call": "_oracle_mpfc_spectral_operators(_dup(psi), 18.0)"},
        # edge: a constant field.  The four derivative entries are exactly
        # zero, the shifted bi-Laplacian returns the field itself, and the
        # inverse-Laplacian entry is exactly zero because the mean is removed.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\nu = np.full((8, 12), -0.7)\n",
         "call": "mpfc_spectral_operators(_dup(u), (6.0, 9.0))",
         "gold_call": "_oracle_mpfc_spectral_operators(_dup(u), (6.0, 9.0))"},
        # edge: a field carrying the Nyquist mode in x, which the two first
        # derivatives annihilate and every even-order operator keeps.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n"
                  "x = np.arange(16) * (16.0 / 16)\ny = np.arange(16) * (16.0 / 16)\n"
                  "X, Y = np.meshgrid(x, y, indexing='ij')\n"
                  "u = np.cos(np.pi * X) + 0.4 * np.sin(2 * np.pi * Y / 16.0)\n",
         "call": "mpfc_spectral_operators(_dup(u), 16.0)",
         "gold_call": "_oracle_mpfc_spectral_operators(_dup(u), 16.0)"},
        # edge: an axis with a single point, which no wavenumber grid can resolve
        # and which must raise ValueError; a one-dimensional field, by contrast,
        # is perfectly legal and is covered above.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n"
                  "def _guard(f, *a, **k):\n"
                  "    try:\n"
                  "        f(*a, **k)\n"
                  "    except ValueError:\n"
                  "        return 1.0\n"
                  "    return 0.0\n",
         "call": "_guard(mpfc_spectral_operators, np.zeros((1, 6)), 8.0)",
         "gold_call": "_guard(_oracle_mpfc_spectral_operators, np.zeros((1, 6)), 8.0)"},
        # edge: the same operators in ONE dimension, where the return has d+4 = 5 planes.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD1,
         "call": "mpfc_spectral_operators(_dup(u1), _dup(L1))",
         "gold_call": "_oracle_mpfc_spectral_operators(_dup(u1), _dup(L1))"},
        # edge: three dimensions, where the return has seven planes and the three
        # derivative planes come first, in axis order.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD3,
         "call": "mpfc_spectral_operators(_dup(u3), _dup(L3))",
         "gold_call": "_oracle_mpfc_spectral_operators(_dup(u3), _dup(L3))"},
        # boundary: odd point counts on both axes, where no Nyquist mode exists.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLDODD,
         "call": "mpfc_spectral_operators(_dup(uo), _dup(Lo))",
         "gold_call": "_oracle_mpfc_spectral_operators(_dup(uo), _dup(Lo))"},
    ]
