"""
Evaluate the discrete free energy of the model on one density field and split it into the two parts the scheme treats differently. Return the real array $[E,\\;E_{\\rm quad},\\;E_1,\\;\\bar\\phi]$ of shape `(4,)`, where $E_{\\rm quad}=\\tfrac12(\\phi,(\\Delta+1)^2\\phi)_h$ is the quadratic block, $E_1$ is everything else - the quartic bulk term, the temperature term and the long-range term - $E=E_{\\rm quad}+E_1$ is the total free energy of the model as printed in the problem statement, and $\\bar\\phi$ is the spatial mean of $\\phi$. The split is not cosmetic: the scheme of the source treats $E_{\\rm quad}$ implicitly through its Fourier symbol and $E_1$ through an auxiliary variable, so an implementation that puts one term on the wrong side of the split is running a different scheme.

The discrete $L^2$ inner product on a uniform periodic grid is the plain grid sum times the cell area, $(f,g)_h=\\sum_{j,k}f_{jk}g_{jk}\\,h_xh_y$ with $h_x=L_x/N_x$ and $h_y=L_y/N_y$, and every integral over $\\Omega$ in the energy is that same sum against the constant field one. Two readings of the long-range term agree in the continuum and differ on the grid: the integral of the squared pointwise first derivatives of $\\chi$, and minus the inner product of $\\chi$ with its Laplacian. **This task fixes the second**, which on the grid is the identity $-(\\chi,\\Delta_h\\chi)_h=(\\chi,\\phi-\\bar\\phi)_h$ obtained from the equation $\\chi$ solves - so the long-range energy is available without differentiating $\\chi$ at all. The quadratic block is degenerate rather than coercive: its symbol $(1-|k|^2)^2$ vanishes on the whole circle $|k|=1$, which is why the minimiser of this energy is a periodic crystal rather than a constant, and why $E_{\\rm quad}$ alone controls nothing on that circle.

****--- Formulas ---****

Every field here may have ****any**** number of spatial dimensions $d\\ge1$, with one edge length per axis; the two-dimensional case is only the instance this task eventually runs. In particular the shifted bi-Laplacian and the mean-free inverse Laplacian are the ****last two**** planes of step 1's return, whose position moves with $d$, and the grid measure of every inner product is $\\prod_i L_i/\\prod_i N_i$. With $\\chi$ the mean-free inverse Laplacian of step 1, $\\bar\\phi$ the spatial mean, and $(\\cdot,\\cdot)_h$ the grid inner product above, $$E_{\\rm quad}=\\tfrac12\\big(\\phi,(\\Delta+1)^2\\phi\\big)_h,\\qquad E_1=\\Big(\\tfrac14\\phi^4-\\tfrac{\\epsilon}{2}\\phi^2,\\;1\\Big)_h+\\tfrac{\\alpha}{2}\\big(\\chi,\\phi-\\bar\\phi\\big)_h,\\qquad E=E_{\\rm quad}+E_1.$$ Note which quadratic-in-$\\phi$ term sits where: the whole of $\\tfrac12\\phi(\\Delta+1)^2\\phi$ is in $E_{\\rm quad}$, and what is left in the bulk density is $-\\tfrac{\\epsilon}{2}\\phi^2$, not $\\tfrac{1-\\epsilon}{2}\\phi^2$ - the two differ by exactly the $\\tfrac12\\phi^2$ that the expansion of the shifted bi-Laplacian already supplies.

Returns
-------
`np.ndarray` of shape `(4,)`, real and finite. For a constant field $c$ the entries are $\\tfrac{1-\\epsilon}{2}c^2|\\Omega|+\\tfrac14c^4|\\Omega|$, $\\tfrac12c^2|\\Omega|$, $\\tfrac14c^4|\\Omega|-\\tfrac{\\epsilon}{2}c^2|\\Omega|$ and $c$; the long-range contribution to $E_1$ is non-negative for every field, and exactly zero when $\\phi$ is constant.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mpfc_free_energy(phi: "np.ndarray", cell: "float | tuple",
        params: "dict | None" = None) -> "np.ndarray":
    """phi: real periodic density field, of ANY dimension d >= 1.
    cell: domain edge lengths, a scalar or a length-d sequence.
    params: dict of overrides for the model parameters
       M (mobility), eps, alpha and C_sav; None means the fixed set.
       M must be strictly positive and alpha must be non-negative;
       eps and C_sav may take any finite value, the positivity of
       E1 + C_sav being checked only where a square root is taken.
    Return the real array [E, E_quad, E1, mean(phi)] of shape (4,).
    Raise ValueError if phi is not a finite real array of rank at least one, if
    cell is not a positive scalar or one length per axis, or if
    params carries an
    unknown key, a non-finite value, a non-positive M or a negative
    alpha."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_mpfc_free_energy(phi: "np.ndarray", cell: "float | tuple",
        params: "dict | None" = None) -> "np.ndarray":
    phi = _check_field(phi, "phi")
    L = _check_cell(cell, phi.shape)
    p = _check_params(params)
    ops = _oracle_mpfc_spectral_operators(phi, L)
    lphi = ops[-2]
    chi = ops[-1]
    mean = float(np.sum(phi)) / float(phi.size)
    e_quad = 0.5 * _ip(phi, lphi, L)
    bulk = _ip(0.25 * phi ** 4 - 0.5 * p["eps"] * phi ** 2, np.ones_like(phi), L)
    e_nl = 0.5 * p["alpha"] * _ip(chi, phi - mean, L)
    e1 = bulk + e_nl
    return np.array([e_quad + e1, e_quad, e1, mean])

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
    PSTIFF = "PS = {'eps': 0.9, 'alpha': 0.05, 'M': 1.0, 'C_sav': 50.0}\n"
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
    return [
        # normal: the fixed parameter set on the rectangular fixture.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD,
         "call": "mpfc_free_energy(_dup(phi), _dup(Lc))",
         "gold_call": "_oracle_mpfc_free_energy(_dup(phi), _dup(Lc))"},
        # normal: a different material - a larger temperature parameter and a
        # weaker long-range coupling shift the two parts in opposite senses.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + PSTIFF,
         "call": "mpfc_free_energy(_dup(phi), _dup(Lc), _dup(PS))",
         "gold_call": "_oracle_mpfc_free_energy(_dup(phi), _dup(Lc), _dup(PS))"},
        # boundary: a scalar cell on a field whose mean is not the fixture's.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD,
         "call": "mpfc_free_energy(_dup(psi), 18.0)",
         "gold_call": "_oracle_mpfc_free_energy(_dup(psi), 18.0)"},
        # boundary: a strictly mean-zero field, where the mean subtraction in
        # the nonlocal term is inactive and the fourth entry is zero.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD,
         "call": "mpfc_free_energy(phi - phi.mean(), _dup(Lc))",
         "gold_call": "_oracle_mpfc_free_energy(phi - phi.mean(), _dup(Lc))"},
        # edge: the long-range coupling switched off, so the nonlocal term
        # must drop out of E1 exactly rather than approximately.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + "PA = {'alpha': 0.0}\n",
         "call": "mpfc_free_energy(_dup(phi), _dup(Lc), _dup(PA))",
         "gold_call": "_oracle_mpfc_free_energy(_dup(phi), _dup(Lc), _dup(PA))"},
        # edge: a constant field, where the quadratic part is exactly half the
        # squared constant times the area and the nonlocal part is zero.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\nu = np.full((8, 12), 0.35)\n",
         "call": "mpfc_free_energy(_dup(u), (6.0, 9.0))",
         "gold_call": "_oracle_mpfc_free_energy(_dup(u), (6.0, 9.0))"},
        # edge: a non-positive mobility, which the parameter contract forbids
        # and which must raise ValueError.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD +
                  "def _guard(f, *a, **k):\n"
                  "    try:\n"
                  "        f(*a, **k)\n"
                  "    except ValueError:\n"
                  "        return 1.0\n"
                  "    return 0.0\n",
         "call": "_guard(mpfc_free_energy, _dup(phi), _dup(Lc), {'M': 0.0})",
         "gold_call": "_guard(_oracle_mpfc_free_energy, _dup(phi), _dup(Lc), {'M': 0.0})"},
        # edge: the free energy of a one-dimensional density.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD1,
         "call": "mpfc_free_energy(_dup(u1), _dup(L1))",
         "gold_call": "_oracle_mpfc_free_energy(_dup(u1), _dup(L1))"},
        # edge: the free energy of a three-dimensional density, where the measure is
        # a volume and the operator planes have moved.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD3,
         "call": "mpfc_free_energy(_dup(u3), _dup(L3))",
         "gold_call": "_oracle_mpfc_free_energy(_dup(u3), _dup(L3))"},
    ]
