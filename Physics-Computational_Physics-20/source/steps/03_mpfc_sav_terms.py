"""
Evaluate the two field-valued quantities the time-stepping needs at a given density layer. Return the real array $[\\,\\mathcal H(\\phi),\\;\\mu(\\phi)\\,]$ of shape `(2,) + phi.shape`, where $\\mu$ is the ****chemical potential of the original model****, the variational derivative of the free energy of step 2, and $\\mathcal H$ is the ****nonlinear map of the source's scalar-auxiliary-variable reformulation****: the variational derivative of $E_1$ alone, divided by the square root of $E_1$ shifted by the positive constant $C_{SAV}$. Both must be evaluated at the **same** argument: in particular the radicand is $E_1$ of the field passed in, never $E_1$ of some earlier layer. Raise `ValueError` if that radicand is not strictly positive, which is the condition on $C_{SAV}$ the source states and the only thing that makes the reformulation legitimate.

The reformulation replaces the nonlinear part of the energy by the square of a single scalar unknown, so that the part of the chemical potential coming from $E_1$ becomes a product of a known field and that unknown, and a discretisation which takes the field explicitly and the unknown implicitly is linear and unconditionally stable by construction. Two facts make it work. The shift constant must be large enough that the shifted energy is strictly positive, not merely non-negative, because the map is a division by its square root. And the map is a **variational** derivative, so it inherits the long-range term of $E_1$ through the same $\\chi$ that entered the energy: differentiating $\\tfrac{\\alpha}{2}(\\chi,\\phi-\\bar\\phi)_h$ in $\\phi$ gives $\\alpha\\chi$, not $\\tfrac{\\alpha}{2}\\chi$, because $\\chi$ itself depends linearly on $\\phi$ and the operator $(-\\Delta)^{-1}$ is symmetric. The price of the reformulation is that the unknown obeys its own evolution equation and slowly drifts away from the square root it started as, so the energy the scheme provably dissipates is a modified one.




****--- Formulas ---****

Every field here may have ****any**** number of spatial dimensions $d\\ge1$, with one edge length per axis; the two-dimensional case is only the instance this task eventually runs. In particular the shifted bi-Laplacian and the mean-free inverse Laplacian are the ****last two**** planes of step 1's return, whose position moves with $d$, and the grid measure of every inner product is $\\prod_i L_i/\\prod_i N_i$. The second returned entry is the chemical potential printed in the problem statement, $\\mu(\\phi)=\\phi^3-\\epsilon\\phi+(\\Delta+1)^2\\phi+\\alpha\\chi$, with $\\chi$ from step 1. The first entry is the source's $\\mathcal H$: a quotient whose denominator is the single scalar $\\sqrt{E_1(\\phi)+C_{SAV}}$, the same at every grid point, with $E_1$ exactly as step 2 returns it, and whose numerator is the variational derivative of that same $E_1$ - so it is $\\mu$ with the one term the scheme keeps implicit and does not quadratize removed, and nothing else changed: the removed term is the shifted bi-Laplacian $(\\Delta+1)^{2}\\phi$, leaving the numerator $\\phi^{3}-\\epsilon\\phi+\\alpha\\chi$, every term with the coefficient it carries in $\\mu$.

Returns
-------
`np.ndarray` of shape `(2,) + phi.shape`, real and finite. The two entries differ only by the shifted bi-Laplacian and by the scalar division, so $\\mu-\\sqrt{E_1+C_{SAV}}\\,\\mathcal H=(\\Delta+1)^2\\phi$ exactly. For a constant field $c$ the first entry is $(c^3-\\epsilon c)/\\sqrt{E_1+C_{SAV}}$ and the second is $c^3-\\epsilon c+c$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mpfc_sav_terms(phi: "np.ndarray", cell: "float | tuple",
        params: "dict | None" = None) -> "np.ndarray":
    """phi: real periodic density field, of ANY dimension d >= 1.
    cell: domain edge lengths, a scalar or a length-d sequence.
    params: as in mpfc_free_energy.
    Return the real array [H(phi), mu(phi)] of shape (2,) + phi.shape.
    Raise ValueError if phi is not a finite real array of rank at least one, if
    cell is not a positive scalar or one length per axis, if params
    is invalid, or if
    E1(phi) + C_sav is not strictly positive."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_mpfc_sav_terms(phi: "np.ndarray", cell: "float | tuple",
        params: "dict | None" = None) -> "np.ndarray":
    phi = _check_field(phi, "phi")
    L = _check_cell(cell, phi.shape)
    p = _check_params(params)
    ops = _oracle_mpfc_spectral_operators(phi, L)
    lphi, chi = ops[-2], ops[-1]
    e1 = _oracle_mpfc_free_energy(phi, L, params)[2]
    rad = e1 + p["C_sav"]
    if not (rad > 0.0):
        raise ValueError("E1(phi) + C_sav must be strictly positive")
    nl = phi ** 3 - p["eps"] * phi + p["alpha"] * chi
    return np.stack([nl / np.sqrt(rad), nl + lphi])

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
         "call": "mpfc_sav_terms(_dup(phi), _dup(Lc))",
         "gold_call": "_oracle_mpfc_sav_terms(_dup(phi), _dup(Lc))"},
        # normal: a different material, which moves the radicand and hence
        # rescales H without touching the chemical potential.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + PSTIFF,
         "call": "mpfc_sav_terms(_dup(phi), _dup(Lc), _dup(PS))",
         "gold_call": "_oracle_mpfc_sav_terms(_dup(phi), _dup(Lc), _dup(PS))"},
        # boundary: a scalar cell and the second fixture field.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD,
         "call": "mpfc_sav_terms(_dup(psi), 18.0)",
         "gold_call": "_oracle_mpfc_sav_terms(_dup(psi), 18.0)"},
        # boundary: a small shift constant, where the radicand is still
        # positive but the two entries differ by three orders of magnitude.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + "PC = {'C_sav': 12.0}\n",
         "call": "mpfc_sav_terms(_dup(phi), _dup(Lc), _dup(PC))",
         "gold_call": "_oracle_mpfc_sav_terms(_dup(phi), _dup(Lc), _dup(PC))"},
        # edge: the long-range coupling switched off.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + "PA = {'alpha': 0.0}\n",
         "call": "mpfc_sav_terms(phi * psi, _dup(Lc), _dup(PA))",
         "gold_call": "_oracle_mpfc_sav_terms(phi * psi, _dup(Lc), _dup(PA))"},
        # edge: a shift constant too small to keep the radicand positive, which
        # must raise ValueError rather than return a complex or nan field.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + "PN = {'C_sav': -1.0e3}\n"
                  "def _guard(f, *a):\n"
                  "    try:\n"
                  "        f(*a)\n"
                  "    except ValueError:\n"
                  "        return 1.0\n"
                  "    return 0.0\n",
         "call": "_guard(mpfc_sav_terms, _dup(phi), _dup(Lc), _dup(PN))",
         "gold_call": "_guard(_oracle_mpfc_sav_terms, _dup(phi), _dup(Lc), _dup(PN))"},
        # edge: the two maps on a one-dimensional density.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD1,
         "call": "mpfc_sav_terms(_dup(u1), _dup(L1))",
         "gold_call": "_oracle_mpfc_sav_terms(_dup(u1), _dup(L1))"},
        # edge: the two maps on a three-dimensional density.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD3,
         "call": "mpfc_sav_terms(_dup(u3), _dup(L3))",
         "gold_call": "_oracle_mpfc_sav_terms(_dup(u3), _dup(L3))"},
    ]
