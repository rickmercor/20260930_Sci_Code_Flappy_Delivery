"""
Evaluate the ****modified discrete energy**** the scheme actually dissipates, and compare it with the original free energy of step 2 on the same layer. Return the real array $[E_{\\rm mod},\\;E,\\;E_{\\rm mod}-E,\\;r-\\sqrt{E_1(\\phi)+C_{SAV}}\\,]$ of shape `(4,)`. The modified energy is the quadratic block of step 2 plus the square of the auxiliary variable minus the shift constant; it is a function of the **pair** $(\\phi,r)$, and it coincides with the original free energy exactly when $r$ is the square root it was initialised as. The last entry is that discrepancy, the quadratization's own consistency error, and it is what makes the two energies separate along a run.

A quadratized scheme does not dissipate the energy of the model; it dissipates the energy of the enlarged system in which the nonlinear part has been replaced by the square of an independent unknown. The two agree at the initial time, because the unknown is initialised as the exact square root, and they drift apart at the rate at which the unknown's own evolution equation ceases to reproduce that square root - a second-order-in-time effect for a second-order scheme, higher order here. The unconditional stability proved for this scheme is a statement about the modified energy only; that the original free energy also decreases along the run is an empirical observation, not a theorem, and a run in which the modified energy decreases while the original does not is not evidence of a bug. The shift constant cancels between the square of the auxiliary variable and the explicit subtraction, so it does not change the modified energy at the initial layer, but it does control how much floating-point cancellation that expression carries.




****--- Formulas ---****

Every field here may have ****any**** number of spatial dimensions $d\\ge1$, with one edge length per axis; the two-dimensional case is only the instance this task eventually runs. In particular the shifted bi-Laplacian and the mean-free inverse Laplacian are the ****last two**** planes of step 1's return, whose position moves with $d$, and the grid measure of every inner product is $\\prod_i L_i/\\prod_i N_i$. $$E_{\\rm mod}(\\phi,r)=\\tfrac12\\big(\\phi,(\\Delta+1)^2\\phi\\big)_h+r^2-C_{SAV},$$ with the same grid inner product as step 2. The second returned entry is the total free energy $E$ of step 2 on the same $\\phi$, the third is their difference, and the fourth is $r-\\sqrt{E_1(\\phi)+C_{SAV}}$. Note that the third and fourth entries are not independent: $E_{\\rm mod}-E=r^2-C_{SAV}-E_1=\\big(r-\\sqrt{E_1+C_{SAV}}\\big)\\big(r+\\sqrt{E_1+C_{SAV}}\\big)$, which is a useful check on both.

Returns
-------
`np.ndarray` of shape `(4,)`, real and finite. When $r=\\sqrt{E_1(\\phi)+C_{SAV}}$ the first two entries agree to round-off and the last two are zero to round-off; the third entry always equals the fourth times $r+\\sqrt{E_1+C_{SAV}}$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mpfc_modified_energy(phi: "np.ndarray", r: float, cell: "float | tuple",
        params: "dict | None" = None) -> "np.ndarray":
    """phi: real periodic density field, of ANY dimension d >= 1.
    r: the auxiliary variable carried by the scheme, a float.
    cell: domain edge lengths, a scalar or a length-d sequence.
    params: as in mpfc_free_energy.
    Return the real array [E_mod, E, E_mod - E, r - sqrt(E1 + C_sav)]
    of shape (4,).
    Raise ValueError if phi is not a finite real array of rank at least one, if r
    is not a finite scalar, if cell or params is invalid, or if
    E1(phi) + C_sav is not strictly positive."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_mpfc_modified_energy(phi: "np.ndarray", r: float, cell: "float | tuple",
        params: "dict | None" = None) -> "np.ndarray":
    phi = _check_field(phi, "phi")
    L = _check_cell(cell, phi.shape)
    p = _check_params(params)
    r = float(r)
    if not np.isfinite(r):
        raise ValueError("r must be a finite scalar")
    e = _oracle_mpfc_free_energy(phi, L, params)
    emod = e[1] + r * r - p["C_sav"]
    rad = e[2] + p["C_sav"]
    if not (rad > 0.0):
        raise ValueError("E1(phi) + C_sav must be strictly positive")
    return np.array([emod, e[0], emod - e[0], r - np.sqrt(rad)])

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
    FLD3 = ('N3 = (8, 10, 6)\n'
            'L3 = (5.0, 7.0, 4.0)\n'
            'g3 = [np.arange(n) * (l / n) for n, l in zip(N3, L3)]\n'
            "X3, Y3, Z3 = np.meshgrid(*g3, indexing='ij')\n"
            'a3, b3, c3 = [2.0 * np.pi / l for l in L3]\n'
            'u3 = 0.15 + 0.25 * np.cos(a3 * X3) * np.cos(2 * b3 * Y3) + 0.20 * np.sin(2 * c3 * Z3) * np.cos(b3 * Y3)\n')
    R0 = "r0 = float(np.sqrt(_oracle_mpfc_free_energy(phi, Lc)[2] + 400.0))\n"
    return [
        # normal: the auxiliary variable exactly on its square root, where the
        # first two entries must agree to round-off and the third is zero.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + R0,
         "call": "mpfc_modified_energy(_dup(phi), _dup(r0), _dup(Lc))",
         "gold_call": "_oracle_mpfc_modified_energy(_dup(phi), _dup(r0), _dup(Lc))"},
        # normal: an auxiliary variable that has drifted, where the two
        # energies separate by a computable amount.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + R0,
         "call": "mpfc_modified_energy(_dup(phi), r0 + 0.03, _dup(Lc))",
         "gold_call": "_oracle_mpfc_modified_energy(_dup(phi), r0 + 0.03, _dup(Lc))"},
        # boundary: a scalar cell on the second fixture field.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD
                  + "r1 = float(np.sqrt(_oracle_mpfc_free_energy(psi, 18.0)[2] + 400.0))\n",
         "call": "mpfc_modified_energy(_dup(psi), _dup(r1), 18.0)",
         "gold_call": "_oracle_mpfc_modified_energy(_dup(psi), _dup(r1), 18.0)"},
        # boundary: a different material and a much smaller shift constant, so
        # that the cancellation between the squared variable and the shift is
        # mild and the drift entry is the dominant one.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + PSTIFF
                  + "r2 = float(np.sqrt(_oracle_mpfc_free_energy(phi, Lc, PS)[2] + 50.0))\n",
         "call": "mpfc_modified_energy(_dup(phi), 1.02 * r2, _dup(Lc), _dup(PS))",
         "gold_call": "_oracle_mpfc_modified_energy(_dup(phi), 1.02 * r2, _dup(Lc), _dup(PS))"},
        # edge: a constant field, where the quadratic part is exactly half the
        # squared constant times the area.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\nu = np.full((8, 12), 0.35)\n"
                  "ru = float(np.sqrt(_oracle_mpfc_free_energy(u, (6.0, 9.0))[2] + 400.0))\n",
         "call": "mpfc_modified_energy(_dup(u), _dup(ru), (6.0, 9.0))",
         "gold_call": "_oracle_mpfc_modified_energy(_dup(u), _dup(ru), (6.0, 9.0))"},
        # edge: a vanishing auxiliary variable, which leaves the modified
        # energy equal to the quadratic part minus the shift constant.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD,
         "call": "mpfc_modified_energy(_dup(phi), 0.0, _dup(Lc))",
         "gold_call": "_oracle_mpfc_modified_energy(_dup(phi), 0.0, _dup(Lc))"},
        # edge: a non-finite auxiliary variable, which must raise ValueError.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD +
                  "def _guard(f, *a, **k):\n"
                  "    try:\n"
                  "        f(*a, **k)\n"
                  "    except ValueError:\n"
                  "        return 1.0\n"
                  "    return 0.0\n",
         "call": "_guard(mpfc_modified_energy, _dup(phi), np.inf, _dup(Lc))",
         "gold_call": "_guard(_oracle_mpfc_modified_energy, _dup(phi), np.inf, _dup(Lc))"},
        # edge: the modified energy of a three-dimensional density.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD3,
         "call": "mpfc_modified_energy(_dup(u3), 12.0, _dup(L3))",
         "gold_call": "_oracle_mpfc_modified_energy(_dup(u3), 12.0, _dup(L3))"},
    ]
