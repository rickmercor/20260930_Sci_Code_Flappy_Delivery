"""
Advance the pair (density, auxiliary variable) by one step of the source's four-stage scheme. Build the two tableaux and the weights with step 4; for each of the four stages assemble the three arguments step 5 needs - the density accumulated with the ****implicit**** tableau, the nonlinear map $\\mathcal H$ of step 3 evaluated at the density accumulated with the ****explicit**** tableau, and the auxiliary variable accumulated with the ****implicit**** tableau - solve the stage, and combine the four stage increments with the weights. Return the real array $[\\phi^{n+1},\\;r^{n+1}]$ of shape `(2,) + phi.shape`, whose second plane is ****constant**** and equal to the new auxiliary variable. Which tableau feeds which argument, and whether the diagonal entry is included in each accumulation, are the two things this step is graded on.

The stage equations of this scheme are implicit in the density and in the auxiliary variable but explicit in every nonlinear coefficient. Written out, the stage carries three separate accumulations of earlier stage increments. Two of them run over the implicit tableau and one over the explicit tableau, and the implicit ones are the delicate case: the diagonal term of the implicit accumulation is the unknown of this stage itself, so it cannot appear on the right-hand side - it is precisely the term that step 5 moved into the denominator of the Fourier solve and into the affine correction. The explicit accumulation has no such issue because its tableau has no diagonal. Getting this wrong does not produce a crash or an unstable run; it produces a method that is stable, mass conserving and energy decreasing, and simply of the wrong order. Note also that the source's coefficients do ****not**** satisfy the usual row-sum consistency condition relating the abscissae to the two tableaux, and do not need to: the right-hand side of this problem carries no explicit time dependence, so no abscissa is ever used.




****--- Formulas ---****

Every field here may have ****any**** number of spatial dimensions $d\\ge1$, with one edge length per axis; the two-dimensional case is only the instance this task eventually runs. In particular the shifted bi-Laplacian and the mean-free inverse Laplacian are the ****last two**** planes of step 1's return, whose position moves with $d$, and the grid measure of every inner product is $\\prod_i L_i/\\prod_i N_i$. Interfaces and orders, which are this task's rather than the source's. Stage $l$ calls step 5 once, with `acc` the density accumulation, $Hl$ the map $\\mathcal H$ of step 3 evaluated at the density accumulation built from the ****explicit**** tableau, `rho` the auxiliary-variable accumulation and $a_ll$ the $l$-th diagonal entry of the ****implicit**** tableau; the three accumulations all start from the layer-$n$ value and add $\\delta t$ times a weighted sum of the stage increments. Both the radicand and the numerator inside $\\mathcal H$ are evaluated at that same explicit argument, never at $\\phi^n$. The layer is closed by the usual weighted combination of the four stage increments, $\\phi^{n+1}=\\phi^n+\\delta t\\sum_{l}b_l\\phi_{kl}$ and $r^{n+1}=r^n+\\delta t\\sum_{l}b_lr_{kl}$, with the same $b$ for both. Every one of the three accumulations runs over the stage increments already computed and stops one short of the diagonal: with $\\phi_{km}$ and $r_{km}$ the increments of the earlier stages, $\\psi_l=\\phi^{n}+\\delta t\\sum_{m=1}^{l-1}\\tilde a_{lm}\\phi_{km}$ and $\\rho_l=r^{n}+\\delta t\\sum_{m=1}^{l-1}\\tilde a_{lm}r_{km}$ for the two implicit accumulations, and $\\phi^{n}+\\delta t\\sum_{m=1}^{l-1}a_{lm}\\phi_{km}$ for the explicit argument of $\\mathcal H$. The diagonal entry $\\tilde a_{ll}$ never enters an accumulation; it is what step 5 solves with.

Returns
-------
`np.ndarray` of shape `(2,) + phi.shape`, real and finite. The spatial mean of the first plane equals that of `phi` **exactly**, whatever the step size, grid or parameters; the second plane is constant. As $\\delta t\\to0$ the first plane tends to `phi` and the second to `r`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mpfc_time_step(phi: "np.ndarray", r: float, dt: float, cell: "float | tuple",
        params: "dict | None" = None,
        coef: "dict | None" = None) -> "np.ndarray":
    """phi: the density at layer n, any dimension d >= 1.
    r: the auxiliary variable at layer n, a float.
    dt: the time step. cell: domain edge lengths, a scalar or one length per axis.
    params: as in mpfc_free_energy.
    coef: dict of overrides for the five free constants a11t, a32t,
       a33t, a31, a43 of the coefficient family; None means this task's
       prescribed set.
    Return the real array [phi_new, r_new] of shape (2,) + phi.shape,
    whose second plane is constant and equal to r_new.
    Raise ValueError if phi is not a finite real array of rank at least one, if r
    is not a finite scalar, if dt is not a finite positive scalar, or if
    cell, params or coef is invalid."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_mpfc_time_step(phi: "np.ndarray", r: float, dt: float, cell: "float | tuple",
        params: "dict | None" = None,
        coef: "dict | None" = None) -> "np.ndarray":
    phi = _check_field(phi, "phi")
    L = _check_cell(cell, phi.shape)
    p = _check_params(params)
    c = _check_coef(coef)
    r = float(r)
    dt = float(dt)
    if not np.isfinite(r):
        raise ValueError("r must be a finite scalar")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive scalar")
    tab = _oracle_mpfc_butcher(**c)
    At = tab[:16].reshape(4, 4)
    A = tab[16:32].reshape(4, 4)
    b = tab[32:36]
    phik = []
    rk = []
    for l in range(4):
        psi = phi.copy()
        for m in range(l):
            if A[l, m] != 0.0:
                psi = psi + dt * A[l, m] * phik[m]
        Hl = _oracle_mpfc_sav_terms(psi, L, params)[0]
        acc = phi.copy()
        rho = r
        for m in range(l):
            if At[l, m] != 0.0:
                acc = acc + dt * At[l, m] * phik[m]
                rho = rho + dt * At[l, m] * rk[m]
        st = _oracle_mpfc_stage_solve(acc, Hl, rho, At[l, l], dt, L, params)
        phik.append(st[2])
        rk.append(float(st[3].flat[0]))
    new = phi.copy()
    rn = r
    for l in range(4):
        if b[l] != 0.0:
            new = new + dt * b[l] * phik[l]
            rn = rn + dt * b[l] * rk[l]
    return np.stack([new, np.full_like(new, rn)])

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
    PE15 = "C15 = {'a11t': 1.0, 'a32t': 1.0, 'a33t': 1.0, 'a31': 1.0, 'a43': 1.0}\n"
    FLD3 = ('N3 = (8, 10, 6)\n'
            'L3 = (5.0, 7.0, 4.0)\n'
            'g3 = [np.arange(n) * (l / n) for n, l in zip(N3, L3)]\n'
            "X3, Y3, Z3 = np.meshgrid(*g3, indexing='ij')\n"
            'a3, b3, c3 = [2.0 * np.pi / l for l in L3]\n'
            'u3 = 0.15 + 0.25 * np.cos(a3 * X3) * np.cos(2 * b3 * Y3) + 0.20 * np.sin(2 * c3 * Z3) * np.cos(b3 * Y3)\n')
    R0 = "r0 = float(np.sqrt(_oracle_mpfc_free_energy(phi, Lc)[2] + 400.0))\n"
    return [
        # normal: one step of the prescribed scheme at the production step size.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + R0,
         "call": "mpfc_time_step(_dup(phi), _dup(r0), 0.125, _dup(Lc))",
         "gold_call": "_oracle_mpfc_time_step(_dup(phi), _dup(r0), 0.125, _dup(Lc))"},
        # normal: the same step driven by the source's displayed tableau, which
        # must give a genuinely different layer.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + R0 + PE15,
         "call": "mpfc_time_step(_dup(phi), _dup(r0), 0.125, _dup(Lc), None, _dup(C15))",
         "gold_call": "_oracle_mpfc_time_step(_dup(phi), _dup(r0), 0.125, _dup(Lc), None, _dup(C15))"},
        # boundary: a scalar cell and a tiny step, where the new layer must
        # differ from the old one at first order in the step.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + R0,
         "call": "mpfc_time_step(_dup(phi), _dup(r0), 1.0e-5, 18.0)",
         "gold_call": "_oracle_mpfc_time_step(_dup(phi), _dup(r0), 1.0e-5, 18.0)"},
        # boundary: a step thirty-two times the production one, which the
        # scheme absorbs because it is unconditionally energy stable.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + R0,
         "call": "mpfc_time_step(_dup(phi), _dup(r0), 4.0, _dup(Lc))",
         "gold_call": "_oracle_mpfc_time_step(_dup(phi), _dup(r0), 4.0, _dup(Lc))"},
        # edge: a different material with a small shift constant, where the
        # auxiliary variable is of order one rather than of order twenty.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + PSTIFF
                  + "r0 = float(np.sqrt(_oracle_mpfc_free_energy(phi, Lc, PS)[2] + 50.0))\n",
         "call": "mpfc_time_step(_dup(phi), _dup(r0), 0.05, _dup(Lc), _dup(PS))",
         "gold_call": "_oracle_mpfc_time_step(_dup(phi), _dup(r0), 0.05, _dup(Lc), _dup(PS))"},
        # edge: an auxiliary variable deliberately off its square root, which
        # the scheme must still advance without reference to that square root.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + R0,
         "call": "mpfc_time_step(_dup(phi), 0.5 * r0, 0.125, _dup(Lc))",
         "gold_call": "_oracle_mpfc_time_step(_dup(phi), 0.5 * r0, 0.125, _dup(Lc))"},
        # edge: an unknown Runge-Kutta coefficient key, which must raise
        # ValueError rather than be silently ignored.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + R0 +
                  "def _guard(f, *a, **k):\n"
                  "    try:\n"
                  "        f(*a, **k)\n"
                  "    except ValueError:\n"
                  "        return 1.0\n"
                  "    return 0.0\n",
         "call": "_guard(mpfc_time_step, _dup(phi), _dup(r0), 0.125, _dup(Lc), None, {'zzz': 1.0})",
         "gold_call": "_guard(_oracle_mpfc_time_step, _dup(phi), _dup(r0), 0.125, _dup(Lc), None, {'zzz': 1.0})"},
        # edge: one whole step of a three-dimensional layer, which exercises the four
        # stages, both tableaux and the closing combination away from two dimensions.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD3 + "r3 = float(np.sqrt(_oracle_mpfc_free_energy(u3, L3)[2] + 400.0))\n",
         "call": "mpfc_time_step(_dup(u3), _dup(r3), 0.125, _dup(L3))",
         "gold_call": "_oracle_mpfc_time_step(_dup(u3), _dup(r3), 0.125, _dup(L3))"},
    ]
