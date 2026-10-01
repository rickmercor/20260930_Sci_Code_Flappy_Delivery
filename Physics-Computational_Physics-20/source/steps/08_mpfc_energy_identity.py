"""
Evaluate, term by term, the exact algebraic identity that the source's energy-dissipation theorem is proved from, for one step taken from the layer $(\\phi^n,r^n)$. Writing $\\mathcal L=(\\Delta+1)^2$ for the quadratic operator and $(\\cdot,\\cdot)_h$ for the grid inner product of step 2, the theorem rewrites the change in the quadratic part of the energy as a ****stage-work**** term and a ****quadratic-form**** term built from the same matrix $P$ that step 4 returns the eigenvalues of. Return the real array of shape `(4,)` holding, in order: $(\\phi^{n+1},\\mathcal L\\phi^{n+1})_h$, $(\\phi^{n},\\mathcal L\\phi^{n})_h$, the stage-work term and the quadratic-form term. The four are not independent: the first equals the second plus the third minus the fourth, to round-off, and that is the identity this step certifies. Raise `ValueError` on the same invalid inputs as the single step of step 6.

A Runge-Kutta method is energy stable when the change it makes to the energy can be written as a sum of terms whose signs are known in advance. For the quadratic part of this energy that is achievable exactly rather than up to a remainder, which is why the stability of this scheme is a theorem and not an estimate: one of the terms is a quadratic form in the four stage increments whose matrix is positive semi-definite for every member of the coefficient family, so that term can only ever subtract. Turning that proof into four computable numbers is a sharper check than watching an energy decrease, because a decreasing energy is consistent with many wrong schemes, while the identity holds only for the right stage increments, the right accumulations and the right matrix.




****--- Formulas ---****

There is no formula sheet for this step. The expansion of the quadratic energy over one step, the two terms it splits into and the matrix that carries the second of them are the source's energy-dissipation proof, and the problem statement lists the identity as something to fetch from it. Derive it, or fetch it; do not guess a decomposition that merely reproduces the difference. Fixed here by the task rather than by the paper, and graded. The four stage increments are exactly those of step 6, built with the same two tableaux. $\\mathcal L$ is the shifted bi-Laplacian of step 1 and every inner product is the grid inner product of step 2, so fields of any rank $d\\ge1$ are allowed. The third returned number is the term of the expansion that is ****linear**** in the stage increments when each of them is paired with the very field that this scheme's own stage equation applies $\\mathcal L$ to at that stage; the fourth is what is left, which is ****bilinear**** in the stage increments and whose coefficient matrix is the $P$ of step 4. Those two sentences fix both numbers uniquely, including their signs and any numerical factor, once the expansion is actually carried out.

Returns
-------
`np.ndarray` of shape `(4,)`, real and finite. The first entry equals the second plus the third minus the fourth to round-off, on any data and any step size - that identity is the point of the step. The fourth entry is **non-negative** for every member of the coefficient family and every step size, which is what makes the quadratic part of the energy dissipative, and both the third and the fourth tend to zero as $\\delta t\\to0$. They are not simple powers of $\\delta t$ here: the stage increments themselves depend on the step size through the implicit denominator.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mpfc_energy_identity(phi: "np.ndarray", r: float, dt: float,
        cell: "float | tuple", params: "dict | None" = None,
        coef: "dict | None" = None) -> "np.ndarray":
    """phi: the density at layer n, any dimension d >= 1.
    r: the auxiliary variable at layer n, a float.
    dt: the time step. cell: domain edge lengths, a scalar or one
       length per axis.
    params: as in mpfc_free_energy. coef: as in mpfc_time_step.
    Return the real array [(phi^{n+1}, L phi^{n+1})_h,
    (phi^n, L phi^n)_h, W, S] of shape (4,), with L the shifted
    bi-Laplacian, W the stage-work term and S the quadratic-form term.
    Raise ValueError if phi is not a finite real array of rank at least
    one, if r is not a finite scalar, if dt is not a finite positive
    scalar, or if cell, params or coef is invalid."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_mpfc_energy_identity(phi: "np.ndarray", r: float, dt: float,
        cell: "float | tuple", params: "dict | None" = None,
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
    for l in range(4):
        if b[l] != 0.0:
            new = new + dt * b[l] * phik[l]
    # the accumulation of the identity INCLUDES the diagonal term, unlike the
    # three accumulations that feed a stage
    phin = [phi + dt * sum(At[l, m] * phik[m] for m in range(l + 1)) for l in range(4)]
    P = np.diag(b) @ At + At.T @ np.diag(b) - np.outer(b, b)
    # the shifted bi-Laplacian is the last but one plane of step 1's return
    lk = [_oracle_mpfc_spectral_operators(f, L)[-2] for f in phik]
    q_next = _ip(new, _oracle_mpfc_spectral_operators(new, L)[-2], L)
    q_now = _ip(phi, _oracle_mpfc_spectral_operators(phi, L)[-2], L)
    work = 2.0 * dt * sum(b[l] * _ip(phin[l], lk[l], L) for l in range(4))
    pform = dt ** 2 * sum(P[l, m] * _ip(phik[l], lk[m], L)
                          for l in range(4) for m in range(4))
    return np.array([q_next, q_now, work, pform])

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
        # normal: the identity on the production layer at the production step.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + R0,
         "call": "mpfc_energy_identity(_dup(phi), _dup(r0), 0.125, _dup(Lc))",
         "gold_call": "_oracle_mpfc_energy_identity(_dup(phi), _dup(r0), 0.125, _dup(Lc))"},
        # normal: a stiffer material, where the two terms of the identity are of
        # very different size and a missing factor shows up immediately.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + PSTIFF + R0,
         "call": "mpfc_energy_identity(_dup(phi), _dup(r0), 0.05, _dup(Lc), _dup(PS))",
         "gold_call": "_oracle_mpfc_energy_identity(_dup(phi), _dup(r0), 0.05, _dup(Lc), _dup(PS))"},
        # boundary: a step size eight times the production one, where the
        # quadratic-form term is no longer a small correction.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + R0,
         "call": "mpfc_energy_identity(_dup(phi), _dup(r0), 1.0, _dup(Lc))",
         "gold_call": "_oracle_mpfc_energy_identity(_dup(phi), _dup(r0), 1.0, _dup(Lc))"},
        # boundary: the source's displayed coefficient member, whose implicit
        # tableau has a different diagonal and so a different accumulation.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + PE15 + R0,
         "call": "mpfc_energy_identity(_dup(phi), _dup(r0), 0.125, _dup(Lc), None, _dup(C15))",
         "gold_call": "_oracle_mpfc_energy_identity(_dup(phi), _dup(r0), 0.125, _dup(Lc), None, _dup(C15))"},
        # edge: three dimensions, where the operator plane and the measure both move.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD3 +
                  "r3 = float(np.sqrt(_oracle_mpfc_free_energy(u3, L3)[2] + 400.0))\n",
         "call": "mpfc_energy_identity(_dup(u3), _dup(r3), 0.125, _dup(L3))",
         "gold_call": "_oracle_mpfc_energy_identity(_dup(u3), _dup(r3), 0.125, _dup(L3))"},
        # edge: an auxiliary variable off its square root, which changes every
        # stage increment and so both terms of the identity.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + R0,
         "call": "mpfc_energy_identity(_dup(phi), 0.4 * r0, 0.125, _dup(Lc))",
         "gold_call": "_oracle_mpfc_energy_identity(_dup(phi), 0.4 * r0, 0.125, _dup(Lc))"},
        # edge: a non-positive step size, which must raise ValueError.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + R0 +
                  "def _guard(f, *a, **k):\n"
                  "    try:\n"
                  "        f(*a, **k)\n"
                  "    except ValueError:\n"
                  "        return 1.0\n"
                  "    return 0.0\n",
         "call": "_guard(mpfc_energy_identity, _dup(phi), _dup(r0), 0.0, _dup(Lc))",
         "gold_call": "_guard(_oracle_mpfc_energy_identity, _dup(phi), _dup(r0), 0.0, _dup(Lc))"},
    ]
