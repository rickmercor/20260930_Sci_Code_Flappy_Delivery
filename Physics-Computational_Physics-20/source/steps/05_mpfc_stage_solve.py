"""
Carry out one Runge-Kutta stage of the source's scheme in Fourier space. The stage is implicit in two coupled unknowns - a field increment and a scalar increment - and the source eliminates them by writing the field increment as an affine function of the scalar. Given the ****accumulated density**** $\\psi$ that the implicit tableau builds for this stage, the nonlinear map $\\mathcal H$ of step 3 already evaluated at this stage's explicit argument, the ****accumulated auxiliary variable**** $\\rho$, the diagonal entry $\\tilde a_{ll}$ of the implicit tableau and the step $\\delta t$, return the real array $[\\mathcal A,\\;\\mathcal B,\\;\\phi_{k},\\;r_{k}]$ of shape `(4,) + acc.shape` - the argument named `acc` in the signature is that $\\psi$ - where $\\mathcal A$ and $\\mathcal B$ are the two fields of the source's affine decomposition, $\\phi_k$ is this stage's density increment and the last plane is ****constant****, equal to this stage's scalar increment $r_k$.

A scheme built on an auxiliary variable is linear at every stage, but it is not decoupled: the stage carries a field unknown and a scalar unknown that depend on each other - the field through the scalar that multiplies the nonlinear map, the scalar through an inner product of that map with the field. The device that makes such a stage cost one transform pair rather than an iteration is to treat the scalar as a parameter, solve the field equation once with it carried along, and then close the system with the single scalar equation that remains. Whether the resulting solve is unconditionally stable is a question about the symbol that is divided by, and whether the stage conserves mass is a question about what that symbol does at the zero wavenumber; both are settled by the construction rather than assumed by it.




****--- Formulas ---****

There is no formula sheet for this step. The stage equation, the elimination that turns it into one division per wavenumber, the two fields of the resulting affine decomposition and the scalar equation that closes the stage are the source's, and the problem statement lists each of them as something to fetch from it. Fetch them; do not reconstruct them by analogy with a scheme that carries no auxiliary variable. Fixed here by the task rather than by the paper, and graded. Every field may have ****any**** number of spatial dimensions $d\\ge1$, with one edge length per axis, and every transform is the $d$-dimensional one; the two-dimensional case is only the instance this task eventually runs. The spatial operators are those of step 1 and the mobility operator is $\\mathcal M\\Delta$, both applied by their symbols; every transform is the unnormalised forward transform of step 1 and every inverse takes the real part. The affine decomposition is written as $\\phi_k=\\mathcal A+\\delta t\\,\\tilde a_{ll}\\,r_k\\,\\mathcal B$, which fixes what the first two returned planes mean: $\\mathcal A$ is the part of the stage increment that does ****not**** depend on this stage's scalar increment, and $\\mathcal B$ is the field multiplying it, normalised so that the factor $\\delta t\\,\\tilde a_{ll}$ sits outside $\\mathcal B$ and not inside it. Every inner product is the grid inner product of step 2. Assembling $\\psi$, $\\mathcal H$ and $\\rho$ for each stage is step 6's job, not this one's.

Returns
-------
`np.ndarray` of shape `(4,) + acc.shape`, real and finite. The first three planes have **exactly zero spatial mean** whatever the data, and the fourth is a constant plane. As $\\delta t\\to0$ the third plane tends to the first.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mpfc_stage_solve(acc: "np.ndarray", Hl: "np.ndarray", rho: float, a_ll: float,
        dt: float, cell: "float | tuple",
        params: "dict | None" = None) -> "np.ndarray":
    """acc: the accumulated density psi of this stage, any dimension.
    Hl: the nonlinear map H evaluated at this stage's explicit
       argument, same shape.
    rho: the accumulated auxiliary variable of this stage, a float.
    a_ll: the diagonal entry of the implicit tableau for this stage.
    dt: the time step. cell: domain edge lengths, a scalar or one length per axis.
    params: as in mpfc_free_energy.
    Return the real array [A, B, phi_k, r_k] of shape (4,) + acc.shape,
    whose last plane is constant and equal to r_k.
    Raise ValueError if acc and Hl are not finite real arrays of rank at least one
    of the same shape, if rho or a_ll is not a finite scalar, if dt is
    not a finite positive scalar, if params is invalid, or if either the
    stage operator or the scalar equation is singular."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_mpfc_stage_solve(acc: "np.ndarray", Hl: "np.ndarray", rho: float, a_ll: float,
        dt: float, cell: "float | tuple",
        params: "dict | None" = None) -> "np.ndarray":
    acc = _check_field(acc, "acc")
    Hl = _check_field(Hl, "Hl")
    if Hl.shape != acc.shape:
        raise ValueError("acc and Hl must have the same shape")
    L = _check_cell(cell, acc.shape)
    p = _check_params(params)
    rho = float(rho)
    a_ll = float(a_ll)
    dt = float(dt)
    if not np.isfinite(rho) or not np.isfinite(a_ll):
        raise ValueError("rho and a_ll must be finite scalars")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive scalar")
    k2 = _k2(acc.shape, L)
    Lt = (k2 - 1.0) ** 2
    Gt = -p["M"] * k2
    den = 1.0 - dt * a_ll * Gt * Lt
    if np.any(den == 0.0):
        raise ValueError("the stage operator is singular for these data")
    Hh = np.fft.fftn(Hl)
    Al = np.real(np.fft.ifftn((Gt * Lt * np.fft.fftn(acc) + rho * Gt * Hh) / den))
    Bl = np.real(np.fft.ifftn((Gt * Hh) / den))
    dn = 2.0 - dt * a_ll * _ip(Hl, Bl, L)
    if dn == 0.0:
        raise ValueError("the stage equation for the auxiliary variable is singular")
    rkl = _ip(Hl, Al, L) / dn
    pkl = Al + dt * a_ll * rkl * Bl
    return np.stack([Al, Bl, pkl, np.full_like(Al, rkl)])

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
    STG = ("Hl = _oracle_mpfc_sav_terms(phi, Lc)[0]\n"
           "acc = phi + 0.01 * psi\n"
           "r0 = float(np.sqrt(_oracle_mpfc_free_energy(phi, Lc)[2] + 400.0))\n")
    return [
        # normal: the first stage of the prescribed tableau.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + STG,
         "call": "mpfc_stage_solve(_dup(acc), _dup(Hl), _dup(r0), 0.5, 0.125, _dup(Lc))",
         "gold_call": "_oracle_mpfc_stage_solve(_dup(acc), _dup(Hl), _dup(r0), 0.5, 0.125, _dup(Lc))"},
        # normal: the third stage, whose diagonal entry is the largest of the
        # four, on a different material.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + PSTIFF
                  + "Hl = _oracle_mpfc_sav_terms(phi, Lc, PS)[0]\n"
                    "acc = phi - 0.02 * psi\n"
                    "r0 = float(np.sqrt(_oracle_mpfc_free_energy(phi, Lc, PS)[2] + 50.0))\n",
         "call": "mpfc_stage_solve(_dup(acc), _dup(Hl), _dup(r0), 1.5, 0.125, _dup(Lc), _dup(PS))",
         "gold_call": "_oracle_mpfc_stage_solve(_dup(acc), _dup(Hl), _dup(r0), 1.5, 0.125, _dup(Lc), _dup(PS))"},
        # boundary: a scalar cell and a much smaller step, where the operator
        # is close to the identity and the stage increment is nearly explicit.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + STG,
         "call": "mpfc_stage_solve(_dup(acc), _dup(Hl), _dup(r0), 2.0 / 3.0, 1.0e-4, 18.0)",
         "gold_call": "_oracle_mpfc_stage_solve(_dup(acc), _dup(Hl), _dup(r0), 2.0 / 3.0, 1.0e-4, 18.0)"},
        # boundary: a large step, where the denominator is dominated by the
        # fourth-order symbol at every mode but the zero one.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + STG,
         "call": "mpfc_stage_solve(_dup(acc), _dup(Hl), _dup(r0), 1.0, 4.0, _dup(Lc))",
         "gold_call": "_oracle_mpfc_stage_solve(_dup(acc), _dup(Hl), _dup(r0), 1.0, 4.0, _dup(Lc))"},
        # edge: a vanishing auxiliary variable, which removes the explicit
        # part of the first numerator and leaves only the linear drive.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + STG,
         "call": "mpfc_stage_solve(_dup(acc), _dup(Hl), 0.0, 0.5, 0.125, _dup(Lc))",
         "gold_call": "_oracle_mpfc_stage_solve(_dup(acc), _dup(Hl), 0.0, 0.5, 0.125, _dup(Lc))"},
        # edge: a constant accumulated field, whose linear drive vanishes
        # identically so that both returned fields have exactly zero mean.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + STG,
         "call": "mpfc_stage_solve(np.full_like(_dup(phi), 0.15), _dup(Hl), _dup(r0), 0.5, 0.125, _dup(Lc))",
         "gold_call": "_oracle_mpfc_stage_solve(np.full_like(_dup(phi), 0.15), _dup(Hl), _dup(r0), 0.5, 0.125, _dup(Lc))"},
        # edge: a non-positive step size, which the stage solve forbids and
        # which must raise ValueError.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD + STG +
                  "def _guard(f, *a, **k):\n"
                  "    try:\n"
                  "        f(*a, **k)\n"
                  "    except ValueError:\n"
                  "        return 1.0\n"
                  "    return 0.0\n",
         "call": "_guard(mpfc_stage_solve, _dup(phi), _dup(Hl), _dup(r0), 0.5, 0.0, _dup(Lc))",
         "gold_call": "_guard(_oracle_mpfc_stage_solve, _dup(phi), _dup(Hl), _dup(r0), 0.5, 0.0, _dup(Lc))"},
        # edge: a three-dimensional stage, where every transform, the measure in the
        # scalar equation and the stage symbol must all follow the rank of the data.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + FLD3 + "H3 = _oracle_mpfc_sav_terms(u3, L3)[0]\n" "r3 = float(np.sqrt(_oracle_mpfc_free_energy(u3, L3)[2] + 400.0))\n",
         "call": "mpfc_stage_solve(_dup(u3), _dup(H3), _dup(r3), 0.5, 0.125, _dup(L3))",
         "gold_call": "_oracle_mpfc_stage_solve(_dup(u3), _dup(H3), _dup(r3), 0.5, 0.125, _dup(L3))"},
    ]
