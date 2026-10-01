"""
Return the ****five indicator properties of the BCC crystal that the source paper adopts as its quantities of interest****, computed from the potential itself, in the order $[a_0,\\; E_{\\mathrm{coh}},\\; c_{11},\\; c_{12},\\; c_{44}]$ - the equilibrium lattice constant in angstrom, the cohesive energy in eV per atom, and the three cubic elastic constants in GPa. ****Nothing beyond the conventions below is derived for you:**** set up the ideal BCC lattice sum, locate its energy minimum, and obtain the elastic constants as second derivatives of the energy per atom with respect to homogeneous strain. The crystal is centrosymmetric, so no internal relaxation is needed.

The energy per atom of a monatomic Bravais crystal is a lattice sum over the shells around one site, with the pair term halved to remove the double count and the embedding term evaluated once at the site's own density. Because a BCC lattice is a simple-cubic lattice with a two-point basis, the shell list is the set of integer triples together with the same triples offset by $(\\tfrac12,\\tfrac12,\\tfrac12)$. The equilibrium lattice constant is where the derivative of that sum with respect to $a$ vanishes; because every distance scales linearly with $a$, that derivative is available analytically - $dr/da = r/a$ for every shell - which is far better conditioned than differencing the energy, and it is what makes the lattice constant reproducible to machine precision rather than to the accuracy of a minimiser. Elastic constants follow from the quadratic response of the energy density to strain: an axial strain isolates $c_{11}$, a biaxial strain gives $c_{11}+c_{12}$, and a shear gives $c_{44}$, with the customary factor of two between tensor and engineering shear that decides whether the off-diagonal strain entry is $e$ or $e/2$.

****--- Formulas ---****

Let $\\boldsymbol c$ run over the integer triples $(i,j,k)$ with $|i|,|j|,|k| \\le \\lceil r_{\\mathrm{cut}}/a\\rceil + 1$ and over the same triples plus $(\\tfrac12,\\tfrac12,\\tfrac12)$, excluding the origin; let $\\boldsymbol d = a\\,(\\mathcal I + \\boldsymbol\\varepsilon)\\boldsymbol c$ and $r = |\\boldsymbol d|$, keeping only those with $0 < r < r_{\\mathrm{cut}}$ ****after**** the strain has been applied. (Selecting the shells on the unstrained distance $a|\\boldsymbol c|$ instead is numerically identical at every instance graded here: the shell nearest the cutoff sits $4.7$ per cent inside it and the nearest one outside sits $10.1$ per cent beyond, against strains of at most $2$ per cent.) Then

$$E(a,\\boldsymbol\\varepsilon) = \\tfrac12\\sum \\phi(r) + F\\Big(\\sum f(r)\\Big), \\qquad \\frac{dE}{da}\\Big|_{\\boldsymbol\\varepsilon=0} = \\tfrac12\\sum \\phi'(r)\\,|\\boldsymbol c| + F'\\Big(\\sum f(r)\\Big)\\sum f'(r)\\,|\\boldsymbol c| .$$

$a_0$ solves $dE/da = 0$; the reference iteration is Newton from $a_start$, with the second derivative taken as $\\big[E'(a+h_a)-E'(a-h_a)\\big]/(2h_a)$, repeated $n_newton$ times. Then, with $V_0 = a_0^3/2$ the volume per atom, $E_0 = E(a_0,\\boldsymbol 0)$ and

$$D[\\boldsymbol M] = \\frac{E\\big(a_0, \\boldsymbol M(h_\\varepsilon)\\big) - 2E_0 + E\\big(a_0, \\boldsymbol M(-h_\\varepsilon)\\big)}{h_\\varepsilon^{2}},$$

$$c_{11} = \\frac{D\\big[\\mathrm{diag}(e,0,0)\\big]}{V_0}, \\qquad c_{11}+c_{12} = \\frac{D\\big[\\mathrm{diag}(e,e,0)\\big]}{2V_0}, \\qquad c_{44} = \\frac{D\\big[\\varepsilon_{yz}=\\varepsilon_{zy}=e/2\\big]}{V_0},$$

all three multiplied by $160.21766208$ to convert eV/A$^3$ to GPa, and $E_{\\mathrm{coh}} = -E_0$.

Returns
-------
`np.ndarray` of shape `(5,)`, real and finite: the lattice constant in angstrom, the cohesive energy in eV per atom (positive for a bound crystal), and $c_{11}$, $c_{12}$, $c_{44}$ in GPa.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def bcc_indicator_properties(params, r_cut=6.0, a_start=3.30, h_a=1e-4,
                             n_newton=8, h_strain=1e-2):
    """params: the same length-20 EAM parameter vector as in step 1.
    r_cut: the hard cutoff. a_start, h_a, n_newton: the Newton iteration for the
    equilibrium lattice constant. h_strain: the strain step of the second-difference
    stencil for the elastic constants.
    Return the real length-5 array [a0 (A), E_coh (eV/atom), c11, c12, c44 (GPa)]."""
    # Implement per the principle above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

EV_A3_TO_GPA = 160.21766208

def _bcc_shells(a, r_cut):
    nmax = int(np.ceil(r_cut / a)) + 1
    n = np.arange(-nmax, nmax + 1)
    I, J, K = np.meshgrid(n, n, n, indexing="ij")
    b = np.stack([I, J, K], -1).reshape(-1, 3).astype(float)
    return np.concatenate([b, b + 0.5])

def _bcc_energy(a, params, r_cut, eps=None):
    d = _bcc_shells(a, r_cut) * a
    if eps is not None:
        d = d @ (np.eye(3) + np.asarray(eps, float)).T
    r = np.sqrt((d ** 2).sum(-1))
    r = r[(r > 1e-10) & (r < r_cut)]
    phi, fd = _oracle_eam_pair_and_density(r, params, r_cut)[:2]
    return 0.5 * phi.sum() + _oracle_eam_embedding(np.array([fd.sum()]), params)[0, 0]

def _bcc_dEda(a, params, r_cut):
    c = np.linalg.norm(_bcc_shells(a, r_cut), axis=1)
    c = c[(c > 1e-10) & (c * a < r_cut)]
    r = c * a
    phi, fd, dphi, dfd = _oracle_eam_pair_and_density(r, params, r_cut)
    Fder = _oracle_eam_embedding(np.array([fd.sum()]), params)[1, 0]
    return 0.5 * (dphi * c).sum() + Fder * (dfd * c).sum()

def _oracle_bcc_indicator_properties(params, r_cut=6.0, a_start=3.30, h_a=1e-4,
                                     n_newton=8, h_strain=1e-2):
    p = _check_params(params)
    if not np.isfinite(a_start) or a_start <= 0.0:
        raise ValueError("a_start must be a finite positive scalar")
    if not (isinstance(n_newton, (int, np.integer)) or float(n_newton).is_integer()):
        raise ValueError("n_newton must be an integer")
    if int(n_newton) < 1:
        raise ValueError("n_newton must be a positive integer")
    if h_a <= 0.0 or h_strain <= 0.0:
        raise ValueError("h_a and h_strain must be positive")
    a = float(a_start)
    for _ in range(int(n_newton)):
        d1 = _bcc_dEda(a, p, r_cut)
        d2 = (_bcc_dEda(a + h_a, p, r_cut) - _bcc_dEda(a - h_a, p, r_cut)) / (2.0 * h_a)
        if not np.isfinite(d2) or d2 == 0.0:
            raise ValueError("the BCC energy has no isolated minimum for these parameters")
        a = a - d1 / d2
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError("the lattice-constant iteration left the physical range")
    e0 = _bcc_energy(a, p, r_cut)
    V0 = a ** 3 / 2.0

    def second(mk):
        return (_bcc_energy(a, p, r_cut, mk(h_strain)) - 2.0 * e0
                + _bcc_energy(a, p, r_cut, mk(-h_strain))) / h_strain ** 2

    c11 = second(lambda e: np.diag([e, 0.0, 0.0])) / V0 * EV_A3_TO_GPA
    c11p12 = second(lambda e: np.diag([e, e, 0.0])) / (2.0 * V0) * EV_A3_TO_GPA

    def m44(e):
        m = np.zeros((3, 3))
        m[1, 2] = m[2, 1] = 0.5 * e
        return m

    c44 = second(m44) / V0 * EV_A3_TO_GPA
    return np.array([a, -e0, c11, c11p12 - c11, c44])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    TA = ("TA = np.array([2.860082, 3.086341, 33.787168, 33.787168, 8.489528, 4.527748,\n"
          "              0.611679, 1.032101, 0.176977, 0.353954,\n"
          "              -5.103845, -0.405524, 1.112997, -3.585325,\n"
          "              -5.14, 0.0, 1.640098, 0.221375, 0.848843, -5.141526])\n")
    return [
        # normal: the production instance. The five values are checkable against the
        # source paper's own ground-truth column for this potential.
        {"setup": "import numpy as np\n" + TA,
         "call": "bcc_indicator_properties(TA)",
         "gold_call": "_oracle_bcc_indicator_properties(TA)"},
        # normal: r_e and A moved, which shifts the minimum and rescales the repulsion.
        {"setup": "import numpy as np\n" + TA + "P = TA.copy()\nP[0] *= 1.02\nP[6] *= 1.15\n",
         "call": "bcc_indicator_properties(P)",
         "gold_call": "_oracle_bcc_indicator_properties(P)"},
        # boundary: beta, B and kappa moved together, changing the attractive range and
        # the repulsive cutoff offset at once.
        {"setup": "import numpy as np\n" + TA + "P = TA.copy()\nP[5] *= 0.92\nP[7] *= 1.10\n"
                  "P[8] *= 1.30\n",
         "call": "bcc_indicator_properties(P)",
         "gold_call": "_oracle_bcc_indicator_properties(P)"},
        # boundary: Newton started far ABOVE the minimum with a different iteration count
        # and a different strain step - the lattice constant must converge to the same
        # stationary point, which separates a converged solve from a lucky recipe.
        {"setup": "import numpy as np\n" + TA,
         "call": "bcc_indicator_properties(TA, 6.0, 3.60, 1e-4, 12, 5e-3)",
         "gold_call": "_oracle_bcc_indicator_properties(TA, 6.0, 3.60, 1e-4, 12, 5e-3)"},
        # boundary: a 4.5 A cutoff, which drops the fourth neighbour shell entirely and
        # so changes all five properties; a hard-coded shell list fails.
        {"setup": "import numpy as np\n" + TA,
         "call": "bcc_indicator_properties(TA, 4.5)",
         "gold_call": "_oracle_bcc_indicator_properties(TA, 4.5)"},
        # edge: eta and F_e moved by 40 and 20 per cent. At the BCC equilibrium the
        # density sits in the MIDDLE branch, so all five properties are unchanged - the
        # control that establishes those two parameters are unidentifiable here.
        {"setup": "import numpy as np\n" + TA + "P = TA.copy()\nP[18] *= 1.40\nP[19] *= 1.20\n",
         "call": "bcc_indicator_properties(P)",
         "gold_call": "_oracle_bcc_indicator_properties(P)"},
        # edge: Newton started BELOW the minimum with a coarse 2 per cent strain step,
        # where the anharmonic contamination of the stencil is visible.
        {"setup": "import numpy as np\n" + TA,
         "call": "bcc_indicator_properties(TA, 6.0, 3.05, 2e-4, 10, 2e-2)",
         "gold_call": "_oracle_bcc_indicator_properties(TA, 6.0, 3.05, 2e-4, 10, 2e-2)"},
    ]
