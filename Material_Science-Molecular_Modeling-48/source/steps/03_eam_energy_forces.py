"""
Evaluate the total energy and ****all per-atom forces**** of a periodic configuration under the EAM potential of steps 1 and 2, and return them as one flat array, the energy first and then the forces in atom-major, Cartesian-minor order: $[E, F_0x, F_0y, F_0z, F_1x, ...]$, of length $3N+1$. The box is orthorhombic, given either as a scalar edge length or as a length-3 vector, and the sums run over ****all periodic images within the cutoff****, not the minimum image alone - at the box sizes used here $L < 2r_{\\mathrm{cut}}$, so an atom interacts with more than one image of some neighbours and with images of itself. The forces are graded as analytic derivatives, and their embedding contribution is the point of this step.

The energy is a pair sum plus a sum of embedding energies of the local densities, and each of the two pieces is a function of the same set of interatomic distances. Differentiating with respect to the position of atom $i$ therefore picks up two kinds of term: the pair derivative $\\phi'$, and - for every neighbour $j$ - the derivative of the embedding energy of ****both**** atoms with respect to the density each contributes to the other. That is the structural difference between an EAM and a pair potential: the coefficient multiplying $f'(r_{ij})$ is $F'(\\rho_i) + F'(\\rho_j)$, not $F'(\\rho_i)$ alone. Because $\\rho_i$ must be complete before any embedding derivative can be evaluated, the calculation is necessarily two-pass: accumulate every density first, then form the forces. A useful invariant is that the forces sum to zero to round-off, by Newton's third law, whatever the configuration.

****--- Formulas ---****

With $\\boldsymbol r_{ij}$ the shortest-image separation plus a lattice translation $\\boldsymbol S$, $r_{ij} = |\\boldsymbol r_{ij}|$, and all pairs with $0 < r_{ij} < r_{\\mathrm{cut}}$ included,

$$\\rho_i = \\sum_{j,\\boldsymbol S} f(r_{ij}), \\qquad E = \\tfrac12\\sum_{i}\\sum_{j,\\boldsymbol S} \\phi(r_{ij}) + \\sum_{i}F(\\rho_i),$$

$$\\boldsymbol F_i = -\\sum_{j,\\boldsymbol S}\\Big[\\phi'(r_{ij}) + \\big(F'(\\rho_i)+F'(\\rho_j)\\big)\\,f'(r_{ij})\\Big]\\frac{\\boldsymbol r_{ij}}{r_{ij}} ,$$

the factor $\\tfrac12$ in $E$ removing the double count of the ordered pair sum. The image translations are $\\boldsymbol S = (n_xL_x, n_yL_y, n_zL_z)$ with $|n_a| \\le \\lceil r_{\\mathrm{cut}}/L_a\\rceil$, applied after the raw separation has been wrapped into $[-L_a/2, L_a/2)$, which guarantees every image inside the cutoff is reached exactly once.

Returns
-------
`np.ndarray`, real, 1-D, of length $3N+1$. Entry $0$ is the total energy in eV; entries $1$ onwards are the forces in eV/A. The three Cartesian sums of the forces are $0$ to round-off.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def eam_energy_forces(positions, cell, params, r_cut=6.0):
    """positions: real (N, 3) array of atomic positions.
    cell: float, or length-3 sequence (Lx, Ly, Lz), the orthorhombic box edges.
    params: the same length-20 EAM parameter vector as in step 1.
    r_cut: the hard cutoff of step 1.
    Return the real 1-D array [E, F_0x, F_0y, F_0z, F_1x, ...] of length 3N + 1."""
    # Implement per the formulas above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_eam_energy_forces(positions, cell, params, r_cut=6.0):
    p = _check_params(params)
    pos = np.asarray(positions, float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if pos.shape[0] < 1:
        raise ValueError("at least one atom is required")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions must be finite")
    L = np.atleast_1d(np.asarray(cell, float))
    if L.size not in (1, 3):
        raise ValueError("cell must be a scalar or a length-3 vector of box lengths")
    L = np.broadcast_to(L, (3,)) if L.size == 3 else np.full(3, float(L[0]))
    if np.any(L <= 0.0) or not np.all(np.isfinite(L)):
        raise ValueError("all box lengths must be finite and positive")
    if not np.isfinite(r_cut) or r_cut <= 0.0:
        raise ValueError("r_cut must be a finite positive scalar")
    nmax = [int(np.ceil(r_cut / L[a])) for a in range(3)]
    g = np.meshgrid(*[np.arange(-n, n + 1) for n in nmax], indexing="ij")
    S = np.stack([g[a].ravel() * L[a] for a in range(3)], axis=1)
    d = pos[:, None, :] - pos[None, :, :]
    d = d - L * np.round(d / L)
    d = d[:, :, None, :] + S[None, None, :, :]
    r = np.sqrt((d ** 2).sum(-1))
    good = (r > 1e-10) & (r < r_cut)
    rs = np.where(good, r, 1.0)
    phi, fd, dphi, dfd = _oracle_eam_pair_and_density(rs, p, r_cut)
    phi = np.where(good, phi, 0.0)
    fd = np.where(good, fd, 0.0)
    dphi = np.where(good, dphi, 0.0)
    dfd = np.where(good, dfd, 0.0)
    rho = fd.sum(axis=(1, 2))
    Fval, Fder = _oracle_eam_embedding(rho, p)
    E = 0.5 * phi.sum() + Fval.sum()
    coef = 0.5 * dphi + 0.5 * (Fder[:, None, None] + Fder[None, :, None]) * dfd
    frc = -2.0 * (coef / rs)[..., None] * d
    frc = np.where(good[..., None], frc, 0.0).sum(axis=(1, 2))
    return np.concatenate([[E], frc.ravel()])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    TA = ("TA = np.array([2.860082, 3.086341, 33.787168, 33.787168, 8.489528, 4.527748,\n"
          "              0.611679, 1.032101, 0.176977, 0.353954,\n"
          "              -5.103845, -0.405524, 1.112997, -3.585325,\n"
          "              -5.14, 0.0, 1.640098, 0.221375, 0.848843, -5.141526])\n")
    CFG = ("def _cfg(m, n_cell=2):\n"
           "    a = 3.05 + 0.06 * (m % 8); L = a * n_cell\n"
           "    amp = 0.08 + 0.04 * (m % 4)\n"
           "    c = np.arange(n_cell, dtype=float)\n"
           "    I, J, K = np.meshgrid(c, c, c, indexing='ij')\n"
           "    b = np.stack([I, J, K], -1).reshape(-1, 3)\n"
           "    R = np.concatenate([b, b + 0.5]) * a\n"
           "    i = np.arange(R.shape[0], dtype=float)\n"
           "    u = amp * np.stack([np.sin(1.7*i + 0.9*m + 0.3),\n"
           "                        np.sin(2.3*i + 1.4*m + 1.1),\n"
           "                        np.sin(3.1*i + 2.2*m + 1.9)], axis=1)\n"
           "    return (R + u) % L, np.full(3, L)\n")
    return [
        # normal: a mid-range configuration of the production pool, middle branch of F.
        {"setup": "import numpy as np\n" + TA + CFG + "pos, cell = _cfg(4)\n",
         "call": "eam_energy_forces(pos, cell, TA)",
         "gold_call": "_oracle_eam_energy_forces(pos, cell, TA)"},
        # boundary: the most compressed configuration, whose densities reach 51.9 and so
        # exercise the third branch of the embedding function for every atom.
        {"setup": "import numpy as np\n" + TA + CFG + "pos, cell = _cfg(0)\n",
         "call": "eam_energy_forces(pos, cell, TA)",
         "gold_call": "_oracle_eam_energy_forces(pos, cell, TA)"},
        # boundary: the most expanded configuration, densities down to 23.2, first branch.
        {"setup": "import numpy as np\n" + TA + CFG + "pos, cell = _cfg(7)\n",
         "call": "eam_energy_forces(pos, cell, TA)",
         "gold_call": "_oracle_eam_energy_forces(pos, cell, TA)"},
        # boundary: a 3x3x3 supercell, 54 atoms, where the number of image shells needed
        # per axis changes and a hard-coded 16-atom or 27-image loop breaks.
        {"setup": "import numpy as np\n" + TA + CFG + "pos, cell = _cfg(2, 3)\n",
         "call": "eam_energy_forces(pos, cell, TA)",
         "gold_call": "_oracle_eam_energy_forces(pos, cell, TA)"},
        # boundary: a RECTANGULAR box. Each axis carries its own length, so the wrap and
        # the image shell count differ per direction.
        {"setup": "import numpy as np\n" + TA + CFG + "pos, cell = _cfg(5)\n"
                  "cell = np.array([cell[0], cell[1]*1.20, cell[2]*0.90])\npos = pos % cell\n",
         "call": "eam_energy_forces(pos, cell, TA)",
         "gold_call": "_oracle_eam_energy_forces(pos, cell, TA)"},
        # edge: ONE atom in a 3 A cube - a simple-cubic crystal of one atom per cell. All
        # interactions are with its own images; the force vanishes by symmetry but the
        # energy does not, so an implementation that skips i == j entirely returns zero.
        {"setup": "import numpy as np\n" + TA + "pos = np.zeros((1, 3))\n",
         "call": "eam_energy_forces(pos, 3.0, TA)",
         "gold_call": "_oracle_eam_energy_forces(pos, 3.0, TA)"},
        # edge: two atoms 1.6 A apart in a 5 A box, deep on the repulsive wall and with
        # many image contributions; the two forces must be exactly opposite.
        {"setup": "import numpy as np\n" + TA + "pos = np.array([[0., 0., 0.], [1.6, 0.2, 0.1]])\n",
         "call": "eam_energy_forces(pos, 5.0, TA)",
         "gold_call": "_oracle_eam_energy_forces(pos, 5.0, TA)"},
        # boundary: a short cutoff, where fewer image shells are needed and the truncation
        # of step 1 must be honoured rather than a fixed 6 A.
        {"setup": "import numpy as np\n" + TA + CFG + "pos, cell = _cfg(9)\n",
         "call": "eam_energy_forces(pos, cell, TA, 3.5)",
         "gold_call": "_oracle_eam_energy_forces(pos, cell, TA, 3.5)"},
    ]
