"""
Return the total potential energy of an adsorbate-on-slab configuration and the forces on the adsorbate atoms, for the surrogate calculator that stands in this task for a machine-learned interatomic potential. Two contributions are summed. The first is a pairwise Morse interaction, tapered smoothly to zero between an inner and an outer radius so that the periodic minimum-image sum is continuous everywhere inside the cell. The second is a bond-order over-coordination penalty that charges an energy whenever the smoothly counted number of bonds at an adsorbate atom exceeds the valence of its element; without it a purely pairwise model has no valence, the reactant and the product of a hydrogen transfer relax into the same structure, and there is no barrier between them. The slab is frozen, so pairs whose two atoms both belong to the slab are omitted and no force is returned for them, and the coordination sum runs over adsorbate neighbours only. The two in-plane directions are periodic, the surface normal is not, and a leading image axis is accepted so that a whole band of images is evaluated in one call.

Automated reaction-barrier workflows became affordable only when machine-learned interatomic potentials replaced density-functional theory in the inner loop: the workflow needs energies and forces for many thousands of configurations, and its logic is indifferent to where they come from as long as the calculator is deterministic, differentiable and periodic. The classical stand-in for such a calculator is a Morse pair potential, which reproduces a bond's depth, equilibrium length and anharmonicity with three parameters per element pair. A pair potential alone, however, is saturation-free: nothing stops an atom from bonding to every neighbour at once, so a transferring hydrogen sees no barrier between a donor and an acceptor. Bond-order and embedded-atom models repair this by making the energy depend on a smoothly counted coordination number rather than on distances alone. A quadratic penalty on coordination in excess of a prescribed valence is the simplest such repair, and it is what separates the two adsorbed states of a surface transfer reaction and creates the saddle between them. Because a periodic sum must not jump when a neighbour crosses the half-cell distance, both the pair term and the coordination counter are multiplied by, respectively evaluated through, a cosine switching function that is one inside an inner radius, zero outside an outer radius and continuously differentiable at both ends.

With $S$ the cosine switch, $D$, $\\alpha$, $r^{0}$ the Morse matrices, $b$ the bond-length matrix, $v$ the valences and $\\kappa$ the penalty strengths,



$$E=\\sum_{\\substack{i<j\\\\ \\text{not both frozen}}} S(r_{ij})\\,D_{s_is_j}\\Big(e^{-2\\alpha_{s_is_j}(r_{ij}-r^{0}_{s_is_j})}-2e^{-\\alpha_{s_is_j}(r_{ij}-r^{0}_{s_is_j})}\\Big)\\;+\\;\\sum_{i\\in\\text{ads}}\\kappa_{s_i}\\big[\\max(0,\\,c_i-v_{s_i})\\big]^{2},$$



$$c_i=\\sum_{\\substack{j\\in\\text{ads},\\,j\\neq i\\\\ b_{s_is_j}>0}} f\\!\\left(r_{ij};\\,f_1b_{s_is_j},\\,f_2b_{s_is_j}\\right),\\qquad S(r)=f(r;r_{\\mathrm{in}},r_{\\mathrm{out}}),$$



$$f(r;r_1,r_2)=\\tfrac12\\Big[1+\\cos\\!\\big(\\pi\\,\\mathrm{clip}\\big(\\tfrac{r-r_1}{r_2-r_1},0,1\\big)\\big)\\Big].$$



All $r_{ij}$ are minimum-image distances in $x$ and $y$; $z$ is not periodic. The forces are $-\\partial E/\\partial \\mathbf{r}_i$ on the adsorbate atoms.

Returns
-------
A tuple `(energy, forces)`: `energy` is a Python `float` for a single configuration and an array of shape $(M,)$ for a band of $M$ images; `forces` has the shape of `ads`, in eV/A.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def surface_energy_forces(ads: "np.ndarray", slab: "np.ndarray",
                          ads_species: "np.ndarray",
                          slab_species: "np.ndarray", cell: "np.ndarray",
                          de: "np.ndarray", alpha: "np.ndarray",
                          r0: "np.ndarray", r_bond: "np.ndarray",
                          valence: "np.ndarray", k_coord: "np.ndarray",
                          f1: float = 1.3, f2: float = 2.0,
                          r_in: float = 4.0, r_out: float = 5.0) -> tuple:
    """Energy and adsorbate forces of the tapered Morse plus coordination model.

    Parameters
    ----------
    ads : numpy.ndarray
        Adsorbate positions in angstrom, shape (N, 3) or (M, N, 3) for a band
        of M images, with N >= 1.
    slab : numpy.ndarray
        Frozen slab positions in angstrom, shape (F, 3); F may be zero.
    ads_species : numpy.ndarray
        Integer species index of each adsorbate atom, shape (N,).
    slab_species : numpy.ndarray
        Integer species index of each slab atom, shape (F,).
    cell : numpy.ndarray
        The two positive in-plane periods (Lx, Ly) in angstrom.
    de, alpha, r0 : numpy.ndarray
        Symmetric (S, S) Morse well depths in eV, decay constants in 1/A and
        equilibrium lengths in A. alpha and r0 must be positive.
    r_bond : numpy.ndarray
        Symmetric non-negative (S, S) reference bond lengths of the
        coordination counter. A zero entry excludes that species pair from
        the coordination sum.
    valence : numpy.ndarray
        Non-negative valence of each species, shape (S,).
    k_coord : numpy.ndarray
        Non-negative over-coordination penalty strength of each species in eV,
        shape (S,).
    f1, f2 : float
        Multiples of r_bond at which the coordination switch leaves one and
        reaches zero; f2 must exceed f1 and both must be positive.
    r_in, r_out : float
        Inner and outer radius in A of the pair-energy taper; r_out must
        exceed r_in and both must be positive.

    Returns
    -------
    result : tuple
        (energy, forces). energy is a float for a single configuration and an
        array of shape (M,) for a band; forces has the shape of ads and
        holds the forces on the adsorbate atoms in eV/A.

    Raises
    ------
    ValueError
        If ads, slab, the species arrays or cell have the wrong shape, if a
        parameter matrix is not square, not symmetric or of a different shape
        from the others, if alpha or r0 is not positive, if r_bond, valence or
        k_coord is negative, if valence or k_coord has the wrong length, if
        f2 <= f1 or r_out <= r_in, or if a species index is out of range.
    """
    return energy, forces

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _switch(d, r1, r2):
    """Smooth 1 -> 0 coordination switch and its derivative with respect to d."""
    x = np.clip((d - r1) / (r2 - r1), 0.0, 1.0)
    f = 0.5 * (1.0 + np.cos(np.pi * x))
    inside = (d > r1) & (d < r2)
    df = np.where(inside,
                  -0.5 * np.pi / (r2 - r1) * np.sin(np.pi * x), 0.0)
    return f, df


def _oracle_surface_energy_forces(ads: "np.ndarray", slab: "np.ndarray",
                                  ads_species: "np.ndarray",
                                  slab_species: "np.ndarray",
                                  cell: "np.ndarray", de: "np.ndarray",
                                  alpha: "np.ndarray", r0: "np.ndarray",
                                  r_bond: "np.ndarray",
                                  valence: "np.ndarray", k_coord: "np.ndarray",
                                  f1: float = 1.3, f2: float = 2.0,
                                  r_in: float = 4.0,
                                  r_out: float = 5.0) -> tuple:
    ads = np.asarray(ads, dtype=float)
    slab = np.asarray(slab, dtype=float)
    ads_species = np.asarray(ads_species, dtype=int)
    slab_species = np.asarray(slab_species, dtype=int)
    de = np.asarray(de, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    r0 = np.asarray(r0, dtype=float)
    r_bond = np.asarray(r_bond, dtype=float)
    valence = np.asarray(valence, dtype=float).reshape(-1)
    k_coord = np.asarray(k_coord, dtype=float).reshape(-1)
    cell = np.asarray(cell, dtype=float).reshape(-1)

    if ads.ndim not in (2, 3) or ads.shape[-1] != 3 or ads.shape[-2] < 1:
        raise ValueError("ads must have shape (N, 3) or (M, N, 3) with N >= 1")
    if slab.ndim != 2 or slab.shape[-1] != 3:
        raise ValueError("slab must have shape (F, 3)")
    if ads_species.shape != (ads.shape[-2],):
        raise ValueError("ads_species must have one entry per adsorbate atom")
    if slab_species.shape != (slab.shape[0],):
        raise ValueError("slab_species must have one entry per slab atom")
    if cell.shape != (2,) or not np.all(cell > 0.0):
        raise ValueError("cell must be two positive in-plane lengths")
    for name, arr in (("de", de), ("alpha", alpha), ("r0", r0),
                      ("r_bond", r_bond)):
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(name + " must be a square matrix")
        if not np.allclose(arr, arr.T, rtol=0.0, atol=1e-12):
            raise ValueError(name + " must be symmetric")
    if not (de.shape == alpha.shape == r0.shape == r_bond.shape):
        raise ValueError("de, alpha, r0 and r_bond must have the same shape")
    if np.any(alpha <= 0.0) or np.any(r0 <= 0.0):
        raise ValueError("alpha and r0 must be positive")
    if np.any(r_bond < 0.0):
        raise ValueError("r_bond must be non-negative")
    nsp = de.shape[0]
    if valence.shape != (nsp,) or k_coord.shape != (nsp,):
        raise ValueError("valence and k_coord must have one entry per species")
    if np.any(k_coord < 0.0) or np.any(valence < 0.0):
        raise ValueError("valence and k_coord must be non-negative")
    if not (f2 > f1 > 0.0):
        raise ValueError("f2 must exceed f1 and both must be positive")
    if not (r_out > r_in > 0.0):
        raise ValueError("r_out must exceed r_in and both must be positive")
    if ads_species.size and (ads_species.min() < 0 or ads_species.max() >= nsp):
        raise ValueError("ads_species out of range of the parameter matrices")
    if slab_species.size and (slab_species.min() < 0 or slab_species.max() >= nsp):
        raise ValueError("slab_species out of range of the parameter matrices")

    flat = ads.reshape(-1, ads.shape[-2], 3)
    nimg, n = flat.shape[0], flat.shape[1]
    lx, ly = float(cell[0]), float(cell[1])

    energy = np.zeros(nimg)
    forces = np.zeros((nimg, n, 3))

    def _wrap(dvec):
        dvec[..., 0] -= np.round(dvec[..., 0] / lx) * lx
        dvec[..., 1] -= np.round(dvec[..., 1] / ly) * ly
        return dvec

    def _morse(d, sp_a, sp_b):
        """Morse pair energy tapered to zero between r_in and r_out."""
        ex = np.exp(-alpha[sp_a, sp_b] * (d - r0[sp_a, sp_b]))
        raw = de[sp_a, sp_b] * (ex * ex - 2.0 * ex)
        draw = 2.0 * de[sp_a, sp_b] * alpha[sp_a, sp_b] * (ex - ex * ex)
        cut, dcut = _switch(d, r_in, r_out)
        return raw * cut, draw * cut + raw * dcut

    if n > 1:
        ii, jj = np.triu_indices(n, k=1)
        dvec = _wrap(flat[:, ii, :] - flat[:, jj, :])
        d = np.maximum(np.sqrt(np.sum(dvec * dvec, axis=-1)), 1e-12)
        sa, sb = ads_species[ii], ads_species[jj]
        e, dedr = _morse(d, sa, sb)
        energy += e.sum(axis=1)

        rb = r_bond[sa, sb]
        live = rb > 0.0
        rb = np.where(live, rb, 1.0)
        f, df = _switch(d, f1 * rb, f2 * rb)
        f = np.where(live, f, 0.0)
        df = np.where(live, df, 0.0)
        coord = np.zeros((nimg, n))
        np.add.at(coord, (slice(None), ii), f)
        np.add.at(coord, (slice(None), jj), f)
        excess = np.maximum(coord - valence[ads_species][None, :], 0.0)
        energy += np.sum(k_coord[ads_species][None, :] * excess ** 2, axis=1)
        dedc = 2.0 * k_coord[ads_species][None, :] * excess
        dedr = dedr + (dedc[:, ii] + dedc[:, jj]) * df

        grad = (dedr / d)[..., None] * dvec
        np.add.at(forces, (slice(None), ii), -grad)
        np.add.at(forces, (slice(None), jj), grad)

    if slab.shape[0] > 0:
        dvec = _wrap(flat[:, :, None, :] - slab[None, None, :, :])
        d = np.maximum(np.sqrt(np.sum(dvec * dvec, axis=-1)), 1e-12)
        e, dedr = _morse(d, ads_species[:, None], slab_species[None, :])
        energy += e.sum(axis=(1, 2))
        forces -= np.sum((dedr / d)[..., None] * dvec, axis=2)

    if ads.ndim == 2:
        return float(energy[0]), forces[0]
    return energy.reshape(ads.shape[:-2]), forces.reshape(ads.shape)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    shared = """import numpy as np

def pack(result):
    # flatten a tuple of arrays of different shapes into one 1-D array
    return np.concatenate([np.asarray(p, dtype=float).ravel() for p in result])

def slab_of(lattice, n_cell, spacing, n_layer):
    # ABAB-stacked square-lattice slab with its top layer at z = 0
    idx = np.arange(n_cell, dtype=float)
    gx, gy = np.meshgrid(idx, idx, indexing="ij")
    out = []
    for lay in range(n_layer):
        off = 0.5 * (lay % 2)
        out.append(np.stack([(gx.ravel() + off) * lattice,
                             (gy.ravel() + off) * lattice,
                             np.full(gx.size, -lay * spacing)], axis=1))
    return np.concatenate(out, axis=0)

DE = np.array([[0.00, 0.25, 0.30, 0.12],
               [0.25, 3.50, 0.30, 3.50],
               [0.30, 0.30, 2.50, 4.00],
               [0.12, 3.50, 4.00, 0.30]])
AL = np.ones((4, 4))
R0 = np.array([[2.80, 2.90, 2.80, 2.60],
               [2.90, 1.55, 1.45, 1.15],
               [2.80, 1.45, 1.50, 1.05],
               [2.60, 1.15, 1.05, 0.95]])
RB = np.array([[0.00, 0.00, 0.00, 0.00],
               [0.00, 1.55, 0.00, 1.15],
               [0.00, 0.00, 1.50, 1.05],
               [0.00, 1.15, 1.05, 0.00]])
VAL = np.array([0.0, 4.0, 2.0, 1.0])
KC = np.array([0.0, 2.0, 2.0, 2.0])
SP = np.array([1, 3, 3, 3, 2, 3])
ADS = np.array([[1.40, 1.40, 1.90], [2.43, 1.40, 2.26],
                [0.88, 2.29, 2.26], [0.88, 0.51, 2.26],
                [4.20, 2.80, 1.80], [4.20, 2.80, 2.77]])
"""
    args = ("slab, SP, np.zeros(slab.shape[0], dtype=int), cell, "
            "DE, AL, R0, RB, VAL, KC")
    return [
        # normal: the benchmark slab and a methyl plus hydroxyl configuration
        {"setup": shared + """
slab = slab_of(2.8, 4, 1.4, 3)
cell = np.array([11.2, 11.2])
ads = ADS.copy()
""",
         "call": "pack(surface_energy_forces(ads, %s))" % args,
         "gold_call": "pack(_oracle_surface_energy_forces(ads, %s))" % args},
        # normal: a three-image band, so the leading image axis is exercised
        {"setup": shared + """
slab = slab_of(2.8, 4, 1.4, 3)
cell = np.array([11.2, 11.2])
shift = np.array([[0.0, 0.0, 0.0]] * 6)
shift[3] = [0.7, 0.35, -0.1]
ads = np.stack([ADS, ADS + 0.5 * shift, ADS + shift])
""",
         "call": "pack(surface_energy_forces(ads, %s))" % args,
         "gold_call": "pack(_oracle_surface_energy_forces(ads, %s))" % args},
        # boundary: the transferring hydrogen midway between donor and acceptor,
        # where the over-coordination penalty is at its largest
        {"setup": shared + """
slab = slab_of(2.8, 4, 1.4, 3)
cell = np.array([11.2, 11.2])
ads = ADS.copy()
ads[4] = [3.20, 2.10, 1.85]
ads[5] = [3.60, 2.90, 2.55]
ads[3] = 0.5 * (ads[0] + ads[4]) + np.array([0.0, 0.0, 0.45])
""",
         "call": "pack(surface_energy_forces(ads, %s))" % args,
         "gold_call": "pack(_oracle_surface_energy_forces(ads, %s))" % args},
        # boundary: an atom pair straddling the periodic boundary, so the
        # minimum-image convention decides which image is seen
        {"setup": shared + """
slab = slab_of(2.8, 4, 1.4, 3)
cell = np.array([11.2, 11.2])
ads = ADS.copy()
ads[:, 0] = np.mod(ads[:, 0] - 1.3, 11.2)
""",
         "call": "pack(surface_energy_forces(ads, %s))" % args,
         "gold_call": "pack(_oracle_surface_energy_forces(ads, %s))" % args},
        # boundary: every pair beyond the outer taper radius, so the Morse sum
        # vanishes and only the coordination term can survive
        {"setup": shared + """
slab = np.zeros((0, 3))
cell = np.array([40.0, 40.0])
ads = np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 9.0, 0.0],
                [0.0, 0.0, 9.0], [9.0, 9.0, 0.0], [9.0, 0.0, 9.0]])
""",
         "call": "pack(surface_energy_forces(ads, %s))" % args,
         "gold_call": "pack(_oracle_surface_energy_forces(ads, %s))" % args},
        # edge: a single adsorbate atom above a one-layer slab, so there is no
        # adsorbate-adsorbate pair and no coordination penalty at all
        {"setup": shared + """
slab = slab_of(2.8, 3, 1.4, 1)
cell = np.array([8.4, 8.4])
ads = np.array([[1.4, 1.4, 1.7]])
one = np.array([2])
""",
         "call": ("pack(surface_energy_forces(ads, slab, one, "
                  "np.zeros(slab.shape[0], dtype=int), cell, DE, AL, R0, RB, "
                  "VAL, KC))"),
         "gold_call": ("pack(_oracle_surface_energy_forces(ads, slab, one, "
                       "np.zeros(slab.shape[0], dtype=int), cell, DE, AL, R0, "
                       "RB, VAL, KC))")},
        # invalid: a non-symmetric Morse matrix
        {"setup": shared + """
slab = slab_of(2.8, 3, 1.4, 1)
cell = np.array([8.4, 8.4])
bad = DE.copy()
bad[0, 1] = 9.0
def run_model():
    try:
        surface_energy_forces(ADS, slab, SP,
                              np.zeros(slab.shape[0], dtype=int), cell,
                              bad, AL, R0, RB, VAL, KC)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_surface_energy_forces(ADS, slab, SP,
                                      np.zeros(slab.shape[0], dtype=int), cell,
                                      bad, AL, R0, RB, VAL, KC)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
        # invalid: an outer taper radius that does not exceed the inner one
        {"setup": shared + """
slab = slab_of(2.8, 3, 1.4, 1)
cell = np.array([8.4, 8.4])
def run_model():
    try:
        surface_energy_forces(ADS, slab, SP,
                              np.zeros(slab.shape[0], dtype=int), cell,
                              DE, AL, R0, RB, VAL, KC, 1.3, 2.0, 5.0, 5.0)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_surface_energy_forces(ADS, slab, SP,
                                      np.zeros(slab.shape[0], dtype=int), cell,
                                      DE, AL, R0, RB, VAL, KC, 1.3, 2.0,
                                      5.0, 5.0)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
    ]
