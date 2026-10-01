"""
Run the whole automated transition-state workflow on the benchmark surface reaction and return the forward barrier of the pathway it selects. The slab, the two adsorbate ensembles and the reaction are those of the benchmark: a hydrogen transfer from a methyl group to a hydroxyl group on a square-lattice (100) surface, with the atom order carbon, three hydrogens, oxygen, hydrogen and the third hydrogen transferring. The stages are, in order: enumerate both ensembles by placing the two fragments on the four in-cell site types with the fragment geometries, heights and in-plane rotation given; relax them all with the supplied calculator; discard structures that have changed their bonding pattern, sunk below the top layer or desorbed; run the two-objective geometry selection on energy and donor-acceptor separation; remove structural duplicates and keep the lowest-energy representatives; align every final state to every initial state and augment it by the point-group operations of the slab; assign the periodic images, build the coarse interpolation of every pairing and rank the pairings by the ranking metric; re-rank the best of those by the transfer metric; relax the modified final states and apply the energy window; recompute both metrics on the dense interpolation and take the best; and finally optimise that band with variable springs and a climbing image. The returned value is the largest image energy of the optimised band minus the energy of its first image.

The individual pieces of an automated barrier search are each unremarkable; what makes the workflow work is the order in which they cut the combinatorial space down. Enumerating adsorption geometries for two fragments on a surface already gives tens of structures per state, pairing them gives hundreds, augmenting each pairing by the point-group operations of the slab gives thousands, and multiplying by the distinct atom mappings gives more than any band optimiser can absorb. Each stage therefore has to be cheaper than the one it feeds and has to discard candidates for a reason that the later stages would have discovered anyway. Structural filters come first because they cost nothing and remove structures that are not the requested chemistry at all. The two-objective selection comes next because it needs only one energy and one distance per structure. Symmetry augmentation then deliberately increases the count, because it is the only stage that can produce a pairing better than anything global optimisation returned, and the interpolation metrics immediately cut that increase back down at the cost of a few interpolation-potential evaluations. Only the handful of survivors ever reach a band optimisation, which is the one stage whose cost is measured in thousands of calculator calls. The final number a workflow of this kind reports is the forward barrier of the pathway it selected, and its value is a statement about the whole cascade rather than about any single stage.



Conventions fixed by this task. Every relaxation and band optimisation uses the fast inertial relaxation engine with the parameters of the benchmark, each independent system (a candidate geometry, or an image of a band) carrying its own time step, mixing parameter, velocity and count of positive steps, velocities starting at zero and systems relaxed together sharing one stopping test on the largest force vector over all of them, applied before each step. One step of one system, with $P=\\mathbf F\\cdot\\mathbf v$ over that system, is: if $P>0$, mix $\\mathbf v\\leftarrow(1-\\alpha)\\mathbf v+\\alpha\\lVert\\mathbf v\\rVert\\,\\mathbf F/\\lVert\\mathbf F\\rVert$, then, if more than $N_{\\min}$ consecutive earlier steps had $P>0$, set $\\Delta t\\leftarrow\\min(f_{\\mathrm{inc}}\\Delta t,\\Delta t_{\\max})$ and $\\alpha\\leftarrow f_a\\alpha$, and count this step as positive; otherwise set $\\mathbf v\\leftarrow0$, $\\Delta t\\leftarrow f_{\\mathrm{dec}}\\Delta t$, $\\alpha\\leftarrow a_{\\mathrm{start}}$ and reset the count to zero. Then $\\mathbf v\\leftarrow\\mathbf v+\\Delta t\\,\\mathbf F$, the displacement is $\\Delta t\\,\\mathbf v$, and when its norm over the whole system exceeds $0.1$ A both the displacement and the velocity are rescaled by the same factor. The parameters are $\\Delta t_0=0.1$, $\\Delta t_{\\max}=0.3$, $N_{\\min}=5$, $f_{\\mathrm{inc}}=1.1$, $f_{\\mathrm{dec}}=0.5$, $a_{\\mathrm{start}}=0.1$ and $f_a=0.99$. The anchoring surface atom of a state is chosen by minimum-image distance to the midpoint of its two fragment centres of mass; slab atoms within 1e-3 A of the smallest distance are tied and the tie goes to the larger slab index.

The reported quantity is



$$\\Delta E^{\\ddagger}=\\max_{m} E\\big(\\mathbf{R}^{\\mathrm{opt}}_m\\big)-E\\big(\\mathbf{R}^{\\mathrm{opt}}_0\\big),$$



where $\\mathbf{R}^{\\mathrm{opt}}$ is the band produced by the two-phase optimisation of the selected interpolation. The two phases are a non-climbing pass, stopped when the largest band-force vector falls below $f_{\\mathrm{switch}}$ or after $\\mathrm{steps}_{\\mathrm{pre}}$ steps, and a climbing pass on the image that is highest in energy at the moment of the switch, stopped when the largest band-force vector falls below $f_{\\mathrm{conv}}$ or after $\\mathrm{steps}_{\\mathrm{ci}}$ steps; the spring constants are rebuilt from the current image energies at every force evaluation, so that the band force stays a continuous function of the images. Ordering inside the selection cascade is by the pair $(\\mu,\\,\\text{index})$ first, then by $(\\tau,\\,\\text{index})$, then by $(\\mu,\\tau,\\text{index})$ on the dense interpolations.

Returns
-------
A Python `float`, the forward barrier $\\Delta E^{\\ddagger}$ of the selected pathway in eV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_nebscape(de: "np.ndarray", alpha: "np.ndarray", r0: "np.ndarray",
                 r_bond: "np.ndarray", valence: "np.ndarray",
                 k_coord: "np.ndarray", lattice: float = 2.8,
                 n_cell: int = 4, n_layer: int = 3, z_layer: float = 1.4,
                 z_c: float = 2.05, z_o: float = 1.95, tilt: float = 25.0,
                 z_max: float = 6.0,
                 n_rep: int = 3, n_coarse: int = 6, n_fine: int = 20,
                 n_neb: int = 20, n_pre: int = 10, e_window: float = 1.2,
                 d_window: float = 4.5, de_window: float = 0.1,
                 mic_alpha: float = 0.68, k_min: float = 0.1,
                 k_max: float = 4.0, f_switch: float = 0.2,
                 f_conv: float = 1e-4, f_relax: float = 1e-5,
                 steps_relax: int = 10000, steps_pre: int = 30000,
                 steps_ci: int = 30000) -> float:
    """Forward barrier of the pathway the automated workflow selects.

    Parameters
    ----------
    de, alpha, r0, r_bond : numpy.ndarray
        Symmetric (S, S) calculator matrices: Morse well depths in eV, decay
        constants in 1/A, equilibrium lengths in A and coordination bond
        lengths in A.
    valence, k_coord : numpy.ndarray
        Valence and over-coordination penalty strength in eV of each species,
        shape (S,).
    lattice : float
        Positive square-lattice constant of the slab in angstrom.
    n_cell : int
        Number of surface cells per direction, at least 2.
    n_layer : int
        Number of slab layers, at least 1.
    z_layer : float
        Positive interlayer spacing in angstrom.
    z_c, z_o : float
        Heights in angstrom at which the carbon and the oxygen fragments are
        placed before relaxation.
    tilt : float
        Angle in degrees through which each rigid fragment is rotated about
        the vertical axis of its own site before placement, measured from the
        x axis towards the y axis.
    z_max : float
        Height in angstrom above which an adsorbate atom counts as desorbed.
    n_rep : int
        Number of lowest-energy representatives kept per ensemble, at least 1.
    n_coarse, n_fine, n_neb : int
        Image counts of the coarse ranking interpolation, the dense ranking
        interpolation and the optimised band; each at least 3.
    n_pre : int
        Number of best candidates under the ranking metric that are re-ranked
        by the transfer metric, at least 1.
    e_window, d_window : float
        Energy window in eV and separation cut-off in angstrom of the
        geometry selection.
    de_window : float
        Energy window in eV within which a relaxed modified final state is
        accepted.
    mic_alpha : float
        Length weight of the periodic-image ranking.
    k_min, k_max : float
        Spring constant range in eV/A^2.
    f_switch, f_conv : float
        Band-force thresholds in eV/A at which the climbing image is switched
        on and at which the band counts as converged.
    f_relax : float
        Force threshold in eV/A of the geometry relaxations.
    steps_relax, steps_pre, steps_ci : int
        Step budgets of the geometry relaxations, of the non-climbing band
        pass and of the climbing band pass.

    Returns
    -------
    barrier : float
        Forward barrier of the selected pathway in eV, the largest image
        energy of the optimised band minus the energy of its first image.

    Raises
    ------
    ValueError
        If lattice or z_layer is not positive, if n_cell is below 2 or
        n_layer below 1, if any image count is below 3, if n_rep or n_pre is
        below 1, if no candidate of either ensemble survives the structure
        filter, or if no interpolation survives the selection cascade.
    """
    return barrier

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _fire_relax(positions, force_fn, fmax, max_steps, batch=False, dt0=0.1,
                dt_max=0.3, n_min=5, f_inc=1.1, f_dec=0.5, a_start=0.1,
                f_a=0.99, max_step=0.1):
    """FIRE relaxation; with batch=True axis 0 holds independent systems."""
    pos = np.array(positions, dtype=float, copy=True)
    vel = np.zeros_like(pos)
    axes = tuple(range(1, pos.ndim)) if batch else tuple(range(pos.ndim))
    nsys = pos.shape[0] if batch else 1
    tail = (1,) * (pos.ndim - 1)
    dt = np.full(nsys, float(dt0))
    a = np.full(nsys, float(a_start))
    n_pos = np.zeros(nsys, dtype=int)

    def _cast(v):
        return np.asarray(v, dtype=float).reshape((nsys,) + tail)

    for _ in range(int(max_steps)):
        f = force_fn(pos)
        if float(np.max(np.sqrt(np.sum(f * f, axis=-1)))) < fmax:
            break
        vf = np.sum(f * vel, axis=axes).reshape(nsys)
        nf = np.maximum(np.sqrt(np.sum(f * f, axis=axes)).reshape(nsys), 1e-300)
        nv = np.sqrt(np.sum(vel * vel, axis=axes)).reshape(nsys)
        up = vf > 0.0
        vel = np.where(_cast(up), (1.0 - _cast(a)) * vel + _cast(a / nf * nv) * f,
                       0.0)
        grow = up & (n_pos > n_min)
        dt = np.where(up, np.where(grow, np.minimum(dt * f_inc, dt_max), dt),
                      dt * f_dec)
        a = np.where(up, np.where(grow, a * f_a, a), a_start)
        n_pos = np.where(up, n_pos + 1, 0)
        vel = vel + _cast(dt) * f
        dr = _cast(dt) * vel
        nrm = np.sqrt(np.sum(dr * dr, axis=axes)).reshape(nsys)
        scale = _cast(np.where(nrm > max_step,
                              max_step / np.maximum(nrm, 1e-300), 1.0))
        vel = vel * scale
        pos = pos + dr * scale
    return pos


def _build_slab(lattice, n_cell, z_layer, n_layer=3):
    """Square-lattice (100) slab, ABAB stacked, whose top layer lies at z = 0."""
    idx = np.arange(n_cell, dtype=float)
    gx, gy = np.meshgrid(idx, idx, indexing="ij")
    out = []
    for lay in range(int(n_layer)):
        off = 0.5 * (lay % 2)
        out.append(np.stack([(gx.ravel() + off) * lattice,
                             (gy.ravel() + off) * lattice,
                             np.full(gx.size, -lay * float(z_layer))], axis=1))
    return np.concatenate(out, axis=0)


def _c4v_operations():
    """The eight point-group operations of the square surface lattice."""
    ops = []
    for r in range(4):
        ang = 0.5 * np.pi * r
        c, s = float(round(np.cos(ang))), float(round(np.sin(ang)))
        ops.append(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))
    mirror = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    for r in range(4):
        ops.append(ops[r] @ mirror)
    return np.asarray(ops, dtype=float)


def _candidate_pool(state, lattice, z_c, z_o, cells, tilt):
    """Deterministic enumeration of adsorbate placements on the square lattice.

    Atom order is C, H1, H2, H3, O, H4 in both states; H3 belongs to the methyl
    group in the initial state and to the water molecule in the final state.
    Every fragment is rotated about the vertical axis through its own site by
    ``tilt`` degrees before placement, which takes each one off every mirror
    plane of the square lattice; without it eight of the sixteen initial-state
    candidates would be invariant under the mirror that exchanges H2 and H3,
    and their relaxation would have to break an exact degeneracy.
    """
    sites = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
    ch3 = np.array([[0.0, 0.0, 0.0], [1.03, 0.0, 0.36],
                    [-0.515, 0.892, 0.36], [-0.515, -0.892, 0.36]])
    ch2 = np.array([[0.0, 0.0, 0.0], [1.03, 0.0, 0.36], [-0.515, 0.892, 0.36]])
    oh = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]])
    h2o = np.array([[-0.757, 0.0, 0.586], [0.0, 0.0, 0.0], [0.757, 0.0, 0.586]])
    ang = np.radians(float(tilt))
    rot = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                    [np.sin(ang), np.cos(ang), 0.0],
                    [0.0, 0.0, 1.0]])
    ch3, ch2, oh, h2o = (f @ rot.T for f in (ch3, ch2, oh, h2o))
    out = []
    for p in sites:
        for q in sites:
            for cx, cy in cells:
                pos = np.zeros((6, 3))
                c_site = np.array([p[0] * lattice, p[1] * lattice, z_c])
                o_site = np.array([(q[0] + cx) * lattice,
                                   (q[1] + cy) * lattice, z_o])
                if state == "is":
                    pos[[0, 1, 2, 3]] = ch3 + c_site
                    pos[[4, 5]] = oh + o_site
                else:
                    pos[[0, 1, 2]] = ch2 + c_site
                    pos[[3, 4, 5]] = h2o + o_site
                out.append(pos)
    return np.asarray(out)


def _is_intact(pos, cell, bonds, z_max, heavy):
    """Structure filter: bonding pattern, intercalation and desorption."""
    if pos[:, 2].min() <= 0.0 or pos[:, 2].max() >= z_max:
        return False
    heavy = np.asarray(heavy)
    for light, host in bonds:
        d = np.sqrt(np.sum(_mic_delta(pos[heavy] - pos[light], cell) ** 2,
                           axis=1))
        if heavy[int(np.argmin(d))] != host:
            return False
    return True


def _distance_fingerprint(pos, cell):
    """Sorted adsorbate pair distances, a translation and rotation invariant."""
    ii, jj = np.triu_indices(pos.shape[0], k=1)
    d = _mic_delta(pos[ii] - pos[jj], cell)
    return np.sort(np.sqrt(np.sum(d * d, axis=1)))


def _dedupe(positions, energies, cell, tol_e=1e-3, tol_d=1e-2):
    """Drop structures whose energy and distance fingerprint repeat an earlier one."""
    keep, prints = [], []
    for i in range(positions.shape[0]):
        fp = _distance_fingerprint(positions[i], cell)
        if any(abs(energies[i] - energies[j]) < tol_e
               and float(np.max(np.abs(fp - q))) < tol_d
               for j, q in zip(keep, prints)):
            continue
        keep.append(i)
        prints.append(fp)
    return np.array(keep, dtype=int)


def _neb_optimize(band, _energy_forces, k_min, k_max, f_switch, f_conv,
                  steps_pre, steps_ci):
    """Non-climbing FIRE pass, then a climbing-image pass on a frozen top image.

    The spring constants are rebuilt at every force evaluation.  Holding them
    fixed over a block of evaluations makes the band force a discontinuous
    function of the images and puts a floor under the residual force that no
    step budget can get below; rebuilding them each time removes that floor.
    """
    state = {"climb": -1}

    def _band_force(pos):
        ener, raw = _energy_forces(pos)
        k = _oracle_variable_spring_constants(ener, k_min, k_max)
        return _oracle_neb_band_forces(pos, ener, raw, k, state["climb"])

    band = _fire_relax(band, _band_force, f_switch, steps_pre, batch=True)
    state["climb"] = int(np.argmax(np.asarray(_energy_forces(band)[0])[1:-1])) + 1
    return _fire_relax(band, _band_force, f_conv, steps_ci, batch=True)


def _oracle_run_nebscape(de: "np.ndarray", alpha: "np.ndarray",
                         r0: "np.ndarray", r_bond: "np.ndarray",
                         valence: "np.ndarray", k_coord: "np.ndarray",
                         lattice: float = 2.8, n_cell: int = 4,
                         n_layer: int = 3, z_layer: float = 1.4,
                         z_c: float = 2.05, z_o: float = 1.95,
                         tilt: float = 25.0,
                         z_max: float = 6.0, n_rep: int = 3,
                         n_coarse: int = 6, n_fine: int = 20, n_neb: int = 20,
                         n_pre: int = 10,
                         e_window: float = 1.2, d_window: float = 4.5,
                         de_window: float = 0.1, mic_alpha: float = 0.68,
                         k_min: float = 0.1, k_max: float = 4.0,
                         f_switch: float = 0.2, f_conv: float = 1e-4,
                         f_relax: float = 1e-5,
                         steps_relax: int = 10000, steps_pre: int = 30000,
                         steps_ci: int = 30000) -> float:
    de = np.asarray(de, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    r0 = np.asarray(r0, dtype=float)
    r_bond = np.asarray(r_bond, dtype=float)
    valence = np.asarray(valence, dtype=float)
    k_coord = np.asarray(k_coord, dtype=float)
    if not (lattice > 0.0 and z_layer > 0.0):
        raise ValueError("lattice and z_layer must be positive")
    if int(n_cell) < 2 or int(n_layer) < 1:
        raise ValueError("n_cell must be at least 2 and n_layer at least 1")
    if min(int(n_coarse), int(n_fine), int(n_neb)) < 3:
        raise ValueError("image counts must be at least 3")
    if min(int(n_rep), int(n_pre)) < 1:
        raise ValueError("n_rep and n_pre must be at least 1")

    lattice = float(lattice)
    n_cell = int(n_cell)
    cell = np.array([n_cell * lattice, n_cell * lattice])
    slab = _build_slab(lattice, n_cell, float(z_layer), int(n_layer))
    slab_species = np.zeros(slab.shape[0], dtype=int)
    ads_species = np.array([1, 3, 3, 3, 2, 3])
    masses = np.array([12.011, 1.008, 1.008, 1.008, 15.999, 1.008])
    moiety = np.array([0, 0, 0, 2, 1, 1])
    frag_is = np.array([0, 0, 0, 0, 1, 1])
    frag_fs = np.array([0, 0, 0, 1, 1, 1])
    heavy = np.array([0, 4])
    bonds_is = ((1, 0), (2, 0), (3, 0), (5, 4))
    bonds_fs = ((1, 0), (2, 0), (3, 4), (5, 4))
    donor, transfer, acceptor = 0, 3, 4
    sym_ops = _c4v_operations()

    def _energy_forces(p):
        return _oracle_surface_energy_forces(p, slab, ads_species, slab_species,
                                             cell, de, alpha, r0, r_bond,
                                             valence, k_coord)

    def _relax(p):
        return _fire_relax(p, lambda q: _energy_forces(q)[1], float(f_relax),
                           int(steps_relax), batch=True)

    raw = {state: _candidate_pool(state, lattice, float(z_c), float(z_o),
                                  ((1, 0),), float(tilt))
           for state in ("is", "fs")}
    split = raw["is"].shape[0]
    relaxed_pool = _relax(np.concatenate([raw["is"], raw["fs"]], axis=0))
    pools = {}
    for state, bonds, block in (("is", bonds_is, relaxed_pool[:split]),
                                ("fs", bonds_fs, relaxed_pool[split:])):
        ok = np.array([i for i in range(block.shape[0])
                       if _is_intact(block[i], cell, bonds, float(z_max),
                                     heavy)],
                      dtype=int)
        if ok.size == 0:
            raise ValueError("no candidate of the %s ensemble survived the "
                             "structure filter" % state)
        rel = block[ok]
        ener = np.asarray(_energy_forces(rel)[0])
        dist = np.sqrt(np.sum(_mic_delta(rel[:, donor, :] - rel[:, acceptor, :],
                                         cell) ** 2, axis=1))
        keep = _oracle_select_pareto_geometries(ener, dist, e_window, d_window)
        keep = keep[_dedupe(rel[keep], ener[keep], cell)]
        keep = keep[np.argsort(ener[keep], kind="stable")[:int(n_rep)]]
        keep = np.sort(keep)
        pools[state] = (rel[keep], ener[keep])

    is_pos = pools["is"][0]
    fs_pos, fs_ener = pools["fs"]

    cand, augmented = [], {}
    for a_i in range(is_pos.shape[0]):
        for b_i in range(fs_pos.shape[0]):
            aug = _oracle_align_final_state(slab, is_pos[a_i], fs_pos[b_i],
                                            masses, frag_is, frag_fs, cell,
                                            sym_ops)[2]
            augmented[(a_i, b_i)] = aug
            for s_i in range(aug.shape[0]):
                disp = _oracle_pbc_displacements(is_pos[a_i], aug[s_i], cell,
                                                 moiety, mic_alpha)[0]
                band, e_idpp = _oracle_build_idpp_band(
                    is_pos[a_i], disp, slab, cell, int(n_coarse))
                mu, tau = _oracle_reaction_distance_metrics(
                    band, e_idpp, donor, transfer, acceptor)
                cand.append((mu, tau, a_i, b_i, s_i))

    cand.sort(key=lambda c: (c[0], c[2], c[3], c[4]))
    pre = sorted(cand[:int(n_pre)], key=lambda c: (c[1], c[2], c[3], c[4]))
    relaxed = _relax(np.asarray([augmented[(c[2], c[3])][c[4]] for c in pre]))
    e_relaxed = np.asarray(_energy_forces(relaxed)[0])

    survivors = []
    for j, (_, _, a_i, b_i, _) in enumerate(pre):
        if abs(float(e_relaxed[j]) - float(fs_ener[b_i])) >= de_window:
            continue
        disp = _oracle_pbc_displacements(is_pos[a_i], relaxed[j], cell, moiety,
                                         mic_alpha)[0]
        band, e_idpp = _oracle_build_idpp_band(is_pos[a_i], disp, slab, cell,
                                               int(n_fine))
        mu2, tau2 = _oracle_reaction_distance_metrics(band, e_idpp, donor,
                                                      transfer, acceptor)
        survivors.append((mu2, tau2, j, a_i, relaxed[j]))

    if not survivors:
        raise ValueError("no interpolation survived the selection pipeline")
    survivors.sort(key=lambda c: (c[0], c[1], c[2]))

    a_i, fs_final = survivors[0][3], survivors[0][4]
    disp = _oracle_pbc_displacements(is_pos[a_i], fs_final, cell, moiety,
                                     mic_alpha)[0]
    band = _oracle_build_idpp_band(is_pos[a_i], disp, slab, cell, int(n_neb))[0]
    band = _neb_optimize(band, _energy_forces, k_min, k_max, f_switch, f_conv,
                         int(steps_pre), int(steps_ci))
    ener = np.asarray(_energy_forces(band)[0])
    return float(np.max(ener) - ener[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    shared = """import numpy as np

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
FAST = dict(n_rep=2, n_coarse=5, n_fine=9, n_neb=9, n_pre=4,
            steps_relax=400, steps_pre=120, steps_ci=120)
"""
    return [
        # normal: the benchmark parameters on a reduced, fast budget
        {"setup": shared + """
kw = dict(FAST)
""",
         "call": "run_nebscape(DE, AL, R0, RB, VAL, KC, **kw)",
         "gold_call": "_oracle_run_nebscape(DE, AL, R0, RB, VAL, KC, **kw)"},
        # normal: a stiffer, more strongly bound calculator on a smaller cell,
        # which changes both ensembles and the selected pathway
        {"setup": shared + """
kw = dict(FAST)
kw.update(lattice=3.0, n_cell=3, n_layer=2, z_layer=1.5)
de2 = DE * 1.25
al2 = AL * 1.15
""",
         "call": "run_nebscape(de2, al2, R0, RB, VAL, KC, **kw)",
         "gold_call": "_oracle_run_nebscape(de2, al2, R0, RB, VAL, KC, **kw)"},
        # boundary: a single representative per ensemble and a single candidate
        # carried into the transfer re-ranking, the narrowest cascade that runs
        {"setup": shared + """
kw = dict(FAST)
kw.update(n_rep=1, n_pre=1)
""",
         "call": "run_nebscape(DE, AL, R0, RB, VAL, KC, **kw)",
         "gold_call": "_oracle_run_nebscape(DE, AL, R0, RB, VAL, KC, **kw)"},
        # boundary: a wider energy window and a tighter separation cut-off, so
        # the two-objective selection keeps a different set of representatives
        {"setup": shared + """
kw = dict(FAST)
kw.update(e_window=2.5, d_window=3.2)
""",
         "call": "run_nebscape(DE, AL, R0, RB, VAL, KC, **kw)",
         "gold_call": "_oracle_run_nebscape(DE, AL, R0, RB, VAL, KC, **kw)"},
        # invalid: with the over-coordination penalty removed the calculator is
        # a bare pair potential, both ensembles collapse onto the same
        # structure, no candidate keeps the requested bonding pattern, and the
        # workflow reports an empty ensemble
        {"setup": shared + """
kw = dict(FAST)
zero = np.zeros(4)
def run_model():
    try:
        run_nebscape(DE, AL, R0, RB, VAL, zero, **kw)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_run_nebscape(DE, AL, R0, RB, VAL, zero, **kw)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
        # edge: uniform springs, which switch the variable-spring scheme off and
        # leave the band at even spacing
        {"setup": shared + """
kw = dict(FAST)
kw.update(k_min=0.999, k_max=1.0)
""",
         "call": "run_nebscape(DE, AL, R0, RB, VAL, KC, **kw)",
         "gold_call": "_oracle_run_nebscape(DE, AL, R0, RB, VAL, KC, **kw)"},
        # invalid: an image count below three
        {"setup": shared + """
kw = dict(FAST)
kw.update(n_neb=2)
def run_model():
    try:
        run_nebscape(DE, AL, R0, RB, VAL, KC, **kw)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_run_nebscape(DE, AL, R0, RB, VAL, KC, **kw)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
        # invalid: a non-positive lattice constant
        {"setup": shared + """
kw = dict(FAST)
kw.update(lattice=0.0)
def run_model():
    try:
        run_nebscape(DE, AL, R0, RB, VAL, KC, **kw)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_run_nebscape(DE, AL, R0, RB, VAL, KC, **kw)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
    ]
