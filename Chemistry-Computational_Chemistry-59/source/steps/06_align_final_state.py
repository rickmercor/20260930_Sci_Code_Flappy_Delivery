"""
Return the translation with which the workflow aligns a final state to an initial state on a periodic slab, the registration operation that must be composed with that translation to restore an equivalent adsorption site, and the set of symmetry-equivalent aligned final states, one for each supplied point-group operation.

Initial and final states that come from independent global searches are not posed for one another. Their adsorbates may sit on different, though symmetry-equivalent, sites of the same surface, and a band drawn between them then spends its energy on surface diffusion and molecular rotation rather than on the bond rearrangement that was asked about; the apparent barrier of such a path can exceed the real one by more than an electronvolt. The repair uses the symmetry of the clean slab, which is known before any adsorbate is placed, and applying the point-group operations of the surface generates all the orientations in which the product is equally stable, so the pairing stage can choose the one closest to the reactant instead of the one that global optimisation happened to return.



Conventions fixed by this task. Distances used to choose an anchoring slab atom are minimum-image in the two periodic directions and full three-dimensional, slab atoms whose distance lies within 1e-3 A of the smallest count as tied and the tie goes to the larger slab index, and fragment centres of mass are mass-weighted with the supplied masses. The tiled environment of an anchor is the set of minimum-image offsets, in the two periodic directions, of the slab atoms from that anchor whose length is at most the cut-off radius; the radius starts at $r_cut$ and grows in steps of 0.5 A until the two environments contain the same number of atoms; the source's additional graph-isomorphism condition on the two environments is not applied. In place of the source's continuous rotation and reflection fits, this task chooses the registration operation from the supplied point-group operations using the registration distance given in the formulas: the identity when that distance for the identity is below $r_tol$, and otherwise the supplied operation that minimises it. Every operation acts on the adsorbate coordinates measured from the initial-state anchor, whose position is put back afterwards; the slab is never moved, and the candidates follow the order of the supplied operations.

Writing $\\mathcal{E}_{s}$ and $\\mathcal{E}_{t}$ for the tiled environments of the final-state and initial-state anchors, the registration distance of a candidate operation $R$ is



$$D(R)=\\Big[\\sum_{\\mathbf{u}\\in\\mathcal{E}_s}\\min_{\\mathbf{w}\\in\\mathcal{E}_t}\\lVert R\\mathbf{u}-\\mathbf{w}\\rVert^{2}\\Big]^{1/2},$$



and $R_{\\mathrm{ref}}=I$ if $D(I)<r_{\\mathrm{tol}}$, otherwise the supplied operation minimising $D$. With $\\mathbf{r}_{\\mathrm{target}}$ the initial-state anchor and $\\mathbf{r}^{\\mathrm{FS}}+\\vec t$ the translated final state, the aligned final state and its symmetry family are



$$\\tilde{\\mathbf{r}}=R_{\\mathrm{ref}}\\big(\\mathbf{r}^{\\mathrm{FS}}+\\vec{t}-\\mathbf{r}_{\\mathrm{target}}\\big)+\\mathbf{r}_{\\mathrm{target}},\\qquad \\tilde{\\mathbf{r}}_{j}=R^{\\mathrm{sym}}_{j}\\big(\\tilde{\\mathbf{r}}-\\mathbf{r}_{\\mathrm{target}}\\big)+\\mathbf{r}_{\\mathrm{target}} .$$

Returns
-------
A tuple `(t, r_ref, candidates)`: `t` is the translation vector of shape $(3,)$ in angstrom, `r_ref` the $(3,3)$ registration matrix, and `candidates` the $(S,N,3)$ stack of symmetry-equivalent aligned final states.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def align_final_state(slab: "np.ndarray", ads_is: "np.ndarray",
                      ads_fs: "np.ndarray", masses: "np.ndarray",
                      frag_is: "np.ndarray", frag_fs: "np.ndarray",
                      cell: "np.ndarray", sym_ops: "np.ndarray",
                      r_cut: float = 4.0, r_tol: float = 1.0) -> tuple:
    """Anchor-based alignment of a final state to an initial state on a slab.

    Parameters
    ----------
    slab : numpy.ndarray
        Frozen slab positions in angstrom, shape (F, 3) with F >= 1.
    ads_is, ads_fs : numpy.ndarray
        Adsorbate positions of the initial and final states in angstrom, both
        of shape (N, 3) with N >= 1.
    masses : numpy.ndarray
        Positive mass of each adsorbate atom, shape (N,).
    frag_is, frag_fs : numpy.ndarray
        Integer fragment label of each adsorbate atom in the initial and in
        the final state, shape (N,); each must define exactly two fragments.
    cell : numpy.ndarray
        The two positive in-plane periods (Lx, Ly) in angstrom.
    sym_ops : numpy.ndarray
        Point-group operations of the clean slab, shape (S, 3, 3) with S >= 1.
    r_cut : float
        Positive initial radius in angstrom of the tiled anchor environment,
        grown in steps of 0.5 A until the two environments have equal size.
    r_tol : float
        Positive registration distance in angstrom below which the two
        environments count as already aligned.

    Returns
    -------
    result : tuple
        (t, r_ref, candidates). t has shape (3,), r_ref shape (3, 3) and
        candidates shape (S, N, 3), one aligned final state per supplied
        operation, in the order the operations were given.

    Raises
    ------
    ValueError
        If slab, ads_is, ads_fs, masses, frag_is, frag_fs, cell or sym_ops has
        the wrong shape, if a mass is not positive, if either fragment label
        array does not define exactly two fragments, if r_cut or r_tol is not
        positive, or if no cut-off radius makes the two environments equal in
        size.
    """
    return t, r_ref, candidates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _anchor_index(slab, ads, masses, fragments, cell):
    """Slab atom closest to the midpoint of the two fragment centres of mass."""
    coms = []
    for lab in np.unique(fragments):
        sel = fragments == lab
        w = masses[sel]
        coms.append(np.sum(ads[sel] * w[:, None], axis=0) / w.sum())
    mid = np.mean(np.asarray(coms), axis=0)
    d = np.sqrt(np.sum(_mic_delta(slab - mid[None, :], cell) ** 2, axis=1))
    tied = np.flatnonzero(d <= d.min() + 1e-3)
    return int(tied.max())


def _oracle_align_final_state(slab: "np.ndarray", ads_is: "np.ndarray",
                              ads_fs: "np.ndarray", masses: "np.ndarray",
                              frag_is: "np.ndarray", frag_fs: "np.ndarray",
                              cell: "np.ndarray", sym_ops: "np.ndarray",
                              r_cut: float = 4.0, r_tol: float = 1.0) -> tuple:
    slab = np.asarray(slab, dtype=float)
    ads_is = np.asarray(ads_is, dtype=float)
    ads_fs = np.asarray(ads_fs, dtype=float)
    masses = np.asarray(masses, dtype=float)
    frag_is = np.asarray(frag_is, dtype=int)
    frag_fs = np.asarray(frag_fs, dtype=int)
    cell = np.asarray(cell, dtype=float).reshape(-1)
    sym_ops = np.asarray(sym_ops, dtype=float)

    if slab.ndim != 2 or slab.shape[1] != 3 or slab.shape[0] < 1:
        raise ValueError("slab must have shape (F, 3) with F >= 1")
    if ads_is.ndim != 2 or ads_is.shape[1] != 3 or ads_is.shape[0] < 1:
        raise ValueError("ads_is must have shape (N, 3) with N >= 1")
    if ads_fs.shape != ads_is.shape:
        raise ValueError("ads_fs must have the same shape as ads_is")
    if masses.shape != (ads_is.shape[0],) or np.any(masses <= 0.0):
        raise ValueError("masses must be one positive value per adsorbate atom")
    if frag_is.shape != (ads_is.shape[0],) or frag_fs.shape != (ads_is.shape[0],):
        raise ValueError("frag_is and frag_fs must have one label per atom")
    if np.unique(frag_is).size != 2 or np.unique(frag_fs).size != 2:
        raise ValueError("frag_is and frag_fs must each define two fragments")
    if cell.shape != (2,) or not np.all(cell > 0.0):
        raise ValueError("cell must be two positive in-plane lengths")
    if sym_ops.ndim != 3 or sym_ops.shape[1:] != (3, 3) or sym_ops.shape[0] < 1:
        raise ValueError("sym_ops must have shape (S, 3, 3) with S >= 1")
    if not (r_cut > 0.0 and r_tol > 0.0):
        raise ValueError("r_cut and r_tol must be positive")

    r_target = slab[_anchor_index(slab, ads_is, masses, frag_is, cell)]
    r_source = slab[_anchor_index(slab, ads_fs, masses, frag_fs, cell)]
    t = r_target - r_source

    def _tiled_environment(anchor, radius):
        # every in-plane periodic copy of the slab within radius of the anchor
        reach = [int(np.ceil(radius / cell[k])) + 1 for k in range(2)]
        shifts = np.array([[i * cell[0], j * cell[1], 0.0]
                           for i in range(-reach[0], reach[0] + 1)
                           for j in range(-reach[1], reach[1] + 1)])
        rel = (slab[None, :, :] + shifts[:, None, :]
               - anchor[None, None, :]).reshape(-1, 3)
        return rel[np.sqrt(np.sum(rel ** 2, axis=1)) <= radius]

    rc = float(r_cut)
    env_s = env_t = np.zeros((0, 3))
    for _ in range(64):
        env_s = _tiled_environment(r_source, rc)
        env_t = _tiled_environment(r_target, rc)
        if env_s.shape[0] == env_t.shape[0]:
            break
        rc += 0.5
    if env_s.shape[0] != env_t.shape[0]:
        raise ValueError("could not match the two anchoring environments")

    def _registration_distance(rot):
        turned = env_s @ np.asarray(rot, dtype=float).T
        gap = np.sqrt(np.sum((turned[:, None, :] - env_t[None, :, :]) ** 2,
                             axis=2))
        return float(np.sqrt(np.sum(np.min(gap, axis=1) ** 2)))

    if _registration_distance(np.eye(3)) < r_tol:
        r_ref = np.eye(3)
    else:
        scores = [_registration_distance(op) for op in sym_ops]
        r_ref = np.asarray(sym_ops[int(np.argmin(scores))], dtype=float)

    aligned = ((ads_fs + t[None, :] - r_target[None, :]) @ r_ref.T
               + r_target[None, :])
    candidates = np.einsum("sij,nj->sni", sym_ops,
                           aligned - r_target[None, :]) + r_target[None, None, :]
    return t, r_ref, candidates

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    shared = """import numpy as np

def pack(result):
    # flatten a tuple of arrays of different shapes into one 1-D array
    return np.concatenate([np.asarray(p, dtype=float).ravel() for p in result])
""" + """
def slab_of(lattice, n_cell, spacing, n_layer):
    idx = np.arange(n_cell, dtype=float)
    gx, gy = np.meshgrid(idx, idx, indexing="ij")
    out = []
    for lay in range(n_layer):
        off = 0.5 * (lay % 2)
        out.append(np.stack([(gx.ravel() + off) * lattice,
                             (gy.ravel() + off) * lattice,
                             np.full(gx.size, -lay * spacing)], axis=1))
    return np.concatenate(out, axis=0)

def c4v():
    ops = []
    for r in range(4):
        ang = 0.5 * np.pi * r
        c, s = float(round(np.cos(ang))), float(round(np.sin(ang)))
        ops.append(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))
    mirror = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    for r in range(4):
        ops.append(ops[r] @ mirror)
    return np.asarray(ops)

MASS = np.array([12.011, 1.008, 1.008, 1.008, 15.999, 1.008])
FIS = np.array([0, 0, 0, 0, 1, 1])
FFS = np.array([0, 0, 0, 1, 1, 1])
AIS = np.array([[1.40, 1.40, 1.90], [2.43, 1.40, 2.26],
                [0.88, 2.29, 2.26], [0.88, 0.51, 2.26],
                [4.20, 2.80, 1.80], [4.20, 2.80, 2.77]])
"""
    return [
        # normal: a final state a full lattice vector away from the initial one
        {"setup": shared + """
slab = slab_of(2.8, 4, 1.4, 3)
cell = np.array([11.2, 11.2])
afs = AIS + np.array([[2.8, 2.8, 0.0]] * 6)
afs[3] = afs[4] + np.array([0.0, 0.0, 1.05])
""",
         "call": "pack(align_final_state(slab, AIS, afs, MASS, FIS, FFS, cell, c4v()))",
         "gold_call":
             "pack(_oracle_align_final_state(slab, AIS, afs, MASS, FIS, FFS, cell, c4v()))"},
        # normal: a final state rotated as well as displaced, and a slab with a
        # different lattice constant and only two layers
        {"setup": shared + """
slab = slab_of(3.1, 4, 1.6, 2)
cell = np.array([12.4, 12.4])
rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
afs = (AIS - AIS.mean(axis=0)) @ rot.T + AIS.mean(axis=0) + np.array([3.1, 0.0, 0.0])
""",
         "call": "pack(align_final_state(slab, AIS, afs, MASS, FIS, FFS, cell, c4v()))",
         "gold_call":
             "pack(_oracle_align_final_state(slab, AIS, afs, MASS, FIS, FFS, cell, c4v()))"},
        # boundary: a single supplied operation, so exactly one candidate is
        # emitted and it is the aligned state itself
        {"setup": shared + """
slab = slab_of(2.8, 4, 1.4, 3)
cell = np.array([11.2, 11.2])
afs = AIS + np.array([[5.6, 0.0, 0.0]] * 6)
one = np.eye(3)[None, :, :]
""",
         "call": "pack(align_final_state(slab, AIS, afs, MASS, FIS, FFS, cell, one))",
         "gold_call":
             "pack(_oracle_align_final_state(slab, AIS, afs, MASS, FIS, FFS, cell, one))"},
        # boundary: a registration tolerance so small that the identity is
        # rejected and an operation has to be searched for
        {"setup": shared + """
slab = slab_of(2.8, 4, 1.4, 3)
cell = np.array([11.2, 11.2])
afs = AIS + np.array([[2.8, 0.0, 0.0]] * 6)
""",
         "call": ("pack(align_final_state(slab, AIS, afs, MASS, FIS, FFS, cell, "
                  "c4v(), 4.0, 1e-9))"),
         "gold_call": ("pack(_oracle_align_final_state(slab, AIS, afs, MASS, FIS, "
                       "FFS, cell, c4v(), 4.0, 1e-9))")},
        # boundary: a larger starting environment radius, which changes which
        # slab atoms enter the registration comparison
        {"setup": shared + """
slab = slab_of(2.8, 4, 1.4, 3)
cell = np.array([11.2, 11.2])
afs = AIS + np.array([[2.8, 2.8, 0.0]] * 6)
""",
         "call": ("pack(align_final_state(slab, AIS, afs, MASS, FIS, FFS, cell, "
                  "c4v(), 6.5, 1.0))"),
         "gold_call": ("pack(_oracle_align_final_state(slab, AIS, afs, MASS, FIS, "
                       "FFS, cell, c4v(), 6.5, 1.0))")},
        # edge: a final state identical to the initial one, so the translation
        # is zero and the identity candidate reproduces the input
        {"setup": shared + """
slab = slab_of(2.8, 4, 1.4, 3)
cell = np.array([11.2, 11.2])
""",
         "call": "pack(align_final_state(slab, AIS, AIS, MASS, FIS, FFS, cell, c4v()))",
         "gold_call":
             "pack(_oracle_align_final_state(slab, AIS, AIS, MASS, FIS, FFS, cell, c4v()))"},
        # normal: two anchoring atoms that are equivalent but whose
        # surroundings are turned by a quarter turn with respect to one
        # another, so the identity fails the registration test and the single
        # point-group operation that maps one environment onto the other has
        # to be composed with the translation
        {"setup": shared + """
cell = np.array([9.0, 9.0])
slab = np.array([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 3.0, 0.0],
                 [5.0, 5.0, 0.0], [5.0, 6.0, 0.0], [3.0, 5.0, 0.0]])
def placed(frag, x, y):
    coms = [np.sum(AIS[frag == k] * MASS[frag == k][:, None], axis=0)
            / MASS[frag == k].sum() for k in (0, 1)]
    return AIS - 0.5 * (coms[0] + coms[1]) + np.array([x, y, 2.0])
ais = placed(FIS, 1.0, 1.0)
afs = placed(FFS, 5.0, 5.0)
""",
         "call": "pack(align_final_state(slab, ais, afs, MASS, FIS, FFS, cell, c4v()))",
         "gold_call":
             "pack(_oracle_align_final_state(slab, ais, afs, MASS, FIS, FFS, cell, c4v()))"},
        # invalid: a fragment labelling with only one fragment
        {"setup": shared + """
slab = slab_of(2.8, 4, 1.4, 3)
cell = np.array([11.2, 11.2])
bad = np.zeros(6, dtype=int)
def run_model():
    try:
        align_final_state(slab, AIS, AIS, MASS, bad, FFS, cell, c4v())
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_align_final_state(slab, AIS, AIS, MASS, bad, FFS, cell, c4v())
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
        # invalid: a non-positive mass
        {"setup": shared + """
slab = slab_of(2.8, 4, 1.4, 3)
cell = np.array([11.2, 11.2])
bad = MASS.copy()
bad[2] = 0.0
def run_model():
    try:
        align_final_state(slab, AIS, AIS, bad, FIS, FFS, cell, c4v())
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_align_final_state(slab, AIS, AIS, bad, FIS, FFS, cell, c4v())
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
        # boundary: four layer-0 slab atoms fall inside the 1e-3 A tie window
        # around the anchoring distance, and the strict minimum is atom 0 while
        # the documented rule takes the largest tied index, atom 5. The case
        # grades the tie-break itself: the two rules select different anchors
        # here, so a nearest-atom implementation fails it.
        {"setup": shared + """
slab = slab_of(2.8, 4, 1.4, 3)
cell = np.array([11.2, 11.2])

def com_mid(ads, frag):
    coms = []
    for lab in np.unique(frag):
        sel = frag == lab
        w = MASS[sel]
        coms.append(np.sum(ads[sel] * w[:, None], axis=0) / w.sum())
    return np.mean(np.asarray(coms), axis=0)

delta = 2.0e-4
ais_t = AIS.copy()
ais_t[:, :2] += np.array([1.4 - delta, 1.4 - delta]) - com_mid(AIS, FIS)[:2]
afs_t = ais_t + np.array([[2.8, 2.8, 0.0]] * 6)
afs_t[3] = afs_t[4] + np.array([0.0, 0.0, 1.05])
""",
         "call": "pack(align_final_state(slab, ais_t, afs_t, MASS, FIS, FFS, cell, c4v()))",
         "gold_call":
             "pack(_oracle_align_final_state(slab, ais_t, afs_t, MASS, FIS, FFS, cell, c4v()))"},
    ]
