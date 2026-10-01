"""
Return, for every adsorbate atom, the displacement vector that carries it from the initial state to the final state through the periodic image that the workflow's periodic-boundary-aware interpolation assigns to it, together with the index of that image. Atoms that the reaction leaves bonded together are grouped into moieties, and the assignment must keep each moiety intact across the periodic boundary.

Interpolating between two periodic configurations is not the same problem as interpolating between two isolated ones. Each atom of the final state exists in every periodic image, and the naive choice, the minimum image convention, treats every atom independently. For a molecule sitting near a cell boundary this is a trap: two atoms of the same rigid group can pick images that differ by a lattice vector, so the straight line between initial and final positions pulls them apart by most of a cell length and lets them re-associate only at the far end. The energy profile of such a path shows a spurious dissociation that has nothing to do with the chemistry being studied. The information needed to avoid it is the atom mapping: it says which bonds survive the reaction, and therefore which atoms must stay together, and which atoms are the active ones whose bonds break and which are consequently free to follow their own shortest route.



Conventions fixed by this task. The nine in-plane images of the final state are indexed $i=3(n_a+1)+(n_b+1)$ for offsets $n_a,n_b\\in\\{-1,0,1\\}$, so $i=4$ is the home cell, and the minimum image of an atom is the image of smallest displacement length, the smaller index winning a tie. Within one moiety, the subgroup that keeps the minimum image is the set of atoms sharing the image index held by the largest number of its atoms, ties going to the smaller index; a moiety of one atom always keeps the minimum image. The parameter `alpha` is the weight the source's image-ranking score gives to its length term, and the angular term of that score is taken as zero whenever either vector in it has zero length. Coordinates are never wrapped back into the cell.

With $\\mathbf{r}^{\\mathrm{IS}}_k$, $\\mathbf{r}^{\\mathrm{FS}}_k$ the two states and $\\mathbf{L}_i=(n_aL_x,\\,n_bL_y,\\,0)$ the nine in-plane offsets, the candidate displacements and the minimum image are



$$\\mathbf{v}_{ki}=\\mathbf{r}^{\\mathrm{FS}}_{k}+\\mathbf{L}_i-\\mathbf{r}^{\\mathrm{IS}}_{k},\\qquad i^{\\mathrm{MIC}}_k=\\arg\\min_i\\lVert\\mathbf{v}_{ki}\\rVert .$$



The returned displacement of atom $k$ is $\\mathbf{v}_{k\\,i_k}$ for the image $i_k$ finally assigned to it.

Returns
-------
A tuple `(disp, index)`: `disp` has shape $(N,3)$ and holds the chosen displacement of each atom in angstrom, `index` has shape $(N,)$ and holds the chosen periodic image index of each atom as a float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pbc_displacements(pos_is: "np.ndarray", pos_fs: "np.ndarray",
                      cell: "np.ndarray", moiety: "np.ndarray",
                      alpha: float = 0.68) -> tuple:
    """Periodic-image-aware displacements from an initial to a final state.

    Parameters
    ----------
    pos_is : numpy.ndarray
        Initial-state positions in angstrom, shape (N, 3) with N >= 1.
    pos_fs : numpy.ndarray
        Final-state positions in angstrom, same shape as pos_is.
    cell : numpy.ndarray
        The two positive in-plane periods (Lx, Ly) in angstrom.
    moiety : numpy.ndarray
        Integer label per atom, shape (N,), grouping the atoms that the
        reaction leaves connected. Atoms whose bonds break carry a label of
        their own.
    alpha : float
        Non-negative weight of the length term relative to the angle term in
        the ranking of candidate periodic images.

    Returns
    -------
    result : tuple
        (disp, index). disp has shape (N, 3) and holds the displacement of
        each atom through its chosen image; index has shape (N,) and holds
        that image index as a float in the range 0 to 8.

    Raises
    ------
    ValueError
        If pos_is is not an (N, 3) array with N >= 1, if pos_fs has a
        different shape, if moiety does not carry one label per atom, if cell
        is not two positive lengths, or if alpha is negative.
    """
    return disp, index

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _mic_delta(vec, cell):
    """Minimum-image displacement in the two periodic in-plane directions."""
    out = np.array(vec, dtype=float, copy=True)
    out[..., 0] -= np.round(out[..., 0] / cell[0]) * cell[0]
    out[..., 1] -= np.round(out[..., 1] / cell[1]) * cell[1]
    return out


def _image_offsets(cell):
    """The nine in-plane periodic offsets, ordered i = 3 * (na + 1) + (nb + 1)."""
    na = np.repeat(np.arange(-1, 2), 3)
    nb = np.tile(np.arange(-1, 2), 3)
    off = np.zeros((9, 3))
    off[:, 0] = na * cell[0]
    off[:, 1] = nb * cell[1]
    return off


def _oracle_pbc_displacements(pos_is: "np.ndarray", pos_fs: "np.ndarray",
                              cell: "np.ndarray", moiety: "np.ndarray",
                              alpha: float = 0.68) -> tuple:
    pos_is = np.asarray(pos_is, dtype=float)
    pos_fs = np.asarray(pos_fs, dtype=float)
    moiety = np.asarray(moiety, dtype=int)
    cell = np.asarray(cell, dtype=float).reshape(-1)

    if pos_is.ndim != 2 or pos_is.shape[1] != 3 or pos_is.shape[0] < 1:
        raise ValueError("pos_is must have shape (N, 3) with N >= 1")
    if pos_fs.shape != pos_is.shape:
        raise ValueError("pos_fs must have the same shape as pos_is")
    if moiety.shape != (pos_is.shape[0],):
        raise ValueError("moiety must have one label per atom")
    if cell.shape != (2,) or not np.all(cell > 0.0):
        raise ValueError("cell must be two positive in-plane lengths")
    if not (alpha >= 0.0):
        raise ValueError("alpha must be non-negative")

    n = pos_is.shape[0]
    disp = (pos_fs[:, None, :] + _image_offsets(cell)[None, :, :]
            - pos_is[:, None, :])
    norm = np.sqrt(np.sum(disp * disp, axis=-1))
    mic = np.argmin(norm, axis=1)
    chosen = mic.copy()

    for label in np.unique(moiety):
        grp = np.flatnonzero(moiety == label)
        if grp.size < 2:
            continue
        counts = np.bincount(mic[grp], minlength=9)
        majority = int(np.argmax(counts))
        inside = grp[mic[grp] == majority]
        outside = grp[mic[grp] != majority]
        if outside.size == 0:
            continue
        v_avg = disp[inside, mic[inside], :].mean(axis=0)
        n_avg = float(np.sqrt(np.dot(v_avg, v_avg)))
        for k in outside:
            nv = norm[k]
            if n_avg <= 1e-12:
                theta = np.zeros(9)
            else:
                safe = np.where(nv < 1e-12, np.inf, nv)
                cosang = (disp[k] @ v_avg) / safe / n_avg
                theta = np.where(nv < 1e-12, 0.0,
                                 np.arccos(np.clip(cosang, -1.0, 1.0)))
            score = theta + alpha * nv / max(float(nv.min()), 1e-12)
            chosen[k] = int(np.argmin(score))

    return disp[np.arange(n), chosen, :], chosen.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    shared = """import numpy as np

def pack(result):
    # flatten a tuple of arrays of different shapes into one 1-D array
    return np.concatenate([np.asarray(p, dtype=float).ravel() for p in result])
"""
    return [
        # normal: a rigid group displaced well inside the cell, so every atom
        # already agrees on the home image
        {"setup": shared + """
cell = np.array([11.2, 11.2])
pos_is = np.array([[1.4, 1.4, 1.9], [2.4, 1.4, 2.3], [0.9, 2.3, 2.3],
                   [0.9, 0.5, 2.3], [4.2, 2.8, 1.8], [4.2, 2.8, 2.8]])
pos_fs = pos_is + np.array([[0.6, 0.3, 0.0]] * 6)
pos_fs[3] = [3.6, 2.4, 2.1]
moiety = np.array([0, 0, 0, 2, 1, 1])
""",
         "call": "pack(pbc_displacements(pos_is, pos_fs, cell, moiety))",
         "gold_call":
             "pack(_oracle_pbc_displacements(pos_is, pos_fs, cell, moiety))"},
        # normal: the whole molecule moved by one lattice vector, so every atom
        # agrees on a non-home image and no repair is needed
        {"setup": shared + """
cell = np.array([11.2, 11.2])
pos_is = np.array([[1.4, 1.4, 1.9], [2.4, 1.4, 2.3], [0.9, 2.3, 2.3],
                   [0.9, 0.5, 2.3], [4.2, 2.8, 1.8], [4.2, 2.8, 2.8]])
pos_fs = pos_is + np.array([[11.2, 0.0, 0.0]] * 6)
moiety = np.array([0, 0, 0, 2, 1, 1])
""",
         "call": "pack(pbc_displacements(pos_is, pos_fs, cell, moiety))",
         "gold_call":
             "pack(_oracle_pbc_displacements(pos_is, pos_fs, cell, moiety))"},
        # boundary: a three-atom moiety straddling the half-cell distance, so
        # two of its atoms take the home image, one takes a neighbouring image
        # under the minimum image convention, and the repair has to fire
        {"setup": shared + """
cell = np.array([6.0, 6.0])
pos_is = np.array([[0.2, 0.0, 2.0], [1.3, 0.0, 2.0], [-0.9, 0.0, 2.0],
                   [3.0, 0.0, 2.0]])
pos_fs = np.array([[3.3, 0.0, 2.0], [4.4, 0.0, 2.0], [2.2, 0.0, 2.0],
                   [0.4, 0.0, 2.0]])
moiety = np.array([0, 0, 0, 1])
""",
         "call": "pack(pbc_displacements(pos_is, pos_fs, cell, moiety))",
         "gold_call":
             "pack(_oracle_pbc_displacements(pos_is, pos_fs, cell, moiety))"},
        # boundary: the same straddling group with the length term switched
        # off, so the angle alone decides the reassigned images
        {"setup": shared + """
cell = np.array([6.0, 6.0])
pos_is = np.array([[0.2, 0.0, 2.0], [1.3, 0.0, 2.0], [-0.9, 0.0, 2.0],
                   [3.0, 0.0, 2.0]])
pos_fs = np.array([[3.3, 0.0, 2.0], [4.4, 0.0, 2.0], [2.2, 0.0, 2.0],
                   [0.4, 0.0, 2.0]])
moiety = np.array([0, 0, 0, 1])
""",
         "call": "pack(pbc_displacements(pos_is, pos_fs, cell, moiety, 0.0))",
         "gold_call":
             "pack(_oracle_pbc_displacements(pos_is, pos_fs, cell, moiety, 0.0))"},
        # edge: a state that does not move at all, so every displacement is the
        # zero vector and the average of the kept group has no direction
        {"setup": shared + """
cell = np.array([5.0, 7.0])
pos_is = np.array([[1.0, 2.0, 3.0], [1.5, 2.0, 3.0], [4.9, 2.0, 3.0]])
pos_fs = pos_is.copy()
moiety = np.array([0, 0, 0])
""",
         "call": "pack(pbc_displacements(pos_is, pos_fs, cell, moiety))",
         "gold_call":
             "pack(_oracle_pbc_displacements(pos_is, pos_fs, cell, moiety))"},
        # edge: a single atom, which is its own moiety and always keeps the
        # minimum image
        {"setup": shared + """
cell = np.array([4.0, 4.0])
pos_is = np.array([[0.3, 0.3, 1.0]])
pos_fs = np.array([[3.8, 3.7, 1.0]])
moiety = np.array([7])
""",
         "call": "pack(pbc_displacements(pos_is, pos_fs, cell, moiety))",
         "gold_call":
             "pack(_oracle_pbc_displacements(pos_is, pos_fs, cell, moiety))"},
        # invalid: a moiety label array of the wrong length
        {"setup": shared + """
cell = np.array([4.0, 4.0])
pos_is = np.zeros((3, 3))
pos_fs = np.ones((3, 3))
bad = np.array([0, 0])
def run_model():
    try:
        pbc_displacements(pos_is, pos_fs, cell, bad)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_pbc_displacements(pos_is, pos_fs, cell, bad)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
        # invalid: a negative length weight
        {"setup": shared + """
cell = np.array([4.0, 4.0])
pos_is = np.zeros((3, 3))
pos_fs = np.ones((3, 3))
moiety = np.array([0, 0, 0])
def run_model():
    try:
        pbc_displacements(pos_is, pos_fs, cell, moiety, -1.0)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_pbc_displacements(pos_is, pos_fs, cell, moiety, -1.0)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
    ]
