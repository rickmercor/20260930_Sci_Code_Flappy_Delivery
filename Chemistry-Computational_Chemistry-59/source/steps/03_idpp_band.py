"""
Return a band of images between an initial state and the final state reached by the supplied displacements, refined on the image-dependent pair potential, together with the value of that potential at every image.

A straight line in Cartesian coordinates is a poor first guess for a reaction path: interpolating positions independently drives atoms through one another, and the resulting band can start thousands of electronvolts above the reactants, which no band optimiser recovers from cheaply. Interpolation schemes that work in the space of interatomic distances instead produce images that are physically sensible arrangements even before any energy has been evaluated, and whose profile is already a usable estimate of where the barrier lies. Including the frozen substrate atoms among the pairs is what stops such a refinement from pushing an adsorbate into the surface.



Conventions fixed by this task. The band starts as the straight line in the supplied displacements, image $m$ of $M$ sitting at $\\lambda_m=m/(M-1)$, so the periodic images already chosen for the atoms are honoured and no atom is re-wrapped. The pair set is every pair of moving atoms, each pair counted once, together with every moving-frozen pair. Distances among the moving atoms are taken from the coordinates as they are, while moving-frozen distances, at the endpoints and in the images, use the minimum image convention in the two periodic directions, because the frozen anchors are periodic and do not move. Each interior image is refined independently by exactly $n_iter$ steepest-descent steps on the image-dependent pair potential, the gradient being the exact gradient of that potential with respect to the moving coordinates; the endpoints are held fixed. The reported value of each interior image is the potential after refinement, and the two endpoints are reported as zero.

With $\\mathbf{v}$ the supplied displacements and $S_m$ the image-dependent pair potential of image $m$,



$$\\mathbf{R}_m^{(0)}=\\mathbf{r}^{\\mathrm{IS}}+\\lambda_m\\,\\mathbf{v},\\qquad \\Delta\\mathbf{R}_j=-\\eta\\,\\partial S_m/\\partial\\mathbf{R}_j,\\qquad \\lVert\\Delta\\mathbf{R}_j\\rVert\\le \\Delta_{\\max},$$



where $\\eta$ is `step`, $\\Delta_{\\max}$ is $max_step$, and the displacement of any atom exceeding the cap in one step is rescaled to the cap.

Returns
-------
A tuple `(band, e_idpp)`: `band` has shape $(M,N,3)$ and holds the refined images in angstrom, `e_idpp` has shape $(M,)$ and holds the image-dependent pair potential of each image, with zeros at the two endpoints.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_idpp_band(pos_is: "np.ndarray", disp: "np.ndarray",
                    frozen: "np.ndarray", cell: "np.ndarray",
                    n_images: int, n_iter: int = 50, step: float = 0.01,
                    max_step: float = 0.05) -> tuple:
    """Linear band in the supplied displacements, refined on the IDPP surface.

    Parameters
    ----------
    pos_is : numpy.ndarray
        Initial-state positions of the moving atoms in angstrom, shape (N, 3)
        with N >= 2.
    disp : numpy.ndarray
        Displacement of each moving atom to the final state, same shape as
        pos_is.
    frozen : numpy.ndarray
        Positions of the frozen anchor atoms in angstrom, shape (F, 3); an
        empty array leaves the objective to the moving pairs alone.
    cell : numpy.ndarray
        The two positive in-plane periods (Lx, Ly) in angstrom, used for the
        moving-to-frozen distances only.
    n_images : int
        Number of images including both endpoints, at least 2.
    n_iter : int
        Non-negative number of steepest-descent steps applied to each
        interior image.
    step : float
        Positive step length of the steepest descent in A per unit gradient.
    max_step : float
        Positive cap in A on the displacement of any one atom in one step.

    Returns
    -------
    result : tuple
        (band, e_idpp). band has shape (n_images, N, 3); e_idpp has shape
        (n_images,) and is zero at both endpoints.

    Raises
    ------
    ValueError
        If pos_is is not an (N, 3) array with N >= 2, if disp has a different
        shape, if cell is not two positive lengths, if n_images is below 2,
        if n_iter is negative, or if step or max_step is not positive.
    """
    return band, e_idpp

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _idpp_value_gradient(pos, frozen, d_mov, d_fix, cell):
    """Image-dependent pair potential of a stack of images and its gradient."""
    n = pos.shape[-2]
    eye = np.eye(n, dtype=bool)
    dvec = pos[..., :, None, :] - pos[..., None, :, :]
    d = np.where(eye, 1.0, np.sqrt(np.sum(dvec * dvec, axis=-1)))
    u = d_mov - d
    value = 0.5 * np.sum(np.where(eye, 0.0, u * u / d ** 4), axis=(-2, -1))
    coeff = np.where(eye, 0.0, -2.0 * u / d ** 4 - 4.0 * u * u / d ** 5)
    grad = np.sum((coeff / d)[..., None] * dvec, axis=-2)
    if frozen.shape[0] > 0:
        fvec = _mic_delta(pos[..., :, None, :] - frozen[None, None, :, :], cell)
        fd = np.maximum(np.sqrt(np.sum(fvec * fvec, axis=-1)), 1e-12)
        fu = d_fix - fd
        value = value + np.sum(fu * fu / fd ** 4, axis=(-2, -1))
        fcoeff = -2.0 * fu / fd ** 4 - 4.0 * fu * fu / fd ** 5
        grad = grad + np.sum((fcoeff / fd)[..., None] * fvec, axis=-2)
    return value, grad


def _oracle_build_idpp_band(pos_is: "np.ndarray", disp: "np.ndarray",
                            frozen: "np.ndarray", cell: "np.ndarray",
                            n_images: int, n_iter: int = 50,
                            step: float = 0.01,
                            max_step: float = 0.05) -> tuple:
    pos_is = np.asarray(pos_is, dtype=float)
    disp = np.asarray(disp, dtype=float)
    frozen = np.asarray(frozen, dtype=float).reshape(-1, 3)
    cell = np.asarray(cell, dtype=float).reshape(-1)
    if pos_is.ndim != 2 or pos_is.shape[1] != 3 or pos_is.shape[0] < 2:
        raise ValueError("pos_is must have shape (N, 3) with N >= 2")
    if disp.shape != pos_is.shape:
        raise ValueError("disp must have the same shape as pos_is")
    if cell.shape != (2,) or not np.all(cell > 0.0):
        raise ValueError("cell must be two positive in-plane lengths")
    if int(n_images) < 2:
        raise ValueError("n_images must be at least 2")
    if int(n_iter) < 0:
        raise ValueError("n_iter must be non-negative")
    if not (step > 0.0 and max_step > 0.0):
        raise ValueError("step and max_step must be positive")

    m = int(n_images)
    lam = np.linspace(0.0, 1.0, m)
    band = pos_is[None, :, :] + lam[:, None, None] * disp[None, :, :]
    e_idpp = np.zeros(m)
    if m == 2:
        return band, e_idpp

    pos_fs = pos_is + disp

    def _moving_distances(p):
        dv = p[:, None, :] - p[None, :, :]
        return np.sqrt(np.sum(dv * dv, axis=-1))

    def _fixed_distances(p):
        dv = _mic_delta(p[:, None, :] - frozen[None, :, :], cell)
        return np.maximum(np.sqrt(np.sum(dv * dv, axis=-1)), 1e-12)

    w = lam[1:-1][:, None, None]
    d_mov = ((1.0 - w) * _moving_distances(pos_is)[None, :, :]
             + w * _moving_distances(pos_fs)[None, :, :])
    if frozen.shape[0] > 0:
        d_fix = ((1.0 - w) * _fixed_distances(pos_is)[None, :, :]
                 + w * _fixed_distances(pos_fs)[None, :, :])
    else:
        d_fix = np.zeros((m - 2, pos_is.shape[0], 0))

    pos = band[1:-1].copy()
    for _ in range(int(n_iter)):
        dr = -float(step) * _idpp_value_gradient(pos, frozen, d_mov, d_fix,
                                                 cell)[1]
        nrm = np.sqrt(np.sum(dr * dr, axis=-1))
        scale = np.where(nrm > max_step, max_step / np.maximum(nrm, 1e-30), 1.0)
        pos = pos + dr * scale[..., None]
    band[1:-1] = pos
    e_idpp[1:-1] = _idpp_value_gradient(pos, frozen, d_mov, d_fix, cell)[0]
    return band, e_idpp

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

POS = np.array([[1.40, 1.40, 1.90], [2.43, 1.40, 2.26],
                [0.88, 2.29, 2.26], [0.88, 0.51, 2.26],
                [4.20, 2.80, 1.80], [4.20, 2.80, 2.77]])
DISP = np.array([[0.10, 0.05, 0.02], [0.08, 0.04, -0.03],
                 [0.05, 0.10, 0.01], [2.60, 1.30, -0.30],
                 [-0.04, 0.02, 0.05], [0.02, -0.03, 0.01]])
"""
    return [
        # normal: the production twenty-image band above the benchmark slab
        {"setup": shared + """
slab = slab_of(2.8, 4, 1.4, 3)
cell = np.array([11.2, 11.2])
""",
         "call": "pack(build_idpp_band(POS, DISP, slab, cell, 20))",
         "gold_call": "pack(_oracle_build_idpp_band(POS, DISP, slab, cell, 20))"},
        # normal: the six-image coarse band with a longer, gentler refinement
        {"setup": shared + """
slab = slab_of(2.8, 4, 1.4, 3)
cell = np.array([11.2, 11.2])
""",
         "call": "pack(build_idpp_band(POS, DISP, slab, cell, 6, 120, 0.005, 0.02))",
         "gold_call":
             "pack(_oracle_build_idpp_band(POS, DISP, slab, cell, 6, 120, 0.005, 0.02))"},
        # boundary: no refinement at all, so the band is the straight line and
        # the reported potential is that of the unrefined images
        {"setup": shared + """
slab = slab_of(2.8, 4, 1.4, 3)
cell = np.array([11.2, 11.2])
""",
         "call": "pack(build_idpp_band(POS, DISP, slab, cell, 7, 0))",
         "gold_call": "pack(_oracle_build_idpp_band(POS, DISP, slab, cell, 7, 0))"},
        # boundary: no frozen anchors, so only the moving pairs enter the
        # objective and the minimum image convention plays no part
        {"setup": shared + """
cell = np.array([11.2, 11.2])
""",
         "call":
             "pack(build_idpp_band(POS, DISP, np.zeros((0, 3)), cell, 9))",
         "gold_call":
             "pack(_oracle_build_idpp_band(POS, DISP, np.zeros((0, 3)), cell, 9))"},
        # boundary: a step cap so small that every atom moves by exactly the cap
        {"setup": shared + """
slab = slab_of(2.8, 4, 1.4, 3)
cell = np.array([11.2, 11.2])
""",
         "call": "pack(build_idpp_band(POS, DISP, slab, cell, 8, 30, 1.0, 1e-4))",
         "gold_call":
             "pack(_oracle_build_idpp_band(POS, DISP, slab, cell, 8, 30, 1.0, 1e-4))"},
        # edge: two images, which are the endpoints alone, so nothing is
        # refined and both reported potentials are zero
        {"setup": shared + """
cell = np.array([11.2, 11.2])
""",
         "call":
             "pack(build_idpp_band(POS, DISP, np.zeros((0, 3)), cell, 2))",
         "gold_call":
             "pack(_oracle_build_idpp_band(POS, DISP, np.zeros((0, 3)), cell, 2))"},
        # edge: a displacement of exactly zero, where every target distance is
        # already met and the potential stays at zero throughout
        {"setup": shared + """
cell = np.array([11.2, 11.2])
""",
         "call": ("pack(build_idpp_band(POS, np.zeros((6, 3)), "
                  "np.zeros((0, 3)), cell, 5))"),
         "gold_call": ("pack(_oracle_build_idpp_band(POS, np.zeros((6, 3)), "
                       "np.zeros((0, 3)), cell, 5))")},
        # invalid: fewer than two images
        {"setup": shared + """
cell = np.array([11.2, 11.2])
def run_model():
    try:
        build_idpp_band(POS, DISP, np.zeros((0, 3)), cell, 1)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_build_idpp_band(POS, DISP, np.zeros((0, 3)), cell, 1)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
        # invalid: a non-positive descent step
        {"setup": shared + """
cell = np.array([11.2, 11.2])
def run_model():
    try:
        build_idpp_band(POS, DISP, np.zeros((0, 3)), cell, 5, 10, 0.0)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_build_idpp_band(POS, DISP, np.zeros((0, 3)), cell, 5, 10, 0.0)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
    ]
