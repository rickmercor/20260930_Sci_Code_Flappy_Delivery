"""
A Fourier solver knows only periodic cells, so a free-standing bar has to be embedded in one. Lay the specimen down first, one voxel per grid point, each grain taking its prescribed run of voxels, then append a block of padding voxels whose job is to stand in for the emptiness outside the specimen. Closing the cell periodically puts that padding between the far end of the bar and its struck end. The cell is required to hold an odd number of voxels, which keeps the integer frequency set of the next stage symmetric about zero and free of a Nyquist entry.

Stiffness in this discretisation does not belong to a voxel. Displacement is sampled at voxel centres and the material is uniform within a voxel, so the elastic link that matters is the one joining two neighbouring centres, and that link runs through half of one voxel and half of the other. Two springs in series: each half contributes a compliance $h/(2 M)$ with its own modulus, the compliances add, and the link therefore carries

$$M_{link} = 2 M_{j} M_{j+1} / (M_{j} + M_{j+1}),$$

the harmonic mean of the two. Inside a grain the two are equal and the rule returns the grain value unchanged, so the rule bites only where two grains meet. Arithmetic averaging, or simply handing the link the modulus of the voxel on one side, both overstate the stiffness of a mismatched pair and both move every resonance of the bar.

The padding is not a material and the series rule does not apply to it. Two conventions are imposed instead. A link with a specimen voxel on one side and a padding voxel on the other carries the modulus of the specimen voxel, so that the padding hangs on a spring of the specimen's own stiffness; there are two such links, one at each end of the bar, because the cell is periodic. A link with padding on both sides carries the padding modulus.

The padding properties themselves are quoted as multiples of specimen averages: each padding modulus is a factor times the voxel mean of the matching specimen modulus, and the padding density is a factor times the specimen density, so that a factor of zero produces padding that is exactly stress free or exactly massless. Three rows come back because the three displacement components separate into one axial and two transverse scalar problems, row one carrying the constrained modulus and rows two and three the shear modulus.

Returns
-------
dict, the padded periodic cell with its voxel and link moduli.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_face_stiffness_cell(
    grain_voxels: np.ndarray,
    longitudinal_modulus: np.ndarray,
    shear_modulus: np.ndarray,
    specimen_density: float,
    n_pad: int,
    pad_stiffness_factor: float,
    pad_density_factor: float,
) -> dict:
    """Lay the specimen and its padding out, and give every link between neighbouring centres its modulus.

    Parameters
    ----------
    grain_voxels : np.ndarray
        Voxels held by each grain, an integer array of shape (n_grains,), none of them below one.
    longitudinal_modulus : np.ndarray
        Constrained longitudinal modulus of each grain in pascal, shape (n_grains,).
    shear_modulus : np.ndarray
        Shear modulus of each grain in pascal, shape (n_grains,).
    specimen_density : float
        Density of the specimen in kilogram per cubic metre.
    n_pad : int
        How many padding voxels follow the specimen, one or more.
    pad_stiffness_factor : float
        Padding modulus as a multiple of the specimen mean modulus, not negative.
    pad_density_factor : float
        Padding density as a multiple of the specimen density, not negative.

    Returns
    -------
    dict
        Under the keys n_specimen, n_total, density, voxel_modulus and face_modulus.

    Raises
    ------
    ValueError
        When grain_voxels is not a one-dimensional integer array whose entries all reach one, when either modulus array fails to match its length or holds an entry that is not finite and above zero, when specimen_density fails to be finite and above zero, when n_pad is not an integer of one or more, when a factor is not finite or drops below zero, or when the voxels total an even number.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _counted(value, label):
    """Return a count as a native int once it is known to be an integer of one or more."""
    if not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError("%s wants an integer of one or more" % label)
    return int(value)


def _checked_factor(value, label):
    """Return a padding factor as a float once it is known to be finite and non-negative."""
    factor = float(value)
    if not np.isfinite(factor):
        raise ValueError("the %s factor must be finite" % label)
    if factor < 0.0:
        raise ValueError("the %s factor must not be negative" % label)
    return factor


def _link_moduli(voxel_row, n_specimen):
    """Modulus of every link, harmonic inside the specimen and prescribed wherever padding is involved."""
    left = voxel_row
    right = np.roll(voxel_row, -1)
    total = left + right
    series = 2.0 * left * right / np.where(total > 0.0, total, 1.0)
    on_left = np.arange(voxel_row.size) < n_specimen
    on_right = np.roll(on_left, -1)
    inside = on_left & on_right
    straddling = on_left ^ on_right
    links = np.where(inside, series, left)
    return np.where(straddling, np.where(on_left, left, right), links)


def _oracle_assemble_face_stiffness_cell(
    grain_voxels: np.ndarray,
    longitudinal_modulus: np.ndarray,
    shear_modulus: np.ndarray,
    specimen_density: float,
    n_pad: int,
    pad_stiffness_factor: float,
    pad_density_factor: float,
) -> dict:
    """Reference implementation."""
    counts = np.asarray(grain_voxels)
    if counts.ndim != 1 or counts.size < 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("grain_voxels wants a one-dimensional array of integers")
    if counts.min() < 1:
        raise ValueError("a grain of zero voxels cannot be laid out")
    axial = np.asarray(longitudinal_modulus, dtype=float)
    lateral = np.asarray(shear_modulus, dtype=float)
    if axial.shape != counts.shape or lateral.shape != counts.shape:
        raise ValueError("one longitudinal and one shear modulus are wanted per grain")
    if not (np.isfinite(axial).all() and np.isfinite(lateral).all()):
        raise ValueError("no grain modulus may be infinite or undefined")
    if min(axial.min(), lateral.min()) <= 0.0:
        raise ValueError("every grain modulus must sit above zero")
    specimen_density = float(specimen_density)
    if not np.isfinite(specimen_density) or specimen_density <= 0.0:
        raise ValueError("density wants a finite value above zero")
    pad = _counted(n_pad, "n_pad")
    stiffness_ratio = _checked_factor(pad_stiffness_factor, "padding stiffness")
    density_ratio = _checked_factor(pad_density_factor, "padding density")

    n_specimen = int(counts.sum())
    n_total = n_specimen + pad
    if n_total % 2 == 0:
        raise ValueError("this configuration lays out an odd cell, and %d is even" % n_total)

    axial_voxels = np.repeat(axial, counts)
    lateral_voxels = np.repeat(lateral, counts)
    axial_row = np.concatenate([axial_voxels, np.full(pad, stiffness_ratio * axial_voxels.mean())])
    lateral_row = np.concatenate([lateral_voxels, np.full(pad, stiffness_ratio * lateral_voxels.mean())])
    voxel_modulus = np.stack([axial_row, lateral_row, lateral_row])
    voxel_density = np.concatenate([
        np.full(n_specimen, specimen_density), np.full(pad, density_ratio * specimen_density)])
    return {
        "n_specimen": n_specimen,
        "n_total": n_total,
        "density": voxel_density,
        "voxel_modulus": voxel_modulus,
        "face_modulus": np.stack([_link_moduli(row, n_specimen) for row in voxel_modulus]),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
# three grains, so two internal joints; the harmonic rule must bite there and nowhere else
SHARE = np.array([4, 3, 5])
AXIAL = np.array([2.0, 3.0, 4.0]) * 1e10
LATERAL = np.array([1.0, 1.5, 2.0]) * 1e10
def digest(out):
    links = out["face_modulus"][0]
    joints = (2 * 2.0 * 3.0 / (2.0 + 3.0) * 1e10, 2 * 3.0 * 4.0 / (3.0 + 4.0) * 1e10)
    return (out["n_specimen"], out["n_total"],
            int(abs(links[3] - joints[0]) < 1.0), int(abs(links[6] - joints[1]) < 1.0),
            tuple(np.round(links / 1e9, 9)), tuple(np.round(out["face_modulus"][1] / 1e9, 9)),
            tuple(np.round(out["density"], 6)))
""",
            "call": "digest(assemble_face_stiffness_cell(SHARE, AXIAL, LATERAL, 2700.0, 3, 0.0, 1.0))",
            "gold_call": "digest(_oracle_assemble_face_stiffness_cell(SHARE, AXIAL, LATERAL, 2700.0, 3, 0.0, 1.0))",
        },
        {
            "setup": """import numpy as np
# a uniform specimen: every internal link must return the grain value, so the rule is invisible
SHARE = np.array([40, 40, 41])
AXIAL = np.full(3, 1.7e11)
LATERAL = np.full(3, 4.4e10)
def digest(out):
    links = out["face_modulus"]
    body = links[:, :120]
    return (out["n_specimen"], out["n_total"], links.shape, out["voxel_modulus"].shape,
            int(float(np.abs(body[0] / 1.7e11 - 1.0).max()) < 1e-15),
            int(float(np.abs(body[1] / 4.4e10 - 1.0).max()) < 1e-15),
            int(np.array_equal(links[1], links[2])),
            round(float(links[0, 120] / 1e9), 9), round(float(links[0, 130] / 1e9), 9),
            round(float(links[0, -1] / 1e9), 9), round(float(out["density"].sum()), 6))
""",
            "call": "digest(assemble_face_stiffness_cell(SHARE, AXIAL, LATERAL, 4506.3, 12, 1e-7, 0.0))",
            "gold_call": "digest(_oracle_assemble_face_stiffness_cell(SHARE, AXIAL, LATERAL, 4506.3, 12, 1e-7, 0.0))",
        },
        {
            "setup": """import numpy as np
SHARE = np.array([5, 11, 7, 15, 16, 18, 9, 16, 5, 6, 10, 11])
AXIAL = 1e9 * np.array([212.0311, 147.8495, 153.5629, 185.5353, 265.6189, 216.9061,
                        161.4598, 214.3794, 267.6499, 134.5997, 136.1903, 177.2734])
LATERAL = 1e9 * np.array([38.5713, 46.1385, 45.4753, 38.482, 37.2008, 40.8484,
                          40.0623, 38.6471, 36.855, 41.0413, 40.7109, 54.9612])
def digest(out):
    links, voxels = out["face_modulus"], out["voxel_modulus"]
    joints = np.flatnonzero(np.abs(np.diff(voxels[0, :129])) > 1.0)
    # a harmonic mean never exceeds the smaller of its two arguments doubled, and always falls
    # below the arithmetic mean unless the two agree
    pairs = np.stack([voxels[0, joints], voxels[0, joints + 1]])
    return (out["n_specimen"], out["n_total"], int(joints.size),
            int(np.all(links[0, joints] < pairs.mean(axis=0))),
            int(np.all(links[0, joints] > pairs.min(axis=0))),
            round(float(links[0, 128] / voxels[0, 128]), 12),
            round(float(links[0, 140] / voxels[0, 0]), 12),
            round(float(links[0, :129].sum() / 1e9), 6), round(float(links[1, :129].sum() / 1e9), 6))
""",
            "call": "digest(assemble_face_stiffness_cell(SHARE, AXIAL, LATERAL, 4506.3, 12, 0.0, 1.0))",
            "gold_call": "digest(_oracle_assemble_face_stiffness_cell(SHARE, AXIAL, LATERAL, 4506.3, 12, 0.0, 1.0))",
        },
        {
            "setup": """import numpy as np
SHARE = np.array([4, 3, 5])
AXIAL = np.array([2e10, 3e10, 4e10])
LATERAL = np.array([1e10, 1.5e10, 2e10])
SPOILT = np.array([2e10, float('nan'), 4e10])
def verdict(fn, pad=4, mass=2700.0, axial=AXIAL, stiff=0.0):
    try:
        fn(SHARE, axial, LATERAL, mass, pad, stiff, 1.0)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": ("(verdict(assemble_face_stiffness_cell, axial=SPOILT), "
                     "verdict(assemble_face_stiffness_cell, mass=float('nan')), "
                     "verdict(assemble_face_stiffness_cell, stiff=-1.0), "
                     "verdict(assemble_face_stiffness_cell, pad=3), "
                     "verdict(assemble_face_stiffness_cell))"),
            "gold_call": ("(verdict(_oracle_assemble_face_stiffness_cell, axial=SPOILT), "
                          "verdict(_oracle_assemble_face_stiffness_cell, mass=float('nan')), "
                          "verdict(_oracle_assemble_face_stiffness_cell, stiff=-1.0), "
                          "verdict(_oracle_assemble_face_stiffness_cell, pad=3), "
                          "verdict(_oracle_assemble_face_stiffness_cell))"),
        },
    ]
