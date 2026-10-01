"""
Return the two reaction-distance metrics by which the workflow ranks candidate interpolations before any band optimisation is attempted: the ranking metric $\\mu$ built from the interpolation-potential profile of the band, and the transfer-specific metric $\\tau$ for a reaction in which one atom moves from a donor centre to an acceptor centre.

The combinatorial stage of an automated barrier search produces far more candidate paths than can be optimised, so the candidates must be ranked by something much cheaper than a band optimisation. For reactions that transfer an atom the ranking needs a second ingredient, since a path can be geometrically smooth and energetically cheap and still describe the wrong elementary process, for instance a dissociation followed by a diffusion rather than a concerted transfer.



Conventions fixed by this task. The path coordinate of an image is the running sum, from zero at the first image, of the Frobenius norms of the differences between consecutive images, and wherever a quantity is accumulated along that coordinate, the trapezoidal rule on the images as they are given is used. The distance set of $\\tau$, which the source leaves generic, is all three mutual distances among the donor, the transferring atom and the acceptor, averaged over all images of the band, with the coordinates as given and no minimum image convention.

With $M$ images and $\\mathbf{R}_m$ the coordinates of image $m$, the path coordinate is



$$s_0=0,\\qquad s_m=s_{m-1}+\\lVert \\mathbf{R}_m-\\mathbf{R}_{m-1}\\rVert_F .$$

Returns
-------
A tuple `(mu, tau)` of two Python `float`s: the area under the interpolation-potential profile and the averaged summed donor-transfer-acceptor distance in angstrom.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reaction_distance_metrics(band: "np.ndarray", e_idpp: "np.ndarray",
                              donor: int, transfer: int,
                              acceptor: int) -> tuple:
    """The two reaction-distance metrics that rank candidate interpolations.

    Parameters
    ----------
    band : numpy.ndarray
        Band coordinates in angstrom, shape (M, N, 3) with M >= 2.
    e_idpp : numpy.ndarray
        Interpolation potential of each image, shape (M,).
    donor, transfer, acceptor : int
        Indices into the atom axis of band naming the donor centre, the
        transferring atom and the acceptor centre.

    Returns
    -------
    result : tuple
        (mu, tau), both Python floats: the ranking metric mu of the
        interpolation in eV*A and its transfer metric tau in A, under the
        conventions stated for this step.

    Raises
    ------
    ValueError
        If band is not an (M, N, 3) array with M >= 2, if e_idpp does not
        carry one entry per image, or if any of the three indices lies
        outside the atom range.
    """
    return mu, tau

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reaction_distance_metrics(band: "np.ndarray", e_idpp: "np.ndarray",
                                      donor: int, transfer: int,
                                      acceptor: int) -> tuple:
    band = np.asarray(band, dtype=float)
    e_idpp = np.asarray(e_idpp, dtype=float)
    if band.ndim != 3 or band.shape[2] != 3 or band.shape[0] < 2:
        raise ValueError("band must have shape (M, N, 3) with M >= 2")
    if e_idpp.shape != (band.shape[0],):
        raise ValueError("e_idpp must have one entry per image")
    n = band.shape[1]
    for name, idx in (("donor", donor), ("transfer", transfer),
                      ("acceptor", acceptor)):
        if not (0 <= int(idx) < n):
            raise ValueError(name + " index is outside the atom range")

    seg = band[1:] - band[:-1]
    ds = np.sqrt(np.sum(seg * seg, axis=(1, 2)))
    r = np.concatenate(([0.0], np.cumsum(ds)))
    mu = float(np.sum(0.5 * (e_idpp[1:] + e_idpp[:-1]) * np.diff(r)))

    d_id, t_id, a_id = int(donor), int(transfer), int(acceptor)
    dt = np.sqrt(np.sum((band[:, d_id] - band[:, t_id]) ** 2, axis=1))
    ta = np.sqrt(np.sum((band[:, t_id] - band[:, a_id]) ** 2, axis=1))
    da = np.sqrt(np.sum((band[:, d_id] - band[:, a_id]) ** 2, axis=1))
    tau = float(np.mean(dt + ta + da))
    return mu, tau

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    shared = """import numpy as np

def band_of(n_img, n_at, amp, seed):
    lam = np.linspace(0.0, 1.0, n_img)
    base = np.array([[float(i), 0.5 * i, 1.0 + 0.1 * i] for i in range(n_at)])
    move = np.zeros((n_at, 3))
    move[seed % n_at] = [amp, 0.5 * amp, -0.2 * amp]
    return base[None, :, :] + lam[:, None, None] * move[None, :, :]
"""
    return [
        # normal: a twenty-image band with a single-peaked interpolation profile
        {"setup": shared + """
band = band_of(20, 6, 2.6, 3)
e = 0.4 * np.sin(np.pi * np.linspace(0.0, 1.0, 20)) ** 2
e[0] = 0.0
e[-1] = 0.0
""",
         "call": "reaction_distance_metrics(band, e, 0, 3, 4)",
         "gold_call": "_oracle_reaction_distance_metrics(band, e, 0, 3, 4)"},
        # normal: a different donor, transfer and acceptor triple on the same
        # band, so tau changes while the path coordinate does not
        {"setup": shared + """
band = band_of(20, 6, 2.6, 3)
e = 0.4 * np.sin(np.pi * np.linspace(0.0, 1.0, 20)) ** 2
e[0] = 0.0
e[-1] = 0.0
""",
         "call": "reaction_distance_metrics(band, e, 5, 1, 2)",
         "gold_call": "_oracle_reaction_distance_metrics(band, e, 5, 1, 2)"},
        # boundary: an interpolation potential that is zero everywhere, so the
        # area vanishes however long the path is
        {"setup": shared + """
band = band_of(11, 5, 3.0, 2)
e = np.zeros(11)
""",
         "call": "reaction_distance_metrics(band, e, 0, 2, 4)",
         "gold_call": "_oracle_reaction_distance_metrics(band, e, 0, 2, 4)"},
        # boundary: a band whose images all coincide, so the path coordinate
        # never advances and the area is zero for a non-zero profile
        {"setup": """import numpy as np
band = np.repeat(np.array([[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0],
                            [0.6, 1.0, 0.0], [2.4, 0.3, 0.0]]]), 6, axis=0)
e = np.array([0.0, 0.3, 0.7, 0.9, 0.2, 0.0])
""",
         "call": "reaction_distance_metrics(band, e, 0, 1, 3)",
         "gold_call": "_oracle_reaction_distance_metrics(band, e, 0, 1, 3)"},
        # edge: two images only, where the trapezoidal rule has a single panel
        {"setup": shared + """
band = band_of(2, 4, 1.5, 1)
e = np.array([0.0, 0.0])
""",
         "call": "reaction_distance_metrics(band, e, 0, 1, 3)",
         "gold_call": "_oracle_reaction_distance_metrics(band, e, 0, 1, 3)"},
        # edge: the donor, transfer and acceptor all the same atom, so every
        # mutual distance is zero and tau vanishes
        {"setup": shared + """
band = band_of(7, 4, 2.0, 0)
e = np.linspace(0.0, 0.6, 7)
""",
         "call": "reaction_distance_metrics(band, e, 2, 2, 2)",
         "gold_call": "_oracle_reaction_distance_metrics(band, e, 2, 2, 2)"},
        # invalid: a profile with the wrong number of entries
        {"setup": shared + """
band = band_of(9, 4, 2.0, 1)
bad = np.zeros(8)
def run_model():
    try:
        reaction_distance_metrics(band, bad, 0, 1, 3)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_reaction_distance_metrics(band, bad, 0, 1, 3)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
        # invalid: an acceptor index outside the atom range
        {"setup": shared + """
band = band_of(9, 4, 2.0, 1)
e = np.zeros(9)
def run_model():
    try:
        reaction_distance_metrics(band, e, 0, 1, 9)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_reaction_distance_metrics(band, e, 0, 1, 9)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
    ]
