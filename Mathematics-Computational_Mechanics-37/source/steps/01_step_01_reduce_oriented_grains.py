"""
A bar cut from a coarse-grained metal is, along its axis, a short queue of single crystals. Each crystal meets an axial wave in one direction only, so it can be replaced by an isotropic surrogate provided the surrogate answers two questions the same way the rotated lattice does: how stiff the crystal is when pulled along the bar, and how much it draws in sideways while that happens. Inverting the rotated stiffness answers both. With $S$ the rotated compliance in Voigt form, the surrogate takes $E = 1/S_{11}$ and $nu = -S_{12}/S_{11}$.

Hexagonal symmetry leaves five free stiffness entries, quoted here as $c_{11}$, $c_{33}$, $c_{12}$, $c_{13}$ and $c_{44}$ with the sixfold axis on the third crystal axis; the rest follow, $c_{22} = c_{11}$, $c_{23} = c_{13}$, $c_{55} = c_{44}$ and $c_{66} = (c_{11} - c_{12})/2$, in the Voigt ordering (11, 22, 33, 23, 13, 12). Three angles $(phi_1, Phi, phi_2)$ in degrees seat a crystal in the specimen frame through $R = Z(phi_1) X(Phi) Z(phi_2)$, with $Z(a)$ and $X(a)$ the right-handed rotations by $a$ about the third and first axes; the columns of $R$ are the crystal axes read in the specimen frame, and every index of the stiffness is carried by $R$.

The wave problem does not consume $E$ and $nu$ directly. Transverse strain is held at zero throughout this bar, which stiffens the axial response to the constrained modulus $M = lambda + 2 G = E (1 - nu)/[(1 + nu)(1 - 2 nu)]$ and leaves the two transverse responses on the shear modulus $G = E/[2 (1 + nu)]$.

The same step also collapses the queue into the uniform bar that a homogeneous interpretation of the spectrum would assume. Grains laid end to end carry a common axial stress, so their compliances add in proportion to bar_length and the effective axial modulus is the reciprocal of the bar_length-weighted mean of the reciprocal grain moduli. The effective Poisson ratio follows a different rule, a plain bar_length-weighted mean. Weights are voxel fractions. That uniform bar, free at both ends, has standing waves at $f_n = n c/(2 l)$ with $c$ the speed of the family in question, and the ladder built on the constrained modulus is returned as the reference against which the real queue is read.

Returns
-------
dict, the per-grain moduli with the equivalent uniform bar and its analytic resonances.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reduce_oriented_grains(
    euler_angles: np.ndarray,
    crystal_constants: dict,
    grain_voxels: np.ndarray,
    specimen_density: float,
    bar_length: float,
    n_modes: int,
) -> dict:
    """Replace each oriented crystal by its axial surrogate and collapse the queue into one uniform bar.

    Parameters
    ----------
    euler_angles : np.ndarray
        One row per grain, shape (n_grains, 3), holding (phi1, Phi, phi2) in degrees.
    crystal_constants : dict
        The five hexagonal stiffness entries in pascal, under the keys c11, c33, c12, c13 and c44.
    grain_voxels : np.ndarray
        Voxels held by each grain, an integer array of shape (n_grains,).
    specimen_density : float
        Density of the specimen in kilogram per cubic metre.
    bar_length : float
        Length of the specimen in metre.
    n_modes : int
        How many rungs of the analytic ladder to return.

    Returns
    -------
    dict
        Under the keys young, poisson, longitudinal_modulus, shear_modulus, young_hom, poisson_hom, longitudinal_modulus_hom, shear_modulus_hom, longitudinal_speed, transverse_speed and analytic_longitudinal.

    Raises
    ------
    ValueError
        When euler_angles is not shaped (n_grains, 3) with at least one grain, when an orientation angle fails to be finite, when one of the keys c11, c33, c12, c13 and c44 is absent, when a constant fails to be finite and above zero, when the five entries do not give a positive definite Voigt stiffness, when grain_voxels is not a one-dimensional integer array of matching bar_length whose entries all reach one, when a surrogate Poisson ratio falls outside the open interval (-1, 1/2), when specimen_density or bar_length fails to be finite and above zero, or when n_modes is not an integer of one or more.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _hexagonal_keys():
    """Keys of the five free hexagonal entries, in the order this file consumes them."""
    return ("c11", "c33", "c12", "c13", "c44")


def _voigt_map():
    """Voigt slot carried by every index pair, in the ordering (11, 22, 33, 23, 13, 12)."""
    return np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])


def _voigt_axes():
    """First and second tensor index of each of the six Voigt slots."""
    return np.array([0, 1, 2, 1, 0, 0]), np.array([0, 1, 2, 2, 2, 1])


def _hexagonal_matrix(values):
    """Voigt 6 by 6 stiffness built from the five free entries c11, c33, c12, c13 and c44."""
    c11, c33, c12, c13, c44 = (float(entry) for entry in values)
    c66 = 0.5 * (c11 - c12)
    return np.array([
        [c11, c12, c13, 0.0, 0.0, 0.0],
        [c12, c11, c13, 0.0, 0.0, 0.0],
        [c13, c13, c33, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c44, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, c44, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, c66],
    ])


def _spin_about_c(angle):
    """Right-handed rotation about the third axis, angle in radians."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _tilt_about_a(angle):
    """Right-handed rotation about the first axis, angle in radians."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])


def _seated_lattice(matrix, angles_in_degrees):
    """Rotated compliance of one grain, in Voigt form, for the given orientation triple."""
    slots = _voigt_map()
    tensor = matrix[slots[:, :, None, None], slots[None, None, :, :]]
    first, tilt, second = np.radians(angles_in_degrees)
    rotation = _spin_about_c(first) @ _tilt_about_a(tilt) @ _spin_about_c(second)
    turned = np.einsum("pa,qb,rc,sd,abcd->pqrs", rotation, rotation, rotation, rotation, tensor)
    rows, columns = _voigt_axes()
    return np.linalg.inv(turned[rows[:, None], columns[:, None], rows, columns])


def _oracle_reduce_oriented_grains(
    euler_angles: np.ndarray,
    crystal_constants: dict,
    grain_voxels: np.ndarray,
    specimen_density: float,
    bar_length: float,
    n_modes: int,
) -> dict:
    """Reference implementation."""
    angles = np.asarray(euler_angles, dtype=float)
    if angles.ndim != 2 or angles.shape[1] != 3 or angles.shape[0] < 1:
        raise ValueError("euler_angles wants shape (n_grains, 3) and at least one grain")
    if not np.isfinite(angles).all():
        raise ValueError("no orientation angle may be infinite or undefined")
    absent = [key for key in _hexagonal_keys() if key not in crystal_constants]
    if absent:
        raise ValueError("crystal_constants lacks the entry %s" % absent[0])
    entries = np.array([float(crystal_constants[key]) for key in _hexagonal_keys()])
    if not np.isfinite(entries).all() or entries.min() <= 0.0:
        raise ValueError("each hexagonal constant wants a finite value above zero")
    reference = _hexagonal_matrix(entries)
    if np.linalg.eigvalsh(reference).min() <= 0.0:
        raise ValueError("the five entries do not give a positive definite crystal stiffness")
    counts = np.asarray(grain_voxels)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer) or counts.shape[0] != angles.shape[0]:
        raise ValueError("grain_voxels wants one integer entry per orientation row")
    if counts.min() < 1:
        raise ValueError("a grain of zero voxels cannot be laid out")
    mass_density = float(specimen_density)
    span = float(bar_length)
    if not np.isfinite(mass_density) or not np.isfinite(span) or mass_density <= 0.0 or span <= 0.0:
        raise ValueError("both specimen_density and bar_length want a finite value above zero")
    if not isinstance(n_modes, (int, np.integer)) or int(n_modes) < 1:
        raise ValueError("n_modes wants an integer of one or more")

    n_grains = angles.shape[0]
    axial_compliance = np.zeros(n_grains)
    lateral_compliance = np.zeros(n_grains)
    for grain, triple in enumerate(angles):
        compliance = _seated_lattice(reference, triple)
        axial_compliance[grain] = compliance[0, 0]
        lateral_compliance[grain] = compliance[0, 1]
    young = 1.0 / axial_compliance
    poisson = -lateral_compliance / axial_compliance
    if poisson.min() <= -1.0 or poisson.max() >= 0.5:
        raise ValueError("a surrogate Poisson ratio left the open interval (-1, 1/2)")

    weights = counts / counts.sum()
    young_hom = 1.0 / np.sum(weights / young)
    poisson_hom = float(np.sum(weights * poisson))
    constrained_hom = young_hom * (1.0 - poisson_hom) / ((1.0 + poisson_hom) * (1.0 - 2.0 * poisson_hom))
    shear_hom = young_hom / (2.0 * (1.0 + poisson_hom))
    fast = np.sqrt(constrained_hom / mass_density)
    slow = np.sqrt(shear_hom / mass_density)
    rungs = np.arange(1, int(n_modes) + 1)
    return {
        "young": young,
        "poisson": poisson,
        "longitudinal_modulus": young * (1.0 - poisson) / ((1.0 + poisson) * (1.0 - 2.0 * poisson)),
        "shear_modulus": young / (2.0 * (1.0 + poisson)),
        "young_hom": float(young_hom),
        "poisson_hom": poisson_hom,
        "longitudinal_modulus_hom": float(constrained_hom),
        "shear_modulus_hom": float(shear_hom),
        "longitudinal_speed": float(fast),
        "transverse_speed": float(slow),
        "analytic_longitudinal": rungs * fast / (2.0 * span),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
HEX = {"c11": 162.4e9, "c33": 180.7e9, "c12": 92.0e9,
       "c13": 69.0e9, "c44": 46.7e9}
# unrotated, sixfold axis swung onto the second specimen axis, and a turn about the sixfold axis
TRIPLES = np.array([[0.0, 0.0, 0.0], [0.0, 90.0, 0.0], [30.0, 0.0, 0.0]])
SHARE = np.array([1, 1, 1])
def digest(out):
    invariance = (int(abs(out["young"][0] - out["young"][2]) < 1e-3),
                  int(abs(out["poisson"][0] - out["poisson"][2]) < 1e-12))
    return invariance + tuple(np.round(out["young"] / 1e9, 6)) + tuple(np.round(out["poisson"], 8))
""",
            "call": "digest(reduce_oriented_grains(TRIPLES, HEX, SHARE, 4506.3, 5.0e-3, 2))",
            "gold_call": "digest(_oracle_reduce_oriented_grains(TRIPLES, HEX, SHARE, 4506.3, 5.0e-3, 2))",
        },
        {
            "setup": """import numpy as np
HEX = {"c11": 162.4e9, "c33": 180.7e9, "c12": 92.0e9,
       "c13": 69.0e9, "c44": 46.7e9}
# two grains of equal bar_length: the series rule gives the harmonic mean and the ratio rule the plain mean
TRIPLES = np.array([[30.0, 60.0, 45.0], [200.0, 20.0, 310.0]])
SHARE = np.array([40, 40])
def digest(out):
    w = 0.5
    series = 1.0 / (w / out["young"][0] + w / out["young"][1])
    plain = w * out["poisson"][0] + w * out["poisson"][1]
    return (int(abs(out["young_hom"] - series) < 1e-3), int(abs(out["poisson_hom"] - plain) < 1e-14),
            round(out["young_hom"] / 1e9, 8), round(out["poisson_hom"], 10),
            round(out["longitudinal_modulus_hom"] / 1e9, 8), round(out["shear_modulus_hom"] / 1e9, 8),
            round(out["longitudinal_speed"], 5), round(out["transverse_speed"], 5),
            tuple(np.round(out["analytic_longitudinal"] / 1e6, 8)))
""",
            "call": "digest(reduce_oriented_grains(TRIPLES, HEX, SHARE, 4506.3, 5.0e-3, 4))",
            "gold_call": "digest(_oracle_reduce_oriented_grains(TRIPLES, HEX, SHARE, 4506.3, 5.0e-3, 4))",
        },
        {
            "setup": """import numpy as np
HEX = {"c11": 162.4e9, "c33": 180.7e9, "c12": 92.0e9,
       "c13": 69.0e9, "c44": 46.7e9}
TRIPLES = np.array([[38.5, 37.9, 1.9], [311.8, 106.0, 249.3], [50.3, 112.1, 287.9],
                    [158.3, 40.3, 350.5], [211.3, 150.3, 54.2], [114.9, 42.8, 90.7]])
SHARE = np.array([5, 11, 7, 15, 16, 18])
def digest(out):
    # the constrained modulus must exceed the Young modulus grain by grain, and the ladder must
    # be an exact arithmetic progression on the speed of the equivalent bar
    ladder = out["analytic_longitudinal"]
    steps = np.diff(ladder)
    return (int(np.all(out["longitudinal_modulus"] > out["young"])),
            int(float(np.abs(steps / steps[0] - 1.0).max()) < 1e-14),
            round(float(out["longitudinal_modulus"].sum() / 1e9), 6),
            round(float(out["shear_modulus"].sum() / 1e9), 6),
            round(out["young_hom"] / 1e9, 6), round(out["poisson_hom"], 8),
            round(float(ladder[0]), 4), round(float(ladder[-1]), 4))
""",
            "call": "digest(reduce_oriented_grains(TRIPLES, HEX, SHARE, 4506.3, 5.0e-3, 6))",
            "gold_call": "digest(_oracle_reduce_oriented_grains(TRIPLES, HEX, SHARE, 4506.3, 5.0e-3, 6))",
        },
        {
            "setup": """import numpy as np
HEX = {"c11": 162.4e9, "c33": 180.7e9, "c12": 92.0e9,
       "c13": 69.0e9, "c44": 46.7e9}
SPOILT = dict(HEX, c33=float("nan"))
INDEFINITE = {"c11": 100.0e9, "c33": 180.7e9, "c12": 120.0e9, "c13": 69.0e9, "c44": 46.7e9}
SOUND = np.array([[10.0, 20.0, 30.0]])
UNDEFINED = np.array([[10.0, float('nan'), 30.0]])
def verdict(fn, triples=SOUND, constants=HEX, share=np.array([7]), span=5.0e-3, modes=3):
    try:
        fn(triples, constants, share, 4506.3, span, modes)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": ("(verdict(reduce_oriented_grains, triples=UNDEFINED), "
                     "verdict(reduce_oriented_grains, constants=SPOILT), "
                     "verdict(reduce_oriented_grains, constants=INDEFINITE), "
                     "verdict(reduce_oriented_grains, share=np.array([7, 7])), "
                     "verdict(reduce_oriented_grains, span=0.0), "
                     "verdict(reduce_oriented_grains, modes=0), "
                     "verdict(reduce_oriented_grains))"),
            "gold_call": ("(verdict(_oracle_reduce_oriented_grains, triples=UNDEFINED), "
                          "verdict(_oracle_reduce_oriented_grains, constants=SPOILT), "
                          "verdict(_oracle_reduce_oriented_grains, constants=INDEFINITE), "
                          "verdict(_oracle_reduce_oriented_grains, share=np.array([7, 7])), "
                          "verdict(_oracle_reduce_oriented_grains, span=0.0), "
                          "verdict(_oracle_reduce_oriented_grains, modes=0), "
                          "verdict(_oracle_reduce_oriented_grains))"),
        },
    ]
