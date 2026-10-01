"""
The microstructure is a square array of heavily doped squares set in a lightly doped host. Within the cell [0, L] x [0, L] the embedded phase occupies

$$x1 < L/2 and x2 < L/2,$$

so it is a square of side L/2 filling exactly a quarter of the cell area, and the grid point of index (0, 0) sits on its lower left corner. Repeated periodically this is a square lattice of squares of side L/2 on a lattice of pitch L, and the corners at which four quadrant boundaries meet are what make it a severe test: the exact field has an integrable singularity at each of them, and a representation by a finite trigonometric series has to approximate a discontinuity that is not aligned with any single mode.

The conductivity used by the solver is this field sampled at the N + 1 grid points per direction, at the positions a * L / (N + 1). Sampling it is not free of consequence. With N even there are N/2 + 1 sample coordinates strictly below L/2 out of N + 1, so the sampled area fraction is

$$[(N/2 + 1) / (N + 1)]^2,$$

which exceeds one quarter by a term of order 1/N. The grid therefore represents a square slightly larger than the real one, and no refinement of the Fourier truncation can remove that: it is an error in the geometry rather than in the representation of the field on it, and it is the reason the achievable error settles at a floor rather than falling to zero. Reporting it explicitly is how that floor is accounted for afterwards rather than mistaken for a failure of the solver.

It matters equally that the sampled field is used for one purpose only, namely applying the constitutive law pointwise inside the iteration. Whenever the resulting field is later measured against the truth, the truth belongs to the real microstructure, with its interface exactly at L/2 and its area fraction exactly one quarter, and not to the sampled copy. Conflating the two removes the whole effect being studied, because a field scored against the material the solver was handed is being scored against its own assumptions.

Returns
-------
dict, the conductivity field sampled on the grid, with the area fraction that sampling actually represents against the exact quarter.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def checkerboard_conductivity(
    n_grid: int,
    c_matrix: float,
    c_inclusion: float,
    period: float,
) -> dict:
    """Sample the two-phase conductivity field on the grid and report what the sampling costs.

    Parameters
    ----------
    n_grid : int
        The number N of grid intervals, even and above zero.
    c_matrix : float
        Conductivity of the surrounding phase in siemens per metre, above zero.
    c_inclusion : float
        Conductivity of the embedded phase in siemens per metre, above zero and above c_matrix.
    period : float
        The cell edge in metre, above zero.

    Returns
    -------
    dict
        Under the keys conductivity, coordinate, inclusion_points, sampled_area_fraction, exact_area_fraction and fraction_defect. Write n for n_grid + 1. conductivity is a real array of shape (n, n) indexed by the two sample indices, and coordinate is the one-dimensional array of the n sample coordinates. The remaining four entries are scalars: inclusion_points is the number of sample coordinates along one edge that fall below half the cell edge, counted in one direction and not over the two-dimensional grid, so that sampled_area_fraction is its square divided by n squared; exact_area_fraction is one quarter; and fraction_defect is sampled_area_fraction minus exact_area_fraction.

    Raises
    ------
    ValueError
        When N fails to be a positive even integer, when either conductivity or the period fails to be finite and above zero, or when the embedded phase fails to be the more conducting of the two.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _even_positive(value, label):
    """Return an argument as an int once it is known to be a positive even integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be an integer" % label)
    out = int(value)
    if out <= 0 or out % 2 != 0:
        raise ValueError("%s must be a positive even integer" % label)
    return out


def _positive_float(value, label):
    """Return an argument as a float once it is known to be finite and above zero."""
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError("%s must be finite and above zero" % label)
    return out


def _oracle_checkerboard_conductivity(
    n_grid: int,
    c_matrix: float,
    c_inclusion: float,
    period: float,
) -> dict:
    """Reference implementation."""
    big_n = _even_positive(n_grid, "n_grid")
    c1 = _positive_float(c_matrix, "c_matrix")
    c2 = _positive_float(c_inclusion, "c_inclusion")
    length = _positive_float(period, "period")

    if c2 <= c1:
        raise ValueError("c_inclusion must exceed c_matrix")

    n = big_n + 1
    coordinate = np.arange(n) * (length / n)
    inside = coordinate < 0.5 * length
    conductivity = np.where(inside[:, None] & inside[None, :], c2, c1)
    count = int(inside.sum())
    sampled = float(count) ** 2 / float(n) ** 2
    return {
        "conductivity": conductivity,
        "coordinate": coordinate,
        "inclusion_points": count,
        "sampled_area_fraction": sampled,
        "exact_area_fraction": 0.25,
        "fraction_defect": sampled - 0.25,
    }

# =============================================================================
# TEST CASES
# =============================================================================

FLAT = """
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            out.extend(flat(v))
        return tuple(out)
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    if isinstance(x, bool):
        return (int(x),)
    return (x,)
"""


def test_cases():
    return [
        {
            # counts and fractions across grid sizes, and the sign and size of the defect
            "setup": """
def digest(out):
    return (out["inclusion_points"], round(out["sampled_area_fraction"], 14),
            round(out["exact_area_fraction"], 14), round(out["fraction_defect"], 14))
""" + FLAT,
            "call": "flat((digest(checkerboard_conductivity(2, 1.0, 100.0, 1.0)), digest(checkerboard_conductivity(4, 1.0, 100.0, 1.0)), digest(checkerboard_conductivity(8, 1.0, 100.0, 1.0)), digest(checkerboard_conductivity(96, 1.0, 100.0, 1.0)), digest(checkerboard_conductivity(512, 1.0, 100.0, 1.0))))",
            "gold_call": "flat((digest(_oracle_checkerboard_conductivity(2, 1.0, 100.0, 1.0)), digest(_oracle_checkerboard_conductivity(4, 1.0, 100.0, 1.0)), digest(_oracle_checkerboard_conductivity(8, 1.0, 100.0, 1.0)), digest(_oracle_checkerboard_conductivity(96, 1.0, 100.0, 1.0)), digest(_oracle_checkerboard_conductivity(512, 1.0, 100.0, 1.0))))",
        },
        {
            # the array itself: corner values, the row sums, and the fact that the mean
            # over the grid is the sampled rather than the exact rule of mixtures
            "setup": """
import numpy as np
def digest(out):
    c = np.asarray(out["conductivity"])
    x = np.asarray(out["coordinate"])
    return (round(float(np.abs(c).sum()), 6), round(float(np.abs(x).sum()), 16),
            round(float(c[0, 0]), 10), round(float(c[-1, -1]), 10),
            round(float(c[0, -1]), 10), round(float(c[-1, 0]), 10),
            round(float(c.mean()), 12), round(float(c[2, 2]), 10),
            round(float(x[1]), 16), round(float(x[-1]), 16), int(c.shape[0]))
""" + FLAT,
            "call": "flat((digest(checkerboard_conductivity(8, 192.26119608, 19226.119608, 2.0e-6)), digest(checkerboard_conductivity(10, 1.0, 100.0, 1.0)), digest(checkerboard_conductivity(6, 2.0, 5.0, 0.5))))",
            "gold_call": "flat((digest(_oracle_checkerboard_conductivity(8, 192.26119608, 19226.119608, 2.0e-6)), digest(_oracle_checkerboard_conductivity(10, 1.0, 100.0, 1.0)), digest(_oracle_checkerboard_conductivity(6, 2.0, 5.0, 0.5))))",
        },
        {
            # the sampled fraction must fall towards one quarter from above as N grows,
            # and the field must scale exactly with a common factor on both phases
            "setup": """
import numpy as np
def monotone(fn):
    d = [fn(n, 1.0, 100.0, 1.0)["fraction_defect"] for n in (8, 16, 32, 64, 128, 256)]
    falling = int(all(d[i] > d[i + 1] > 0.0 for i in range(len(d) - 1)))
    return (falling, round(d[0], 12), round(d[-1], 12))
def scaling(fn):
    a = np.asarray(fn(12, 1.0, 100.0, 1.0)["conductivity"])
    b = np.asarray(fn(12, 7.5, 750.0, 1.0)["conductivity"])
    return (round(float(np.max(np.abs(b - 7.5 * a))), 12),)
""" + FLAT,
            "call": "flat((monotone(checkerboard_conductivity), scaling(checkerboard_conductivity)))",
            "gold_call": "flat((monotone(_oracle_checkerboard_conductivity), scaling(_oracle_checkerboard_conductivity)))",
        },
        {
            "setup": """
def verdict(fn, n=8, c1=1.0, c2=100.0, L=1.0):
    try:
        fn(n, c1, c2, L)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(checkerboard_conductivity, n=7), verdict(checkerboard_conductivity, n=0), verdict(checkerboard_conductivity, n=-4), verdict(checkerboard_conductivity, c1=0.0), verdict(checkerboard_conductivity, c2=-1.0), verdict(checkerboard_conductivity, L=float('nan')), verdict(checkerboard_conductivity, c1=100.0, c2=100.0), verdict(checkerboard_conductivity, c1=200.0, c2=100.0), verdict(checkerboard_conductivity)))",
            "gold_call": "flat((verdict(_oracle_checkerboard_conductivity, n=7), verdict(_oracle_checkerboard_conductivity, n=0), verdict(_oracle_checkerboard_conductivity, n=-4), verdict(_oracle_checkerboard_conductivity, c1=0.0), verdict(_oracle_checkerboard_conductivity, c2=-1.0), verdict(_oracle_checkerboard_conductivity, L=float('nan')), verdict(_oracle_checkerboard_conductivity, c1=100.0, c2=100.0), verdict(_oracle_checkerboard_conductivity, c1=200.0, c2=100.0), verdict(_oracle_checkerboard_conductivity)))",
        },
    ]
