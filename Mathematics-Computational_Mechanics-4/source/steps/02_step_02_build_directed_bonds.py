"""
Enumerate the directed bonds of the peridynamic discretisation and remove those severed by the pre-crack.

Given the lattice produced by the previous step, point i owns a bond to point j whenever j is distinct from i and lies inside i's own horizon, that is ||X_j - X_i|| <= lattice[i, 3] within a relative tolerance of 1e-12. Because horizons vary in space this relation is not symmetric: a bond (i, j) may exist while (j, i) does not.

A bond is severed by the pre-crack when its reference segment crosses the horizontal line y = crack_y at an abscissa not exceeding crack_x_max. The segment crosses when (y_i - crack_y) * (y_j - crack_y) < 0, so a bond with an endpoint lying exactly on the crack line does not cross it; the crossing abscissa is then obtained by linear interpolation along the segment. Severed bonds are omitted from the output entirely, in both directions in which they appear.

Bonds are emitted in a fixed order: owner index ascending, and for a given owner, neighbour index ascending. Every downstream step indexes bonds by their position in this ordering, so the ordering is part of the contract.

The function returns one integer array with two columns [owner, neighbour], one row per surviving directed bond.

Raises ValueError if: lattice does not have shape (N, 5) with N >= 2; lattice contains non-finite values; any horizon in column 3 is not strictly positive; any volume in column 4 is not strictly positive; crack_y or crack_x_max is not finite; the lattice contains two coincident points.

The family of a material point is the index set of points with which it interacts, obtained by testing every candidate against that point's horizon. Under a uniform horizon this relation is symmetric, so families come in reciprocal pairs and the discrete interactions inherit the action-reaction structure of the continuum theory automatically. Under a spatially varying horizon that symmetry is lost. A coarse point with a large horizon may reach across a refinement interface to a fine point whose own, smaller horizon does not reach back. The resulting one-sided membership is the discrete origin of the ghost forces and spurious wave reflections that motivate the dual-horizon formulation, and it is why families must be enumerated and stored as directed pairs rather than as unordered ones.

Choosing the horizon ratio slightly above an integer keeps family membership insensitive to floating-point rounding, since no candidate point then sits within rounding distance of the horizon boundary. The same care is needed at the crack tip: a pre-crack terminating exactly on a lattice column would place a bond's crossing abscissa exactly on the acceptance threshold, where algebraically equivalent interpolation formulas can round to opposite sides and yield different families.

Fracture in peridynamics is represented purely through the presence or absence of bonds, with no explicit crack surface, no remeshing, and no enrichment of the approximation space. A pre-existing crack is therefore imposed as an initial condition on connectivity: every bond whose reference segment crosses the crack is deleted before the simulation begins, so that the two faces exert no force on one another. Because the crack is nonlocal in effect, it influences all points within a horizon of the crack line, and the width of that influence zone scales with the local horizon rather than with the crack opening.

Returns
-------
np.ndarray, a (B, 2) int64 array of [owner, neighbour] index pairs in owner-major, neighbour-ascending order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_directed_bonds(lattice: np.ndarray, crack_y: float,
                         crack_x_max: float) -> np.ndarray:
    '''Enumerate directed peridynamic bonds, omitting those severed by the pre-crack.

    Parameters
    ----------
    lattice : np.ndarray
        (N, 5) float array with columns [x, y, spacing, horizon, volume].
    crack_y : float
        Ordinate of the horizontal pre-crack line.
    crack_x_max : float
        Largest abscissa at which the pre-crack severs a bond.

    Returns
    -------
    bonds : np.ndarray
        (B, 2) integer array with columns [owner, neighbour], ordered by owner
        ascending and then by neighbour ascending.
    '''
    bonds = np.zeros((0, 2), dtype=np.int64)
    return bonds  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_directed_bonds(lattice: np.ndarray, crack_y: float,
                                 crack_x_max: float) -> np.ndarray:
    """Reference implementation."""
    lat = np.asarray(lattice, dtype=float)
    if lat.ndim != 2 or lat.shape[1] != 5 or lat.shape[0] < 2:
        raise ValueError("lattice must have shape (N, 5) with N >= 2")
    if not np.all(np.isfinite(lat)):
        raise ValueError("lattice must be finite")
    if np.any(lat[:, 3] <= 0.0):
        raise ValueError("horizons (column 3) must be > 0")
    if np.any(lat[:, 4] <= 0.0):
        raise ValueError("volumes (column 4) must be > 0")
    if not (np.isfinite(crack_y) and np.isfinite(crack_x_max)):
        raise ValueError("crack_y and crack_x_max must be finite")

    X = lat[:, 0]
    Y = lat[:, 1]
    horizon = lat[:, 3]
    N = lat.shape[0]

    owners = []
    nbrs = []
    for i in range(N):
        r = np.hypot(X - X[i], Y - Y[i])
        if np.count_nonzero(r == 0.0) > 1:
            raise ValueError("lattice contains coincident points")
        js = np.where((r > 0.0) & (r <= horizon[i] * (1.0 + 1e-12)))[0]
        for j in js:
            if (Y[i] - crack_y) * (Y[j] - crack_y) < 0.0:
                t = (crack_y - Y[i]) / (Y[j] - Y[i])
                if X[i] + t * (X[j] - X[i]) <= crack_x_max:
                    continue
            owners.append(i)
            nbrs.append(j)
    return np.asarray([owners, nbrs], dtype=np.int64).T.reshape(-1, 2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: production two-region lattice with the pre-crack ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
lattice = _mk([[0.0, 0.02], [0.02, 0.04]], [0.002, 0.001], 0.02, 3.015)
crack_y = 0.0101
crack_x_max = 0.0151
""",
            "call": "build_directed_bonds(lattice, crack_y, crack_x_max)",
            "gold_call": "_oracle_build_directed_bonds(lattice, crack_y, crack_x_max)",
        },
        # --- Boundary: uniform lattice, crack line outside the domain (no bond severed) ---
        {
            "setup": """import numpy as np
rows = []
for ix in range(4):
    for iy in range(4):
        rows.append(((ix + 0.5) * 1.0, (iy + 0.5) * 1.0, 1.0, 1.5, 1.0))
lattice = np.asarray(rows, dtype=float)
crack_y = -5.0
crack_x_max = 10.0
""",
            "call": "build_directed_bonds(lattice, crack_y, crack_x_max)",
            "gold_call": "_oracle_build_directed_bonds(lattice, crack_y, crack_x_max)",
        },
        # --- Edge: partial crack terminating between lattice columns ---
        {
            "setup": """import numpy as np
rows = []
for ix in range(4):
    for iy in range(4):
        rows.append(((ix + 0.5) * 1.0, (iy + 0.5) * 1.0, 1.0, 1.5, 1.0))
lattice = np.asarray(rows, dtype=float)
crack_y = 2.05
crack_x_max = 1.6
""",
            "call": "build_directed_bonds(lattice, crack_y, crack_x_max)",
            "gold_call": "_oracle_build_directed_bonds(lattice, crack_y, crack_x_max)",
        },
        # --- Invalid: coincident points ---
        {
            "setup": """import numpy as np
rows = []
for ix in range(3):
    for iy in range(3):
        rows.append(((ix + 0.5) * 0.5, (iy + 0.5) * 0.5, 0.5, 1.2, 0.25))
rows.append((0.75, 0.75, 0.5, 1.2, 0.25))
lattice = np.asarray(rows, dtype=float)
def run_model():
    try:
        build_directed_bonds(lattice, 10.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_directed_bonds(lattice, 10.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: lattice with four columns instead of five ---
        {
            "setup": """import numpy as np
lattice = np.array([[0.0, 0.0, 1.0, 1.5], [1.0, 0.0, 1.0, 1.5]], dtype=float)
def run_model():
    try:
        build_directed_bonds(lattice, 10.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_directed_bonds(lattice, 10.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive horizon ---
        {
            "setup": """import numpy as np
lattice = np.array([[0.0, 0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.5, 1.0]], dtype=float)
def run_model():
    try:
        build_directed_bonds(lattice, 10.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_directed_bonds(lattice, 10.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
