"""
Construct the cell-centred material-point lattice for a horizontally partitioned rectangular plate whose regions may use different uniform grid spacings.

The domain is the rectangle [x_bounds[0,0], x_bounds[-1,1]] x [0, height]. It is divided into R contiguous vertical regions; region r spans x in [x_bounds[r,0], x_bounds[r,1]] and is discretised with the uniform spacing spacings[r] in both coordinate directions. Within a region, points sit at the centres of square cells of side spacings[r], so the point at cell index (ix, iy) has coordinates (x_lo + (ix + 0.5) * d, (iy + 0.5) * d) with d = spacings[r]. Each point carries its local spacing d, its horizon horizon_ratio * d, and its voxel volume d ** 2 (per unit out-of-plane thickness).

Points are emitted in a fixed order: regions in the order given by x_bounds, then within each region ix ascending as the outer loop and iy ascending as the inner loop. Every downstream step indexes points by their position in this ordering, so the ordering is part of the contract.

The function returns one float array with five columns [x, y, spacing, horizon, volume], one row per point.

Raises ValueError if: x_bounds does not have shape (R, 2) with R >= 1; spacings does not have shape (R,) matching x_bounds; x_bounds or spacings contain non-finite values; any region has x_hi <= x_lo; consecutive regions are not contiguous in x within an absolute tolerance of 1e-12; any spacing is not strictly positive; height is not finite and strictly positive; horizon_ratio is not finite and strictly positive; any region width is not a positive integer multiple of that region's spacing within a relative tolerance of 1e-12; height is not a positive integer multiple of every region's spacing within a relative tolerance of 1e-12.

Peridynamic discretisations replace the continuum body by a finite set of material points, each identified with the centre of mass of a small voxel of the reference configuration. Integrals over a point's neighbourhood become sums weighted by these voxel volumes, so the volume attached to each point is a quadrature weight rather than a cosmetic attribute, and it appears explicitly in every internal-force and energy sum.

The horizon delta sets the range of nonlocal interaction. It is conventionally chosen proportional to the local point spacing, delta = m * Delta with m a fixed ratio, because the accuracy of the nonlocal quadrature depends on how many points fall inside a neighbourhood rather than on the horizon's absolute size. A consequence is that local mesh refinement automatically produces a spatially varying horizon field: halving the spacing in a region halves the horizon there. Ratios slightly above an integer, such as m = 3.015, are used so that points sitting exactly at a lattice distance of 3 * Delta fall unambiguously inside the neighbourhood rather than on its boundary, which removes the sensitivity of family membership to floating-point comparison at the horizon.

Refining only where resolution is needed, such as along an expected crack path, keeps the point count and the internal-force cost far below those of a globally fine discretisation. The price is that the discretisation is no longer translation invariant across the interface between regions, and neighbourhood relations there become asymmetric, which the force formulation must subsequently account for.

Returns
-------
np.ndarray, an (N, 5) float array of per-point [x, y, spacing, horizon, volume] in the region/ix/iy ordering
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_point_lattice(x_bounds: np.ndarray, spacings: np.ndarray,
                        height: float, horizon_ratio: float) -> np.ndarray:
    '''Build the cell-centred material-point lattice of a multi-region plate.

    Parameters
    ----------
    x_bounds : np.ndarray
        (R, 2) float array; row r gives the x-interval [x_lo, x_hi] of region r.
        Regions must be contiguous and ordered by increasing x.
    spacings : np.ndarray
        (R,) float array of uniform grid spacings, one per region.
    height : float
        Extent of the plate in the y-direction, spanning [0, height].
    horizon_ratio : float
        Ratio m in delta = m * spacing, applied within each region.

    Returns
    -------
    lattice : np.ndarray
        (N, 5) float array with columns [x, y, spacing, horizon, volume],
        ordered by region, then by ix ascending, then by iy ascending.
    '''
    lattice = np.zeros((0, 5), dtype=float)
    return lattice  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_point_lattice(x_bounds: np.ndarray, spacings: np.ndarray,
                                height: float, horizon_ratio: float) -> np.ndarray:
    """Reference implementation."""
    xb = np.asarray(x_bounds, dtype=float)
    sp = np.asarray(spacings, dtype=float)
    if xb.ndim != 2 or xb.shape[1] != 2 or xb.shape[0] < 1:
        raise ValueError("x_bounds must have shape (R, 2) with R >= 1")
    if sp.ndim != 1 or sp.shape[0] != xb.shape[0]:
        raise ValueError("spacings must have shape (R,) matching x_bounds")
    if not np.all(np.isfinite(xb)) or not np.all(np.isfinite(sp)):
        raise ValueError("x_bounds and spacings must be finite")
    if np.any(xb[:, 1] <= xb[:, 0]):
        raise ValueError("each region must satisfy x_lo < x_hi")
    if xb.shape[0] > 1 and not np.allclose(xb[1:, 0], xb[:-1, 1], rtol=0.0, atol=1e-12):
        raise ValueError("regions must be contiguous in x")
    if np.any(sp <= 0.0):
        raise ValueError("spacings must be > 0")
    if not (np.isfinite(height) and height > 0.0):
        raise ValueError("height must be finite and > 0")
    if not (np.isfinite(horizon_ratio) and horizon_ratio > 0.0):
        raise ValueError("horizon_ratio must be finite and > 0")

    rows = []
    for r in range(xb.shape[0]):
        d = sp[r]
        lo = xb[r, 0]
        width = xb[r, 1] - xb[r, 0]
        nx = int(round(width / d))
        ny = int(round(height / d))
        if nx < 1 or abs(nx * d - width) > 1e-12 * width:
            raise ValueError("region width must be a positive integer multiple of its spacing")
        if ny < 1 or abs(ny * d - height) > 1e-12 * height:
            raise ValueError("height must be a positive integer multiple of every spacing")
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d,
                             (iy + 0.5) * d,
                             d,
                             horizon_ratio * d,
                             d * d))
    return np.asarray(rows, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the two-region production lattice (500 points) ---
        {
            "setup": """import numpy as np
x_bounds = np.array([[0.0, 0.02], [0.02, 0.04]], dtype=float)
spacings = np.array([0.002, 0.001], dtype=float)
height = 0.02
horizon_ratio = 3.015
""",
            "call": "build_point_lattice(x_bounds, spacings, height, horizon_ratio)",
            "gold_call": "_oracle_build_point_lattice(x_bounds, spacings, height, horizon_ratio)",
        },
        # --- Boundary: single region, smallest non-degenerate lattice (2 x 2) ---
        {
            "setup": """import numpy as np
x_bounds = np.array([[0.0, 1.0]], dtype=float)
spacings = np.array([0.5], dtype=float)
height = 1.0
horizon_ratio = 2.0
""",
            "call": "build_point_lattice(x_bounds, spacings, height, horizon_ratio)",
            "gold_call": "_oracle_build_point_lattice(x_bounds, spacings, height, horizon_ratio)",
        },
        # --- Edge: three regions, coarse-fine-coarse, exercises ordering across refinement ---
        {
            "setup": """import numpy as np
x_bounds = np.array([[0.0, 0.25], [0.25, 0.75], [0.75, 1.0]], dtype=float)
spacings = np.array([0.25, 0.125, 0.25], dtype=float)
height = 0.25
horizon_ratio = 1.5
""",
            "call": "build_point_lattice(x_bounds, spacings, height, horizon_ratio)",
            "gold_call": "_oracle_build_point_lattice(x_bounds, spacings, height, horizon_ratio)",
        },
        # --- Invalid: non-contiguous regions ---
        {
            "setup": """import numpy as np
x_bounds = np.array([[0.0, 0.02], [0.03, 0.04]], dtype=float)
spacings = np.array([0.002, 0.001], dtype=float)
def run_model():
    try:
        build_point_lattice(x_bounds, spacings, 0.02, 3.015)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_point_lattice(x_bounds, spacings, 0.02, 3.015)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: region width not an integer multiple of the spacing ---
        {
            "setup": """import numpy as np
x_bounds = np.array([[0.0, 0.02]], dtype=float)
spacings = np.array([0.003], dtype=float)
def run_model():
    try:
        build_point_lattice(x_bounds, spacings, 0.02, 3.015)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_point_lattice(x_bounds, spacings, 0.02, 3.015)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive horizon ratio ---
        {
            "setup": """import numpy as np
x_bounds = np.array([[0.0, 0.02]], dtype=float)
spacings = np.array([0.002], dtype=float)
def run_model():
    try:
        build_point_lattice(x_bounds, spacings, 0.02, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_point_lattice(x_bounds, spacings, 0.02, 0.0)
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
