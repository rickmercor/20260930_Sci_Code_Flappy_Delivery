"""
Populate a rectangular design domain with a uniform lattice of material points

and return their reference coordinates together with their reference volumes.

In the material point method the continuum is represented by Lagrangian points

that carry mass, volume, deformation gradient and stress, while a fixed Eulerian

grid is used only to solve the force balance. The points also act as the

quadrature points of the internal virtual work integral, so the reference volume

V_p^0 assigned to each point is the quadrature weight of the region it

represents. Filling a domain of side lengths (L_x, L_y) with a lattice of

spacing d places one point at the centre of every lattice cell, so that a point

of index (i, j) sits at ((i + 1/2) d, (j + 1/2) d) and carries V_p^0 = d^2 for a

body of unit thickness. The lattice must tile the domain exactly, otherwise the

quadrature weights no longer sum to the domain volume.

Returns
-------
tuple (points, volumes) of np.ndarray with shapes (n_points, 2) and (n_points,), both float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def initialize_material_points(domain: np.ndarray, point_spacing: float):
    """Lay out material points on a uniform lattice filling a rectangular domain.

    Parameters
    ----------
    domain : np.ndarray
        Side lengths (L_x, L_y) of the design domain.
    point_spacing : float
        Lattice spacing d between neighbouring material points.

    Returns
    -------
    point_state : tuple of np.ndarray
        The pair (points, volumes), holding the reference coordinates of shape
        (n_points, 2) ordered with the x index varying fastest, and the
        reference volume of each material point, of shape (n_points,).

    Raises
    ------
    ValueError
        If `domain` does not hold two positive side lengths, if `point_spacing`
        is not positive, or if either side length is not an integer multiple of
        `point_spacing`.
    """
    return point_state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_LATTICE_TOLERANCE = 1e-9


def _oracle_initialize_material_points(domain: np.ndarray, point_spacing: float):
    """Reference implementation."""
    domain = np.asarray(domain, dtype=float)
    if domain.shape != (2,) or np.any(domain <= 0.0):
        raise ValueError("domain must hold two positive side lengths")
    spacing = float(point_spacing)
    if spacing <= 0.0:
        raise ValueError("point_spacing must be > 0")
    counts = domain / spacing
    if np.any(np.abs(counts - np.rint(counts)) > _LATTICE_TOLERANCE):
        raise ValueError("domain must be an integer multiple of point_spacing")
    n_x = round(float(counts[0]))
    n_y = round(float(counts[1]))
    xs = (np.arange(n_x) + 0.5) * spacing
    ys = (np.arange(n_y) + 0.5) * spacing
    points = np.stack([np.tile(xs, n_y), np.repeat(ys, n_x)], axis=1)
    volumes = np.full(n_x * n_y, spacing * spacing)
    return points, volumes

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: the lattice used by the problem configuration ---
        {
            "setup": """import numpy as np
domain = np.array([2.0, 1.0])
point_spacing = 0.2
""",
            "call": "[np.round(initialize_material_points(domain, point_spacing)[0], 10).tolist(),"
                    " np.round(initialize_material_points(domain, point_spacing)[1], 10).tolist()]",
            "gold_call": "[np.round(_oracle_initialize_material_points(domain, point_spacing)[0], 10).tolist(),"
                         " np.round(_oracle_initialize_material_points(domain, point_spacing)[1], 10).tolist()]",
        },
        # --- Boundary case: the smallest admissible lattice, a single point ---
        {
            "setup": """import numpy as np
domain = np.array([0.2, 0.2])
point_spacing = 0.2
""",
            "call": "[np.round(initialize_material_points(domain, point_spacing)[0], 10).tolist(),"
                    " np.round(initialize_material_points(domain, point_spacing)[1], 10).tolist()]",
            "gold_call": "[np.round(_oracle_initialize_material_points(domain, point_spacing)[0], 10).tolist(),"
                         " np.round(_oracle_initialize_material_points(domain, point_spacing)[1], 10).tolist()]",
        },
        # --- Edge case: a spacing that does not tile the domain must raise ---
        {
            "setup": """import numpy as np
domain = np.array([2.0, 1.0])
def run_model():
    try:
        initialize_material_points(domain, 0.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_initialize_material_points(domain, 0.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
