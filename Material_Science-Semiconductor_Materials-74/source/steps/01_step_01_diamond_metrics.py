"""
Compute the area and the directional coupling factor of one interior diamond cell.

A diamond cell is the quadrilateral spanned by the two primal cell centres and the two dual cell centres that share an interface. It is fixed by the primal edge length |sigma|, the dual edge length |sigma*|, the angle theta between the primal and dual directions, and the two outward unit normals n_KL and n_K**L**.

|D| = 0.5 * |sigma| * |sigma*| * sin(theta)

c   = n_KL . n_K*L*

The scalar c is the factor that multiplies every cross-direction term of the flux discretization; it vanishes exactly when the two normals are orthogonal.

A diamond cell is the quadrilateral spanned by the two primal cell centres and the two dual cell centres that share an interface. Two scalars summarise it. Its area sets the common prefactor of every flux built on the cell, and the projection of the two outward unit normals onto each other measures how far the cell is from being orthogonal. That projection is the switch that turns the transverse coupling on: it is zero exactly when the two directions are perpendicular, and the whole point of the discretization is that they need not be.

Returns
-------
`numpy.ndarray` of shape `(2,)` containing `[diamond_area, normal_projection]` as native floats.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_diamond_metrics(
    primal_edge_length: float,
    dual_edge_length: float,
    angle_radians: float,
    primal_normal: np.ndarray,
    dual_normal: np.ndarray,
) -> np.ndarray:
    """Return the diamond-cell area and the primal-dual normal projection.

    Parameters
    ----------
    primal_edge_length : float
        Length |sigma| of the primal edge. Must be finite and > 0.
    dual_edge_length : float
        Length |sigma*| of the dual edge. Must be finite and > 0.
    angle_radians : float
        Angle theta between the primal and dual directions, in radians.
        Must be finite and must give a strictly positive diamond area.
    primal_normal : np.ndarray
        Outward unit normal n_KL of the primal edge, shape (2,), finite.
    dual_normal : np.ndarray
        Outward unit normal n_K*L* of the dual edge, shape (2,), finite.

    Returns
    -------
    metrics : np.ndarray
        Array of shape (2,) holding [diamond_area, normal_projection], where
        diamond_area = 0.5 * |sigma| * |sigma*| * sin(theta) and
        normal_projection = n_KL . n_K*L*.

    Raises
    ------
    ValueError
        If primal_edge_length or dual_edge_length is not finite or is <= 0,
        if angle_radians is not finite, if primal_normal or dual_normal is not
        a finite array of shape (2,), or if the resulting diamond area is
        <= 0.
    """
    return metrics  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_diamond_metrics(
    primal_edge_length: float,
    dual_edge_length: float,
    angle_radians: float,
    primal_normal: np.ndarray,
    dual_normal: np.ndarray,
) -> np.ndarray:
    if not np.isfinite(primal_edge_length) or primal_edge_length <= 0.0:
        raise ValueError("primal_edge_length must be finite and > 0")
    if not np.isfinite(dual_edge_length) or dual_edge_length <= 0.0:
        raise ValueError("dual_edge_length must be finite and > 0")
    if not np.isfinite(angle_radians):
        raise ValueError("angle_radians must be finite")

    primal = np.asarray(primal_normal, dtype=float)
    dual = np.asarray(dual_normal, dtype=float)
    if primal.shape != (2,) or dual.shape != (2,):
        raise ValueError("Both normals must be arrays of shape (2,)")
    if not np.all(np.isfinite(primal)) or not np.all(np.isfinite(dual)):
        raise ValueError("Both normals must be finite")

    area = 0.5 * float(primal_edge_length) * float(dual_edge_length) * np.sin(angle_radians)
    if area <= 0.0:
        raise ValueError("angle_radians must define a strictly positive diamond area")

    coupling = float(np.dot(primal, dual))
    return np.array([float(area), coupling], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: the non-orthogonal diamond of the task (theta = pi/3)
        {
            "setup": "import numpy as np\np = np.array([1.0, 0.0])\nd = np.array([0.5, np.sqrt(3) / 2])\n",
            "call": "compute_diamond_metrics(1.6, 2.1, np.pi / 3, p, d)",
            "gold_call": "_oracle_compute_diamond_metrics(1.6, 2.1, np.pi / 3, p, d)",
        },
        # Boundary: orthogonal normals, theta = pi/2 (coupling exactly zero)
        {
            "setup": "import numpy as np\np = np.array([1.0, 0.0])\nd = np.array([0.0, 1.0])\n",
            "call": "compute_diamond_metrics(2.0, 3.0, np.pi / 2, p, d)",
            "gold_call": "_oracle_compute_diamond_metrics(2.0, 3.0, np.pi / 2, p, d)",
        },
        # Edge: strongly distorted diamond, theta = pi/60, normals not aligned with theta
        {
            "setup": "import numpy as np\np = np.array([0.0, 1.0])\nd = np.array([0.6, 0.8])\n",
            "call": "compute_diamond_metrics(0.75, 1.25, np.pi / 60, p, d)",
            "gold_call": "_oracle_compute_diamond_metrics(0.75, 1.25, np.pi / 60, p, d)",
        },
        # Invalid: non-positive primal edge length
        {
            "setup": """import numpy as np
p = np.array([1.0, 0.0])
d = np.array([0.5, np.sqrt(3) / 2])
def run_model():
    try:
        compute_diamond_metrics(0.0, 2.1, np.pi / 3, p, d)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_diamond_metrics(0.0, 2.1, np.pi / 3, p, d)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Invalid: degenerate angle giving zero area
        {
            "setup": """import numpy as np
p = np.array([1.0, 0.0])
d = np.array([0.5, np.sqrt(3) / 2])
def run_model():
    try:
        compute_diamond_metrics(1.6, 2.1, 0.0, p, d)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_diamond_metrics(1.6, 2.1, 0.0, p, d)
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
