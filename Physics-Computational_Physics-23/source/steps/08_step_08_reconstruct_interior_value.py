"""
Close the representation formula at one interior point by combining the assembled boundary quadrature with the assembled domain quadrature.

For a source point strictly inside the layer the free coefficient of the representation formula is unity, so the temperature there follows from the two boundary pairings of the fundamental solution and of its normal derivative together with the domain term that carries the transient and source contributions. The normal derivative of the temperature at a boundary quadrature point is the projection of its gradient on the outward unit normal.

Returns
-------
float: the temperature at the interior source point implied by the representation formula, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reconstruct_interior_value(boundary_terms: "np.ndarray", boundary_fields: "np.ndarray",
                               domain_terms: "np.ndarray", domain_values: "np.ndarray") -> float:
    """Evaluate the representation formula at one interior point.

    Parameters
    ----------
    boundary_terms : np.ndarray
        Float array of shape ``(m, 6)`` gathered over every boundary element of
        the layer, whose columns are the x and y coordinates of the boundary
        quadrature points, the weight that multiplies the normal derivative of
        the temperature there, the weight that multiplies the temperature
        there, and the two components of the outward unit normal.
    boundary_fields : np.ndarray
        Float array of shape ``(m, 3)`` holding the temperature and its two
        spatial derivatives at the same boundary quadrature points.
    domain_terms : np.ndarray
        Float array of shape ``(q, 3)`` gathered over every angular sector of
        the layer, whose columns are the x and y coordinates of the domain
        quadrature points and the weight of the fundamental solution there.
    domain_values : np.ndarray
        Float array of shape ``(q,)`` holding the density of the domain
        integral at the same domain quadrature points.

    Returns
    -------
    value : float
        Temperature at the interior source point implied by the representation
        formula, as a native Python float.

    Raises
    ------
    ValueError
        If ``boundary_terms`` is not a finite real array of shape ``(m, 6)``
        with ``m >= 1``, if ``boundary_fields`` is not a finite real array of
        shape ``(m, 3)``, if ``domain_terms`` is not a finite real array of
        shape ``(q, 3)`` with ``q >= 1``, or if ``domain_values`` is not a
        finite real array of shape ``(q,)``.

    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_reconstruct_interior_value(boundary_terms: "np.ndarray", boundary_fields: "np.ndarray",
                                       domain_terms: "np.ndarray", domain_values: "np.ndarray") -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    terms = np.asarray(boundary_terms, dtype=float)
    fields = np.asarray(boundary_fields, dtype=float)
    sectors = np.asarray(domain_terms, dtype=float)
    density = np.asarray(domain_values, dtype=float)

    if terms.ndim != 2 or terms.shape[0] < 1 or terms.shape[1] != 6:
        raise ValueError("boundary_terms must have shape (m, 6) with m >= 1")
    if fields.shape != (terms.shape[0], 3):
        raise ValueError("boundary_fields must have shape (m, 3)")
    if sectors.ndim != 2 or sectors.shape[0] < 1 or sectors.shape[1] != 3:
        raise ValueError("domain_terms must have shape (q, 3) with q >= 1")
    if density.shape != (sectors.shape[0],):
        raise ValueError("domain_values must have shape (q,)")
    for name, block in (("boundary_terms", terms), ("boundary_fields", fields),
                        ("domain_terms", sectors), ("domain_values", density)):
        if not np.all(np.isfinite(block)):
            raise ValueError(f"{name} must be finite")

    # Normal derivative of the temperature at each boundary quadrature point.
    flux = fields[:, 1] * terms[:, 4] + fields[:, 2] * terms[:, 5]

    boundary = float(np.sum(terms[:, 2] * flux - terms[:, 3] * fields[:, 0]))
    domain = float(np.sum(sectors[:, 2] * density))

    return float(boundary - domain)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned: a hand built pair of quadrature tables whose combination
        # is a two line closed form, evaluated without the oracle.
        {
            "setup": """import numpy as np
bt = np.array([[0.0, 0.0, 2.0, 3.0, 1.0, 0.0],
               [1.0, 0.0, -1.0, 0.5, 0.0, -1.0]])
bf = np.array([[5.0, 7.0, 11.0],
               [2.0, -4.0, 6.0]])
dt = np.array([[0.0, 0.0, 0.25], [1.0, 1.0, -0.5]])
dv = np.array([8.0, 3.0])
""",
            "call": "reconstruct_interior_value(bt.copy(), bf.copy(), dt.copy(), dv.copy())",
            "gold_call": "_oracle_reconstruct_interior_value(bt.copy(), bf.copy(), dt.copy(), dv.copy())",
        },
        # --- Pinned: a vanishing domain density leaves only the boundary sum,
        # so the routine must return exactly the boundary contribution.
        {
            "setup": """import numpy as np
rng = np.random.default_rng(5)
bt = rng.normal(size=(6, 6))
bf = rng.normal(size=(6, 3))
dt = rng.normal(size=(4, 3))
dv = np.zeros(4)
""",
            "call": "reconstruct_interior_value(bt.copy(), bf.copy(), dt.copy(), dv.copy())",
            "gold_call": "_oracle_reconstruct_interior_value(bt.copy(), bf.copy(), dt.copy(), dv.copy())",
        },
        # --- Valid: tables of the size the coating produces ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(31)
bt = rng.normal(size=(120, 6))
bf = rng.normal(size=(120, 3)) + 3.0
dt = rng.normal(size=(300, 3))
dv = rng.normal(size=300)
""",
            "call": "float(1.0 + 1.0e6 * reconstruct_interior_value(bt.copy(), bf.copy(), dt.copy(), dv.copy()))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_reconstruct_interior_value(bt.copy(), bf.copy(), dt.copy(), dv.copy()))",
        },
        # --- Valid: a boundary contribution far larger than the domain one,
        # which is the balance an ultra-thin layer produces ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(77)
bt = rng.normal(size=(48, 6))
bf = rng.normal(size=(48, 3)) + 2.0
dt = 1.0e-6 * rng.normal(size=(64, 3))
dv = rng.normal(size=64)
""",
            "call": "float(1.0 + 1.0e6 * reconstruct_interior_value(bt.copy(), bf.copy(), dt.copy(), dv.copy()))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_reconstruct_interior_value(bt.copy(), bf.copy(), dt.copy(), dv.copy()))",
        },
        # --- Boundary: one boundary point and one domain point ---
        {
            "setup": """import numpy as np
bt = np.array([[0.5, -0.5, 1.25, -0.75, 0.6, 0.8]])
bf = np.array([[1.5, -2.5, 0.5]])
dt = np.array([[0.1, 0.2, 0.9]])
dv = np.array([-1.25])
""",
            "call": "float(1.0 + 1.0e6 * reconstruct_interior_value(bt.copy(), bf.copy(), dt.copy(), dv.copy()))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_reconstruct_interior_value(bt.copy(), bf.copy(), dt.copy(), dv.copy()))",
        },
        # --- Edge: an outward normal aligned with the y axis only, so that the
        # x derivative of the field cannot enter the flux ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(101)
bt = rng.normal(size=(10, 6))
bt[:, 4] = 0.0
bt[:, 5] = -1.0
bf = rng.normal(size=(10, 3))
dt = rng.normal(size=(5, 3))
dv = rng.normal(size=5)
""",
            "call": "float(1.0 + 1.0e6 * reconstruct_interior_value(bt.copy(), bf.copy(), dt.copy(), dv.copy()))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_reconstruct_interior_value(bt.copy(), bf.copy(), dt.copy(), dv.copy()))",
        },
        # --- Invalid: mismatched boundary field rows ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        reconstruct_interior_value(np.zeros((4, 6)), np.zeros((3, 3)), np.zeros((2, 3)), np.zeros(2))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reconstruct_interior_value(np.zeros((4, 6)), np.zeros((3, 3)), np.zeros((2, 3)), np.zeros(2))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a boundary table with the wrong number of columns ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        reconstruct_interior_value(np.zeros((4, 5)), np.zeros((4, 3)), np.zeros((2, 3)), np.zeros(2))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reconstruct_interior_value(np.zeros((4, 5)), np.zeros((4, 3)), np.zeros((2, 3)), np.zeros(2))
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
