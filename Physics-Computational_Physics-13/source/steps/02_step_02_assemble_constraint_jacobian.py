"""
Assemble the constraint Jacobian, the off-diagonal block of the saddle-point system, whose block-sparse pattern mirrors the topology of the mechanism.

The holonomic bilateral constraints of a mechanism are collected in a vector-valued function of the generalised coordinates, and its derivative with respect to those coordinates is the constraint Jacobian. In the linearised saddle-point system this Jacobian occupies the off-diagonal blocks, appearing transposed above the diagonal and untransposed below it, while the lower-right block is exactly zero because the Lagrange multipliers enter the equations of motion linearly and are not themselves differentiated. That structural zero is what makes the whole coefficient matrix indefinite rather than positive definite, and it is the reason the multiplier variables cannot be preconditioned directly and must instead be reached through a Schur complement.

The essential structural property of the Jacobian is locality. A kinematic constraint ties exactly two bodies together, so the corresponding row is nonzero only in the columns belonging to those two bodies. Reading the rows of the Jacobian therefore recovers the kinematic constraint graph: the vertices are bodies and the edges are joints. This is the property that the preconditioner of the later steps exploits, since two constraints interact through the Schur complement only when they share a body.

Each joint in the testbed is a spherical joint contributing three scalar constraints. A spherical joint coincides the attachment points of its two bodies, so each of its three rows differences one translational coordinate of the two bodies and, because the attachment point is offset from each body's reference frame, also picks up rotational coordinates through a moment arm. The moment-arm entries are what make the Jacobian rows of a joint linearly independent of one another and of neighbouring joints, so that the Jacobian has full row rank and the exact Schur complement is nonsingular. In the testbed those arms vary with the joint index through a fixed trigonometric law, which keeps the whole matrix reproducible while still exercising the general case in which rotational and translational coordinates are coupled.

Returns
-------
np.ndarray of shape (3 * n_edges, 6 * n_bodies), float: the block-sparse constraint Jacobian, three rows per spherical joint.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_constraint_jacobian(n_bodies: int, n_chords: int) -> np.ndarray:
    """Assemble the constraint Jacobian of the multibody testbed.

    Parameters
    ----------
    n_bodies : int
        Number of rigid bodies in the mechanism (n_bodies >= 4).
    n_chords : int
        Number of long-range chords added to the closed chain (n_chords >= 0).

    Returns
    -------
    jacobian : np.ndarray
        Full-row-rank matrix of shape (3 * n_edges, 6 * n_bodies), where
        n_edges is the number of joints of the kinematic graph. Rows are
        ordered by joint and then by the three scalar constraints of that
        joint.
    """
    return jacobian  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_assemble_constraint_jacobian(n_bodies: int, n_chords: int) -> np.ndarray:
    # Local imports and nested helpers keep the oracle self-contained when the
    # harness executes it in isolation.
    import numpy as np

    def _kinematic_edges(n_bodies, n_chords):
        """Return the edge list of the kinematic constraint graph."""
        edges = [tuple(sorted((b, (b + 1) % n_bodies))) for b in range(n_bodies)]
        seen = set(edges)
        stride = max(1, n_bodies // max(1, n_chords))
        for c in range(n_chords):
            i = (c * stride) % n_bodies
            j = (i + n_bodies // 3 + c) % n_bodies
            if i == j:
                continue
            edge = tuple(sorted((i, j)))
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)
        return edges

    if not (isinstance(n_bodies, (int, np.integer)) and not isinstance(n_bodies, bool)
            and n_bodies >= 4):
        raise ValueError("n_bodies must be an integer >= 4")
    if not (isinstance(n_chords, (int, np.integer)) and not isinstance(n_chords, bool)
            and n_chords >= 0):
        raise ValueError("n_chords must be an integer >= 0")

    n_bodies = int(n_bodies)
    n_chords = int(n_chords)

    edges = _kinematic_edges(n_bodies, n_chords)
    n_edges = len(edges)
    n_dof = 6 * n_bodies
    jacobian = np.zeros((3 * n_edges, n_dof), dtype=float)

    for k, (i, j) in enumerate(edges):
        phi = 2.0 * np.pi * (k + 1) / n_edges
        for d in range(3):
            row = 3 * k + d
            # Coincidence of the two attachment points along direction d.
            jacobian[row, 6 * i + d] = 1.0
            jacobian[row, 6 * j + d] = -1.0
            # Moment arms coupling the rotational coordinates of both bodies.
            jacobian[row, 6 * i + 3 + (d + 1) % 3] = np.sin(phi + d)
            jacobian[row, 6 * j + 3 + (d + 2) % 3] = -np.cos(phi + d)

    return jacobian

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark mechanism (normal scenario) ---
        {
            "setup": """import numpy as np
n_bodies = 24
n_chords = 4
""",
            "call": "assemble_constraint_jacobian(n_bodies, n_chords)",
            "gold_call": "_oracle_assemble_constraint_jacobian(n_bodies, n_chords)",
        },
        # --- Valid: smaller mechanism with more chords ---
        {
            "setup": """import numpy as np
n_bodies = 12
n_chords = 6
""",
            "call": "assemble_constraint_jacobian(n_bodies, n_chords)",
            "gold_call": "_oracle_assemble_constraint_jacobian(n_bodies, n_chords)",
        },
        # --- Boundary: no chords, so the graph carries a single cycle ---
        {
            "setup": """import numpy as np
n_bodies = 8
n_chords = 0
""",
            "call": "assemble_constraint_jacobian(n_bodies, n_chords)",
            "gold_call": "_oracle_assemble_constraint_jacobian(n_bodies, n_chords)",
        },
        # --- Edge: smallest admissible mechanism, chords collapsing onto chain edges ---
        {
            "setup": """import numpy as np
n_bodies = 4
n_chords = 3
""",
            "call": "assemble_constraint_jacobian(n_bodies, n_chords)",
            "gold_call": "_oracle_assemble_constraint_jacobian(n_bodies, n_chords)",
        },
        # --- Invalid: mechanism too small to close a chain ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        assemble_constraint_jacobian(3, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_constraint_jacobian(3, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative chord count ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        assemble_constraint_jacobian(8, -1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_constraint_jacobian(8, -1)
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
