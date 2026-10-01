"""
Assemble the effective stiffness matrix, the (1,1) block of the saddle-point system produced by the Newton-Raphson linearisation of the index-3 multibody equations of motion.

Implicit integration of the index-3 differential-algebraic equations of constrained multibody dynamics, followed by Newton-Raphson linearisation, produces at every time step a symmetric indefinite saddle-point system whose (1,1) block is the effective stiffness matrix. That tangent operator superposes two physically distinct contributions. The first is the inertial term, which enters as the generalised mass matrix divided by beta * h ** 2, where h is the integration step and beta is the Newmark parameter of the implicit integrator; because each body's mass matrix is independent of every other body, this contribution is block diagonal, one 6 x 6 block per body, holding three translational masses and three principal moments of inertia. The second is the elastic term, which couples bodies that share a joint and therefore inherits the sparsity pattern of the kinematic constraint graph: each joint contributes a stiffness that penalises the relative motion of its two bodies, adding its stiffness to both diagonal blocks and subtracting it from the two off-diagonal blocks, exactly as an edge contributes to a weighted graph Laplacian.

The relative weight of the two contributions is what governs the conditioning of the whole calculation. As the time step is reduced the inertial term grows as one over h ** 2 and comes to dominate, so the spectral condition number of the effective stiffness behaves as h ** (-2); simultaneously, a realistic mechanism carries joint stiffnesses spread over many orders of magnitude, from compliant bushings to effectively rigid welds, and that coefficient variation adds its own ill-conditioning on top. The testbed used here reproduces both effects deterministically: bodies are indexed cyclically in mass, and the joint stiffness varies log-periodically with the joint index over six orders of magnitude. The resulting operator is symmetric positive definite, which is what allows the multigrid hierarchy of the later steps to be built on it directly rather than on a positive-definite surrogate.

The kinematic graph itself is a closed chain of bodies plus a small number of chords. The chords are what make the graph cycle-rich rather than tree-like, and the number of independent cycles they create is what later determines the size of the dense low-rank correction in the constraint preconditioner. An open chain would make that correction empty and would hide the part of the method that handles kinematic loops.

Returns
-------
np.ndarray of shape (6 * n_bodies, 6 * n_bodies), float: the symmetric positive-definite effective stiffness matrix in SI units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_effective_stiffness(n_bodies: int, n_chords: int, h: float,
                                 beta: float, k_scale: float) -> np.ndarray:
    """Assemble the effective stiffness matrix of the multibody testbed.

    Parameters
    ----------
    n_bodies : int
        Number of rigid bodies in the mechanism (n_bodies >= 4).
    n_chords : int
        Number of long-range chords added to the closed chain (n_chords >= 0).
    h : float
        Integration time step in seconds (h > 0).
    beta : float
        Newmark parameter of the implicit integrator (0 < beta <= 1).
    k_scale : float
        Reference joint stiffness in N/m (k_scale > 0).

    Returns
    -------
    stiffness : np.ndarray
        Symmetric positive-definite matrix of shape (6 * n_bodies,
        6 * n_bodies) holding the inertial and elastic contributions to the
        tangent operator.
    """
    return stiffness  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_assemble_effective_stiffness(n_bodies: int, n_chords: int, h: float,
                                         beta: float, k_scale: float) -> np.ndarray:
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
    if not (isinstance(h, (int, float)) and np.isfinite(h) and float(h) > 0.0):
        raise ValueError("h must be a finite number > 0")
    if not (isinstance(beta, (int, float)) and np.isfinite(beta)
            and 0.0 < float(beta) <= 1.0):
        raise ValueError("beta must be a finite number in the half-open interval (0, 1]")
    if not (isinstance(k_scale, (int, float)) and np.isfinite(k_scale)
            and float(k_scale) > 0.0):
        raise ValueError("k_scale must be a finite number > 0")

    n_bodies = int(n_bodies)
    n_chords = int(n_chords)
    h = float(h)
    beta = float(beta)
    k_scale = float(k_scale)

    edges = _kinematic_edges(n_bodies, n_chords)
    n_dof = 6 * n_bodies
    stiffness = np.zeros((n_dof, n_dof), dtype=float)

    # Inertial term: block-diagonal generalised mass scaled by 1 / (beta * h^2).
    for b in range(n_bodies):
        mass = 1.0 + 0.5 * (b % 5)
        block = np.array([mass, mass, mass,
                          0.100 * mass, 0.125 * mass, 0.150 * mass], dtype=float)
        rows = np.arange(6 * b, 6 * b + 6)
        stiffness[rows, rows] += block / (beta * h * h)

    # Elastic term: one graph-Laplacian contribution per joint, with the joint
    # stiffness spread log-periodically over six orders of magnitude.
    n_edges = len(edges)
    for k, (i, j) in enumerate(edges):
        k_edge = k_scale * 10.0 ** (3.0 * np.cos(2.0 * np.pi * k / n_edges))
        for d in range(6):
            ii, jj = 6 * i + d, 6 * j + d
            stiffness[ii, ii] += k_edge
            stiffness[jj, jj] += k_edge
            stiffness[ii, jj] -= k_edge
            stiffness[jj, ii] -= k_edge

    return stiffness

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
h = 1.0e-3
beta = 0.25
k_scale = 1.0e8
""",
            "call": "assemble_effective_stiffness(n_bodies, n_chords, h, beta, k_scale)",
            "gold_call": "_oracle_assemble_effective_stiffness(n_bodies, n_chords, h, beta, k_scale)",
        },
        # --- Valid: smaller mechanism with more chords ---
        {
            "setup": """import numpy as np
n_bodies = 12
n_chords = 6
h = 5.0e-4
beta = 0.3
k_scale = 5.0e7
""",
            "call": "assemble_effective_stiffness(n_bodies, n_chords, h, beta, k_scale)",
            "gold_call": "_oracle_assemble_effective_stiffness(n_bodies, n_chords, h, beta, k_scale)",
        },
        # --- Boundary: no chords, so the graph is a single closed loop ---
        {
            "setup": """import numpy as np
n_bodies = 8
n_chords = 0
h = 1.0e-3
beta = 0.25
k_scale = 1.0e8
""",
            "call": "assemble_effective_stiffness(n_bodies, n_chords, h, beta, k_scale)",
            "gold_call": "_oracle_assemble_effective_stiffness(n_bodies, n_chords, h, beta, k_scale)",
        },
        # --- Edge: very small step, so the inertial term dominates completely ---
        {
            "setup": """import numpy as np
n_bodies = 4
n_chords = 2
h = 1.0e-5
beta = 1.0
k_scale = 1.0e3
""",
            "call": "assemble_effective_stiffness(n_bodies, n_chords, h, beta, k_scale)",
            "gold_call": "_oracle_assemble_effective_stiffness(n_bodies, n_chords, h, beta, k_scale)",
        },
        # --- Invalid: mechanism too small to close a chain ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        assemble_effective_stiffness(3, 1, 1.0e-3, 0.25, 1.0e8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_effective_stiffness(3, 1, 1.0e-3, 0.25, 1.0e8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: Newmark parameter outside its admissible interval ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        assemble_effective_stiffness(8, 2, 1.0e-3, 0.0, 1.0e8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_effective_stiffness(8, 2, 1.0e-3, 0.0, 1.0e8)
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
