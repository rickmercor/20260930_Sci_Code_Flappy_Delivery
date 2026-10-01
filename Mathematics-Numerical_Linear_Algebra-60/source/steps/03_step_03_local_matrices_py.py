"""
Compute the local Mini finite-element matrices on one physical triangle. The local displacement basis consists of the three P1 vertex basis functions and one normalized cubic bubble, for each displacement component. The pressure basis consists of the three P1 vertex functions.

The Mini element enriches the continuous piecewise-linear displacement space by a cubic element bubble. The local bilinear forms therefore require integration of products involving P1 and cubic basis functions. The paper defines the bubble space $B_3$ and the associated Mini discretization.

Returns
-------
dict[str, np.ndarray], containing A_local, B_local, C_local, and M_local
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def local_matrices(
    triangle_vertices: np.ndarray,
    mu: float,
    lam: float,
) -> dict[str, np.ndarray]:
    """
    Assemble the local Mini-element matrices for one triangle.

    The scalar Mini basis is ordered as

        [l1, l2, l3, phi_b],

    where l1, l2, and l3 are the barycentric P1 basis functions associated
    with triangle_vertices[0], triangle_vertices[1], and triangle_vertices[2],
    respectively, and the normalized cubic bubble is

        phi_b = 27 * l1 * l2 * l3.

    The local displacement space has 8 degrees of freedom: the four scalar
    Mini basis functions for the x component followed by the same four basis
    functions for the y component. Thus the local displacement ordering is

        [x_l1, x_l2, x_l3, x_b, y_l1, y_l2, y_l3, y_b].

    The pressure space uses the three scalar P1 basis functions in the
    vertex order [l1, l2, l3].

    Parameters
    ----------
    triangle_vertices : np.ndarray
        Coordinates of one triangle, shape (3, 2). Row 0, row 1, and row 2
        define the vertex order associated with l1, l2, and l3.
    mu : float
        Positive shear modulus.
    lam : float
        Positive Lame parameter.

    Returns
    -------
    matrices : dict[str, np.ndarray]
        Dictionary containing the local matrices for the Mini mixed element.
        The displacement-related matrices use the 8-DOF ordering described
        above, while the pressure-related matrix uses the 3-DOF ordering
        [l1, l2, l3].

        The returned dictionary contains the local displacement stiffness,
        displacement mass, divergence coupling, and pressure matrix.

    Raises
    ------
    ValueError
        If triangle_vertices does not have shape (3, 2), if the triangle is
        degenerate, or if mu <= 0 or lam <= 0.
    """
    return matrices

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_local_matrices(
    triangle_vertices: np.ndarray,
    mu: float,
    lam: float,
) -> dict[str, np.ndarray]:
    """Reference implementation."""
    x = np.asarray(triangle_vertices, dtype=float)

    if x.shape != (3, 2):
        raise ValueError("triangle_vertices must have shape (3,2)")

    if not np.all(np.isfinite(x)):
        raise ValueError("triangle coordinates must be finite")

    if not np.isscalar(mu) or float(mu) <= 0.0:
        raise ValueError("mu must be positive")

    if not np.isscalar(lam) or float(lam) <= 0.0:
        raise ValueError("lam must be positive")

    p0, p1, p2 = x

    # Physical map:
    # X(r,s) = p0 + J [r,s]^T
    J = np.column_stack((p1 - p0, p2 - p0))
    detJ = float(np.linalg.det(J))

    # Scale-aware degeneracy check.
    # det(J) has units of length^2, so compare it against
    # a tolerance scaled by the squared triangle size.
    edge_scale = max(
        np.linalg.norm(p1 - p0),
        np.linalg.norm(p2 - p0),
        np.linalg.norm(p2 - p1),
    )

    if edge_scale <= 0.0:
        raise ValueError("degenerate triangle")

    if abs(detJ) <= 1e-14 * edge_scale**2:
        raise ValueError("degenerate triangle")

    invJ = np.linalg.inv(J)

    # 7-point Gauss-Legendre rule after Duffy transformation.
    xi, wi = np.polynomial.legendre.leggauss(7)
    nodes = 0.5 * (xi + 1.0)
    weights = 0.5 * wi

    # Four scalar displacement basis functions:
    # P1 vertex functions plus cubic bubble.
    n_scalar_u = 4

    # Three scalar pressure basis functions.
    n_pressure = 3

    A_scalar = np.zeros((n_scalar_u, n_scalar_u), dtype=float)
    M_scalar = np.zeros((n_scalar_u, n_scalar_u), dtype=float)
    B_local = np.zeros((n_pressure, 2 * n_scalar_u), dtype=float)
    C_local = np.zeros((n_pressure, n_pressure), dtype=float)

    def _reference_basis(r: float, s: float):
        l1 = 1.0 - r - s
        l2 = r
        l3 = s

        values = np.array(
            [
                l1,
                l2,
                l3,
                27.0 * l1 * l2 * l3,
            ],
            dtype=float,
        )

        gradients = np.array(
            [
                [-1.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [
                    27.0 * l3 * (l1 - l2),
                    27.0 * l2 * (l1 - l3),
                ],
            ],
            dtype=float,
        )

        return values, gradients

    def _pressure_basis(r: float, s: float):
        return np.array(
            [
                1.0 - r - s,
                r,
                s,
            ],
            dtype=float,
        )

    for a, wa in zip(nodes, weights):
        for b, wb in zip(nodes, weights):

            # Duffy map:
            # r = a
            # s = (1-a)b
            r = float(a)
            s = float((1.0 - a) * b)

            # Jacobian of Duffy transformation.
            duffy_det = 1.0 - a

            # Physical integration factor.
            w = float(
                wa * wb * duffy_det * abs(detJ)
            )

            u_values, grads_ref = _reference_basis(r, s)

            # Row-vector gradient transformation.
            grads_phys = grads_ref @ invJ

            q_values = _pressure_basis(r, s)

            # Scalar P1+bubble displacement stiffness.
            A_scalar += (
                w
                * float(mu)
                * (grads_phys @ grads_phys.T)
            )

            # Scalar displacement mass.
            M_scalar += (
                w
                * np.outer(u_values, u_values)
            )

            # Divergence terms.
            B_local[:, :n_scalar_u] += (
                w
                * np.outer(q_values, grads_phys[:, 0])
            )

            B_local[:, n_scalar_u:] += (
                w
                * np.outer(q_values, grads_phys[:, 1])
            )

            # Pressure bilinear form.
            C_local += (
                w
                / (float(lam) + float(mu))
                * np.outer(q_values, q_values)
            )

    # Vector displacement operator is block diagonal in x/y.
    A_local = np.zeros(
        (2 * n_scalar_u, 2 * n_scalar_u),
        dtype=float,
    )
    A_local[:n_scalar_u, :n_scalar_u] = A_scalar
    A_local[n_scalar_u:, n_scalar_u:] = A_scalar

    M_local = np.zeros_like(A_local)
    M_local[:n_scalar_u, :n_scalar_u] = M_scalar
    M_local[n_scalar_u:, n_scalar_u:] = M_scalar

    return {
        "A_local": A_local,
        "B_local": B_local,
        "C_local": C_local,
        "M_local": M_local,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
tri = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
], dtype=float)

def pack_local(matrices):
    return np.concatenate([
        np.asarray(matrices["A_local"], dtype=float).ravel(),
        np.asarray(matrices["B_local"], dtype=float).ravel(),
        np.asarray(matrices["C_local"], dtype=float).ravel(),
        np.asarray(matrices["M_local"], dtype=float).ravel(),
    ])
""",
            "call": "pack_local(local_matrices(tri, 1.0, 7777.0))",
            "gold_call": "pack_local(_oracle_local_matrices(tri, 1.0, 7777.0))",
        },
        {
            "setup": """import numpy as np
tri = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
    [0.0, 2.0],
], dtype=float)

def pack_local(matrices):
    return np.concatenate([
        np.asarray(matrices["A_local"], dtype=float).ravel(),
        np.asarray(matrices["B_local"], dtype=float).ravel(),
        np.asarray(matrices["C_local"], dtype=float).ravel(),
        np.asarray(matrices["M_local"], dtype=float).ravel(),
    ])
""",
            "call": "pack_local(local_matrices(tri, 1.0, 1.0))",
            "gold_call": "pack_local(_oracle_local_matrices(tri, 1.0, 1.0))",
        },
        {
            "setup": """import numpy as np
tri = np.array([
    [0.0, 0.0],
    [1e-4, 0.0],
    [0.0, 1e-4],
], dtype=float)

def pack_local(matrices):
    return np.concatenate([
        np.asarray(matrices["A_local"], dtype=float).ravel(),
        np.asarray(matrices["B_local"], dtype=float).ravel(),
        np.asarray(matrices["C_local"], dtype=float).ravel(),
        np.asarray(matrices["M_local"], dtype=float).ravel(),
    ])
""",
            "call": "pack_local(local_matrices(tri, 2.0, 100.0))",
            "gold_call": "pack_local(_oracle_local_matrices(tri, 2.0, 100.0))",
        },
        {
            "setup": """import numpy as np
tri = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [2.0, 0.0],
], dtype=float)

def run_model():
    try:
        local_matrices(tri, 1.0, 100.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_local_matrices(tri, 1.0, 100.0)
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
