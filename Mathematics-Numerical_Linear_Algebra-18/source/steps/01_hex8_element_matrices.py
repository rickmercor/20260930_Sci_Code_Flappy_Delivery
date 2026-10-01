"""
Form the stiffness and consistent mass matrices of a single eight-node hexahedral element of an isotropic linearly elastic solid.

node_coords is an (8, 3) array of nodal coordinates. The first four rows traverse the face at the lower value of the third natural coordinate and the last four traverse the opposite face in the same rotational sense. E, nu and rho are Young's modulus, Poisson's ratio and density.

Degrees of freedom are ordered node by node and, within a node, as the three translations in ascending coordinate direction, so entry 3*a + d belongs to direction d of local node a. Both matrices are integrated by two-point Gauss quadrature in each natural coordinate, that is eight integration points, applied to the trilinear shape functions.

The function returns one float array of shape (2, 24, 24); plane 0 is the stiffness matrix and plane 1 is the consistent mass matrix.

Raises ValueError if node_coords does not have shape (8, 3); if E is not strictly positive; if rho is not strictly positive; if nu does not lie strictly between -1.0 and 0.5; or if the Jacobian determinant is not strictly positive at any integration point.

A finite element discretisation replaces a continuous body by a union of simple subdomains over which the displacement field is interpolated from nodal values. The eight-node hexahedron with trilinear interpolation is the workhorse of three-dimensional structural analysis: it is the lowest-order element that reproduces a general linear displacement field exactly, and its element matrices are cheap enough that large models remain tractable.

Two matrices characterise the element. The stiffness matrix arises from the strain energy of the interpolated field and depends on the elasticity of the material and the geometry of the element. The mass matrix arises from the kinetic energy. When the same interpolation is used for both, the mass matrix is called consistent; it is not diagonal, and it differs from the lumped alternative in which mass is concentrated at the nodes. The choice matters because the two produce different discrete spectra, the consistent form bounding frequencies from above.

Both matrices are evaluated by numerical quadrature over the natural coordinates of a reference cube, the mapping to physical coordinates entering through the Jacobian. Two-point Gauss quadrature in each direction integrates the trilinear mass terms exactly and is the standard full-integration rule for this element. It is known to overestimate stiffness in bending when elements are thin, a behaviour usually described as shear locking, which is a property of the element and the rule rather than an error in their implementation.

Returns
-------
np.ndarray of shape (2, 24, 24), dtype float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hex8_element_matrices(node_coords: 'np.ndarray', E: float, nu: float,
                          rho: float) -> 'np.ndarray':
    '''Stiffness and consistent mass matrix of one eight-node hexahedral element.

    Parameters
    ----------
    node_coords : np.ndarray
        (8, 3) float array of nodal coordinates in the element's own ordering.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    rho : float
        Mass density.

    Returns
    -------
    np.ndarray
        (2, 24, 24) float array; plane 0 is stiffness, plane 1 is consistent mass.
    '''
    return element_matrices  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_hex8_element_matrices(node_coords: 'np.ndarray', E: float, nu: float,
                                  rho: float) -> 'np.ndarray':
    import numpy as np
    CORNERS = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.shape != (8, 3):
        raise ValueError("node_coords must have shape (8, 3)")
    if not E > 0:
        raise ValueError("E must be strictly positive")
    if not rho > 0:
        raise ValueError("rho must be strictly positive")
    if not -1.0 < nu < 0.5:
        raise ValueError("nu must lie strictly between -1.0 and 0.5")
    sgn = np.array([[-1.0 if c == 0 else 1.0 for c in cc] for cc in CORNERS])
    lam = E*nu/((1.0+nu)*(1.0-2.0*nu))
    mu = E/(2.0*(1.0+nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0*mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    Ke = np.zeros((24, 24))
    Me = np.zeros((24, 24))
    g = 1.0/np.sqrt(3.0)
    for xi in (-g, g):
        for eta in (-g, g):
            for ze in (-g, g):
                N = 0.125*(1+sgn[:, 0]*xi)*(1+sgn[:, 1]*eta)*(1+sgn[:, 2]*ze)
                dN = np.empty((8, 3))
                dN[:, 0] = 0.125*sgn[:, 0]*(1+sgn[:, 1]*eta)*(1+sgn[:, 2]*ze)
                dN[:, 1] = 0.125*(1+sgn[:, 0]*xi)*sgn[:, 1]*(1+sgn[:, 2]*ze)
                dN[:, 2] = 0.125*(1+sgn[:, 0]*xi)*(1+sgn[:, 1]*eta)*sgn[:, 2]
                J = dN.T @ node_coords
                detJ = np.linalg.det(J)
                if not detJ > 0:
                    raise ValueError("non-positive Jacobian determinant at an integration point")
                dNx = np.linalg.solve(J, dN.T).T
                B = np.zeros((6, 24))
                Nm = np.zeros((3, 24))
                for a in range(8):
                    B[0, 3*a+0] = dNx[a, 0]
                    B[1, 3*a+1] = dNx[a, 1]
                    B[2, 3*a+2] = dNx[a, 2]
                    B[3, 3*a+0] = dNx[a, 1]; B[3, 3*a+1] = dNx[a, 0]
                    B[4, 3*a+1] = dNx[a, 2]; B[4, 3*a+2] = dNx[a, 1]
                    B[5, 3*a+0] = dNx[a, 2]; B[5, 3*a+2] = dNx[a, 0]
                    Nm[0, 3*a+0] = N[a]; Nm[1, 3*a+1] = N[a]; Nm[2, 3*a+2] = N[a]
                Ke += (B.T @ D @ B)*detJ
                Me += rho*(Nm.T @ Nm)*detJ
    return np.stack([Ke, Me])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- normal: the production element, steel ---
        {
            "setup": """import numpy as np
C = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
def box(a, b, c):
    return np.array([[q[0]*a, q[1]*b, q[2]*c] for q in C], dtype=float)
xe = box(1.0/24, 0.05, 0.01)
""",
            "call": "hex8_element_matrices(xe, 210e9, 0.3, 7850.0)",
            "gold_call": "_oracle_hex8_element_matrices(xe, 210e9, 0.3, 7850.0)",
        },
        # --- boundary: unit cube ---
        {
            "setup": """import numpy as np
C = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
def box(a, b, c):
    return np.array([[q[0]*a, q[1]*b, q[2]*c] for q in C], dtype=float)
xe = box(1.0, 1.0, 1.0)
""",
            "call": "hex8_element_matrices(xe, 210e9, 0.3, 7850.0)",
            "gold_call": "_oracle_hex8_element_matrices(xe, 210e9, 0.3, 7850.0)",
        },
        # --- edge: aspect ratio 1e4 ---
        {
            "setup": """import numpy as np
C = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
def box(a, b, c):
    return np.array([[q[0]*a, q[1]*b, q[2]*c] for q in C], dtype=float)
xe = box(1.0, 1.0, 1.0e-4)
""",
            "call": "hex8_element_matrices(xe, 210e9, 0.3, 7850.0)",
            "gold_call": "_oracle_hex8_element_matrices(xe, 210e9, 0.3, 7850.0)",
        },
        # --- edge: near-incompressible ---
        {
            "setup": """import numpy as np
C = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
def box(a, b, c):
    return np.array([[q[0]*a, q[1]*b, q[2]*c] for q in C], dtype=float)
xe = box(1.0/24, 0.05, 0.01)
""",
            "call": "hex8_element_matrices(xe, 210e9, 0.4999, 7850.0)",
            "gold_call": "_oracle_hex8_element_matrices(xe, 210e9, 0.4999, 7850.0)",
        },
        # --- invalid: seven nodes ---
        {
            "setup": """import numpy as np
C = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
def box(a, b, c):
    return np.array([[q[0]*a, q[1]*b, q[2]*c] for q in C], dtype=float)
xe = box(1.0, 1.0, 1.0)[:7]
def run_model():
    try:
        hex8_element_matrices(xe, 210e9, 0.3, 7850.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_hex8_element_matrices(xe, 210e9, 0.3, 7850.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: Poisson ratio at the incompressible limit ---
        {
            "setup": """import numpy as np
C = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
def box(a, b, c):
    return np.array([[q[0]*a, q[1]*b, q[2]*c] for q in C], dtype=float)
xe = box(1.0, 1.0, 1.0)
def run_model():
    try:
        hex8_element_matrices(xe, 210e9, 0.5, 7850.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_hex8_element_matrices(xe, 210e9, 0.5, 7850.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: collapsed element, zero Jacobian ---
        {
            "setup": """import numpy as np
C = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
def box(a, b, c):
    return np.array([[q[0]*a, q[1]*b, q[2]*c] for q in C], dtype=float)
xe = box(1.0, 1.0, 0.0)
def run_model():
    try:
        hex8_element_matrices(xe, 210e9, 0.3, 7850.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_hex8_element_matrices(xe, 210e9, 0.3, 7850.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: negative density ---
        {
            "setup": """import numpy as np
C = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
def box(a, b, c):
    return np.array([[q[0]*a, q[1]*b, q[2]*c] for q in C], dtype=float)
xe = box(1.0, 1.0, 1.0)
def run_model():
    try:
        hex8_element_matrices(xe, 210e9, 0.3, -7850.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_hex8_element_matrices(xe, 210e9, 0.3, -7850.0)
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
