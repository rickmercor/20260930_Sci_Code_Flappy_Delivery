"""
Construct the value and directional variation of the nodal geometry for the family of angles $\theta+\alpha v$ on a fixed triangular mesh.

Use the continuous affine nodal basis and its dimensionless Dirichlet bilinear form without boundary elimination or lumping.

A jet stores a value in layer 0 and its ordinary first derivative with respect to the dimensionless parameter $\alpha$ in layer 1, without factorial scaling.

The geometry array has shape $(2,N,N+8)$: the first $N$ columns hold the fixed stiffness matrix and its zero derivative; the remaining pairs hold $n,t,\nu,\tau$ and their derivatives, with counterclockwise tangents and $\nu=(2nn^{\mathsf T}-I)e_1$.

*The mesh is fixed while the orientations vary. The tensor representation and the magnetic vector have different angular responses, and their tangent frames must be differentiated consistently.*

Returns
-------
A dimensionless floating-point array of shape (2, N, N + 8) containing the stiffness and coupled nodal frames in layer zero and their directional derivatives in layer one.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def prepare_geometry_jet(
    vertices: np.ndarray,
    triangles: np.ndarray,
    angles: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    r"""Evaluate the stated numerical value and directional-derivative contract.

    Parameters
    ----------
    vertices : np.ndarray
        Finite dimensionless planar coordinates, shape (N, 2), N >= 3.
    triangles : np.ndarray
        Integer connectivity, shape (M, 3), M >= 1, using every vertex; each
        triangle has distinct zero-based indices and nonzero area, with either
        orientation.
    angles : np.ndarray
        Finite initial angles in radians, shape (N,), in vertex order.
    direction : np.ndarray
        Finite angle derivatives with respect to alpha, shape (N,), in vertex order.

    Returns
    -------
    result : np.ndarray
        A dimensionless floating-point array of shape (2, N, N + 8) containing the
        stiffness and coupled nodal frames in layer zero and their directional
        derivatives in layer one.

    Raises
    ------
    ValueError
        If an input violates its shape, finiteness, domain, or stated convention.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _array(value, ndim, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != ndim or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite rank-{ndim} array")
    return value


def _geometry(geometry):
    geometry = _array(geometry, 3, "geometry")
    if geometry.shape[0] != 2 or geometry.shape[1] < 3:
        raise ValueError("geometry must have shape (2, N, N + 8), N >= 3")
    size = geometry.shape[1]
    if geometry.shape[2] != size + 8:
        raise ValueError("geometry must have shape (2, N, N + 8)")
    K = geometry[0, :, :size]
    if not np.allclose(K, K.T, rtol=0, atol=1e-12):
        raise ValueError("stiffness must be symmetric")
    if not np.allclose(K.sum(axis=1), 0, rtol=0, atol=1e-12):
        raise ValueError("stiffness must have zero row sums")
    if np.any(K - np.diag(np.diag(K)) > 1e-12):
        raise ValueError("stiffness off-diagonals must be nonpositive")
    if np.any(geometry[1, :, :size] != 0):
        raise ValueError("the mesh is fixed, so stiffness derivatives must be zero")
    n, dn = geometry[:, :, size : size + 2]
    if not np.allclose(np.sum(n * n, axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("magnetic directions must have unit length")
    if not np.allclose(np.sum(n * dn, axis=1), 0, rtol=0, atol=1e-12):
        raise ValueError("direction derivatives must be tangent")
    expected = _frames(n, dn)
    if not np.allclose(geometry[:, :, size:], expected, rtol=1e-12, atol=1e-12):
        raise ValueError("the supplied frames and their derivatives are inconsistent")
    return geometry


def _frames(n, dn):
    t = np.column_stack((-n[:, 1], n[:, 0]))
    dt = np.column_stack((-dn[:, 1], dn[:, 0]))
    nu = np.column_stack((n[:, 0] ** 2 - n[:, 1] ** 2, 2 * n[:, 0] * n[:, 1]))
    dnu = np.column_stack(
        (
            2 * n[:, 0] * dn[:, 0] - 2 * n[:, 1] * dn[:, 1],
            2 * (dn[:, 0] * n[:, 1] + n[:, 0] * dn[:, 1]),
        )
    )
    tau = np.column_stack((-nu[:, 1], nu[:, 0]))
    dtau = np.column_stack((-dnu[:, 1], dnu[:, 0]))
    return np.stack(
        (np.column_stack((n, t, nu, tau)), np.column_stack((dn, dt, dnu, dtau)))
    )


def _oracle_prepare_geometry_jet(
    vertices: np.ndarray,
    triangles: np.ndarray,
    angles: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    vertices = _array(vertices, 2, "vertices")
    triangles = np.asarray(triangles)
    if vertices.shape[0] < 3 or vertices.shape[1] != 2:
        raise ValueError("vertices must have shape (N, 2), N >= 3")
    size = len(vertices)
    if (
        triangles.ndim != 2
        or triangles.shape[0] < 1
        or triangles.shape[1] != 3
        or not np.issubdtype(triangles.dtype, np.integer)
    ):
        raise ValueError("triangles must be a nonempty integer array of shape (M, 3)")
    if (
        np.any(triangles < 0)
        or np.any(triangles >= size)
        or len(np.unique(triangles)) != size
    ):
        raise ValueError("connectivity must be in range and use every vertex")
    angles = _array(angles, 1, "angles")
    direction = _array(direction, 1, "direction")
    if angles.shape != (size,) or direction.shape != (size,):
        raise ValueError("angles and direction must have shape (N,)")
    K = np.zeros((size, size))
    for indices in triangles:
        if len(np.unique(indices)) != 3:
            raise ValueError("triangle indices must be distinct")
        xy = vertices[indices]
        area = abs(np.linalg.det(np.column_stack((xy[1] - xy[0], xy[2] - xy[0])))) / 2
        if area == 0 or not np.isfinite(area):
            raise ValueError("triangle area must be finite and positive")
        gradients = np.linalg.inv(np.column_stack((np.ones(3), xy)))[1:, :]
        K[np.ix_(indices, indices)] += area * gradients.T @ gradients
    n = np.column_stack((np.cos(angles), np.sin(angles)))
    dn = direction[:, None] * np.column_stack((-n[:, 1], n[:, 0]))
    result = np.zeros((2, size, size + 8))
    result[0, :, :size] = K
    result[:, :, size:] = _frames(n, dn)
    return _geometry(result)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic numerical cases for the stated contract."""
    return [
        {
            "setup": """import numpy as np
vertices = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.5,.5]])
triangles = np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
boundary = np.array([True,True,True,True,False])
angles = np.array([.10,.75,1.55,2.90,2.35])
direction = np.array([.4,-.1,.25,-.3,.2])
""",
            "call": "prepare_geometry_jet(vertices, triangles, angles, direction)",
            "gold_call": "_oracle_prepare_geometry_jet(vertices, triangles, angles, direction)",
        },
        {
            "setup": """import numpy as np
vertices = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.5,.5]])
triangles = np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
boundary = np.array([True,True,True,True,False])
angles = np.array([.10,.75,1.55,2.90,2.35])
direction = np.array([.4,-.1,.25,-.3,.2])
direction[:]=0.
""",
            "call": "prepare_geometry_jet(vertices, triangles, angles, direction)",
            "gold_call": "_oracle_prepare_geometry_jet(vertices, triangles, angles, direction)",
        },
        {
            "setup": """import numpy as np
vertices = np.array([[i/3,j/3] for j in range(4) for i in range(4)])
triangles = np.array([t for j in range(3) for i in range(3)
    for t in [(4*j+i,4*j+i+1,4*j+i+5),(4*j+i,4*j+i+5,4*j+i+4)]])
boundary = np.array([i in (0,3) or j in (0,3) for j in range(4) for i in range(4)])
angles = .1+.8*vertices[:,0]+.6*vertices[:,1]
angles[~boundary] += np.array([1.6,-1.3,1.1,-1.5])
direction = np.where(boundary,np.cos(np.pi*vertices[:,0])*np.sin(np.pi*(vertices[:,1]+.25)),0.)
""",
            "call": "prepare_geometry_jet(vertices, triangles, angles, direction)",
            "gold_call": "_oracle_prepare_geometry_jet(vertices, triangles, angles, direction)",
        },
        {
            "setup": """import numpy as np
vertices = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.5,.5]])
triangles = np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
boundary = np.array([True,True,True,True,False])
angles = np.array([.10,.75,1.55,2.90,2.35])
direction = np.array([.4,-.1,.25,-.3,.2])
vertices[4]=vertices[0]

def run_model():
    try:
        prepare_geometry_jet(vertices, triangles, angles, direction)
        return 0.0
    except ValueError:
        return 1.0
def run_gold():
    try:
        _oracle_prepare_geometry_jet(vertices, triangles, angles, direction)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
