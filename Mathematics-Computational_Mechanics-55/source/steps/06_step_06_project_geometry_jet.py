"""
Determine the updated geometric jet after nodewise normalization of the magnetic tangent increment.

Differentiate both the increment and its normalization, then reconstruct the nematic representation and both tangent frames.

A zero magnetic increment at a prescribed node does not set that node's direction derivative to zero.

A jet stores a value in layer 0 and its ordinary first derivative with respect to the dimensionless parameter $\alpha$ in layer 1, without factorial scaling.

The geometry array has shape $(2,N,N+8)$: the first $N$ columns hold the fixed stiffness matrix and its zero derivative; the remaining pairs hold $n,t,\nu,\tau$ and their derivatives, with counterclockwise tangents and $\nu=(2nn^{\mathsf T}-I)e_1$.

Differentiating the unit-length constraint removes the radial part of a direction variation. The same operation must be followed by the nonlinear tensor reconstruction, rather than projecting an independently evolved nematic jet.

Returns
-------
A dimensionless floating-point array of shape (2, N, N + 8) containing the projected geometry and its directional derivative with the fixed stiffness block preserved.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def project_geometry_jet(geometry: np.ndarray, r_jet: np.ndarray) -> np.ndarray:
    r"""Evaluate the stated numerical value and directional-derivative contract.

    Parameters
    ----------
    geometry : np.ndarray
        Finite geometric jet, shape (2, N, N + 8), with the layout stated in the
        description; magnetic rows are unit, their derivatives tangent, frames
        consistent, and stiffness has zero row sums and nonpositive off-diagonals,
        checked to absolute 1e-12 (frame comparison also uses relative 1e-12).
    r_jet : np.ndarray
        Finite magnetic coefficient jet, shape (2, N), with base coefficients
        strictly in (-1, 1).

    Returns
    -------
    result : np.ndarray
        A dimensionless floating-point array of shape (2, N, N + 8) containing the
        projected geometry and its directional derivative with the fixed stiffness
        block preserved.

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


def _oracle_project_geometry_jet(geometry: np.ndarray, r_jet: np.ndarray) -> np.ndarray:
    geometry = _geometry(geometry)
    size = geometry.shape[1]
    r_jet = _array(r_jet, 2, "r_jet")
    if r_jet.shape != (2, size) or np.any(np.abs(r_jet[0]) >= 1):
        raise ValueError("r_jet must have shape (2, N) and base r in (-1, 1)")
    n, dn = geometry[:, :, size : size + 2]
    t, dt = geometry[:, :, size + 2 : size + 4]
    r, dr = r_jet
    x = n + r[:, None] * t
    dx = dn + dr[:, None] * t + r[:, None] * dt
    lengths = np.linalg.norm(x, axis=1)
    projected = x / lengths[:, None]
    dprojected = (dx - projected * np.sum(projected * dx, axis=1)[:, None]) / lengths[
        :, None
    ]
    unchanged = (r == 0) & (dr == 0)
    projected[unchanged] = n[unchanged]
    dprojected[unchanged] = dn[unchanged]
    result = geometry.copy()
    result[:, :, size:] = _frames(projected, dprojected)
    return result

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
K = np.array([[1.,0.,0.,0.,-1.],[0.,1.,0.,0.,-1.],
              [0.,0.,1.,0.,-1.],[0.,0.,0.,1.,-1.],[-1.,-1.,-1.,-1.,4.]])
n = np.column_stack((np.cos(angles),np.sin(angles)))
t = np.column_stack((-n[:,1],n[:,0]))
nu = np.column_stack((np.cos(2*angles),np.sin(2*angles)))
tau = np.column_stack((-nu[:,1],nu[:,0]))
geometry = np.zeros((2,5,13))
geometry[0,:,:5] = K
geometry[0,:,5:] = np.column_stack((n,t,nu,tau))
geometry[1,:,5:] = np.column_stack((direction[:,None]*t,-direction[:,None]*n,
                                  2*direction[:,None]*tau,-2*direction[:,None]*nu))
r_jet=np.array([[0.,0.,0.,0.,.2],[0.,0.,0.,0.,-.3]])
""",
            "call": "project_geometry_jet(geometry, r_jet)",
            "gold_call": "_oracle_project_geometry_jet(geometry, r_jet)",
        },
        {
            "setup": """import numpy as np
vertices = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.5,.5]])
triangles = np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
boundary = np.array([True,True,True,True,False])
angles = np.array([.10,.75,1.55,2.90,2.35])
direction = np.array([.4,-.1,.25,-.3,.2])
K = np.array([[1.,0.,0.,0.,-1.],[0.,1.,0.,0.,-1.],
              [0.,0.,1.,0.,-1.],[0.,0.,0.,1.,-1.],[-1.,-1.,-1.,-1.,4.]])
n = np.column_stack((np.cos(angles),np.sin(angles)))
t = np.column_stack((-n[:,1],n[:,0]))
nu = np.column_stack((np.cos(2*angles),np.sin(2*angles)))
tau = np.column_stack((-nu[:,1],nu[:,0]))
geometry = np.zeros((2,5,13))
geometry[0,:,:5] = K
geometry[0,:,5:] = np.column_stack((n,t,nu,tau))
geometry[1,:,5:] = np.column_stack((direction[:,None]*t,-direction[:,None]*n,
                                  2*direction[:,None]*tau,-2*direction[:,None]*nu))
r_jet=np.zeros((2,5))
""",
            "call": "project_geometry_jet(geometry, r_jet)",
            "gold_call": "_oracle_project_geometry_jet(geometry, r_jet)",
        },
        {
            "setup": """import numpy as np
vertices = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.5,.5]])
triangles = np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
boundary = np.array([True,True,True,True,False])
angles = np.array([.10,.75,1.55,2.90,2.35])
direction = np.array([.4,-.1,.25,-.3,.2])
K = np.array([[1.,0.,0.,0.,-1.],[0.,1.,0.,0.,-1.],
              [0.,0.,1.,0.,-1.],[0.,0.,0.,1.,-1.],[-1.,-1.,-1.,-1.,4.]])
n = np.column_stack((np.cos(angles),np.sin(angles)))
t = np.column_stack((-n[:,1],n[:,0]))
nu = np.column_stack((np.cos(2*angles),np.sin(2*angles)))
tau = np.column_stack((-nu[:,1],nu[:,0]))
geometry = np.zeros((2,5,13))
geometry[0,:,:5] = K
geometry[0,:,5:] = np.column_stack((n,t,nu,tau))
geometry[1,:,5:] = np.column_stack((direction[:,None]*t,-direction[:,None]*n,
                                  2*direction[:,None]*tau,-2*direction[:,None]*nu))
r_jet=np.array([[0.,0.,0.,0.,-.9],[.2,-.3,.1,.4,.7]])
""",
            "call": "project_geometry_jet(geometry, r_jet)",
            "gold_call": "_oracle_project_geometry_jet(geometry, r_jet)",
        },
        {
            "setup": """import numpy as np
vertices = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.5,.5]])
triangles = np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
boundary = np.array([True,True,True,True,False])
angles = np.array([.10,.75,1.55,2.90,2.35])
direction = np.array([.4,-.1,.25,-.3,.2])
K = np.array([[1.,0.,0.,0.,-1.],[0.,1.,0.,0.,-1.],
              [0.,0.,1.,0.,-1.],[0.,0.,0.,1.,-1.],[-1.,-1.,-1.,-1.,4.]])
n = np.column_stack((np.cos(angles),np.sin(angles)))
t = np.column_stack((-n[:,1],n[:,0]))
nu = np.column_stack((np.cos(2*angles),np.sin(2*angles)))
tau = np.column_stack((-nu[:,1],nu[:,0]))
geometry = np.zeros((2,5,13))
geometry[0,:,:5] = K
geometry[0,:,5:] = np.column_stack((n,t,nu,tau))
geometry[1,:,5:] = np.column_stack((direction[:,None]*t,-direction[:,None]*n,
                                  2*direction[:,None]*tau,-2*direction[:,None]*nu))
r_jet=np.ones((2,5))

def run_model():
    try:
        project_geometry_jet(geometry, r_jet)
        return 0.0
    except ValueError:
        return 1.0
def run_gold():
    try:
        _oracle_project_geometry_jet(geometry, r_jet)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
