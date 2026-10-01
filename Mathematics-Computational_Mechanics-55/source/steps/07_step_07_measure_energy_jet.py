"""
Determine the physical harmonic energy and its directional derivative on the fixed mesh.

The physical nodal state concatenates the nematic direction scaled by $Q_c$ and magnetic direction scaled by $M_c$.

Use the energy of its four-component affine interpolant with the factor of one half in the Dirichlet integral.

A jet stores a value in layer 0 and its ordinary first derivative with respect to the dimensionless parameter $\alpha$ in layer 1, without factorial scaling.

The geometry array has shape $(2,N,N+8)$: the first $N$ columns hold the fixed stiffness matrix and its zero derivative; the remaining pairs hold $n,t,\nu,\tau$ and their derivatives, with counterclockwise tangents and $\nu=(2nn^{\mathsf T}-I)e_1$.

Differentiation must act on the discrete interpolated field. A continuous chain-rule identity for a constrained vector field cannot replace this discrete energy variation.

Returns
-------
A dimensionless floating-point vector of length two containing the discrete harmonic energy followed by its directional derivative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def measure_energy_jet(geometry: np.ndarray, Qc: float, Mc: float) -> np.ndarray:
    r"""Evaluate the stated numerical value and directional-derivative contract.

    Parameters
    ----------
    geometry : np.ndarray
        Finite geometric jet, shape (2, N, N + 8), with the layout stated in the
        description; magnetic rows are unit, their derivatives tangent, frames
        consistent, and stiffness has zero row sums and nonpositive off-diagonals,
        checked to absolute 1e-12 (frame comparison also uses relative 1e-12).
    Qc : float
        Positive finite dimensionless nematic length, constant with respect to
        alpha.
    Mc : float
        Positive finite dimensionless magnetic length, constant with respect to
        alpha.

    Returns
    -------
    result : np.ndarray
        A dimensionless floating-point vector of length two containing the discrete
        harmonic energy followed by its directional derivative.

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


def _positive(value, name):
    if np.ndim(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


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


def _oracle_measure_energy_jet(
    geometry: np.ndarray, Qc: float, Mc: float
) -> np.ndarray:
    geometry = _geometry(geometry)
    size = geometry.shape[1]
    Qc, Mc = _positive(Qc, "Qc"), _positive(Mc, "Mc")
    K = geometry[0, :, :size]
    n, dn = geometry[:, :, size : size + 2]
    nu, dnu = geometry[:, :, size + 4 : size + 6]
    energy = 0.5 * np.sum(K * (Qc * Qc * (nu @ nu.T) + Mc * Mc * (n @ n.T)))
    variation = np.sum(K * (Qc * Qc * (dnu @ nu.T) + Mc * Mc * (dn @ n.T)))
    result = np.array([energy, variation])
    if not np.all(np.isfinite(result)):
        raise ValueError("energy jet is nonfinite")
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
""",
            "call": "measure_energy_jet(geometry, 1.0013, 1.0025)",
            "gold_call": "_oracle_measure_energy_jet(geometry, 1.0013, 1.0025)",
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
geometry[1]=0.
""",
            "call": "measure_energy_jet(geometry, 1.0013, 1.0025)",
            "gold_call": "_oracle_measure_energy_jet(geometry, 1.0013, 1.0025)",
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
""",
            "call": "measure_energy_jet(geometry, .4, 2.5)",
            "gold_call": "_oracle_measure_energy_jet(geometry, .4, 2.5)",
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

def run_model():
    try:
        measure_energy_jet(geometry, 0., 1.)
        return 0.0
    except ValueError:
        return 1.0
def run_gold():
    try:
        _oracle_measure_energy_jet(geometry, 0., 1.)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
