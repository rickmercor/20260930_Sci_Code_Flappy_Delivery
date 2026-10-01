"""
Determine the magnetic block minimizer and its total directional derivative for the supplied quadratic and state jets.

Use the Lagrangian with positive multiplier pairing against $p-\varphi(r)$ and identity-metric quadratic penalty of strength $\zeta$.

In this block only, replace compatibility by its first-order affine expansion at the old magnetic coefficients, freeze the supplied auxiliary and multiplier values during minimization, and eliminate boundary magnetic coefficients.

Differentiate the resulting minimizer with respect to the supplied jet family, including the moving expansion point, rather than freezing the affine model when taking the derivative.

For a unit magnetic vector $n$, let $R(n)=(2nn^{\mathsf T}-I)e_1$, let $J$ be counterclockwise quarter-turn rotation, and let $P$ normalize a nonzero vector.

The exact compatibility is the continuous scalar branch through zero satisfying

$$

R(P(n+rJn))=P(R(n)+\varphi(r)JR(n)),\qquad |r|<1.

$$

Differentiating a linearized minimization problem involves changes in both its matrix and right-hand side. The moving expansion point contributes through curvature of the nonlinear compatibility relation.

Returns
-------
A dimensionless floating-point array of shape (2, N) containing the new magnetic coefficients and their total directional derivatives, with boundary entries zero in both layers.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_magnetic_jet(
    systems: np.ndarray, boundary: np.ndarray, state: np.ndarray, zeta: float
) -> np.ndarray:
    r"""Evaluate the stated numerical value and directional-derivative contract.

    Parameters
    ----------
    systems : np.ndarray
        Finite quadratic jet, shape (2, 2, N, N + 1); first axis is
        value/derivative, second is magnetic/nematic, last column is gradient offset
        and preceding columns are Hessian; both Hessian layers are symmetric and
        only the base layer must be positive semidefinite, checked to absolute
        1e-12.
    boundary : np.ndarray
        Boolean mask of shape (N,); True fixes a magnetic increment at zero in both
        jet layers, while prescribed direction derivatives may remain nonzero.
    state : np.ndarray
        Finite state jet, shape (2, 3, N), with rows r, p, multiplier within each
        layer; base r lies in (-1, 1), and boundary entries of both r layers are
        zero.
    zeta : float
        Positive finite augmentation parameter in the identity metric, constant with
        respect to alpha.

    Returns
    -------
    result : np.ndarray
        A dimensionless floating-point array of shape (2, N) containing the new
        magnetic coefficients and their total directional derivatives, with boundary
        entries zero in both layers.

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


def _mask(boundary, size):
    boundary = np.asarray(boundary)
    if boundary.dtype != bool or boundary.shape != (size,):
        raise ValueError("boundary must be a Boolean vector of length N")
    return boundary


def _systems(systems):
    systems = _array(systems, 4, "systems")
    if systems.shape[:2] != (2, 2) or systems.shape[2] < 1:
        raise ValueError("systems must have shape (2, 2, N, N + 1)")
    size = systems.shape[2]
    if systems.shape[3] != size + 1:
        raise ValueError("systems must have shape (2, 2, N, N + 1)")
    for layer in range(2):
        for block in systems[layer, :, :, :size]:
            if not np.allclose(block, block.T, rtol=0, atol=1e-12):
                raise ValueError("Hessians and their derivatives must be symmetric")
            if layer == 0 and np.linalg.eigvalsh(block)[0] < -1e-12:
                raise ValueError("base Hessians must be positive semidefinite")
    return systems


def _state(state, size):
    state = _array(state, 3, "state")
    if state.shape != (2, 3, size):
        raise ValueError("state must have shape (2, 3, N)")
    if np.any(np.abs(state[0, 0]) >= 1):
        raise ValueError("base magnetic coefficients must lie in (-1, 1)")
    return state


def _phi(r):
    return 2 * r / (1 - r * r)


def _dphi(r):
    return 2 * (1 + r * r) / (1 - r * r) ** 2


def _ddphi(r):
    return 4 * r * (3 + r * r) / (1 - r * r) ** 3


def _linear_jet(A, dA, b, db):
    try:
        q, upper = np.linalg.qr(A)
        x = np.linalg.solve(upper, q.T @ b)
        dx = np.linalg.solve(upper, q.T @ (db - dA @ x))
    except np.linalg.LinAlgError as exc:
        raise ValueError("linear system is singular") from exc
    result = np.stack((x, dx))
    if not np.all(np.isfinite(result)):
        raise ValueError("linear solve produced a nonfinite jet")
    return result


def _oracle_solve_magnetic_jet(
    systems: np.ndarray, boundary: np.ndarray, state: np.ndarray, zeta: float
) -> np.ndarray:
    systems = _systems(systems)
    size = systems.shape[2]
    boundary = _mask(boundary, size)
    state = _state(state, size)
    if np.any(state[:, 0, boundary] != 0):
        raise ValueError("both magnetic boundary layers must be zero")
    zeta = _positive(zeta, "zeta")
    r, p, lam = state[0]
    dr, dp, dlam = state[1]
    a = _dphi(r)
    da = _ddphi(r) * dr
    b = _phi(r) - a * r
    db = -da * r
    A = systems[0, 0, :, :size] + zeta * np.diag(a * a)
    dA = systems[1, 0, :, :size] + 2 * zeta * np.diag(a * da)
    rhs = -systems[0, 0, :, size] + a * lam + zeta * a * (p - b)
    drhs = -systems[1, 0, :, size] + da * lam + a * dlam
    drhs += zeta * (da * (p - b) + a * (dp - db))
    free = np.flatnonzero(~boundary)
    result = np.zeros((2, size))
    if free.size:
        result[:, free] = _linear_jet(
            A[np.ix_(free, free)], dA[np.ix_(free, free)], rhs[free], drhs[free]
        )
    if np.any(np.abs(result[0]) >= 1):
        raise ValueError("magnetic iterate left (-1, 1)")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic numerical cases for the stated contract."""
    return [
        {
            "setup": """import numpy as np
systems = np.array([
    [[[2.,-.4,.2],[-.4,1.,-.3]],[[1.5,.25,-.1],[.25,2.,.5]]],
    [[[.2,.05,-.12],[.05,-.1,.23]],[[-.1,.07,.15],[.07,.3,-.2]]]])
boundary = np.array([True,False])
state = np.array([[[0.,.12],[.03,-.07],[.11,-.08]],
                  [[0.,-.13],[.2,.1],[-.08,.04]]])
r_jet = state[:,0].copy()
multiplier_jet = state[:,2].copy()
""",
            "call": "solve_magnetic_jet(systems, boundary, state, 4.)",
            "gold_call": "_oracle_solve_magnetic_jet(systems, boundary, state, 4.)",
        },
        {
            "setup": """import numpy as np
systems = np.array([
    [[[2.,-.4,.2],[-.4,1.,-.3]],[[1.5,.25,-.1],[.25,2.,.5]]],
    [[[.2,.05,-.12],[.05,-.1,.23]],[[-.1,.07,.15],[.07,.3,-.2]]]])
boundary = np.array([True,False])
state = np.array([[[0.,.12],[.03,-.07],[.11,-.08]],
                  [[0.,-.13],[.2,.1],[-.08,.04]]])
r_jet = state[:,0].copy()
multiplier_jet = state[:,2].copy()
state[1]=0.
systems[1]=0.
""",
            "call": "solve_magnetic_jet(systems, boundary, state, 4.)",
            "gold_call": "_oracle_solve_magnetic_jet(systems, boundary, state, 4.)",
        },
        {
            "setup": """import numpy as np
systems = np.array([
    [[[2.,-.4,.2],[-.4,1.,-.3]],[[1.5,.25,-.1],[.25,2.,.5]]],
    [[[.2,.05,-.12],[.05,-.1,.23]],[[-.1,.07,.15],[.07,.3,-.2]]]])
boundary = np.array([True,False])
state = np.array([[[0.,.12],[.03,-.07],[.11,-.08]],
                  [[0.,-.13],[.2,.1],[-.08,.04]]])
r_jet = state[:,0].copy()
multiplier_jet = state[:,2].copy()
state[0,0,1]=.85
state[1,0,1]=.3
""",
            "call": "solve_magnetic_jet(systems, boundary, state, 4.)",
            "gold_call": "_oracle_solve_magnetic_jet(systems, boundary, state, 4.)",
        },
        {
            "setup": """import numpy as np
systems = np.array([
    [[[2.,-.4,.2],[-.4,1.,-.3]],[[1.5,.25,-.1],[.25,2.,.5]]],
    [[[.2,.05,-.12],[.05,-.1,.23]],[[-.1,.07,.15],[.07,.3,-.2]]]])
boundary = np.array([True,False])
state = np.array([[[0.,.12],[.03,-.07],[.11,-.08]],
                  [[0.,-.13],[.2,.1],[-.08,.04]]])
r_jet = state[:,0].copy()
multiplier_jet = state[:,2].copy()
state[1,0,0]=1.

def run_model():
    try:
        solve_magnetic_jet(systems, boundary, state, 4.)
        return 0.0
    except ValueError:
        return 1.0
def run_gold():
    try:
        _oracle_solve_magnetic_jet(systems, boundary, state, 4.)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
