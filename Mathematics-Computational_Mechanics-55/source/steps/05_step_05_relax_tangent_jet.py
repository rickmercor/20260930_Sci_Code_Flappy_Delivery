"""
Return the terminal inner state and its directional derivative along the executed stopping branch.

Each cycle performs the selectively linearized magnetic minimization, exact nonlinear auxiliary minimization on all nodes, and dual ascent with residual $p-\varphi(r)$, propagating the supplied jets through those operations.

Use the first completed cycle whose base all-node RMS compatibility residual is at most the tolerance; derivatives do not enter the stopping test, and the base cycle count is held fixed when differentiating.

Retain all state rows and both layers for the next outer update.

The preceding magnetic and auxiliary contracts define the minimizations, compatibility branch, and boundary treatment.

The derivative of an executed finite iteration differs from a derivative obtained by substituting its infinite-iteration stationarity equations. A stopping branch is locally fixed for the specified directional calculation.

Returns
-------
A dimensionless floating-point array of shape (2, 3, N) containing the terminal magnetic, auxiliary, and multiplier state and its derivative on the base stopping branch.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def relax_tangent_jet(
    systems: np.ndarray,
    boundary: np.ndarray,
    state: np.ndarray,
    zeta: float = 4.0,
    rho: float = 1.0,
    tolerance: float = 1e-10,
    max_iterations: int = 20000,
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
    rho : float
        Positive finite dual-ascent step, constant with respect to alpha.
    tolerance : float
        Positive finite all-node RMS tolerance applied only to the base
        compatibility residual.
    max_iterations : int
        Positive integer inner-cycle cap; Boolean values are rejected.

    Returns
    -------
    result : np.ndarray
        A dimensionless floating-point array of shape (2, 3, N) containing the
        terminal magnetic, auxiliary, and multiplier state and its derivative on the
        base stopping branch.

    Raises
    ------
    ValueError
        If an input violates its shape, finiteness, domain, or stated convention.
    RuntimeError
        If the inner iteration cap is reached before the base tolerance.
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


def _count(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


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


def _solve_magnetic_jet(
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


def _solve_auxiliary_jet(
    systems: np.ndarray,
    r_jet: np.ndarray,
    multiplier_jet: np.ndarray,
    zeta: float,
    rho: float,
) -> np.ndarray:
    systems = _systems(systems)
    size = systems.shape[2]
    r_jet = _array(r_jet, 2, "r_jet")
    multiplier_jet = _array(multiplier_jet, 2, "multiplier_jet")
    if (
        r_jet.shape != (2, size)
        or multiplier_jet.shape != (2, size)
        or np.any(np.abs(r_jet[0]) >= 1)
    ):
        raise ValueError("vector jets must have shape (2, N) and base r in (-1, 1)")
    zeta, rho = _positive(zeta, "zeta"), _positive(rho, "rho")
    r, dr = r_jet
    lam, dlam = multiplier_jet
    f, df = _phi(r), _dphi(r) * dr
    B = systems[0, 1, :, :size] + zeta * np.eye(size)
    dB = systems[1, 1, :, :size]
    rhs = -systems[0, 1, :, size] - lam + zeta * f
    drhs = -systems[1, 1, :, size] - dlam + zeta * df
    p, dp = _linear_jet(B, dB, rhs, drhs)
    updated = np.stack((p, lam + rho * (p - f)))
    variation = np.stack((dp, dlam + rho * (dp - df)))
    return np.stack((updated, variation))


def _oracle_relax_tangent_jet(
    systems: np.ndarray,
    boundary: np.ndarray,
    state: np.ndarray,
    zeta: float = 4.0,
    rho: float = 1.0,
    tolerance: float = 1e-10,
    max_iterations: int = 20000,
) -> np.ndarray:
    systems = _systems(systems)
    size = systems.shape[2]
    boundary = _mask(boundary, size)
    state = _state(state, size).copy()
    if np.any(state[:, 0, boundary] != 0):
        raise ValueError("both magnetic boundary layers must be zero")
    zeta, rho = _positive(zeta, "zeta"), _positive(rho, "rho")
    tolerance = _positive(tolerance, "tolerance")
    max_iterations = _count(max_iterations, "max_iterations")
    for _ in range(max_iterations):
        r = _solve_magnetic_jet(systems, boundary, state, zeta)
        auxiliary = _solve_auxiliary_jet(systems, r, state[:, 2], zeta, rho)
        state = np.concatenate((r[:, None, :], auxiliary), axis=1)
        residual = np.sqrt(np.mean((state[0, 1] - _phi(state[0, 0])) ** 2))
        if residual <= tolerance:
            return state
    raise RuntimeError("inner cycle cap reached before compatibility tolerance")

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
            "call": "relax_tangent_jet(systems, boundary, state)",
            "gold_call": "_oracle_relax_tangent_jet(systems, boundary, state)",
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
            "call": "relax_tangent_jet(systems, boundary, state)",
            "gold_call": "_oracle_relax_tangent_jet(systems, boundary, state)",
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
boundary[:]=False
""",
            "call": "relax_tangent_jet(systems, boundary, state)",
            "gold_call": "_oracle_relax_tangent_jet(systems, boundary, state)",
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

def run_model():
    try:
        relax_tangent_jet(systems, boundary, state, max_iterations=1)
        return 0.0
    except RuntimeError:
        return 1.0
def run_gold():
    try:
        _oracle_relax_tangent_jet(systems, boundary, state, max_iterations=1)
        return 0.0
    except RuntimeError:
        return 1.0
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
