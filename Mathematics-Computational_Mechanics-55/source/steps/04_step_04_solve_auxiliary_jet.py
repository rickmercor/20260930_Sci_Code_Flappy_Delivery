"""
Determine the full-node auxiliary minimizer, subsequent dual-ascent update, and total directional derivatives.

Use the nematic quadratic from the supplied jet, positive multiplier pairing against $p-\varphi(r)$, and an identity-metric quadratic penalty of strength $\zeta$.

Minimize over auxiliary coefficients at every node at the supplied magnetic state, then advance the multiplier with step $\rho$ evaluated at this minimizer.

Both operations retain exact nonlinear compatibility, including in their directional derivatives.

For a unit magnetic vector $n$, let $R(n)=(2nn^{\mathsf T}-I)e_1$, let $J$ be counterclockwise quarter-turn rotation, and let $P$ normalize a nonzero vector.

The exact compatibility is the continuous scalar branch through zero satisfying

$$

R(P(n+rJn))=P(R(n)+\varphi(r)JR(n)),\qquad |r|<1.

$$

Boundary constraints on magnetic increments do not remove boundary auxiliary variables. Their primal and dual variations remain part of the coupled state, even when the corresponding increment is zero.

Returns
-------
A dimensionless floating-point array of shape (2, 2, N) containing the new auxiliary and multiplier rows in the value layer and their directional derivatives in the derivative layer.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_auxiliary_jet(
    systems: np.ndarray,
    r_jet: np.ndarray,
    multiplier_jet: np.ndarray,
    zeta: float,
    rho: float,
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
    r_jet : np.ndarray
        Finite magnetic coefficient jet, shape (2, N), with base coefficients
        strictly in (-1, 1).
    multiplier_jet : np.ndarray
        Finite old multiplier jet, shape (2, N), including all nodes.
    zeta : float
        Positive finite augmentation parameter in the identity metric, constant with
        respect to alpha.
    rho : float
        Positive finite dual-ascent step, constant with respect to alpha.

    Returns
    -------
    result : np.ndarray
        A dimensionless floating-point array of shape (2, 2, N) containing the new
        auxiliary and multiplier rows in the value layer and their directional
        derivatives in the derivative layer.

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


def _phi(r):
    return 2 * r / (1 - r * r)


def _dphi(r):
    return 2 * (1 + r * r) / (1 - r * r) ** 2


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


def _oracle_solve_auxiliary_jet(
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
            "call": "solve_auxiliary_jet(systems, r_jet, multiplier_jet, 4., 1.)",
            "gold_call": "_oracle_solve_auxiliary_jet(systems, r_jet, multiplier_jet, 4., 1.)",
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
systems[1]=0.
r_jet[1]=0.
multiplier_jet[1]=0.
""",
            "call": "solve_auxiliary_jet(systems, r_jet, multiplier_jet, 4., 1.)",
            "gold_call": "_oracle_solve_auxiliary_jet(systems, r_jet, multiplier_jet, 4., 1.)",
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
r_jet[0]=[-.7,.85]
""",
            "call": "solve_auxiliary_jet(systems, r_jet, multiplier_jet, 4., .7)",
            "gold_call": "_oracle_solve_auxiliary_jet(systems, r_jet, multiplier_jet, 4., .7)",
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
r_jet[0,0]=1.

def run_model():
    try:
        solve_auxiliary_jet(systems, r_jet, multiplier_jet, 4., 1.)
        return 0.0
    except ValueError:
        return 1.0
def run_gold():
    try:
        _oracle_solve_auxiliary_jet(systems, r_jet, multiplier_jet, 4., 1.)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
