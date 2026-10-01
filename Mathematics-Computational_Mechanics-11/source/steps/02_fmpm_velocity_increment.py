"""
Advances one increment of the source's revised recursion: from the increment of order l - 1 it forms the increment of order l, scaled by alpha_blend, with the inactive nodes and then the controlled nodes of the result zeroed.

Each increment is the difference between the velocities of successive orders, so lumped-mass features such as grid velocity conditions can be imposed on every increment instead of conflicting with the expansion; blending only rescales the increment.

Returns
-------
A float64 array of shape (n,), the next velocity increment.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fmpm_velocity_increment(ops: "np.ndarray", N: int, dv_prev: "np.ndarray", bc_nodes: list, alpha_blend: float) -> "np.ndarray":
    """Advances one increment of the source's revised recursion: from the increment of order l - 1 it forms the increment
    of order l, scaled by alpha_blend, with the inactive nodes and then the controlled nodes of the result zeroed.

    Model and conventions (the source's revised FMPM(k) and its vibrating bar): a one-dimensional uniform background
    grid with nodes x_i = i dx, i = 0..n_nodes - 1; linear (tent) shape functions S_pi = max(0, 1 - |X_p - x_i|/dx)
    with gradients dS_pi/dx = -1/dx on the node below a particle and +1/dx on the node above it; N material points
    with positions X_p, positive masses M_p and undeformed volumes Omega_p; S is the N x n matrix of the S_pi (n =
    n_nodes); lumped nodal masses m_i = sum_p M_p S_pi, a node with m_i = 0 being inactive and carrying nothing; the
    lumped reverse map S+ is the n x N operator with entries (S+)_ip = M_p S_pi/m_i on the active nodes and zero rows
    at the inactive nodes, so that S+ S is n x n, the full mass matrix is m~ = S^T M S = m (I - A) and A = I - S+ S;
    the stacked operator array stores the transpose of S+, an N x n block indexed like S. The revised FMPM loop: the
    FMPM(k) grid velocities v+(k) are the sum of k increments Delta v(l), l = 1..k, the increment of order l being the
    difference between the velocities of the source's truncated expansions of the full mass matrix inverse at orders l
    and l - 1, seeded by the lumped-mass velocity Delta v(1) = m^-1 p+ on the active nodes; every increment, the seed
    included, is zeroed on the inactive nodes and on the controlled (zero-velocity) nodes, so a controlled node
    carries its prescribed zero velocity whatever momentum is supplied there; the generalised blend of the source
    scales the increment of order l >= 2 by alpha_blend when blend_period = 1 or l mod blend_period = 1 and by 1
    otherwise, and never scales the seed (blend_period = 1 blends with the lumped mass matrix, blend_period = 2 with
    the FMPM(2) matrix); the early exit stops the loop once the Euclidean norm of the increment just added falls below
    tol (tested after each increment beyond the seed). The source's original, non-incremental expansion writes the
    same order-k velocities as an alternating sum sum_l (-1)^(l+1) v*_l over l = 1..k whose terms are multiples of (S+
    S)^(l-1) m^-1 p+ by coefficients that depend on k and l; it must agree with the unconstrained, unblended loop.
    Particle updates: the source's FMPM(k) update is PIC-style, replacing every particle velocity by the extrapolated
    grid velocity, V(n+1) = S v+(k), and advancing the positions over the step with the velocity alpha V(n+1) + (1 -
    alpha) V(n) (time-integration parameter alpha); its effective acceleration is the change of the particle velocity
    over the step per unit time, and its Lagrangian velocity mismatch is the difference between the velocity that
    advances the positions and the mean of V(n) and V(n+1); the FLIP update adds the extrapolated lumped-mass
    acceleration increment, V(n+1) = V(n) + S m^-1 f dt. The loop as an operator: K(k) is the n x n matrix with v+(k)
    = K(k) p+ for the constrained, blended loop without early exit; K_inf is the closed-form limit of the same
    recursion continued to infinite order, which exists when the recursion converges and equals the full mass matrix
    inverse on the active nodes when no node is controlled (K(k) and K_inf have zero rows at the controlled nodes); T
    denotes the unblended increment map, the linear map that carries one increment to the next with alpha_blend = 1,
    the zeroing of the inactive and controlled nodes included; the convergence rate of the loop is the spectral rate
    per increment of its blended recursion, the spectral radius of the map that carries the increments over one blend
    period taken to the power 1/blend_period; the truncation error of order k is the 2-norm (largest singular value)
    of K(k) - K_inf. The vibrating bar: L_r = n_cells dx, two particles per cell at (i + s) dx and (i + 1 - s) dx for
    a placement fraction 0 < s < 0.5, volumes dx/2 and masses rho dx/2, linear-elastic small-strain material sigma = E
    eps with wave speed v_wave = sqrt(E/rho), the left node 0 held at zero velocity (the reaction sets p+_0 = 0 and
    the loop zeroes the controlled entry of every increment), the right end free, initial particle velocities v0
    sin(pi X_p/(2 L_r)), fundamental period T = 4 L_r/v_wave, time step dt = C dx/v_wave for the Courant number C. One
    USL time step linearised at frozen particle positions acts on the state z = [V; eps] (particle velocities, then
    particle strains): momenta p = S^T M V, internal forces f = -G^T Omega E eps with G the N x n matrix of the shape
    function gradients, p+ = p + f dt with the reaction p+_0 = 0, grid velocities v+ from the scheme (FLIP: m^-1 p+;
    FMPM: the loop to order k with the given blend; EXACT: the closed-form limit of the loop), particle velocities
    from the scheme's update (the PIC-style update for FMPM and EXACT; the FLIP update, whose grid force includes the
    reaction at the controlled node), strains eps(n+1) = eps + dt G v+ with the same grid velocities. The
    amplification matrix G(C) is the (2N x 2N) matrix of that map. The linear stability limit C_lin is the smallest
    Courant number at which the spectral radius of G(C) exceeds 1 + tol: C is scanned at dC, 2 dC, ... up to C_max,
    the first scan point over the threshold and the point before it bracket the crossing, and the crossing is located
    by bisection to 1e-10. The energy retention after P periods is (kinetic + strain energy)/(initial kinetic energy)
    after round(P T/dt) applications of G to the initial state, with kinetic energy sum_p M_p V_p^2/2 and strain
    energy sum_p Omega_p E eps_p^2/2.

    Args:
        ops: array of shape (2N + 1, n), the stacked rows [S; (S+)^T; m] returned by fmpm_grid_operators.
        N: int, the particle count that splits the rows of ops.
        dv_prev: array-like of shape (n,), the previous velocity increment Delta v(l - 1).
        bc_nodes: sequence of int node indices carrying a zero-velocity condition, or None.
        alpha_blend: float in (0, 1], the blend fraction applied to the increment (1 for no blending).

    Returns:
        A float64 array of shape (n,), the next velocity increment.

    Raises:
        ValueError: for an ops array of the wrong shape or with non-finite entries, a dv_prev of the wrong length,
        alpha_blend outside (0, 1] or a controlled node index outside the grid.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _check_finite(a, what):
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite " + what)

def _split(ops, N):
    ops = _f64(ops)
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or N < 1:
        raise ValueError("bad N")
    if ops.ndim != 2 or ops.shape[0] != 2 * int(N) + 1:
        raise ValueError("bad ops shape")
    _check_finite(ops, "ops")
    return ops[0:N, :], ops[N:2 * N, :], ops[2 * N, :]

def _SpS(S, Sp, v):
    # (S+ S v)_i = sum_p Sp[p,i] * (S v)_p
    return np.dot(Sp.T, np.dot(S, v))

def _bc_array(bc_nodes, n):
    bc = np.asarray(bc_nodes, dtype=np.int64).ravel() if bc_nodes is not None else np.zeros(0, np.int64)
    if bc.size and (bc.min() < 0 or bc.max() >= n):
        raise ValueError("bad bc node index")
    return bc

def _oracle_fmpm_velocity_increment(ops: "np.ndarray", N: int, dv_prev: "np.ndarray", bc_nodes: list, alpha_blend: float) -> "np.ndarray":
    S, Sp, m = _split(ops, int(N))
    n = m.size
    dv = _f64(dv_prev).ravel()
    if dv.size != n:
        raise ValueError("bad dv_prev length")
    _check_finite(dv, "dv_prev")
    a = float(alpha_blend)
    if not np.isfinite(a) or a <= 0.0 or a > 1.0:
        raise ValueError("bad alpha_blend")
    bc = _bc_array(bc_nodes, n)
    dv_new = a * (dv - _SpS(S, Sp, dv))
    dv_new[m <= 0.0] = 0.0                     # inactive nodes carry nothing
    if bc.size:
        dv_new[bc] = 0.0                       # constraint imposed on the increment
    return dv_new.astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\nn = 10\nN = 14\nXp, Mp, dx, rng = _mk(n, N, 11)\nops = _ops(Xp, Mp, n, dx)\ndv = rng.normal(size=n)\nbc = None\nab = 1.0\n', 'call': 'fmpm_velocity_increment(ops, N, dv, bc, ab)', 'gold_call': '_oracle_fmpm_velocity_increment(ops, N, dv, bc, ab)'},
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\nn = 10\nN = 14\nXp, Mp, dx, rng = _mk(n, N, 11)\nops = _ops(Xp, Mp, n, dx)\ndv = rng.normal(size=n)\nbc = [0]\nab = 1.0\n', 'call': 'fmpm_velocity_increment(ops, N, dv, bc, ab)', 'gold_call': '_oracle_fmpm_velocity_increment(ops, N, dv, bc, ab)'},
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\nn = 12\nN = 20\nXp, Mp, dx, rng = _mk(n, N, 12)\nops = _ops(Xp, Mp, n, dx)\ndv = rng.normal(size=n)\nbc = [0, 11]\nab = 0.8\n', 'call': 'fmpm_velocity_increment(ops, N, dv, bc, ab)', 'gold_call': '_oracle_fmpm_velocity_increment(ops, N, dv, bc, ab)'},
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\n# boundary: an increment that lives only on an inactive node returns zero everywhere\nn = 6\nXp = np.array([2.2, 2.7, 3.4])\nMp = np.array([1.0, 1.0, 1.0])\ndx = 1.0\nN = 3\nops = _ops(Xp, Mp, n, dx)\ndv = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])\nbc = None\nab = 1.0\n', 'call': 'fmpm_velocity_increment(ops, N, dv, bc, ab)', 'gold_call': '_oracle_fmpm_velocity_increment(ops, N, dv, bc, ab)'},
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\n# invalid input: a blend fraction above 1 is not a blend and must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\nn = 10\nN = 14\nXp, Mp, dx, rng = _mk(n, N, 11)\nops = _ops(Xp, Mp, n, dx)\ndv = rng.normal(size=n)\n', 'call': '_catches_value_error(lambda: fmpm_velocity_increment(ops, N, dv, None, 1.5))', 'gold_call': '_catches_value_error(lambda: _oracle_fmpm_velocity_increment(ops, N, dv, None, 1.5))'},
    ]
