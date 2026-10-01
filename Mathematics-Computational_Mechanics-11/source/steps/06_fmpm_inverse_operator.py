"""
Builds the n x n matrix K(k) that the constrained, blended FMPM loop applies to the momenta, v+(k) = K(k) p+ (the revised loop driven by one unit momentum per column, without early exit), the closed-form limit K_inf of the same recursion continued to infinite order (so K(k) and K_inf have zero rows at the controlled nodes), and the diagnostics of the series: the spectral radius of the unblended increment map T, the convergence rate per increment of the blended recursion, the 2-norm and the largest entry of K(k) - K_inf, and the number of active nodes.

The loop is a truncated Neumann series; writing it as an operator exposes its limit, which is the full mass matrix inverse when no node is controlled, and its convergence rate, which is what deteriorates when particles cluster inside a cell and the full mass matrix approaches singularity.

Returns
-------
A float64 array of shape (2n + 1, n): rows 0..n-1 hold K(k), rows n..2n-1 hold K_inf, and row 2n holds [spectral radius of T, convergence rate per increment, 2-norm (largest singular value) of K(k) - K_inf, largest absolute entry of K(k) - K_inf, number of active nodes] followed by zeros.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fmpm_inverse_operator(ops: "np.ndarray", N: int, k: int, bc_nodes: list, alpha_blend: float, blend_period: int) -> "np.ndarray":
    """Builds the n x n matrix K(k) that the constrained, blended FMPM loop applies to the momenta, v+(k) = K(k) p+ (the
    revised loop driven by one unit momentum per column, without early exit), the closed-form limit K_inf of the same
    recursion continued to infinite order (so K(k) and K_inf have zero rows at the controlled nodes), and the
    diagnostics of the series: the spectral radius of the unblended increment map T, the convergence rate per
    increment of the blended recursion, the 2-norm and the largest entry of K(k) - K_inf, and the number of active
    nodes.

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
        k: int, the FMPM order (at least 1); FMPM(k) sums exactly k increments, so k = 1 gives the lumped-mass
            velocities.
        bc_nodes: sequence of int node indices carrying a zero-velocity condition, or None.
        alpha_blend: float in (0, 1], the blend fraction applied to the increment (1 for no blending).
        blend_period: int, the blend period m of the source's generalised blend (at least 1): the increment of order l
            >= 2 is scaled by alpha_blend when m = 1 or l mod m = 1; the seed is never scaled.

    Returns:
        A float64 array of shape (2n + 1, n): rows 0..n-1 hold K(k), rows n..2n-1 hold K_inf, and row 2n holds
        [spectral radius of T, convergence rate per increment, 2-norm (largest singular value) of K(k) - K_inf,
        largest absolute entry of K(k) - K_inf, number of active nodes] followed by zeros.

    Raises:
        ValueError: for a malformed ops array, k < 1, alpha_blend outside (0, 1], blend_period < 1, a controlled node
        index outside the grid, or a blended recursion that does not converge (spectral radius of its map over one
        blend period at or above 1), so that the series has no limit.
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

def _bc_array(bc_nodes, n):
    bc = np.asarray(bc_nodes, dtype=np.int64).ravel() if bc_nodes is not None else np.zeros(0, np.int64)
    if bc.size and (bc.min() < 0 or bc.max() >= n):
        raise ValueError("bad bc node index")
    return bc

def _check_order(k):
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError("bad k")
    return int(k)

def _check_blend(alpha_blend, blend_period):
    a = float(alpha_blend)
    if not np.isfinite(a) or a <= 0.0 or a > 1.0:
        raise ValueError("bad alpha_blend")
    if not isinstance(blend_period, (int, np.integer)) or isinstance(blend_period, bool) or blend_period < 1:
        raise ValueError("bad blend_period")
    return a, int(blend_period)

def _loop_matrices(ops, N, bc_nodes):
    """D = seed map m^-1 on the active nodes with zero rows at the controlled nodes, T = Z A with A = I - S+ S and Z zeroing inactive and controlled nodes"""
    S, Sp, m = _split(ops, int(N))
    n = m.size
    bc = _bc_array(bc_nodes, n)
    active = m > 0.0
    D = np.zeros((n, n))
    D[active, active] = 1.0 / m[active]
    if bc.size:
        D[bc, :] = 0.0
    A = np.eye(n) - np.dot(Sp.T, S)
    z = active.astype(np.float64)
    if bc.size:
        z[bc] = 0.0
    T = z[:, None] * A
    return D, T, n

def _oracle_fmpm_inverse_operator(ops: "np.ndarray", N: int, k: int, bc_nodes: list, alpha_blend: float, blend_period: int) -> "np.ndarray":
    # K(k) and K_inf depend only on the arguments below, never on the Courant number, so a stability
    # scan that walks C over hundreds of points rebuilds an identical operator each time. Memoise it on
    # the exact inputs; the result is bit-identical and history cannot affect it.
    arr = np.ascontiguousarray(np.asarray(ops, dtype=np.float64))
    key = (arr.tobytes(), arr.shape, int(N), int(k),
           (None if bc_nodes is None else tuple(int(b) for b in bc_nodes)),
           float(alpha_blend), int(blend_period))
    cache = getattr(_oracle_fmpm_inverse_operator, "_cache", None)
    if cache is None:
        cache = {}
        _oracle_fmpm_inverse_operator._cache = cache
    hit = cache.get(key)
    if hit is not None:
        return hit.copy()
    out = _fmpm_inverse_operator_uncached(ops, N, k, bc_nodes, alpha_blend, blend_period)
    if len(cache) >= 64:
        cache.pop(next(iter(cache)))
    cache[key] = np.array(out, copy=True)
    return out

def _fmpm_inverse_operator_uncached(ops: "np.ndarray", N: int, k: int, bc_nodes: list, alpha_blend: float, blend_period: int) -> "np.ndarray":
    k = _check_order(k)
    a, period = _check_blend(alpha_blend, blend_period)
    D, T, n = _loop_matrices(ops, N, bc_nodes)
    # K(k): the revised loop of the source driven by one unit momentum per column (no early exit)
    eye = np.eye(n)
    K = np.column_stack([_oracle_fmpm_loop(ops, N, eye[:, j], k, bc_nodes, a, period, 0.0)[0, :] for j in range(n)])
    # the exact limit of the same recursion: sum_{j>=0} alpha^j T^(j m) (I + T + ... + T^(m-1)) D = (I - alpha T^m)^-1 (I + ... + T^(m-1)) D
    partial = np.eye(n)
    powT = np.eye(n)
    for _ in range(1, period):
        powT = np.dot(powT, T)
        partial = partial + powT
    Tm = np.dot(powT, T)                                   # T^m
    rhoT = float(np.max(np.abs(np.linalg.eigvals(T))))
    rho_block = float(np.max(np.abs(np.linalg.eigvals(a * Tm))))
    if rho_block >= 1.0:
        raise ValueError("the FMPM series does not converge: the blended recursion has spectral radius >= 1")
    K_inf = np.linalg.solve(np.eye(n) - a * Tm, np.dot(partial, D))
    diff = K - K_inf
    out = np.zeros((2 * n + 1, n), dtype=np.float64)
    out[0:n, :] = K
    out[n:2 * n, :] = K_inf
    out[2 * n, 0] = rhoT
    out[2 * n, 1] = rho_block ** (1.0 / period)
    out[2 * n, 2] = float(np.linalg.norm(diff, 2))
    out[2 * n, 3] = float(np.max(np.abs(diff)))
    out[2 * n, 4] = float(np.count_nonzero(_split(ops, int(N))[2] > 0.0))
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\nn = 10\nN = 14\nXp, Mp, dx, rng = _mk(n, N, 51)\nops = _ops(Xp, Mp, n, dx)\nk=4\nbc=[0]\nab=1.0\nbp=1\n', 'call': 'fmpm_inverse_operator(ops, N, k, bc, ab, bp)', 'gold_call': '_oracle_fmpm_inverse_operator(ops, N, k, bc, ab, bp)'},
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\n# boundary: no controlled node, so the limit is the full mass matrix inverse on the active nodes\nn = 10\nN = 14\nXp, Mp, dx, rng = _mk(n, N, 51)\nops = _ops(Xp, Mp, n, dx)\nk=1\nbc=None\nab=1.0\nbp=1\n', 'call': 'fmpm_inverse_operator(ops, N, k, bc, ab, bp)', 'gold_call': '_oracle_fmpm_inverse_operator(ops, N, k, bc, ab, bp)'},
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\nn = 12\nN = 20\nXp, Mp, dx, rng = _mk(n, N, 52)\nops = _ops(Xp, Mp, n, dx)\nk=6\nbc=[0, 11]\nab=0.8\nbp=1\n', 'call': 'fmpm_inverse_operator(ops, N, k, bc, ab, bp)', 'gold_call': '_oracle_fmpm_inverse_operator(ops, N, k, bc, ab, bp)'},
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\n# the FMPM(2) blend and its closed-form limit\nn = 12\nN = 20\nXp, Mp, dx, rng = _mk(n, N, 52)\nops = _ops(Xp, Mp, n, dx)\nk=7\nbc=[0]\nab=0.8\nbp=2\n', 'call': 'fmpm_inverse_operator(ops, N, k, bc, ab, bp)', 'gold_call': '_oracle_fmpm_inverse_operator(ops, N, k, bc, ab, bp)'},
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\n# invalid input: a controlled node outside the grid must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\nn = 10\nN = 14\nXp, Mp, dx, rng = _mk(n, N, 51)\nops = _ops(Xp, Mp, n, dx)\n', 'call': '_catches_value_error(lambda: fmpm_inverse_operator(ops, N, 3, [10], 1.0, 1))', 'gold_call': '_catches_value_error(lambda: _oracle_fmpm_inverse_operator(ops, N, 3, [10], 1.0, 1))'},
    ]
