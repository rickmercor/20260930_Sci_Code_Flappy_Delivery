"""
Builds the linear shape function matrix S (N x n_nodes), the lumped nodal masses m and the lumped reverse map S+ (n_nodes x N) of a particle set on the uniform grid, stacked as the rows [S; (S+)^T; m] of one (2N + 1) x n_nodes array, the transpose of S+ being indexed like S; nodes carrying no mass are inactive.

Every mass matrix of the material point method is assembled from these operators: the lumped matrix from the column sums of S^T M, the full matrix from S^T M S, and the source's expansion of the full inverse from the product S+ S that is meant to approximate the identity.

Returns
-------
A float64 array of shape (2N + 1, n_nodes): rows 0..N-1 hold S, rows N..2N-1 hold (S+)^T (entry [N + p, i] = (S+)_ip = M_p S_pi/m_i, zero at an inactive node) and row 2N holds m.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fmpm_grid_operators(Xp: "np.ndarray", Mp: "np.ndarray", n_nodes: int, dx: float) -> "np.ndarray":
    """Builds the linear shape function matrix S (N x n_nodes), the lumped nodal masses m and the lumped reverse map S+
    (n_nodes x N) of a particle set on the uniform grid, stacked as the rows [S; (S+)^T; m] of one (2N + 1) x n_nodes
    array, the transpose of S+ being indexed like S; nodes carrying no mass are inactive.

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
        Xp: array-like of shape (N,), particle positions inside the grid (0 <= X_p <= (n_nodes - 1) dx).
        Mp: array-like of shape (N,), positive particle masses.
        n_nodes: int, the number of grid nodes (at least 2) at x_i = i dx.
        dx: float, the positive cell size.

    Returns:
        A float64 array of shape (2N + 1, n_nodes): rows 0..N-1 hold S, rows N..2N-1 hold (S+)^T (entry [N + p, i] =
        (S+)_ip = M_p S_pi/m_i, zero at an inactive node) and row 2N holds m.

    Raises:
        ValueError: for empty or mismatched particle arrays, n_nodes < 2, a non-positive dx, a non-finite entry, a
        non-positive mass or a particle outside the grid.
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

def _oracle_fmpm_grid_operators(Xp: "np.ndarray", Mp: "np.ndarray", n_nodes: int, dx: float) -> "np.ndarray":
    Xp = _f64(Xp).ravel()
    Mp = _f64(Mp).ravel()
    if Xp.size < 1 or Mp.size != Xp.size:
        raise ValueError("bad particle arrays")
    if not isinstance(n_nodes, (int, np.integer)) or isinstance(n_nodes, bool) or n_nodes < 2:
        raise ValueError("bad n_nodes")
    dxf = float(dx)
    if not np.isfinite(dxf) or dxf <= 0.0:
        raise ValueError("bad dx")
    _check_finite(Xp, "positions")
    _check_finite(Mp, "masses")
    if np.any(Mp <= 0.0):
        raise ValueError("non-positive mass")
    n = int(n_nodes)
    N = Xp.size
    xi = np.arange(n, dtype=np.float64) * dxf
    if np.any(Xp < 0.0) or np.any(Xp > xi[-1]):
        raise ValueError("particle outside grid")
    # linear (tent) shape functions
    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dxf
    S = np.where(S > 0.0, S, 0.0)
    m = (S * Mp[:, None]).sum(axis=0)          # m = diag(S^T M)
    active = m > 0.0
    Sp = np.zeros_like(S)
    Sp[:, active] = (Mp[:, None] * S[:, active]) / m[None, active]
    out = np.zeros((2 * N + 1, n), dtype=np.float64)
    out[0:N, :] = S
    out[N:2 * N, :] = Sp
    out[2 * N, :] = m
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\nn=10\nN=14\nXp, Mp, dx, rng = _mk(n, N, 1)\n', 'call': 'fmpm_grid_operators(Xp, Mp, n, dx)', 'gold_call': '_oracle_fmpm_grid_operators(Xp, Mp, n, dx)'},
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\nn=8\nN=9\nXp, Mp, dx, rng = _mk(n, N, 2)\n', 'call': 'fmpm_grid_operators(Xp, Mp, n, dx)', 'gold_call': '_oracle_fmpm_grid_operators(Xp, Mp, n, dx)'},
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\nn=16\nN=40\nXp, Mp, dx, rng = _mk(n, N, 3)\n', 'call': 'fmpm_grid_operators(Xp, Mp, n, dx)', 'gold_call': '_oracle_fmpm_grid_operators(Xp, Mp, n, dx)'},
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\n# boundary: a single particle sitting exactly on an interior node, so one node carries all the mass and its neighbours are inactive\nn=5\nXp = np.array([2.0])\nMp = np.array([1.3])\ndx = 1.0\n', 'call': 'fmpm_grid_operators(Xp, Mp, n, dx)', 'gold_call': '_oracle_fmpm_grid_operators(Xp, Mp, n, dx)'},
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\n# invalid input: a particle outside the grid must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\nn=6\n', 'call': '_catches_value_error(lambda: fmpm_grid_operators(np.array([1.0, 7.5]), np.array([1.0, 1.0]), n, 1.0))', 'gold_call': '_catches_value_error(lambda: _oracle_fmpm_grid_operators(np.array([1.0, 7.5]), np.array([1.0, 1.0]), n, 1.0))'},
    ]
