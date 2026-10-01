"""
Applies the source's PIC-style FMPM(k) particle velocity and position updates with time-integration parameter alpha and also reports the effective acceleration, the mismatch between the two Lagrangian velocities implied by the position and velocity updates, and the velocity a FLIP update would have produced.

Replacing particle velocities by extrapolated grid velocities removes null-space noise at the price of dissipation; recasting the update as a FLIP update with a damping term shows what is damped, and the choice alpha = 1/2 removes the inconsistency between the position and velocity updates.

Returns
-------
A float64 array of shape (5, N): rows hold the updated particle velocities, the updated particle positions, the effective acceleration, the Lagrangian velocity mismatch and the FLIP velocity.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fmpm_particle_update(ops: "np.ndarray", N: int, v_plus: "np.ndarray", Vn: "np.ndarray", Xn: "np.ndarray", f_grid: "np.ndarray", dt: float, alpha: float) -> "np.ndarray":
    """Applies the source's PIC-style FMPM(k) particle velocity and position updates with time-integration parameter
    alpha and also reports the effective acceleration, the mismatch between the two Lagrangian velocities implied by
    the position and velocity updates, and the velocity a FLIP update would have produced.

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
        v_plus: array-like of shape (n,), the grid velocities v+(k).
        Vn: array-like of shape (N,), the current particle velocities.
        Xn: array-like of shape (N,), the current particle positions.
        f_grid: array-like of shape (n,), the nodal forces.
        dt: float, the positive time step.
        alpha: float, the time-integration parameter of the position update.

    Returns:
        A float64 array of shape (5, N): rows hold the updated particle velocities, the updated particle positions,
        the effective acceleration, the Lagrangian velocity mismatch and the FLIP velocity.

    Raises:
        ValueError: for a malformed ops array, arrays of the wrong sizes or with non-finite entries, a non-positive dt
        or a non-finite alpha.
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

def _oracle_fmpm_particle_update(ops: "np.ndarray", N: int, v_plus: "np.ndarray", Vn: "np.ndarray", Xn: "np.ndarray", f_grid: "np.ndarray", dt: float, alpha: float) -> "np.ndarray":
    S, Sp, m = _split(ops, int(N))
    n = m.size
    Np = int(N)
    v = _f64(v_plus).ravel()
    Vn = _f64(Vn).ravel()
    Xn = _f64(Xn).ravel()
    f = _f64(f_grid).ravel()
    if v.size != n or f.size != n or Vn.size != Np or Xn.size != Np:
        raise ValueError("bad array sizes")
    for arr, nm in ((v, "v_plus"), (Vn, "Vn"), (Xn, "Xn"), (f, "f_grid")):
        _check_finite(arr, nm)
    d = float(dt)
    a = float(alpha)
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError("bad dt")
    if not np.isfinite(a):
        raise ValueError("bad alpha")
    active = m > 0.0
    acc_grid = np.zeros(n, dtype=np.float64)
    acc_grid[active] = f[active] / m[active]   # a = m^-1 f  (lumped)
    V_new = np.dot(S, v)                       # V(n+1) = S v+(k)
    X_new = Xn + (a * V_new + (1.0 - a) * Vn) * d
    A_eff = (V_new - Vn) / d                   # Eq (9)
    mismatch = (a - 0.5) * (V_new - Vn)        # dX/dt - <V> for FMPM(k)
    V_flip = Vn + np.dot(S, acc_grid) * d      # Eq (8) FLIP velocity update
    out = np.zeros((5, Np), dtype=np.float64)
    out[0, :] = V_new
    out[1, :] = X_new
    out[2, :] = A_eff
    out[3, :] = mismatch
    out[4, :] = V_flip
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\nn = 10\nN = 14\nXp, Mp, dx, rng = _mk(n, N, 41)\nops = _ops(Xp, Mp, n, dx)\nv = rng.normal(size=n)\nVn = rng.normal(size=N)\nXn = np.sort(rng.uniform(0,(n-1)*dx,N))\nf = rng.normal(size=n)\ndt=0.01\nal=0.5\n', 'call': 'fmpm_particle_update(ops, N, v, Vn, Xn, f, dt, al)', 'gold_call': '_oracle_fmpm_particle_update(ops, N, v, Vn, Xn, f, dt, al)'},
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\nn = 10\nN = 14\nXp, Mp, dx, rng = _mk(n, N, 41)\nops = _ops(Xp, Mp, n, dx)\nv = rng.normal(size=n)\nVn = rng.normal(size=N)\nXn = np.sort(rng.uniform(0,(n-1)*dx,N))\nf = rng.normal(size=n)\ndt=0.01\nal=1.0\n', 'call': 'fmpm_particle_update(ops, N, v, Vn, Xn, f, dt, al)', 'gold_call': '_oracle_fmpm_particle_update(ops, N, v, Vn, Xn, f, dt, al)'},
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\nn = 12\nN = 20\nXp, Mp, dx, rng = _mk(n, N, 42)\nops = _ops(Xp, Mp, n, dx)\nv = rng.normal(size=n)\nVn = rng.normal(size=N)\nXn = np.sort(rng.uniform(0,(n-1)*dx,N))\nf = rng.normal(size=n)\ndt=0.05\nal=0.0\n', 'call': 'fmpm_particle_update(ops, N, v, Vn, Xn, f, dt, al)', 'gold_call': '_oracle_fmpm_particle_update(ops, N, v, Vn, Xn, f, dt, al)'},
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\n# boundary: grid velocities equal to the extrapolated particle velocities leave the effective acceleration at zero\nn = 8\nN = 6\nXp, Mp, dx, rng = _mk(n, N, 43)\nops = _ops(Xp, Mp, n, dx)\nv = np.full(n, 0.7)\nVn = np.full(N, 0.7)\nXn = Xp.copy()\nf = rng.normal(size=n)\ndt=0.02\nal=0.5\n', 'call': 'fmpm_particle_update(ops, N, v, Vn, Xn, f, dt, al)', 'gold_call': '_oracle_fmpm_particle_update(ops, N, v, Vn, Xn, f, dt, al)'},
        {'setup': 'import numpy as np\ndef _mk(n, N, seed, dx=1.0):\n    rng = np.random.default_rng(seed)\n    Xp = np.sort(rng.uniform(0.0, (n - 1) * dx, N))\n    Mp = rng.uniform(0.5, 1.5, N)\n    return Xp, Mp, dx, rng\ndef _ops(Xp, Mp, n, dx):\n    Xp = np.asarray(Xp, dtype=np.float64); Mp = np.asarray(Mp, dtype=np.float64)\n    N = Xp.size\n    xi = np.arange(n, dtype=np.float64) * dx\n    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dx\n    S = np.where(S > 0.0, S, 0.0)\n    m = (S * Mp[:, None]).sum(axis=0)\n    act = m > 0.0\n    Sp = np.zeros_like(S)\n    Sp[:, act] = (Mp[:, None] * S[:, act]) / m[None, act]\n    out = np.zeros((2 * N + 1, n), dtype=np.float64)\n    out[0:N, :] = S; out[N:2 * N, :] = Sp; out[2 * N, :] = m\n    return out\n# invalid input: a non-positive time step must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\nn = 10\nN = 14\nXp, Mp, dx, rng = _mk(n, N, 41)\nops = _ops(Xp, Mp, n, dx)\nv = rng.normal(size=n)\nVn = rng.normal(size=N)\nXn = np.sort(rng.uniform(0,(n-1)*dx,N))\nf = rng.normal(size=n)\n', 'call': '_catches_value_error(lambda: fmpm_particle_update(ops, N, v, Vn, Xn, f, 0.0, 0.5))', 'gold_call': '_catches_value_error(lambda: _oracle_fmpm_particle_update(ops, N, v, Vn, Xn, f, 0.0, 0.5))'},
    ]
