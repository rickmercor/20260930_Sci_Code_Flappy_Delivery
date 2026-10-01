"""
Assembles the (2N x 2N) single-step amplification matrix G(C) of one USL time step of the source's vibrating bar linearised at frozen particle positions, for the FLIP, FMPM(k) (with the given blend) or EXACT grid-velocity scheme, acting on the state [particle velocities; particle strains]; the particle-velocity block follows the scheme's particle update (PIC-style for FMPM and EXACT; FLIP with the reaction included in the grid force).

Linearising one time step at a fixed particle configuration turns the explicit update into a matrix whose eigenvalues decide whether any mode grows; the source measured its stability limits by trial simulations and cites such a spectral analysis only for the lumped-mass update.

Returns
-------
A float64 array of shape (2N, 2N), the amplification matrix on the state [V; eps].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fmpm_bar_operator(L_r: float, dx: float, s: float, E: float, rho: float, C: float, scheme: str, k: int, alpha_blend: float, blend_period: int) -> "np.ndarray":
    """Assembles the (2N x 2N) single-step amplification matrix G(C) of one USL time step of the source's vibrating bar
    linearised at frozen particle positions, for the FLIP, FMPM(k) (with the given blend) or EXACT grid-velocity
    scheme, acting on the state [particle velocities; particle strains]; the particle-velocity block follows the
    scheme's particle update (PIC-style for FMPM and EXACT; FLIP with the reaction included in the grid force).

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
        L_r: float, the bar length, a whole number of cells.
        dx: float, the positive cell size.
        s: float, the placement fraction of the two particles of each cell, at (i + s) dx and (i + 1 - s) dx, with 0 <
            s < 0.5.
        E: float, the positive elastic modulus.
        rho: float, the positive density.
        C: float, the positive Courant number, dt = C dx/v_wave.
        scheme: str, one of 'FLIP', 'FMPM' or 'EXACT' (the closed-form limit of the loop).
        k: int, the FMPM order (at least 1); FMPM(k) sums exactly k increments, so k = 1 gives the lumped-mass
            velocities.
        alpha_blend: float in (0, 1], the blend fraction applied to the increment (1 for no blending).
        blend_period: int, the blend period m of the source's generalised blend (at least 1): the increment of order l
            >= 2 is scaled by alpha_blend when m = 1 or l mod m = 1; the seed is never scaled.

    Returns:
        A float64 array of shape (2N, 2N), the amplification matrix on the state [V; eps].

    Raises:
        ValueError: for a bar length that is not a whole number of cells, a non-positive dx, E, rho or C, a placement
        fraction outside (0, 0.5), an unknown scheme, k < 1, alpha_blend outside (0, 1] or blend_period < 1.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

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

def _bar_particles(L_r, dx, s, rho):
    L = float(L_r); d = float(dx); sf = float(s); r = float(rho)
    if not (np.isfinite(L) and np.isfinite(d) and np.isfinite(sf) and np.isfinite(r)) or L <= 0.0 or d <= 0.0 or r <= 0.0:
        raise ValueError("bad bar geometry or density")
    ncell = int(round(L / d))
    if ncell < 1 or abs(ncell * d - L) > 1e-9 * max(1.0, L):
        raise ValueError("the bar length must be a whole number of cells")
    if not (0.0 < sf < 0.5):
        raise ValueError("the placement fraction must lie strictly between 0 and 0.5")
    n = ncell + 1
    Xp = np.array([(i + t) * d for i in range(ncell) for t in (sf, 1.0 - sf)])
    Vol = np.full(Xp.size, 0.5 * d)
    Mp = r * Vol
    return Xp, Mp, Vol, n, d

def _bar_gradients(Xp, n, dx):
    """d S_pi / dx for the tent functions (particles never sit on a node for 0 < s < 0.5)"""
    xi = np.arange(n, dtype=np.float64) * dx
    G = np.zeros((Xp.size, n))
    for p in range(Xp.size):
        i0 = int(np.floor(Xp[p] / dx))
        if 0 <= i0 < n:
            G[p, i0] = -1.0 / dx
        if 0 <= i0 + 1 < n:
            G[p, i0 + 1] = 1.0 / dx
    return G

def _check_scheme(scheme):
    if scheme not in ("FLIP", "FMPM", "EXACT"):
        raise ValueError("scheme must be FLIP, FMPM or EXACT")
    return scheme

def _oracle_fmpm_bar_operator(L_r: float, dx: float, s: float, E: float, rho: float, C: float, scheme: str, k: int, alpha_blend: float, blend_period: int) -> "np.ndarray":
    _check_scheme(scheme)
    k = _check_order(k)
    a, period = _check_blend(alpha_blend, blend_period)
    Ef = float(E); Cf = float(C)
    if not np.isfinite(Ef) or Ef <= 0.0 or not np.isfinite(Cf) or Cf <= 0.0:
        raise ValueError("bad modulus or Courant number")
    Xp, Mp, Vol, n, d = _bar_particles(L_r, dx, s, rho)
    N = Xp.size
    ops = _oracle_fmpm_grid_operators(Xp, Mp, n, d)
    S = ops[0:N, :]
    m = ops[2 * N, :]
    G = _bar_gradients(Xp, n, d)
    vwave = np.sqrt(Ef / float(rho))
    dt = Cf * d / vwave
    bc = [0]
    # linear maps of the state z = [V; eps]: momenta p = S^T M V, forces f = -G^T Omega E eps, reaction p+_0 = 0
    P_V = np.dot(S.T, np.diag(Mp))
    F_eps = -np.dot(G.T, np.diag(Vol * Ef))
    Pp_V = P_V.copy(); Pp_eps = F_eps * dt
    Pp_V[0, :] = 0.0; Pp_eps[0, :] = 0.0
    active = m > 0.0
    Dinv = np.zeros((n, n)); Dinv[active, active] = 1.0 / m[active]
    if scheme == "FLIP":
        vV = np.dot(Dinv, Pp_V); veps = np.dot(Dinv, Pp_eps)                       # lumped v+(1)
    else:
        inv = _oracle_fmpm_inverse_operator(ops, N, k, bc, a, period)
        K = inv[0:n, :] if scheme == "FMPM" else inv[n:2 * n, :]
        vV = np.dot(K, Pp_V); veps = np.dot(K, Pp_eps)
    # particle velocities through the particle update of the source, one column per unit state entry:
    # FMPM and EXACT take the PIC-style row V(n+1) = S v+, FLIP the row V(n) + S m^-1 f dt with f dt = p+ - p
    upd_row = 4 if scheme == "FLIP" else 0
    VV = np.zeros((N, N)); Veps = np.zeros((N, N))
    eyeN = np.eye(N)
    for j in range(N):
        VV[:, j] = _oracle_fmpm_particle_update(ops, N, vV[:, j], eyeN[:, j], Xp, (Pp_V[:, j] - P_V[:, j]) / dt, dt, 1.0)[upd_row, :]
        Veps[:, j] = _oracle_fmpm_particle_update(ops, N, veps[:, j], np.zeros(N), Xp, Pp_eps[:, j] / dt, dt, 1.0)[upd_row, :]
    EV = dt * np.dot(G, vV)
    Eeps = np.eye(N) + dt * np.dot(G, veps)                                           # USL strain update with the same grid velocities
    out = np.zeros((2 * N, 2 * N), dtype=np.float64)
    out[0:N, 0:N] = VV; out[0:N, N:] = Veps
    out[N:, 0:N] = EV; out[N:, N:] = Eeps
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': "import numpy as np\nL_r = 8.0\ndx = 1.0\ns = 0.25\nE = 2.0\nrho = 0.5\nC = 0.5\nscheme = 'FMPM'\nk = 4\nab = 1.0\nbp = 1\n", 'call': 'fmpm_bar_operator(L_r, dx, s, E, rho, C, scheme, k, ab, bp)', 'gold_call': '_oracle_fmpm_bar_operator(L_r, dx, s, E, rho, C, scheme, k, ab, bp)'},
        {'setup': "import numpy as np\n# the lumped-mass FLIP step of the same bar\nL_r = 8.0\ndx = 1.0\ns = 0.25\nE = 2.0\nrho = 0.5\nC = 0.5\nscheme = 'FLIP'\nk = 1\nab = 1.0\nbp = 1\n", 'call': 'fmpm_bar_operator(L_r, dx, s, E, rho, C, scheme, k, ab, bp)', 'gold_call': '_oracle_fmpm_bar_operator(L_r, dx, s, E, rho, C, scheme, k, ab, bp)'},
        {'setup': "import numpy as np\n# the closed-form limit of the loop with a clustered placement and a coarser cell\nL_r = 12.0\ndx = 2.0\ns = 0.4\nE = 3.0\nrho = 0.75\nC = 0.3\nscheme = 'EXACT'\nk = 1\nab = 1.0\nbp = 1\n", 'call': 'fmpm_bar_operator(L_r, dx, s, E, rho, C, scheme, k, ab, bp)', 'gold_call': '_oracle_fmpm_bar_operator(L_r, dx, s, E, rho, C, scheme, k, ab, bp)'},
        {'setup': "import numpy as np\n# boundary: the FMPM(2) blend at the placement where the two particles of a cell nearly coincide\nL_r = 6.0\ndx = 1.0\ns = 0.49\nE = 2.0\nrho = 0.5\nC = 0.2\nscheme = 'FMPM'\nk = 5\nab = 0.8\nbp = 2\n", 'call': 'fmpm_bar_operator(L_r, dx, s, E, rho, C, scheme, k, ab, bp)', 'gold_call': '_oracle_fmpm_bar_operator(L_r, dx, s, E, rho, C, scheme, k, ab, bp)'},
        {'setup': 'import numpy as np\n# invalid input: a placement fraction of 0.5 puts both particles of a cell on the same point and must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n', 'call': "_catches_value_error(lambda: fmpm_bar_operator(8.0, 1.0, 0.5, 2.0, 0.5, 0.5, 'FMPM', 4, 1.0, 1))", 'gold_call': "_catches_value_error(lambda: _oracle_fmpm_bar_operator(8.0, 1.0, 0.5, 2.0, 0.5, 0.5, 'FMPM', 4, 1.0, 1))"},
    ]
