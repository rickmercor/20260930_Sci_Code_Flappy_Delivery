"""
Runs the complete chain and assembles the stability audit of the vibrating bar for ten schemes, in this row order: FLIP; FMPM(1); FMPM(2); FMPM(k_ref); FMPM(k_high); EXACT; FMPM(k_ref) blended with the lumped matrix (alpha_blend, period 1); FMPM(k_ref) blended with the FMPM(2) matrix (alpha_blend, period 2); FMPM(k_high) blended with the lumped matrix (alpha_blend, period 1); FMPM(k_high) blended with the FMPM(2) matrix (alpha_blend, period 2). Each scheme row holds the 12 columns [scheme code (1 FLIP, 2 FMPM, 3 EXACT), k (1 in the FLIP and EXACT rows), alpha_blend (1 in the unblended rows), blend period (1 in the unblended rows), C_lin at s, C_lin at s_clustered, energy retention after the given periods at C_ret and placement s, convergence rate of the loop at s (0 for FLIP), 2-norm of K(k) - K_inf at s (0 for FLIP and EXACT), number of steps of the retention run, 0, 0]. Head row 0 holds [C_lin of FMPM(k_ref) at s, C_lin of FLIP at s, C_lin of FMPM(1) at s, C_lin of EXACT at s, C_lin of EXACT at s_clustered, C_lin of FLIP at s_clustered, energy retention at C_ref of FMPM(1), of FMPM(k_ref), of the lumped-matrix blend and of the FMPM(2) blend, the spectral radius of T for the unblended loop at s and at s_clustered]; head row 1 holds [N, n_nodes, v_wave, T, number of steps of the retention runs at C_ret (the count tabulated in the scheme rows), C_lin of FMPM(2) at s, C_lin of FMPM(k_high) at s, C_lin of the lumped-matrix blend at s, C_lin of the FMPM(2) blend at s, 2-norm of K(k_ref) - K_inf at s and at s_clustered, and the largest absolute entry of the difference between the operators of the source's original expansion and of the revised loop at order k_ref on the design bar without controlled nodes and without blend (a consistency check, zero to rounding)]. The scheme rows follow.

The head scalar is the linear stability limit of the source's reference order on its own vibrating bar, a number the source only approaches by trial simulation; the table records how that limit depends on order, blending and particle placement and what each scheme does to the energy, so that the source's measured limits, its dissipation figures and its statement on conditioning can each be set against the linear analysis.

Returns
-------
A float64 array of shape (12, 12): two head rows followed by one row per scheme.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fmpm_stability_audit(L_r: float, dx: float, s: float, s_clustered: float, E: float, rho: float, v0: float, k_ref: int, k_high: int, alpha_blend: float, C_ref: float, C_ret: float, periods: float, tol: float, C_max: float, dC: float) -> "np.ndarray":
    """Runs the complete chain and assembles the stability audit of the vibrating bar for ten schemes, in this row order:
    FLIP; FMPM(1); FMPM(2); FMPM(k_ref); FMPM(k_high); EXACT; FMPM(k_ref) blended with the lumped matrix (alpha_blend,
    period 1); FMPM(k_ref) blended with the FMPM(2) matrix (alpha_blend, period 2); FMPM(k_high) blended with the
    lumped matrix (alpha_blend, period 1); FMPM(k_high) blended with the FMPM(2) matrix (alpha_blend, period 2). Each
    scheme row holds the 12 columns [scheme code (1 FLIP, 2 FMPM, 3 EXACT), k (1 in the FLIP and EXACT rows),
    alpha_blend (1 in the unblended rows), blend period (1 in the unblended rows), C_lin at s, C_lin at s_clustered,
    energy retention after the given periods at C_ret and placement s, convergence rate of the loop at s (0 for FLIP),
    2-norm of K(k) - K_inf at s (0 for FLIP and EXACT), number of steps of the retention run, 0, 0]. Head row 0 holds
    [C_lin of FMPM(k_ref) at s, C_lin of FLIP at s, C_lin of FMPM(1) at s, C_lin of EXACT at s, C_lin of EXACT at
    s_clustered, C_lin of FLIP at s_clustered, energy retention at C_ref of FMPM(1), of FMPM(k_ref), of the
    lumped-matrix blend and of the FMPM(2) blend, the spectral radius of T for the unblended loop at s and at
    s_clustered]; head row 1 holds [N, n_nodes, v_wave, T, number of steps of the retention runs at C_ret (the count
    tabulated in the scheme rows), C_lin of FMPM(2) at s, C_lin of FMPM(k_high) at s, C_lin of the lumped-matrix blend
    at s, C_lin of the FMPM(2) blend at s, 2-norm of K(k_ref) - K_inf at s and at s_clustered, and the largest
    absolute entry of the difference between the operators of the source's original expansion and of the revised loop
    at order k_ref on the design bar without controlled nodes and without blend (a consistency check, zero to
    rounding)]. The scheme rows follow.

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
        s_clustered: float, a second placement fraction (0 < s_clustered < 0.5) at which the limits are repeated.
        E: float, the positive elastic modulus.
        rho: float, the positive density.
        v0: float, the non-zero amplitude of the initial velocity v0 sin(pi X_p/(2 L_r)).
        k_ref: int, the reference order (at least 1) whose limit is the head-row answer.
        k_high: int, a high order (at least 1) audited against the closed-form limit; the blends are audited at k_ref
            and at k_high.
        alpha_blend: float in (0, 1], the blend fraction applied to the increment (1 for no blending).
        C_ref: float, the positive Courant number of the source's dissipation comparison, at which the head row
            reports energy retention.
        C_ret: float, the positive Courant number at which every audited scheme's energy retention is tabulated.
        periods: float, the positive number of fundamental periods over which the state is advanced.
        tol: float, the non-negative excess over 1 that the spectral radius must exceed for the operator to count as
            unstable.
        C_max: float, the largest Courant number of the scan.
        dC: float, the positive scan step.

    Returns:
        A float64 array of shape (12, 12): two head rows followed by one row per scheme.

    Raises:
        ValueError: for invalid bar, order, blend or Courant arguments of the underlying steps, a retention
        configuration beyond its linear limit, or a scan that finds no loss of stability.
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

def _oracle_fmpm_stability_audit(L_r: float, dx: float, s: float, s_clustered: float, E: float, rho: float, v0: float, k_ref: int, k_high: int, alpha_blend: float, C_ref: float, C_ret: float, periods: float, tol: float, C_max: float, dC: float) -> "np.ndarray":
    k_ref = _check_order(k_ref); k_high = _check_order(k_high)
    a, _ = _check_blend(alpha_blend, 1)
    Cr = float(C_ref); Ct = float(C_ret)
    if not np.isfinite(Cr) or Cr <= 0.0 or not np.isfinite(Ct) or Ct <= 0.0:
        raise ValueError("bad reference Courant numbers")
    configs = [(1.0, "FLIP", 1, 1.0, 1), (2.0, "FMPM", 1, 1.0, 1), (2.0, "FMPM", 2, 1.0, 1), (2.0, "FMPM", k_ref, 1.0, 1),
               (2.0, "FMPM", k_high, 1.0, 1), (3.0, "EXACT", 1, 1.0, 1), (2.0, "FMPM", k_ref, a, 1), (2.0, "FMPM", k_ref, a, 2),
               (2.0, "FMPM", k_high, a, 1), (2.0, "FMPM", k_high, a, 2)]
    ncol = 12
    rows = np.zeros((len(configs), ncol), dtype=np.float64)
    Xp, Mp, Vol, n, d = _bar_particles(L_r, dx, s, rho)
    N = Xp.size
    ops = _oracle_fmpm_grid_operators(Xp, Mp, n, d)
    Xc, Mc, Vc, nc, dc = _bar_particles(L_r, dx, s_clustered, rho)
    opsc = _oracle_fmpm_grid_operators(Xc, Mc, nc, dc)
    for r, (code, scheme, k, al, per) in enumerate(configs):
        Clim = _oracle_fmpm_stability_limit(L_r, dx, s, E, rho, scheme, k, al, per, tol, C_max, dC)
        Cclu = _oracle_fmpm_stability_limit(L_r, dx, s_clustered, E, rho, scheme, k, al, per, tol, C_max, dC)
        ret = _oracle_fmpm_energy_retention(L_r, dx, s, E, rho, scheme, k, al, per, Ct, v0, periods)
        if scheme == "FLIP":
            rate, err = 0.0, 0.0
        else:
            inv = _oracle_fmpm_inverse_operator(ops, N, k, [0], al, per)
            rate = inv[2 * n, 1]
            err = 0.0 if scheme == "EXACT" else inv[2 * n, 2]
        rows[r, :] = [code, float(k), al, float(per), Clim, Cclu, ret[0], rate, err, ret[3], 0.0, 0.0]
    inv_s = _oracle_fmpm_inverse_operator(ops, N, k_ref, [0], 1.0, 1)
    inv_c = _oracle_fmpm_inverse_operator(opsc, Xc.size, k_ref, [0], 1.0, 1)
    ret_ref = [_oracle_fmpm_energy_retention(L_r, dx, s, E, rho, sc, k, al, per, Cr, v0, periods)[0]
               for (sc, k, al, per) in (("FMPM", 1, 1.0, 1), ("FMPM", k_ref, 1.0, 1), ("FMPM", k_ref, a, 1), ("FMPM", k_ref, a, 2))]
    vwave = np.sqrt(float(E) / float(rho))
    # the source's original expansion against the revised loop at order k_ref on the design bar without controlled nodes
    eye = np.eye(n)
    K_loop = _oracle_fmpm_inverse_operator(ops, N, k_ref, None, 1.0, 1)[0:n, :]
    K_prior = np.column_stack([_oracle_fmpm_prior_series(ops, N, eye[:, j], k_ref)[0, :] for j in range(n)])
    expansion_gap = float(np.max(np.abs(K_loop - K_prior)))
    out = np.zeros((2 + len(configs), ncol), dtype=np.float64)
    out[0, :] = [rows[3, 4], rows[0, 4], rows[1, 4], rows[5, 4], rows[5, 5], rows[0, 5], ret_ref[0], ret_ref[1], ret_ref[2], ret_ref[3], inv_s[2 * n, 0], inv_c[2 * nc, 0]]
    out[1, :] = [float(N), float(n), vwave, 4.0 * float(L_r) / vwave, rows[3, 9], rows[2, 4], rows[4, 4], rows[6, 4], rows[7, 4], inv_s[2 * n, 2], inv_c[2 * nc, 2], expansion_gap]
    out[2:, :] = rows
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nL_r = 10.0\ndx = 1.0\ns = 0.25\ns_clustered = 0.4\nE = 2.0\nrho = 0.5\nv0 = 0.16\nk_ref = 4\nk_high = 20\nab = 0.8\nC_ref = 0.5\nC_ret = 0.4\nperiods = 2.0\ntol = 1e-8\nC_max = 1.5\ndC = 0.01\n', 'call': 'fmpm_stability_audit(L_r, dx, s, s_clustered, E, rho, v0, k_ref, k_high, ab, C_ref, C_ret, periods, tol, C_max, dC)', 'gold_call': '_oracle_fmpm_stability_audit(L_r, dx, s, s_clustered, E, rho, v0, k_ref, k_high, ab, C_ref, C_ret, periods, tol, C_max, dC)', 'tol': 1e-07},
        {'setup': 'import numpy as np\n# boundary: a coarser cell, a stiffer bar and a reference order of 2\nL_r = 12.0\ndx = 2.0\ns = 0.3\ns_clustered = 0.45\nE = 3.0\nrho = 0.75\nv0 = 0.1\nk_ref = 2\nk_high = 16\nab = 0.9\nC_ref = 0.3\nC_ret = 0.25\nperiods = 1.0\ntol = 1e-8\nC_max = 1.2\ndC = 0.02\n', 'call': 'fmpm_stability_audit(L_r, dx, s, s_clustered, E, rho, v0, k_ref, k_high, ab, C_ref, C_ret, periods, tol, C_max, dC)', 'gold_call': '_oracle_fmpm_stability_audit(L_r, dx, s, s_clustered, E, rho, v0, k_ref, k_high, ab, C_ref, C_ret, periods, tol, C_max, dC)', 'tol': 1e-07},
        {'setup': "import numpy as np\n# the source's bar at the design settings; this case grades head entry [0, 0], the FMPM(k_ref) limit,\n# which does not depend on k_high, so the high order is small here and the full k_high = 40 chain is\n# covered by the integration benchmark\nL_r = 40.0\ndx = 1.0\ns = 0.25\ns_clustered = 0.4\nE = 2.0\nrho = 0.5\nv0 = 0.16\nk_ref = 4\nk_high = 8\nab = 0.8\nC_ref = 0.539\nC_ret = 0.45\nperiods = 5.0\ntol = 1e-8\nC_max = 1.5\ndC = 0.01\n", 'call': 'float(np.asarray(fmpm_stability_audit(L_r, dx, s, s_clustered, E, rho, v0, k_ref, k_high, ab, C_ref, C_ret, periods, tol, C_max, dC))[0, 0])', 'gold_call': 'float(np.asarray(_oracle_fmpm_stability_audit(L_r, dx, s, s_clustered, E, rho, v0, k_ref, k_high, ab, C_ref, C_ret, periods, tol, C_max, dC))[0, 0])', 'tol': 1e-07},
        {'setup': 'import numpy as np\n# invalid input: a reference Courant number beyond the linear limit of FMPM(1) leaves no finite retention and must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n', 'call': '_catches_value_error(lambda: fmpm_stability_audit(10.0, 1.0, 0.25, 0.4, 2.0, 0.5, 0.16, 4, 20, 0.8, 0.95, 0.4, 2.0, 1e-8, 1.5, 0.01))', 'gold_call': '_catches_value_error(lambda: _oracle_fmpm_stability_audit(10.0, 1.0, 0.25, 0.4, 2.0, 0.5, 0.16, 4, 20, 0.8, 0.95, 0.4, 2.0, 1e-8, 1.5, 0.01))'},
    ]
