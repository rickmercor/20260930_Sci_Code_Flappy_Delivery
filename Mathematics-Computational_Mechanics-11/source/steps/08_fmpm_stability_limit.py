"""
Finds the linear stability limit C_lin of a scheme on the vibrating bar: the smallest Courant number at which the spectral radius of the amplification matrix exceeds 1 + tol, located by scanning C = dC, 2 dC, ... up to C_max for the first point over the threshold and bisecting the bracketing interval to 1e-10.

The Courant limit of an explicit update is where its amplification matrix first acquires an eigenvalue outside the unit circle; measuring it from trial simulations, as the source does, mixes this linear limit with cell-crossing effects that the frozen-position analysis separates out.

Returns
-------
A float, the linear stability limit C_lin.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fmpm_stability_limit(L_r: float, dx: float, s: float, E: float, rho: float, scheme: str, k: int, alpha_blend: float, blend_period: int, tol: float, C_max: float, dC: float) -> float:
    """Finds the linear stability limit C_lin of a scheme on the vibrating bar: the smallest Courant number at which the
    spectral radius of the amplification matrix exceeds 1 + tol, located by scanning C = dC, 2 dC, ... up to C_max for
    the first point over the threshold and bisecting the bracketing interval to 1e-10.

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
        scheme: str, one of 'FLIP', 'FMPM' or 'EXACT' (the closed-form limit of the loop).
        k: int, the FMPM order (at least 1); FMPM(k) sums exactly k increments, so k = 1 gives the lumped-mass
            velocities.
        alpha_blend: float in (0, 1], the blend fraction applied to the increment (1 for no blending).
        blend_period: int, the blend period m of the source's generalised blend (at least 1): the increment of order l
            >= 2 is scaled by alpha_blend when m = 1 or l mod m = 1; the seed is never scaled.
        tol: float, the non-negative excess over 1 that the spectral radius must exceed for the operator to count as
            unstable.
        C_max: float, the largest Courant number of the scan.
        dC: float, the positive scan step.

    Returns:
        A float, the linear stability limit C_lin.

    Raises:
        ValueError: for invalid bar or scheme arguments, a negative tol, a non-positive dC, C_max not above dC, an
        operator unstable at every scanned Courant number, or no loss of stability up to C_max.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _check_scheme(scheme):
    if scheme not in ("FLIP", "FMPM", "EXACT"):
        raise ValueError("scheme must be FLIP, FMPM or EXACT")
    return scheme

def _spectral_radius(L_r, dx, s, E, rho, C, scheme, k, alpha_blend, blend_period):
    return float(np.max(np.abs(np.linalg.eigvals(_oracle_fmpm_bar_operator(L_r, dx, s, E, rho, C, scheme, k, alpha_blend, blend_period)))))

def _oracle_fmpm_stability_limit(L_r: float, dx: float, s: float, E: float, rho: float, scheme: str, k: int, alpha_blend: float, blend_period: int, tol: float, C_max: float, dC: float) -> float:
    t = float(tol); cm = float(C_max); dc = float(dC)
    if not np.isfinite(t) or t < 0.0 or not np.isfinite(cm) or not np.isfinite(dc) or dc <= 0.0 or cm <= dc:
        raise ValueError("bad scan settings")
    _check_scheme(scheme)
    rho_of = lambda C: _spectral_radius(L_r, dx, s, E, rho, C, scheme, k, alpha_blend, blend_period) - 1.0 - t
    C_a = 0.0
    C_b = None
    j = 1
    while j * dc <= cm + 1e-12:
        Cj = j * dc
        if rho_of(Cj) > 0.0:
            C_b = Cj
            break
        C_a = Cj
        j += 1
    if C_b is None:
        raise ValueError("no loss of stability below C_max")
    if C_a == 0.0:
        C_a = 1e-6 * dc
        if rho_of(C_a) > 0.0:
            raise ValueError("unstable at every scanned Courant number")
    return float(brentq(rho_of, C_a, C_b, xtol=1e-10, rtol=1e-12, maxiter=200))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': "import numpy as np\nL_r = 10.0\ndx = 1.0\ns = 0.25\nE = 2.0\nrho = 0.5\nscheme = 'FMPM'\nk = 4\nab = 1.0\nbp = 1\ntol = 1e-8\nC_max = 1.5\ndC = 0.01\n", 'call': 'float(fmpm_stability_limit(L_r, dx, s, E, rho, scheme, k, ab, bp, tol, C_max, dC))', 'gold_call': 'float(_oracle_fmpm_stability_limit(L_r, dx, s, E, rho, scheme, k, ab, bp, tol, C_max, dC))', 'tol': 1e-07},
        {'setup': "import numpy as np\n# the lumped-mass FLIP limit of the short bar\nL_r = 10.0\ndx = 1.0\ns = 0.25\nE = 2.0\nrho = 0.5\nscheme = 'FLIP'\nk = 1\nab = 1.0\nbp = 1\ntol = 1e-8\nC_max = 1.5\ndC = 0.01\n", 'call': 'float(fmpm_stability_limit(L_r, dx, s, E, rho, scheme, k, ab, bp, tol, C_max, dC))', 'gold_call': 'float(_oracle_fmpm_stability_limit(L_r, dx, s, E, rho, scheme, k, ab, bp, tol, C_max, dC))', 'tol': 1e-07},
        {'setup': "import numpy as np\n# the closed-form limit at a clustered placement, where the limit is small\nL_r = 10.0\ndx = 1.0\ns = 0.4\nE = 2.0\nrho = 0.5\nscheme = 'EXACT'\nk = 1\nab = 1.0\nbp = 1\ntol = 1e-8\nC_max = 1.5\ndC = 0.01\n", 'call': 'float(fmpm_stability_limit(L_r, dx, s, E, rho, scheme, k, ab, bp, tol, C_max, dC))', 'gold_call': 'float(_oracle_fmpm_stability_limit(L_r, dx, s, E, rho, scheme, k, ab, bp, tol, C_max, dC))', 'tol': 1e-07},
        {'setup': "import numpy as np\n# boundary: a coarse scan step still brackets the same crossing\nL_r = 10.0\ndx = 1.0\ns = 0.25\nE = 2.0\nrho = 0.5\nscheme = 'FMPM'\nk = 4\nab = 0.8\nbp = 2\ntol = 1e-8\nC_max = 1.2\ndC = 0.1\n", 'call': 'float(fmpm_stability_limit(L_r, dx, s, E, rho, scheme, k, ab, bp, tol, C_max, dC))', 'gold_call': 'float(_oracle_fmpm_stability_limit(L_r, dx, s, E, rho, scheme, k, ab, bp, tol, C_max, dC))', 'tol': 1e-07},
        {'setup': 'import numpy as np\n# invalid input: a scan that ends below every instability must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n', 'call': "_catches_value_error(lambda: fmpm_stability_limit(10.0, 1.0, 0.25, 2.0, 0.5, 'FLIP', 1, 1.0, 1, 1e-8, 0.3, 0.01))", 'gold_call': "_catches_value_error(lambda: _oracle_fmpm_stability_limit(10.0, 1.0, 0.25, 2.0, 0.5, 'FLIP', 1, 1.0, 1, 1e-8, 0.3, 0.01))"},
    ]
