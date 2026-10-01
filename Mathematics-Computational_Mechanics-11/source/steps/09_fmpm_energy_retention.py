"""
Advances the linearised bar from its initial velocity profile v0 sin(pi X_p/(2 L_r)) through round(periods T/dt) applications of the amplification matrix at Courant number C and returns the retained total energy, the kinetic and the strain parts, each relative to the initial kinetic energy, and the number of steps taken.

The source characterises its schemes by the energy left after five vibration periods: FLIP and high-order FMPM(k) keep it, the lumped-mass PIC update FMPM(1) loses a large fraction, and blending with the lumped matrix inherits that loss while blending with the FMPM(2) matrix does not.

Returns
-------
A float64 array of shape (4,): [retained energy fraction, kinetic fraction, strain fraction, number of steps].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fmpm_energy_retention(L_r: float, dx: float, s: float, E: float, rho: float, scheme: str, k: int, alpha_blend: float, blend_period: int, C: float, v0: float, periods: float) -> "np.ndarray":
    """Advances the linearised bar from its initial velocity profile v0 sin(pi X_p/(2 L_r)) through round(periods T/dt)
    applications of the amplification matrix at Courant number C and returns the retained total energy, the kinetic
    and the strain parts, each relative to the initial kinetic energy, and the number of steps taken.

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
        C: float, the positive Courant number, dt = C dx/v_wave.
        v0: float, the non-zero amplitude of the initial velocity v0 sin(pi X_p/(2 L_r)).
        periods: float, the positive number of fundamental periods over which the state is advanced.

    Returns:
        A float64 array of shape (4,): [retained energy fraction, kinetic fraction, strain fraction, number of steps].

    Raises:
        ValueError: for invalid bar or scheme arguments, a zero or non-finite v0, a non-positive periods, or a
        configuration whose spectral radius exceeds 1 + 1e-8 so that its energy has no finite retention.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

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

def _oracle_fmpm_energy_retention(L_r: float, dx: float, s: float, E: float, rho: float, scheme: str, k: int, alpha_blend: float, blend_period: int, C: float, v0: float, periods: float) -> "np.ndarray":
    v0f = float(v0); per = float(periods)
    if not np.isfinite(v0f) or v0f == 0.0 or not np.isfinite(per) or per <= 0.0:
        raise ValueError("bad amplitude or period count")
    Gm = _oracle_fmpm_bar_operator(L_r, dx, s, E, rho, C, scheme, k, alpha_blend, blend_period)
    if float(np.max(np.abs(np.linalg.eigvals(Gm)))) > 1.0 + 1e-8:
        raise ValueError("the configuration is linearly unstable at this Courant number; its energy has no finite retention")
    Xp, Mp, Vol, n, d = _bar_particles(L_r, dx, s, rho)
    N = Xp.size
    Ef = float(E)
    vwave = np.sqrt(Ef / float(rho))
    dt = float(C) * d / vwave
    T = 4.0 * float(L_r) / vwave                       # fundamental period of the fixed-free bar
    nsteps = int(round(per * T / dt))
    z = np.concatenate([v0f * np.sin(np.pi * Xp / (2.0 * float(L_r))), np.zeros(N)])
    E0 = 0.5 * float(np.sum(Mp * z[:N] ** 2))
    for _ in range(nsteps):
        z = np.dot(Gm, z)
    ke = 0.5 * float(np.sum(Mp * z[:N] ** 2))
    se = 0.5 * float(np.sum(Vol * Ef * z[N:] ** 2))
    return np.array([(ke + se) / E0, ke / E0, se / E0, float(nsteps)], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': "import numpy as np\nL_r = 10.0\ndx = 1.0\ns = 0.25\nE = 2.0\nrho = 0.5\nscheme = 'FMPM'\nk = 1\nab = 1.0\nbp = 1\nC = 0.5\nv0 = 0.16\nperiods = 2.0\n", 'call': 'fmpm_energy_retention(L_r, dx, s, E, rho, scheme, k, ab, bp, C, v0, periods)', 'gold_call': '_oracle_fmpm_energy_retention(L_r, dx, s, E, rho, scheme, k, ab, bp, C, v0, periods)', 'tol': 1e-07},
        {'setup': "import numpy as np\nL_r = 10.0\ndx = 1.0\ns = 0.25\nE = 2.0\nrho = 0.5\nscheme = 'FMPM'\nk = 4\nab = 1.0\nbp = 1\nC = 0.5\nv0 = 0.16\nperiods = 2.0\n", 'call': 'fmpm_energy_retention(L_r, dx, s, E, rho, scheme, k, ab, bp, C, v0, periods)', 'gold_call': '_oracle_fmpm_energy_retention(L_r, dx, s, E, rho, scheme, k, ab, bp, C, v0, periods)', 'tol': 1e-07},
        {'setup': "import numpy as np\n# the FLIP update over one period\nL_r = 10.0\ndx = 1.0\ns = 0.25\nE = 2.0\nrho = 0.5\nscheme = 'FLIP'\nk = 1\nab = 1.0\nbp = 1\nC = 0.6\nv0 = 0.1\nperiods = 1.0\n", 'call': 'fmpm_energy_retention(L_r, dx, s, E, rho, scheme, k, ab, bp, C, v0, periods)', 'gold_call': '_oracle_fmpm_energy_retention(L_r, dx, s, E, rho, scheme, k, ab, bp, C, v0, periods)', 'tol': 1e-07},
        {'setup': "import numpy as np\n# boundary: the lumped-matrix blend of FMPM(4) at the source's blend fraction\nL_r = 12.0\ndx = 1.0\ns = 0.3\nE = 2.0\nrho = 0.5\nscheme = 'FMPM'\nk = 4\nab = 0.8\nbp = 1\nC = 0.4\nv0 = 0.16\nperiods = 3.0\n", 'call': 'fmpm_energy_retention(L_r, dx, s, E, rho, scheme, k, ab, bp, C, v0, periods)', 'gold_call': '_oracle_fmpm_energy_retention(L_r, dx, s, E, rho, scheme, k, ab, bp, C, v0, periods)', 'tol': 1e-07},
        {'setup': 'import numpy as np\n# invalid input: a Courant number beyond the linear limit has no finite retention and must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n', 'call': "_catches_value_error(lambda: fmpm_energy_retention(10.0, 1.0, 0.25, 2.0, 0.5, 'FMPM', 4, 1.0, 1, 0.9, 0.16, 1.0))", 'gold_call': "_catches_value_error(lambda: _oracle_fmpm_energy_retention(10.0, 1.0, 0.25, 2.0, 0.5, 'FMPM', 4, 1.0, 1, 0.9, 0.16, 1.0))"},
    ]
