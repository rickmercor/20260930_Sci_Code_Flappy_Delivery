"""
Assemble the rate correction factor of the perfect-sink site and its diagnostics at the given truncation order, from the coefficient sequences, the kernel and the truncated solution of the previous steps. The total capture flux through the unit sphere follows from the canonical unknowns as J = X_0 / 2. Also return: the zeroth-order analytical approximation J(0), one half of the zeroth iterate of the leading equation of the resolving system (the inhomogeneous term Q_00); the first-order approximation J(1), one half of the first iterate of the leading equation (Q_00 divided by 1 - q_0 Q_00), which is the source's first approximation for a free cavity behind a perfectly permeable shell; the relative percentage error delta1 = 100 (J - J(1)) / J; the sup-norm of the truncated system matrix M_lm = q_m Q_lm, that is the largest row sum of absolute values over l, m = 0..order; the effective steric factor f(theta0) = J(theta0; eps = 0) of the same site in an unbounded corona with the same hindrance exponent, from the same truncated system at eps = 0; and the coupling factor C = J(theta0; eps, Bi, p) / (f(theta0) J_shell), with J_shell the rate correction factor of the confined isotropic perfect sink in the same corona behind the same shell (Berg's 1 / (1 - eps) when Bi = inf and p = 0), which is one if anisotropy and confinement acted multiplicatively. Raise ValueError if theta0 is not in (0, pi], eps not in [0, 1), biot not positive or hindrance negative or not finite.

The source tabulates J against theta0 / pi for several thickness ratios together with the first approximation J(1), shows that J(1) is a uniform lower bound of J with a maximum error that grows as the shell gets thinner, and points out that anisotropic trapping and the diffusive interaction with the cavity wall are coupled rather than multiplicative effects.

Returns
-------
A (7,) float64 array [J, J(0), J(1), delta1 in percent, sup-norm of M, f(theta0), C].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rate_correction_summary(order, theta0, eps, biot, hindrance):
    """Assemble the rate correction factor of the perfect-sink site and its diagnostics at the
    given truncation order, from the coefficient sequences, the kernel and the truncated
    solution of the previous steps. A (7,) float64 array [J, J(0), J(1), delta1 in percent,
    sup-norm of M, f(theta0), C]."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def _oracle_rate_correction_summary(order, theta0, eps, biot, hindrance):
    n = int(order); t = float(theta0); e = float(eps); bi = float(biot); p = float(hindrance)
    if not (0.0 < t <= np.pi):
        raise ValueError("theta0 must lie in (0, pi]")
    if not (0.0 <= e < 1.0):
        raise ValueError("eps must lie in [0, 1)")
    if not (bi > 0.0):
        raise ValueError("biot must be positive")
    if not (p >= 0.0) or not np.isfinite(p):
        raise ValueError("hindrance exponent must be finite and non-negative")
    q = _oracle_dual_series_coefficients(n, e, bi, p)[4]
    Q = _oracle_minkov_matrix(n, t)
    X = _oracle_perfect_sink_solution(q, Q)
    J = 0.5 * X[0]
    mnorm = float(np.max(np.sum(np.abs(Q * q[None, :]), axis=1)))
    j0 = 0.5 * Q[0, 0]
    j1 = 0.5 * Q[0, 0] / (1.0 - q[0] * Q[0, 0])
    delta1 = 100.0 * (J - j1) / J
    f_esf = 0.5 * _oracle_perfect_sink_solution(_oracle_dual_series_coefficients(n, 0.0, bi, p)[4], Q)[0]
    eps0 = e ** (1.0 + p) if not np.isfinite(bi) else e ** (1.0 + p) - (1.0 + p) * e * e / bi
    j_shell = (1.0 + p) / (1.0 - eps0)
    coupling = J / (f_esf * j_shell)
    return np.array([J, j0, j1, delta1, mnorm, f_esf, coupling], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\norder = 200\ntheta0 = float(np.arccos(0.7))\neps = 0.625\nbiot = 2.5\nhindrance = 1.0\ndamkohler = 3.0\n',
         'call': 'rate_correction_summary(order, theta0, eps, biot, hindrance)',
         'gold_call': '_oracle_rate_correction_summary(order, theta0, eps, biot, hindrance)'},
        {'setup': 'import numpy as np\norder = 100\ntheta0 = 0.4 * np.pi\neps = 0.5\nbiot = np.inf\nhindrance = 0.0\n',
         'call': 'rate_correction_summary(order, theta0, eps, biot, hindrance)',
         'gold_call': '_oracle_rate_correction_summary(order, theta0, eps, biot, hindrance)'},
        {'setup': 'import numpy as np\norder = 60\ntheta0 = float(np.pi)\neps = 0.8\nbiot = 0.25\nhindrance = 2.0\n',
         'call': 'rate_correction_summary(order, theta0, eps, biot, hindrance)',
         'gold_call': '_oracle_rate_correction_summary(order, theta0, eps, biot, hindrance)'},
        {'setup': 'import numpy as np\norder = 400\ntheta0 = float(np.pi / 2)\neps = 2.0 / 4.2\nbiot = 0.5\nhindrance = 1.5\ndamkohler = 0.5\n',
         'call': 'rate_correction_summary(order, theta0, eps, biot, hindrance)',
         'gold_call': '_oracle_rate_correction_summary(order, theta0, eps, biot, hindrance)'},
    ]
