"""
Run the complete chain for every nanoreactor design in design_table, a list of rows (R, R0, site_fraction, D, kappa, P, p) in nm, nm, -, nm^2/us, nm/us, nm/us and - with finite kappa (P may be inf), at the given truncation order, and return one table. For each design map the parameters, derive the coefficient sequences, build the Minkov kernel, solve the truncated perfect-sink system from q and the kernel and assemble its summary, reconstruct the perfect-sink field at the two mid-shell probe points xi_mid = (1 + 1 / eps) / 2 on the site axis (theta = 0) and at the inert pole (theta = pi) together with the fraction of the capture entering through the inner half-angle sub-cap 0 <= theta <= theta0 / 2 and the fraction of the shell entry flux through the hemisphere facing the site, solve the projected radiation system from q, theta0 and Da, evaluate the naive-reduction diagnostic from q, the kernel and Da, the series-resistance deviation and, from the radiation solution, the same half-angle capture fraction. Each design row holds the 29 columns [eps, h, theta0, Da, Bi, p, k_S, J_shell, J_iso, J_sink, J(0), J(1), delta1, sup-norm of M, f(theta0), C, F_half of the sink, u at the axis probe, u at the pole probe, G of the sink, J_Da of the projected system, sup-norm of the naive matrix, J of the naive system, J_react, J_CK, delta_CK, k = k_S J_Da in nm^3/us, F_half of the radiation solution, J_Da / J_sink]. Row 0 is the head row [J_Da / J_sink of the first design, number of designs, order, sum of delta_CK over the designs, zeros]. Raise ValueError if design_table is empty or a row does not have seven entries, or if order is not a positive integer.

The head scalar, the fraction of its own diffusion-controlled capture rate that the partially reactive site of the first design actually achieves behind its shell, is the quantity a nanoreactor designer needs: it tells how much of the rate is lost to finite surface kinetics once the site is small, the cage is tight, the corona hinders diffusion near the core and the shell is only partly permeable, and it is not obtainable from the perfect-sink theory of the source, from Berg's isotropic factor or from the isotropic Collins-Kimball formula.

Returns
-------
A (1 + n_designs, 29) float64 array; row 0 is the head row and row i the row of design i, with the columns listed in the description.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nanoreactor_audit(design_table, order):
    """Run the complete chain for every nanoreactor design in design_table, a list of rows (R,
    R0, site_fraction, D, kappa, P, p) in nm, nm, -, nm^2/us, nm/us, nm/us and - with finite
    kappa (P may be inf), at the given truncation order, and return one table. A (1 +
    n_designs, 29) float64 array; row 0 is the head row and row i the row of design i, with
    the columns listed in the description."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def _oracle_nanoreactor_audit(design_table, order):
    rows = [tuple(float(v) for v in r) for r in design_table]
    if not rows or any(len(r) != 7 for r in rows):
        raise ValueError("design_table must hold rows (R, R0, site_fraction, D, kappa, permeability, hindrance)")
    n = int(order)
    if n < 1 or n != order:
        raise ValueError("order must be a positive integer")
    NC = 29
    out = np.zeros((1 + len(rows), NC))
    for i, (R, R0, phi, D, kappa, perm, p) in enumerate(rows, 1):
        par = _oracle_nanoreactor_parameters(R, R0, phi, D, kappa, perm, p)
        eps, h, t0, da, bi, pp, ks, j_shell, j_iso = par
        co = _oracle_dual_series_coefficients(n, eps, bi, pp)
        Q = _oracle_minkov_matrix(n, t0)
        X = _oracle_perfect_sink_solution(co[4], Q)
        summ = _oracle_rate_correction_summary(n, t0, eps, bi, pp)
        J, j0, j1, d1, mnorm, f_esf, coup = summ
        xi_mid = 0.5 * (1.0 + 1.0 / eps)
        loc = _oracle_local_fields(X, co, eps, [[xi_mid, 0.0], [xi_mid, np.pi]], 0.5 * t0)
        u_front, u_back, f_half, g_front = loc[0], loc[1], loc[2], loc[3]
        Xd = _oracle_partially_reactive_solution(co[4], t0, da)
        j_da = 0.5 * Xd[0]
        reg = _oracle_radiation_islae_regularity(co[4], Q, da)
        sr = _oracle_series_resistance_deviation(J, t0, da, j_da)
        loc_da = _oracle_local_fields(Xd, co, eps, [[xi_mid, 0.0]], 0.5 * t0)
        out[i] = [eps, h, t0, da, bi, pp, ks, j_shell, j_iso, J, j0, j1, d1, mnorm, f_esf, coup,
                  f_half, u_front, u_back, g_front, j_da, reg[0], reg[1], sr[0], sr[1], sr[2],
                  ks * j_da, loc_da[1], j_da / J]
    out[0, 0] = out[1, 28]
    out[0, 1] = len(rows)
    out[0, 2] = n
    out[0, 3] = float(np.sum(out[1:, 25]))
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ndesign_table = [(2.5, 4.0, 0.15, 500.0, 600.0, 500.0, 1.0)]\norder = 200\n',
         'call': 'nanoreactor_audit(design_table, order)',
         'gold_call': '_oracle_nanoreactor_audit(design_table, order)'},
        {'setup': 'import numpy as np\ndesign_table = [(1.4, 4.0, 0.10, 500.0, 250.0, np.inf, 0.0), (2.0, 4.2, 0.50, 500.0, 125.0, 125.0, 1.5)]\norder = 120\n',
         'call': 'nanoreactor_audit(design_table, order)',
         'gold_call': '_oracle_nanoreactor_audit(design_table, order)'},
        {'setup': 'import numpy as np\ndesign_table = [(3.0, 4.0, 0.30, 400.0, 1600.0, 800.0, 0.5)]\norder = 400\n',
         'call': 'nanoreactor_audit(design_table, order)',
         'gold_call': '_oracle_nanoreactor_audit(design_table, order)'},
    ]
