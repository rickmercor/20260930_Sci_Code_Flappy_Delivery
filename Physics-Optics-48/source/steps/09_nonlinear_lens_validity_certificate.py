"""
Construct an exact Appendix-D/E validity certificate from Gaussian moments and correlated lens offsets.

Use the fixed electron rest-energy convention 0.000511 GeV, so gamma=E/0.000511. Call transport_nonlinear_lens_pair at tau_x and tau_x/2 on tensor-product physicists' Gauss-Hermite nodes sqrt(2)sigma u_i with weights w_i w_j/pi, and call nonlinear_geometric_growth for the asymptotic pair. With eps=eps_n/gamma and alpha=0, use sigma_position^2=eps beta and sigma_slope^2=eps/beta. Add the independent initial positional variance to the centered mapped-position variance, form the centered 2x2 covariance determinant with the exact mapped angle, and define g_exact=sqrt(max[(eps_out/eps)^2-1,0]). Define n=log2[s(tau_x)/s(tau_x/2)] from centered rms final positional residuals. For alignment, direct composition of the paper's preceding drift-kick equations gives M=(1/L+1/l)[[-L,L],[-(1+2l/L),1]], resolving the inconsistent printed simplification in Eq. (E5). Use Sigma_Delta=sigma_Delta^2[[1,rho],[rho,1]] and <J_x>=gamma[(M Sigma_Delta M^T)_00/beta+beta(M Sigma_Delta M^T)_11]/2. Return the equal-rms tolerance in units of matched sigma_x. Exact map, asymptotic model, cancellation order and alignment action must remain separate. To keep the cancellation-order probe numerically resolved, the supported taper domain is 1e-12 <= |tau_x| max(L,l,beta) <= 1e6; finite inputs outside it must be refused with ValueError.

Returns
-------
Return [g_x_exact,g_y_exact,g_x_asym,g_y_asym,n_x,n_y,action_ratio,tolerance_in_sigma_x] as a length-8 NumPy array.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def nonlinear_lens_validity_certificate(energy_gev, length_m, half_gap_m, beta_m,
                                        tau_per_m, eps_nx_m_rad, eps_ny_m_rad,
                                        offset_fraction, offset_correlation,
                                        quadrature_order):
    """Return exact-map, asymptotic, cancellation and alignment diagnostics.

    Args:
        energy_gev: Positive energy in GeV.
        length_m, half_gap_m, beta_m: Positive lengths L,l,beta in metres.
        tau_per_m: Finite taper tau_x in inverse metres satisfying
            1e-12 <= abs(tau_x)*max(L,l,beta) <= 1e6.
        eps_nx_m_rad, eps_ny_m_rad: Positive normalized emittances in m rad.
        offset_fraction: Positive rms lens offset in units of matched sigma_x.
        offset_correlation: Correlation rho of the two equal-rms lens offsets.
        quadrature_order: Odd integer from 5 through 15 inclusive.

    Returns:
        numpy.ndarray: Length-8 float array
        [g_x_exact,g_y_exact,g_x_asym,g_y_asym,n_x,n_y,<J_x>/eps_nx,
         sigma_Delta_max/sigma_x].

    Raises:
        ValueError: If an input is nonfinite; a physical scale, emittance, or
            offset_fraction is nonpositive; rho is outside [-1,1];
            quadrature_order is not a finite odd integer from 5 through 15;
            the taper fails the documented resolved-probe range; or any map,
            moment, action, or returned diagnostic is not finite and resolved.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_nonlinear_lens_validity_certificate(energy_gev, length_m, half_gap_m, beta_m,
                                                        tau_per_m, eps_nx_m_rad, eps_ny_m_rad,
                                                        offset_fraction, offset_correlation,
                                                        quadrature_order):
    import math
    import numpy as np
    vals = [float(x) for x in (energy_gev, length_m, half_gap_m, beta_m,
                               tau_per_m, eps_nx_m_rad, eps_ny_m_rad,
                               offset_fraction, offset_correlation)]
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("inputs must be finite")
    (energy_gev, length_m, half_gap_m, beta_m, tau_per_m,
     eps_nx_m_rad, eps_ny_m_rad, offset_fraction, offset_correlation) = vals
    order_float = float(quadrature_order)
    if not math.isfinite(order_float):
        raise ValueError("quadrature_order must be finite")
    order = int(order_float)
    if order_float != order or order < 5 or order > 15 or order % 2 == 0:
        raise ValueError("quadrature_order must be an odd integer from 5 through 15")
    if min(energy_gev, length_m, half_gap_m, beta_m,
           eps_nx_m_rad, eps_ny_m_rad, offset_fraction) <= 0 or tau_per_m == 0:
        raise ValueError("physical scales and offset_fraction are positive; tau is nonzero")
    if offset_correlation < -1.0 or offset_correlation > 1.0:
        raise ValueError("offset correlation must lie in [-1,1]")
    taper_scale = abs(tau_per_m) * max(length_m, half_gap_m, beta_m)
    if taper_scale < 1e-12 or taper_scale > 1e6:
        raise ValueError("taper is outside the resolved finite probe range")
    gamma = energy_gev / 0.000511
    eps_x = eps_nx_m_rad / gamma
    eps_y = eps_ny_m_rad / gamma
    sigma_xp = math.sqrt(eps_x / beta_m)
    sigma_yp = math.sqrt(eps_y / beta_m)
    nodes, one_weights = np.polynomial.hermite.hermgauss(order)
    xp = (math.sqrt(2.0) * sigma_xp * nodes[:, None] +
          np.zeros((order, order), dtype=float))
    yp = (math.sqrt(2.0) * sigma_yp * nodes[None, :] +
          np.zeros((order, order), dtype=float))
    rays = np.column_stack((xp.ravel(), yp.ravel()))
    weights = np.outer(one_weights, one_weights).ravel() / math.pi

    def mapped(tau):
        return _oracle_transport_nonlinear_lens_pair(
            rays, length_m, half_gap_m, tau)

    full = mapped(tau_per_m)
    half = mapped(0.5 * tau_per_m)

    def centered_moment(a, b=None):
        a = np.asarray(a, dtype=float)
        ac = a - np.dot(weights, a)
        if b is None:
            return float(np.dot(weights, ac * ac))
        b = np.asarray(b, dtype=float)
        bc = b - np.dot(weights, b)
        return float(np.dot(weights, ac * bc))

    def exact_growth(pos, angle, eps_geo):
        var_pos = eps_geo * beta_m + centered_moment(pos)
        var_angle = centered_moment(angle)
        cov = centered_moment(pos, angle)
        determinant = max(var_pos * var_angle - cov * cov, 0.0)
        ratio_sq = determinant / (eps_geo * eps_geo)
        return math.sqrt(max(ratio_sq - 1.0, 0.0))

    gx_exact = exact_growth(full[:, 0], full[:, 1], eps_x)
    gy_exact = exact_growth(full[:, 2], full[:, 3], eps_y)
    asym = _oracle_nonlinear_geometric_growth(
        energy_gev, length_m, half_gap_m, beta_m, tau_per_m,
        eps_nx_m_rad, eps_ny_m_rad)
    sx_full = math.sqrt(centered_moment(full[:, 0]))
    sx_half = math.sqrt(centered_moment(half[:, 0]))
    sy_full = math.sqrt(centered_moment(full[:, 2]))
    sy_half = math.sqrt(centered_moment(half[:, 2]))
    if (not all(math.isfinite(x) for x in (sx_full, sx_half, sy_full, sy_half))
            or min(sx_full, sx_half, sy_full, sy_half) <= 0):
        raise ValueError("degenerate cancellation-order probe")
    nx = math.log(sx_full / sx_half, 2.0)
    ny = math.log(sy_full / sy_half, 2.0)

    sigma_x = math.sqrt(eps_x * beta_m)
    sigma_offset = offset_fraction * sigma_x
    factor = 1.0 / length_m + 1.0 / half_gap_m
    matrix = factor * np.array([
        [-length_m, length_m],
        [-(1.0 + 2.0 * half_gap_m / length_m), 1.0]], dtype=float)
    offset_cov = sigma_offset ** 2 * np.array(
        [[1.0, offset_correlation], [offset_correlation, 1.0]], dtype=float)
    final_cov = matrix @ offset_cov @ matrix.T
    mean_action = 0.5 * gamma * (final_cov[0, 0] / beta_m +
                                 beta_m * final_cov[1, 1])
    action_ratio = mean_action / eps_nx_m_rad
    if not math.isfinite(action_ratio) or action_ratio <= 0:
        raise ValueError("degenerate offset-action probe")
    tolerance_sigma = offset_fraction / math.sqrt(action_ratio)
    out = np.array([gx_exact, gy_exact, asym[0], asym[1], nx, ny,
                    action_ratio, tolerance_sigma], dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError("diagnostics exceed the supported finite domain")
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"", "call":"nonlinear_lens_validity_certificate(50.,5**.5,2*5**.5,.015*5**.5,-61.42,8e-6,.1e-6,.2,-.35,7)", "gold_call":"_oracle_nonlinear_lens_validity_certificate(50.,5**.5,2*5**.5,.015*5**.5,-61.42,8e-6,.1e-6,.2,-.35,7)"},
        {"setup":"", "call":"nonlinear_lens_validity_certificate(800.,8.94427191,17.88854382,.1341640786,-232.698,12e-6,.18e-6,.15,.4,9)", "gold_call":"_oracle_nonlinear_lens_validity_certificate(800.,8.94427191,17.88854382,.1341640786,-232.698,12e-6,.18e-6,.15,.4,9)"},
        {"setup":"", "call":"nonlinear_lens_validity_certificate(120.,2.6,4.1,.052,-90.,4e-6,4e-6,.3,0.,5)", "gold_call":"_oracle_nonlinear_lens_validity_certificate(120.,2.6,4.1,.052,-90.,4e-6,4e-6,.3,0.,5)"},
        {"setup":"", "call":"nonlinear_lens_validity_certificate(300.,4.2,9.7,.063,-155.,9e-6,.25e-6,.1,1.,11)", "gold_call":"_oracle_nonlinear_lens_validity_certificate(300.,4.2,9.7,.063,-155.,9e-6,.25e-6,.1,1.,11)"},
        {"setup":"", "call":"nonlinear_lens_validity_certificate(300.,4.2,9.7,.063,-155.,9e-6,.25e-6,.1,-1.,11)", "gold_call":"_oracle_nonlinear_lens_validity_certificate(300.,4.2,9.7,.063,-155.,9e-6,.25e-6,.1,-1.,11)"},
        {"setup":"", "call":"nonlinear_lens_validity_certificate(75.,1.1,3.8,.22,28.,2e-6,13e-6,.42,-.7,13)", "gold_call":"_oracle_nonlinear_lens_validity_certificate(75.,1.1,3.8,.22,28.,2e-6,13e-6,.42,-.7,13)"},
        {"setup":"", "call":"nonlinear_lens_validity_certificate(2500.,15.,27.,.31,430.,15e-6,.07e-6,.08,.65,15)", "gold_call":"_oracle_nonlinear_lens_validity_certificate(2500.,15.,27.,.31,430.,15e-6,.07e-6,.08,.65,15)"},
        {"setup":"def _ve(fn):\n try: fn()\n except ValueError: return 1.0\n except Exception: return -1.0\n return 0.0", "call":"_ve(lambda: nonlinear_lens_validity_certificate(50.,5**.5,2*5**.5,.015*5**.5,-61.42,8e-6,.1e-6,.2,-.35,float('inf')))", "gold_call":"_ve(lambda: _oracle_nonlinear_lens_validity_certificate(50.,5**.5,2*5**.5,.015*5**.5,-61.42,8e-6,.1e-6,.2,-.35,float('inf')))"},
        {"setup":"def _ve(fn):\n try: fn()\n except ValueError: return 1.0\n except Exception: return -1.0\n return 0.0", "call":"_ve(lambda: nonlinear_lens_validity_certificate(50.,5**.5,2*5**.5,.015*5**.5,1e-20,8e-6,.1e-6,.2,-.35,7))", "gold_call":"_ve(lambda: _oracle_nonlinear_lens_validity_certificate(50.,5**.5,2*5**.5,.015*5**.5,1e-20,8e-6,.1e-6,.2,-.35,7))"}
    ]
