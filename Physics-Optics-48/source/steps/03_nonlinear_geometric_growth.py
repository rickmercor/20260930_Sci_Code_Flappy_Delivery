"""
Evaluate the residual O(tau_x^2) normalized-emittance increments after the paper's -I cancellation.

Use gamma=E/(0.000511 GeV) and C=tau_x^2 L^3(1+L/l)(l/L+1)/(beta^2 gamma). Equations (29)-(30) give g_x=C sqrt(6 eps_nx^2+18 eps_ny^2) and g_y=C sqrt(18 eps_nx^2+6 eps_ny^2). E is in GeV, L,l,beta in m, tau_x in m^-1, normalized emittances in m rad, and both returned growths are dimensionless in [x,y] order.

Returns
-------
Return a length-2 NumPy float array [g_x,g_y], both dimensionless.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def nonlinear_geometric_growth(energy_gev, length_m, half_gap_m, beta_m,
                               tau_per_m, eps_nx_m_rad, eps_ny_m_rad):
    """Return residual geometric increments [g_x,g_y].

    Args:
        energy_gev: Positive beam energy in GeV.
        length_m: Positive lattice length L in metres.
        half_gap_m: Positive half-cell gap l in metres.
        beta_m: Positive beta function in metres.
        tau_per_m: Finite nonlinear-lens taper tau_x in inverse metres.
        eps_nx_m_rad: Positive horizontal normalized emittance in m rad.
        eps_ny_m_rad: Positive vertical normalized emittance in m rad.

    Returns:
        numpy.ndarray: Length-2 nonnegative float array [g_x,g_y], dimensionless.

    Raises:
        ValueError: If an input is nonfinite; energy, a length, or an emittance
            is nonpositive; or either result is outside the finite supported
            numerical domain.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_nonlinear_geometric_growth(energy_gev, length_m, half_gap_m, beta_m,
                                               tau_per_m, eps_nx_m_rad, eps_ny_m_rad):
    import math
    import numpy as np
    vals = [float(x) for x in (energy_gev, length_m, half_gap_m, beta_m,
                               tau_per_m, eps_nx_m_rad, eps_ny_m_rad)]
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("inputs must be finite")
    energy_gev, length_m, half_gap_m, beta_m, tau_per_m, eps_nx_m_rad, eps_ny_m_rad = vals
    if min(energy_gev, length_m, half_gap_m, beta_m, eps_nx_m_rad, eps_ny_m_rad) <= 0:
        raise ValueError("energy, lengths and emittances must be positive")
    gamma = energy_gev / 0.000511
    try:
        coeff = (tau_per_m ** 2 * length_m ** 3 /
                 (beta_m ** 2 * gamma) * (1.0 + length_m / half_gap_m) *
                 (half_gap_m / length_m + 1.0))
        gx = coeff * math.hypot(math.sqrt(6.0) * eps_nx_m_rad,
                                math.sqrt(18.0) * eps_ny_m_rad)
        gy = coeff * math.hypot(math.sqrt(18.0) * eps_nx_m_rad,
                                math.sqrt(6.0) * eps_ny_m_rad)
    except (OverflowError, ZeroDivisionError):
        raise ValueError("result exceeds the supported finite domain")
    out = np.array([gx, gy], dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError("result exceeds the supported finite domain")
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"", "call":"nonlinear_geometric_growth(50.,5**.5,2*5**.5,.015*5**.5,-61.42,8e-6,.1e-6)", "gold_call":"_oracle_nonlinear_geometric_growth(50.,5**.5,2*5**.5,.015*5**.5,-61.42,8e-6,.1e-6)"},
        {"setup":"", "call":"nonlinear_geometric_growth(80.,1.2,2.4,.04,0.,5e-6,5e-6)", "gold_call":"_oracle_nonlinear_geometric_growth(80.,1.2,2.4,.04,0.,5e-6,5e-6)"},
        {"setup":"", "call":"nonlinear_geometric_growth(500.,3.1,4.7,.06,180.,2e-6,14e-6)", "gold_call":"_oracle_nonlinear_geometric_growth(500.,3.1,4.7,.06,180.,2e-6,14e-6)"},
        {"setup":"", "call":"nonlinear_geometric_growth(500.,3.1,4.7,.06,-180.,2e-6,14e-6)", "gold_call":"_oracle_nonlinear_geometric_growth(500.,3.1,4.7,.06,-180.,2e-6,14e-6)"},
        {"setup":"", "call":"nonlinear_geometric_growth(37.,.9,8.2,.013,22.,.08e-6,16e-6)", "gold_call":"_oracle_nonlinear_geometric_growth(37.,.9,8.2,.013,22.,.08e-6,16e-6)"},
        {"setup":"", "call":"nonlinear_geometric_growth(3200.,14.,28.,.21,480.,12e-6,.18e-6)", "gold_call":"_oracle_nonlinear_geometric_growth(3200.,14.,28.,.21,480.,12e-6,.18e-6)"},
        {"setup":"", "call":"nonlinear_geometric_growth(125.,2.3,.7,.4,75.,3e-6,3e-6)", "gold_call":"_oracle_nonlinear_geometric_growth(125.,2.3,.7,.4,75.,3e-6,3e-6)"},
        {"setup":"def _ve(fn):\n try: fn()\n except ValueError: return 1.0\n except Exception: return -1.0\n return 0.0", "call":"_ve(lambda: nonlinear_geometric_growth(0.,2.,4.,.03,10.,1e-6,1e-6))", "gold_call":"_ve(lambda: _oracle_nonlinear_geometric_growth(0.,2.,4.,.03,10.,1e-6,1e-6))"}
    ]
