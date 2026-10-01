"""
Propagate the calibrated bending-plane ISR increment under a general dipole-field exponent.

The paper gives Delta epsilon_nx^ISR proportional to E L^4 B^5. With L proportional to E^{1/2} and B proportional to E^{-p}, the exponent is 3-5p.

Returns
-------
Return one nonnegative dimensionless float g_ISR.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

def incoherent_radiation_growth(energy_gev, exponent, reference_energy_gev,
                                reference_growth):
    """Return the dimensionless horizontal ISR emittance increment.

    Args:
        energy_gev: Positive beam energy E in GeV.
        exponent: Finite dipole scaling exponent p.
        reference_energy_gev: Positive reference energy E_r in GeV.
        reference_growth: Nonnegative dimensionless ISR increment at E_r.

    Returns:
        float: One finite nonnegative dimensionless scalar.

    Raises:
        ValueError: If an input is nonfinite, either energy is nonpositive,
            reference_growth is negative, or a nonzero result cannot be
            represented as a finite float.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_incoherent_radiation_growth(energy_gev, exponent, reference_energy_gev,
                                                reference_growth):
    import math
    energy_gev = float(energy_gev)
    exponent = float(exponent)
    reference_energy_gev = float(reference_energy_gev)
    reference_growth = float(reference_growth)
    if not all(math.isfinite(x) for x in (energy_gev, exponent, reference_energy_gev, reference_growth)):
        raise ValueError("inputs must be finite")
    if min(energy_gev, reference_energy_gev) <= 0 or reference_growth < 0:
        raise ValueError("energies are positive and growth is nonnegative")
    if reference_growth == 0.0:
        return 0.0
    log_result = (math.log(reference_growth) +
                  (3.0 - 5.0 * exponent) * math.log(energy_gev / reference_energy_gev))
    if log_result > math.log(float.fromhex('0x1.fffffffffffffp+1023')):
        raise ValueError("result exceeds the supported finite domain")
    if log_result < math.log(float.fromhex('0x0.0000000000001p-1022')):
        return 0.0
    result = math.exp(log_result)
    if not math.isfinite(result):
        raise ValueError("result exceeds the supported finite domain")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"", "call":"incoherent_radiation_growth(50.,.6,50.,.04)", "gold_call":"_oracle_incoherent_radiation_growth(50.,.6,50.,.04)"},
        {"setup":"", "call":"incoherent_radiation_growth(5000.,.6,50.,.04)", "gold_call":"_oracle_incoherent_radiation_growth(5000.,.6,50.,.04)"},
        {"setup":"", "call":"incoherent_radiation_growth(500.,0.,50.,.025)", "gold_call":"_oracle_incoherent_radiation_growth(500.,0.,50.,.025)"},
        {"setup":"", "call":"incoherent_radiation_growth(5.,.3,50.,.011)", "gold_call":"_oracle_incoherent_radiation_growth(5.,.3,50.,.011)"},
        {"setup":"", "call":"incoherent_radiation_growth(900.,.8,30.,.07)", "gold_call":"_oracle_incoherent_radiation_growth(900.,.8,30.,.07)"},
        {"setup":"", "call":"incoherent_radiation_growth(123.,.47,123.,0.)", "gold_call":"_oracle_incoherent_radiation_growth(123.,.47,123.,0.)"},
        {"setup":"", "call":"incoherent_radiation_growth(1e4,1/3,25.,.0035)", "gold_call":"_oracle_incoherent_radiation_growth(1e4,1/3,25.,.0035)"},
        {"setup":"", "call":"incoherent_radiation_growth(500.,-1000.,50.,0.)", "gold_call":"_oracle_incoherent_radiation_growth(500.,-1000.,50.,0.)"}
    ]
