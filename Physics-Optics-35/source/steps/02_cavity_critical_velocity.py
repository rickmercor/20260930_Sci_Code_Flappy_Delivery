"""
Compute the resonance-crossing critical mirror speed from the storage-time criterion in Eq. (4).

Use SI units throughout: wavelength and cavity length in metres, c in m/s, and finesse dimensionless.

Returns
-------
Return one Python float in m/s.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cavity_critical_velocity(wavelength_m, length_m, finesse, c_m_s=299792458.0):
    """Return the positive critical velocity in m/s."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_cavity_critical_velocity(wavelength_m, length_m, finesse, c_m_s=299792458.0):
    """Critical speed from Eq. (4) of the attached paper."""
    import math
    wavelength_m = float(wavelength_m)
    length_m = float(length_m)
    finesse = float(finesse)
    c_m_s = float(c_m_s)
    if not all(math.isfinite(value) for value in
               (wavelength_m, length_m, finesse, c_m_s)):
        raise ValueError("all physical inputs must be finite")
    if wavelength_m <= 0 or length_m <= 0 or finesse <= 0 or c_m_s <= 0:
        raise ValueError("all physical inputs must be positive")
    return float(wavelength_m * math.pi * c_m_s / (4.0 * length_m * finesse**2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"", "call":"cavity_critical_velocity(1064e-9,2.75,400.)", "gold_call":"_oracle_cavity_critical_velocity(1064e-9,2.75,400.)"},
        {"setup":"", "call":"cavity_critical_velocity(1550e-9,1.0,1000.)", "gold_call":"_oracle_cavity_critical_velocity(1550e-9,1.0,1000.)"},
        {"setup":"", "call":"cavity_critical_velocity(532e-9,5.5,200.)", "gold_call":"_oracle_cavity_critical_velocity(532e-9,5.5,200.)"},
        {"setup":"", "call":"cavity_critical_velocity(1064e-9,3000.,450.)", "gold_call":"_oracle_cavity_critical_velocity(1064e-9,3000.,450.)"},
        {"setup":"", "call":"cavity_critical_velocity(780e-9,0.12,10000.)", "gold_call":"_oracle_cavity_critical_velocity(780e-9,0.12,10000.)"},
        {"setup":"", "call":"cavity_critical_velocity(1064e-9,2.75,800.)", "gold_call":"_oracle_cavity_critical_velocity(1064e-9,2.75,800.)"},
        {"setup":"", "call":"cavity_critical_velocity(2128e-9,5.5,400.)", "gold_call":"_oracle_cavity_critical_velocity(2128e-9,5.5,400.)"}
    ]
