"""
Recover and implement the paper's adaptive effective-sampling selection across its three physical regimes.

Inputs Ra and Rb are intensity reflectivities. Follow Algorithms 1--3 of the cited main paper, including its storage-memory cutoff, sub-history convention, inverse-curve discretization, strict comparisons, and positive half-tie rule. For the low-rate branch, use Nmax=ceil(5/abs(ln(sqrt(Ra*Rb)))), the least integer not below five field-amplitude lifetimes. Return columns in the signature's documented order.

Returns
-------
Return a length-6 NumPy float array in exactly the documented order; N and n_subhistories are represented as exact integer-valued floats.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def adaptive_sampling_plan(desired_hz, length_m, reflectivity_a, reflectivity_b, c_m_s=299792458.0):
    """Return [f_calc, theta, N, n_subhistories, partial_flag, accuracy]: effective sampling rate f_calc in Hz, time step theta = 1/f_calc in s, round trips N per integration step, number of interleaved sub-histories (1 when none), partial_flag 1.0 in the capped partial-update branch and 0.0 otherwise, and accuracy = 1 - |f_calc - desired_hz|/desired_hz."""
    return np.zeros(6, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_adaptive_sampling_plan(desired_hz, length_m, reflectivity_a, reflectivity_b, c_m_s=299792458.0):
    """Paper Algorithms 1--3 with explicitly frozen integer conventions."""
    import math
    import numpy as np
    desired_hz = float(desired_hz)
    length_m = float(length_m)
    ra = float(reflectivity_a)
    rb = float(reflectivity_b)
    c_m_s = float(c_m_s)
    if not np.all(np.isfinite(np.asarray(
            [desired_hz, length_m, ra, rb, c_m_s], dtype=float))):
        raise ValueError("inputs must be finite")
    if desired_hz <= 0 or length_m <= 0 or c_m_s <= 0:
        raise ValueError("frequencies, length, and c must be positive")
    if not (0 < ra < 1 and 0 < rb < 1):
        raise ValueError("intensity reflectivities must lie in (0,1)")
    f_2t = c_m_s / (2.0 * length_m)
    q = math.sqrt(ra * rb)
    n_eff = 1.0 / abs(math.log(q))
    n_max = int(math.ceil(5.0 * n_eff))
    eta = f_2t / desired_hz
    n_roundtrips = 1
    n_subhistories = 1
    partial = 0.0
    if desired_hz > f_2t:
        n_subhistories = max(1, int(math.floor(1.0 / eta + 0.5)))
        f_calc = f_2t * n_subhistories
    elif desired_hz < f_2t / n_max:
        n_roundtrips = n_max
        f_calc = desired_hz
        partial = 1.0
    else:
        k0 = int(math.floor(eta))
        boundary = 2.0 * k0 * (k0 + 1.0) / (2.0 * k0 + 1.0)
        n_roundtrips = k0 if eta < boundary else k0 + 1
        f_calc = f_2t / n_roundtrips
    accuracy = 1.0 - abs(f_calc - desired_hz) / desired_hz
    return np.array([f_calc, 1.0 / f_calc, float(n_roundtrips),
                     float(n_subhistories), partial, accuracy], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"", "call":"adaptive_sampling_plan(360000.,2.75,0.985,0.9995)", "gold_call":"_oracle_adaptive_sampling_plan(360000.,2.75,0.985,0.9995)"},
        {"setup":"", "call":"adaptive_sampling_plan(1.5*299792458./(2*2.75),2.75,0.985,0.9995)", "gold_call":"_oracle_adaptive_sampling_plan(1.5*299792458./(2*2.75),2.75,0.985,0.9995)"},
        {"setup":"", "call":"adaptive_sampling_plan(50000.,2.75,0.985,0.9995)", "gold_call":"_oracle_adaptive_sampling_plan(50000.,2.75,0.985,0.9995)"},
        {"setup":"", "call":"adaptive_sampling_plan(299792458./(2*2.75),2.75,0.985,0.9995)", "gold_call":"_oracle_adaptive_sampling_plan(299792458./(2*2.75),2.75,0.985,0.9995)"},
        {"setup":"", "call":"adaptive_sampling_plan(2.5*299792458./(2*2.75),2.75,0.985,0.9995)", "gold_call":"_oracle_adaptive_sampling_plan(2.5*299792458./(2*2.75),2.75,0.985,0.9995)"},
        {"setup":"", "call":"adaptive_sampling_plan((299792458./(2*2.75))/(220./21.+1e-6),2.75,0.985,0.9995)", "gold_call":"_oracle_adaptive_sampling_plan((299792458./(2*2.75))/(220./21.+1e-6),2.75,0.985,0.9995)"},
        {"setup":"", "call":"adaptive_sampling_plan((299792458./(2*2.75))/(220./21.-1e-6),2.75,0.985,0.9995)", "gold_call":"_oracle_adaptive_sampling_plan((299792458./(2*2.75))/(220./21.-1e-6),2.75,0.985,0.9995)"},
        {"setup":"", "call":"adaptive_sampling_plan(100000.,10.,0.999,0.9999)", "gold_call":"_oracle_adaptive_sampling_plan(100000.,10.,0.999,0.9999)"}
    ]
