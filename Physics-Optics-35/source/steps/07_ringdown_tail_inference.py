"""
Infer incident power from the paper's nonadiabatic continuum tail on full and retained grids, then quantify sampling bias.

For positive v in the paper's nonadiabatic regime, Eqs. (5)--(6), specialized to unit incident field, give g(t)=t_a*(sqrt(i*pi/(2*k*v*T))*exp(i*T/(2*k*v*tau**2))*exp(-t/tau-i*k*v*t**2/(2*T)) + 1/(1-q*exp(-2*i*k*v*t))). Use T=L/c, tau=2*T/abs(ln(q)), k=2*pi/lambda, q=sqrt(Ra*Rb), and t_a=sqrt(1-Ra). For this task's author-defined signed extension, admit |v|>=v_cr, retain signed v in the same analytic expression, and use NumPy's principal complex square root for v<0. Use the supplied fixed tail and fit alpha=<g,E>/<g,g> independently on the full tail (all samples in the window) and the retained tail (indices 0,stride,2*stride,... of the whole trace inside the window).

Returns
-------
Return one length-4 NumPy float array in the documented order.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def ringdown_tail_inference(time_s, field, stride, velocity_m_s,
                            length_m, wavelength_m, reflectivity_a,
                            reflectivity_b, finesse,
                            tail_start_storage=4.0,
                            tail_end_storage=24.0,
                            c_m_s=299792458.0):
    """Return [P_full_fit,P_retained_fit,relative_bias,n_retained_tail]."""
    return np.zeros(4, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_ringdown_tail_inference(time_s, field, stride, velocity_m_s,
                                    length_m, wavelength_m, reflectivity_a,
                                    reflectivity_b, finesse,
                                    tail_start_storage=4.0,
                                    tail_end_storage=24.0,
                                    c_m_s=299792458.0):
    """Tail-only complex LS incident-power bias using paper Eqs. (5)--(6)."""
    import math
    import numpy as np
    time_s = np.asarray(time_s, dtype=float)
    field = np.asarray(field, dtype=complex)
    stride_i = int(stride)
    if (time_s.ndim != 1 or field.shape != time_s.shape or time_s.size < 5
            or np.any(np.diff(time_s) <= 0) or not np.all(np.isfinite(time_s))
            or not np.all(np.isfinite(field))):
        raise ValueError("time and field must be finite aligned vectors")
    if stride_i != stride or stride_i < 1:
        raise ValueError("stride must be a positive integer")
    if (length_m <= 0 or wavelength_m <= 0 or finesse <= 0 or c_m_s <= 0
            or not (0 < reflectivity_a < 1 and 0 < reflectivity_b < 1)):
        raise ValueError("invalid optical parameters")
    if not (0 < tail_start_storage < tail_end_storage):
        raise ValueError("tail bounds must be positive and ordered")
    v = float(velocity_m_s)
    v_critical = wavelength_m * math.pi * c_m_s / (4.0 * length_m * finesse**2)
    if not np.isfinite(v) or abs(v) < v_critical * (1.0 - 1e-12):
        raise ValueError("continuum tail fit requires |v| >= v_cr")
    T = float(length_m) / float(c_m_s)
    tau = 2.0 * T / abs(math.log(math.sqrt(float(reflectivity_a) * float(reflectivity_b))))
    k = 2.0 * math.pi / float(wavelength_m)
    q = math.sqrt(float(reflectivity_a) * float(reflectivity_b))
    t_a = math.sqrt(1.0 - float(reflectivity_a))

    def unit_input_kernel(t):
        transient_prefactor = np.sqrt(1j * math.pi / (2.0 * k * v * T))
        transient_prefactor *= np.exp(1j * T / (2.0 * k * v * tau**2))
        transient = transient_prefactor * np.exp(
            -t / tau - 1j * k * v * t**2 / (2.0 * T)
        )
        adiabatic = 1.0 / (1.0 - q * np.exp(-2j * k * v * t))
        return t_a * (transient + adiabatic)

    full_index = np.where((time_s >= tail_start_storage * tau)
                          & (time_s <= tail_end_storage * tau))[0]
    retained_index = np.arange(0, time_s.size, stride_i, dtype=int)
    retained_index = retained_index[
        (time_s[retained_index] >= tail_start_storage * tau)
        & (time_s[retained_index] <= tail_end_storage * tau)
    ]
    if full_index.size < 16 or retained_index.size < 3:
        raise ValueError("insufficient samples in the continuum-valid tail")
    g_full = unit_input_kernel(time_s[full_index])
    g_retained = unit_input_kernel(time_s[retained_index])
    alpha_full = np.vdot(g_full, field[full_index]) / np.vdot(g_full, g_full)
    alpha_retained = (np.vdot(g_retained, field[retained_index])
                      / np.vdot(g_retained, g_retained))
    power_full = float(abs(alpha_full)**2)
    power_retained = float(abs(alpha_retained)**2)
    if power_full <= 0 or not np.isfinite(power_full + power_retained):
        raise ValueError("tail fit produced invalid incident power")
    relative_power_bias = abs(power_retained - power_full) / power_full
    return np.array([power_full, power_retained, relative_power_bias,
                     float(retained_index.size)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nt=np.linspace(-1e-4,1e-4,4001); e=np.exp(-t/3e-5)*np.exp(1j*2e4*t)","call":"ringdown_tail_inference(t,e,5,7e-4,2.75,1064e-9,.985,.9995,400.)","gold_call":"_oracle_ringdown_tail_inference(t,e,5,7e-4,2.75,1064e-9,.985,.9995,400.)"},
        {"setup":"import numpy as np\nt=np.linspace(-1.2e-4,1.2e-4,5001); e=(1+.1*np.cos(2e4*t))*np.exp(-1j*3e4*t)","call":"ringdown_tail_inference(t,e,7,-8e-4,2.75,1064e-9,.982,.999,380.)","gold_call":"_oracle_ringdown_tail_inference(t,e,7,-8e-4,2.75,1064e-9,.982,.999,380.)"},
        {"setup":"import numpy as np\nt=np.linspace(-8e-5,8e-5,6001); e=(.8+.2j)*np.ones(t.size)","call":"ringdown_tail_inference(t,e,3,1.2e-2,1.5,780e-9,.97,.998,300.)","gold_call":"_oracle_ringdown_tail_inference(t,e,3,1.2e-2,1.5,780e-9,.97,.998,300.)"},
        {"setup":"import numpy as np\nt=np.linspace(-2e-4,2e-4,8001); e=np.exp(-abs(t)/7e-5)*(1+.4j)","call":"ringdown_tail_inference(t,e,11,-5e-4,3.,1550e-9,.99,.999,500.)","gold_call":"_oracle_ringdown_tail_inference(t,e,11,-5e-4,3.,1550e-9,.99,.999,500.)"},
        {"setup":"import numpy as np\nt=np.linspace(-1e-4,1e-4,4501); e=np.exp(1j*(2e4*t+4e8*t*t))","call":"ringdown_tail_inference(t,e,9,9e-3,2.,1064e-9,.98,.999,350.,5.,20.)","gold_call":"_oracle_ringdown_tail_inference(t,e,9,9e-3,2.,1064e-9,.98,.999,350.,5.,20.)"},
        {"setup":"import numpy as np\nt=np.linspace(-1.5e-4,1.5e-4,7001); e=(1+t/1e-3)*np.exp(-1j*5e4*t)","call":"ringdown_tail_inference(t,e,13,-1.1e-2,2.5,1550e-9,.975,.997,320.,3.,18.)","gold_call":"_oracle_ringdown_tail_inference(t,e,13,-1.1e-2,2.5,1550e-9,.975,.997,320.,3.,18.)"},
        {"setup":"import numpy as np\nt=np.linspace(-9e-5,9e-5,5501); e=(.7-.3j)*np.exp(-t/8e-5)","call":"ringdown_tail_inference(t,e,4,1.5e-3,1.8,532e-9,.96,.996,280.)","gold_call":"_oracle_ringdown_tail_inference(t,e,4,1.5e-3,1.8,532e-9,.96,.996,280.)"},
    ]
