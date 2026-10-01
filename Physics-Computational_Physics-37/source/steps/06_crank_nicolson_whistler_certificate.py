"""
Certify the modal Crank--Nicolson source update against the continuous whistler mode.

Let omega = omega_real + i*omega_imag. The requested continuous horizon is



duration = periods * 2*pi / omega_real.



Use the smallest whole number of time steps spanning that horizon:



N = ceil(duration / time_step),

t = N * time_step.



For the modal equation u_t = -i*omega*u, the Crank-Nicolson amplification factor is



g = (1 - i*omega*time_step/2) / (1 + i*omega*time_step/2).



Starting from unit amplitude, compute



u_CN = g^N,

u_exact = exp(-i*omega*t).



The diagnostics are



amplitude_error = abs(abs(u_CN)/abs(u_exact) - 1),

phase_error = abs(principal_angle(u_CN/u_exact)),

log_error = abs(log(abs(u_CN)) - log(abs(u_exact))),

absolute_mode = abs(u_CN).



Return [amplitude_error, phase_error, log_error, N, t, absolute_mode], with N represented numerically in the returned real array.

Returns
-------
Return one length-6 real NumPy array in documented order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def crank_nicolson_whistler_certificate(omega_real, omega_imag,
                                        time_step, periods=1.0):
    """Compare a centered modal update with continuous propagation.

    Parameters
    ----------
    omega_real : float
        Positive real modal frequency.
    omega_imag : float
        Nonpositive imaginary modal frequency.
    time_step : float
        Positive update size.
    periods : float
        Positive comparison horizon in periods.
    Returns
    -------
    ndarray, shape (6,)
        [amplitude_error,phase_error,log_error,steps,time,absolute_mode].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_crank_nicolson_whistler_certificate(omega_real, omega_imag,
                                                time_step, periods=1.0):
    """Compare the paper's Crank--Nicolson source update with one whistler mode."""
    import math
    import numpy as np
    wr = float(omega_real)
    wi = float(omega_imag)
    tau = float(time_step)
    periods = float(periods)
    if not all(math.isfinite(x) for x in (wr, wi, tau, periods)):
        raise ValueError("inputs must be finite")
    if wr <= 0 or wi > 0 or tau <= 0 or periods <= 0:
        raise ValueError("invalid modal or time-step domain")
    omega = complex(wr, wi)
    duration = periods * 2.0 * math.pi / wr
    steps = int(math.ceil(duration / tau))
    elapsed = steps * tau
    factor = (1.0 - 0.5j * omega * tau) / (1.0 + 0.5j * omega * tau)
    numerical = factor ** steps
    exact = np.exp(-1j * omega * elapsed)
    amplitude_error = abs(abs(numerical) / abs(exact) - 1.0)
    phase_error = abs(float(np.angle(numerical / exact)))
    log_damping_error = abs(math.log(abs(numerical)) - math.log(abs(exact)))
    return np.array([amplitude_error, phase_error, log_damping_error,
                     float(steps), elapsed, abs(numerical)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '', 'call': 'crank_nicolson_whistler_certificate(4.,-.1,.01)', 'gold_call': '_oracle_crank_nicolson_whistler_certificate(4.,-.1,.01)'}, {'setup': '', 'call': 'crank_nicolson_whistler_certificate(20.,-.5,.002)', 'gold_call': '_oracle_crank_nicolson_whistler_certificate(20.,-.5,.002)'}, {'setup': '', 'call': 'crank_nicolson_whistler_certificate(8.,0.,.005,2.)', 'gold_call': '_oracle_crank_nicolson_whistler_certificate(8.,0.,.005,2.)'}, {'setup': '', 'call': 'crank_nicolson_whistler_certificate(30.,-1.2,.0015,.5)', 'gold_call': '_oracle_crank_nicolson_whistler_certificate(30.,-1.2,.0015,.5)'}, {'setup': '', 'call': 'crank_nicolson_whistler_certificate(2.5,-.02,.02,3.)', 'gold_call': '_oracle_crank_nicolson_whistler_certificate(2.5,-.02,.02,3.)'}, {'setup': '', 'call': 'crank_nicolson_whistler_certificate(55.,-.8,.0004,1.5)', 'gold_call': '_oracle_crank_nicolson_whistler_certificate(55.,-.8,.0004,1.5)'}, {'setup': '', 'call': 'crank_nicolson_whistler_certificate(12.,-.3,.003,2.5)', 'gold_call': '_oracle_crank_nicolson_whistler_certificate(12.,-.3,.003,2.5)'}]
