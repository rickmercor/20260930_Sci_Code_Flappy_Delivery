"""
Measure the three defined time-domain sampling errors and the phase-sensitive two-tone PDH spectral distortion after retaining every Nth round-trip sample.

Retain indices 0,N,2N,... without appending the final point, and truncate the full reference at the last retained time. Let P_ref be the maximum power on that truncated reference and P_ret the maximum retained power; peak_error=|P_ret-P_ref|/P_ref. Find all sign-changing PDH zero crossings by linear interpolation. Choose the reference crossing nearest the time of P_ref, then choose the retained crossing nearest that reference crossing; zero_error_s is their absolute time difference. Linearly interpolate retained PDH onto the truncated reference times and set PDH_NRMSE to the RMS interpolation error divided by sqrt(mean(PDH_ref**2)). For the fourth metric, on each grid use a Hann window sin(pi*u)**2 with u=(t-t_first)/(t_last-t_first) on that grid, subtract that grid's window-weighted PDH mean, form the two complex coefficients at the supplied frequencies normalized by that grid's window sum, and divide the Euclidean coefficient-difference norm by the reference coefficient norm.

Returns
-------
Return one length-4 real NumPy array in the fixed diagnostic order.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def sampling_diagnostics(time_s, observables, stride, spectral_hz):
    """Return a length-4 float array
    [peak_error, zero_error_s, PDH_NRMSE, spectral_distortion].

    The two columns of observables are power and signed PDH respectively. The retained
    grid uses indices 0,stride,2*stride,...; spectral_hz contains two positive frequencies.
    Invalid shapes, nonfinite data, non-increasing time, an invalid stride, an unavailable
    zero crossing, or a zero normalization denominator raise ValueError.
    """
    return np.zeros(4, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_sampling_diagnostics(time_s, observables, stride, spectral_hz):
    """Time-domain errors plus two-tone Hann-windowed PDH spectral distortion."""
    import numpy as np
    time_s = np.asarray(time_s, dtype=float)
    observables = np.asarray(observables, dtype=float)
    try:
        stride_value = float(stride)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("stride must be a finite positive integer") from exc
    if not np.isfinite(stride_value):
        raise ValueError("stride must be a finite positive integer")
    stride_i = int(stride_value)
    if stride_i != stride_value or stride_i < 1:
        raise ValueError("stride must be a positive integer")
    if time_s.ndim != 1 or observables.shape != (time_s.size, 2) or time_s.size < 5:
        raise ValueError("observables must have shape (len(time_s),2)")
    if (not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0)
            or not np.all(np.isfinite(observables))):
        raise ValueError("time must increase and inputs must be finite")
    spectral_hz = np.asarray(spectral_hz, dtype=float)
    if (spectral_hz.shape != (2,) or np.any(spectral_hz <= 0)
            or not np.all(np.isfinite(spectral_hz))):
        raise ValueError("spectral_hz must contain two finite positive frequencies")
    if np.max(spectral_hz) >= 0.5 / np.min(np.diff(time_s)):
        raise ValueError("spectral frequencies exceed the physical-grid Nyquist rate")
    sample_index = np.arange(0, time_s.size, stride_i, dtype=int)
    if sample_index.size < 4:
        raise ValueError("at least four samples are required")
    last = int(sample_index[-1])
    tref = time_s[:last + 1]
    pref = observables[:last + 1, 0]
    dref = observables[:last + 1, 1]
    ts = time_s[sample_index]
    ps = observables[sample_index, 0]
    ds = observables[sample_index, 1]
    peak_ref = float(np.max(pref))
    if peak_ref <= 0:
        raise ValueError("reference peak power must be positive")
    peak_error = abs(float(np.max(ps)) - peak_ref) / peak_ref

    def crossings(t, y):
        idx = np.where(((y[:-1] <= 0) & (y[1:] > 0)) |
                       ((y[:-1] >= 0) & (y[1:] < 0)))[0]
        values = []
        for i in idx:
            denominator = y[i + 1] - y[i]
            if denominator != 0:
                values.append(t[i] - y[i] * (t[i + 1] - t[i]) / denominator)
        return np.asarray(values, dtype=float)

    reference_crossings = crossings(tref, dref)
    sampled_crossings = crossings(ts, ds)
    if reference_crossings.size == 0 or sampled_crossings.size == 0:
        raise ValueError("PDH trace must contain a zero crossing")
    peak_time = tref[int(np.argmax(pref))]
    z_ref = reference_crossings[int(np.argmin(np.abs(reference_crossings - peak_time)))]
    z_sample = sampled_crossings[int(np.argmin(np.abs(sampled_crossings - z_ref)))]
    zero_time_error = abs(float(z_sample - z_ref))
    interpolated = np.interp(tref, ts, ds)
    denominator = float(np.sqrt(np.mean(dref**2)))
    if denominator == 0:
        raise ValueError("reference PDH RMS must be nonzero")
    pdh_nrmse = float(np.sqrt(np.mean((interpolated - dref)**2)) / denominator)

    def coefficients(t, y):
        u = (t - t[0]) / (t[-1] - t[0])
        window = np.sin(np.pi * u)**2
        window_sum = float(np.sum(window))
        centered = y - float(np.sum(window * y) / window_sum)
        return np.asarray([
            np.sum(window * centered * np.exp(-2j * np.pi * f * t)) / window_sum
            for f in spectral_hz
        ], dtype=complex)

    reference_coefficients = coefficients(tref, dref)
    sampled_coefficients = coefficients(ts, ds)
    spectral_denominator = float(np.sum(np.abs(reference_coefficients)**2))
    if spectral_denominator <= 0:
        raise ValueError("reference two-tone PDH energy must be positive")
    spectral_distortion = float(np.sqrt(
        np.sum(np.abs(sampled_coefficients - reference_coefficients)**2)
        / spectral_denominator
    ))
    return np.array([peak_error, zero_time_error, pdh_nrmse,
                     spectral_distortion], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nt=np.linspace(-1,1,101); o=np.column_stack((np.exp(-t*t),np.sin(5*t)+.1*np.sin(9*t)))", "call":"sampling_diagnostics(t,o,2,np.array([1.,2.]))", "gold_call":"_oracle_sampling_diagnostics(t,o,2,np.array([1.,2.]))"},
        {"setup":"import numpy as np\nt=np.linspace(-1,1,101); o=np.column_stack((np.exp(-((t-.13)/.3)**2),np.sin(5*t+.2)))", "call":"sampling_diagnostics(t,o,4,np.array([1.,2.]))", "gold_call":"_oracle_sampling_diagnostics(t,o,4,np.array([1.,2.]))"},
        {"setup":"import numpy as np\nt=np.linspace(0,2,121); o=np.column_stack((1+.2*np.cos(3*t),np.sin(7*t+.1)+.2*np.sin(11*t)))", "call":"sampling_diagnostics(t,o,3,np.array([1.,2.]))", "gold_call":"_oracle_sampling_diagnostics(t,o,3,np.array([1.,2.]))"},
        {"setup":"import numpy as np\nt=np.linspace(0,1,129); o=np.column_stack((1+.3*np.exp(-((t-.4)/.08)**2),np.sin(2*np.pi*8*t)))", "call":"sampling_diagnostics(t,o,1,np.array([8.,13.]))", "gold_call":"_oracle_sampling_diagnostics(t,o,1,np.array([8.,13.]))"},
        {"setup":"import numpy as np\nt=np.linspace(-.5,.5,151); o=np.column_stack((1+.4*np.cos(2*np.pi*3*t),np.sin(2*np.pi*12*t+.4)+.3*np.sin(2*np.pi*19*t)))", "call":"sampling_diagnostics(t,o,5,np.array([12.,19.]))", "gold_call":"_oracle_sampling_diagnostics(t,o,5,np.array([12.,19.]))"},
        {"setup":"import numpy as np\nt=np.linspace(0,.2,201); o=np.column_stack((1+np.exp(-((t-.08)/.015)**2),np.sin(2*np.pi*30*t)-.2*np.sin(2*np.pi*45*t)))", "call":"sampling_diagnostics(t,o,7,np.array([30.,45.]))", "gold_call":"_oracle_sampling_diagnostics(t,o,7,np.array([30.,45.]))"},
        {"setup":"import numpy as np\nt=np.linspace(-2,2,161); o=np.column_stack((2+.1*np.cos(t),np.sin(4*t+.7)+.15*np.sin(13*t)))", "call":"sampling_diagnostics(t,o,8,np.array([.5,1.5]))", "gold_call":"_oracle_sampling_diagnostics(t,o,8,np.array([.5,1.5]))"}
    ]
