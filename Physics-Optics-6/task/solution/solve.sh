#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def raman_response(t: "np.ndarray", tau1: float, tau2: float) -> "np.ndarray":
    t = np.asarray(t, dtype=float)
    # tau1 is the PERIOD, so the angular frequency of the ringing is 2*pi/tau1 and the
    # oscillator time constant that enters the normalisation is a = tau1/(2*pi).
    a = tau1 / (2.0 * np.pi)
    amp = (a * a + tau2 * tau2) / (a * tau2 * tau2)
    tp = np.where(t > 0.0, t, 0.0)
    return np.where(t > 0.0, amp * np.sin(2.0 * np.pi * tp / tau1) * np.exp(-tp / tau2), 0.0)

import numpy as np


def overlap_expansion_coefficients(f_R: float, tau1: float, tau2: float) -> "np.ndarray":
    # sech(u)^2 tanh(u) = u - (4/3) u^3 + O(u^5), so the expansion coefficients are the first
    # and third moments of the response, the third carrying that rational factor.
    a = tau1 / (2.0 * np.pi)
    amp = (a * a + tau2 * tau2) / (a * tau2 * tau2)
    w = 1.0 / a
    s = 1.0 / tau2
    # int_0^inf t^n exp(-s t) sin(w t) dt = Im[ n! / (s - i w)^(n+1) ]
    moment1 = amp * 2.0 * s * w / (s * s + w * w) ** 2
    moment3 = amp * 24.0 * s * w * (s * s - w * w) / (s * s + w * w) ** 4
    return np.array([f_R * moment1, -(4.0 / 3.0) * f_R * moment3], dtype=float)

import numpy as np


def cavity_scales(lambda0: float, D1: float, D2: float, Q_int: float, Q_ext: float) -> "np.ndarray":
    c_light = 299792458.0
    omega0 = 2.0 * np.pi * c_light / lambda0
    kappa_ext = omega0 / Q_ext
    kappa = omega0 / Q_int + kappa_ext
    # The mode-family curvature already absorbs the index and the speed of light:
    # D2 = -c D1^2 beta2 / n0, so c|beta2|/n0 = D2/D1^2 and tau0^2 = D2/(kappa D1^2).
    tau0 = np.sqrt(D2 / (kappa * D1 * D1))
    return np.array([kappa, kappa_ext, tau0, 2.0 * np.pi / D1], dtype=float)

import numpy as np


def _overlap_time(tau_s: float, f_R: float, tau1: float, tau2: float) -> float:
    # The response decays on tau2, so the half line is truncated at sixty lifetimes; the grid
    # resolves both the ringing period tau1 and the pulse duration tau_s many times over.
    horizon = 60.0 * float(tau2)
    n = 50000
    t = np.linspace(0.0, horizon, n + 1)
    u = np.clip(t / float(tau_s), -300.0, 300.0)
    kernel = (1.0 / np.cosh(u)) ** 2 * np.tanh(u)
    return float(f_R * np.trapezoid(raman_response(t, tau1, tau2) * kernel, t))


def nonadiabatic_soliton_duration(omega_target: float, tau0: float, f_R: float,
                                          tau1: float, tau2: float) -> float:
    if not (float(omega_target) < 0.0):
        raise ValueError("omega_target must be strictly negative: the delayed response "
                         "shifts energy only towards lower frequency")
    target = float(omega_target)

    def _shift(tau_s):
        x = np.pi * np.pi * tau_s / float(tau1)
        return -2.0 * tau0 * tau0 * (x / np.sinh(x)) * _overlap_time(tau_s, f_R, tau1, tau2) / tau_s ** 3

    # the slowly varying inversion is the right order of magnitude, so it seeds the bracket
    tau_A = float(overlap_expansion_coefficients(f_R, tau1, tau2)[0])
    seed = (8.0 * tau0 * tau0 * tau_A / (15.0 * abs(target))) ** 0.25
    lower, upper = seed / 16.0, seed * 16.0
    if not (_shift(lower) < target < _shift(upper)):
        raise ValueError("omega_target is not reachable on the bracketed range of durations")
    for _ in range(200):
        mid = 0.5 * (lower + upper)
        if _shift(mid) < target:
            lower = mid
        else:
            upper = mid
        if upper - lower < 1e-11 * upper:
            break
    return float(0.5 * (lower + upper))

import numpy as np


def raman_kernel_spectrum(n_modes: int, window: float, tau0: float,
                                  tau1: float, tau2: float) -> "np.ndarray":
    spacing = window / n_modes
    lag = np.arange(n_modes) * spacing
    # tau0 converts the response from per second to per unit dimensionless time.
    samples = raman_response(lag * tau0, tau1, tau2) * tau0
    return np.fft.fft(samples) * spacing

import numpy as np


def soliton_seed(zeta: float, n_modes: int, window: float) -> "np.ndarray":
    spacing = window / n_modes
    theta = (np.arange(n_modes) - n_modes // 2) * spacing
    arg = np.clip(theta * np.sqrt(zeta), -700.0, 700.0)
    return (np.sqrt(2.0 * zeta) / np.cosh(arg)).astype(complex)

import numpy as np


def lle_propagators(zeta: float, n_modes: int, window: float, dtau: float) -> "np.ndarray":
    offsets = 2.0 * np.pi * np.fft.fftfreq(n_modes, d=window / n_modes)
    linear = -(1.0 + 1j * zeta) - 1j * offsets * offsets
    field = np.exp(linear * dtau / 2.0)
    # The real part of `linear` is -1 everywhere, so it never vanishes and the drive
    # propagator needs no removable-singularity branch.
    return np.vstack([field, (field - 1.0) / linear])

import numpy as np


def propagate_soliton(psi0: "np.ndarray", kernel_spectrum: "np.ndarray",
                              propagators: "np.ndarray", f_R: float, f_pump: float,
                              dtau: float, n_steps: int) -> "np.ndarray":
    psi = np.asarray(psi0, dtype=complex).copy()
    n = psi.size
    field_op = propagators[0]
    drive_op = propagators[1]
    drive_hat = np.zeros(n, dtype=complex)
    drive_hat[0] = f_pump * n
    for _ in range(int(n_steps)):
        psi = np.fft.ifft(field_op * np.fft.fft(psi) + drive_op * drive_hat)
        intensity = np.abs(psi) ** 2
        delayed = np.real(np.fft.ifft(kernel_spectrum * np.fft.fft(intensity)))
        psi = psi * np.exp(1j * dtau * ((1.0 - f_R) * intensity + f_R * delayed))
        psi = np.fft.ifft(field_op * np.fft.fft(psi) + drive_op * drive_hat)
    return np.abs(np.fft.fft(psi) / n) ** 2

import numpy as np
from scipy.optimize import curve_fit


def soliton_observables(spectrum: "np.ndarray", window: float, tau0: float) -> "np.ndarray":
    power = np.asarray(spectrum, dtype=float).copy()
    n = power.size
    power[0] = 0.0                                   # the transmitted drive is not comb light
    offsets = 2.0 * np.pi * np.fft.fftfreq(n, d=window / n)
    order = np.argsort(offsets)
    x = offsets[order]
    y = power[order]
    keep = y > y.max() * 1e-4                        # forty decibels below the strongest line
    x = x[keep]
    y = y[keep]
    mean = np.sum(x * y) / np.sum(y)
    rms = np.sqrt(np.sum(y * (x - mean) ** 2) / np.sum(y))

    def _envelope(v, log_amp, centre, width):
        arg = np.clip(np.pi * (v - centre) * width / 2.0, -300.0, 300.0)
        return log_amp - 2.0 * np.log(np.cosh(arg))

    guess = [np.log(y.max()), 0.0, 2.0 / max(rms, 1e-12)]
    fitted, _ = curve_fit(_envelope, x, np.log(y), p0=guess, maxfev=80000)
    return np.array([abs(fitted[2]) * tau0, -fitted[1] / tau0], dtype=float)

import numpy as np


def detuning_for_target_shift(lambda0: float, D1: float, D2: float, Q_int: float, Q_ext: float,
                                      f_R: float, tau1: float, tau2: float, omega_target: float,
                                      pump_power: float, zeta_floor: float, n_modes: int, dtau: float,
                                      n_steps: int) -> float:
    if not (float(f_R) > 0.0):
        raise ValueError("f_R must be strictly positive")
    if not (float(omega_target) < 0.0):
        raise ValueError("omega_target must be strictly negative")

    scales = cavity_scales(lambda0, D1, D2, Q_int, Q_ext)
    tau0 = scales[2]
    window = scales[3] / tau0
    kernel = raman_kernel_spectrum(n_modes, window, tau0, tau1, tau2)
    amplitude = np.sqrt(pump_power)
    target = float(omega_target)

    def _shift(zeta):
        seed = soliton_seed(zeta, n_modes, window)
        props = lle_propagators(zeta, n_modes, window, dtau)
        spectrum = propagate_soliton(seed, kernel, props, f_R, amplitude, dtau, n_steps)
        return soliton_observables(spectrum, window, tau0)[1]

    # the delayed-response estimate over-assigns the detuning, so it closes the interval from above
    lower = float(zeta_floor)
    upper = (tau0 / nonadiabatic_soliton_duration(target, tau0, f_R, tau1, tau2)) ** 2
    if not (upper > lower):
        raise ValueError("the delayed-response estimate does not lie above zeta_floor")
    # the shift is monotone on this interval, so bracketing it here is what makes the root unique
    if not (_shift(lower) > target):
        raise ValueError("the shift at zeta_floor is already below omega_target")
    if not (_shift(upper) < target):
        raise ValueError("the shift at the delayed-response estimate does not reach omega_target")

    for _ in range(60):
        mid = 0.5 * (lower + upper)
        if _shift(mid) > target:
            lower = mid
        else:
            upper = mid
        if upper - lower < 1e-5 * upper:
            break
    return float(0.5 * (lower + upper))
SCICODE_GOLD_EOF
