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


def fermi_occupation(energies, temperature):
    # A string passes np.isscalar but has no float value, and asking numpy
    # whether it is finite raises TypeError rather than the ValueError this
    # function documents. Convert first and turn every failure into that
    # documented error, for the temperature and for the energies alike.
    if not np.isscalar(temperature) or isinstance(temperature, (str, bytes)):
        raise ValueError("temperature must be a finite non-negative scalar")
    try:
        temperature = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("temperature must be a finite non-negative scalar") from exc
    if not np.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be a finite non-negative scalar")
    try:
        e = np.asarray(energies, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("energies must all be finite") from exc
    if not np.all(np.isfinite(e)):
        raise ValueError("energies must all be finite")
    if temperature == 0.0:
        return np.where(e < 0.0, 1.0, np.where(e > 0.0, 0.0, 0.5))
    kb_mev_per_k = 0.08617333262
    x = np.clip(e / (kb_mev_per_k * temperature), -500.0, 500.0)
    return 1.0 / (np.exp(x) + 1.0)

import numpy as np
from scipy.integrate import quad


def gap_kernel(gap, temperature, cutoff):
    values = ((gap, "gap"), (temperature, "temperature"), (cutoff, "cutoff"))
    converted = []
    for value, name in values:
        if not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError(name + " must be a finite scalar")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(numeric):
            raise ValueError(name + " must be a finite scalar")
        converted.append(numeric)
    gap, temperature, cutoff = converted
    if gap <= 0.0:
        raise ValueError("gap must be strictly greater than zero")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if cutoff <= 0.0:
        raise ValueError("cutoff must be strictly greater than zero")

    def integrand(xi):
        E = np.sqrt(xi**2 + gap**2)
        occupation = float(np.asarray(fermi_occupation(np.array([E]), temperature))[0])
        return (1.0 - 2.0 * occupation) / (2.0 * E)

    # Magnitudes far outside the millielectronvolt range overflow the squared
    # band energy or underflow the quasiparticle energy to zero, and the
    # quadrature then raises an arithmetic error or returns a non-finite number.
    # Both are refusals of the input rather than results, so report them as the
    # documented ValueError instead of leaking the arithmetic exception.
    try:
        result, _ = quad(integrand, -cutoff, cutoff, limit=200, points=[0.0])
    except (OverflowError, ZeroDivisionError, FloatingPointError) as exc:
        raise ValueError("the band integral does not evaluate to a finite number at these magnitudes") from exc
    if not np.isfinite(result):
        raise ValueError("the band integral does not evaluate to a finite number at these magnitudes")
    return float(result)

import numpy as np
from scipy.optimize import brentq


def equilibrium_gap(temperature, cutoff, coupling):
    values = ((temperature, "temperature"), (cutoff, "cutoff"), (coupling, "coupling"))
    converted = []
    for value, name in values:
        if not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError(name + " must be a finite scalar")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(numeric):
            raise ValueError(name + " must be a finite scalar")
        converted.append(numeric)
    temperature, cutoff, coupling = converted
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if cutoff <= 0.0:
        raise ValueError("cutoff must be strictly greater than zero")
    if coupling <= 0.0:
        raise ValueError("coupling must be strictly greater than zero")

    def residual(gap):
        return gap_kernel(gap, temperature, cutoff) - 1.0 / coupling

    lo = 1e-9
    # Parameters far outside the millielectronvolt range overflow inside the
    # kernel or while the bracket grows. The root finder cannot serve such an
    # input, so it is refused with the documented error rather than leaking an
    # arithmetic exception.
    try:
        h_lo = residual(lo)
        if h_lo < 0.0:
            # The kernel is largest at a vanishing gap, so a negative residual
            # there leaves no positive root and the normal state is the only
            # solution.
            return 0.0

        # The root moves far above the cutoff once the coupling is strong, since
        # the kernel falls off only logarithmically in the gap. A bracket fixed
        # at a small multiple of the cutoff therefore misses the root entirely
        # and, with a sign test alone to fall back on, reports the normal state
        # for a comfortably superconducting parameter set. Grow the upper end
        # until the residual actually changes sign instead of assuming a scale.
        hi = 5.0 * cutoff
        h_hi = residual(hi)
        for _ in range(200):
            if h_hi <= 0.0:
                break
            hi *= 2.0
            h_hi = residual(hi)
        if h_hi > 0.0:
            raise ValueError("no positive gap root was bracketed below the search limit")
        return float(brentq(residual, lo, hi, xtol=1e-14, rtol=1e-14))
    except (OverflowError, ZeroDivisionError, FloatingPointError) as exc:
        raise ValueError("no finite equilibrium gap at these magnitudes") from exc

import numpy as np
from scipy.optimize import brentq


def critical_temperature(cutoff, coupling):
    values = ((cutoff, "cutoff"), (coupling, "coupling"))
    converted = []
    for value, name in values:
        if not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError(name + " must be a finite scalar")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(numeric):
            raise ValueError(name + " must be a finite scalar")
        converted.append(numeric)
    cutoff, coupling = converted
    if cutoff <= 0.0:
        raise ValueError("cutoff must be strictly greater than zero")
    if coupling <= 0.0:
        raise ValueError("coupling must be strictly greater than zero")

    lo = 0.01
    hi = 100.0
    threshold = 1e-6

    def residual(temperature):
        return equilibrium_gap(temperature, cutoff, coupling) - threshold

    # A cutoff or coupling far outside the millielectronvolt range overflows
    # inside the gap solver. That is a refusal of the input rather than a
    # result, so it is reported as the documented ValueError.
    try:
        h_lo = residual(lo)
        h_hi = residual(hi)
        if not ((h_lo < 0.0 < h_hi) or (h_hi < 0.0 < h_lo)):
            raise ValueError("no transition found in the temperature search window")
        return float(brentq(residual, lo, hi, xtol=1e-10))
    except (OverflowError, ZeroDivisionError, FloatingPointError) as exc:
        raise ValueError("no finite critical temperature at these magnitudes") from exc

import numpy as np
from scipy.integrate import quad


def inertia_coefficient(gap, temperature, cutoff):
    values = ((gap, "gap"), (temperature, "temperature"), (cutoff, "cutoff"))
    converted = []
    for value, name in values:
        if not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError(name + " must be a finite scalar")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(numeric):
            raise ValueError(name + " must be a finite scalar")
        converted.append(numeric)
    gap, temperature, cutoff = converted
    if gap <= 0.0:
        raise ValueError("gap must be strictly greater than zero")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if cutoff <= 0.0:
        raise ValueError("cutoff must be strictly greater than zero")

    kb_mev_per_k = 0.08617333262

    def integrand(xi):
        # The bracket is (2 f(E) - 1) / (2 E). With 2 f(E) - 1 = -tanh(E / 2 k T)
        # its energy derivative is available in closed form, so no step size
        # enters and the value is exact to quadrature accuracy. A finite
        # difference here would leave a truncation error of order 1e-9, which is
        # the same size as the comparison tolerance and would fail a solver who
        # differentiated exactly.
        E = np.sqrt(xi**2 + gap**2)
        if temperature <= 0.0:
            derivative = 1.0 / (2.0 * E * E)
        else:
            x = E / (2.0 * kb_mev_per_k * temperature)
            # cosh overflows near x = 710 while sech squared has already
            # underflowed to zero, so take that limit directly rather than
            # dividing by an infinity.
            sech2 = 0.0 if x > 350.0 else 1.0 / np.cosh(x) ** 2
            derivative = np.tanh(x) / (2.0 * E * E) - sech2 / (4.0 * kb_mev_per_k * temperature * E)
        return derivative / (2.0 * E)

    # Magnitudes far outside the millielectronvolt range overflow the squared
    # band energy or underflow the quasiparticle energy to zero, and the
    # quadrature then raises an arithmetic error or returns a non-finite number.
    # Both are refusals of the input rather than results, so report them as the
    # documented ValueError instead of leaking the arithmetic exception.
    try:
        result, _ = quad(integrand, -cutoff, cutoff, limit=200, points=[0.0])
    except (OverflowError, ZeroDivisionError, FloatingPointError) as exc:
        raise ValueError("the band integral does not evaluate to a finite number at these magnitudes") from exc
    if not np.isfinite(result):
        raise ValueError("the band integral does not evaluate to a finite number at these magnitudes")
    return float(result)

import numpy as np


def pulse_envelope(times, t_center, sigma_before, sigma_after, frequency, phase):
    values = (
        (t_center, "t_center"),
        (sigma_before, "sigma_before"),
        (sigma_after, "sigma_after"),
        (frequency, "frequency"),
        (phase, "phase"),
    )
    converted = []
    for value, name in values:
        if not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError(name + " must be a finite scalar")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(numeric):
            raise ValueError(name + " must be a finite scalar")
        converted.append(numeric)
    t_center, sigma_before, sigma_after, frequency, phase = converted
    if sigma_before <= 0.0:
        raise ValueError("sigma_before must be strictly greater than zero")
    if sigma_after <= 0.0:
        raise ValueError("sigma_after must be strictly greater than zero")
    try:
        t = np.asarray(times, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("times must all be finite") from exc
    if not np.all(np.isfinite(t)):
        raise ValueError("times must all be finite")
    sigma = np.where(t < t_center, sigma_before, sigma_after)
    envelope = np.exp(-((t - t_center) ** 2) / sigma**2)
    carrier = np.cos(2.0 * np.pi * frequency * t + phase)
    return np.asarray(envelope * carrier, dtype=float)

import numpy as np


def drive_coefficient(drive_strength, t_sim, t_ref, cutoff, coupling):
    values = (
        (drive_strength, "drive_strength"),
        (t_sim, "t_sim"),
        (t_ref, "t_ref"),
        (cutoff, "cutoff"),
        (coupling, "coupling"),
    )
    converted = []
    for value, name in values:
        if not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError(name + " must be a finite scalar")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(numeric):
            raise ValueError(name + " must be a finite scalar")
        converted.append(numeric)
    drive_strength, t_sim, t_ref, cutoff, coupling = converted
    if drive_strength < 0.0:
        raise ValueError("drive_strength must be non-negative")
    if t_sim <= 0.0:
        raise ValueError("t_sim must be strictly greater than zero")
    if t_ref <= 0.0:
        raise ValueError("t_ref must be strictly greater than zero")
    if cutoff <= 0.0:
        raise ValueError("cutoff must be strictly greater than zero")
    if coupling <= 0.0:
        raise ValueError("coupling must be strictly greater than zero")

    # Magnitudes far outside the millielectronvolt range overflow inside the
    # upstream gap and inertia integrals, or leave a product of two huge
    # numbers that is no longer finite. Either is a refusal of the input
    # rather than a result, so it is reported as the documented ValueError.
    try:
        gap_sim = float(equilibrium_gap(t_sim, cutoff, coupling))
        gap_ref = float(equilibrium_gap(t_ref, cutoff, coupling))
        if gap_sim <= 0.0 or gap_ref <= 0.0:
            raise ValueError("equilibrium gap must not vanish at either temperature")
        inertia_sim = float(inertia_coefficient(gap_sim, t_sim, cutoff))
        inertia_ref = float(inertia_coefficient(gap_ref, t_ref, cutoff))
        result = float(drive_strength * inertia_sim / inertia_ref)
    except (OverflowError, ZeroDivisionError, FloatingPointError) as exc:
        raise ValueError("no finite drive coefficient at these magnitudes") from exc
    if not np.isfinite(result):
        raise ValueError("no finite drive coefficient at these magnitudes")
    return result

import numpy as np
from scipy.integrate import solve_ivp


def run_pipeline(cutoff, coupling, sim_fraction, ref_fraction, drive_strength,
                       damping, frequency, sigma_before, sigma_after, phase,
                       t_center, t_max):
    scalars = (cutoff, coupling, sim_fraction, ref_fraction, drive_strength,
               damping, frequency, sigma_before, sigma_after, phase, t_center, t_max)
    converted = []
    for value in scalars:
        # A string is a scalar to numpy but has no float value, and asking
        # numpy whether it is finite raises TypeError rather than the ValueError
        # this function documents. Convert first and turn every failure into the
        # documented error.
        if not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError("every argument must be a finite scalar")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("every argument must be a finite scalar") from exc
        if not np.isfinite(numeric):
            raise ValueError("every argument must be a finite scalar")
        converted.append(numeric)
    (cutoff, coupling, sim_fraction, ref_fraction, drive_strength,
     damping, frequency, sigma_before, sigma_after, phase,
     t_center, t_max) = converted
    if cutoff <= 0.0 or coupling <= 0.0:
        raise ValueError("cutoff and coupling must be strictly positive")
    if damping <= 0.0 or t_max <= 0.0:
        raise ValueError("damping and t_max must be strictly positive")
    if drive_strength < 0.0:
        raise ValueError("drive_strength must be non-negative")
    for fraction in (sim_fraction, ref_fraction):
        if not 0.0 < fraction < 1.0:
            raise ValueError("temperature fractions must lie strictly between zero and one")

    t_c = float(critical_temperature(cutoff, coupling))
    t_sim = sim_fraction * t_c
    t_ref = ref_fraction * t_c

    gap_0 = float(equilibrium_gap(t_sim, cutoff, coupling))
    if gap_0 <= 0.0:
        raise ValueError("no superconducting solution at the simulation temperature")

    inertia = float(inertia_coefficient(gap_0, t_sim, cutoff))
    coefficient = float(drive_coefficient(drive_strength, t_sim, t_ref, cutoff, coupling))
    inverse_coupling = 1.0 / coupling

    # The reduced inertia coefficient carries units of inverse energy squared,
    # because its outer weight contributes one inverse energy, the derivative with
    # respect to quasiparticle energy contributes two more and the band measure
    # returns one. Multiplying the second time derivative by it therefore lands in
    # inverse energy per squared time while every other term is an energy, and the
    # two are reconciled by the squared reduced Planck constant, which carries
    # squared energy times squared time. The damping value is already a time, so
    # its term is an energy as it stands and takes no further factor.
    hbar_mev_ps = 0.6582119569
    inertial = hbar_mev_ps * hbar_mev_ps * inertia

    def derivative(t, state):
        gap, rate = state
        envelope = float(np.asarray(pulse_envelope(
            np.array([t]), t_center, sigma_before, sigma_after, frequency, phase))[0])
        safe_gap = max(gap, 1e-12)
        kernel = float(gap_kernel(safe_gap, t_sim, cutoff))
        force = (-coefficient * envelope ** 2 * gap
                 - inverse_coupling * gap
                 + gap * kernel)
        return [rate, (force - damping * rate) / inertial]

    solution = solve_ivp(derivative, (0.0, t_max), [gap_0, 0.0], dense_output=True,
                         rtol=1e-11, atol=1e-14, max_step=0.01)
    if not solution.success:
        raise ValueError("the gap dynamics failed to integrate")

    # The target is the smallest magnitude anywhere in the window, so a sampled
    # grid is not enough: the extremum falls between samples, and refining one
    # grid family by powers of two keeps landing on the same points and looks
    # convergent while missing it. Scan coarsely for the bracket, then descend
    # on the dense output by golden-section, which needs no derivative and
    # cannot step outside the bracket.
    scan = np.linspace(0.0, t_max, 4001)
    magnitude = np.abs(solution.sol(scan)[0])
    index = int(np.argmin(magnitude))
    left = scan[max(index - 1, 0)]
    right = scan[min(index + 1, scan.size - 1)]

    def magnitude_at(t):
        return abs(float(np.asarray(solution.sol(t))[0]))

    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(left), float(right)
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = magnitude_at(c), magnitude_at(d)
    for _ in range(200):
        if b - a < 1e-13:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = magnitude_at(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = magnitude_at(d)
    minimum = min(magnitude_at(0.5 * (a + b)), float(magnitude[index]))
    return float(minimum / gap_0)
SCICODE_GOLD_EOF
