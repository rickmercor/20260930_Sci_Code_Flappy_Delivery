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
from scipy.special import i0e, i1e


def _bessel_ratio(x):
    """I_1(x) / I_0(x), evaluated stably and with the value 0 at x = 0."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    nz = x > 0.0
    out[nz] = i1e(x[nz]) / i0e(x[nz])
    return out


def sensor_alignment_statistics(sigma_phi_sq, gamma_values):
    """Reference implementation of sensor_alignment_statistics."""
    if isinstance(sigma_phi_sq, bool):
        raise ValueError("sigma_phi_sq must be a real number, got a bool")
    try:
        sigma_phi_sq = float(sigma_phi_sq)
    except (TypeError, ValueError):
        raise ValueError("sigma_phi_sq must be a real number")
    if not np.isfinite(sigma_phi_sq) or sigma_phi_sq <= 0.0:
        raise ValueError("sigma_phi_sq must be finite and strictly positive")

    try:
        gam = np.asarray(gamma_values, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("gamma_values must be an array of real numbers")
    if gam.ndim != 1:
        raise ValueError("gamma_values must be one-dimensional")
    if gam.size < 1:
        raise ValueError("gamma_values must have at least one entry")
    if not np.all(np.isfinite(gam)):
        raise ValueError("gamma_values must be finite")
    if np.any(gam < 0.0) or np.any(gam > 1.0):
        raise ValueError("every entry of gamma_values must lie in [0, 1]")

    conc = gam / sigma_phi_sq
    mean_cos = _bessel_ratio(conc)

    # <sin^2> = c0(s)/s, which tends to 1/2 as s -> 0.
    mean_sin_sq = np.full(conc.shape, 0.5, dtype=np.float64)
    nz = conc > 0.0
    mean_sin_sq[nz] = mean_cos[nz] / conc[nz]

    stats = np.empty((gam.size, 4), dtype=np.float64)
    stats[:, 0] = conc
    stats[:, 1] = mean_cos
    stats[:, 2] = mean_sin_sq
    stats[:, 3] = gam * mean_cos
    return stats

import numpy as np
from scipy.special import i0e, i1e

# Rational series coefficients of the four gradient-expansion functions, in
# ascending powers of the squared steering strength, truncated at order 20.
_C1_NUM = (1, -5, 23, -677, 7313, -218491, 863897, -874088357,
           27545803997, -423385249313, 19488418951523)
_C1_DEN = (2, 32, 576, 73728, 3686400, 530841600, 10404495360,
           53271016243200, 8629904631398400, 690392370511872000,
           167074953663873024000)
_C2_NUM = (0, -1, 1, -131, 25, -41851, 20209, -33334307, 133205867,
           -19173165917, 3470122403)
_C2_DEN = (1, 8, 16, 6144, 4096, 26542080, 53084160, 380507258880,
           6849130659840, 4566087106560000, 3913788948480000)
_C3_NUM = (0, 3, -29, 1325, -19553, 247777, -397169, 473737993,
           -154836151097, 13477269234097, -693144302217667)
_C3_DEN = (1, 32, 576, 73728, 3686400, 176947200, 1156055040,
           5919001804800, 8629904631398400, 3451961852559360000,
           835374768319365120000)
_C4_NUM = (0, 3, -17, 545, -6173, 190111, -767717, 788938397,
           -25160566037, 390389770937, -18107708487467)
_C4_DEN = (1, 32, 576, 73728, 3686400, 530841600, 10404495360,
           53271016243200, 8629904631398400, 690392370511872000,
           167074953663873024000)


def _series_value(num, den, x):
    """Evaluate sum_n (num[n]/den[n]) * x**(2n) for an array x."""
    x = np.asarray(x, dtype=np.float64)
    z = x * x
    total = np.zeros_like(x)
    power = np.ones_like(x)
    for n, d in zip(num, den):
        total = total + (float(n) / float(d)) * power
        power = power * z
    return total


def closure_series_coefficients(kappa_values):
    """Reference implementation of closure_series_coefficients."""
    try:
        kap = np.asarray(kappa_values, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("kappa_values must be an array of real numbers")
    if kap.ndim != 1:
        raise ValueError("kappa_values must be one-dimensional")
    if kap.size < 1:
        raise ValueError("kappa_values must have at least one entry")
    if not np.all(np.isfinite(kap)):
        raise ValueError("kappa_values must be finite")
    if np.any(kap < 0.0) or np.any(kap > 1.2):
        raise ValueError("every entry of kappa_values must lie in [0, 1.2]")

    c0 = np.zeros_like(kap)
    nz = kap > 0.0
    c0[nz] = i1e(kap[nz]) / i0e(kap[nz])

    coeffs = np.empty((kap.size, 5), dtype=np.float64)
    coeffs[:, 0] = c0
    coeffs[:, 1] = _series_value(_C1_NUM, _C1_DEN, kap)
    coeffs[:, 2] = _series_value(_C2_NUM, _C2_DEN, kap)
    coeffs[:, 3] = _series_value(_C3_NUM, _C3_DEN, kap)
    coeffs[:, 4] = _series_value(_C4_NUM, _C4_DEN, kap)
    return coeffs

import numpy as np


def region_transport_parameters(kappa_tilde, mean_cos_values, peclet):
    """Reference implementation of region_transport_parameters."""
    for name, value in (("kappa_tilde", kappa_tilde), ("peclet", peclet)):
        if isinstance(value, bool):
            raise ValueError("%s must be a real number, got a bool" % name)
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("%s must be a real number" % name)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("%s must be finite and strictly positive" % name)
    kappa_tilde = float(kappa_tilde)
    peclet = float(peclet)

    try:
        mc = np.asarray(mean_cos_values, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("mean_cos_values must be an array of real numbers")
    if mc.ndim != 1:
        raise ValueError("mean_cos_values must be one-dimensional")
    if mc.size < 1:
        raise ValueError("mean_cos_values must have at least one entry")
    if not np.all(np.isfinite(mc)):
        raise ValueError("mean_cos_values must be finite")
    if np.any(mc < 0.0) or np.any(mc > 1.0):
        raise ValueError("every entry of mean_cos_values must lie in [0, 1]")

    kappa_eff = kappa_tilde * mc
    coeffs = closure_series_coefficients(kappa_eff)
    c0k = coeffs[:, 0]
    c1 = coeffs[:, 1]
    c2 = coeffs[:, 2]
    c3 = coeffs[:, 3]

    effective_diffusivity = 1.0 / peclet + c1 + c2

    transport = np.empty((mc.size, 5), dtype=np.float64)
    transport[:, 0] = c0k / effective_diffusivity
    transport[:, 1] = c3 / effective_diffusivity
    transport[:, 2] = kappa_eff
    transport[:, 3] = effective_diffusivity
    steered = c0k > 0.0
    transport[:, 4] = 0.0
    transport[steered, 4] = effective_diffusivity[steered] / c0k[steered]
    return transport

import numpy as np


def _check_transport_and_radii(transport, radii):
    """Shared validation of a (n, 2) transport table and its n - 1 radii."""
    try:
        tr = np.asarray(transport, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("transport must be an array of real numbers")
    if tr.ndim != 2 or tr.shape[1] < 2:
        raise ValueError("transport must be two-dimensional with at least two "
                         "columns")
    if tr.shape[0] < 1:
        raise ValueError("transport must have at least one row")
    if not np.all(np.isfinite(tr)):
        raise ValueError("transport must be finite")
    if np.any(tr[:, 0] < 0.0):
        raise ValueError("inverse decay lengths must be non-negative")

    try:
        rad = np.asarray(radii, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("radii must be an array of real numbers")
    if rad.ndim != 1:
        raise ValueError("radii must be one-dimensional")
    if rad.size != tr.shape[0] - 1:
        raise ValueError(
            "radii must have exactly one entry fewer than transport has rows")
    if not np.all(np.isfinite(rad)):
        raise ValueError("radii must be finite")
    if rad.size > 0:
        if np.any(rad <= 0.0):
            raise ValueError("radii must be strictly positive")
        if np.any(np.diff(rad) <= 0.0):
            raise ValueError("radii must be strictly increasing")
    return tr, rad


def matched_density_amplitudes(transport, radii):
    """Reference implementation of matched_density_amplitudes."""
    try:
        tr, rad = _check_transport_and_radii(transport, radii)
    except ValueError as exc:
        raise ValueError("matched_density_amplitudes: %s" % exc)
    n = tr.shape[0]

    log_amp = np.zeros(n, dtype=np.float64)          # outermost log-amplitude 0
    for j in range(n - 2, -1, -1):
        r = rad[j]
        log_shape_out = tr[j + 1, 1] * np.log(r) - tr[j + 1, 0] * r
        log_shape_in = tr[j, 1] * np.log(r) - tr[j, 0] * r
        log_amp[j] = log_amp[j + 1] + log_shape_out - log_shape_in
    amplitudes = np.exp(log_amp)
    if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes <= 0.0):
        raise ValueError(
            "the matched amplitudes overflowed or underflowed; the transport "
            "parameters and radii are too widely separated to match")
    return amplitudes

import numpy as np
from scipy.integrate import quad as _quad
from scipy.special import exp1, gamma as _gamma_fn, gammaincc


def _upper_incomplete_gamma(order, x):
    """Upper incomplete gamma for any real order, with x > 0.

    scipy covers order > 0 only. Order 0 is the exponential integral, and
    negative orders follow from the standard downward recurrence
    Gamma(a, x) = (Gamma(a + 1, x) - x**a * exp(-x)) / a.
    """
    if order > 0.0:
        return _gamma_fn(order) * gammaincc(order, x)
    if order == 0.0:
        return float(exp1(x))
    return (_upper_incomplete_gamma(order + 1.0, x)
            - x ** order * np.exp(-x)) / order


def _finite_panel_moment(index, inverse_length, lower, upper):
    """Integral of r**index * exp(-inverse_length * r) over a finite panel
    whose lower edge is strictly positive.

    The panel is mapped onto [0, 1], integrated adaptively there and scaled
    back by its width. Differencing two upper incomplete gammas instead loses
    the panel when the region barely decays, because both values then sit on
    Gamma(order) itself and their difference falls below the rounding of
    either one.
    """
    width = float(upper) - float(lower)
    if width <= 0.0:
        return 0.0

    def integrand(t):
        r = float(lower) + width * t
        return r ** index * np.exp(-float(inverse_length) * r)

    return width * _quad(integrand, 0.0, 1.0, epsabs=0.0, epsrel=1e-13,
                         limit=200)[0]


def _radial_power_moment(exponent, inverse_length, power, lower, upper):
    """Integral of r**(exponent + power) * exp(-inverse_length * r) over
    [lower, upper]. upper may be numpy.inf."""
    order = float(exponent) + float(power) + 1.0
    if inverse_length > 0.0:
        if lower > 0.0 and not np.isinf(upper):
            return _finite_panel_moment(order - 1.0, inverse_length,
                                        lower, upper)
        scale = 1.0 / float(inverse_length)
        if lower > 0.0:
            lo = _upper_incomplete_gamma(order, float(lower) / scale)
        elif order > 0.0:
            lo = _gamma_fn(order)
        else:
            raise ValueError(
                "the integral diverges at the origin for a non-positive order")
        hi = (0.0 if np.isinf(upper)
              else _upper_incomplete_gamma(order, float(upper) / scale))
        return scale ** order * (lo - hi)
    if np.isinf(upper):
        raise ValueError(
            "an unbounded region needs a strictly positive inverse decay length")
    if lower <= 0.0 and order <= 0.0:
        raise ValueError(
            "the integral diverges at the origin for a non-positive order")
    if order == 0.0:
        return float(np.log(float(upper) / float(lower)))
    return (float(upper) ** order - float(lower) ** order) / order


def _check_profile(transport, amplitudes, radii):
    """Validate a matched profile: transport table, amplitudes and radii."""
    try:
        tr, rad = _check_transport_and_radii(transport, radii)
    except ValueError as exc:
        raise ValueError("invalid profile: %s" % exc)
    try:
        amp = np.asarray(amplitudes, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("amplitudes must be an array of real numbers")
    if amp.ndim != 1:
        raise ValueError("amplitudes must be one-dimensional")
    if amp.size != tr.shape[0]:
        raise ValueError("amplitudes must have one entry per region")
    if not np.all(np.isfinite(amp)):
        raise ValueError("amplitudes must be finite")
    if np.any(amp <= 0.0):
        raise ValueError("amplitudes must be strictly positive")
    if tr[-1, 0] <= 0.0:
        raise ValueError(
            "the outermost inverse decay length must be strictly positive")
    return tr, amp, rad


def _region_edges(rad, n):
    """[(lower, upper)] for each region, innermost first, outermost unbounded."""
    lowers = np.concatenate(([0.0], rad))
    uppers = np.concatenate((rad, [np.inf]))
    return list(zip(lowers[:n], uppers[:n]))


def radial_normalisation(transport, amplitudes, radii):
    """Reference implementation of radial_normalisation."""
    tr, amp, rad = _check_profile(transport, amplitudes, radii)
    n = tr.shape[0]

    pieces = np.empty(n, dtype=np.float64)
    for j, (lo, hi) in enumerate(_region_edges(rad, n)):
        pieces[j] = amp[j] * _radial_power_moment(tr[j, 1], tr[j, 0], 1.0, lo, hi)

    total = float(np.sum(pieces))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("the profile has no finite, positive normalisation")
    summary = np.empty(n + 1, dtype=np.float64)
    summary[0] = total
    summary[1:] = pieces / total
    return summary

import numpy as np


def radial_density_profile(transport, amplitudes, radii, normalisation,
                                   r_values):
    """Reference implementation of radial_density_profile."""
    tr, amp, rad = _check_profile(transport, amplitudes, radii)

    if isinstance(normalisation, bool):
        raise ValueError("normalisation must be a real number, got a bool")
    try:
        norm = float(normalisation)
    except (TypeError, ValueError):
        raise ValueError("normalisation must be a real number")
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("normalisation must be finite and strictly positive")

    try:
        rv = np.asarray(r_values, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("r_values must be an array of real numbers")
    if rv.ndim != 1:
        raise ValueError("r_values must be one-dimensional")
    if rv.size < 1:
        raise ValueError("r_values must have at least one entry")
    if not np.all(np.isfinite(rv)):
        raise ValueError("r_values must be finite")
    if np.any(rv < 0.0):
        raise ValueError("r_values must be non-negative")

    # Region index of each query radius; an interface radius goes outward.
    index = np.searchsorted(rad, rv, side="right") if rad.size else np.zeros(
        rv.size, dtype=np.int64)
    index = np.asarray(index, dtype=np.int64)

    density = np.zeros(rv.size, dtype=np.float64)
    positive = rv > 0.0
    if np.any(positive):
        j = index[positive]
        r = rv[positive]
        density[positive] = (amp[j] * r ** (tr[j, 1] + 1.0)
                             * np.exp(-tr[j, 0] * r) / norm)
    return density

import numpy as np


def conditional_inverse_square_moments(transport, amplitudes, radii,
                                               normalisation):
    """Reference implementation of conditional_inverse_square_moments."""
    tr, amp, rad = _check_profile(transport, amplitudes, radii)
    n = tr.shape[0]
    if n < 2:
        raise ValueError("at least two regions are required")

    if isinstance(normalisation, bool):
        raise ValueError("normalisation must be a real number, got a bool")
    try:
        norm = float(normalisation)
    except (TypeError, ValueError):
        raise ValueError("normalisation must be a real number")
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("normalisation must be finite and strictly positive")

    edges = _region_edges(rad, n)
    moments = np.empty(n - 1, dtype=np.float64)
    for j in range(1, n):
        lo, hi = edges[j]
        moments[j - 1] = (
            amp[j] * _radial_power_moment(tr[j, 1], tr[j, 0], -1.0, lo, hi)
            / norm)
    return moments

import numpy as np


def _profile_quadrature(transport, radii):
    """Integrate each sensing region on panels uniform in log radius."""
    nodes, gl_weights = np.polynomial.legendre.leggauss(48)
    tr = np.asarray(transport, dtype=np.float64)
    rad = np.asarray(radii, dtype=np.float64)
    rules = []
    for j in range(1, tr.shape[0]):
        lo = float(rad[j - 1])
        if j < tr.shape[0] - 1:
            hi = float(rad[j])
        else:
            decay = float(tr[j, 0])
            if decay <= 0.0:
                raise ValueError("the outermost region does not decay")
            hi = lo + 100.0 / decay
        log_lo, log_hi = np.log(lo), np.log(hi)
        panels = max(1, int(np.ceil(log_hi - log_lo)))
        edges = np.linspace(log_lo, log_hi, panels + 1)
        r_all, w_all = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            half = 0.5 * (b - a)
            r = np.exp(0.5 * (a + b) + half * nodes)
            r_all.append(r)
            w_all.append(half * gl_weights * r)
        rules.append((np.concatenate(r_all), np.concatenate(w_all)))
    return rules


def sensing_entropy_production_rate(kappa_tilde=0.48,
                                            sigma_phi_sq=0.30, peclet=2.5,
                                            r_inner=0.06, r_outer=0.55,
                                            gamma_mid=0.50):
    """Reference implementation of sensing_entropy_production_rate."""
    scalars = (("kappa_tilde", kappa_tilde), ("sigma_phi_sq", sigma_phi_sq),
               ("peclet", peclet), ("r_inner", r_inner), ("r_outer", r_outer),
               ("gamma_mid", gamma_mid))
    values = {}
    for name, value in scalars:
        if isinstance(value, bool):
            raise ValueError("%s must be a real number, got a bool" % name)
        try:
            values[name] = float(value)
        except (TypeError, ValueError):
            raise ValueError("%s must be a real number" % name)
        if not np.isfinite(values[name]):
            raise ValueError("%s must be finite" % name)
    for name in ("kappa_tilde", "sigma_phi_sq", "peclet", "r_inner"):
        if values[name] <= 0.0:
            raise ValueError("%s must be strictly positive" % name)
    if values["r_outer"] <= values["r_inner"]:
        raise ValueError("r_outer must be strictly greater than r_inner")
    if not (0.0 <= values["gamma_mid"] <= 1.0):
        raise ValueError("gamma_mid must lie in [0, 1]")

    gamma_values = np.array([0.0, values["gamma_mid"], 1.0], dtype=np.float64)
    radii = np.array([values["r_inner"], values["r_outer"]], dtype=np.float64)

    stats = sensor_alignment_statistics(values["sigma_phi_sq"],
                                                gamma_values)
    mean_cos = stats[:, 1]

    transport = region_transport_parameters(values["kappa_tilde"],
                                                    mean_cos, values["peclet"])
    amplitudes = matched_density_amplitudes(transport, radii)
    summary = radial_normalisation(transport, amplitudes, radii)
    normalisation = float(summary[0])
    moments = conditional_inverse_square_moments(
        transport, amplitudes, radii, normalisation)

    # The two restricted averages are integrals of the matched profile that
    # step 06 evaluates, so the density readout is what certifies them before
    # the rate is assembled from them. Integrating that readout region by
    # region must reproduce step 07's closed-form values; a profile that is
    # wrong anywhere, replaced by zeros, or replaced by any constant fails
    # here and no rate is returned. Step 02 is called directly as well, since
    # it is otherwise reached only through step 03.
    closure_series_coefficients(values["kappa_tilde"] * mean_cos)
    profile_moments = np.empty(moments.size, dtype=np.float64)
    for j, (r_nodes, r_weights) in enumerate(
            _profile_quadrature(transport, radii)):
        density = radial_density_profile(transport, amplitudes, radii,
                                                 normalisation, r_nodes)
        profile_moments[j] = float(np.sum(r_weights * density
                                          / (r_nodes * r_nodes)))
    if not np.all(np.isfinite(profile_moments)):
        raise ValueError("the matched profile is not integrable over a "
                         "sensing region")
    if not np.all(np.abs(profile_moments - moments)
                  <= 1e-7 * np.abs(moments) + 1e-15):
        raise ValueError("the radial integral of the matched density does not "
                         "reproduce the restricted inverse-square averages")

    weights = stats[1:, 3]
    rate = float(np.sum(weights * moments)
                 / (values["sigma_phi_sq"] * values["peclet"]))

    return rate
SCICODE_GOLD_EOF
