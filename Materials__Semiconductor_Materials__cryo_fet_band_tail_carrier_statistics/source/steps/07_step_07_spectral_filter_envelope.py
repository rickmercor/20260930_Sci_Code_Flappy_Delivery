"""
A constant added to the confinement, U to U + U0, is a change of energy reference and nothing else, so every physical prediction must be blind to it. The coupling of stage 5 is linear in the weighting function, so under that change Delta moves to Delta + U0 R, where

R = sum_n C_n integral dz exp(-i (2 k0 + n G0z) z) conj(f(z))^2

is stage 5 evaluated with a unit weight. The splitting is blind to the reference exactly when R = 0. Whether it is depends on the envelope and not on the potential. Write conj(f)^2 = exp(2 i k0 z) conj(F)^2 with F = exp(i k0 z) f. If every plane wave of F lies in the open sector (0, G0z / 2), the wave numbers of conj(F)^2 lie in the open interval (-G0z, 0), and the n-th harmonic of R picks out its plane wave at n G0z, a whole multiple of G0z that the open interval never contains; every term vanishes and R is identically zero. An envelope with plane waves outside the sector has no such protection.

This stage measures R for a given envelope and then applies the cheapest repair available to an envelope computed without the sector restriction: project it onto its sector after the fact. With F_hat the discrete Fourier transform of exp(i k0 z_j) f(z_j) on the supercell, the filtered envelope keeps the indices m = 1, ..., N_FBZ / 2 - 1, zeroes all others, transforms back, removes the carrier again, and is renormalised so that dz sum_j |f|^2 = 1. The retained weight is the squared norm of the projection divided by that of the original, before renormalisation. The filtered envelope is complex in general even when the input is real, because its spectrum is no longer symmetric about the valley. Its R is evaluated as well, and on the grid it vanishes to rounding for the same reason as in the continuum: the wave numbers of conj(F)^2 are sums of two sector grid wave numbers, strictly between zero and G0z in magnitude, and no such sum is congruent to a whole multiple of G0z modulo the resolved band N_BZ G0z.

Filtering is not the same construction as solving the band-limited eigenproblem. It restores the protection of R, but the filtered function is not an eigenstate of the band-limited Hamiltonian, so its coupling approximates the band-limited result rather than reproducing it.

Returns
-------
dict holding envelope, the complex filtered and renormalised envelope; retained_weight, the fraction of the norm the projection keeps; ambiguity_real, ambiguity_imag and ambiguity_abs, the response R of the input envelope; and filtered_ambiguity_abs, the modulus of R for the filtered envelope. R is dimensionless, since Delta is in eV and U0 in eV.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectral_filter_envelope(
    z: np.ndarray,
    envelope: np.ndarray,
    valley_wavenumber: float,
    zone_vector: float,
    orders,
    coefficients,
) -> dict:
    """Measure the reference response of an envelope, project it onto its valley sector and measure again.

    Parameters
    ----------
    z : np.ndarray
        Uniform supercell positions in nm.
    envelope : np.ndarray
        Slowly varying envelope, normalised in the continuum sense.
    valley_wavenumber : float
        Valley minimum k0 in reciprocal nm, inside the open sector.
    zone_vector : float
        Strained growth-direction reciprocal-lattice vector G0z in reciprocal nm, above zero.
    orders : sequence of int
        Orders n, without repetition.
    coefficients : sequence of complex
        Overlap sums C_n.

    Returns
    -------
    dict
        Under the keys envelope, retained_weight, ambiguity_real, ambiguity_imag, ambiguity_abs and filtered_ambiguity_abs.

    Raises
    ------
    ValueError
        When the grid, envelope, orders or coefficients fail the checks of the coupling stage, when the cell fails to hold an even whole number of grid wave numbers per zone, when k0 fails to lie strictly inside the open sector, or when the projection retains no weight.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_spectral_filter_envelope(
    z: np.ndarray,
    envelope: np.ndarray,
    valley_wavenumber: float,
    zone_vector: float,
    orders,
    coefficients,
) -> dict:
    """Reference implementation."""
    grid = np.asarray(z, dtype=float)
    f = np.asarray(envelope)
    ones = np.ones(grid.shape if grid.ndim == 1 else (0,))
    before = _oracle_intervalley_coupling(grid, f, ones, valley_wavenumber, zone_vector, orders, coefficients)  # noqa: F821

    g0 = float(zone_vector)
    k0 = float(valley_wavenumber)
    if not 0.0 < k0 < 0.5 * g0:
        raise ValueError("valley_wavenumber must lie strictly inside the open sector")
    n = grid.size
    dz = float(grid[1] - grid[0])
    length = n * dz
    per_zone_real = length * g0 / (2.0 * math.pi)
    per_zone = int(round(per_zone_real))
    if per_zone < 4 or per_zone % 2 or abs(per_zone_real - per_zone) > 1e-7 * per_zone:
        raise ValueError("the cell must hold an even whole number of grid wave numbers per zone")

    carrier = np.fft.fft(np.exp(1j * k0 * grid) * f)
    mask = np.zeros(n, dtype=bool)
    mask[1:per_zone // 2] = True
    kept = np.where(mask, carrier, 0.0)
    total = float(np.sum(np.abs(carrier) ** 2))
    retained = float(np.sum(np.abs(kept) ** 2)) / total
    if retained <= 0.0:
        raise ValueError("the projection retains no weight")
    filtered = np.exp(-1j * k0 * grid) * np.fft.ifft(kept)
    filtered = filtered / math.sqrt(dz * float(np.sum(np.abs(filtered) ** 2)))

    after = _oracle_intervalley_coupling(grid, filtered, ones, valley_wavenumber, zone_vector, orders, coefficients)  # noqa: F821
    return {
        "envelope": filtered,
        "retained_weight": retained,
        "ambiguity_real": before["delta_real"],
        "ambiguity_imag": before["delta_imag"],
        "ambiguity_abs": before["delta_abs"],
        "filtered_ambiguity_abs": after["delta_abs"],
    }

# =============================================================================
# TEST CASES
# =============================================================================

FLAT = """
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            out.extend(flat(v))
        return tuple(out)
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    if isinstance(x, bool):
        return (int(x),)
    if isinstance(x, complex):
        return (x.real, x.imag)
    return (x,)
"""

SETUP = """
import math
import numpy as np
A = 0.543
G0 = 4.0 * math.pi / A * (1.0 + 0.00884297)
K0 = 0.8394 * 2.0 * math.pi / A
ORDERS = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
C = (1.84e-3, 5.99e-4, -2.44e-2, -3.18e-2, -0.221, -4.02e-4, 1.58e-3, 1.04e-5, 3.35e-5)
L = 2.0 * math.pi * 128 / G0
z = 10.0 - L + (L / 1280) * np.arange(1280)
dz = z[1] - z[0]
def normalised(f):
    return f / math.sqrt(dz * float(np.sum(np.abs(f) ** 2)))
"""


def test_cases():
    return [
        {
            # a sharply kinked real envelope: large leakage, large R, and R removed by the filter
            "setup": SETUP + """
f = normalised(np.exp(-np.abs(z + 3.0) / 0.35) + 0.4 * np.exp(-((z + 5.0) / 0.2) ** 2))
def digest(out):
    g = out["envelope"]
    pick = [int(np.argmax(np.abs(g))), 600, 900]
    return (round(out["retained_weight"], 10), round(out["ambiguity_real"], 10), round(out["ambiguity_imag"], 10),
            round(out["ambiguity_abs"], 10), int(out["filtered_ambiguity_abs"] < 1e-12),
            round(dz * float(np.sum(np.abs(g) ** 2)), 12),
            [(round(complex(g[i]).real, 8), round(complex(g[i]).imag, 8)) for i in pick])
""" + FLAT,
            "call": "flat(digest(spectral_filter_envelope(z, f, K0, G0, ORDERS, C)))",
            "gold_call": "flat(digest(_oracle_spectral_filter_envelope(z, f, K0, G0, ORDERS, C)))",
        },
        {
            # an envelope already built from sector plane waves only must pass through unchanged, up to
            # rounding, with R already zero before the filter
            "setup": SETUP + """
m = np.arange(1, 64)
coeff = np.exp(-((2.0 * math.pi / L * m - K0) / 0.6) ** 2) * np.exp(0.8j * m)
F = np.exp(1j * np.outer(z, 2.0 * math.pi / L * m)) @ coeff
f = normalised(np.exp(-1j * K0 * z) * F)
def passthrough(fn):
    out = fn(z, f, K0, G0, ORDERS, C)
    return (round(out["retained_weight"], 12), int(out["ambiguity_abs"] < 1e-12), int(out["filtered_ambiguity_abs"] < 1e-12),
            round(float(np.max(np.abs(out["envelope"] - f))), 10))
""" + FLAT,
            "call": "flat(passthrough(spectral_filter_envelope))",
            "gold_call": "flat(passthrough(_oracle_spectral_filter_envelope))",
        },
        {
            # smoother envelopes leak less: R and the discarded weight must fall together as the width grows
            "setup": SETUP + """
def trend(fn):
    rows = []
    for width in (0.3, 0.8, 2.0):
        f = normalised(np.exp(-((z + 3.0) / width) ** 2))
        out = fn(z, f, K0, G0, ORDERS, C)
        rows.append((round(1.0 - out["retained_weight"], 12), round(out["ambiguity_abs"], 12)))
    return rows
""" + FLAT,
            "call": "flat(trend(spectral_filter_envelope))",
            "gold_call": "flat(trend(_oracle_spectral_filter_envelope))",
        },
        {
            "setup": SETUP + """
f = normalised(np.exp(-((z + 3.0) / 1.0) ** 2))
def verdict(fn, **kw):
    args = dict(z=z, envelope=f, valley_wavenumber=K0, zone_vector=G0, orders=ORDERS, coefficients=C)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(spectral_filter_envelope, envelope=3.0 * f), verdict(spectral_filter_envelope, zone_vector=0.97 * G0), verdict(spectral_filter_envelope, valley_wavenumber=0.55 * G0), verdict(spectral_filter_envelope, envelope=np.where(z > 5.0, np.nan, f)), verdict(spectral_filter_envelope, coefficients=C[:3]), verdict(spectral_filter_envelope)))",
            "gold_call": "flat((verdict(_oracle_spectral_filter_envelope, envelope=3.0 * f), verdict(_oracle_spectral_filter_envelope, zone_vector=0.97 * G0), verdict(_oracle_spectral_filter_envelope, valley_wavenumber=0.55 * G0), verdict(_oracle_spectral_filter_envelope, envelope=np.where(z > 5.0, np.nan, f)), verdict(_oracle_spectral_filter_envelope, coefficients=C[:3]), verdict(_oracle_spectral_filter_envelope)))",
        },
    ]
