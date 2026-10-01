"""
The two valley ground states are degenerate at zeroth order, one the complex conjugate of the other, and the intervalley part of the Hamiltonian lifts the degeneracy. At first order in degenerate perturbation theory the problem closes on the two-dimensional subspace spanned by the two valley states, the effective Hamiltonian there is [[0, Delta], [Delta*, 0]] with a complex intervalley coupling Delta, the two eigenvalues are plus and minus |Delta|, and the valley splitting is E_VS = 2 |Delta|.

Written in terms of the slowly varying envelopes, the coupling is a sum over the harmonics that the Bloch factors of the two valleys can exchange. The valley-coupling selection rule G - G' = n G0 between the plane-wave coefficients of the two Bloch factors defines the overlap sums C_n, and each order n picks out the component of the confinement at wave number 2 k0 + n G0z. For the growth-direction problem,

Delta = sum_n C_n integral dz exp(-i (2 k0 + n G0z) z) conj(f_+(z)) U(z) f_-(z),

and since the envelope of the -k0 valley is the complex conjugate of that of the +k0 valley, f_- = conj(f_+), the integrand carries the square of conj(f_+) and not |f_+|^2. The integral is taken on the supercell as dz times the sum over grid points; for an envelope confined well inside the cell this is the trapezoidal rule of a function that vanishes at both ends of the cell and it converges faster than any power of dz.

The order n = 0 samples the confinement at 2 k0, the short-period resonance of an oscillating germanium profile. The order n = -1 samples it at 2 k0 - G0z = -2 k1, the long-period resonance at twice the distance from the valley to the strained zone boundary, and its coefficient is proportional to the in-plane shear strain, so that this channel is closed in an unsheared crystal. The stage reports the contribution of each order separately, so that the dominant channel of a given profile can be identified.

The same formula accepts any envelope and any weighting function in place of U, which later stages use: a real envelope from a conventional calculation, and a unit weight, which measures how the coupling responds to a constant added to the confinement.

Returns
-------
dict holding the floats delta_real, delta_imag and delta_abs, the coupling in eV; splitting, E_VS = 2 |Delta| in eV; the arrays contribution_real and contribution_imag, the contribution of each order in eV in the order supplied; and the integer dominant_order, the order whose contribution has the largest modulus.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def intervalley_coupling(
    z: np.ndarray,
    envelope: np.ndarray,
    potential: np.ndarray,
    valley_wavenumber: float,
    zone_vector: float,
    orders,
    coefficients,
) -> dict:
    """Evaluate the first-order intervalley coupling of a valley envelope and the valley splitting it implies.

    Parameters
    ----------
    z : np.ndarray
        Uniform supercell positions in nm.
    envelope : np.ndarray
        Slowly varying envelope of the +k0 valley, normalised in the continuum sense.
    potential : np.ndarray
        Real weighting function in eV, usually the confinement energy.
    valley_wavenumber : float
        Valley minimum k0 in reciprocal nm, above zero.
    zone_vector : float
        Strained growth-direction reciprocal-lattice vector G0z in reciprocal nm, above zero.
    orders : sequence of int
        Orders n, without repetition.
    coefficients : sequence of complex
        Overlap sums C_n.

    Returns
    -------
    dict
        Under the keys delta_real, delta_imag, delta_abs, splitting, contribution_real, contribution_imag and dominant_order.

    Raises
    ------
    ValueError
        When the grid fails to be uniform and finite, when the envelope or weight fails to be finite or to match the grid, when the weight is complex, when the envelope fails to be normalised, when k0 or G0z fails to be finite and above zero, or when the orders and coefficients fail to be integers and finite values of equal length without repetition.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_intervalley_coupling(
    z: np.ndarray,
    envelope: np.ndarray,
    potential: np.ndarray,
    valley_wavenumber: float,
    zone_vector: float,
    orders,
    coefficients,
) -> dict:
    """Reference implementation."""
    grid = np.asarray(z, dtype=float)
    if grid.ndim != 1 or grid.size < 8 or not np.all(np.isfinite(grid)):
        raise ValueError("z must be a finite one-dimensional grid")
    steps = np.diff(grid)
    dz = float(steps[0])
    if dz <= 0.0 or np.max(np.abs(steps - dz)) > 1e-9 * dz:
        raise ValueError("z must be uniformly increasing")
    f = np.asarray(envelope)
    w = np.asarray(potential)
    if f.shape != grid.shape or not np.all(np.isfinite(f)):
        raise ValueError("envelope must be finite and match z")
    if w.shape != grid.shape or np.iscomplexobj(w) or not np.all(np.isfinite(w)):
        raise ValueError("potential must be real, finite and match z")
    if abs(dz * float(np.sum(np.abs(f) ** 2)) - 1.0) > 1e-8:
        raise ValueError("envelope must be normalised")
    k0 = float(valley_wavenumber)
    g0 = float(zone_vector)
    if not (math.isfinite(k0) and math.isfinite(g0)) or k0 <= 0.0 or g0 <= 0.0:
        raise ValueError("valley_wavenumber and zone_vector must be finite and above zero")
    ns = [int(n) for n in orders]
    for n, raw in zip(ns, orders):
        if isinstance(raw, bool) or float(raw) != n:
            raise ValueError("orders must be integers")
    cs = [complex(c) for c in coefficients]
    if not ns or len(ns) != len(cs) or len(set(ns)) != len(ns):
        raise ValueError("orders and coefficients must be non-empty, of equal length and without repetition")
    if not all(math.isfinite(c.real) and math.isfinite(c.imag) for c in cs):
        raise ValueError("coefficients must be finite")

    density = np.conj(f.astype(complex)) ** 2 * w.astype(float)
    parts = np.array([c * dz * np.sum(np.exp(-1j * (2.0 * k0 + n * g0) * grid) * density)
                      for n, c in zip(ns, cs)])
    delta = complex(np.sum(parts))
    return {
        "delta_real": delta.real,
        "delta_imag": delta.imag,
        "delta_abs": abs(delta),
        "splitting": 2.0 * abs(delta),
        "contribution_real": parts.real,
        "contribution_imag": parts.imag,
        "dominant_order": ns[int(np.argmax(np.abs(parts)))],
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
def normalised(f):
    return f / math.sqrt((z[1] - z[0]) * float(np.sum(np.abs(f) ** 2)))
def digest(out):
    return (round(out["delta_real"] * 1e6, 6), round(out["delta_imag"] * 1e6, 6), round(out["delta_abs"] * 1e6, 6),
            round(out["splitting"] * 1e6, 6), [round(float(v) * 1e6, 6) for v in out["contribution_real"]],
            [round(float(v) * 1e6, 6) for v in out["contribution_imag"]], out["dominant_order"])
"""


def test_cases():
    return [
        {
            # a complex Gaussian envelope with a chirp, in a wiggle-well weight near the 2 k1 resonance
            "setup": SETUP + """
f = normalised(np.exp(-((z + 3.0) / 2.2) ** 2) * np.exp(0.4j * z + 0.03j * z * z))
xi = 0.5 * (np.tanh((z + 8.0) / 0.4) - np.tanh(z / 0.4))
u = 0.15 * (1.0 - xi) + 0.05 * (1.0 + np.cos(2.0 * math.pi / (3 * A) * z)) * xi - 3.0e-3 * z
""" + FLAT,
            "call": "flat(digest(intervalley_coupling(z, f, u, K0, G0, ORDERS, C)))",
            "gold_call": "flat(digest(_oracle_intervalley_coupling(z, f, u, K0, G0, ORDERS, C)))",
        },
        {
            # conjugation and the harmonic phases: an envelope multiplied by a global phase must rotate
            # Delta by twice that phase and leave the splitting alone, and a real envelope in a unit
            # weight must return the Fourier component of its square at 2 k0 + n G0z
            "setup": SETUP + """
f = normalised(np.exp(-((z + 2.0) / 1.5) ** 2) * (1.0 + 0.3j * np.tanh(z + 2.0)))
g = normalised(np.exp(-((z + 2.0) / 0.3) ** 2))
u = 0.2 * np.exp(-((z + 1.0) / 0.25) ** 2)
def rotate(fn):
    a = fn(z, f, u, K0, G0, ORDERS, C)
    b = fn(z, f * np.exp(0.7j), u, K0, G0, ORDERS, C)
    da = complex(a["delta_real"], a["delta_imag"])
    db = complex(b["delta_real"], b["delta_imag"])
    c = fn(z, g, np.ones_like(z), K0, G0, (-1, 0), (1.0, 1.0))
    return (round(b["splitting"] - a["splitting"], 15), round(abs(db - da * np.exp(-1.4j)), 15),
            round(a["splitting"] * 1e6, 6), [round(float(v), 9) for v in c["contribution_real"]],
            [round(float(v), 9) for v in c["contribution_imag"]], c["dominant_order"])
""" + FLAT,
            "call": "flat(rotate(intervalley_coupling))",
            "gold_call": "flat(rotate(_oracle_intervalley_coupling))",
        },
        {
            # linearity in the weight and in the coefficients: doubling U doubles Delta, and dropping
            # every order but one leaves exactly that order's contribution
            "setup": SETUP + """
f = normalised(np.exp(-((z + 4.0) / 1.8) ** 2) * np.exp(0.2j * z))
u = 0.3 * np.cos(2.0 * math.pi / (3 * A) * z) * np.exp(-((z + 4.0) / 4.0) ** 2)
def linear(fn):
    a = fn(z, f, u, K0, G0, ORDERS, C)
    b = fn(z, f, 2.0 * u, K0, G0, ORDERS, C)
    c = fn(z, f, u, K0, G0, (-1,), (C[3],))
    return (round(b["delta_abs"] / a["delta_abs"], 12), round(c["delta_real"] - a["contribution_real"][3], 15),
            round(c["delta_imag"] - a["contribution_imag"][3], 15), a["dominant_order"], round(a["splitting"] * 1e6, 6))
""" + FLAT,
            "call": "flat(linear(intervalley_coupling))",
            "gold_call": "flat(linear(_oracle_intervalley_coupling))",
        },
        {
            "setup": SETUP + """
f = normalised(np.exp(-((z + 3.0) / 2.0) ** 2))
u = 0.1 * np.ones_like(z)
def verdict(fn, **kw):
    args = dict(z=z, envelope=f, potential=u, valley_wavenumber=K0, zone_vector=G0, orders=ORDERS, coefficients=C)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(intervalley_coupling, envelope=2.0 * f), verdict(intervalley_coupling, potential=u + 0.0j), verdict(intervalley_coupling, envelope=f[:-1]), verdict(intervalley_coupling, orders=(0, 0), coefficients=(1.0, 1.0)), verdict(intervalley_coupling, coefficients=C[:-1]), verdict(intervalley_coupling, valley_wavenumber=-1.0), verdict(intervalley_coupling, z=z[::-1]), verdict(intervalley_coupling)))",
            "gold_call": "flat((verdict(_oracle_intervalley_coupling, envelope=2.0 * f), verdict(_oracle_intervalley_coupling, potential=u + 0.0j), verdict(_oracle_intervalley_coupling, envelope=f[:-1]), verdict(_oracle_intervalley_coupling, orders=(0, 0), coefficients=(1.0, 1.0)), verdict(_oracle_intervalley_coupling, coefficients=C[:-1]), verdict(_oracle_intervalley_coupling, valley_wavenumber=-1.0), verdict(_oracle_intervalley_coupling, z=z[::-1]), verdict(_oracle_intervalley_coupling)))",
        },
    ]
