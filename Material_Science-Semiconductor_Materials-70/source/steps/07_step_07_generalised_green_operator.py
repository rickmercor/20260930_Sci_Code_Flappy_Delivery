"""
The Green operator used so far follows from the spectral derivative rule, in which differentiating an exponential multiplies it by its own wavevector. That is not the only rule available. Replacing the derivative by a difference quotient taken over a finite spacing h gives a different operator, and because the fields here are continuous functions rather than tables of nodal values the spacing need not be the grid spacing: it is a free parameter, and letting it run recovers a one-parameter family that contains the spectral rule at one end and the familiar finite-difference construction at the other.

Take the potential forward and the current backward, which is the pairing that makes the resulting operator symmetric,

$$
forward: [phi(x + h) - phi(x)] / h, backward: [J(x) - J(x - h)] / h,
$$

each applied in its own direction with its own spacing. Acting on the exponential of index i these multiply it by

$$
[exp(i * xi_i * h) - 1] / h and [1 - exp(-i * xi_i * h)] / h
$$

respectively. Collecting the second derivative that the forward rule applied to the potential and then contracted with the backward rule produces gives a real, non-negative quantity per direction,

$$
alpha_i = 2 * sin(xi_i * h / 2) / h,
$$

while the first-derivative factor that contracts with the polarisation is the complex

$$
beta_i = [1 - exp(-i * xi_i * h)] / (i * h).
$$

The two are not independent: the modulus squared of beta in each direction is exactly alpha squared in that direction, which is worth checking numerically because it is the identity that lets the operator be written compactly. With it the generalised operator is

$$
Gamma_G(xi) = conj(beta) tensor beta / [c0 * conj(beta) . beta],
$$

the denominator being c0 times the sum of the alpha squared over the directions, a real positive number. The denominator carries that sum to the first power and not to the second. Any expression that squares it is not an operator of the right physical dimension, and the error is easy to make. What it does is not subtle: the squared denominator divides the operator by a quantity that is tiny in these units and varies by more than three orders of magnitude across the retained modes, which suppresses the update altogether, so the field collapses to the uniform applied mean, the macroscopic conductivity collapses onto the arithmetic mean of the sampled conductivity, and the equilibrium residual never falls below any useful tolerance.

Two limits fix the family and both should be checked rather than assumed. As the spacing goes to zero, beta tends to the wavevector, alpha tends to the wavevector as well, and the operator tends to the spectral one; a candidate expression that fails this limit is wrong however plausible it looks, and this is the cheapest available test of the assembly. When the spacing equals the grid spacing, the operator is the one a finite-difference discretisation on that grid produces, with the difference that here the field remains a trigonometric polynomial defined everywhere rather than a table of nodal values. Between the two the spacing interpolates continuously.

The structural property survives the generalisation: c0 times the operator is still an orthogonal projector, idempotent and of unit trace, now onto the direction of the conjugated symbol rather than onto the wavevector. What changes is which direction that is, and therefore which fields the iteration is allowed to produce. Whether moving that direction removes the oscillations that a truncated series shows near a material discontinuity is a separate question, and the answer is not settled by the fact that the operator is a projector.

Returns
-------
dict, the difference-quotient symbol and the generalised Green operator built from it, with the defects that verify its structure and its spectral limit.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def generalised_green_operator(
    wavevector,
    spacing: float,
    reference: float,
    retained,
) -> dict:
    """Build the difference-quotient Green operator at an arbitrary spacing and verify its structure.

    Parameters
    ----------
    wavevector : array
        Wavevector components of shape (2, n, n) in reciprocal metre.
    spacing : float
        Difference spacing in metre, above zero.
    reference : float
        Reference conductivity in siemens per metre, above zero.
    retained : array
        Boolean mask of shape (n, n) marking the retained non-zero modes.

    Returns
    -------
    dict
        Under the keys symbol, alpha, green, symbol_defect, spectral_defect, projector_defect and trace_defect. symbol is a complex array and alpha a real array, both of the shape (2, n, n) of the supplied wavevector, and green is a complex array of shape (2, 2, n, n) laid out as the spectral Green operator is. The remaining four entries are scalars.

    Raises
    ------
    ValueError
        When the wavevector is not a real array of shape (2, n, n), when the mask does not have shape (n, n) or marks no mode, or when the spacing or the reference conductivity fails to be finite and above zero.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _positive_float(value, label):
    """Return an argument as a float once it is known to be finite and above zero."""
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError("%s must be finite and above zero" % label)
    return out


def _green_from_symbol(symbol, reference):
    """Assemble conj(beta) tensor beta divided by reference times conj(beta) . beta."""
    n = symbol.shape[1]
    weight = np.real(np.sum(np.conj(symbol) * symbol, axis=0))
    green = np.zeros((2, 2, n, n), complex)
    live = weight > 0.0
    for a in range(2):
        for b in range(2):
            numerator = np.conj(symbol[a]) * symbol[b]
            green[a, b][live] = numerator[live] / (reference * weight[live])
    return green


def _projector_defects(green, reference, retained):
    """Return the departures of reference times green from idempotence and from unit trace."""
    scaled = reference * green
    square = np.zeros_like(scaled)
    for a in range(2):
        for b in range(2):
            square[a, b] = scaled[a, 0] * scaled[0, b] + scaled[a, 1] * scaled[1, b]
    idem = 0.0
    for a in range(2):
        for b in range(2):
            idem = max(idem, float(np.max(np.abs((square[a, b] - scaled[a, b])[retained]))))
    trace = float(np.max(np.abs((scaled[0, 0] + scaled[1, 1])[retained] - 1.0)))
    return idem, trace


def _oracle_generalised_green_operator(
    wavevector,
    spacing: float,
    reference: float,
    retained,
) -> dict:
    """Reference implementation."""
    wave = np.asarray(wavevector, dtype=float)
    if wave.ndim != 3 or wave.shape[0] != 2 or wave.shape[1] != wave.shape[2]:
        raise ValueError("wavevector must have shape (2, n, n)")
    if not np.all(np.isfinite(wave)):
        raise ValueError("wavevector must be finite")
    n = wave.shape[1]
    keep = np.asarray(retained, dtype=bool)
    if keep.shape != (n, n):
        raise ValueError("retained must have shape (n, n)")
    if not keep.any():
        raise ValueError("at least one mode must be retained")
    h = _positive_float(spacing, "spacing")
    c0 = _positive_float(reference, "reference")

    symbol = (1.0 - np.exp(-1j * wave * h)) / (1j * h)
    alpha = 2.0 * np.sin(wave * h / 2.0) / h
    green = _green_from_symbol(symbol, c0)

    scale = float(np.max(np.abs(wave)))
    if scale == 0.0:
        scale = 1.0
    symbol_defect = float(np.max(np.abs(np.conj(symbol) * symbol - alpha ** 2))) / scale ** 2
    spectral_defect = float(np.max(np.abs(symbol - wave))) / scale
    idem, trace = _projector_defects(green, c0, keep)
    return {
        "symbol": symbol,
        "alpha": alpha,
        "green": green,
        "symbol_defect": symbol_defect,
        "spectral_defect": spectral_defect,
        "projector_defect": idem,
        "trace_defect": trace,
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
    return (x,)
"""

SETUP = """
import numpy as np
def lattice(N, M, L, c0):
    n = N + 1
    idx = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    wave = 2.0 * np.pi * idx / L
    k = np.zeros((2, n, n))
    k[0] = wave[:, None]
    k[1] = wave[None, :]
    keep = np.abs(idx) <= M // 2
    ret = keep[:, None] & keep[None, :]
    ret[0, 0] = False
    beta = k.astype(complex)
    w = np.real(np.sum(np.conj(beta) * beta, axis=0))
    g = np.zeros((2, 2, n, n), complex)
    live = w > 0.0
    for a in range(2):
        for b in range(2):
            num = np.conj(beta[a]) * beta[b]
            g[a, b][live] = num[live] / (c0 * w[live])
    return k, ret, g, L / n
"""


def test_cases():
    return [
        {
            # the two identities that fix the symbol, and the projector structure, at
            # several spacings including one far below and one at the grid spacing
            "setup": SETUP + """
def digest(fn, N, M, ratio, L=1.0, c0=50.5):
    k, ret, g, dx = lattice(N, M, L, c0)
    out = fn(k, ratio * dx, c0, ret)
    gg = np.asarray(out["green"])
    al = np.asarray(out["alpha"])
    sy = np.asarray(out["symbol"])
    vals = [int(out["symbol_defect"] < 1.0e-12), int(out["projector_defect"] < 1.0e-12),
            int(out["trace_defect"] < 1.0e-12), round(float(out["spectral_defect"]), 8)]
    # the operator itself, so that a perturbed assembly cannot pass on the flags alone
    for (a, b) in ((0, 1), (1, 1), (2, 1)):
        vals.append(round(float(np.real(gg[0, 0, a, b])), 13))
        vals.append(round(float(np.imag(gg[0, 1, a, b])), 13))
        vals.append(round(float(al[1, a, b]), 8))
        vals.append(round(float(np.real(sy[0, a, b])), 8))
    vals.append(round(float(np.sum(np.abs(gg[:, :, ret]))), 10))
    # whole-array checksums for the two symbol arrays
    vals.append(round(float(np.abs(sy).sum()), 6))
    vals.append(round(float(np.abs(np.imag(sy)).sum()), 6))
    vals.append(round(float(np.abs(al).sum()), 6))
    return tuple(vals)
""" + FLAT,
            "call": "flat((digest(generalised_green_operator, 8, 8, 1.0), digest(generalised_green_operator, 8, 8, 0.5), digest(generalised_green_operator, 8, 4, 0.25), digest(generalised_green_operator, 12, 12, 1.0e-7), digest(generalised_green_operator, 12, 6, 2.0)))",
            "gold_call": "flat((digest(_oracle_generalised_green_operator, 8, 8, 1.0), digest(_oracle_generalised_green_operator, 8, 8, 0.5), digest(_oracle_generalised_green_operator, 8, 4, 0.25), digest(_oracle_generalised_green_operator, 12, 12, 1.0e-7), digest(_oracle_generalised_green_operator, 12, 6, 2.0)))",
        },
        {
            # the spectral limit: as the spacing falls the operator must approach the
            # one built from the wavevector itself, and the approach must be first order
            "setup": SETUP + """
def digest(fn, N, M):
    L, c0 = 1.0, 50.5
    k, ret, g, dx = lattice(N, M, L, c0)
    d = []
    for r in (1.0e-3, 1.0e-4, 1.0e-5):
        out = fn(k, r * dx, c0, ret)
        gg = np.asarray(out["green"])
        d.append(float(np.max(np.abs(gg[:, :, ret] - g[:, :, ret]))) * c0)
    order = tuple(round(d[i] / d[i + 1], 3) for i in range(len(d) - 1))
    # carry the magnitudes as well as their ratios, so the limit is pinned and not only its rate
    mags = tuple(round(x, 10) for x in d)
    mid = fn(k, 0.5 * dx, c0, ret)
    gm = np.asarray(mid["green"])
    return order + mags + (int(d[0] > d[1] > d[2] > 0.0), int(d[2] < 1.0e-4),
                           round(float(np.sum(np.abs(gm[:, :, ret]))), 10))
""" + FLAT,
            "call": "flat((digest(generalised_green_operator, 8, 8), digest(generalised_green_operator, 12, 8)))",
            "gold_call": "flat((digest(_oracle_generalised_green_operator, 8, 8), digest(_oracle_generalised_green_operator, 12, 8)))",
        },
        {
            # the operator entries themselves at a spacing equal to the grid spacing,
            # sampled at several modes, and their dependence on c0 and on the spacing
            "setup": SETUP + """
def digest(fn, N, ratio, c0):
    L = 1.0
    k, ret, g, dx = lattice(N, N, L, c0)
    out = fn(k, ratio * dx, c0, ret)
    gg = np.asarray(out["green"])
    al = np.asarray(out["alpha"])
    sy = np.asarray(out["symbol"])
    vals = []
    for (a, b) in ((0, 1), (1, 2), (3, 3), (2, 0)):
        vals.append(round(float(np.real(gg[0, 0, a, b])), 13))
        vals.append(round(float(np.real(gg[0, 1, a, b])), 13))
        vals.append(round(float(np.imag(gg[0, 1, a, b])), 13))
        vals.append(round(float(al[0, a, b]), 8))
        vals.append(round(float(np.imag(sy[1, a, b])), 8))
    return tuple(vals)
""" + FLAT,
            "call": "flat((digest(generalised_green_operator, 8, 1.0, 50.5), digest(generalised_green_operator, 8, 0.5, 50.5), digest(generalised_green_operator, 8, 1.0, 202.0)))",
            "gold_call": "flat((digest(_oracle_generalised_green_operator, 8, 1.0, 50.5), digest(_oracle_generalised_green_operator, 8, 0.5, 50.5), digest(_oracle_generalised_green_operator, 8, 1.0, 202.0)))",
        },
        {
            "setup": SETUP + """
def verdict(fn, ratio=0.5, c0=50.5, wave=None, ret=None):
    k, r, g, dx = lattice(8, 8, 1.0, 50.5)
    if wave is not None:
        k = wave
    if ret is not None:
        r = ret
    try:
        fn(k, ratio * dx, c0, r)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(generalised_green_operator), verdict(generalised_green_operator, ratio=0.0), verdict(generalised_green_operator, ratio=-1.0), verdict(generalised_green_operator, c0=0.0), verdict(generalised_green_operator, c0=float('nan')), verdict(generalised_green_operator, wave=__import__('numpy').zeros((9, 9))), verdict(generalised_green_operator, ret=__import__('numpy').zeros((9, 9), bool)), verdict(generalised_green_operator, ret=__import__('numpy').ones((8, 8), bool))))",
            "gold_call": "flat((verdict(_oracle_generalised_green_operator), verdict(_oracle_generalised_green_operator, ratio=0.0), verdict(_oracle_generalised_green_operator, ratio=-1.0), verdict(_oracle_generalised_green_operator, c0=0.0), verdict(_oracle_generalised_green_operator, c0=float('nan')), verdict(_oracle_generalised_green_operator, wave=__import__('numpy').zeros((9, 9))), verdict(_oracle_generalised_green_operator, ret=__import__('numpy').zeros((9, 9), bool)), verdict(_oracle_generalised_green_operator, ret=__import__('numpy').ones((8, 8), bool))))",
        },
    ]
