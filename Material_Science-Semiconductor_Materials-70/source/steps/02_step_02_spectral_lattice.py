"""
The auxiliary problem that the iterative solver rests on is written in Fourier space, so before anything can be solved the lattice of frequencies has to be laid out, and the convention chosen here is not the one a routine FFT setup produces.

The cell is the square of side L, and it is sampled at N + 1 points along each edge with spacing

$$dx = L / (N + 1),$$

the point of index a sitting at a * dx for a = 0, ..., N. There are therefore N + 1 samples per direction, one period wide, and the discrete transform has odd length N + 1. That is deliberate. With N even the frequency indices

$$i = -N/2, ..., N/2$$

are N + 1 consecutive integers, which is exactly one complete residue system modulo N + 1, so every index names a distinct exponential and the set is symmetric about zero. An even-length transform cannot do both at once: it has a single unpaired mode at the folding frequency whose exponential is real on every sample point, and that mode has to be treated as a special case in any scheme built on it. Here there is no such mode and no special case. The wavevector attached to index i is

$$xi_i = 2 * pi * i / L,$$

and in two dimensions the pair (i, j) carries xi = (xi_i, xi_j).

Separately from the grid, the unknown potential fluctuation is represented by a partial Fourier series of order M + 1 per direction, retaining the modes with |i| <= M/2 and |j| <= M/2, with M an even integer no larger than N. This is the whole point of the construction: M and N are two independent numbers. N says how many points the trapezoidal rule uses to estimate Fourier coefficients, and M says how many of those coefficients the represented field is allowed to keep. The condition N >= M is required, because estimating a coefficient needs at least as many sample points as there are coefficients to estimate; the reverse inequality is not required by anything, and the habitual choice M = N is a convention rather than a consequence. The mode at i = j = 0 is excluded from the retained set, because the mean of the fluctuation is fixed at zero and the mean of the field is fixed by the applied loading instead.

The Green operator is assembled from a symbol vector beta which, for the standard spectral derivative rule, is the wavevector itself, beta = xi. In general it is built as

$$Gamma(xi) = conj(beta) tensor beta / [c0 * conj(beta) . beta],$$

with c0 the conductivity of the homogeneous reference medium. Written this way one property is visible immediately and is worth using as the check on the assembly: c0 * Gamma is an orthogonal projector. It is idempotent, because conj(beta) tensor beta applied to itself returns itself times the scalar conj(beta) . beta, and its trace is exactly one, so it projects each mode of the polarisation onto the single direction that a gradient field is allowed to occupy at that frequency. Any assembly that fails either property has the wrong operator, and the failure will not otherwise show until the iteration has converged to the wrong field.

Returns
-------
dict, the frequency lattice and its retained-mode set, the Green operator built on it, and the two defects that certify the operator is a projector.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectral_lattice(
    n_grid: int,
    n_modes: int,
    period: float,
    reference: float,
) -> dict:
    """Lay out the frequency lattice, the retained-mode set and the Green operator.

    Parameters
    ----------
    n_grid : int
        The number N of grid intervals, even and above zero.
    n_modes : int
        The truncation order M, even, above zero and not above N.
    period : float
        The cell edge in metre, above zero.
    reference : float
        The reference conductivity in siemens per metre, above zero.

    Returns
    -------
    dict
        Under the keys frequency_index, wavevector, symbol, green, retained, n_points, n_retained, spacing, projector_defect and trace_defect. Write n for n_grid + 1. frequency_index is the one-dimensional integer array of the n signed frequency indices, in the order the discrete transform uses. wavevector is a real array of shape (2, n, n) and symbol a complex array of the same shape, the leading axis running over the two directions. green is a complex array of shape (2, 2, n, n), its two leading axes running over the operator's row and column direction. retained is a boolean array of shape (n, n). The remaining five entries are scalars: n_points is n itself, the number of sample points along one edge and not over the two-dimensional grid; n_retained is the number of true entries of retained, counted over the whole (n, n) mask; spacing is the sample spacing, the period divided by n; and projector_defect and trace_defect are the departures of the scaled operator from idempotence and from unit trace, each taken over the retained modes.

    Raises
    ------
    ValueError
        When N or M fails to be a positive even integer, when M exceeds N, or when the period or the reference conductivity fails to be finite and above zero.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _even_positive(value, label):
    """Return an argument as an int once it is known to be a positive even integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be an integer" % label)
    out = int(value)
    if out <= 0 or out % 2 != 0:
        raise ValueError("%s must be a positive even integer" % label)
    return out


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


def _oracle_spectral_lattice(
    n_grid: int,
    n_modes: int,
    period: float,
    reference: float,
) -> dict:
    """Reference implementation."""
    big_n = _even_positive(n_grid, "n_grid")
    small_m = _even_positive(n_modes, "n_modes")
    if small_m > big_n:
        raise ValueError("n_modes must not exceed n_grid")
    length = _positive_float(period, "period")
    c0 = _positive_float(reference, "reference")

    n = big_n + 1
    index = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    wave = 2.0 * np.pi * index / length
    wavevector = np.zeros((2, n, n))
    wavevector[0] = wave[:, None]
    wavevector[1] = wave[None, :]
    symbol = wavevector.astype(complex)

    keep = np.abs(index) <= small_m // 2
    retained = keep[:, None] & keep[None, :]
    retained[0, 0] = False

    green = _green_from_symbol(symbol, c0)
    idem, trace = _projector_defects(green, c0, retained)
    return {
        "frequency_index": index,
        "wavevector": wavevector,
        "symbol": symbol,
        "green": green,
        "retained": retained,
        "n_points": n,
        "n_retained": int(retained.sum()),
        "spacing": length / n,
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


def test_cases():
    return [
        {
            # counts, spacing, index range and the projector identities
            "setup": """
import numpy as np
def digest(out):
    idx = np.asarray(out["frequency_index"])
    g = np.asarray(out["green"])
    r = np.asarray(out["retained"])
    k = np.asarray(out["wavevector"])
    b = np.asarray(out["symbol"])
    # whole-array checksums, so that no entry of any returned array is left unconstrained
    return (round(float(np.abs(k).sum()), 6), round(float(np.abs(b).sum()), 6),
            round(float(np.abs(np.imag(b)).sum()), 10),
            out["n_points"], out["n_retained"], round(out["spacing"], 16),
            int(idx.min()), int(idx.max()), int(np.unique(idx).size),
            int(out["projector_defect"] < 1e-12), int(out["trace_defect"] < 1e-12),
            round(float(np.sum(np.abs(g[:, :, r]))), 10),
            round(float(np.real(g[0, 0, 1, 1])), 13))
""" + FLAT,
            "call": "flat((digest(spectral_lattice(8, 4, 2.0e-6, 1000.0)), digest(spectral_lattice(12, 12, 1.0, 1.0)), digest(spectral_lattice(20, 6, 3.5e-6, 250.0))))",
            "gold_call": "flat((digest(_oracle_spectral_lattice(8, 4, 2.0e-6, 1000.0)), digest(_oracle_spectral_lattice(12, 12, 1.0, 1.0)), digest(_oracle_spectral_lattice(20, 6, 3.5e-6, 250.0))))",
        },
        {
            # the operator itself, sampled at a few modes, and its scaling in c0 and L
            "setup": """
import numpy as np
def digest(out):
    g = np.asarray(out["green"])
    k = np.asarray(out["wavevector"])
    picks = [(0, 1), (1, 0), (1, 1), (2, 3), (3, 2)]
    vals = []
    for (a, b) in picks:
        for c in range(2):
            for d in range(2):
                vals.append(round(float(np.real(g[c, d, a, b])), 14))
        vals.append(round(float(k[0, a, b]), 8))
        vals.append(round(float(k[1, a, b]), 8))
    return tuple(vals)
""" + FLAT,
            "call": "flat((digest(spectral_lattice(8, 8, 1.0, 1.0)), digest(spectral_lattice(8, 8, 1.0, 4.0)), digest(spectral_lattice(8, 8, 0.5, 1.0))))",
            "gold_call": "flat((digest(_oracle_spectral_lattice(8, 8, 1.0, 1.0)), digest(_oracle_spectral_lattice(8, 8, 1.0, 4.0)), digest(_oracle_spectral_lattice(8, 8, 0.5, 1.0))))",
        },
        {
            # the retained set: shape, symmetry about zero, the excluded mean mode, and
            # the count (M + 1) squared minus one whenever M is below N
            "setup": """
import numpy as np
def digest(out):
    r = np.asarray(out["retained"])
    idx = np.asarray(out["frequency_index"])
    n = out["n_points"]
    sym = int(bool(np.array_equal(r, r[np.argsort((-idx) % n)][:, np.argsort((-idx) % n)])))
    g = np.asarray(out["green"])
    return (int(r.sum()), int(r[0, 0]), sym, int(r.shape[0]), int(r.shape[1]),
            round(float(np.sum(np.abs(g[:, :, r]))), 10))
""" + FLAT,
            "call": "flat((digest(spectral_lattice(16, 4, 1.0, 1.0)), digest(spectral_lattice(16, 10, 1.0, 1.0)), digest(spectral_lattice(16, 16, 1.0, 1.0)), digest(spectral_lattice(6, 2, 1.0, 1.0))))",
            "gold_call": "flat((digest(_oracle_spectral_lattice(16, 4, 1.0, 1.0)), digest(_oracle_spectral_lattice(16, 10, 1.0, 1.0)), digest(_oracle_spectral_lattice(16, 16, 1.0, 1.0)), digest(_oracle_spectral_lattice(6, 2, 1.0, 1.0))))",
        },
        {
            "setup": """
def verdict(fn, n=8, m=4, L=1.0, c0=1.0):
    try:
        fn(n, m, L, c0)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(spectral_lattice, n=7), verdict(spectral_lattice, m=3), verdict(spectral_lattice, m=10), verdict(spectral_lattice, n=0), verdict(spectral_lattice, L=0.0), verdict(spectral_lattice, c0=-1.0), verdict(spectral_lattice, L=float('inf')), verdict(spectral_lattice)))",
            "gold_call": "flat((verdict(_oracle_spectral_lattice, n=7), verdict(_oracle_spectral_lattice, m=3), verdict(_oracle_spectral_lattice, m=10), verdict(_oracle_spectral_lattice, n=0), verdict(_oracle_spectral_lattice, L=0.0), verdict(_oracle_spectral_lattice, c0=-1.0), verdict(_oracle_spectral_lattice, L=float('inf')), verdict(_oracle_spectral_lattice)))",
        },
    ]
