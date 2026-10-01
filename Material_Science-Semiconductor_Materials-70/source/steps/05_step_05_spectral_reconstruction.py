"""
The field the solver produces is not a list of values at grid points. It is a partial Fourier series of order M + 1 per direction, and a partial Fourier series is a function defined at every point of the cell. The grid entered only as the quadrature rule that estimated the coefficients; once the coefficients are known the grid has done its work, and evaluating the series anywhere else is not interpolation but simply the same function read at other points.

That distinction has consequences that are easy to miss. A truncated series approximating a discontinuous field oscillates near the discontinuity, and the oscillation has a wavelength set by the highest retained mode, which is comparable with the grid spacing when M is comparable with N. Sampled at the grid points, oscillations of that wavelength are largely invisible: the samples can sit near the nodes of the ripple and return a field that looks clean. Evaluated between the grid points the same function shows the ripple at full amplitude. A field judged only at the points it was computed on can therefore appear far better than it is, and any measure of its quality has to be taken from the function rather than from the samples.

Reconstruction is done by evaluating

$$E(x) = sum over |i| <= M/2, |j| <= M/2 of F_ij * exp(i * xi_i * x1) * exp(i * xi_j * x2)$$

on a finer uniform grid. The Fourier series coefficients F are obtained from the discrete transform of the field by dividing by the number of grid points, since the transform approximates the coefficients through the trapezoidal rule. Carried out directly this costs one operation per fine point per retained mode. Done instead by placing the retained coefficients into a longer array whose remaining entries are zero and inverting a transform of that length, it costs the usual transform complexity in the length of the fine grid, and it returns exactly the same values: zero padding in the frequency domain is evaluation of the same trigonometric polynomial at more points, not smoothing and not resampling.

Taking the fine grid to be a whole multiple K of the computation grid, so that it carries K * (N + 1) points per direction, makes the computation grid a subset of it. That is worth doing for one reason beyond convenience: every K-th value of the reconstruction must then agree with the field on the computation grid to rounding error, which is the check that the padding was placed at the right positions. The frequency index of a mode is not its array position, and putting the retained coefficients at their array positions rather than at their indices modulo the fine length produces a plausible-looking field that is wrong. The node agreement is what catches that.

Reported alongside is the ratio of the largest magnitude the first field component reaches on the reconstruction grid to the largest it reaches on the computation grid. It is one when the coarse grid sees everything the refined one does and above one by the size of whatever the coarse grid was missing. Note what this ratio is not: the numerator is a maximum over a finite set of points, not the supremum of the underlying polynomial, and the two differ. Refining further moves the numerator, and not monotonically, because a finer grid samples a different set of points on the same ripple. The statistic is therefore tied to a stated refinement factor and is a diagnostic of what the computation grid hides rather than a property of the field alone.

Returns
-------
dict, the field reconstructed between the sample points, with the peak magnitude the computation grid hides and the agreement at the points the two grids share.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectral_reconstruction(
    field_transform,
    n_grid: int,
    n_modes: int,
    refinement: int,
) -> dict:
    """Evaluate the retained trigonometric polynomial on a refined grid and measure what the coarse grid hides.

    The reported peak ratio is taken over the reconstruction grid this call builds, so it
    depends on the refinement factor and is not the supremum of the polynomial.

    Parameters
    ----------
    field_transform : array
        Complex array of shape (2, N + 1, N + 1) holding the transform of the converged field.
    n_grid : int
        The number N of grid intervals, even and above zero.
    n_modes : int
        The truncation order M, even, above zero and not above N.
    refinement : int
        Whole refinement factor K, above zero.

    Returns
    -------
    dict
        Under the keys fine_field, coarse_field, n_fine, node_defect, peak_ratio, fine_peak and coarse_peak. fine_field is a real array of shape (2, n_fine, n_fine) and coarse_field a real array of shape (2, n_grid + 1, n_grid + 1), the leading axis running over the two directions as in field_transform. The remaining five entries are scalars.

    Raises
    ------
    ValueError
        When N or M fails to be a positive even integer, when M exceeds N, when the refinement factor fails to be an integer above zero, or when the transform does not have shape (2, N + 1, N + 1).
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _even_positive(value, label):
    """Return an argument as an int once it is known to be a positive even integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be an integer" % label)
    out = int(value)
    if out <= 0 or out % 2 != 0:
        raise ValueError("%s must be a positive even integer" % label)
    return out


def _oracle_spectral_reconstruction(
    field_transform,
    n_grid: int,
    n_modes: int,
    refinement: int,
) -> dict:
    """Reference implementation."""
    big_n = _even_positive(n_grid, "n_grid")
    small_m = _even_positive(n_modes, "n_modes")
    if small_m > big_n:
        raise ValueError("n_modes must not exceed n_grid")
    if isinstance(refinement, bool) or not isinstance(refinement, (int, np.integer)):
        raise ValueError("refinement must be an integer")
    factor = int(refinement)
    if factor <= 0:
        raise ValueError("refinement must be above zero")

    transform = np.asarray(field_transform)
    n = big_n + 1
    if transform.shape != (2, n, n):
        raise ValueError("field_transform must have shape (2, n_grid + 1, n_grid + 1)")

    n_fine = factor * n
    index = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    # the Fourier series coefficients, the transform carrying the trapezoidal factor
    series = transform / float(n) ** 2
    chosen = np.where(np.abs(index) <= small_m // 2)[0]
    # position by frequency index modulo the fine length, not by array position
    rows = index[chosen] % n_fine
    padded = np.zeros((2, n_fine, n_fine), complex)
    padded[np.ix_([0, 1], rows, rows)] = series[np.ix_([0, 1], chosen, chosen)]
    fine = np.real(np.fft.ifft2(padded * float(n_fine) ** 2, axes=(1, 2)))
    coarse = np.real(np.fft.ifft2(transform, axes=(1, 2)))

    scale = float(np.max(np.abs(coarse)))
    if scale == 0.0:
        scale = 1.0
    shared = fine[:, ::factor, ::factor]
    node_defect = float(np.max(np.abs(shared - coarse))) / scale
    fine_peak = float(np.max(np.abs(fine[0])))
    coarse_peak = float(np.max(np.abs(coarse[0])))
    ratio = fine_peak / coarse_peak if coarse_peak > 0.0 else float("inf")
    return {
        "fine_field": fine,
        "coarse_field": coarse,
        "n_fine": n_fine,
        "node_defect": node_defect,
        "peak_ratio": ratio,
        "fine_peak": fine_peak,
        "coarse_peak": coarse_peak,
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
def planted(N, M, seed):
    # a band-limited real field whose transform is supported exactly on the retained set
    n = N + 1
    idx = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    keep = np.abs(idx) <= M // 2
    ret = keep[:, None] & keep[None, :]
    rng = np.random.default_rng(seed)
    real = np.zeros((2, n, n))
    real[0] = rng.standard_normal((n, n))
    real[1] = rng.standard_normal((n, n))
    t = np.fft.fft2(real, axes=(1, 2))
    t[0][~ret] = 0.0
    t[1][~ret] = 0.0
    # re-symmetrise so the inverse transform is real to rounding error
    f = np.real(np.fft.ifft2(t, axes=(1, 2)))
    t = np.fft.fft2(f, axes=(1, 2))
    t[0][~ret] = 0.0
    t[1][~ret] = 0.0
    return t
"""


def test_cases():
    return [
        {
            # the reconstruction must reproduce the computation grid exactly at the
            # points the two grids share, at several truncations and refinements
            "setup": SETUP + """
def digest(fn, N, M, K, seed):
    out = fn(planted(N, M, seed), N, M, K)
    fine = np.asarray(out["fine_field"])
    coarse = np.asarray(out["coarse_field"])
    return (round(float(np.abs(fine).sum()), 8), round(float(np.abs(coarse).sum()), 8),
            round(float(out["fine_peak"]), 10), round(float(out["coarse_peak"]), 10),
            out["n_fine"], int(out["node_defect"] < 1.0e-12),
            round(float(out["peak_ratio"]), 10) if out["peak_ratio"] < 1e6 else -1.0,
            round(float(out["coarse_peak"]), 10))
""" + FLAT,
            "call": "flat((digest(spectral_reconstruction, 12, 12, 4, 1), digest(spectral_reconstruction, 12, 8, 4, 2), digest(spectral_reconstruction, 12, 4, 3, 3), digest(spectral_reconstruction, 20, 10, 2, 4), digest(spectral_reconstruction, 8, 8, 1, 5)))",
            "gold_call": "flat((digest(_oracle_spectral_reconstruction, 12, 12, 4, 1), digest(_oracle_spectral_reconstruction, 12, 8, 4, 2), digest(_oracle_spectral_reconstruction, 12, 4, 3, 3), digest(_oracle_spectral_reconstruction, 20, 10, 2, 4), digest(_oracle_spectral_reconstruction, 8, 8, 1, 5)))",
        },
        {
            # a single mode placed by hand: the reconstruction must be the cosine that
            # mode names, sampled on the fine grid, which fixes the padding positions
            "setup": """
import numpy as np
def single(N, i, j, amp):
    n = N + 1
    t = np.zeros((2, n, n), complex)
    t[0, i % n, j % n] = amp * n * n / 2.0
    t[0, (-i) % n, (-j) % n] = amp * n * n / 2.0
    return t
def digest(fn, N, i, j, K, amp=3.0):
    M = N
    out = fn(single(N, i, j, amp), N, M, K)
    g = np.asarray(out["fine_field"])[0]
    nf = out["n_fine"]
    x = np.arange(nf) / nf
    want = amp * np.cos(2.0 * np.pi * (i * x[:, None] + j * x[None, :]))
    return (round(float(np.max(np.abs(g - want))), 10), int(out["node_defect"] < 1.0e-12),
            round(float(out["peak_ratio"]), 8))
""" + FLAT,
            "call": "flat((digest(spectral_reconstruction, 8, 1, 0, 4), digest(spectral_reconstruction, 8, 3, 2, 4), digest(spectral_reconstruction, 8, 4, 4, 3), digest(spectral_reconstruction, 12, 6, 0, 2)))",
            "gold_call": "flat((digest(_oracle_spectral_reconstruction, 8, 1, 0, 4), digest(_oracle_spectral_reconstruction, 8, 3, 2, 4), digest(_oracle_spectral_reconstruction, 8, 4, 4, 3), digest(_oracle_spectral_reconstruction, 12, 6, 0, 2)))",
        },
        {
            # refining further must not change values already computed: the fine grid at
            # factor 2 must be contained in the fine grid at factor 6
            "setup": SETUP + """
def digest(fn, N, M, seed):
    t = planted(N, M, seed)
    a = np.asarray(fn(t, N, M, 2)["fine_field"])
    b = np.asarray(fn(t, N, M, 6)["fine_field"])
    return (round(float(np.max(np.abs(b[:, ::3, ::3] - a))), 11),
            round(float(fn(t, N, M, 2)["peak_ratio"]), 8),
            round(float(fn(t, N, M, 6)["peak_ratio"]), 8))
""" + FLAT,
            "call": "flat((digest(spectral_reconstruction, 10, 10, 7), digest(spectral_reconstruction, 10, 6, 8)))",
            "gold_call": "flat((digest(_oracle_spectral_reconstruction, 10, 10, 7), digest(_oracle_spectral_reconstruction, 10, 6, 8)))",
        },
        {
            "setup": SETUP + """
def verdict(fn, N=8, M=4, K=4, shape=None):
    t = planted(8, 4, 1)
    if shape is not None:
        t = np.zeros(shape, complex)
    try:
        fn(t, N, M, K)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(spectral_reconstruction), verdict(spectral_reconstruction, N=7), verdict(spectral_reconstruction, M=3), verdict(spectral_reconstruction, M=10), verdict(spectral_reconstruction, K=0), verdict(spectral_reconstruction, K=-2), verdict(spectral_reconstruction, shape=(2, 8, 8)), verdict(spectral_reconstruction, shape=(9, 9))))",
            "gold_call": "flat((verdict(_oracle_spectral_reconstruction), verdict(_oracle_spectral_reconstruction, N=7), verdict(_oracle_spectral_reconstruction, M=3), verdict(_oracle_spectral_reconstruction, M=10), verdict(_oracle_spectral_reconstruction, K=0), verdict(_oracle_spectral_reconstruction, K=-2), verdict(_oracle_spectral_reconstruction, shape=(2, 8, 8)), verdict(_oracle_spectral_reconstruction, shape=(9, 9))))",
        },
    ]
