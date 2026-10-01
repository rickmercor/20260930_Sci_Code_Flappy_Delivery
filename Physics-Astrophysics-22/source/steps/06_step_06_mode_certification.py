"""
Given a background and a starting guess for the frequency, assemble the pencil of step 3 at two resolutions and solve it at both with the iteration of step 5, then return the refined frequency, the distance between the two frequencies, the relative weight carried by the top quarter of the Chebyshev spectrum of the coarse eigenvector, and the residual of the coarse shifted pencil on that eigenvector. The certification passes when the frequency drift and the spectral tail are both below the requested tolerance. The Chebyshev coefficients are those of the Gauss-Lobatto transform of the nodal values, in which the first and last samples carry half weight.

A spectral discretisation of this problem returns numbers whether or not they mean anything, so a candidate mode has to be certified before it is used. Two independent signatures separate a genuine mode from an artefact of the grid, and they fail in different ways, which is why both are worth recording.

The first is resolution independence. A genuine eigenvalue of the continuum problem is approximated by the discrete pencil with an error that falls rapidly as the grid is refined, so the frequency obtained at two resolutions agrees to many digits. An artefact has no continuum counterpart and its position is set by the grid, so it moves. This is the standard discriminator in the pseudospectrum literature for asymptotically flat backgrounds, where the branch cut along the imaginary axis seeds families of spurious eigenvalues that can be told from physical ones in no other way, and it is the reason a single resolution is never enough.

The second is the spectral decay of the eigenfunction. Expanding the nodal values of the converged right eigenvector in the Chebyshev basis of the grid gives coefficients whose magnitude, for a function analytic on the closed interval, falls below rounding level well before the top of the spectrum. An artefact is instead supported at the grid scale, so its expansion has most of its weight in the highest coefficients and the tail is of order one. The ratio of the largest coefficient in the top quarter of the spectrum to the largest overall is therefore a direct read-out: tiny for a resolved mode, of order one for a grid-scale vector. This check also certifies that the resolution is adequate, which the first check alone does not, since two under-resolved grids can agree with one another.

A third quantity, the residual of the shifted pencil on the eigenvector, records that the iteration converged at all. It is necessary but far from sufficient, because the pencil of this problem is close to singular over wide regions of the frequency plane, so a small residual on its own certifies nothing.

Returns
-------
dict holding the float frequency_real and the float frequency_imag of the mode obtained at the fine resolution; the float frequency_drift, the modulus of the difference between the fine and the coarse frequencies; the float spectral_tail, the largest Chebyshev coefficient modulus in the top quarter of the coarse spectrum divided by the largest overall; the float residual, the largest modulus of the coarse shifted pencil applied to the coarse right eigenvector; and the int certified, one when the drift and the tail are both below the tolerance and zero otherwise.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mode_certification(
    resolution: int,
    refinement: int,
    d: int,
    ell: int,
    s: float,
    frequency_guess: complex,
    tolerance: float,
) -> dict:
    """Certify a candidate mode by resolution independence and by spectral decay of its eigenfunction.

    Parameters
    ----------
    resolution : int
        Coarse grid resolution.
    refinement : int
        Increment added to the coarse resolution to form the fine one.
    d : int
        Spacetime dimension.
    ell : int
        Multipole number.
    s : float
        Spin label.
    frequency_guess : complex
        Starting guess for the frequency.
    tolerance : float
        Tolerance applied to the drift and to the spectral tail.

    Returns
    -------
    dict
        Under the keys frequency_real, frequency_imag, frequency_drift, spectral_tail, residual and
        certified.

    Raises
    ------
    ValueError
        When the resolution is not an integer of at least eight, when the refinement is not a
        positive integer, when the tolerance is not finite and strictly positive, or when the
        background or the guess is outside the range the earlier steps accept.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _chebyshev_coefficients(values):
    """Gauss-Lobatto Chebyshev transform of nodal values on a grid of N + 1 points."""
    samples = np.asarray(values)
    count = samples.size - 1
    index = np.arange(count + 1)
    scale = np.where((index == 0) | (index == count), 2.0, 1.0)
    basis = np.cos(np.pi * np.outer(index, index) / count)
    return (2.0 / count) * (basis @ (samples / scale)) / scale


def _oracle_mode_certification(
    resolution: int,
    refinement: int,
    d: int,
    ell: int,
    s: float,
    frequency_guess: complex,
    tolerance: float,
) -> dict:
    """Reference implementation."""
    if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)):
        raise ValueError("resolution must be an integer")
    if int(resolution) < 8:
        raise ValueError("resolution must be at least eight")
    if isinstance(refinement, bool) or not isinstance(refinement, (int, np.integer)):
        raise ValueError("refinement must be an integer")
    if int(refinement) < 1:
        raise ValueError("refinement must be at least one")
    tol = float(tolerance)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance must be finite and strictly positive")

    coarse_size = int(resolution)
    fine_size = coarse_size + int(refinement)
    coarse = _oracle_transmission_pencil(coarse_size, d, ell, s)  # noqa: F821
    fine = _oracle_transmission_pencil(fine_size, d, ell, s)  # noqa: F821
    coarse_mode = _oracle_regular_mode(  # noqa: F821
        coarse["operator_a"], coarse["operator_b"], frequency_guess, 1e-13, 200
    )
    fine_mode = _oracle_regular_mode(  # noqa: F821
        fine["operator_a"], fine["operator_b"], frequency_guess, 1e-13, 200
    )

    coarse_frequency = complex(coarse_mode["frequency_real"], coarse_mode["frequency_imag"])
    fine_frequency = complex(fine_mode["frequency_real"], fine_mode["frequency_imag"])
    drift = float(abs(fine_frequency - coarse_frequency))

    vector = coarse_mode["right_vector"]
    coefficients = np.abs(_chebyshev_coefficients(vector))
    largest = float(np.max(coefficients))
    top = coefficients[(3 * coarse_size) // 4:]
    tail = float(np.max(top) / largest) if largest > 0.0 else float("inf")

    shifted = coarse["operator_a"] - (1j * coarse_frequency) * coarse["operator_b"]
    residual = float(np.max(np.abs(shifted @ vector)))

    return {
        "frequency_real": float(fine_frequency.real),
        "frequency_imag": float(fine_frequency.imag),
        "frequency_drift": drift,
        "spectral_tail": tail,
        "residual": residual,
        "certified": int(drift < tol and tail < tol),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    FLAT = """
def flat(x):
    if isinstance(x, dict):
        return flat([x[k] for k in sorted(x)])
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
def digest(out):
    return (round(out["frequency_real"], 10), round(out["frequency_imag"], 10),
            int(out["frequency_drift"] < 1e-10), int(out["spectral_tail"] < 1e-10),
            int(out["residual"] < 1e-8), int(out["certified"]))
"""
    return [
        {
            # the graded sector, certified at a moderate pair of resolutions
            "setup": SETUP + FLAT,
            "call": "flat(digest(mode_certification(60, 30, 14, 2, 2.0, 1.1j, 1e-10)))",
            "gold_call": "flat(digest(_oracle_mode_certification(60, 30, 14, 2, 2.0, 1.1j, 1e-10)))",
        },
        {
            # the two signatures are independent: the mode of the twenty-dimensional background is also
            # resolved and resolution-independent, while its frequency differs from the fourteen-
            # dimensional one, and the exact four-dimensional mode has a polynomial eigenfunction whose
            # Chebyshev tail is at rounding level
            "setup": SETUP + """
def signatures(fn):
    big = fn(60, 30, 20, 2, 2.0, 1.05j, 1e-10)
    small = fn(60, 30, 14, 2, 2.0, 1.1j, 1e-10)
    exact = fn(8, 2, 4, 2, 2.0, 3.9j, 1e-4)
    return (int(big["certified"]), int(small["certified"]),
            int(big["frequency_imag"] < small["frequency_imag"]),
            int(exact["spectral_tail"] < 1e-8),
            int(abs(exact["frequency_imag"] - 4.0) < 1e-4))
""" + FLAT,
            "call": "flat(signatures(mode_certification))",
            "gold_call": "flat(signatures(_oracle_mode_certification))",
        },
        {
            # edge: a guess placed away from any mode on the axis drives the iteration onto a
            # resolution-dependent artefact, which the certification must reject
            "setup": SETUP + """
def artefact(fn):
    out = fn(40, 20, 14, 2, 0.0, 3.0j, 1e-10)
    return (int(out["certified"] == 0), int(out["frequency_drift"] > 1e-10))
""" + FLAT,
            "call": "flat(artefact(mode_certification))",
            "gold_call": "flat(artefact(_oracle_mode_certification))",
        },
        {
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(resolution=40, refinement=20, d=14, ell=2, s=2.0,
                frequency_guess=1.1j, tolerance=1e-10)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(mode_certification, resolution=6), "
                    "verdict(mode_certification, refinement=0), "
                    "verdict(mode_certification, tolerance=0.0), "
                    "verdict(mode_certification, d=3), "
                    "verdict(mode_certification, frequency_guess=complex('nan'))))",
            "gold_call": "flat((verdict(_oracle_mode_certification, resolution=6), "
                         "verdict(_oracle_mode_certification, refinement=0), "
                         "verdict(_oracle_mode_certification, tolerance=0.0), "
                         "verdict(_oracle_mode_certification, d=3), "
                         "verdict(_oracle_mode_certification, frequency_guess=complex('nan'))))",
        },
    ]
