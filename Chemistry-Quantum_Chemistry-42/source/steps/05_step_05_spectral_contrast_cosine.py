"""
Step 05: Unshifted spectral contrast cosine of broadened lineshapes.

Cosine of the spectral contrast angle between two Gaussian-broadened vibronic lineshapes computed from autocorrelation functions.

Approximate vibronic spectra are judged against an exact reference by treating each spectrum as a vector and measuring
the angle between the two vectors: the cosine of the spectral contrast angle is the inner product of the spectra divided
by the product of their norms, with the inner product defined as the frequency integral of the product of the two
spectra. It equals one only when the two spectra are proportional, it ignores an overall intensity scale, and when no
frequency shift is applied it penalizes both misplaced peaks and a wrong envelope.

Here the spectra are the Gaussian-broadened lineshapes
    I(omega) = Re int_0^infinity C(t) exp(i omega t) exp(-t^2 / (2 tau^2)) dt
of the sampled autocorrelation functions, with broadening time tau, integrated over all real omega. The frequency
integrals can be evaluated without any frequency grid.

Returns
-------
numpy.ndarray of shape (3,), unshifted spectral contrast cosine and the squared norms of the approximate and reference broadened lineshapes
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectral_contrast_cosine(approx_autocorrelation: "np.ndarray", reference_autocorrelation: "np.ndarray", time_step: float, broadening_time: float) -> "np.ndarray":
    '''Unshifted spectral contrast cosine between the broadened lineshapes of an approximate and a reference autocorrelation.

    Both autocorrelations are sampled at t_n = n time_step, n = 0, ..., N - 1. The frequency integrals of the products and
    squares of the two lineshapes are converted exactly into time integrals over [0, infinity) of the corresponding products
    of autocorrelations and damping factors, and those time integrals are evaluated over [0, t_(N-1)] with the composite
    trapezoidal rule on the samples.

    Parameters
    ----------
    approx_autocorrelation : np.ndarray
        Array of shape (N, 2), N >= 2, with rows [Re C, Im C] of the approximate autocorrelation.
    reference_autocorrelation : np.ndarray
        Array of shape (N, 2) with rows [Re C, Im C] of the reference autocorrelation on the same times.
    time_step : float
        Sampling interval dt > 0.
    broadening_time : float
        Broadening time tau > 0 of the Gaussian damping exp(-t^2 / (2 tau^2)).

    Returns
    -------
    result : np.ndarray
        Array of shape (3,) holding [cos(theta), squared norm of the approximate lineshape, squared norm of the reference
        lineshape], where each squared norm is the frequency integral of the square of that lineshape.

    Raises
    ------
    ValueError
        If the arrays are not finite with equal shapes (N, 2) and N >= 2, if time_step or broadening_time is not strictly
        positive, or if either lineshape has zero norm.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_spectral_contrast_cosine(approx_autocorrelation: "np.ndarray", reference_autocorrelation: "np.ndarray", time_step: float, broadening_time: float) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    ca = np.asarray(approx_autocorrelation, dtype=float)
    cr = np.asarray(reference_autocorrelation, dtype=float)
    if ca.ndim != 2 or ca.shape != cr.shape or ca.shape[1] != 2 or ca.shape[0] < 2:
        raise ValueError("autocorrelations must share a shape (N, 2) with N >= 2")
    if not (np.all(np.isfinite(ca)) and np.all(np.isfinite(cr))):
        raise ValueError("autocorrelations must be finite")
    if not time_step > 0.0 or not broadening_time > 0.0:
        raise ValueError("time_step and broadening_time must be strictly positive")
    za = ca[:, 0] + 1j * ca[:, 1]
    zr = cr[:, 0] + 1j * cr[:, 1]
    t = time_step * np.arange(za.size)
    damp2 = np.exp(-(t / broadening_time) ** 2)
    # int I_1 I_2 domega = pi Re int_0^inf C_1 conj(C_2) exp(-t^2 / tau^2) dt
    cross = np.pi * np.trapezoid(np.real(za * np.conj(zr)) * damp2, t)
    norm_a = np.pi * np.trapezoid(np.abs(za) ** 2 * damp2, t)
    norm_r = np.pi * np.trapezoid(np.abs(zr) ** 2 * damp2, t)
    if norm_a <= 0.0 or norm_r <= 0.0:
        raise ValueError("lineshapes must have nonzero norm")
    return np.array([cross / np.sqrt(norm_a * norm_r), norm_a, norm_r])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: two damped vibronic progressions with slightly different spacings and anharmonic phases ---
        {
            "setup": "import numpy as np\n"
                     "t = 0.05 * np.arange(601)\n"
                     "za = np.exp(-1.2 * (1.0 - np.exp(-0.85j * t)) - 2.1j * t)\n"
                     "zr = np.exp(-1.25 * (1.0 - np.exp(-0.83j * t)) - 2.08j * t - 0.004 * t * t * 1j)\n"
                     "A = np.column_stack([za.real, za.imag])\n"
                     "B = np.column_stack([zr.real, zr.imag])\n",
            "call": "spectral_contrast_cosine(A.copy(), B.copy(), 0.05, 5.0)",
            "gold_call": "_oracle_spectral_contrast_cosine(A.copy(), B.copy(), 0.05, 5.0)",
            "tol": 1e-10,
        },
        # --- Boundary: an approximation that differs from the reference only by an intensity factor gives cosine one ---
        {
            "setup": "import numpy as np\n"
                     "t = 0.1 * np.arange(121)\n"
                     "zr = np.exp(-0.7 * (1.0 - np.exp(-1.3j * t)) - 0.4j * t)\n"
                     "B = np.column_stack([zr.real, zr.imag])\n"
                     "A = 2.5 * B\n",
            "call": "spectral_contrast_cosine(A.copy(), B.copy(), 0.1, 2.0)",
            "gold_call": "_oracle_spectral_contrast_cosine(A.copy(), B.copy(), 0.1, 2.0)",
            "tol": 1e-12,
        },
        # --- Edge: a rigid frequency shift of the whole spectrum with short broadening ---
        {
            "setup": "import numpy as np\n"
                     "t = 0.02 * np.arange(2001)\n"
                     "zr = np.exp(-2.0 * (1.0 - np.exp(-0.6j * t)) - 1.5j * t)\n"
                     "za = zr * np.exp(-0.35j * t)\n"
                     "A = np.column_stack([za.real, za.imag])\n"
                     "B = np.column_stack([zr.real, zr.imag])\n",
            "call": "spectral_contrast_cosine(A.copy(), B.copy(), 0.02, 1.5)",
            "gold_call": "_oracle_spectral_contrast_cosine(A.copy(), B.copy(), 0.02, 1.5)",
            "tol": 1e-10,
        },
        # --- Boundary: two samples only, where the trapezoidal rule uses a single interval ---
        {
            "setup": "import numpy as np\n"
                     "A = np.array([[1.0, 0.0], [0.2, -0.6]])\n"
                     "B = np.array([[1.0, 0.0], [-0.1, -0.7]])\n",
            "call": "spectral_contrast_cosine(A.copy(), B.copy(), 0.5, 0.8)",
            "gold_call": "_oracle_spectral_contrast_cosine(A.copy(), B.copy(), 0.5, 0.8)",
            "tol": 1e-12,
        },
        # --- Error: mismatched sample counts must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.ones((5, 2)), np.ones((6, 2)), 0.1, 1.0)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(spectral_contrast_cosine)",
            "gold_call": "_probe(_oracle_spectral_contrast_cosine)",
        },
    ]
