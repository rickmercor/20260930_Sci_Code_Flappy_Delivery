"""
Run the complete inverse-EFT reconstruction, including recovery of one omitted coefficient.

The input may be complete or may contain exactly one NaN marking an intentionally
withheld EFT coefficient. When one coefficient is missing, recover it from the
finite-spectrum consistency conditions before carrying out the standard source
reconstruction. The orchestrator derives the source configuration, infers the
finite spectrum, reconstructs and classifies the spectral locations, checks the
EFT re-expansion residual, and evaluates the rational amplitude.

Parameters
----------
b : np.ndarray
    One-dimensional real EFT coefficient array with nonzero known b[0]. It may
    contain no NaNs or exactly one NaN marking an omitted coefficient.
rank_rtol : float
    Positive finite relative SVD threshold.
imag_tol : float
    Nonnegative finite generalized-eigenvalue imaginary-part tolerance.
residual_tol : float
    Positive finite upper bound required for the EFT re-expansion residual.
s_eval : float
    Finite real evaluation point that is not a reconstructed pole.

Returns
-------
value : float
    Reconstructed amplitude at s_eval, rounded to ten decimal places.

Raises
------
ValueError
    Under invalid-input conditions declared by the preceding steps, if more
    than one EFT coefficient is missing, if the reconstruction residual is not
    smaller than residual_tol, or if s_eval is a reconstructed pole.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_inverse_eft_reconstruction(
    b: "np.ndarray",
    rank_rtol: float,
    imag_tol: float,
    residual_tol: float,
    s_eval: float,
) -> float:
    """Recover any single omitted coefficient, reconstruct A, and return A_rec(s_eval)."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_evaluate_inverse_eft_reconstruction(
    b: "np.ndarray",
    rank_rtol: float,
    imag_tol: float,
    residual_tol: float,
    s_eval: float,
) -> float:
    """Recover any single omitted coefficient, reconstruct A, and return A_rec(s_eval)."""
    raw = np.asarray(b)
    if np.iscomplexobj(raw) and np.any(np.imag(raw[np.isfinite(raw)]) != 0.0):
        raise ValueError("b must be real-valued")
    try:
        coeff = np.asarray(raw, dtype=float).copy()
        residual_limit = float(residual_tol)
        point = float(s_eval)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("inputs must be real numeric values") from exc
    if coeff.ndim != 1 or coeff.size < 2 or np.any(np.isinf(coeff)):
        raise ValueError("b must be a one-dimensional array with no infinite entries")
    if np.isnan(coeff[0]) or coeff[0] == 0.0:
        raise ValueError("b[0] must be known and nonzero")
    missing = np.flatnonzero(np.isnan(coeff))
    if missing.size > 1:
        raise ValueError("at most one EFT coefficient may be omitted")
    if not np.isfinite(residual_limit) or residual_limit <= 0.0:
        raise ValueError("residual_tol must be strictly positive and finite")
    if not np.isfinite(point):
        raise ValueError("s_eval must be finite")

    if missing.size == 1:
        recovered = _oracle_recover_missing_eft_coefficient(coeff, rank_rtol, imag_tol)
        coeff[int(missing[0])] = recovered

    c = _oracle_compute_log_derivative_coefficients(coeff)
    r, probe_size = _oracle_choose_source_configuration(c)
    probe = _oracle_build_hankel_probe(c, r, probe_size)
    d = _oracle_infer_finite_spectrum_size(probe, rank_rtol)
    if d < 1:
        raise ValueError("the inferred finite spectrum is empty")
    current, shifted = _oracle_build_shifted_hankel_pencil(c, r, d)
    spectral_data = _oracle_recover_spectral_locations(current, shifted, imag_tol)
    classified, _weights = _oracle_classify_spectral_locations(current, r, spectral_data)
    numerator, denominator, residual = _oracle_reconstruct_rational_amplitude(coeff, classified)
    if residual >= residual_limit:
        raise ValueError("the EFT reconstruction residual exceeds residual_tol")

    denominator_value = float(np.polynomial.polynomial.polyval(point, denominator))
    if abs(denominator_value) <= 1.0e-12:
        raise ValueError("s_eval is a reconstructed pole")
    numerator_value = float(np.polynomial.polynomial.polyval(point, numerator))
    return float(np.round(numerator_value / denominator_value, 10))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return missing-coefficient, complete-input, and alternate complete-input cases."""
    return [
        {
            "setup": "import numpy as np\nb=np.array([1.255458,0.076295438118,-0.035757230423273274,-0.075823351552458039254394,-0.083749978089300131915797567674,-0.078338731191377881605205437737333754,np.nan,-0.057419982857692864178695909526227899110314926714,-0.047357424351643017636639948800385481733980392431793594,-0.038625674826665615587987604470468827548796529186628134990074,-0.031299339343808186683923366066895834773625422979666139953711748154],dtype=float)",
            "call": "evaluate_inverse_eft_reconstruction(b,1e-10,1e-10,1e-8,0.37)",
            "gold_call": "_oracle_evaluate_inverse_eft_reconstruction(b,1e-10,1e-10,1e-8,0.37)",
        },
        {
            "setup": "import numpy as np\nb=np.array([2.0,1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125],dtype=float)",
            "call": "evaluate_inverse_eft_reconstruction(b,1e-10,1e-10,1e-8,0.4)",
            "gold_call": "_oracle_evaluate_inverse_eft_reconstruction(b,1e-10,1e-10,1e-8,0.4)",
        },
        {
            "setup": "import numpy as np\nb=np.array([1.2,0.6,0.45,0.3375,0.253125,0.18984375,0.1423828125,0.106787109375,0.08009033203125],dtype=float)",
            "call": "evaluate_inverse_eft_reconstruction(b,1e-10,1e-10,1e-8,0.3)",
            "gold_call": "_oracle_evaluate_inverse_eft_reconstruction(b,1e-10,1e-10,1e-8,0.3)",
        },
    ]
