"""
The treatment Boltzmann fit of the peak flux-voltage curve.

The treatment validation summary (its Eq. 15) fits the ensemble-averaged PEAK flux at each pulse energy with a three-parameter Boltzmann form, F = F_max / (1 + exp(-(|E| - |E_bar|) / kappa)): asymptote, midpoint energy, slope factor. The fit is an unweighted least squares, that run against the magnitudes of the (non-positive) pulse energies.

Both the midpoint and the slope factor come out with the signs of the treatment own details, which means that the midpoint is written as a negative energy and the slope factor as a positive quantity. This step returns the three parameters as a float array. The choice of objective and the abscissa convention are part of the scientific contract, not just fitting mechanics: they pin the only reading of the fitted slope that the treatment energy-to-voltage conversion applies to.

Returns
-------
params : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.optimize import curve_fit

def fit_boltzmann(eps_family: np.ndarray, peak_fluxes: np.ndarray) -> np.ndarray:
    """Three-parameter Boltzmann fit of the peak flux-energy curve.

    Parameters
    ----------
    eps_family : np.ndarray
        Shape (m,), the pulse energies in kT (non-positive in the supplied family).
    peak_fluxes : np.ndarray
        Shape (m,), the ensemble-averaged peak flux at each pulse energy.

    Returns
    -------
    np.ndarray
        Shape (3,), native floats: [F_max, eps_bar, kappa], where eps_bar carries the
        treatment's sign convention (negative energy) and kappa is the positive slope
        factor, in kT.

    Raises
    ------
    ValueError
        If the inputs are inconsistent, shorter than three points, or non-finite.
    """
    params = np.empty(3, dtype=float)
    return params  # placeholder to complete!

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import curve_fit


def _oracle_fit_boltzmann(eps_family, peak_fluxes):
    """Reference implementation for fit_boltzmann."""
    import numpy as np
    from scipy.optimize import curve_fit

    def _fit_model(x, fm, xb, k):
        return fm / (1.0 + np.exp(-(x - xb) / k))

    eps_family = np.asarray(eps_family, dtype=float).ravel()
    peak_fluxes = np.asarray(peak_fluxes, dtype=float).ravel()
    if eps_family.size != peak_fluxes.size:
        raise ValueError("eps_family and peak_fluxes must be the same length")
    if eps_family.size < 3:
        raise ValueError("at least three points are needed for a three-parameter fit")
    if not (np.all(np.isfinite(eps_family)) and np.all(np.isfinite(peak_fluxes))):
        raise ValueError("inputs must be finite")
    p, _ = curve_fit(_fit_model, np.abs(eps_family), peak_fluxes,
                     p0=[float(peak_fluxes.max()), 7.0, 1.2], maxfev=200000)
    return np.array([float(p[0]), float(-p[1]), float(p[2])])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = (
        "import numpy as np\n"
        "eps = np.array([0., -2., -4., -6., -7., -8., -9., -10., -12., -14.])\n"
    )
    synth = setup + (
        "def boltz(x, fm, xb, k):\n"
        "    return fm / (1.0 + np.exp(-(np.abs(x) - xb) / k))\n"
    )
    invalid = setup + (
        "def run_model(e, p):\n"
        "    try:\n"
        "        fit_boltzmann(e, p)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
        "def run_gold(e, p):\n"
        "    try:\n"
        "        _oracle_fit_boltzmann(e, p)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
    )
    curves = [
        # Normal: the measured graded peaks. Fit reproduces the published-convention values.
        setup + "peaks = np.array([0.0024595516, 0.0136904502, 0.2716549121, 2.7152015702, 5.7895651440, 9.7709977838, 13.6833034428, 16.8673351408, 20.9637150550, 22.8610852062])\n",
        # Boundary: exact Boltzmann data on saturating support; the fit is well-posed and
        # kappa comes out positive against the magnitudes.
        synth + "peaks = boltz(eps, 100.0, 6.5, 1.1)\n",
        # Edge: a shallower curve over the same family (kappa large, F_max barely above the
        # largest peak).
        synth + "peaks = boltz(eps, 200.0, 4.0, 3.3)\n",
    ]
    return [
        # Normal: the measured graded peaks. Fit reproduces the published-convention values.
        {"setup": curves[0],
         "call": "fit_boltzmann(eps, peaks)",
         "gold_call": "_oracle_fit_boltzmann(eps, peaks)"},
        # Boundary: exact Boltzmann data on saturating support; the fit is well-posed and
        # kappa comes out positive against the magnitudes.
        {"setup": curves[1],
         "call": "fit_boltzmann(eps, peaks)",
         "gold_call": "_oracle_fit_boltzmann(eps, peaks)"},
        # Edge: a shallower curve over the same family (kappa large, F_max barely above the
        # largest peak).
        {"setup": curves[2],
         "call": "fit_boltzmann(eps, peaks)",
         "gold_call": "_oracle_fit_boltzmann(eps, peaks)"},
    ] + [
        # Invalid: two points for a three-parameter fit.
        {"setup": invalid,
         "call": "run_model(np.array([0., -2.]), np.array([0.0, 1.0]))",
         "gold_call": "run_gold(np.array([0., -2.]), np.array([0.0, 1.0]))"},
        # Invalid: mismatched lengths.
        {"setup": invalid,
         "call": "run_model(np.array([0., -2., -4.]), np.array([0.0, 1.0]))",
         "gold_call": "run_gold(np.array([0., -2., -4.]), np.array([0.0, 1.0]))"},
    ]
