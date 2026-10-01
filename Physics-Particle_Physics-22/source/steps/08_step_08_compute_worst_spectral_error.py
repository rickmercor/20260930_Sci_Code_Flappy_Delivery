"""
Return the worst continuous-phase error of the unrefined matter-spectrum propagation.

For every supplied energy and zenith cosine, form the identical finite

path-averaged layer model for unrefined and exact propagation. Include both

neutrino signs and the muon-to-electron and muon-to-muon channels. The scalar is



$$

\mathcal E=10^6\max_{E,c,\sigma,\beta}\max_{\delta\in[0,2\pi)}

|P_{\mu\to\beta}^{(0)}-P_{\mu\to\beta}^{\mathrm{exact}}|.

$$



The outer scan is discrete; the inner phase optimization is continuous.

The finite piecewise-constant Earth model is shared by both branches,

so its density discretization error is excluded from this statistic.

Do not replace the zero-correction branch by a refined spectrum or impose

unitarity on its approximate spectral matrices.

Returns
-------
A finite nonnegative Python float containing one million times the largest continuous-phase probability error.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_worst_spectral_error(
    radii_km: "np.ndarray",
    density_coefficients: "np.ndarray",
    electron_fractions: "np.ndarray",
    energies_gev: "np.ndarray",
    cos_zeniths: "np.ndarray",
    mixing: "np.ndarray",
    detector_depth_km: float = 2.0,
    subdivisions: int = 3,
    quadrature_order: int = 16,
) -> float:
    r"""Return the worst continuous-phase error of the unrefined matter-spectrum propagation.

    Parameters
    ----------
    radii_km : np.ndarray
        Finite positive increasing outer shell radii, shape (S,), in km.
    density_coefficients : np.ndarray
        Finite real shape (S, 4) ascending polynomial coefficients in r/R,
        in grams per cubic centimeter; sampled densities must be nonnegative.
    electron_fractions : np.ndarray
        Finite real shape (S,) fractions in [0, 1].
    energies_gev : np.ndarray
        Nonempty finite real positive energy vector, shape (J,), in GeV.
    cos_zeniths : np.ndarray
        Nonempty finite real direction cosines, shape (K,), in [-1, 1].
    mixing : np.ndarray
        Finite real shape (5,) vector s12_sq, s13_sq, s23_sq, m21, m31;
        first two entries in (0, 1), third in [0, 1], and 0 < m21 < m31.
        Mass-squared entries are in eV squared; CP phase is optimized.
    detector_depth_km : float, default 2.0
        Finite detector depth in [0, R), in km.
    subdivisions : int, default 3
        Positive integer pieces per geometric shell interval.
    quadrature_order : int, default 16
        Positive integer midpoint samples per piece.

    Returns
    -------
    error : float
        Finite nonnegative dimensionless error scaled by one million.

    Raises
    ------
    ValueError
        If an input violates its domain, sampled density is negative,
        an approximate spectrum or projector pivot cannot be recovered,
        or a probability harmonic or final maximum is nonfinite.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_worst_spectral_error(
    radii_km: "np.ndarray",
    density_coefficients: "np.ndarray",
    electron_fractions: "np.ndarray",
    energies_gev: "np.ndarray",
    cos_zeniths: "np.ndarray",
    mixing: "np.ndarray",
    detector_depth_km: float = 2.0,
    subdivisions: int = 3,
    quadrature_order: int = 16,
) -> float:
    parameters = _real_array(mixing, (5,), "mixing")
    reduced = parameters[[0, 1, 3, 4]]
    _validate_mixing(*reduced)
    if not 0 <= parameters[2] <= 1:
        raise ValueError("s23_sq must be in [0, 1]")
    raw_energies, raw_cosines = np.asarray(energies_gev), np.asarray(cos_zeniths)
    if (
        raw_energies.ndim != 1
        or raw_energies.size == 0
        or raw_cosines.ndim != 1
        or raw_cosines.size == 0
    ):
        raise ValueError("energy and direction vectors must be nonempty")
    energies = _real_array(energies_gev, raw_energies.shape, "energies_gev")
    cosines = _real_array(cos_zeniths, raw_cosines.shape, "cos_zeniths")
    if np.any(energies <= 0) or np.any(np.abs(cosines) > 1):
        raise ValueError("invalid energy or direction")
    maximum = 0.0
    for direction in cosines:
        segments = _oracle_trace_shell_segments(
            radii_km, direction, detector_depth_km, subdivisions
        )
        layers = _oracle_average_segment_densities(
            segments,
            radii_km,
            density_coefficients,
            electron_fractions,
            direction,
            detector_depth_km,
            quadrature_order,
        )
        for energy in energies:
            for charge in (1, -1):
                amplitudes = _oracle_propagate_spectral_pair(
                    layers, energy, reduced, charge
                )
                harmonics = _oracle_compute_error_harmonics(
                    amplitudes, parameters[2], charge
                )
                for coefficients in harmonics:
                    maximum = max(maximum, _oracle_maximize_phase_error(coefficients))
    result = float(1e6 * maximum)
    if not np.isfinite(result):
        raise ValueError("scaled maximum must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative numerical and invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nr=np.array([1221.5,3480.,5701.,6371.])\nb=np.array([[13.,0.,-2.,0.],[12.,-1.2,-2.5,.3],[7.5,-2.,-.4,0.],[5.2,-.7,0.,0.]])\ny=np.array([.467,.467,.495,.495])\ne=np.array([2.1,3.8,6.2,9.7])\nc=np.array([-.97,-.84,-.41])\np=np.array([.307,.02195,.561,7.49e-5,.002534])\n",
            "call": "compute_worst_spectral_error(r.copy(), b.copy(), y.copy(), e.copy(), c.copy(), p.copy())",
            "gold_call": "_oracle_compute_worst_spectral_error(r.copy(), b.copy(), y.copy(), e.copy(), c.copy(), p.copy())",
            "tol": 0.0001,
        },
        {
            "setup": "import numpy as np\nr=np.array([1221.5,3480.,5701.,6371.])\nb=np.array([[13.,0.,-2.,0.],[12.,-1.2,-2.5,.3],[7.5,-2.,-.4,0.],[5.2,-.7,0.,0.]])\ny=np.array([.467,.467,.495,.495])\ne=np.array([2.1,3.8,6.2,9.7])\nc=np.array([-.97,-.84,-.41])\np=np.array([.307,.02195,.561,7.49e-5,.002534])\nc=np.array([.4])\n",
            "call": "compute_worst_spectral_error(r.copy(), b.copy(), y.copy(), e.copy(), c.copy(), p.copy(), 0.0)",
            "gold_call": "_oracle_compute_worst_spectral_error(r.copy(), b.copy(), y.copy(), e.copy(), c.copy(), p.copy(), 0.0)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nr=np.array([1221.5,3480.,5701.,6371.])\nb=np.array([[13.,0.,-2.,0.],[12.,-1.2,-2.5,.3],[7.5,-2.,-.4,0.],[5.2,-.7,0.,0.]])\ny=np.array([.467,.467,.495,.495])\ne=np.array([2.1,3.8,6.2,9.7])\nc=np.array([-.97,-.84,-.41])\np=np.array([.307,.02195,.561,7.49e-5,.002534])\ne=np.array([1.7,4.6])\nc=np.array([-.92,-.2])\np[2]=.38\n",
            "call": "compute_worst_spectral_error(r.copy(), b.copy(), y.copy(), e.copy(), c.copy(), p.copy(), 1.0, 2, 11)",
            "gold_call": "_oracle_compute_worst_spectral_error(r.copy(), b.copy(), y.copy(), e.copy(), c.copy(), p.copy(), 1.0, 2, 11)",
            "tol": 0.0001,
        },
        {
            "setup": "import numpy as np\nr=np.array([1221.5,3480.,5701.,6371.])\nb=np.array([[13.,0.,-2.,0.],[12.,-1.2,-2.5,.3],[7.5,-2.,-.4,0.],[5.2,-.7,0.,0.]])\ny=np.array([.467,.467,.495,.495])\ne=np.array([2.1,3.8,6.2,9.7])\nc=np.array([-.97,-.84,-.41])\np=np.array([.307,.02195,.561,7.49e-5,.002534])\ne[0]=0.0\ndef _raises_value_error(function):\n    try:\n        function()\n    except ValueError:\n        return 1\n    return 0\n",
            "call": "_raises_value_error(lambda: compute_worst_spectral_error(r.copy(), b.copy(), y.copy(), e.copy(), c.copy(), p.copy()))",
            "gold_call": "_raises_value_error(lambda: _oracle_compute_worst_spectral_error(r.copy(), b.copy(), y.copy(), e.copy(), c.copy(), p.copy()))",
            "tol": 0.0,
        },
    ]
