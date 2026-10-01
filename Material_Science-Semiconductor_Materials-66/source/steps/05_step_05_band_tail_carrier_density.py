"""
Return the band-tail-assisted mobile electron density at the supplied local electrostatic potential or potentials.

This routine supplies the mobile-carrier statistical response required by the selected cryogenic transistor model. Only that response is modified; other material and occupation terms retain the actual lattice temperature. Follow the supplied scalar/1D-array shape contract, return densities in cm^-3, and use n_nodes for Gauss-Legendre potential quadrature with stable finite-precision evaluation.

Returns
-------
float or np.ndarray, the band-tail-assisted electron density nBT in cm^-3, scalar in and scalar out
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def band_tail_carrier_density(psi: Union[float, np.ndarray], Vch: float, T: float,
                              UBT: float, Usp: float, Voffset: float,
                              n_nodes: int) -> Union[float, np.ndarray]:
    '''Band-tail-assisted mobile electron density at one or more local potentials.

    Parameters
    ----------
    psi : float or np.ndarray
        Local electrostatic potential in volts, referenced to the intrinsic
        level. Either a finite scalar or a finite 1D array.
    Vch : float
        Channel potential in volts, finite.
    T : float
        Lattice temperature in kelvin, strictly positive.
    UBT : float
        Characteristic band-tail statistical voltage in volts, strictly
        positive.
    Usp : float
        Smoothing voltage of the statistical crossover, in volts, strictly
        positive.
    Voffset : float
        Offset in volts locating the crossover relative to the conduction-band
        reference potential. Finite, may be negative.
    n_nodes : int
        Number of Gauss-Legendre nodes, an integer of at least 2.

    Returns
    -------
    nBT : float or np.ndarray
        Mobile electron density in cm^-3. A native Python float when `psi` is a
        scalar, otherwise a 1D array of the same length as `psi`.

    Raises
    ------
    ValueError
        If `psi` is not a finite scalar or finite 1D array, if `Vch` or
        `Voffset` is not a finite scalar, if any of `T`, `UBT`, `Usp` is not a
        finite strictly positive scalar, or if `n_nodes` is not an integer
        greater than or equal to 2.
'''
    return nBT

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from typing import Union


def _bandgap(T):
    """Silicon band gap in eV at temperature T in kelvin."""
    return 1.170 - 4.730e-4 * T * T / (T + 636.0)


def _thermal_voltage(T):
    """Thermal voltage kT/q in volts."""
    return 1.380649e-23 * T / 1.602176634e-19


def _oracle_band_tail_carrier_density(psi: Union[float, np.ndarray], Vch: float, T: float,
                                      UBT: float, Usp: float, Voffset: float,
                                      n_nodes: int) -> Union[float, np.ndarray]:
    arr = np.asarray(psi, dtype=float)
    if arr.ndim > 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("psi must be a finite scalar or a finite 1D array")
    for name, v in (("Vch", Vch), ("Voffset", Voffset)):
        if not np.isscalar(v) or not np.isfinite(float(v)):
            raise ValueError("%s must be a finite scalar" % name)
    for name, v in (("T", T), ("UBT", UBT), ("Usp", Usp)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    if not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer greater than or equal to 2")

    T = float(T); Vch = float(Vch)
    ut = _thermal_voltage(T)
    psi_X = Vch + _bandgap(T) / 2.0 + float(Voffset)
    log_nX = _oracle_log_intrinsic_carrier_density(T, 2.86e19, 2.66e19) + (psi_X - Vch) / ut

    p = np.atleast_1d(arr)
    x, w = np.polynomial.legendre.leggauss(int(n_nodes))
    t = 0.5 * (p[:, None] - psi_X) * x[None, :] + 0.5 * (p[:, None] + psi_X)
    ww = 0.5 * (p[:, None] - psi_X) * w[None, :]
    W = 1.0 / (1.0 + np.exp(np.clip((t - psi_X) / float(Usp), -700.0, 700.0)))
    U_stat = W * float(UBT) + (1.0 - W) * ut
    out = np.exp(np.clip(log_nX + np.sum(ww / U_stat, axis=1), -700.0, 700.0))
    return float(out[0]) if arr.ndim == 0 else out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        band_tail_carrier_density(0.5739, 0.0, -1.0, 0.0291, 0.0998, 0.004088, 400)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_band_tail_carrier_density(0.5739, 0.0, -1.0, 0.0291, 0.0998, 0.004088, 400)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "import numpy as np",
            "call": "band_tail_carrier_density(0.5739, 0.0, 12.0, 0.0291, 0.0998, 0.004088, 400)",
            "gold_call": "_oracle_band_tail_carrier_density(0.5739, 0.0, 12.0, 0.0291, 0.0998, 0.004088, 400)",
        },
        {
            "setup": "import numpy as np\nP = np.array([-0.40, 0.0, 0.30, 0.5890, 0.65])\n",
            "call": "band_tail_carrier_density(P, 0.0, 12.0, 0.0291, 0.0998, 0.004088, 400)",
            "gold_call": "_oracle_band_tail_carrier_density(P, 0.0, 12.0, 0.0291, 0.0998, 0.004088, 400)",
        },
        {
            "setup": "import numpy as np",
            "call": "band_tail_carrier_density(0.5890354444444444, 0.0, 12.0, 0.0291, 0.0998, 0.004088, 400)",
            "gold_call": "_oracle_band_tail_carrier_density(0.5890354444444444, 0.0, 12.0, 0.0291, 0.0998, 0.004088, 400)",
        },
        {
            "setup": "import numpy as np",
            "call": "band_tail_carrier_density(0.6168, 0.1, 12.0, 0.0291, 0.0998, 0.004088, 400)",
            "gold_call": "_oracle_band_tail_carrier_density(0.6168, 0.1, 12.0, 0.0291, 0.0998, 0.004088, 400)",
        },
        {
            "setup": "import numpy as np",
            "call": "band_tail_carrier_density(0.55, 0.0, 250.0, 0.0291, 0.0998, 0.004088, 400)",
            "gold_call": "_oracle_band_tail_carrier_density(0.55, 0.0, 250.0, 0.0291, 0.0998, 0.004088, 400)",
        },
        {
            "setup": "import numpy as np\nP = np.array([0.20, 0.45])\n",
            "call": "band_tail_carrier_density(P, 0.0, 12.0, 0.0120, 0.0400, -0.010, 200)",
            "gold_call": "_oracle_band_tail_carrier_density(P, 0.0, 12.0, 0.0120, 0.0400, -0.010, 200)",
        },
            {   # a strong band tail with an abrupt crossover placed below the reference
            "setup": 'import numpy as np',
            "call": "band_tail_carrier_density(np.array([0.45, 0.55, 0.60, 0.65]), 0.0, 12.0, 0.08, 0.02, -0.02, 400)",
            "gold_call": "_oracle_band_tail_carrier_density(np.array([0.45, 0.55, 0.60, 0.65]), 0.0, 12.0, 0.08, 0.02, -0.02, 400)",
        },
        {   # intermediate temperature, where the two limits are closer together
            "setup": 'import numpy as np',
            "call": "band_tail_carrier_density(0.50, 0.0, 77.0, 0.0291, 0.0998, 0.004088, 400)",
            "gold_call": "_oracle_band_tail_carrier_density(0.50, 0.0, 77.0, 0.0291, 0.0998, 0.004088, 400)",
        },
]
