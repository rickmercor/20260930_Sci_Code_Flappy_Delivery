"""
Return the mechanical work W_n(F) in pN nm associated with the formation of the n-th base pair (the n_closed - 1 -> n_closed transition) of an n_bp duplex under shear at temperature T (K): the integral from 0 to F of the change in construct length dx_n(f), where dx_n(f) is the extension of the n_closed-pair duplex rod (helical length, previous steps) minus that of the (n_closed - 1)-pair rod (a zero-pair duplex has zero length) plus the extension of the released strand with n_bp - n_closed nucleotides minus that with n_bp - n_closed + 1 nucleotides. Use k_B = 1.380649e-23 J/K.

Forming a base pair shortens the released single strands and lengthens the duplex; under a constant force the free-energy change of the transition acquires the mechanical work of that length change, integrated over force because both extensions are force dependent.

Returns
-------
float, the work W_n(F) in pN nm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mechanical_work(n_closed: int, force: float, n_bp: int, temperature: float, persistence_length: float,
                            interphosphate_distance: float) -> float:
    """Return the mechanical work W_n(F) in pN nm associated with the formation of the n-th base pair (the n_closed - 1 -> n_closed transition) of an n_bp duplex under shear at temperature T (K): the integral from 0 to F of the change in construct length dx_n(f), where dx_n(f) is the extension of the n_closed-pair duplex rod (helical length, previous steps) minus that of the (n_closed - 1)-pair rod (a zero-pair duplex has zero length) plus the extension of the released strand with n_bp - n_closed nucleotides minus that with n_bp - n_closed + 1 nucleotides. Use k_B = 1.380649e-23 J/K.

    Parameters
    ----------
    n_closed : int
        Number of closed pairs after the transition (2 <= n_closed <= n_bp).
    force : float
        Shear force F in pN (> 0).
    n_bp : int
        Total number of base pairs of the duplex (>= 2).
    temperature : float
        Temperature in K (> 0).
    persistence_length : float
        lambda_ss in nm.
    interphosphate_distance : float
        l_ss in nm.

    Returns
    -------
    work : float
        Work in pN nm.

    Raises
    ------
    ValueError
        If n_closed or n_bp are out of range, or force, temperature or the polymer parameters are not finite and positive.
    """
    return work

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit


def _check_pos(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(x)


def _check_int(n, name, lo):
    if isinstance(n, bool) or int(n) != n or int(n) < lo:
        raise ValueError(f"{name} must be an integer >= {lo}")
    return int(n)


def _thermal_energy_pn_nm(temperature):
    """k_B T in pN nm."""
    return 1.380649e-23 * temperature * 1e21


def _length_change(f, len_n, len_n1, n_released, kt, lam, l_ss):
    """dx_n(f): duplex rod gain minus single-strand loss for the (n-1) -> n base-pair formation."""
    ds = _oracle_rod_extension(len_n, f, kt) - _oracle_rod_extension(len_n1, f, kt)
    ss = (_oracle_ssdna_extension(n_released, f, kt, lam, l_ss)
          - _oracle_ssdna_extension(n_released + 1, f, kt, lam, l_ss))
    return ds + ss


def _oracle_mechanical_work(n_closed: int, force: float, n_bp: int, temperature: float, persistence_length: float,
                            interphosphate_distance: float) -> float:
    """W_n(F) = int_0^F dx_n(f) df (pN nm), eqs (3)-(5): the work of the n-1 -> n base-pair formation step.

    dx_n(f) = [rod extension of the helical n-bp duplex - rod extension of the (n-1)-bp duplex]
              + [<z>_{N-n} - <z>_{N-n+1}] with the Marko-Siggia strand extensions; a 0-bp duplex has zero length.
    """
    n = _check_int(n_closed, "n_closed", 2)
    N = _check_int(n_bp, "n_bp", 2)
    if n > N:
        raise ValueError("n_closed must not exceed n_bp")
    F = _check_pos(force, "force")
    T = _check_pos(temperature, "temperature")
    lam = _check_pos(persistence_length, "persistence_length")
    l_ss = _check_pos(interphosphate_distance, "interphosphate_distance")
    kt = _thermal_energy_pn_nm(T)
    Ln = _oracle_helical_end_to_end_distance(n)
    Ln1 = _oracle_helical_end_to_end_distance(n - 1) if n - 1 >= 1 else 0.0
    val, _ = quad(_length_change, 1e-9, F, args=(Ln, Ln1, N - n, kt, lam, l_ss), limit=200, epsabs=1e-12, epsrel=1e-12)
    return float(val)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nn_closed, force, n_bp, temperature, persistence_length, interphosphate_distance = 5, 6.0, 9, 303.15, 0.77, 0.7\n",
            "call": "mechanical_work(n_closed, force, n_bp, temperature, persistence_length, interphosphate_distance)",
            "gold_call": "_oracle_mechanical_work(n_closed, force, n_bp, temperature, persistence_length, interphosphate_distance)",
        },
        {
            "setup": "import numpy as np\nn_closed, force, n_bp, temperature, persistence_length, interphosphate_distance = 2, 3.0, 9, 303.15, 0.77, 0.7\n",
            "call": "mechanical_work(n_closed, force, n_bp, temperature, persistence_length, interphosphate_distance)",
            "gold_call": "_oracle_mechanical_work(n_closed, force, n_bp, temperature, persistence_length, interphosphate_distance)",
        },
        {
            "setup": "import numpy as np\nn_closed, force, n_bp, temperature, persistence_length, interphosphate_distance = 11, 12.0, 11, 298.15, 0.77, 0.7\n",
            "call": "mechanical_work(n_closed, force, n_bp, temperature, persistence_length, interphosphate_distance)",
            "gold_call": "_oracle_mechanical_work(n_closed, force, n_bp, temperature, persistence_length, interphosphate_distance)",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        mechanical_work(10, 6.0, 9, 303.15, 0.77, 0.7)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_mechanical_work(10, 6.0, 9, 303.15, 0.77, 0.7)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
