"""
From the reduced transition rates, form the two effective parameters per pair that govern the source's scalar iterated map for the copy: for each copy unit m opposite template unit n with following unit nn, an effective rate of completed incorporation and an effective rate of completed removal, both obtained by eliminating the closed state exactly. Return params[m, n, nn, :] = (alpha_mn, beta_mnnn) with the same index convention as the rates.

Once the polymerase is closed on a nucleotide it either polymerizes or reopens and loses it, and once it has closed on the pyrophosphate of the last incorporated unit it either reopens or removes that unit. The competition between the two exits of each closed state defines the effective rates with which the copy grows and shrinks.

Returns
-------
ndarray of float64, shape (2, 2, 2, 2): params[m, n, nn] = (alpha, beta) in 1/s.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def backward_map_parameters(rates: "np.ndarray") -> "np.ndarray":
    """From the reduced transition rates, form the two effective parameters per pair that govern the source's scalar iterated map for the copy: for each copy unit m opposite template unit n with following unit nn, an effective rate of completed incorporation and an effective rate of completed removal, both obtained by eliminating the closed state exactly. Return params[m, n, nn, :] = (alpha_mn, beta_mnnn) with the same index convention as the rates.

    Parameters
    ----------
    rates : np.ndarray
        Reduced transition rates of shape (2, 2, 2, 4) from the previous step.

    Returns
    -------
    params : np.ndarray
        Effective incorporation and removal rates of shape (2, 2, 2, 2).

    Raises
    ------
    ValueError
        If rates is not a (2, 2, 2, 4) array of finite nonnegative values or some w_pol + w_FE is zero.
    """
    return params

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_backward_map_parameters(rates: "np.ndarray") -> "np.ndarray":
    """Eqs. (30)-(31): alpha[m,n] and beta[m,n,nn] of the scalar iterated function system.

    Returns params[m, n, nn, :] = (alpha, beta) with alpha = w_pol w_EF / (w_pol + w_FE) (effective
    rate of a completed incorporation) and beta = w_depol w_FE / (w_pol + w_FE) (effective rate of a
    completed removal); alpha does not depend on nn.
    """
    r = np.asarray(rates, dtype=np.float64)
    if r.shape != (2, 2, 2, 4) or not np.all(np.isfinite(r)) or np.any(r < 0.0):
        raise ValueError("rates must be a (2, 2, 2, 4) array of finite nonnegative rates")
    w_ef, w_fe, w_pol, w_dep = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    if np.any(w_pol + w_fe <= 0.0):
        raise ValueError("w_pol + w_FE must be positive for every pair")
    params = np.zeros((2, 2, 2, 2), dtype=np.float64)
    params[..., 0] = w_pol * w_ef / (w_pol + w_fe)
    params[..., 1] = w_dep * w_fe / (w_pol + w_fe)
    return params

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([2.0e-8, 4.0e-9])\nppi, kp_const = 1.0e-4, 5.6e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\n",
            "call": "np.asarray(backward_map_parameters(rates_m))",
            "gold_call": "np.asarray(_oracle_backward_map_parameters(rates_o))",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([1.0, 1.0])\nppi, kp_const = 1.0e-4, 5.6e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\n",
            "call": "np.asarray(backward_map_parameters(rates_m))",
            "gold_call": "np.asarray(_oracle_backward_map_parameters(rates_o))",
        },
        {
            "setup": "import numpy as np\nrates_m = np.full((2, 2, 2, 4), 1.0)\nrates_m[..., 3] = 0.0\nrates_o = rates_m.copy()\n",
            "call": "np.asarray(backward_map_parameters(rates_m))",
            "gold_call": "np.asarray(_oracle_backward_map_parameters(rates_o))",
        },
        {
            "setup": "import numpy as np\nrates_m = np.full((2, 2, 2, 4), 1.0)\nrates_m[0, 0, 0, 1] = -1.0\nrates_o = rates_m.copy()\ndef run_model():\n    try:\n        backward_map_parameters(rates_m)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_backward_map_parameters(rates_o)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
