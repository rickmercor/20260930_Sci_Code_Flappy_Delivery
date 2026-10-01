"""
Orchestrate the whole pipeline: build the reduced rates and the effective parameters, check with the onset step that the configuration grows (return 0.0 if the onset scale is at or above 1), run the backward iteration around the period, form the site transfer entries, verify that the error probability is a valid probability, and return the mean growth velocity in nucleotides per second. Call the earlier step functions rather than reimplementing them.

The end-to-end quantity is the steady-state elongation rate of the polymerase on the given periodic template at the given nucleotide and pyrophosphate concentrations, the observable that single-molecule and bulk replication assays measure.

Returns
-------
float, the mean growth velocity v in nucleotides per second (0.0 if growth is stalled).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def steady_growth_velocity(kin: "np.ndarray", conc: "np.ndarray", ppi: float, kp_const: float,
                                   template: "np.ndarray") -> float:
    """Orchestrate the whole pipeline: build the reduced rates and the effective parameters, check with the onset step that the configuration grows (return 0.0 if the onset scale is at or above 1), run the backward iteration around the period, form the site transfer entries, verify that the error probability is a valid probability, and return the mean growth velocity in nucleotides per second. Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    kin : np.ndarray
        Kinetic table of shape (2, 2, 4).
    conc : np.ndarray
        Nucleotide concentrations ([dATP], [dTTP]) in M.
    ppi : float
        Pyrophosphate concentration in M.
    kp_const : float
        Pyrophosphorolysis constant K_P in M.
    template : np.ndarray
        One period of the template as integers 0 (A) and 1 (T).

    Returns
    -------
    v : float
        Mean growth velocity in nt/s.

    Raises
    ------
    ValueError
        If any input is invalid as in the earlier steps.
    """
    return v

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_kin(kin):
    k = np.asarray(kin, dtype=np.float64)
    if k.shape != (2, 2, 4) or not np.all(np.isfinite(k)) or np.any(k <= 0.0):
        raise ValueError("kin must be a (2, 2, 4) array of finite positive constants")
    return k


def _check_conc(conc):
    c = np.asarray(conc, dtype=np.float64).ravel()
    if c.shape != (2,) or not np.all(np.isfinite(c)) or np.any(c <= 0.0):
        raise ValueError("conc must be two finite positive concentrations (dATP, dTTP)")
    return c


def _check_positive(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(x)


def _check_template(template):
    t = np.asarray(template).ravel()
    if t.size < 1 or t.dtype.kind not in "iu" or np.any((t != 0) & (t != 1)):
        raise ValueError("template must be a non-empty integer array of 0 (A) and 1 (T)")
    return t.astype(np.int64)


def _oracle_steady_growth_velocity(kin: "np.ndarray", conc: "np.ndarray", ppi: float, kp_const: float,
                                   template: "np.ndarray") -> float:
    """End to end: reduced rates -> IFS parameters -> backward iteration -> transfer factors -> v.
    Returns the mean growth velocity in nucleotides per second; 0.0 if the configuration lies at or
    below the onset of growth (onset scale factor >= 1), where the exact solution has x_l -> 0."""
    k = _check_kin(kin)
    c = _check_conc(conc)
    ppi = _check_positive(ppi, "ppi")
    kp_const = _check_positive(kp_const, "kp_const")
    t = _check_template(template)
    if _oracle_onset_of_growth_scale(k, c, ppi, kp_const, t) >= 1.0:
        return 0.0
    rates = _oracle_reduced_transition_rates(k, c, ppi, kp_const)
    params = _oracle_backward_map_parameters(rates)
    x = _oracle_backward_iteration(params, t)
    Y = _oracle_site_transfer_factors(x, params, rates, t)
    eta = _oracle_error_probability(Y, t)
    if not (0.0 <= eta <= 1.0):
        raise ValueError("error probability outside [0, 1]: inconsistent transfer factors")
    return _oracle_mean_growth_velocity(x, Y)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([2.0e-8, 4.0e-9])\nppi, kp_const = 1.0e-4, 5.6e-3\n",
            "call": "steady_growth_velocity(kin, conc, ppi, kp_const, template)",
            "gold_call": "_oracle_steady_growth_velocity(kin, conc, ppi, kp_const, template)",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([1.0, 1.0])\nppi, kp_const = 1.0e-4, 5.6e-3\n",
            "call": "steady_growth_velocity(kin, conc, ppi, kp_const, template)",
            "gold_call": "_oracle_steady_growth_velocity(kin, conc, ppi, kp_const, template)",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([1.0e-9, 2.0e-10])\nppi, kp_const = 1.0e-4, 5.6e-3\n",
            "call": "steady_growth_velocity(kin, conc, ppi, kp_const, template)",
            "gold_call": "_oracle_steady_growth_velocity(kin, conc, ppi, kp_const, template)",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([2.0e-8, 4.0e-9])\nppi, kp_const = 1.0e-4, 5.6e-3\ndef run_model():\n    try:\n        steady_growth_velocity(kin, conc, ppi, kp_const, np.array([]))\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_steady_growth_velocity(kin, conc, ppi, kp_const, np.array([]))\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
