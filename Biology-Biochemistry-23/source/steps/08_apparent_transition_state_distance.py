"""
Orchestrate the pipeline: for each shear force build the transition rates and the master-equation generator, obtain the off-rate k(F) by the single-exponential fit over the thermal dissociation window t_end = exp(-(dG_non + (n_bp - 1) dG_s) / RT) in units of 1 / k_a with n_points samples, and return the apparent transition-state distance d = k_B T times the slope of the least-squares straight line of ln k(F) against F (Bell's relation), in nm. Call the earlier step functions rather than reimplementing them.

Bell's phenomenological law ln k(F) = ln k(0) + F d / k_B T turns the force dependence of the rupture rate into a single length, the apparent transition-state distance, which is the quantity single-molecule experiments report and which the kinetic model must reproduce.

Returns
-------
float, the apparent transition-state distance in nm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apparent_transition_state_distance(forces: "np.ndarray", n_bp: int, temperature: float, dh_stack: float,
                                               ds_stack: float, dh_non: float, ds_non: float, persistence_length: float,
                                               interphosphate_distance: float, n_points: int) -> float:
    """Orchestrate the pipeline: for each shear force build the transition rates and the master-equation generator, obtain the off-rate k(F) by the single-exponential fit over the thermal dissociation window t_end = exp(-(dG_non + (n_bp - 1) dG_s) / RT) in units of 1 / k_a with n_points samples, and return the apparent transition-state distance d = k_B T times the slope of the least-squares straight line of ln k(F) against F (Bell's relation), in nm. Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    forces : np.ndarray
        Increasing 1-D array of shear forces in pN (>= 2 values).
    n_bp : int
        Number of base pairs (>= 2).
    temperature : float
        Temperature in K.
    dh_stack : float
        Stacking enthalpy, kJ/mol.
    ds_stack : float
        Salt-corrected stacking entropy, kJ/(mol K).
    dh_non : float
        Non-stacking enthalpy, kJ/mol.
    ds_non : float
        Salt-corrected non-stacking entropy, kJ/(mol K).
    persistence_length : float
        lambda_ss in nm.
    interphosphate_distance : float
        l_ss in nm.
    n_points : int
        Number of sample times of the exponential fit (>= 3).

    Returns
    -------
    d : float
        Apparent transition-state distance in nm.

    Raises
    ------
    ValueError
        If forces is not an increasing array of at least two positive values, n_bp < 2, temperature is not positive, or n_points < 3.
    """
    return d

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


def _thermal_energy_kj_mol(temperature):
    """R T in kJ/mol."""
    return 1.380649e-23 * 6.02214076e23 * temperature / 1000.0


def _oracle_apparent_transition_state_distance(forces: "np.ndarray", n_bp: int, temperature: float, dh_stack: float,
                                               ds_stack: float, dh_non: float, ds_non: float, persistence_length: float,
                                               interphosphate_distance: float, n_points: int) -> float:
    """Apparent transition-state distance d (nm) = k_B T * slope of ln k(F) versus F over the given forces (Bell)."""
    Fs = np.asarray(forces, dtype=float)
    if Fs.ndim != 1 or Fs.size < 2 or not np.all(np.isfinite(Fs)) or np.any(Fs <= 0.0) or np.any(np.diff(Fs) <= 0.0):
        raise ValueError("forces must be an increasing 1-D array of at least two positive values")
    N = _check_int(n_bp, "n_bp", 2)
    T = _check_pos(temperature, "temperature")
    npts = _check_int(n_points, "n_points", 3)
    rt = _thermal_energy_kj_mol(T)
    kt = _thermal_energy_pn_nm(T)
    dg_s = float(dh_stack) - T * float(ds_stack)
    dg_non = float(dh_non) - T * float(ds_non)
    t_end = float(np.exp(-(dg_non + (N - 1) * dg_s) / rt))       # thermal dissociation time in 1/k_a
    # geometric and mechanical ingredients, evaluated here so that the chain is complete (reported, not returned)
    rod_lengths = [_oracle_helical_end_to_end_distance(n) for n in range(1, N + 1)]
    _oracle_rod_extension(rod_lengths[-1], float(Fs[0]), kt)
    _oracle_ssdna_extension(1, float(Fs[0]), kt, persistence_length, interphosphate_distance)
    _oracle_mechanical_work(2, float(Fs[0]), N, T, persistence_length, interphosphate_distance)
    ks = []
    for F in Fs:
        r = _oracle_transition_rates(F, N, T, dh_stack, ds_stack, dh_non, ds_non, persistence_length, interphosphate_distance)
        Tm = _oracle_master_equation_generator(N, r)
        ks.append(_oracle_rupture_off_rate(Tm, t_end, npts))
    slope = np.polyfit(Fs, np.log(np.asarray(ks)), 1)[0]
    return float(kt * slope)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndh_stack, ds_stack, dh_non, ds_non = -7.6 * 4.184, -0.0920, 4.6 * 4.184, 0.0502\nlam_ss, l_ss = 0.77, 0.7\nforces, n_bp, temperature, n_points = np.array([3.0, 6.0, 9.0]), 9, 303.15, 1000\n",
            "call": "apparent_transition_state_distance(forces, n_bp, temperature, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss, n_points)",
            "gold_call": "_oracle_apparent_transition_state_distance(forces, n_bp, temperature, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss, n_points)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\ndh_stack, ds_stack, dh_non, ds_non = -7.6 * 4.184, -0.0920, 4.6 * 4.184, 0.0502\nlam_ss, l_ss = 0.77, 0.7\nforces, n_bp, temperature, n_points = np.array([2.5, 5.0, 10.0]), 11, 298.15, 1000\n",
            "call": "apparent_transition_state_distance(forces, n_bp, temperature, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss, n_points)",
            "gold_call": "_oracle_apparent_transition_state_distance(forces, n_bp, temperature, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss, n_points)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\ndh_stack, ds_stack, dh_non, ds_non = -7.6 * 4.184, -0.0920, 4.6 * 4.184, 0.0502\nlam_ss, l_ss = 0.77, 0.7\nforces, n_bp, temperature, n_points = np.array([5.0, 10.0, 15.0]), 10, 303.15, 1000\n",
            "call": "apparent_transition_state_distance(forces, n_bp, temperature, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss, n_points)",
            "gold_call": "_oracle_apparent_transition_state_distance(forces, n_bp, temperature, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss, n_points)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\ndh_stack, ds_stack, dh_non, ds_non = -7.6 * 4.184, -0.0920, 4.6 * 4.184, 0.0502\nlam_ss, l_ss = 0.77, 0.7\ndef run_model():\n    try:\n        apparent_transition_state_distance(np.array([9.0, 3.0]), 9, 303.15, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss, 1000)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_apparent_transition_state_distance(np.array([9.0, 3.0]), 9, 303.15, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss, 1000)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
