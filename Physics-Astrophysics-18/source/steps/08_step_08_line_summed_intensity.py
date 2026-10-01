"""
Trace every listed screen point through the full pipeline, apply the horizon-capture and

escape rules, and return the summed observed specific intensity.

Two termination rules apply to every ray:



* Horizon capture.  A ray that reaches 1.0001 times the outer horizon radius

  is terminated there and contributes the intensity accumulated up to that

  point.

* Escape.  A ray is terminated on an outward crossing of the observer radius

  and contributes the intensity accumulated up to that point.



Conventions fixed for this step:

* Integration uses scipy.integrate.solve_ivp with method "DOP853" and

  terminal events, at the supplied tolerances.

* This step orchestrates the earlier steps ml_parameters_and_horizon,

  screen_to_initial_conditions and transfer_rhs.

Returns
-------
float, the sum of the observed specific intensity over the listed screen points, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from typing import Sequence
import numpy as np
from scipy.integrate import solve_ivp


def line_summed_intensity(alphas: Sequence[float], beta: float, mass: float, chi: float, b: float,
                          r_o: float, inclination: float, emis_params: dict,
                          absorb_params: dict, flow_params: dict,
                          rtol: float, atol: float) -> float:
    '''Summed observed specific intensity over a row of screen points.

    Parameters
    ----------
    alphas : Sequence[float]
        Horizontal screen coordinates of the pixels, in units of mass.
    beta : float
        Vertical screen coordinate shared by all the pixels.
    mass : float
        Physical mass M of the black hole.
    chi : float
        Dimensionless spin J / M^2.
    b : float
        Spindle deformation parameter B.
    r_o : float
        Observer radius.
    inclination : float
        Observer polar angle in radians.
    emis_params : dict
        Emission parameters; see step 05.
    absorb_params : dict
        Absorption parameters with keys alpha0, alpha1, alpha2, alpha3.
    flow_params : dict
        Flow parameters with keys v_max, p3, psi, lam.
    rtol : float
        Relative tolerance of the integrator.
    atol : float
        Absolute tolerance of the integrator.

    Returns
    -------
    result : float
        The sum of the observed specific intensity over the pixels, as a
        native Python float.

    Raises
    ------
    ValueError
        If alphas is empty, or either tolerance is not positive.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from typing import Sequence
import numpy as np
from scipy.integrate import solve_ivp


def _oracle_line_summed_intensity(alphas: Sequence[float], beta: float, mass: float, chi: float, b: float, r_o: float, inclination: float, emis_params: dict, absorb_params: dict, flow_params: dict, rtol: float, atol: float) -> float:
    """Reference implementation."""
    if len(alphas) == 0:
        raise ValueError("alphas must not be empty")
    if rtol <= 0.0 or atol <= 0.0:
        raise ValueError("tolerances must be positive")
    params = _oracle_ml_parameters_and_horizon(mass, chi, b)
    r_in = params[10]

    def _captured(tau, state, *args):
        return state[1] - 1.0001 * r_in
    _captured.terminal = True
    _captured.direction = -1

    def _escaped(tau, state, *args):
        return state[1] - (r_o - 1e-6)
    _escaped.terminal = True
    _escaped.direction = +1

    total = 0.0
    for alpha in alphas:
        ic = _oracle_screen_to_initial_conditions(float(alpha), beta, params, b,
                                                  r_o, inclination)
        state0 = np.concatenate([ic, [0.0, 0.0]])
        sol = solve_ivp(_oracle_transfer_rhs, (0.0, 5.0e4), state0, method="DOP853",
                        rtol=rtol, atol=atol, events=(_captured, _escaped),
                        args=(params, b, r_in, emis_params, absorb_params, flow_params))
        total += float(sol.y[9, -1])      # captured or escaped: keep what was accumulated
    return float(total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Candidate and oracle start from separate equivalent inputs."""
    exception_setup = 'def _exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n'
    return [
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\nEM = dict(j0=1.0, j1=1.0, j2=1.0, j3=1.0, p1=-1.5, p2=-0.5, sigma_dtheta=0.1, beta=0.1, r_p=1.5, w_p=3.0, j_p=1.0, r_b=6.0, sigma_br=0.5, sigma_btheta=0.15, r_s=8.0, theta_s=np.pi / 2, phi_s=3 * np.pi / 2, sigma_sr=2.0, sigma_stheta=np.pi / 36, sigma_sphi=np.pi / 18)\nAB = dict(alpha0=0.02, alpha1=1.0, alpha2=0.5, alpha3=0.5)\nFL = dict(v_max=0.9, p3=0.5, psi=0.9, lam=10.0)\nROW60 = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]',
            "call": 'line_summed_intensity(*_review_deepcopy((ROW60, 0.0, 1.0, 0.94, 0.01, 50.0, np.deg2rad(60.0), EM, AB, FL, 1e-09, 1e-11)))',
            "gold_call": '_oracle_line_summed_intensity(*_review_deepcopy((ROW60, 0.0, 1.0, 0.94, 0.01, 50.0, np.deg2rad(60.0), EM, AB, FL, 1e-09, 1e-11)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\nEM = dict(j0=1.0, j1=1.0, j2=1.0, j3=1.0, p1=-1.5, p2=-0.5, sigma_dtheta=0.1, beta=0.1, r_p=1.5, w_p=3.0, j_p=1.0, r_b=6.0, sigma_br=0.5, sigma_btheta=0.15, r_s=8.0, theta_s=np.pi / 2, phi_s=3 * np.pi / 2, sigma_sr=2.0, sigma_stheta=np.pi / 36, sigma_sphi=np.pi / 18)\nAB = dict(alpha0=0.02, alpha1=1.0, alpha2=0.5, alpha3=0.5)\nFL = dict(v_max=0.9, p3=0.5, psi=0.9, lam=10.0)',
            "call": 'line_summed_intensity(*_review_deepcopy(([0.0], 0.0, 1.0, 0.94, 0.01, 50.0, np.deg2rad(80.0), EM, AB, FL, 1e-09, 1e-11)))',
            "gold_call": '_oracle_line_summed_intensity(*_review_deepcopy(([0.0], 0.0, 1.0, 0.94, 0.01, 50.0, np.deg2rad(80.0), EM, AB, FL, 1e-09, 1e-11)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\nEM = dict(j0=1.0, j1=1.0, j2=1.0, j3=1.0, p1=-1.5, p2=-0.5, sigma_dtheta=0.1, beta=0.1, r_p=1.5, w_p=3.0, j_p=1.0, r_b=6.0, sigma_br=0.5, sigma_btheta=0.15, r_s=8.0, theta_s=np.pi / 2, phi_s=3 * np.pi / 2, sigma_sr=2.0, sigma_stheta=np.pi / 36, sigma_sphi=np.pi / 18)\nAB = dict(alpha0=0.02, alpha1=1.0, alpha2=0.5, alpha3=0.5)\nFL = dict(v_max=0.9, p3=0.5, psi=0.9, lam=10.0)\nRECEDING = [-10.0, -8.0]',
            "call": 'line_summed_intensity(*_review_deepcopy((RECEDING, 0.0, 1.0, 0.94, 0.01, 50.0, np.deg2rad(80.0), EM, AB, FL, 1e-09, 1e-11)))',
            "gold_call": '_oracle_line_summed_intensity(*_review_deepcopy((RECEDING, 0.0, 1.0, 0.94, 0.01, 50.0, np.deg2rad(80.0), EM, AB, FL, 1e-09, 1e-11)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\nEM0 = dict(j0=0.0, j1=1.0, j2=1.0, j3=1.0, p1=-1.5, p2=-0.5, sigma_dtheta=0.1, beta=0.1, r_p=1.5, w_p=3.0, j_p=1.0, r_b=6.0, sigma_br=0.5, sigma_btheta=0.15, r_s=8.0, theta_s=np.pi / 2, phi_s=3 * np.pi / 2, sigma_sr=2.0, sigma_stheta=np.pi / 36, sigma_sphi=np.pi / 18)\nAB = dict(alpha0=0.02, alpha1=1.0, alpha2=0.5, alpha3=0.5)\nFL = dict(v_max=0.9, p3=0.5, psi=0.9, lam=10.0)',
            "call": 'line_summed_intensity(*_review_deepcopy(([-10.0, 10.0], 0.0, 1.0, 0.94, 0.01, 50.0, np.deg2rad(80.0), EM0, AB, FL, 1e-09, 1e-11)))',
            "gold_call": '_oracle_line_summed_intensity(*_review_deepcopy(([-10.0, 10.0], 0.0, 1.0, 0.94, 0.01, 50.0, np.deg2rad(80.0), EM0, AB, FL, 1e-09, 1e-11)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\nEM = dict(j0=1.0, j1=1.0, j2=1.0, j3=1.0, p1=-1.5, p2=-0.5, sigma_dtheta=0.1, beta=0.1, r_p=1.5, w_p=3.0, j_p=1.0, r_b=6.0, sigma_br=0.5, sigma_btheta=0.15, r_s=8.0, theta_s=np.pi / 2, phi_s=3 * np.pi / 2, sigma_sr=2.0, sigma_stheta=np.pi / 36, sigma_sphi=np.pi / 18)\nAB = dict(alpha0=0.02, alpha1=1.0, alpha2=0.5, alpha3=0.5)\nFL = dict(v_max=0.9, p3=0.5, psi=0.9, lam=10.0)\n\n\n' + exception_setup,
            "call": '_exception_code(line_summed_intensity, *_review_deepcopy(([], 0.0, 1.0, 0.94, 0.01, 50.0, np.deg2rad(80.0), EM, AB, FL, 1e-09, 1e-11)))',
            "gold_call": '_exception_code(_oracle_line_summed_intensity, *_review_deepcopy(([], 0.0, 1.0, 0.94, 0.01, 50.0, np.deg2rad(80.0), EM, AB, FL, 1e-09, 1e-11)))',
        },
    ]
