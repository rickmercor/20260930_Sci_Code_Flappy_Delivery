"""
Build the four-velocity of the accreting matter from the prescribed three-velocity

components measured in the local zero-angular-momentum frame.

The accreting matter is referred to the local zero-angular-momentum frame of

the stationary axisymmetric spacetime, whose lapse and frame-dragging angular

velocity are



    kappa = sqrt(-g_tt + g_tph^2 / g_phph),      Omega = -g_tph / g_phph.



The flow's three-velocity is measured in that frame.  Its polar component

vanishes (the matter is injected along conical surfaces), and the radial and

azimuthal components are prescribed by the source as



    vhat^r   = -v_max (r_in / r)^p3

    vhat^phi = psi sqrt(1 / r) / (1 + lam / r)



Conventions fixed for this step:

* vhat^theta = 0 exactly, so u^x = 0 exactly.

* The Lorentz factor is built from the three-velocity measured in that frame.

Returns
-------
np.ndarray of shape (4,), the contravariant four-velocity [u^t, u^r, u^x, u^phi]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def zamo_flow_four_velocity(r: float, x: float, params: np.ndarray, b: float,
                            r_in: float, v_max: float, p3: float,
                            psi: float, lam: float) -> np.ndarray:
    '''Four-velocity of the accreting matter in the local black-hole frame.

    Parameters
    ----------
    r : float
        Radial coordinate.
    x : float
        Polar coordinate x = cos(theta).
    params : np.ndarray
        Array of shape (11,) as returned by ml_parameters_and_horizon.
    b : float
        Spindle deformation parameter B.
    r_in : float
        Inner boundary of the accretion model.
    v_max : float
        Maximum radial three-velocity of the flow.
    p3 : float
        Exponent controlling the radial acceleration of the flow.
    psi : float
        Overall rotational speed of the flow.
    lam : float
        Suppression scale for the azimuthal motion in the inner region.

    Returns
    -------
    result : np.ndarray
        Array of shape (4,), the contravariant four-velocity [u^t, u^r, u^x, u^phi].

    Raises
    ------
    ValueError
        If r is not positive, |x| is not less than 1, or the prescribed
        three-velocity is not subluminal.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_zamo_flow_four_velocity(r: float, x: float, params: np.ndarray, b: float, r_in: float, v_max: float, p3: float, psi: float, lam: float) -> np.ndarray:
    """Reference implementation."""
    if r <= 0.0:
        raise ValueError("r must be positive")
    if abs(x) >= 1.0:
        raise ValueError("|x| must be less than 1")
    g_tt, g_rr, g_xx, g_pp, g_tp = _oracle_ml_metric_components(r, x, params, b)
    kappa = np.sqrt(-g_tt + g_tp * g_tp / g_pp)
    omega = -g_tp / g_pp
    v_r = -v_max * (r_in / r) ** p3
    v_ph = psi * np.sqrt(1.0 / r) / (1.0 + lam / r)
    speed2 = v_r * v_r + v_ph * v_ph
    if speed2 >= 1.0:
        raise ValueError("three-velocity is not subluminal")
    gam = 1.0 / np.sqrt(1.0 - speed2)
    return np.array([gam / kappa,
                     gam * v_r / np.sqrt(g_rr),
                     0.0,
                     gam * (omega / kappa + v_ph / np.sqrt(g_pp))])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Candidate and oracle start from separate equivalent inputs."""
    exception_setup = 'def _exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n'
    return [
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)',
            "call": 'zamo_flow_four_velocity(*_review_deepcopy((8.0, np.cos(np.deg2rad(80.0)), par, 0.01, par[10], 0.9, 0.5, 0.9, 10.0)))',
            "gold_call": '_oracle_zamo_flow_four_velocity(*_review_deepcopy((8.0, np.cos(np.deg2rad(80.0)), par, 0.01, par[10], 0.9, 0.5, 0.9, 10.0)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)',
            "call": 'zamo_flow_four_velocity(*_review_deepcopy((5.0, 0.0, par, 0.01, 5.0, 0.9, 0.5, 0.9, 10.0)))',
            "gold_call": '_oracle_zamo_flow_four_velocity(*_review_deepcopy((5.0, 0.0, par, 0.01, 5.0, 0.9, 0.5, 0.9, 10.0)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)',
            "call": 'zamo_flow_four_velocity(*_review_deepcopy((3.0, 0.2, par, 0.01, par[10], 0.9, 0.5, 0.9, 0.0)))',
            "gold_call": '_oracle_zamo_flow_four_velocity(*_review_deepcopy((3.0, 0.2, par, 0.01, par[10], 0.9, 0.5, 0.9, 0.0)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\n\n\n' + exception_setup,
            "call": '_exception_code(zamo_flow_four_velocity, *_review_deepcopy((8.0, 0.0, par, 0.01, par[10], 3.0, 0.5, 0.9, 10.0)))',
            "gold_call": '_exception_code(_oracle_zamo_flow_four_velocity, *_review_deepcopy((8.0, 0.0, par, 0.01, par[10], 3.0, 0.5, 0.9, 10.0)))',
        },
    ]
