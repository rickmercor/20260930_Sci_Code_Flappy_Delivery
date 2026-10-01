"""
Convert a point on the observer's screen into the position and covariant momentum of the

backward-traced ray that reaches it.

The spacetime is not asymptotically flat, so an image plane at large distance

with parallel rays is invalid.  The observer is placed at a finite radius in

the locally nonrotating frame, with lapse and angular velocity



    N_o = sqrt((g_tph^2 - g_tt g_phph) / g_phph),   omega_o = -g_tph / g_phph,



evaluated at the observer, and screen coordinates are defined by central

projection of the locally measured photon momentum with respect to the local

radial direction, scaled by the observer radius:



    alpha = r_o p^(phi) / p^(r),      beta = -r_o p^(x) / p^(r).



This step uses that relation in the inverse direction.



Conventions fixed for this step:

* The overall normalisation of the null momentum is fixed by requiring the

  photon frequency measured in the observer's frame to be unity.

* The returned ray is the backward-traced ray: the arriving photon's

  momentum with every component reversed, so it is past-directed (p_t > 0)

  and moving inward (p_r < 0).

* inclination is the observer's polar angle in radians; x_o = cos(inclination).

* t = 0 and phi = 0 at the observer.

Returns
-------
np.ndarray of shape (8,), [t, r, x, phi, p_t, p_r, p_x, p_phi]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def screen_to_initial_conditions(alpha: float, beta: float, params: np.ndarray,
                                 b: float, r_o: float, inclination: float) -> np.ndarray:
    '''Initial conditions of a backward-traced ray from a point on the observer screen.

    Parameters
    ----------
    alpha : float
        Horizontal screen coordinate, in units of mass.
    beta : float
        Vertical screen coordinate, in units of mass.
    params : np.ndarray
        Array of shape (11,) as returned by ml_parameters_and_horizon.
    b : float
        Spindle deformation parameter B.
    r_o : float
        Observer radius.
    inclination : float
        Observer polar angle in radians.

    Returns
    -------
    result : np.ndarray
        Array of shape (8,): [t, r, x, phi, p_t, p_r, p_x, p_phi].

    Raises
    ------
    ValueError
        If r_o is not positive, or inclination is not strictly between 0 and
        pi (the observer frame is singular on the axis).
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_screen_to_initial_conditions(alpha: float, beta: float, params: np.ndarray, b: float, r_o: float, inclination: float) -> np.ndarray:
    """Reference implementation."""
    if r_o <= 0.0:
        raise ValueError("r_o must be positive")
    if not (0.0 < inclination < np.pi):
        raise ValueError("inclination must lie strictly between 0 and pi")
    x_o = np.cos(inclination)
    g_tt, g_rr, g_xx, g_pp, g_tp = _oracle_ml_metric_components(r_o, x_o, params, b)
    omega = -g_tp / g_pp
    lapse = np.sqrt((g_tp * g_tp - g_tt * g_pp) / g_pp)
    norm = np.sqrt(1.0 + (alpha * alpha + beta * beta) / (r_o * r_o))
    n_r = -1.0 / norm
    n_ph = -(alpha / r_o) / norm
    n_x = (beta / r_o) / norm
    p_ph = n_ph * np.sqrt(g_pp)
    p_r = n_r * np.sqrt(g_rr)
    p_x = n_x * np.sqrt(g_xx)
    p_t = lapse - omega * p_ph
    return np.array([0.0, r_o, x_o, 0.0, p_t, p_r, p_x, p_ph])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Candidate and oracle start from separate equivalent inputs."""
    exception_setup = 'def _exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n'
    return [
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)',
            "call": 'screen_to_initial_conditions(*_review_deepcopy((4.0, 0.0, par, 0.01, 50.0, np.deg2rad(80.0))))',
            "gold_call": '_oracle_screen_to_initial_conditions(*_review_deepcopy((4.0, 0.0, par, 0.01, 50.0, np.deg2rad(80.0))))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)',
            "call": 'screen_to_initial_conditions(*_review_deepcopy((0.0, 0.0, par, 0.01, 50.0, np.deg2rad(80.0))))',
            "gold_call": '_oracle_screen_to_initial_conditions(*_review_deepcopy((0.0, 0.0, par, 0.01, 50.0, np.deg2rad(80.0))))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)',
            "call": 'screen_to_initial_conditions(*_review_deepcopy((-6.0, 3.0, par, 0.01, 30.0, np.deg2rad(60.0))))',
            "gold_call": '_oracle_screen_to_initial_conditions(*_review_deepcopy((-6.0, 3.0, par, 0.01, 30.0, np.deg2rad(60.0))))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\n\n\n' + exception_setup,
            "call": '_exception_code(screen_to_initial_conditions, *_review_deepcopy((4.0, 0.0, par, 0.01, 0.0, np.deg2rad(80.0))))',
            "gold_call": '_exception_code(_oracle_screen_to_initial_conditions, *_review_deepcopy((4.0, 0.0, par, 0.01, 0.0, np.deg2rad(80.0))))',
        },
    ]
