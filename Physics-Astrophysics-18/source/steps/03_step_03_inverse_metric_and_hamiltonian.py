"""
Invert the metric's t-phi block to obtain the contravariant components, and evaluate the

photon Hamiltonian for a given covariant momentum.

Photon propagation is governed by the Hamiltonian H = (1/2) g^{mu nu} p_mu

p_nu, which requires the contravariant metric.  No closed form for g^{mu nu}

is published for this spacetime, so it has to be obtained by inverting the

covariant components: r and x decouple, while t and phi mix through g_tph.

H = 0 is the null condition.



Conventions fixed for this step:

* The returned Hamiltonian carries the factor 1/2.

* The implementation must remain valid for complex r and x.

Returns
-------
np.ndarray of shape (6,), [g^tt, g^rr, g^xx, g^phph, g^tph, H]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def inverse_metric_and_hamiltonian(r: float, x: float, params: np.ndarray,
                                   b: float, p_cov: np.ndarray) -> np.ndarray:
    '''Contravariant metric components and the photon Hamiltonian.

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
    p_cov : np.ndarray
        Covariant momentum of shape (4,), ordered [p_t, p_r, p_x, p_phi].

    Returns
    -------
    result : np.ndarray
        Array of shape (6,): [g^tt, g^rr, g^xx, g^phph, g^tph, H].

    Raises
    ------
    ValueError
        If r is not positive, |x| is not less than 1, or p_cov does not have
        shape (4,).
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_inverse_metric_and_hamiltonian(r: float, x: float, params: np.ndarray, b: float, p_cov: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    if np.real(r) <= 0.0:
        raise ValueError("r must be positive")
    if abs(np.real(x)) >= 1.0:
        raise ValueError("|x| must be less than 1")
    p_cov = np.asarray(p_cov)
    if p_cov.shape != (4,):
        raise ValueError("p_cov must have shape (4,)")
    g_tt, g_rr, g_xx, g_pp, g_tp = _oracle_ml_metric_components(r, x, params, b)
    det = g_tt * g_pp - g_tp * g_tp
    i_tt = g_pp / det
    i_rr = 1.0 / g_rr
    i_xx = 1.0 / g_xx
    i_pp = g_tt / det
    i_tp = -g_tp / det
    p_t, p_r, p_x, p_ph = p_cov[0], p_cov[1], p_cov[2], p_cov[3]
    ham = 0.5 * (i_tt * p_t * p_t + i_rr * p_r * p_r + i_xx * p_x * p_x
                 + i_pp * p_ph * p_ph + 2.0 * i_tp * p_t * p_ph)
    return np.array([i_tt, i_rr, i_xx, i_pp, i_tp, ham])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Candidate and oracle start from separate equivalent inputs."""
    exception_setup = 'def _exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n'
    return [
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\npg = np.array([-1.0, 0.4, 0.1, 2.0])',
            "call": 'inverse_metric_and_hamiltonian(*_review_deepcopy((8.0, np.cos(np.deg2rad(80.0)), par, 0.01, pg)))',
            "gold_call": '_oracle_inverse_metric_and_hamiltonian(*_review_deepcopy((8.0, np.cos(np.deg2rad(80.0)), par, 0.01, pg)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\nxo = np.cos(np.deg2rad(80.0))\ng2 = _oracle_ml_metric_components(50.0, xo, par, 0.01)\ndet2 = g2[0] * g2[3] - g2[4] ** 2\ngi = np.array([g2[3] / det2, 1.0 / g2[1], 1.0 / g2[2], g2[0] / det2, -g2[4] / det2])\npr_, px_, pph_ = (0.3, 0.05, 2.0)\ncc = gi[1] * pr_ ** 2 + gi[2] * px_ ** 2 + gi[3] * pph_ ** 2\nbq = 2.0 * gi[4] * pph_\npt_ = (-bq - np.sqrt(bq * bq - 4.0 * gi[0] * cc)) / (2.0 * gi[0])\npn = np.array([pt_, pr_, px_, pph_])',
            "call": 'inverse_metric_and_hamiltonian(*_review_deepcopy((50.0, xo, par, 0.01, pn)))',
            "gold_call": '_oracle_inverse_metric_and_hamiltonian(*_review_deepcopy((50.0, xo, par, 0.01, pn)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar0 = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.0)\npr = np.array([-1.0, 1.0, 0.0, 0.0])',
            "call": 'inverse_metric_and_hamiltonian(*_review_deepcopy((20.0, 0.0, par0, 0.0, pr)))',
            "gold_call": '_oracle_inverse_metric_and_hamiltonian(*_review_deepcopy((20.0, 0.0, par0, 0.0, pr)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\n\n\n' + exception_setup,
            "call": '_exception_code(inverse_metric_and_hamiltonian, *_review_deepcopy((8.0, 1.5, par, 0.01, np.zeros(4))))',
            "gold_call": '_exception_code(_oracle_inverse_metric_and_hamiltonian, *_review_deepcopy((8.0, 1.5, par, 0.01, np.zeros(4))))',
        },
    ]
