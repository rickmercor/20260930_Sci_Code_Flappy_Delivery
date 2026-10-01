"""
Evaluate the affine-parameter derivative of the combined geodesic and radiative-transfer

state at a point along a ray.

The Hamilton-Jacobi equation does not separate in this spacetime, so each ray

is obtained by integrating Hamilton's equations for the Hamiltonian of step 03,

with tau its affine parameter, and the radiative transfer is carried along the

same ray.  In Lorentz-invariant form the transfer equation reads



    dI/dtau = J - A I,   with   I = I_nu / nu^3,   J = j_nu / nu^2,   A = nu alpha_nu,



where nu is the photon frequency measured by the emitting matter (minus the

physical photon momentum contracted with the matter four-velocity of step 04),

and j_nu and alpha_nu are the emission and absorption coefficients of step 05

in the matter frame.



Conventions fixed for this step:

* The observed frequency is normalised to unity, so the observed specific

  intensity of a ray equals its invariant intensity.

* The state carries the backward-traced ray of step 06: its momentum is the

  arriving photon's momentum with every component reversed.

* optical_depth is the optical depth accumulated along the ray from the

  observer to the current point, and intensity is the observed specific

  intensity contributed by the part of the ray between the observer and the

  current point; both are zero at the observer.

* t and phi are cyclic, so dp_t/dtau and dp_phi/dtau are exactly zero.

* Metric derivatives must be accurate to 1e-10 relative at every radius down

  to 1.0001 r_+.

Returns
-------
np.ndarray of shape (10,), d/dtau of [t, r, x, phi, p_t, p_r, p_x, p_phi, optical_depth, intensity]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def transfer_rhs(tau: float, state: np.ndarray, params: np.ndarray, b: float,
                 r_in: float, emis_params: dict, absorb_params: dict,
                 flow_params: dict) -> np.ndarray:
    '''Right-hand side of the coupled geodesic and radiative-transfer system.

    Parameters
    ----------
    tau : float
        Affine parameter; the system is autonomous, so this is unused.
    state : np.ndarray
        Array of shape (10,):
        [t, r, x, phi, p_t, p_r, p_x, p_phi, optical_depth, intensity],
        where optical_depth is the optical depth accumulated along the ray
        from the observer to the current point and intensity is the observed
        specific intensity contributed by the ray between the observer and
        the current point.
    params : np.ndarray
        Array of shape (11,) as returned by ml_parameters_and_horizon.
    b : float
        Spindle deformation parameter B.
    r_in : float
        Inner boundary of the accretion model.
    emis_params : dict
        Emission parameters; see step 05.
    absorb_params : dict
        Absorption parameters with keys alpha0, alpha1, alpha2, alpha3.
    flow_params : dict
        Flow parameters with keys v_max, p3, psi, lam.

    Returns
    -------
    result : np.ndarray
        Array of shape (10,), the affine-parameter derivative of the state.

    Raises
    ------
    ValueError
        If state does not have shape (10,), its radial coordinate is not
        positive, its polar coordinate has |x| >= 1, r_in is not positive, or
        flow_params lacks a required key.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _inverse_metric_derivatives(r, x, params, b, cs_step=1e-20):
    """Complex-step derivatives of the contravariant metric w.r.t. r and x.

    The increment is a default argument rather than a module-level constant:
    the grading namespace keeps imports and function definitions only.
    """
    zero_p = np.zeros(4)
    d_r = [c.imag / cs_step for c in _oracle_inverse_metric_and_hamiltonian(
        complex(r, cs_step), complex(x, 0.0), params, b, zero_p)[:5]]
    d_x = [c.imag / cs_step for c in _oracle_inverse_metric_and_hamiltonian(
        complex(r, 0.0), complex(x, cs_step), params, b, zero_p)[:5]]
    return d_r, d_x


def _oracle_transfer_rhs(tau: float, state: np.ndarray, params: np.ndarray, b: float, r_in: float, emis_params: dict, absorb_params: dict, flow_params: dict) -> np.ndarray:
    """Reference implementation."""
    state = np.asarray(state, dtype=float)
    if state.shape != (10,):
        raise ValueError("state must have shape (10,)")
    if state[1] <= 0.0:
        raise ValueError("the radial coordinate must be positive")
    if abs(state[2]) >= 1.0:
        raise ValueError("the polar coordinate must satisfy |x| < 1")
    if r_in <= 0.0:
        raise ValueError("r_in must be positive")
    for key in ("v_max", "p3", "psi", "lam"):
        if key not in flow_params:
            raise ValueError(f"flow_params is missing the key {key!r}")
    r, x, phi = state[1], state[2], state[3]
    p_cov = state[4:8]
    p_t, p_r, p_x, p_ph = p_cov[0], p_cov[1], p_cov[2], p_cov[3]
    rho = state[8]
    inv = _oracle_inverse_metric_and_hamiltonian(r, x, params, b, p_cov)
    i_tt, i_rr, i_xx, i_pp, i_tp = inv[0], inv[1], inv[2], inv[3], inv[4]
    d_r, d_x = _inverse_metric_derivatives(r, x, params, b)
    quad = (p_t * p_t, p_r * p_r, p_x * p_x, p_ph * p_ph, 2.0 * p_t * p_ph)
    dp_r = -0.5 * sum(d_r[i] * quad[i] for i in range(5))
    dp_x = -0.5 * sum(d_x[i] * quad[i] for i in range(5))
    u = _oracle_zamo_flow_four_velocity(r, x, params, b, r_in, flow_params["v_max"],
                                        flow_params["p3"], flow_params["psi"],
                                        flow_params["lam"])
    redshift = 1.0 / (p_t * u[0] + p_r * u[1] + p_x * u[2] + p_ph * u[3])
    j_nu, a_nu = _oracle_emission_and_absorption(r, x, phi, r_in, emis_params, absorb_params)
    return np.array([i_tt * p_t + i_tp * p_ph,
                     i_rr * p_r,
                     i_xx * p_x,
                     i_pp * p_ph + i_tp * p_t,
                     0.0, dp_r, dp_x, 0.0,
                     a_nu / redshift,
                     redshift * redshift * j_nu * np.exp(-rho)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Candidate and oracle start from separate equivalent inputs."""
    exception_setup = 'def _exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n'
    return [
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\nEM = dict(j0=1.0, j1=1.0, j2=1.0, j3=1.0, p1=-1.5, p2=-0.5, sigma_dtheta=0.1, beta=0.1, r_p=1.5, w_p=3.0, j_p=1.0, r_b=6.0, sigma_br=0.5, sigma_btheta=0.15, r_s=8.0, theta_s=np.pi / 2, phi_s=3 * np.pi / 2, sigma_sr=2.0, sigma_stheta=np.pi / 36, sigma_sphi=np.pi / 18)\nAB = dict(alpha0=0.02, alpha1=1.0, alpha2=0.5, alpha3=0.5)\nFL = dict(v_max=0.9, p3=0.5, psi=0.9, lam=10.0)\ns9 = _oracle_screen_to_initial_conditions(4.0, 0.0, par, 0.01, 50.0, np.deg2rad(80.0))\nst = np.concatenate([s9, [0.05, 0.2]])\nst[1] = 8.0\nst[2] = 0.05\nst[3] = 3 * np.pi / 2',
            "call": 'transfer_rhs(*_review_deepcopy((0.0, st, par, 0.01, par[10], EM, AB, FL)))',
            "gold_call": '_oracle_transfer_rhs(*_review_deepcopy((0.0, st, par, 0.01, par[10], EM, AB, FL)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\nEM = dict(j0=1.0, j1=1.0, j2=1.0, j3=1.0, p1=-1.5, p2=-0.5, sigma_dtheta=0.1, beta=0.1, r_p=1.5, w_p=3.0, j_p=1.0, r_b=6.0, sigma_br=0.5, sigma_btheta=0.15, r_s=8.0, theta_s=np.pi / 2, phi_s=3 * np.pi / 2, sigma_sr=2.0, sigma_stheta=np.pi / 36, sigma_sphi=np.pi / 18)\nAB0 = dict(alpha0=0.0, alpha1=1.0, alpha2=0.5, alpha3=0.5)\nFL = dict(v_max=0.9, p3=0.5, psi=0.9, lam=10.0)\ns9b = _oracle_screen_to_initial_conditions(-6.0, 0.0, par, 0.01, 50.0, np.deg2rad(80.0))\nstb = np.concatenate([s9b, [0.0, 0.0]])\nstb[1] = 6.0\nstb[2] = 0.0\nstb[3] = 1.0',
            "call": 'transfer_rhs(*_review_deepcopy((0.0, stb, par, 0.01, par[10], EM, AB0, FL)))',
            "gold_call": '_oracle_transfer_rhs(*_review_deepcopy((0.0, stb, par, 0.01, par[10], EM, AB0, FL)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\nEM = dict(j0=1.0, j1=1.0, j2=1.0, j3=1.0, p1=-1.5, p2=-0.5, sigma_dtheta=0.1, beta=0.1, r_p=1.5, w_p=3.0, j_p=1.0, r_b=6.0, sigma_br=0.5, sigma_btheta=0.15, r_s=8.0, theta_s=np.pi / 2, phi_s=3 * np.pi / 2, sigma_sr=2.0, sigma_stheta=np.pi / 36, sigma_sphi=np.pi / 18)\nAB = dict(alpha0=0.02, alpha1=1.0, alpha2=0.5, alpha3=0.5)\nFL = dict(v_max=0.9, p3=0.5, psi=0.9, lam=10.0)\ns9c = _oracle_screen_to_initial_conditions(10.0, 0.0, par, 0.01, 50.0, np.deg2rad(80.0))\nstc = np.concatenate([s9c, [0.3, 1.0]])',
            "call": 'transfer_rhs(*_review_deepcopy((0.0, stc, par, 0.01, par[10], EM, AB, FL)))',
            "gold_call": '_oracle_transfer_rhs(*_review_deepcopy((0.0, stc, par, 0.01, par[10], EM, AB, FL)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\nEM = dict(j0=1.0, j1=1.0, j2=1.0, j3=1.0, p1=-1.5, p2=-0.5, sigma_dtheta=0.1, beta=0.1, r_p=1.5, w_p=3.0, j_p=1.0, r_b=6.0, sigma_br=0.5, sigma_btheta=0.15, r_s=8.0, theta_s=np.pi / 2, phi_s=3 * np.pi / 2, sigma_sr=2.0, sigma_stheta=np.pi / 36, sigma_sphi=np.pi / 18)\nAB = dict(alpha0=0.02, alpha1=1.0, alpha2=0.5, alpha3=0.5)\nFL = dict(v_max=0.9, p3=0.5, psi=0.9, lam=10.0)\ns9h = _oracle_screen_to_initial_conditions(0.0, 0.0, par, 0.01, 50.0, np.deg2rad(80.0))\nsth = np.concatenate([s9h, [0.05, 0.2]])\nsth[1] = 1.01 * par[10]\nsth[2] = 0.1\nsth[3] = 1.0\ngih = _oracle_inverse_metric_and_hamiltonian(sth[1], sth[2], par, 0.01, np.zeros(4))\ncch = gih[0] * sth[4] ** 2 + gih[2] * sth[6] ** 2 + gih[3] * sth[7] ** 2 + 2.0 * gih[4] * sth[4] * sth[7]\nsth[5] = -np.sqrt(-cch / gih[1])',
            "call": 'transfer_rhs(*_review_deepcopy((0.0, sth, par, 0.01, par[10], EM, AB, FL)))',
            "gold_call": '_oracle_transfer_rhs(*_review_deepcopy((0.0, sth, par, 0.01, par[10], EM, AB, FL)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\nEM = dict(j0=1.0, j1=1.0, j2=1.0, j3=1.0, p1=-1.5, p2=-0.5, sigma_dtheta=0.1, beta=0.1, r_p=1.5, w_p=3.0, j_p=1.0, r_b=6.0, sigma_br=0.5, sigma_btheta=0.15, r_s=8.0, theta_s=np.pi / 2, phi_s=3 * np.pi / 2, sigma_sr=2.0, sigma_stheta=np.pi / 36, sigma_sphi=np.pi / 18)\nAB = dict(alpha0=0.02, alpha1=1.0, alpha2=0.5, alpha3=0.5)\nFL = dict(v_max=0.9, p3=0.5, psi=0.9, lam=10.0)\ns9 = _oracle_screen_to_initial_conditions(4.0, 0.0, par, 0.01, 50.0, np.deg2rad(80.0))\nstx = np.concatenate([s9, [0.0, 0.0]])\nstx[1] = 8.0\nstx[2] = 1.5\n\n\n' + exception_setup,
            "call": '_exception_code(transfer_rhs, *_review_deepcopy((0.0, stx, par, 0.01, par[10], EM, AB, FL)))',
            "gold_call": '_exception_code(_oracle_transfer_rhs, *_review_deepcopy((0.0, stx, par, 0.01, par[10], EM, AB, FL)))',
        },
    ]
