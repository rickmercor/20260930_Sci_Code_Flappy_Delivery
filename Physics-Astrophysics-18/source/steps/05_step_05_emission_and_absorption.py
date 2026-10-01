"""
Evaluate the total emission and absorption coefficients of the multi-component accretion

environment at a point, from the disk, plateau, bump and localized-spot laws under their

independent weights.

The emissivity is a weighted superposition of an attenuated disk background

j_d, a ring-like Gaussian bump j_b representing an outward-propagating shock,

and a localized spot j_s representing a flare, and the absorption coefficient

is built from the same three shape functions under its own weights:



    j_nu     = j0 (j1 j_d + j2 j_b + j3 j_s)

    alpha_nu = alpha0 (alpha1 j_d + alpha2 j_b + alpha3 j_s)



Throughout, theta = arccos(x) is the polar angle measured from the pole.



The disk's radial attenuation is evaluated not at the coordinate radius but

at an effective radius carrying a plateau, a region of locally sustained

radiative efficiency, realised by a hyperbolic-tangent step centred at r_p

with width w_p and strength j_p:



    r_eff = r - j_p w_p [ tanh((r - r_p) / w_p) - tanh((r_in - r_p) / w_p) ]



and the disk component is, in logarithmic form,



    ln j_d = p1 L + p2 L^2 - (theta - pi/2)^2 / (2 (sigma_dtheta e^(beta r))^2),

    L = ln(r_eff / r_in).



The bump is axisymmetric and centred on the equator,



    ln j_b = -(r - r_b)^2 / (2 sigma_br^2) - (theta - pi/2)^2 / (2 sigma_btheta^2).



The spot is a Gaussian in the three spatial coordinates, centred at

(r_s, theta_s, phi_s) with independent widths, whose azimuthal factor is

periodic because the azimuth accumulated along a ray is not wrapped:



    ln j_s = -(r - r_s)^2 / (2 sigma_sr^2) - (theta - theta_s)^2 / (2 sigma_stheta^2)

             - [1 - cos(phi - phi_s)] / (2 sigma_sphi^2)



with sigma_sphi entering unrescaled.



Conventions fixed for this step:

* emis_params is a dict with keys

  j0, j1, j2, j3, p1, p2, sigma_dtheta, beta, r_p, w_p, j_p,

  r_b, sigma_br, sigma_btheta,

  r_s, theta_s, phi_s, sigma_sr, sigma_stheta, sigma_sphi

* absorb_params is a dict with keys alpha0, alpha1, alpha2, alpha3.

Returns
-------
np.ndarray of shape (2,), [j_nu, alpha_nu] as native floats
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def emission_and_absorption(r: float, x: float, phi: float, r_in: float,
                            emis_params: dict, absorb_params: dict) -> np.ndarray:
    '''Total emission and absorption coefficients of the accretion environment.

    Parameters
    ----------
    r : float
        Radial coordinate.
    x : float
        Polar coordinate x = cos(theta).
    phi : float
        Azimuthal coordinate, accumulated along the ray and not wrapped.
    r_in : float
        Inner boundary of the accretion model.
    emis_params : dict
        Emission parameters; see the step background for the key list.
    absorb_params : dict
        Absorption parameters with keys alpha0, alpha1, alpha2, alpha3.

    Returns
    -------
    result : np.ndarray
        Array of shape (2,): [j_nu, alpha_nu], as native floats.

    Raises
    ------
    ValueError
        If |x| exceeds 1, r_in is not positive, any plateau, disk, bump or
        spot width is not positive, a required key is missing from
        emis_params or absorb_params, or the effective radius is not positive.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _plateau_effective_radius(r, r_in, r_p, w_p, j_p):
    """Effective radius of the plateau substitution, anchored at r_in."""
    anchor = np.tanh((r_in - r_p) / w_p)
    return float(r - j_p * w_p * (np.tanh((r - r_p) / w_p) - anchor))


def _disk_emissivity(r, x, r_in, p1, p2, sigma_dtheta, beta, r_p, w_p, j_p):
    """Attenuated, flaring disk background with the radial law at the effective radius."""
    r_eff = _plateau_effective_radius(r, r_in, r_p, w_p, j_p)
    if r_eff <= 0.0:
        raise ValueError("effective radius must be positive")
    theta = np.arccos(x)
    ll = np.log(r_eff / r_in)
    sig = sigma_dtheta * np.exp(beta * r)
    return float(np.exp(p1 * ll + p2 * ll * ll
                        - (theta - np.pi / 2.0) ** 2 / (2.0 * sig * sig)))


def _bump_emissivity(r, x, r_b, sigma_br, sigma_btheta):
    """Ring-like Gaussian bump; axisymmetric, centred on the equator."""
    theta = np.arccos(x)
    return float(np.exp(-(r - r_b) ** 2 / (2.0 * sigma_br ** 2)
                        - (theta - np.pi / 2.0) ** 2 / (2.0 * sigma_btheta ** 2)))


def _spot_emissivity(r, x, phi, r_s, theta_s, phi_s, sigma_sr, sigma_stheta, sigma_sphi):
    """Localized spot with a 2 pi periodic azimuthal factor."""
    theta = np.arccos(x)
    azimuthal = 1.0 - np.cos(phi - phi_s)
    return float(np.exp(-(r - r_s) ** 2 / (2.0 * sigma_sr ** 2)
                        - (theta - theta_s) ** 2 / (2.0 * sigma_stheta ** 2)
                        - azimuthal / (2.0 * sigma_sphi ** 2)))


def _oracle_emission_and_absorption(r: float, x: float, phi: float, r_in: float, emis_params: dict, absorb_params: dict) -> np.ndarray:
    """Reference implementation."""
    if abs(x) > 1.0:
        raise ValueError("|x| must not exceed 1")
    if r_in <= 0.0:
        raise ValueError("r_in must be positive")
    e, al = emis_params, absorb_params
    for key in ("j0", "j1", "j2", "j3", "p1", "p2", "sigma_dtheta", "beta", "r_p", "w_p",
                "j_p", "r_b", "sigma_br", "sigma_btheta", "r_s", "theta_s", "phi_s",
                "sigma_sr", "sigma_stheta", "sigma_sphi"):
        if key not in e:
            raise ValueError(f"emis_params is missing the key {key!r}")
    for key in ("alpha0", "alpha1", "alpha2", "alpha3"):
        if key not in al:
            raise ValueError(f"absorb_params is missing the key {key!r}")
    for key in ("w_p", "sigma_dtheta", "sigma_br", "sigma_btheta",
                "sigma_sr", "sigma_stheta", "sigma_sphi"):
        if e[key] <= 0.0:
            raise ValueError(f"{key} must be positive")
    j_d = _disk_emissivity(r, x, r_in, e["p1"], e["p2"], e["sigma_dtheta"],
                           e["beta"], e["r_p"], e["w_p"], e["j_p"])
    j_b = _bump_emissivity(r, x, e["r_b"], e["sigma_br"], e["sigma_btheta"])
    j_s = _spot_emissivity(r, x, phi, e["r_s"], e["theta_s"], e["phi_s"],
                           e["sigma_sr"], e["sigma_stheta"], e["sigma_sphi"])
    j_nu = e["j0"] * (e["j1"] * j_d + e["j2"] * j_b + e["j3"] * j_s)
    a_nu = al["alpha0"] * (al["alpha1"] * j_d + al["alpha2"] * j_b + al["alpha3"] * j_s)
    return np.array([float(j_nu), float(a_nu)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Candidate and oracle start from separate equivalent inputs."""
    exception_setup = 'def _exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n'
    return [
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\nEM = dict(j0=1.0, j1=1.0, j2=1.0, j3=1.0, p1=-1.5, p2=-0.5, sigma_dtheta=0.1, beta=0.1, r_p=1.5, w_p=3.0, j_p=1.0, r_b=6.0, sigma_br=0.5, sigma_btheta=0.15, r_s=8.0, theta_s=np.pi / 2, phi_s=3 * np.pi / 2, sigma_sr=2.0, sigma_stheta=np.pi / 36, sigma_sphi=np.pi / 18)\nAB = dict(alpha0=0.02, alpha1=1.0, alpha2=0.5, alpha3=0.5)',
            "call": 'emission_and_absorption(*_review_deepcopy((8.0, 0.0, 3 * np.pi / 2, par[10], EM, AB)))',
            "gold_call": '_oracle_emission_and_absorption(*_review_deepcopy((8.0, 0.0, 3 * np.pi / 2, par[10], EM, AB)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\nEM = dict(j0=1.0, j1=1.0, j2=1.0, j3=1.0, p1=-1.5, p2=-0.5, sigma_dtheta=0.1, beta=0.1, r_p=1.5, w_p=3.0, j_p=1.0, r_b=6.0, sigma_br=0.5, sigma_btheta=0.15, r_s=8.0, theta_s=np.pi / 2, phi_s=3 * np.pi / 2, sigma_sr=2.0, sigma_stheta=np.pi / 36, sigma_sphi=np.pi / 18)\nAB = dict(alpha0=0.02, alpha1=1.0, alpha2=0.5, alpha3=0.5)',
            "call": 'emission_and_absorption(*_review_deepcopy((par[10], 0.0, 0.0, par[10], EM, AB)))',
            "gold_call": '_oracle_emission_and_absorption(*_review_deepcopy((par[10], 0.0, 0.0, par[10], EM, AB)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\nEM = dict(j0=1.0, j1=1.0, j2=1.0, j3=1.0, p1=-1.5, p2=-0.5, sigma_dtheta=0.1, beta=0.1, r_p=1.5, w_p=3.0, j_p=1.0, r_b=6.0, sigma_br=0.5, sigma_btheta=0.15, r_s=8.0, theta_s=np.pi / 2, phi_s=3 * np.pi / 2, sigma_sr=2.0, sigma_stheta=np.pi / 36, sigma_sphi=np.pi / 18)\nAB = dict(alpha0=0.02, alpha1=1.0, alpha2=0.5, alpha3=0.5)',
            "call": 'emission_and_absorption(*_review_deepcopy((8.0, 0.0, 3 * np.pi / 2 + 0.05 - 2 * np.pi, par[10], EM, AB)))',
            "gold_call": '_oracle_emission_and_absorption(*_review_deepcopy((8.0, 0.0, 3 * np.pi / 2 + 0.05 - 2 * np.pi, par[10], EM, AB)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\nEM = dict(j0=1.0, j1=1.0, j2=1.0, j3=1.0, p1=-1.5, p2=-0.5, sigma_dtheta=0.1, beta=0.1, r_p=1.5, w_p=3.0, j_p=1.0, r_b=6.0, sigma_br=0.5, sigma_btheta=0.15, r_s=8.0, theta_s=np.pi / 2, phi_s=3 * np.pi / 2, sigma_sr=2.0, sigma_stheta=np.pi / 36, sigma_sphi=np.pi / 18)\nAB0 = dict(alpha0=0.0, alpha1=1.0, alpha2=0.5, alpha3=0.5)',
            "call": 'emission_and_absorption(*_review_deepcopy((6.0, 0.05, 1.0, par[10], EM, AB0)))',
            "gold_call": '_oracle_emission_and_absorption(*_review_deepcopy((6.0, 0.05, 1.0, par[10], EM, AB0)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\nEMb = dict(j0=1.0, j1=1.0, j2=0.0, j3=1.0, p1=-1.5, p2=-0.5, sigma_dtheta=0.1, beta=0.1, r_p=1.5, w_p=3.0, j_p=1.0, r_b=6.0, sigma_br=0.5, sigma_btheta=0.15, r_s=8.0, theta_s=np.pi / 2, phi_s=3 * np.pi / 2, sigma_sr=2.0, sigma_stheta=np.pi / 36, sigma_sphi=np.pi / 18)\nAB = dict(alpha0=0.02, alpha1=1.0, alpha2=0.5, alpha3=0.5)',
            "call": 'emission_and_absorption(*_review_deepcopy((6.0, 0.0, 3 * np.pi / 2 - 2 * np.pi, par[10], EMb, AB)))',
            "gold_call": '_oracle_emission_and_absorption(*_review_deepcopy((6.0, 0.0, 3 * np.pi / 2 - 2 * np.pi, par[10], EMb, AB)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\nEM = dict(j0=1.0, j1=1.0, j2=1.0, j3=1.0, p1=-1.5, p2=-0.5, sigma_dtheta=0.1, beta=0.1, r_p=1.5, w_p=3.0, j_p=1.0, r_b=6.0, sigma_br=0.5, sigma_btheta=0.15, r_s=8.0, theta_s=np.pi / 2, phi_s=3 * np.pi / 2, sigma_sr=2.0, sigma_stheta=np.pi / 36, sigma_sphi=np.pi / 18)\nAB = dict(alpha0=0.02, alpha1=1.0, alpha2=0.5, alpha3=0.5)\n\n\n' + exception_setup,
            "call": '_exception_code(emission_and_absorption, *_review_deepcopy((8.0, 1.5, 0.0, par[10], EM, AB)))',
            "gold_call": '_exception_code(_oracle_emission_and_absorption, *_review_deepcopy((8.0, 1.5, 0.0, par[10], EM, AB)))',
        },
    ]
