"""
Compose the complete transmission-matched chiral continuation pipeline.

The calculation constructs one shared spectral representation, solves the two distinct arbitrary-center circular-channel hierarchies with chiral-to-vacuum interface matching, reconstructs the physical electric-field series, forms its volume-intensity series, and returns the specified rational curvature.

Returns
-------
native float equal to the second deformation derivative of the Pade-continued normalized volume-averaged electric intensity
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def chiral_field_intensity_curvature(d: float, h: float, t: float, sharpness: float, wavelength: float, theta: float, chi_bar: float, chi_amplitude: float, lateral_scale: float, delta_center: float, delta_target: float, nx: int, nz: int, max_order: int, pade_numerator: int, pade_denominator: int) -> float:
    '''Run the complete arbitrary-center chiral-field intensity calculation.

    Parameters
    ----------
    d, h : float
        Positive lateral period and slab half-height.
    t : float
        Positive envelope half-thickness below h.
    sharpness : float
        Positive tanh-profile sharpness.
    wavelength : float
        Positive vacuum wavelength.
    theta : float
        Incidence angle in radians with abs(theta) < pi/2.
    chi_bar, chi_amplitude : float
        Finite background chirality and envelope amplitude.
    lateral_scale : float
        Finite multiplier of the four fixed lateral harmonics.
    delta_center, delta_target : float
        Finite global expansion center and requested physical deformation.
    nx : int
        Even Fourier-node count, at least 4.
    nz : int
        Chebyshev degree, at least 2.
    max_order : int
        Nonnegative maximum deformation order.
    pade_numerator, pade_denominator : int
        Nonnegative Pade degrees satisfying their sum no greater than
        max_order.

    Returns
    -------
    curvature : float
        Native Python float containing the analytic second derivative, at
        delta_target - delta_center, of the normalized Pade approximant built
        from the volume-averaged electric-intensity coefficient series.  The
        complete two-channel recursive pipeline is used.  No random state is
        used, and no intermediate rounding occurs.

    Raises
    ------
    ValueError
        Under the invalid conditions documented by the component functions,
        if the Pade degrees exceed max_order, or if the physical target
        profile violates max(abs(k0*chi)) < 1.
    '''
    return curvature

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_chiral_field_intensity_curvature(d: float, h: float, t: float, sharpness: float, wavelength: float, theta: float, chi_bar: float, chi_amplitude: float, lateral_scale: float, delta_center: float, delta_target: float, nx: int, nz: int, max_order: int, pade_numerator: int, pade_denominator: int) -> float:
    """Reference implementation composed exclusively from earlier oracles."""
    if not isinstance(max_order, (int, np.integer)) or isinstance(max_order, (bool, np.bool_)) or max_order < 0:
        raise ValueError("max_order must be a nonnegative integer")
    for value, name in ((pade_numerator, "pade_numerator"), (pade_denominator, "pade_denominator")):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if pade_numerator + pade_denominator > max_order:
        raise ValueError("Pade degrees must sum to at most max_order")
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("wavelength must be a positive finite scalar")
    if not np.isfinite(delta_target):
        raise ValueError("delta_target must be finite")

    k0 = 2.0 * np.pi / float(wavelength)
    alpha = k0 * np.sin(float(theta))
    operators = _oracle_spectral_operators_and_weights(nx, nz, d, h, alpha)
    radiation = _oracle_radiation_and_bohren_data(nx, d, h, wavelength, theta)
    left_profiles = _oracle_centered_channel_profiles(
        1, nx, nz, d, h, t, sharpness, wavelength, chi_bar,
        chi_amplitude, lateral_scale, delta_center
    )
    right_profiles = _oracle_centered_channel_profiles(
        -1, nx, nz, d, h, t, sharpness, wavelength, chi_bar,
        chi_amplitude, lateral_scale, delta_center
    )
    envelope = left_profiles[0]
    target_chi = float(chi_bar) - float(delta_target) * envelope
    if np.max(np.abs(k0 * target_chi)) >= 1.0:
        raise ValueError("the target chirality violates abs(k0*chi) < 1")

    left_matrix = _oracle_assemble_transmission_operator(
        left_profiles, operators, radiation, nx, nz
    )
    right_matrix = _oracle_assemble_transmission_operator(
        right_profiles, operators, radiation, nx, nz
    )
    left_series = _oracle_solve_transmission_channel_series(
        1, left_matrix, left_profiles, operators, radiation, max_order
    )
    right_series = _oracle_solve_transmission_channel_series(
        -1, right_matrix, right_profiles, operators, radiation, max_order
    )
    electric = _oracle_reconstruct_electric_series(
        left_series, right_series, left_profiles, right_profiles, operators, k0
    )
    coefficients = _oracle_electric_intensity_coefficients(electric, operators)
    return _oracle_pade_second_derivative(
        coefficients, pade_numerator, pade_denominator,
        float(delta_target) - float(delta_center)
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
args = (0.8,0.95,0.5,7.0,0.72,0.28,0.004,0.022,0.8,0.15,-0.55,8,8,8,4,4)
""",
            "call": "chiral_field_intensity_curvature(*args)",
            "gold_call": "_oracle_chiral_field_intensity_curvature(*args)",
            "tol": 2e-7,
        },
        {
            "setup": """import numpy as np
args = (0.82,0.9,0.43,6.0,0.76,0.0,0.003,0.018,0.0,-0.1,-0.6,6,6,6,3,3)
""",
            "call": "chiral_field_intensity_curvature(*args)",
            "gold_call": "_oracle_chiral_field_intensity_curvature(*args)",
            "tol": 2e-7,
        },
        {
            "setup": """import numpy as np
args = (0.78,0.88,0.4,5.5,0.71,-0.22,0.002,0.01,1.0,0.2,-0.3,8,7,8,4,4)
""",
            "call": "chiral_field_intensity_curvature(*args)",
            "gold_call": "_oracle_chiral_field_intensity_curvature(*args)",
            "tol": 2e-7,
        },
        {
            "setup": """import numpy as np
args = (0.8,0.95,0.5,8.0,0.7,np.pi/6,0.006,0.04,1.0,0.25,-1.0,18,22,18,10,9)
def catch_value_error(fn):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "catch_value_error(chiral_field_intensity_curvature)",
            "gold_call": "catch_value_error(_oracle_chiral_field_intensity_curvature)",
        },
        {
            "setup": """import numpy as np
args = (0.8,0.95,0.5,8.0,0.0,np.pi/6,0.006,0.04,1.0,0.25,-1.0,18,22,18,9,9)
def catch_value_error(fn):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "catch_value_error(chiral_field_intensity_curvature)",
            "gold_call": "catch_value_error(_oracle_chiral_field_intensity_curvature)",
        },
    ]
