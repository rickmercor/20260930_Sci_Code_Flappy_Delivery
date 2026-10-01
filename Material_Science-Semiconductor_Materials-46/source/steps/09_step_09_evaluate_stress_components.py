"""
Combine the complementary amplitudes of one layer with the thermally generated contributions at the same point to return the four axisymmetric stress components of one axial mode.

Each stress component is a different differential combination of the two potentials, and the axial dependence sorts them into two families. The four amplitudes multiply I0(ηr), K0(ηr), ηr·I1(ηr) and ηr·K1(ηr) in the biharmonic function L = [E·I0(ηr) + F·K0(ηr) + P·ηr·I1(ηr) + Q·ηr·K1(ηr)]·sin(ηz), whose complementary stresses are σ_rr = 2G·∂/∂z(ν∇²L − ∂²L/∂r²), σ_θθ = 2G·∂/∂z(ν∇²L − (1/r)·∂L/∂r), σ_zz = 2G·∂/∂z((2 − ν)∇²L − ∂²L/∂z²) and σ_rz = 2G·∂/∂r((1 − ν)∇²L − ∂²L/∂z²); the radial, hoop, axial and shear stress entries of the thermally generated terms are added to the matching complementary stress before the cosine (normal stresses) or sine (shear stress) of ηz is applied. The three normal components inherit the cosine of the axial mode, so they are largest on the insulated face and vanish on the face held at ambient, while the shear component inherits the sine and does the opposite. That split is a direct consequence of the assumed field: the representation used here enforces, without being asked to, a frictionless rigid support on the insulated face and a normal-traction-free condition on the pinned face, which is the appropriate idealisation for a via bonded to a thick carrier at one end and terminated at a free surface at the other.




Within a layer the complementary contribution to the radial normal stress involves the second radial derivative of the biharmonic function while the hoop stress involves the first derivative divided by the radius; the two coincide on the axis, as axisymmetry demands, and separate as the radius grows. The axial normal stress instead combines the Laplacian of the biharmonic function with the function itself, which is why it is the component most sensitive to the interplay between the layer stiffness and the axial wavelength, and why it is the component that a plane-strain treatment of a via discards entirely.




Adding the thermally generated contributions completes the field. Their role differs by component: they enter the radial and hoop stresses both through the curvature of the thermoelastic potential and through the direct subtraction of the free thermal dilatation, whereas in the axial stress those two effects partially cancel. Because the thermal contributions carry the material's own expansion coefficient and stiffness while the complementary amplitudes are common to the whole stack through the interface conditions, evaluating the same radius from the two sides of an interface reproduces continuous radial and shear stresses but discontinuous hoop and axial stresses, and the size of those jumps is the quantitative signature of the expansion mismatch that drives via failure.

Returns
-------
np.ndarray of shape (4,), float: the radial normal, hoop, axial normal and shear stress contributed by one axial mode at the requested point, in Pa.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_stress_components(love_coefficients: np.ndarray,
                               thermal_terms: np.ndarray, eta: float,
                               poisson: float, shear_modulus: float,
                               radius: float,
                               axial_position: float) -> np.ndarray:
    """Return the four stress components contributed by one axial mode.

    Parameters
    ----------
    love_coefficients : np.ndarray
        Array of shape (4,) holding the amplitudes of the regular harmonic,
        singular harmonic, regular biharmonic and singular biharmonic shapes
        of the layer.
    thermal_terms : np.ndarray
        Array of shape (6,) holding the thermally generated displacement and
        stress contributions of the layer at this radius, stripped of their
        axial factor.
    eta : float
        Axial eigenvalue of the mode in 1/m (eta > 0).
    poisson : float
        Poisson's ratio of the layer, -1 < poisson < 0.5.
    shear_modulus : float
        Shear modulus of the layer in Pa (shear_modulus > 0).
    radius : float
        Radial coordinate in m (radius >= 0).
    axial_position : float
        Axial coordinate in m (axial_position >= 0).

    Returns
    -------
    components : np.ndarray
        Array of shape (4,) holding the radial normal, hoop, axial normal and
        shear stresses in Pa contributed by this axial mode.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above; invalid input
        must raise ValueError rather than return a sentinel value. The two
        singular amplitudes, the second and fourth entries of
        love_coefficients, must be zero when radius is 0, since the singular
        shapes are unbounded on the axis; a non-zero value there is invalid.
    """
    return components  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_stress_components(love_coefficients: np.ndarray,
                                       thermal_terms: np.ndarray, eta: float,
                                       poisson: float, shear_modulus: float,
                                       radius: float,
                                       axial_position: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np
    from scipy.special import i0, i1, k0, k1

    love_coefficients = np.asarray(love_coefficients, dtype=float).ravel()
    thermal_terms = np.asarray(thermal_terms, dtype=float).ravel()
    if love_coefficients.size != 4:
        raise ValueError("love_coefficients must hold four amplitudes")
    if thermal_terms.size != 6:
        raise ValueError("thermal_terms must hold six entries")
    if not np.all(np.isfinite(love_coefficients)) or not np.all(np.isfinite(thermal_terms)):
        raise ValueError("love_coefficients and thermal_terms must be finite")
    if not (isinstance(eta, (int, float, np.floating, np.integer))
            and not isinstance(eta, bool) and np.isfinite(eta) and float(eta) > 0.0):
        raise ValueError("eta must be a finite number > 0")
    if not (isinstance(poisson, (int, float, np.floating, np.integer))
            and not isinstance(poisson, bool) and np.isfinite(poisson)
            and -1.0 < float(poisson) < 0.5):
        raise ValueError("poisson must be a finite number in (-1, 0.5)")
    if not (isinstance(shear_modulus, (int, float, np.floating, np.integer))
            and not isinstance(shear_modulus, bool) and np.isfinite(shear_modulus)
            and float(shear_modulus) > 0.0):
        raise ValueError("shear_modulus must be a finite number > 0")
    for name, value in (("radius", radius), ("axial_position", axial_position)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool)
                and np.isfinite(value) and float(value) >= 0.0):
            raise ValueError(f"{name} must be a finite number >= 0")

    eta = float(eta)
    poisson = float(poisson)
    shear_modulus = float(shear_modulus)
    radius = float(radius)
    axial_position = float(axial_position)

    if radius == 0.0:
        if love_coefficients[1] != 0.0 or love_coefficients[3] != 0.0:
            raise ValueError("the singular amplitudes must vanish on the axis")
        shape = np.array([1.0, 0.0, 0.0, 0.0])
        slope = np.zeros(4)
        curvature = eta ** 2 * np.array([0.5, 0.0, 1.0, 0.0])
        slope_over_r = curvature.copy()
        laplacian = 2.0 * eta ** 2 * np.array([0.0, 0.0, 1.0, 0.0])
        laplacian_slope = np.zeros(4)
    else:
        x = eta * radius
        bess_i0, bess_i1, bess_k0, bess_k1 = i0(x), i1(x), k0(x), k1(x)
        shape = np.array([bess_i0, bess_k0, x * bess_i1, x * bess_k1])
        slope = eta * np.array([bess_i1, -bess_k1, x * bess_i0, -x * bess_k0])
        curvature = eta ** 2 * np.array([bess_i0 - bess_i1 / x,
                                         bess_k0 + bess_k1 / x,
                                         bess_i0 + x * bess_i1,
                                         -bess_k0 + x * bess_k1])
        slope_over_r = slope / radius
        laplacian = 2.0 * eta ** 2 * np.array([0.0, 0.0, bess_i0, -bess_k0])
        laplacian_slope = 2.0 * eta ** 3 * np.array([0.0, 0.0, bess_i1, bess_k1])

    value_f = float(shape @ love_coefficients)
    value_fp = float(slope @ love_coefficients)
    value_fpp = float(curvature @ love_coefficients)
    value_fpr = float(slope_over_r @ love_coefficients)
    value_g = float(laplacian @ love_coefficients)
    value_gp = float(laplacian_slope @ love_coefficients)

    two_g = 2.0 * shear_modulus
    cosine = np.cos(eta * axial_position)
    sine = np.sin(eta * axial_position)

    radial = (two_g * eta * (poisson * value_g - value_fpp) + thermal_terms[2]) * cosine
    hoop = (two_g * eta * (poisson * value_g - value_fpr) + thermal_terms[3]) * cosine
    axial = (two_g * eta * ((2.0 - poisson) * value_g + eta ** 2 * value_f)
             + thermal_terms[4]) * cosine
    shear = (two_g * ((1.0 - poisson) * value_gp + eta ** 2 * value_fp)
             + thermal_terms[5]) * sine

    return np.array([radial, hoop, axial, shear])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: silicon substrate at the liner interface (normal scenario) ---
        {
            "setup": """import numpy as np
love_coefficients = np.array([-1.3e-19, 4.1e-19, 8.7e-20, -2.2e-19])
thermal_terms = np.array([-1.1e-9, 3.4e-9, -8.2e7, -8.2e7, 0.0, -4.5e6])
eta = 7853.981633974483
poisson = 0.28
shear_modulus = 170.0e9 / 2.56
radius = 16.0e-6
axial_position = 200.0e-6 / 3.0
""",
            "call": "evaluate_stress_components(love_coefficients, thermal_terms, eta, poisson, shear_modulus, radius, axial_position)",
            "gold_call": "_oracle_evaluate_stress_components(love_coefficients, thermal_terms, eta, poisson, shear_modulus, radius, axial_position)",
        },
        # --- Valid: copper core away from the axis at a higher axial mode ---
        {
            "setup": """import numpy as np
love_coefficients = np.array([2.5e-19, 0.0, -6.0e-20, 0.0])
thermal_terms = np.array([-9.0e-10, 2.0e-9, -6.5e7, -6.5e7, -1.0e7, -3.0e6])
eta = 5.0 * np.pi / (2.0 * 200.0e-6)
poisson = 0.35
shear_modulus = 110.0e9 / 2.7
radius = 9.0e-6
axial_position = 20.0e-6
""",
            "call": "evaluate_stress_components(love_coefficients, thermal_terms, eta, poisson, shear_modulus, radius, axial_position)",
            "gold_call": "_oracle_evaluate_stress_components(love_coefficients, thermal_terms, eta, poisson, shear_modulus, radius, axial_position)",
        },
        # --- Boundary: on the axis, where the radial and hoop stresses coincide ---
        {
            "setup": """import numpy as np
love_coefficients = np.array([2.5e-19, 0.0, -6.0e-20, 0.0])
thermal_terms = np.array([0.0, 2.0e-9, -6.5e7, -6.5e7, -1.0e7, 0.0])
eta = 7853.981633974483
poisson = 0.35
shear_modulus = 110.0e9 / 2.7
radius = 0.0
axial_position = 0.0
""",
            "call": "evaluate_stress_components(love_coefficients, thermal_terms, eta, poisson, shear_modulus, radius, axial_position)",
            "gold_call": "_oracle_evaluate_stress_components(love_coefficients, thermal_terms, eta, poisson, shear_modulus, radius, axial_position)",
        },
        # --- Edge: on the pinned face, where the normal components vanish ---
        {
            "setup": """import numpy as np
love_coefficients = np.array([-1.3e-19, 4.1e-19, 8.7e-20, -2.2e-19])
thermal_terms = np.array([-1.1e-9, 3.4e-9, -8.2e7, -8.2e7, 0.0, -4.5e6])
eta = np.pi / (2.0 * 200.0e-6)
poisson = 0.28
shear_modulus = 170.0e9 / 2.56
radius = 20.0e-6
axial_position = 200.0e-6
""",
            "call": "evaluate_stress_components(love_coefficients, thermal_terms, eta, poisson, shear_modulus, radius, axial_position)",
            "gold_call": "_oracle_evaluate_stress_components(love_coefficients, thermal_terms, eta, poisson, shear_modulus, radius, axial_position)",
        },
        # --- Invalid: singular amplitudes requested on the axis ---
        {
            "setup": """import numpy as np
love_coefficients = np.array([2.5e-19, 1.0e-19, -6.0e-20, 0.0])
thermal_terms = np.zeros(6)
def run_model():
    try:
        evaluate_stress_components(love_coefficients, thermal_terms, 7853.98, 0.35, 4.0e10, 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_stress_components(love_coefficients, thermal_terms, 7853.98, 0.35, 4.0e10, 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: wrong number of thermal terms ---
        {
            "setup": """import numpy as np
love_coefficients = np.zeros(4)
thermal_terms = np.zeros(4)
def run_model():
    try:
        evaluate_stress_components(love_coefficients, thermal_terms, 7853.98, 0.35, 4.0e10, 1.0e-6, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_stress_components(love_coefficients, thermal_terms, 7853.98, 0.35, 4.0e10, 1.0e-6, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
