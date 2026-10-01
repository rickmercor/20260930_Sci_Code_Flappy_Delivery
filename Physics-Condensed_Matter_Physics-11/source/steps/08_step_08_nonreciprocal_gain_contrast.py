"""
This step runs the whole chain twice and returns the asymmetry between the two runs. Step one builds the Fourier coefficients of the travelling modulus and the harmonic coupling operator they induce. Step two turns those into the quadratic matrix polynomial in the wavenumber at the prescribed frequency. Step three linearises that polynomial and solves it for every mode the segment supports, confirming along the way that the supersonic regime leaves the spectrum real. Step four sorts those modes into forward and backward families by the sign of their group velocity and labels them by basic-mode order. Step five imposes displacement and stress continuity at the two interfaces, retaining the scattering orders the truncation can actually balance. Step six solves that system for the transmission and reflection magnitudes. Step seven converts them into the power the segment returns, referred to the power it received.

The asymmetry comes from the direction of the modulation, not from the geometry, which is symmetric. Reversing the sign of the modulation wavenumber reverses the direction in which the stiffness pattern travels and therefore exchanges the case of a wave running with the modulation for the case of a wave running against it, while leaving the segment, its length and the incident frequency untouched. Running the chain once with each sign and taking the ratio of the two total gains isolates that dependence. Its value is one for a reciprocal scatterer and departs from one only because the modulation has a direction, which is the whole content of the phenomenon.

Only the modulation wavenumber is reversed; the modulation frequency is left alone. The harmonic frequencies omega_n = omega_0 + n omega_m are therefore the same in both runs and the scattering orders are not relabelled. What changes is which orders the segment feeds, because the phase matching that couples a forward branch to a backward one depends on the direction the modulus travels. Posing the graded quantity as a total over all retained orders rather than as a single order keeps it independent of which individual order happens to dominate in each direction.

Returns
-------
dict holding the native floats positive_gain, negative_gain and contrast, the last being the ratio of the first to the second, together with the native floats positive_transmission_zero and positive_reflection_minus_one, the zeroth-order transmission and the minus-first-order reflection coefficient under positive incidence.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nonreciprocal_gain_contrast(
    alpha_m: float,
    E0: float,
    rho0: float,
    kappa_m: float,
    omega_m: float,
    omega: float,
    segment_length: float,
    n_order: int,
) -> dict:
    """Run the scattering chain for both directions of incidence and return the ratio of the total gains.

    Parameters
    ----------
    alpha_m : float
        Normalised modulation depth, of magnitude below one.
    E0 : float
        Elastic modulus of the unmodulated rod in pascal.
    rho0 : float
        Mass density in kilogram per cubic metre.
    kappa_m : float
        Modulation wavenumber in radians per metre for positive incidence.
    omega_m : float
        Modulation angular frequency in radians per second.
    omega : float
        Incident angular frequency in radians per second.
    segment_length : float
        Length of the modulated segment in metres.
    n_order : int
        Truncation order of the harmonic expansion, at least two.

    Returns
    -------
    dict
        Under the keys positive_gain, negative_gain, contrast,
        positive_transmission_zero and positive_reflection_minus_one, each a native
        float. The contrast entry is the ratio of positive_gain to negative_gain.

    Raises
    ------
    ValueError
        If any argument violates its stated range, if n_order is below two, or if
        the negative-incidence run returns a gain of zero.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_nonreciprocal_gain_contrast(
    alpha_m: float,
    E0: float,
    rho0: float,
    kappa_m: float,
    omega_m: float,
    omega: float,
    segment_length: float,
    n_order: int,
) -> dict:
    if int(n_order) != n_order or int(n_order) < 2:
        raise ValueError("n_order must be an integer of two or more")
    n_order = int(n_order)

    built = _oracle_modulation_coupling_matrix(alpha_m, E0, n_order)  # noqa: F821
    coefficients = built["coefficients"]
    coupling = built["coupling"]

    results = {}
    for label, sign in (("positive", 1.0), ("negative", -1.0)):
        travelling = sign * float(kappa_m)
        operator = _oracle_floquet_quadratic_operator(  # noqa: F821
            omega, coupling, rho0, travelling, omega_m
        )
        spectrum = _oracle_floquet_wavenumber_spectrum(  # noqa: F821
            operator["a2"], operator["a1"], operator["a0"]
        )
        families = _oracle_directional_mode_families(  # noqa: F821
            spectrum["wavenumbers"], spectrum["mode_shapes"], coupling,
            omega, rho0, travelling, omega_m,
        )
        system = _oracle_interface_coupling_system(  # noqa: F821
            families["forward_wavenumbers"], families["forward_modes"],
            families["backward_wavenumbers"], families["backward_modes"],
            coefficients, omega, rho0, travelling, omega_m, segment_length,
        )
        scattered = _oracle_harmonic_scattering_coefficients(  # noqa: F821
            system["matrix"], system["rhs"], system["j_order"]
        )
        budget = _oracle_harmonic_power_budget(  # noqa: F821
            scattered["transmission"], scattered["reflection"], omega, omega_m
        )
        results[label] = (budget, scattered, system["j_order"])

    positive_gain = float(results["positive"][0]["total_gain"])
    negative_gain = float(results["negative"][0]["total_gain"])
    if negative_gain == 0.0:
        raise ValueError("the negative-incidence run returned a gain of zero")

    scattered_positive, j_order = results["positive"][1], results["positive"][2]
    return {
        "positive_gain": positive_gain,
        "negative_gain": negative_gain,
        "contrast": positive_gain / negative_gain,
        "positive_transmission_zero": float(scattered_positive["transmission"][j_order]),
        "positive_reflection_minus_one": float(scattered_positive["reflection"][j_order - 1]),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\ndef digest(d):\n    return (round(d['contrast'], 6), round(d['positive_gain'], 6), round(d['negative_gain'], 6))\n",
            "call": "digest(nonreciprocal_gain_contrast(0.3, 1.0, 1.0, 10.0, 20.0, 15.0, 0.5 * np.pi, 8))",
            "gold_call": "digest(_oracle_nonreciprocal_gain_contrast(0.3, 1.0, 1.0, 10.0, 20.0, 15.0, 0.5 * np.pi, 8))",
        },
        {
            "setup": "import numpy as np\ndef digest(d):\n    return (round(d['contrast'], 6), round(d['positive_transmission_zero'], 6), round(d['positive_reflection_minus_one'], 6))\n",
            "call": "digest(nonreciprocal_gain_contrast(0.3, 1.0, 1.0, 10.0, 20.0, 15.0, 0.6 * np.pi, 4))",
            "gold_call": "digest(_oracle_nonreciprocal_gain_contrast(0.3, 1.0, 1.0, 10.0, 20.0, 15.0, 0.6 * np.pi, 4))",
        },
        {
            "setup": "import numpy as np\ndef digest(d):\n    return (round(d['contrast'], 8), round(d['positive_gain'], 8))\n",
            "call": "digest(nonreciprocal_gain_contrast(0.0, 1.0, 1.0, 10.0, 20.0, 15.0, 0.5 * np.pi, 3))",
            "gold_call": "digest(_oracle_nonreciprocal_gain_contrast(0.0, 1.0, 1.0, 10.0, 20.0, 15.0, 0.5 * np.pi, 3))",
        },
        {
            "setup": "import numpy as np\ndef verdict(fn, am=0.3, e=1.0, r=1.0, km=10.0, wm=20.0, w=15.0, L=0.5 * np.pi, n=4):\n    try:\n        fn(am, e, r, km, wm, w, L, n)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "(verdict(nonreciprocal_gain_contrast, n=1), verdict(nonreciprocal_gain_contrast, am=1.5), verdict(nonreciprocal_gain_contrast, e=0.0), verdict(nonreciprocal_gain_contrast, km=0.0), verdict(nonreciprocal_gain_contrast, L=-1.0), verdict(nonreciprocal_gain_contrast, w=0.0), verdict(nonreciprocal_gain_contrast))",
            "gold_call": "(verdict(_oracle_nonreciprocal_gain_contrast, n=1), verdict(_oracle_nonreciprocal_gain_contrast, am=1.5), verdict(_oracle_nonreciprocal_gain_contrast, e=0.0), verdict(_oracle_nonreciprocal_gain_contrast, km=0.0), verdict(_oracle_nonreciprocal_gain_contrast, L=-1.0), verdict(_oracle_nonreciprocal_gain_contrast, w=0.0), verdict(_oracle_nonreciprocal_gain_contrast))",
        },
    ]
