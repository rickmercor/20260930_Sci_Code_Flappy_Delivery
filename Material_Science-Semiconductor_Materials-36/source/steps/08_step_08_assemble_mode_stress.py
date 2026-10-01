"""
Combine the Love displacement function of a layer with the thermoelastic potential of the same layer to obtain the four stress amplitudes of one axial mode at one radius.

The stress of an axial mode inside a layer is the sum of two contributions that were computed separately for good reason. The particular part, carried by the thermoelastic displacement potential, is fixed entirely by the local temperature and the local material properties and knows nothing about the neighbouring layers. The homogeneous part, carried by the Love displacement function, knows nothing about the temperature and exists only to enforce the bonded interfaces and the traction-free outer surface. Neither is meaningful on its own: the potential part alone would leave a discontinuous traction at every interface and a loaded outer surface, while the Love part alone would describe an unheated body.

Adding them layer by layer produces the characteristic signature of a via cross section. The radial stress and the shear stress are continuous everywhere, because their continuity was imposed as an equation; the hoop stress and the axial stress jump at each interface, because nothing constrains them, and the size of the jump is set by the mismatch in expansion coefficient and elastic modulus. A copper core that is trying to expand more than its surroundings is held in a state of nearly hydrostatic compression, with the radial and hoop components almost equal there because the core is unaware of the interface until very close to it, while the silicon just outside the liner is thrown into strong hoop tension. That reversal across a one-micron oxide is what makes the liner both the peak stress location and the effective compliance that protects the substrate.

Only the four stresses are returned. The two displacement amplitudes were needed to write the interface conditions, but the observable quantities of a reliability or piezoresistive assessment are the stresses, and returning them alone keeps the interface of this step narrow. The amplitudes still have to be multiplied by cos(eta z) for the three normal components and by sin(eta z) for the shear component before the modes are summed.

Returns
-------
np.ndarray of shape (4,), float: the radial, hoop, axial and shear stress amplitudes of this axial mode, in Pa.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_mode_stress(radius: float, radii: np.ndarray, poisson: np.ndarray,
                         shear: np.ndarray, axial_eigenvalue: float,
                         love_coefficients: np.ndarray,
                         potential_terms: np.ndarray) -> np.ndarray:
    """Total stress amplitudes of one axial mode at one radius.

    Parameters
    ----------
    radius : float
        Radial coordinate in metres, 0 < radius <= radii[-1].
    radii : np.ndarray
        Shape (l,). Outer radius of each layer in metres, strictly increasing,
        with l >= 2.
    poisson : np.ndarray
        Shape (l,). Poisson ratio of each layer, each in (0, 0.5).
    shear : np.ndarray
        Shape (l,). Shear modulus of each layer in Pa, all > 0.
    axial_eigenvalue : float
        Axial eigenvalue eta of this mode in inverse metres (> 0).
    love_coefficients : np.ndarray
        Shape (4 * l - 2,). Love coefficients ordered from the innermost layer
        outward, the innermost layer carrying only its two bounded ones.
    potential_terms : np.ndarray
        Shape (6,). Potential amplitudes of the layer containing radius,
        evaluated at that radius.

    Returns
    -------
    stress : np.ndarray
        Array of shape (4,) holding the radial, hoop, axial and shear stress
        amplitudes in Pa.

    Raises
    ------
    ValueError
        If radii, poisson and shear do not share a single length l >= 2;
        if radii is not finite, positive and strictly increasing; if any
        poisson entry is not finite and inside the open interval (0,
        0.5); if any shear entry is not finite and greater than 0; if
        love_coefficients is not a finite array of length 4 * l - 2; if
        potential_terms is not a finite array of length 6; if
        axial_eigenvalue is not a finite number greater than 0; if
        radius is not a finite number; or if radius lies outside the
        half-open interval (0, radii[-1]].
    """
    return stress  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np 

def _oracle_assemble_mode_stress(radius: float, radii: np.ndarray,
                                 poisson: np.ndarray, shear: np.ndarray,
                                 axial_eigenvalue: float,
                                 love_coefficients: np.ndarray,
                                 potential_terms: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
    from scipy.special import i0, i1, k0, k1

    radii = np.asarray(radii, dtype=float).ravel()
    poisson = np.asarray(poisson, dtype=float).ravel()
    shear = np.asarray(shear, dtype=float).ravel()
    love_coefficients = np.asarray(love_coefficients, dtype=float).ravel()
    potential_terms = np.asarray(potential_terms, dtype=float).ravel()
    n_layers = radii.size
    if n_layers < 2 or poisson.size != n_layers or shear.size != n_layers:
        raise ValueError("radii, poisson and shear must share one length l >= 2")
    if not np.all(np.isfinite(radii)) or radii[0] <= 0.0 or np.any(np.diff(radii) <= 0.0):
        raise ValueError("radii must be finite, positive and strictly increasing")
    if not (np.all(np.isfinite(poisson)) and np.all(poisson > 0.0) and np.all(poisson < 0.5)):
        raise ValueError("poisson entries must be finite and in the open interval (0, 0.5)")
    if not (np.all(np.isfinite(shear)) and np.all(shear > 0.0)):
        raise ValueError("shear entries must be finite and > 0")
    if love_coefficients.size != 4 * n_layers - 2 or not np.all(np.isfinite(love_coefficients)):
        raise ValueError("love_coefficients must be a finite array of length 4 * l - 2")
    if potential_terms.size != 6 or not np.all(np.isfinite(potential_terms)):
        raise ValueError("potential_terms must be a finite array of length 6")
    if not (isinstance(axial_eigenvalue, (int, float, np.floating, np.integer))
            and np.isfinite(axial_eigenvalue) and float(axial_eigenvalue) > 0.0):
        raise ValueError("axial_eigenvalue must be a finite number > 0")
    if not (isinstance(radius, (int, float, np.floating, np.integer))
            and np.isfinite(radius)):
        raise ValueError("radius must be a finite number")
    if float(radius) <= 0.0 or float(radius) > radii[-1]:
        raise ValueError("radius must lie in the half-open interval (0, radii[-1]]")

    radius = float(radius)
    eta = float(axial_eigenvalue)

    layer = int(min(np.searchsorted(radii, radius, side="left"), n_layers - 1))
    nu = poisson[layer]
    modulus = shear[layer]

    x = eta * radius
    bessel_i0, bessel_i1 = i0(x), i1(x)
    bessel_k0, bessel_k1 = k0(x), k1(x)
    eta2, eta3 = eta ** 2, eta ** 3

    normal = 2.0 * modulus * np.array([
        -eta3 * (bessel_i0 - bessel_i1 / x),
        -eta3 * (bessel_k0 + bessel_k1 / x),
        2.0 * nu * eta3 * bessel_i0 - eta3 * (bessel_i0 + x * bessel_i1),
        -2.0 * nu * eta3 * bessel_k0 - eta3 * (-bessel_k0 + x * bessel_k1)])
    hoop = 2.0 * modulus * np.array([
        -eta2 / radius * bessel_i1,
        eta2 / radius * bessel_k1,
        2.0 * nu * eta3 * bessel_i0 - eta2 / radius * x * bessel_i0,
        -2.0 * nu * eta3 * bessel_k0 + eta2 / radius * x * bessel_k0])
    axial = 2.0 * modulus * np.array([
        eta3 * bessel_i0, eta3 * bessel_k0,
        2.0 * (2.0 - nu) * eta3 * bessel_i0 + eta3 * x * bessel_i1,
        -2.0 * (2.0 - nu) * eta3 * bessel_k0 + eta3 * x * bessel_k1])
    shear_row = 2.0 * modulus * np.array([
        eta3 * bessel_i1, -eta3 * bessel_k1,
        2.0 * (1.0 - nu) * eta3 * bessel_i1 + eta3 * x * bessel_i0,
        2.0 * (1.0 - nu) * eta3 * bessel_k1 - eta3 * x * bessel_k0])

    if layer == 0:
        # The innermost layer carries only the coefficients bounded on the axis.
        active = np.array([love_coefficients[0], 0.0, love_coefficients[1], 0.0])
    else:
        start = 2 + 4 * (layer - 1)
        active = love_coefficients[start:start + 4]

    return np.array([
        float(normal @ active + potential_terms[2]),
        float(hoop @ active + potential_terms[3]),
        float(axial @ active + potential_terms[4]),
        float(shear_row @ active + potential_terms[5])])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: silicon substrate of the benchmark via (normal scenario) ---
        {
            "setup": """import numpy as np
def digest(*parts):
    # The harness compares plain numbers, so each array this step returns is
    # reduced to two descriptors of order one: a normalized weighted sum that
    # pins down its shape, and the log of its weighted magnitude that pins down
    # its scale. Both stay of order one whatever the component is worth, so an
    # error in a small component cannot hide behind a large one.
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
poisson = np.array([0.35, 0.17, 0.28])
shear = np.array([110.0e9, 70.0e9, 170.0e9]) / (2.0 * (1.0 + poisson))
eta = np.pi / (2.0 * 200.0e-6)
love = np.linspace(-2.0, 3.0, 10) * 1.0e-19
potential = np.array([1.0e-9, -2.0e-9, -5.0e7, 1.2e8, 6.0e7, -3.0e6])
""",
            "call": "digest(assemble_mode_stress(20.0e-6, radii, poisson, shear, eta, love, potential))",
            "gold_call": "digest(_oracle_assemble_mode_stress(20.0e-6, radii, poisson, shear, eta, love, potential))",
        },
        # --- Valid: copper core, where only two Love coefficients act ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
poisson = np.array([0.35, 0.17, 0.28])
shear = np.array([110.0e9, 70.0e9, 170.0e9]) / (2.0 * (1.0 + poisson))
eta = np.pi / (2.0 * 200.0e-6)
love = np.linspace(-2.0, 3.0, 10) * 1.0e-19
potential = np.array([2.0e-9, -1.0e-9, -1.1e8, -1.1e8, -2.0e8, -4.0e5])
""",
            "call": "digest(assemble_mode_stress(7.5e-6, radii, poisson, shear, eta, love, potential))",
            "gold_call": "digest(_oracle_assemble_mode_stress(7.5e-6, radii, poisson, shear, eta, love, potential))",
        },
        # --- Boundary: exactly on the outer surface ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
poisson = np.array([0.35, 0.17, 0.28])
shear = np.array([110.0e9, 70.0e9, 170.0e9]) / (2.0 * (1.0 + poisson))
eta = np.pi / (2.0 * 200.0e-6)
love = np.linspace(-2.0, 3.0, 10) * 1.0e-19
potential = np.array([1.0e-9, -2.0e-9, -5.0e7, 1.2e8, 6.0e7, -3.0e6])
""",
            "call": "digest(assemble_mode_stress(30.0e-6, radii, poisson, shear, eta, love, potential))",
            "gold_call": "digest(_oracle_assemble_mode_stress(30.0e-6, radii, poisson, shear, eta, love, potential))",
        },
        # --- Edge: no Love content, so the stress is the bare potential part ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
poisson = np.array([0.35, 0.17, 0.28])
shear = np.array([110.0e9, 70.0e9, 170.0e9]) / (2.0 * (1.0 + poisson))
eta = np.pi / (2.0 * 200.0e-6)
love = np.zeros(10)
potential = np.array([1.0e-9, -2.0e-9, -5.0e7, 1.2e8, 6.0e7, -3.0e6])
def check(fn):
    s = np.asarray(fn(20.0e-6, radii, poisson, shear, eta, love, potential), dtype=float)
    return int(np.max(np.abs(s - potential[[2, 3, 4, 5]])) < 1e-9)
""",
            "call": "check(assemble_mode_stress)",
            "gold_call": "check(_oracle_assemble_mode_stress)",
        },
        # --- Consistency: the result is linear in the Love coefficients ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
poisson = np.array([0.35, 0.17, 0.28])
shear = np.array([110.0e9, 70.0e9, 170.0e9]) / (2.0 * (1.0 + poisson))
eta = np.pi / (2.0 * 200.0e-6)
love = np.linspace(-2.0, 3.0, 10) * 1.0e-19
zero_potential = np.zeros(6)
def check(fn):
    single = np.asarray(fn(20.0e-6, radii, poisson, shear, eta, love, zero_potential), dtype=float)
    double = np.asarray(fn(20.0e-6, radii, poisson, shear, eta, 2.0 * love, zero_potential), dtype=float)
    scale = np.maximum(np.abs(double), 1.0e-30)
    return int(np.max(np.abs(2.0 * single - double) / scale) < 1e-12)
""",
            "call": "check(assemble_mode_stress)",
            "gold_call": "check(_oracle_assemble_mode_stress)",
        },
        # --- Consistency: only the two bounded coefficients act inside the core ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
poisson = np.array([0.35, 0.17, 0.28])
shear = np.array([110.0e9, 70.0e9, 170.0e9]) / (2.0 * (1.0 + poisson))
eta = np.pi / (2.0 * 200.0e-6)
love = np.linspace(-2.0, 3.0, 10) * 1.0e-19
altered = love.copy()
altered[2:] *= -7.0
potential = np.array([2.0e-9, -1.0e-9, -1.1e8, -1.1e8, -2.0e8, -4.0e5])
def check(fn):
    a = np.asarray(fn(7.5e-6, radii, poisson, shear, eta, love, potential), dtype=float)
    b = np.asarray(fn(7.5e-6, radii, poisson, shear, eta, altered, potential), dtype=float)
    return int(np.max(np.abs(a - b)) < 1e-9)
""",
            "call": "check(assemble_mode_stress)",
            "gold_call": "check(_oracle_assemble_mode_stress)",
        },
        # --- Invalid: Love vector of the wrong length ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
poisson = np.array([0.35, 0.17, 0.28])
shear = np.array([40.7e9, 29.9e9, 66.4e9])
love = np.zeros(12)
potential = np.zeros(6)
def run_model():
    try:
        assemble_mode_stress(20.0e-6, radii, poisson, shear, 7854.0, love, potential)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_mode_stress(20.0e-6, radii, poisson, shear, 7854.0, love, potential)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: radius beyond the outer surface ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
poisson = np.array([0.35, 0.17, 0.28])
shear = np.array([40.7e9, 29.9e9, 66.4e9])
love = np.zeros(10)
potential = np.zeros(6)
def run_model():
    try:
        assemble_mode_stress(40.0e-6, radii, poisson, shear, 7854.0, love, potential)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_mode_stress(40.0e-6, radii, poisson, shear, 7854.0, love, potential)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: potential vector of the wrong length ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 30.0e-6])
poisson = np.array([0.35, 0.17, 0.28])
shear = np.array([40.7e9, 29.9e9, 66.4e9])
love = np.zeros(10)
potential = np.zeros(4)
def run_model():
    try:
        assemble_mode_stress(20.0e-6, radii, poisson, shear, 7854.0, love, potential)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_mode_stress(20.0e-6, radii, poisson, shear, 7854.0, love, potential)
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
