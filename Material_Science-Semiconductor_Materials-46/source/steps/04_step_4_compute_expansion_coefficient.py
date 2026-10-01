"""
Return the amplitude with which one product eigenmode enters the expansion of a via that starts at a spatially uniform temperature above ambient.

The transient temperature difference field is a double series over the axial modes and, for each of them, over the decay rates of the layered radial problem. The amplitudes are fixed by projecting the initial condition onto the eigenmodes, which requires an orthogonality relation, and the relation that holds here is not the naive one. The radial eigenfunctions of a layered cylinder are orthogonal only when the integral over the cross-section is weighted by the volumetric heat capacity of the material occupying each annulus, that is, by the ratio of the layer's thermal conductivity to its thermal diffusivity. That weight is what makes the layered radial operator self-adjoint once the interface flux condition is imposed, and it is a physical statement: the modes are orthogonal with respect to stored thermal energy, not with respect to volume.

With that weight in place the projection is a product of an axial factor and a radial factor, both of which are available in closed form for a uniform initial temperature. The radial moment of an eigenfunction over an annulus and the square norm over the same annulus both follow from the Bessel equation itself rather than from tabulated antiderivatives: the moment is minus the radius times the slope divided by the squared radial parameter, and the square norm is half the squared radius times the sum of the eigenfunction squared and the squared slope divided by the squared radial parameter, each evaluated as a difference between the annulus endpoints. Both expressions carry over unchanged to layers whose radial parameter is negative, which is why they are preferable to formulas written specifically for oscillatory Bessel functions.

The projection coefficient is not invariant to the arbitrary scale of the eigenfunction. If the eigenfunction is multiplied by a factor c, its radial moment is multiplied by c and its square norm by c squared, so the coefficient is divided by c; only the product of the coefficient and eigenfunction is invariant. This projection is also the only place the initial condition enters the whole calculation: everything downstream, including the thermoelastic potential and the stress field, is linear in these modal products, so a uniform initial temperature rise simply scales the entire transient response.

Returns
-------
float: the expansion amplitude of the product eigenmode in K, as a native Python float.float: the expansion amplitude of the product eigenmode in K, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np



def compute_expansion_coefficient(radii: np.ndarray, capacities: np.ndarray,
                                  eigenfunction: np.ndarray,
                                  axial_mode: np.ndarray,
                                  initial_rise: float) -> float:
    """Return the expansion amplitude of one product eigenmode.

    Parameters
    ----------
    radii : np.ndarray
        Outer radius of each layer in m, shape (n_layers,), strictly
        increasing and positive.
    capacities : np.ndarray
        Volumetric heat capacity of each layer in J/(m^3 K), shape
        (n_layers,), all positive.
    eigenfunction : np.ndarray
        Array of shape (n_layers, 3) holding the squared radial parameter and
        the two radial amplitudes of every layer.
    axial_mode : np.ndarray
        Array of shape (3,) holding the axial eigenvalue in 1/m, the axial
        square norm in m and the axial eigenfunction integral in m.
    initial_rise : float
        Spatially uniform initial temperature of the via above ambient in K.

    Returns
    -------
    coefficient : float
        Expansion amplitude of the mode in K, as a native Python float.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, or if the
        heat-capacity-weighted square norm of the eigenfunction vanishes;
        invalid input must raise ValueError rather than return a sentinel
        value.
    """
    return coefficient  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_expansion_coefficient(radii: np.ndarray,
                                          capacities: np.ndarray,
                                          eigenfunction: np.ndarray,
                                          axial_mode: np.ndarray,
                                          initial_rise: float) -> float:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np
    from scipy.special import i0, i1, j0, j1, k0, k1, y0, y1

    if not (isinstance(initial_rise, (int, float, np.floating, np.integer))
            and not isinstance(initial_rise, bool) and np.isfinite(initial_rise)):
        raise ValueError("initial_rise must be a finite number")

    radii = np.asarray(radii, dtype=float).ravel()
    capacities = np.asarray(capacities, dtype=float).ravel()
    eigenfunction = np.asarray(eigenfunction, dtype=float)
    axial_mode = np.asarray(axial_mode, dtype=float).ravel()
    n_layers = radii.size
    if n_layers < 1:
        raise ValueError("radii must hold at least one layer")
    if capacities.size != n_layers:
        raise ValueError("radii and capacities must agree in length")
    if eigenfunction.shape != (n_layers, 3):
        raise ValueError("eigenfunction must have shape (n_layers, 3)")
    if axial_mode.size != 3:
        raise ValueError("axial_mode must have three entries")
    if not np.all(np.isfinite(radii)) or radii[0] <= 0.0 or np.any(np.diff(radii) <= 0.0):
        raise ValueError("radii must be finite, positive and strictly increasing")
    if not np.all(np.isfinite(capacities)) or np.any(capacities <= 0.0):
        raise ValueError("capacities must be finite and positive")

    eta, axial_norm, axial_moment = (float(axial_mode[0]), float(axial_mode[1]),
                                     float(axial_mode[2]))
    if not (np.isfinite(eta) and eta > 0.0):
        raise ValueError("the axial eigenvalue must be finite and > 0")
    if not (np.isfinite(axial_norm) and axial_norm > 0.0):
        raise ValueError("the axial square norm must be finite and > 0")

    inner = np.concatenate([[0.0], radii[:-1]])

    def _value_and_slope(mu, amp_a, amp_b, radius):
        """Radial eigenfunction and its derivative at a radius."""
        if radius == 0.0:
            return amp_a, 0.0
        if mu > 0.0:
            b = np.sqrt(mu)
            x = b * radius
            return (amp_a * j0(x) + amp_b * y0(x),
                    -b * (amp_a * j1(x) + amp_b * y1(x)))
        if mu < 0.0:
            b = np.sqrt(-mu)
            x = b * radius
            return (amp_a * i0(x) + amp_b * k0(x),
                    b * (amp_a * i1(x) - amp_b * k1(x)))
        return amp_a + amp_b * np.log(radius), amp_b / radius

    def _moments(mu, amp_a, amp_b, radius):
        """Antiderivatives of r*R and r*R^2 evaluated at a radius."""
        if radius == 0.0:
            return 0.0, 0.0
        if mu != 0.0:
            value, slope = _value_and_slope(mu, amp_a, amp_b, radius)
            return (-radius * slope / mu,
                    0.5 * radius ** 2 * (slope ** 2 / mu + value ** 2))
        log_r = np.log(radius)
        first = (0.5 * amp_a * radius ** 2
                 + amp_b * (0.5 * radius ** 2 * log_r - 0.25 * radius ** 2))
        second = (0.5 * amp_a ** 2 * radius ** 2
                  + amp_a * amp_b * (radius ** 2 * log_r - 0.5 * radius ** 2)
                  + amp_b ** 2 * (0.5 * radius ** 2 * log_r ** 2
                                  - 0.5 * radius ** 2 * log_r + 0.25 * radius ** 2))
        return first, second

    # Both radial integrals are weighted by the volumetric heat capacity,
    # which is the weight that makes the layered radial operator self-adjoint.
    radial_moment = 0.0
    radial_norm = 0.0
    for i in range(n_layers):
        mu, amp_a, amp_b = (float(eigenfunction[i, 0]), float(eigenfunction[i, 1]),
                            float(eigenfunction[i, 2]))
        outer_first, outer_second = _moments(mu, amp_a, amp_b, radii[i])
        inner_first, inner_second = _moments(mu, amp_a, amp_b, inner[i])
        radial_moment += capacities[i] * (outer_first - inner_first)
        radial_norm += capacities[i] * (outer_second - inner_second)

    if radial_norm == 0.0:
        raise ValueError("the radial square norm of the eigenfunction vanishes")

    return float(initial_rise) * axial_moment * radial_moment / (axial_norm * radial_norm)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: slowest mode of the benchmark via (normal scenario) ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 25.0e-6])
capacities = np.array([8960.0 * 385.0, 2200.0 * 730.0, 2329.0 * 700.0])
eigenfunction = np.array([
    [-1.0342822695e+07, 1.0000000000e+00, 0.0000000000e+00],
    [ 6.7677263268e+09, 1.0260381408e+00, 1.3245291089e+00],
    [ 1.2975309035e+07, 1.0318116676e+00, 6.4857891790e-03]])
axial_mode = np.array([7853.981633974483, 1.0e-4, 1.2732395447e-04])
initial_rise = 100.0
""",
            "call": "compute_expansion_coefficient(radii, capacities, eigenfunction, axial_mode, initial_rise)",
            "gold_call": "_oracle_compute_expansion_coefficient(radii, capacities, eigenfunction, axial_mode, initial_rise)",
        },
        # --- Valid: a negative initial rise simply flips the amplitude ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 25.0e-6])
capacities = np.array([8960.0 * 385.0, 2200.0 * 730.0, 2329.0 * 700.0])
eigenfunction = np.array([
    [-1.0342822695e+07, 1.0000000000e+00, 0.0000000000e+00],
    [ 6.7677263268e+09, 1.0260381408e+00, 1.3245291089e+00],
    [ 1.2975309035e+07, 1.0318116676e+00, 6.4857891790e-03]])
axial_mode = np.array([7853.981633974483, 1.0e-4, 1.2732395447e-04])
initial_rise = -45.0
""",
            "call": "compute_expansion_coefficient(radii, capacities, eigenfunction, axial_mode, initial_rise)",
            "gold_call": "_oracle_compute_expansion_coefficient(radii, capacities, eigenfunction, axial_mode, initial_rise)",
        },
        # --- Boundary: homogeneous cylinder whose radial eigenfunction is flat ---
        {
            "setup": """import numpy as np
radii = np.array([25.0e-6])
capacities = np.array([2329.0 * 700.0])
eigenfunction = np.array([[0.0, 1.0, 0.0]])
axial_mode = np.array([7853.981633974483, 1.0e-4, 1.2732395447e-04])
initial_rise = 100.0
""",
            "call": "compute_expansion_coefficient(radii, capacities, eigenfunction, axial_mode, initial_rise)",
            "gold_call": "_oracle_compute_expansion_coefficient(radii, capacities, eigenfunction, axial_mode, initial_rise)",
        },
        # --- Edge: scaling the eigenfunction reduces the coefficient inversely ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 16.0e-6, 25.0e-6])
capacities = np.array([8960.0 * 385.0, 2200.0 * 730.0, 2329.0 * 700.0])
eigenfunction = np.array([
    [-1.0342822695e+07, 1.0000000000e+03, 0.0000000000e+00],
    [ 6.7677263268e+09, 1.0260381408e+03, 1.3245291089e+03],
    [ 1.2975309035e+07, 1.0318116676e+03, 6.4857891790e+00]])
axial_mode = np.array([7853.981633974483, 1.0e-4, 1.2732395447e-04])
initial_rise = 100.0
""",
            "call": "compute_expansion_coefficient(radii, capacities, eigenfunction, axial_mode, initial_rise)",
            "gold_call": "_oracle_compute_expansion_coefficient(radii, capacities, eigenfunction, axial_mode, initial_rise)",
        },
        # --- Invalid: eigenfunction array of the wrong shape ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 25.0e-6])
capacities = np.array([3.4496e6, 1.6303e6])
eigenfunction = np.array([[0.0, 1.0], [0.0, 1.0]])
axial_mode = np.array([7853.98, 1.0e-4, 1.27e-4])
def run_model():
    try:
        compute_expansion_coefficient(radii, capacities, eigenfunction, axial_mode, 100.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_expansion_coefficient(radii, capacities, eigenfunction, axial_mode, 100.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive volumetric heat capacity ---
        {
            "setup": """import numpy as np
radii = np.array([15.0e-6, 25.0e-6])
capacities = np.array([3.4496e6, 0.0])
eigenfunction = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
axial_mode = np.array([7853.98, 1.0e-4, 1.27e-4])
def run_model():
    try:
        compute_expansion_coefficient(radii, capacities, eigenfunction, axial_mode, 100.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_expansion_coefficient(radii, capacities, eigenfunction, axial_mode, 100.0)
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
