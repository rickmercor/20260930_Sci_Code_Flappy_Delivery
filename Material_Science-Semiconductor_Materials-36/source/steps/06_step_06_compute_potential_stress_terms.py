"""
Evaluate the displacement and stress contributions of the thermoelastic displacement potential in one layer, for one axial mode, at one radius and instant.

The temperature rise above the 300 K ambient, which is also the stress-free temperature, is the series theta = sum over n of c_mn R_imn(r) cos(eta_m z) exp(-mu_mn t) for the axial mode m considered here. The thermoelastic displacement potential Phi of a layer is the particular solution of the quasi-static thermoelastic problem that carries the same terms as that series, with displacements u_r = dPhi/dr and u_z = dPhi/dz; its stresses are those of that displacement field under the isotropic thermoelastic law of the layer, free thermal strain included. This function evaluates, for one axial mode and the layer of interest, the six potential contributions summed over the radial modes at the given radius and time.

Return convention. The axial dependence is stripped: the returned u_r, sigma_rr, sigma_phiphi and sigma_zz are the coefficients of cos(eta_m z), and u_z and sigma_rz are the coefficients of sin(eta_m z), each carrying its own sign. Displacements are in metres and stresses in Pa, with tension positive. The radial factor R_imn uses the amplitudes and separation constant of the layer supplied in layer_eigen_coefficients, in the convention of sub-problem 02.

Returns
-------
np.ndarray of shape (6,), float: the potential amplitudes for u_r, u_z, sigma_rr, sigma_phiphi, sigma_zz and sigma_rz.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_potential_stress_terms(radius: float, axial_eigenvalue: float,
                                   decay_rates: np.ndarray,
                                   modal_coefficients: np.ndarray,
                                   layer_eigen_coefficients: np.ndarray,
                                   expansion: float, poisson: float, shear: float,
                                   time: float) -> np.ndarray:
    """Thermoelastic-potential contribution of one layer for one axial mode.

    Parameters
    ----------
    radius : float
        Radial coordinate in metres (> 0).
    axial_eigenvalue : float
        Axial eigenvalue eta of this mode in inverse metres (> 0).
    decay_rates : np.ndarray
        Shape (n_radial,). Thermal decay rates of this axial mode, in inverse
        seconds.
    modal_coefficients : np.ndarray
        Shape (n_radial,). Expansion coefficients of this axial mode, in kelvin.
    layer_eigen_coefficients : np.ndarray
        Shape (n_radial, 3). For each radial mode, the two amplitudes and the
        separation constant of the layer of interest.
    expansion : float
        Coefficient of thermal expansion of the layer in 1/K (> 0).
    poisson : float
        Poisson ratio of the layer, 0 < poisson < 0.5.
    shear : float
        Shear modulus of the layer in Pa (> 0).
    time : float
        Elapsed time since the quench in seconds (>= 0).

    Returns
    -------
    terms : np.ndarray
        Array of shape (6,) holding the amplitudes of the radial displacement,
        the axial displacement, the radial stress, the hoop stress, the axial
        stress and the shear stress, in metres for the displacements and Pa for
        the stresses.

    Raises
    ------
    ValueError
        If decay_rates and modal_coefficients do not share a single
        length greater than or equal to 1; if layer_eigen_coefficients
        does not have shape (n_radial, 3); if any decay_rates entry is
        not finite and greater than 0; if modal_coefficients or
        layer_eigen_coefficients is not finite; if radius,
        axial_eigenvalue, expansion or shear is not a finite number
        greater than 0; if poisson is not a finite number in the open
        interval (0, 0.5); or if time is not a finite number greater
        than or equal to 0.
    """
    return terms  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_potential_stress_terms(radius: float, axial_eigenvalue: float,
                                           decay_rates: np.ndarray,
                                           modal_coefficients: np.ndarray,
                                           layer_eigen_coefficients: np.ndarray,
                                           expansion: float, poisson: float,
                                           shear: float,
                                           time: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
    from scipy.special import i0, i1, j0, j1, k0, k1, y0, y1

    decay_rates = np.asarray(decay_rates, dtype=float).ravel()
    modal_coefficients = np.asarray(modal_coefficients, dtype=float).ravel()
    layer_eigen_coefficients = np.asarray(layer_eigen_coefficients, dtype=float)
    n_radial = decay_rates.size
    if n_radial < 1 or modal_coefficients.size != n_radial:
        raise ValueError("decay_rates and modal_coefficients must share one length >= 1")
    if layer_eigen_coefficients.shape != (n_radial, 3):
        raise ValueError("layer_eigen_coefficients must have shape (n_radial, 3)")
    if not (np.all(np.isfinite(decay_rates)) and np.all(decay_rates > 0.0)):
        raise ValueError("decay_rates entries must be finite and > 0")
    if not np.all(np.isfinite(modal_coefficients)):
        raise ValueError("modal_coefficients must be finite")
    if not np.all(np.isfinite(layer_eigen_coefficients)):
        raise ValueError("layer_eigen_coefficients must be finite")
    for name, value in (("radius", radius), ("axial_eigenvalue", axial_eigenvalue),
                        ("expansion", expansion), ("shear", shear)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(poisson, (int, float, np.floating, np.integer))
            and np.isfinite(poisson) and 0.0 < float(poisson) < 0.5):
        raise ValueError("poisson must be a finite number in the open interval (0, 0.5)")
    if not (isinstance(time, (int, float, np.floating, np.integer))
            and np.isfinite(time) and float(time) >= 0.0):
        raise ValueError("time must be a finite number >= 0")

    radius = float(radius)
    eta = float(axial_eigenvalue)
    time = float(time)
    shear = float(shear)

    def _basis(q, r):
        """Value and radial derivative of the two solutions of the radial ODE."""
        if q > 0.0:
            c = np.sqrt(q)
            return j0(c * r), y0(c * r), -c * j1(c * r), -c * y1(c * r)
        if q < 0.0:
            c = np.sqrt(-q)
            return i0(c * r), k0(c * r), c * i1(c * r), -c * k1(c * r)
        return 1.0, np.log(r), 0.0, 1.0 / r

    thermal_factor = float(expansion) * (1.0 + float(poisson)) / (1.0 - float(poisson))

    terms = np.zeros(6, dtype=float)
    for n in range(n_radial):
        amp_a, amp_b, separation = layer_eigen_coefficients[n]
        first, second, d_first, d_second = _basis(separation, radius)
        shape = amp_a * first + amp_b * second
        slope = amp_a * d_first + amp_b * d_second
        # The radial factor solves R'' + R'/r + beta^2 R = 0, so the curvature
        # is available in closed form and needs no numerical differentiation.
        curvature = -slope / radius - separation * shape
        laplace_eigenvalue = separation + eta ** 2
        weight = modal_coefficients[n] * np.exp(-decay_rates[n] * time)
        potential = -thermal_factor * weight / laplace_eigenvalue
        terms[0] += potential * slope
        terms[1] += -potential * eta * shape
        terms[2] += 2.0 * shear * (potential * curvature - thermal_factor * weight * shape)
        terms[3] += 2.0 * shear * (potential * slope / radius
                                   - thermal_factor * weight * shape)
        terms[4] += 2.0 * shear * (-eta ** 2 * potential * shape
                                   - thermal_factor * weight * shape)
        terms[5] += 2.0 * shear * (-eta * potential * slope)

    return terms

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
eta = np.pi / (2.0 * 200.0e-6)
mus = np.array([5.701567849e3, 9.697475846e4])
coef = np.array([125.532767, 1.782778])
layer = np.array([[1.038174, 7.102796e-3, 9.817019e6],
                  [0.412355, -2.115468e-2, 1.207142e9]])
""",
            "call": "digest(compute_potential_stress_terms(20.0e-6, eta, mus, coef, layer, 2.6e-6, 0.28, 66.40625e9, 5.0e-5))",
            "gold_call": "digest(_oracle_compute_potential_stress_terms(20.0e-6, eta, mus, coef, layer, 2.6e-6, 0.28, 66.40625e9, 5.0e-5))",
        },
        # --- Valid: copper core on the modified Bessel branch ---
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
eta = np.pi / (2.0 * 200.0e-6)
mus = np.array([5.701567849e3, 9.697475846e4])
coef = np.array([125.532767, 1.782778])
layer = np.array([[1.0, 0.0, -1.25147063e7],
                  [1.0, 0.0, 7.74450e8]])
""",
            "call": "digest(compute_potential_stress_terms(7.5e-6, eta, mus, coef, layer, 17.0e-6, 0.35, 40.7407407e9, 1.0e-5))",
            "gold_call": "digest(_oracle_compute_potential_stress_terms(7.5e-6, eta, mus, coef, layer, 17.0e-6, 0.35, 40.7407407e9, 1.0e-5))",
        },
        # --- Boundary: a single radial mode evaluated at time zero ---
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
eta = np.pi / (2.0 * 200.0e-6)
mus = np.array([5.701567849e3])
coef = np.array([125.532767])
layer = np.array([[1.0, 0.0, -1.25147063e7]])
""",
            "call": "digest(compute_potential_stress_terms(1.0e-6, eta, mus, coef, layer, 17.0e-6, 0.35, 40.7407407e9, 0.0))",
            "gold_call": "digest(_oracle_compute_potential_stress_terms(1.0e-6, eta, mus, coef, layer, 17.0e-6, 0.35, 40.7407407e9, 0.0))",
        },
        # --- Consistency: the shear amplitude is fixed by the radial displacement ---
        # Both come from the same potential, so sigma_rz + 2 G eta u_r vanishes identically.
        {
            "setup": """import numpy as np
eta = np.pi / (2.0 * 200.0e-6)
mus = np.array([5.701567849e3, 9.697475846e4])
coef = np.array([125.532767, 1.782778])
layer = np.array([[1.038174, 7.102796e-3, 9.817019e6],
                  [0.412355, -2.115468e-2, 1.207142e9]])
def check(fn):
    g = 66.40625e9
    worst = 0.0
    for r in (17.0e-6, 20.0e-6, 29.0e-6):
        terms = np.asarray(fn(r, eta, mus, coef, layer, 2.6e-6, 0.28, g, 5.0e-5), dtype=float)
        scale = max(abs(terms[5]), 1.0e-30)
        worst = max(worst, abs(terms[5] + 2.0 * g * eta * terms[0]) / scale)
    return int(worst < 1e-10)
""",
            "call": "check(compute_potential_stress_terms)",
            "gold_call": "check(_oracle_compute_potential_stress_terms)",
        },
        # --- Consistency: the trace identity of a potential-only stress state ---
        # Since the potential satisfies the Poisson equation, the three normal
        # amplitudes and the axial displacement obey
        # sigma_rr + sigma_phiphi - sigma_zz + 4 G eta u_z = 0.
        {
            "setup": """import numpy as np
eta = np.pi / (2.0 * 200.0e-6)
mus = np.array([5.701567849e3])
coef = np.array([125.532767])
layer = np.array([[1.0, 0.0, -1.25147063e7]])
def check(fn):
    g = 40.7407407e9
    worst = 0.0
    for r in (2.0e-6, 7.5e-6, 14.0e-6):
        t = np.asarray(fn(r, eta, mus, coef, layer, 17.0e-6, 0.35, g, 1.0e-5), dtype=float)
        residual = t[2] + t[3] - t[4] + 4.0 * g * eta * t[1]
        worst = max(worst, abs(residual) / max(abs(t[4]), 1.0e-30))
    return int(worst < 1e-10)
""",
            "call": "check(compute_potential_stress_terms)",
            "gold_call": "check(_oracle_compute_potential_stress_terms)",
        },
        # --- Invalid: Poisson ratio outside the admissible range ---
        {
            "setup": """import numpy as np
mus = np.array([5.7e3])
coef = np.array([125.5])
layer = np.array([[1.0, 0.0, -1.25e7]])
def run_model():
    try:
        compute_potential_stress_terms(7.5e-6, 7854.0, mus, coef, layer, 17.0e-6, 0.6, 4.07e10, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_potential_stress_terms(7.5e-6, 7854.0, mus, coef, layer, 17.0e-6, 0.6, 4.07e10, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: eigen-coefficient table of the wrong shape ---
        {
            "setup": """import numpy as np
mus = np.array([5.7e3, 9.7e4])
coef = np.array([125.5, 1.8])
layer = np.array([[1.0, 0.0, -1.25e7]])
def run_model():
    try:
        compute_potential_stress_terms(7.5e-6, 7854.0, mus, coef, layer, 17.0e-6, 0.35, 4.07e10, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_potential_stress_terms(7.5e-6, 7854.0, mus, coef, layer, 17.0e-6, 0.35, 4.07e10, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: radius on the axis ---
        {
            "setup": """import numpy as np
mus = np.array([5.7e3])
coef = np.array([125.5])
layer = np.array([[1.0, 0.0, -1.25e7]])
def run_model():
    try:
        compute_potential_stress_terms(0.0, 7854.0, mus, coef, layer, 17.0e-6, 0.35, 4.07e10, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_potential_stress_terms(0.0, 7854.0, mus, coef, layer, 17.0e-6, 0.35, 4.07e10, 0.0)
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
