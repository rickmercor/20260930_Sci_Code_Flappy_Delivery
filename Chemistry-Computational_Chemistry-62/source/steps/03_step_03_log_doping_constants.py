"""
Logarithms of the doping constants of the z binding modes of a z-valent cation, each corrected by the random-phase correlation integral of the free ions.

Doping constants of the binding modes, corrected for ion correlations.

A z-valent cation M(z+) can take a polyanion site in z different ways. In mode k (k = 1..z) the
cation is attached to k polyanion repeat units and keeps z - k monovalent anions, so k = 1 is the
fully mixed mode and k = z the fully bridged mode. The doping reaction of mode k breaks intrinsic
polycation-polyanion pairs with 1/k cation and z/k anions per broken pair.

Each mode has a doping constant K_k. It combines the standard free energy of the reaction, Delta G_k
(in units of k_B T), with the electrostatic correlation chemical potentials of the free ions. For a
symmetric complex whose charged sites are all occupied, the random-phase correlation free energy of
Gaussian-smeared free ions gives

    ln K_k = -Delta G_k + (z / k) (l_B / (pi l)) * I + 1

    I = integral from 0 to infinity of [ z G(q, a_M) + G(q, a_A) ] / (1 + k2(q) / q^2) dq

with G(q, a) = exp(-q^2 a^2), a = omega^(1/3), q dimensionless (units of 1/l), l = (29.7)^(1/3)
Angstrom, l_B the Bjerrum length in Angstrom and k2(q) the squared inverse screening length of the
free ions defined in the previous step. The integral must be evaluated to a relative accuracy of about
1e-8.

Returns
-------
np.ndarray of shape (z,): ln K_k of the binding modes k = 1..z
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def log_doping_constants(delta_g: "np.ndarray", phi_free: "np.ndarray", omega: "np.ndarray", valence: int,
                         temperature: float, dielectric: float) -> "np.ndarray":
    '''Natural logarithms of the correlation-corrected doping constants of the z binding modes.

    Parameters
    ----------
    delta_g : np.ndarray
        Shape (z,): standard free energies of the binding modes k = 1..z in units of k_B T.
    phi_free : np.ndarray
        Shape (2,): free volume fractions of cation and anion.
    omega : np.ndarray
        Shape (2,): effective volumes of cation and anion relative to water.
    valence : int
        Cation charge number z.
    temperature : float
        Absolute temperature in K.
    dielectric : float
        Relative permittivity used in the Bjerrum length.

    Returns
    -------
    result : np.ndarray
        Shape (z,): ln K_k for k = 1..z.

    Raises
    ------
    ValueError
        If delta_g does not have shape (valence,), or on any condition that makes the screening
        wavenumber of the free ions invalid (phi_free or omega not of shape (2,), a negative phi_free
        entry, a non-positive omega entry, valence not an integer >= 1, non-positive temperature or
        dielectric).
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gauss_legendre_24():
    import numpy as np
    rule = getattr(_gauss_legendre_24, "rule", None)
    if rule is None:
        rule = np.polynomial.legendre.leggauss(24)
        _gauss_legendre_24.rule = rule
    return rule


def _oracle_log_doping_constants(delta_g: "np.ndarray", phi_free: "np.ndarray", omega: "np.ndarray", valence: int,
                                 temperature: float, dielectric: float) -> "np.ndarray":
    import numpy as np
    _oracle_screening_wavenumber_squared(np.array([1.0]), phi_free, omega, valence, temperature, dielectric)
    z = int(valence)
    delta_g = np.array(delta_g, dtype=float)
    if delta_g.shape != (z,):
        raise ValueError("delta_g must have shape (valence,)")
    phi_free = np.array(phi_free, dtype=float)
    omega = np.array(omega, dtype=float)
    l, l_b = _water_and_bjerrum(temperature, dielectric)
    a = omega ** (1.0 / 3.0)
    # composite Gauss-Legendre rule on panels graded geometrically from far below the screening
    # wavenumber sqrt(k2(0)) up to the point where both Gaussian factors are below exp(-46)
    k2_zero = float(_oracle_screening_wavenumber_squared(np.array([0.0]), phi_free, omega, z, temperature,
                                                         dielectric)[0])
    q_max = np.sqrt(46.0) / a.min()
    q_small = min(q_max * 1e-8, np.sqrt(k2_zero) * 1e-3) if k2_zero > 0.0 else q_max * 1e-8
    n_panels = int(np.ceil(np.log(q_max / q_small) / np.log(2.0))) + 1
    edges = np.concatenate([[0.0], np.geomspace(q_small, q_max, n_panels)])
    nodes, weights = _gauss_legendre_24()
    half = 0.5 * (edges[1:] - edges[:-1])
    mid = 0.5 * (edges[1:] + edges[:-1])
    q = (half[:, None] * nodes[None, :] + mid[:, None]).ravel()
    w = (half[:, None] * weights[None, :]).ravel()
    k2 = _oracle_screening_wavenumber_squared(q, phi_free, omega, z, temperature, dielectric)
    integrand = (z * np.exp(-(q * a[0]) ** 2) + np.exp(-(q * a[1]) ** 2)) * q * q / (q * q + k2)
    integral = float(np.dot(w, integrand))
    k = np.arange(1, z + 1, dtype=float)
    return -delta_g + (z / k) * l_b / (np.pi * l) * integral + 1.0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal: divalent cation at a moderate free-ion content
        {
            "setup": """import numpy as np
phi = np.array([0.0131, 0.0101])
omega = np.array([7.3410, 2.8363])
""",
            "call": "log_doping_constants(np.array([-4.2, -0.1]), phi.copy(), omega.copy(), 2, 298.15, 79.0)",
            "gold_call": "_oracle_log_doping_constants(np.array([-4.2, -0.1]), phi.copy(), omega.copy(), 2, 298.15, 79.0)",
            "tol": 1e-5,
        },
        # boundary: vanishing free ions, where the screening disappears
        {
            "setup": """import numpy as np
phi = np.array([1e-9, 2e-9])
omega = np.array([10.0526, 2.8363])
""",
            "call": "log_doping_constants(np.array([-2.6, -0.7]), phi.copy(), omega.copy(), 2, 298.15, 79.0)",
            "gold_call": "_oracle_log_doping_constants(np.array([-2.6, -0.7]), phi.copy(), omega.copy(), 2, 298.15, 79.0)",
            "tol": 1e-5,
        },
        # edge: concentrated monovalent salt at another permittivity
        {
            "setup": """import numpy as np
phi = np.array([0.098, 0.076])
omega = np.array([3.646, 2.836])
""",
            "call": "log_doping_constants(np.array([-1.5]), phi.copy(), omega.copy(), 1, 310.0, 67.0)",
            "gold_call": "_oracle_log_doping_constants(np.array([-1.5]), phi.copy(), omega.copy(), 1, 310.0, 67.0)",
            "tol": 1e-5,
        },
        # edge: trivalent cation with three binding modes and a small anion
        {
            "setup": """import numpy as np
phi = np.array([0.021, 0.0079])
omega = np.array([9.154, 1.12])
""",
            "call": "log_doping_constants(np.array([-3.0, -1.5, 0.4]), phi.copy(), omega.copy(), 3, 298.15, 79.0)",
            "gold_call": "_oracle_log_doping_constants(np.array([-3.0, -1.5, 0.4]), phi.copy(), omega.copy(), 3, 298.15, 79.0)",
            "tol": 1e-5,
        },
        # edge: almost no free ions, the screening wavenumber sits six orders of magnitude below the ion sizes
        {
            "setup": """import numpy as np
phi = np.array([1e-12, 3e-12])
omega = np.array([7.34, 2.84])
""",
            "call": "log_doping_constants(np.array([-1.0, 0.5]), phi.copy(), omega.copy(), 2, 298.15, 79.0)",
            "gold_call": "_oracle_log_doping_constants(np.array([-1.0, 0.5]), phi.copy(), omega.copy(), 2, 298.15, 79.0)",
            "tol": 1e-5,
        },
        # invalid: one free energy per mode is required
        {
            "setup": """import numpy as np
phi = np.array([0.0131, 0.0101])
omega = np.array([7.3410, 2.8363])
def run_model():
    try:
        log_doping_constants(np.array([-4.2]), phi.copy(), omega.copy(), 2, 298.15, 79.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_log_doping_constants(np.array([-4.2]), phi.copy(), omega.copy(), 2, 298.15, 79.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
