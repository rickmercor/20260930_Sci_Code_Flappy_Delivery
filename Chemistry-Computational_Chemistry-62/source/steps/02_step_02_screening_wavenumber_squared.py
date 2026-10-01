"""
Wavenumber-dependent squared inverse screening length of the free, Gaussian-smeared salt ions in units of the water size.

Wavenumber-dependent screening of the free salt ions.

The electrostatic correlations of the free ions are treated in a random-phase approximation in which
every ion carries a Gaussian charge cloud. All lengths are measured in units of the water size
l = v_w^(1/3) with v_w = 29.7 Angstrom^3, so the wavenumber q is dimensionless (q times l), and the
smearing size of an ion of effective volume omega (relative to water) is a = omega^(1/3). The squared
inverse screening length at wavenumber q, also in units of 1/l^2, is

    k2(q) = 4 pi (l_B / l) * [ (phi_M / omega_M) z^2 G(q, a_M) + (phi_A / omega_A) G(q, a_A) ]

with G(q, a) = exp(-q^2 a^2) the square of the Gaussian smearing factor exp(-q^2 a^2 / 2), z the
cation charge number (the anion is monovalent), phi the free-ion volume fractions and omega the
effective ion volumes. The Bjerrum length in Angstrom is

    l_B = e^2 / (4 pi eps0 eps_r k_B T) * 1e10

with e = 1.602176634e-19 C, eps0 = 8.8541878128e-12 F/m and k_B = 1.380649e-23 J/K.

Returns
-------
np.ndarray with the shape of q: squared inverse screening length k2(q) in units of 1/l^2
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def screening_wavenumber_squared(q: "np.ndarray", phi_free: "np.ndarray", omega: "np.ndarray", valence: int,
                                 temperature: float, dielectric: float) -> "np.ndarray":
    '''Squared inverse screening length k2(q) of Gaussian-smeared free ions, in units of 1/l^2.

    Parameters
    ----------
    q : np.ndarray
        Dimensionless wavenumbers (in units of 1/l).
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
        k2(q), same shape as q.

    Raises
    ------
    ValueError
        If phi_free or omega does not have shape (2,), an entry of phi_free is negative, an entry of
        omega is not positive, valence is not an integer >= 1, or temperature or dielectric is not
        positive.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _water_and_bjerrum(temperature: float, dielectric: float) -> tuple:
    import numpy as np
    e, eps0, k_b, v_w = 1.602176634e-19, 8.8541878128e-12, 1.380649e-23, 29.7
    l = v_w ** (1.0 / 3.0)
    l_b = e ** 2 / (4.0 * np.pi * eps0 * float(dielectric) * k_b * float(temperature)) * 1e10
    return l, l_b


def _oracle_screening_wavenumber_squared(q: "np.ndarray", phi_free: "np.ndarray", omega: "np.ndarray", valence: int,
                                         temperature: float, dielectric: float) -> "np.ndarray":
    import numpy as np
    phi_free = np.array(phi_free, dtype=float)
    omega = np.array(omega, dtype=float)
    if phi_free.shape != (2,) or omega.shape != (2,):
        raise ValueError("phi_free and omega must have shape (2,)")
    if np.any(phi_free < 0.0) or np.any(omega <= 0.0):
        raise ValueError("phi_free must be non-negative and omega positive")
    if isinstance(valence, bool) or not float(valence).is_integer() or int(valence) < 1:
        raise ValueError("valence must be an integer >= 1")
    if not (temperature > 0.0 and dielectric > 0.0):
        raise ValueError("temperature and dielectric must be positive")
    z = int(valence)
    l, l_b = _water_and_bjerrum(temperature, dielectric)
    a = omega ** (1.0 / 3.0)
    q = np.array(q, dtype=float)
    return 4.0 * np.pi * l_b / l * (phi_free[0] / omega[0] * z ** 2 * np.exp(-(q * a[0]) ** 2)
                                    + phi_free[1] / omega[1] * np.exp(-(q * a[1]) ** 2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal: divalent cation, grid across the smearing scale
        {
            "setup": """import numpy as np
q = np.linspace(0.0, 3.0, 13)
phi = np.array([0.0131, 0.0101])
omega = np.array([7.3410, 2.8363])
""",
            "call": "screening_wavenumber_squared(q.copy(), phi.copy(), omega.copy(), 2, 298.15, 79.0)",
            "gold_call": "_oracle_screening_wavenumber_squared(q.copy(), phi.copy(), omega.copy(), 2, 298.15, 79.0)",
            "tol": 1e-10,
        },
        # boundary: q = 0 and a monovalent salt at another temperature and permittivity
        {
            "setup": """import numpy as np
q = np.array([0.0, 0.05, 0.5])
phi = np.array([0.094, 0.077])
omega = np.array([3.646, 2.836])
""",
            "call": "screening_wavenumber_squared(q.copy(), phi.copy(), omega.copy(), 1, 310.0, 67.0)",
            "gold_call": "_oracle_screening_wavenumber_squared(q.copy(), phi.copy(), omega.copy(), 1, 310.0, 67.0)",
            "tol": 1e-10,
        },
        # edge: trivalent cation, large and negative wavenumbers, no free anions
        {
            "setup": """import numpy as np
q = np.array([-4.0, -0.7, 0.3, 8.0])
phi = np.array([0.0025, 0.0])
omega = np.array([9.154, 3.06])
""",
            "call": "screening_wavenumber_squared(q.copy(), phi.copy(), omega.copy(), 3, 280.0, 85.0)",
            "gold_call": "_oracle_screening_wavenumber_squared(q.copy(), phi.copy(), omega.copy(), 3, 280.0, 85.0)",
            "tol": 1e-10,
        },
        # invalid: non-positive effective volume
        {
            "setup": """import numpy as np
q = np.array([0.1, 1.0])
def run_model():
    try:
        screening_wavenumber_squared(q.copy(), np.array([0.01, 0.01]), np.array([0.0, 2.8]), 2, 298.15, 79.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_screening_wavenumber_squared(q.copy(), np.array([0.01, 0.01]), np.array([0.0, 2.8]), 2, 298.15, 79.0)
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
