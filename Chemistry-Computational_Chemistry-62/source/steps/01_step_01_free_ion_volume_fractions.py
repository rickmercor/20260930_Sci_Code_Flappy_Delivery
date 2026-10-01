"""
Effective volumes of the salt ions relative to water and the volume fractions of the ions that remain free in the bath once the complex has taken up salt.

Free salt ions around a doped polyelectrolyte complex.

A symmetric polyelectrolyte complex is immersed in a solution of a z:1 salt M(z+) A(-)_z. Doping
binds salt ions to polyion repeat units, so the salt left free in solution is reduced by the amount
taken up by the complex:

    C_S_free = C_S - C_P0 * xi_PS

where C_S is the salt concentration (mol/L), C_P0 the polyelectrolyte repeat-unit concentration of
the solution before phase separation (mol/L) and xi_PS the total fraction of polyion repeat units in
extrinsic (salt-bound) pairs. Each formula unit of salt releases one cation and z anions.

Every ion is given a size relative to a water molecule. With the bare ionic radius r (Angstrom),
the hydration number n_h and the water molecular volume v_w = 29.7 Angstrom^3,

    omega = (4/3) pi r^3 / v_w + n_h

and the volume fraction of a free ion species of molar concentration C is phi = C * omega / C_W with
the molarity of water C_W = 55.56 mol/L.

Returns
-------
np.ndarray of shape (4,): [omega_cation, omega_anion, phi_cation_free, phi_anion_free]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def free_ion_volume_fractions(salt_conc: float, xi_total: float, polymer_conc: float, valence: int,
                              ions: "np.ndarray") -> "np.ndarray":
    '''Effective ion volumes and volume fractions of the free salt ions.

    Parameters
    ----------
    salt_conc : float
        Salt concentration C_S of the bathing solution in mol/L.
    xi_total : float
        Total fraction xi_PS of polyion repeat units in salt-bound pairs.
    polymer_conc : float
        Repeat-unit concentration C_P0 of the polyelectrolyte solution before phase separation, mol/L.
    valence : int
        Charge number z of the cation; each salt unit carries one cation and z anions.
    ions : np.ndarray
        Shape (2, 2): row 0 cation, row 1 anion; column 0 bare radius (Angstrom), column 1
        hydration number.

    Returns
    -------
    result : np.ndarray
        Float array [omega_cation, omega_anion, phi_cation_free, phi_anion_free].

    Raises
    ------
    ValueError
        If valence is not an integer >= 1, ions is not a (2, 2) array, a radius is not positive, a
        hydration number is negative, salt_conc or polymer_conc is not positive, xi_total is outside
        [0, 1), or the free salt concentration salt_conc - polymer_conc * xi_total is not positive.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_free_ion_volume_fractions(salt_conc: float, xi_total: float, polymer_conc: float, valence: int,
                                      ions: "np.ndarray") -> "np.ndarray":
    import numpy as np
    if isinstance(valence, bool) or not float(valence).is_integer() or int(valence) < 1:
        raise ValueError("valence must be an integer >= 1")
    z = int(valence)
    ions = np.array(ions, dtype=float)
    if ions.shape != (2, 2):
        raise ValueError("ions must have shape (2, 2)")
    if np.any(ions[:, 0] <= 0.0) or np.any(ions[:, 1] < 0.0):
        raise ValueError("radii must be positive and hydration numbers non-negative")
    if not (salt_conc > 0.0 and polymer_conc > 0.0):
        raise ValueError("concentrations must be positive")
    if not (0.0 <= xi_total < 1.0):
        raise ValueError("xi_total must lie in [0, 1)")
    free = float(salt_conc) - float(polymer_conc) * float(xi_total)
    if free <= 0.0:
        raise ValueError("free salt concentration must be positive")
    v_w, c_w = 29.7, 55.56
    omega = 4.0 / 3.0 * np.pi * ions[:, 0] ** 3 / v_w + ions[:, 1]
    return np.array([omega[0], omega[1], free * omega[0] / c_w, z * free * omega[1] / c_w])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal: CaCl2 at 0.10 M with a doped complex
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.81, 2.0]])
""",
            "call": "free_ion_volume_fractions(0.10, 0.07, 0.05, 2, ions.copy())",
            "gold_call": "_oracle_free_ion_volume_fractions(0.10, 0.07, 0.05, 2, ions.copy())",
            "tol": 1e-10,
        },
        # boundary: monovalent salt, undoped complex
        {
            "setup": """import numpy as np
ions = np.array([[1.02, 3.5], [1.81, 2.0]])
""",
            "call": "free_ion_volume_fractions(1.5, 0.0, 0.05, 1, ions.copy())",
            "gold_call": "_oracle_free_ion_volume_fractions(1.5, 0.0, 0.05, 1, ions.copy())",
            "tol": 1e-10,
        },
        # edge: trivalent cation with the salt almost exhausted by doping
        {
            "setup": """import numpy as np
ions = np.array([[1.03, 9.0], [1.96, 1.8]])
""",
            "call": "free_ion_volume_fractions(0.0201, 0.4, 0.05, 3, ions.copy())",
            "gold_call": "_oracle_free_ion_volume_fractions(0.0201, 0.4, 0.05, 3, ions.copy())",
            "tol": 1e-10,
        },
        # edge: small bare anion with no hydration shell
        {
            "setup": """import numpy as np
ions = np.array([[0.72, 10.0], [0.5, 0.0]])
""",
            "call": "free_ion_volume_fractions(0.03, 0.2, 0.1, 2, ions.copy())",
            "gold_call": "_oracle_free_ion_volume_fractions(0.03, 0.2, 0.1, 2, ions.copy())",
            "tol": 1e-10,
        },
        # invalid: doping would consume more salt than is present
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.81, 2.0]])
def run_model():
    try:
        free_ion_volume_fractions(0.01, 0.3, 0.05, 2, ions.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_free_ion_volume_fractions(0.01, 0.3, 0.05, 2, ions.copy())
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
