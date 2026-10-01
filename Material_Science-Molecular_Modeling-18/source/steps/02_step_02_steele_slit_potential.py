"""
Reduced external field of a structureless carbon slit pore.

A graphitic wall built from evenly spaced planes of carbon exerts a layered dispersion field on an adsorbed segment that is conventionally summed analytically over the planes, leaving a three term expression in the distance to the surface. The two walls of a slit act independently and their fields add, so a segment near one wall still feels the tail of the other. Solid-fluid size and energy parameters follow the usual geometric and arithmetic combining rules. Return the field already divided by the thermal energy, so the result is dimensionless, and let it diverge where a segment centre cannot go. Distances are in angstrom, energies as temperatures in kelvin, and the solid density in reciprocal cubic angstrom.

Returns
-------
numpy.ndarray, the reduced external potential on the grid, positive infinity where a segment centre is excluded
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def steele_slit_potential(z: "np.ndarray", pore_width: float, sigma_f: float, epsilon_f: float, sigma_s: float, epsilon_s: float, rho_s: float, delta_s: float, alpha: float, temperature: float) -> "np.ndarray":
    '''Reduced external potential of a carbon slit pore on its grid.

    Parameters
    ----------
    z : numpy.ndarray
        Distances from the first wall in angstrom, spanning the pore.
    pore_width : float
        Wall to wall separation in angstrom.
    sigma_f, epsilon_f : float
        Fluid segment diameter in angstrom and well depth in kelvin.
    sigma_s, epsilon_s : float
        Solid site diameter in angstrom and well depth in kelvin.
    rho_s : float
        Solid number density in angstrom^-3.
    delta_s : float
        Interlayer spacing of the solid in angstrom.
    alpha : float
        Adjustable coefficient of the third term.
    temperature : float
        Absolute temperature in kelvin.

    Returns
    -------
    numpy.ndarray
        Reduced external potential on the grid, positive infinity where a segment centre is
        excluded.

    Raises
    ------
    ValueError
        If pore_width or temperature is not positive.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_steele_slit_potential(z: "np.ndarray", pore_width: float, sigma_f: float, epsilon_f: float, sigma_s: float, epsilon_s: float, rho_s: float, delta_s: float, alpha: float, temperature: float) -> "np.ndarray":
    if pore_width <= 0 or temperature <= 0:
        raise ValueError("pore_width and temperature must be positive")
    z = np.asarray(z, dtype=float)
    s_c = 0.5*(sigma_f + sigma_s)
    e_c = np.sqrt(epsilon_f*epsilon_s)

    def _one_wall(d):
        out = np.full(d.shape, np.inf)
        m = d > 1e-9
        dd = d[m]
        with np.errstate(over='ignore'):
            out[m] = 2.0*np.pi*rho_s*e_c*s_c**2*delta_s*(
                0.4*(s_c/dd)**10 - (s_c/dd)**4
                - s_c**4/(3.0*delta_s*(dd + alpha*delta_s)**3))
        return out

    return (_one_wall(z) + _one_wall(pore_width - z))/temperature

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nZ = np.linspace(0.0, 14.0, 1401)",
            "call": "steele_slit_potential(Z, 14.0, 3.161, 488.75, 3.4, 28.0, 0.114, 3.35, 0.61, 425.0)",
            "gold_call": "_oracle_steele_slit_potential(Z, 14.0, 3.161, 488.75, 3.4, 28.0, 0.114, 3.35, 0.61, 425.0)",
        },
        {
            "setup": "import numpy as np\nZ = np.linspace(0.0, 8.0, 801)",
            "call": "steele_slit_potential(Z, 8.0, 3.161, 488.75, 3.4, 28.0, 0.114, 3.35, 0.61, 425.0)",
            "gold_call": "_oracle_steele_slit_potential(Z, 8.0, 3.161, 488.75, 3.4, 28.0, 0.114, 3.35, 0.61, 425.0)",
        },
        {
            "setup": "import numpy as np\nZ = np.linspace(0.0, 30.0, 601)",
            "call": "steele_slit_potential(Z, 30.0, 3.161, 488.75, 3.4, 28.0, 0.114, 3.35, 0.61, 700.0)",
            "gold_call": "_oracle_steele_slit_potential(Z, 30.0, 3.161, 488.75, 3.4, 28.0, 0.114, 3.35, 0.61, 700.0)",
        },
        {
            "setup": "import numpy as np\nZ = np.array([7.0])",
            "call": "steele_slit_potential(Z, 14.0, 3.161, 488.75, 3.4, 28.0, 0.114, 3.35, 0.61, 425.0)",
            "gold_call": "_oracle_steele_slit_potential(Z, 14.0, 3.161, 488.75, 3.4, 28.0, 0.114, 3.35, 0.61, 425.0)",
        },
        {
            "setup": "import numpy as np\nZ = np.array([0.0, 14.0])",
            "call": "steele_slit_potential(Z, 14.0, 3.161, 488.75, 3.4, 28.0, 0.114, 3.35, 0.61, 425.0)",
            "gold_call": "_oracle_steele_slit_potential(Z, 14.0, 3.161, 488.75, 3.4, 28.0, 0.114, 3.35, 0.61, 425.0)",
        },
    ]
