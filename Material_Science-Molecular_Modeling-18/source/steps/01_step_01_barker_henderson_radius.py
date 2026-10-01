"""
Effective hard-sphere radius of a Mie segment at a given temperature.

A perturbation treatment of a soft repulsive fluid replaces the true segment by a hard sphere whose size is set by how far the repulsive branch of the pair potential actually excludes a neighbour at the temperature of interest. For a Mie segment that size is obtained from an integral of the Mayer factor of the potential taken across the repulsive core, and it shrinks as temperature rises. Every geometric quantity built later in the pipeline, the weight functions, the packing measures and the bonding volume, is fixed by the value returned here, so a factor of two taken in the wrong place propagates into every subsequent number. Evaluate the integral on a uniform grid of 20001 points spanning a lower limit of 1e-8 angstrom to the segment diameter with the trapezoidal rule; the integrand is numerically zero across the inner part of that range, so the lower limit only has to keep the potential finite. Lengths are in angstrom and the temperature in kelvin.

Returns
-------
float, the effective hard-sphere radius in angstrom
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def barker_henderson_radius(sigma: float, epsilon: float, lambda_a: float, lambda_r: float, temperature: float) -> float:
    '''Effective hard-sphere radius of a Mie segment.

    Parameters
    ----------
    sigma : float
        Mie segment diameter in angstrom.
    epsilon : float
        Mie potential well depth divided by the Boltzmann constant, in kelvin.
    lambda_a : float
        Attractive exponent of the Mie potential.
    lambda_r : float
        Repulsive exponent of the Mie potential.
    temperature : float
        Absolute temperature in kelvin.

    Returns
    -------
    float
        Effective hard-sphere radius in angstrom.

    Raises
    ------
    ValueError
        If sigma, epsilon or temperature is not positive, or if lambda_r <= lambda_a.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_barker_henderson_radius(sigma: float, epsilon: float, lambda_a: float, lambda_r: float, temperature: float) -> float:
    if sigma <= 0 or epsilon <= 0 or temperature <= 0:
        raise ValueError("sigma, epsilon and temperature must be positive")
    if lambda_r <= lambda_a:
        raise ValueError("lambda_r must exceed lambda_a")
    n = 20001
    r = np.linspace(1e-8, sigma, n)
    pref = lambda_r/(lambda_r - lambda_a)*(lambda_r/lambda_a)**(lambda_a/(lambda_r - lambda_a))
    with np.errstate(over='ignore'):
        phi = epsilon*pref*((sigma/r)**lambda_r - (sigma/r)**lambda_a)
    integrand = 1.0 - np.exp(-phi/temperature)
    return float(0.5*np.trapezoid(integrand, r))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "barker_henderson_radius(3.161, 488.75, 6.0, 52.367, 425.0)",
            "gold_call": "_oracle_barker_henderson_radius(3.161, 488.75, 6.0, 52.367, 425.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "barker_henderson_radius(3.161, 488.75, 6.0, 52.367, 300.0)",
            "gold_call": "_oracle_barker_henderson_radius(3.161, 488.75, 6.0, 52.367, 300.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "barker_henderson_radius(3.161, 488.75, 6.0, 52.367, 700.0)",
            "gold_call": "_oracle_barker_henderson_radius(3.161, 488.75, 6.0, 52.367, 700.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "barker_henderson_radius(3.740, 253.30, 6.0, 12.650, 425.0)",
            "gold_call": "_oracle_barker_henderson_radius(3.740, 253.30, 6.0, 12.650, 425.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "barker_henderson_radius(2.800, 900.00, 6.0, 20.000, 150.0)",
            "gold_call": "_oracle_barker_henderson_radius(2.800, 900.00, 6.0, 20.000, 150.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "barker_henderson_radius(3.161, 488.75, 6.0, 7.0, 425.0)",
            "gold_call": "_oracle_barker_henderson_radius(3.161, 488.75, 6.0, 7.0, 425.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "barker_henderson_radius(3.161, 488.75, 6.0, 52.367, 2000.0)",
            "gold_call": "_oracle_barker_henderson_radius(3.161, 488.75, 6.0, 52.367, 2000.0)",
        },
    ]
