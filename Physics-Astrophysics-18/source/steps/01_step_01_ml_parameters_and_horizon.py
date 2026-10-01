"""
Convert the physical mass, dimensionless spin and deformation parameter into the metric's

integration constants and the constants of the thermodynamically normalised Killing frame,

and return the outer horizon radius alongside them.

The spacetime is a Ricci-flat rotating black hole carrying, besides mass and

rotation, a third integration constant B that deforms the geometry and makes

it non-asymptotically flat.  Its metric is written in terms of integration

constants (m, a, B), and m and a are not the physical mass and mass times

spin: with no canonical normalisation of the timelike Killing vector at

infinity, the conserved charges depend on a choice of Killing frame, fixed by

the first law of black-hole thermodynamics through two constants lam1 and

lam2 that every later step needs.



Writing M = mass, chi for the dimensionless spin and B = b, the source's

abbreviations are



    I1    = 1 - B^2 a^2 / 2

    I2    = 1 - B^2 a^2

    P0    = 2 I2 (I1^2 + B^2 m^2) / (I1^2 (1 + sqrt(I2)))

    Delta = (1 - B^2 m^2 I2 / I1^2) r^2 - 2 m (I2 / I1) r + a^2

    eps1  = sqrt(I2)

    eps2  = sqrt(1 + B^2 r_plus^2)

    gamma = sqrt((1 + eps1 + eps1^2 - eps1^3) eps2^2

                 + 2 (eps1 - 1) eps1 eps2 - 2 eps1)

    lam1  = (2 eps1 (eps1 + 1) (eps2^2 - 1) - gamma^2)

            / (gamma sqrt(eps2 (eps1 + 1) S))

    lam2  = (2 eps1 eps2 + 2 eps1 - (eps1 + 1)^2 eps2^2)

            * sqrt(eps2 (1 - eps1) S) / ((eps1 + 1) (eps2^2 - 1) gamma)



with S = eps1^2 eps2 - 2 eps1 + eps2, and the integration constants, obtained

by inverting the conserved-charge relations to O(B^2), are



    m = M + B^2 M^3 (6 - 5 chi^2) / 4

    a = M chi + B^2 M^3 chi (2 - chi^2) / 4



Conventions fixed for this step:

* r_plus is the larger root of Delta.

* At b = 0 the Killing-frame constants are 0/0; return their analytic

  limits gamma = 0.0, lam1 = 1.0, lam2 = 0.0 there.

Returns
-------
np.ndarray of shape (11,) of native floats, in the order [m, a, I1, I2, P0, eps1, eps2, gamma, lam1, lam2, r_plus]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def ml_parameters_and_horizon(mass: float, chi: float, b: float) -> np.ndarray:
    '''Derived constants and outer horizon radius of the spindle-deformed black hole.

    Parameters
    ----------
    mass : float
        Physical mass M of the black hole.
    chi : float
        Dimensionless spin J / M^2, with |chi| <= 1.
    b : float
        Spindle deformation parameter B, in units of inverse mass.

    Returns
    -------
    result : np.ndarray
        Array of shape (11,) of native floats, in the order
        [m, a, I1, I2, P0, eps1, eps2, gamma, lam1, lam2, r_plus].

    Raises
    ------
    ValueError
        If mass is not positive, |chi| exceeds 1, b is negative, or the
        parameter set admits no horizon.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_ml_parameters_and_horizon(mass: float, chi: float, b: float) -> np.ndarray:
    """Reference implementation."""
    if mass <= 0.0:
        raise ValueError("mass must be positive")
    if abs(chi) > 1.0:
        raise ValueError("|chi| must not exceed 1")
    if b < 0.0:
        raise ValueError("b must be non-negative")
    m = mass + b ** 2 * mass ** 3 * (6.0 - 5.0 * chi ** 2) / 4.0
    a = mass * chi + b ** 2 * mass ** 3 * chi * (2.0 - chi ** 2) / 4.0
    i1 = 1.0 - 0.5 * b * b * a * a
    i2 = 1.0 - b * b * a * a
    p0 = 2.0 * i2 * (i1 * i1 + b * b * m * m) / (i1 * i1 * (1.0 + np.sqrt(i2)))
    disc = 4.0 * i2 * (m * m - a * a) - a ** 6 * b ** 4
    if disc < 0.0:
        raise ValueError("no horizon for the given parameters")
    r_plus = ((1.0 + i2) * (2.0 * i2 * m + np.sqrt(disc))
              / (a ** 4 * b ** 4 + 4.0 * i2 * (1.0 - b * b * m * m)))
    eps1 = np.sqrt(i2)
    eps2 = np.sqrt(1.0 + b * b * r_plus * r_plus)
    if b == 0.0:
        gamma, lam1, lam2 = 0.0, 1.0, 0.0
    else:
        gamma = np.sqrt((1.0 + eps1 + eps1 ** 2 - eps1 ** 3) * eps2 ** 2
                        + 2.0 * (eps1 - 1.0) * eps1 * eps2 - 2.0 * eps1)
        s = eps1 * eps1 * eps2 - 2.0 * eps1 + eps2
        lam1 = ((2.0 * eps1 * (eps1 + 1.0) * (eps2 * eps2 - 1.0) - gamma * gamma)
                / (gamma * np.sqrt(eps2 * (eps1 + 1.0) * s)))
        lam2 = ((2.0 * eps1 * eps2 + 2.0 * eps1 - (eps1 + 1.0) ** 2 * eps2 * eps2)
                * np.sqrt(eps2 * (1.0 - eps1) * s)
                / ((eps1 + 1.0) * (eps2 * eps2 - 1.0) * gamma))
    return np.array([m, a, i1, i2, p0, eps1, eps2, gamma, lam1, lam2, r_plus], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    exception_setup = 'def _exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n'
    return [
        {   # normal: the production configuration
            "setup": 'import numpy as np',
            "call": 'ml_parameters_and_horizon(1.0, 0.94, 0.01)',
            "gold_call": '_oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)',
        },
        {   # boundary: b = 0, where the Killing-frame constants are 0/0 and the
            # analytic limits must be returned instead of nan
            "setup": 'import numpy as np',
            "call": 'ml_parameters_and_horizon(1.0, 0.94, 0.0)',
            "gold_call": '_oracle_ml_parameters_and_horizon(1.0, 0.94, 0.0)',
        },
        {   # edge: a ten-times larger deformation, where the O(B^2) shifts in m,
            # a and r_plus are no longer negligible
            "setup": 'import numpy as np',
            "call": 'ml_parameters_and_horizon(1.0, 0.5, 0.1)',
            "gold_call": '_oracle_ml_parameters_and_horizon(1.0, 0.5, 0.1)',
        },
        {   # invalid: non-positive mass
            "setup": 'import numpy as np\n' + exception_setup,
            "call": '_exception_code(ml_parameters_and_horizon, -1.0, 0.94, 0.01)',
            "gold_call": '_exception_code(_oracle_ml_parameters_and_horizon, -1.0, 0.94, 0.01)',
        },
    ]
