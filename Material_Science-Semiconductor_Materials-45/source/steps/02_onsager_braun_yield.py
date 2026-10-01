"""
Evaluate the Onsager-Braun field-assisted dissociation yield of a bound electron-hole pair anchored to its zero-field value: with b_F = q^3 |F| / [8 pi eps eps0 (kT)^2] for the electric field F (V/m), f(F) = I_1(2 sqrt(2 b_F)) / sqrt(2 b_F) with I_1 the modified Bessel function of the first kind of order one (f(0) = 1, and f(F) = 1 + b_F + b_F^2/3 + b_F^3/18 + b_F^4/180 + ...), and P_gen(F) = P_gen^0 f(F) / [P_gen^0 f(F) + 1 - P_gen^0] for the zero-field yield P_gen^0. Return [P_gen(F), f(F), b_F]. Raise ValueError if the zero-field yield is not in (0, 1] or the permittivity or temperature is not positive.

In the Onsager-Braun picture a photogenerated charge-transfer pair dissociates with a probability that rises with the electric field, so a field dependence of the free-charge generation can mimic, in a photocurrent-versus-voltage curve, a voltage-dependent collection of already-free carriers.

Returns
-------
A (3,) float64 array [P_gen(F), f(F), b_F].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def onsager_braun_yield(field: float, zero_field_yield: float, permittivity: float, temperature: float) -> "np.ndarray":
    """Evaluate the Onsager-Braun field-assisted dissociation yield of a bound electron-hole
        pair anchored to its zero-field value: with b_F = q^3 |F| / [8 pi eps eps0 (kT)^2] for
        the electric field F (V/m), f(F) = I_1(2 sqrt(2 b_F)) / sqrt(2 b_F) with I_1 the
        modified Bessel function of the first kind of order one (f(0) = 1, and f(F) = 1 + b_F +
        b_F^2/3 + b_F^3/18 + b_F^4/180 + ...), and P_gen(F) = P_gen^0 f(F) / [P_gen^0 f(F) + 1 -
        P_gen^0] for the zero-field yield P_gen^0. A (3,) float64 array [P_gen(F), f(F), b_F].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _Q():
    return 1.602176634e-19


def _KB():
    return 1.380649e-23


def _E0():
    return 8.8541878128e-12


def _besseli1(x):
    """Modified Bessel function I1(x) by its power series."""
    x = float(x)
    term = x / 2.0
    s = term
    k = 0

    while k < 1000:
        k += 1
        term *= (x / 2.0) ** 2 / (k * (k + 1))
        s += term

        if term < 1e-17 * s:
            break

    return s


def _oracle_onsager_braun_yield(
    field: float,
    zero_field_yield: float,
    permittivity: float,
    temperature: float,
) -> "np.ndarray":
    """Onsager-Braun field-assisted dissociation yield anchored to its zero-field value."""
    F = abs(float(field))
    P0 = float(zero_field_yield)
    eps = float(permittivity)
    T = float(temperature)

    if not (0.0 < P0 <= 1.0) or eps <= 0 or T <= 0:
        raise ValueError(
            "the zero-field yield must lie in (0, 1], "
            "the permittivity and temperature must be positive"
        )

    bF = (
        _Q() ** 3
        * F
        / (8.0 * np.pi * eps * _E0() * (_KB() * T) ** 2)
    )

    if bF > 0.0:
        u = np.sqrt(2.0 * bF)
        f = _besseli1(2.0 * u) / u
    else:
        f = 1.0

    Pgen = P0 * f / (P0 * f + 1.0 - P0)

    return np.array([Pgen, f, bF], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\n',
         'call': 'onsager_braun_yield(9.0e6, 0.8, 3.5, 300.0)',
         'gold_call': '_oracle_onsager_braun_yield(9.0e6, 0.8, 3.5, 300.0)'},
        {'setup': 'import numpy as np\n',
         'call': 'onsager_braun_yield(9.0e6, 0.6, 3.5, 300.0)',
         'gold_call': '_oracle_onsager_braun_yield(9.0e6, 0.6, 3.5, 300.0)'},
        {'setup': 'import numpy as np\n',
         'call': 'onsager_braun_yield(0.0, 0.5, 3.0, 290.0)',
         'gold_call': '_oracle_onsager_braun_yield(0.0, 0.5, 3.0, 290.0)'},
        {'setup': 'import numpy as np\n',
         'call': 'onsager_braun_yield(-4.0e7, 0.3, 3.0, 250.0)',
         'gold_call': '_oracle_onsager_braun_yield(-4.0e7, 0.3, 3.0, 250.0)'},
    ]
