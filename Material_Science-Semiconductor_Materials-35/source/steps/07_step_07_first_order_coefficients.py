"""
Compute the first-order Poincare-Lindstedt quadrature, harmonic and frequency coefficients at one bias.

The circuit is the diode in parallel with a capacitance C, an inductance L and a load conductance G_l, all per unit diode area, with the ac voltage v about the bias obeying

    C dv/dt + G_l v + (1/L) integral(v dt) + i_D(v) = 0,   i_D(v) = I(Vdc + v) - I(Vdc).

Written for the scaled voltage v/dV and the phase tau = omega t, this becomes a generalised van der Pol equation whose right-hand side is proportional to the small parameter eps = G0/(C omega0), with omega0 = 1/sqrt(L C) and G0 the maximum negative differential conductance. A Poincare-Lindstedt expansion in eps, with the phase fixed by d(v/dV)/dtau = 0 at tau = 0 and the zero-order amplitude a0 set by the gain balance of step 04 (which eliminates G_l), gives

    v/dV = a0 cos(tau) + eps [ b1 sin(tau) + sum_{n=2}^{nmax} b1^(n) sin(n tau) ] + O(eps^2),
    omega = omega0 [ 1 + eps^2 kappa1 + O(eps^3) ].

The quadrature coefficient b1 and the harmonic coefficients b1^(n) follow from the order-eps equation and the phase condition, and the frequency coefficient kappa1 from the condition that the next-order correction stays periodic (no secular term). All three depend only on the bias, a0, dV, G0 and the dc characteristic, through the coefficients c_n of step 03 for n = 2 ... nmax, and not on C, L or the frequency. Every harmonic sum is truncated at nmax.

Returns
-------
numpy.ndarray [b1, b1^(2), ..., b1^(nmax), kappa1], dimensionless first-order coefficients
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def first_order_coefficients(Vdc: float, a0: float, dV: float, G0: float, iv_params: np.ndarray, nmax: int) -> np.ndarray:
    '''First-order Poincare-Lindstedt coefficients at one bias.

    Parameters
    ----------
    Vdc : float
        Operating bias in volts.
    a0 : float
        Zero-order amplitude in units of dV.
    dV : float
        Peak-to-valley voltage separation in volts.
    G0 : float
        Maximum negative differential conductance in mS/um^2.
    iv_params : numpy.ndarray
        The characteristic parameters [A1, V1, w1, A2, V2, w2, B] in mA/um^2,
        V, V, mA/um^2, V, V and mA/(um^2 V^3).
    nmax : int
        Highest harmonic retained.

    Returns
    -------
    coeffs : numpy.ndarray
        Array [b1, b1^(2), ..., b1^(nmax), kappa1] of length nmax + 1, all
        dimensionless, in the normalisation defined above.

    Raises
    ------
    ValueError
        If a0 <= 0, dV <= 0, G0 <= 0 or nmax < 2, or if iv_params is invalid
        as in step 01.
    '''
    return coeffs

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_first_order_coefficients(Vdc: float, a0: float, dV: float, G0: float, iv_params: np.ndarray, nmax: int) -> np.ndarray:
    """Reference implementation. Chains step 03."""
    if a0 <= 0.0 or dV <= 0.0 or G0 <= 0.0 or int(nmax) < 2:
        raise ValueError("need a0 > 0, dV > 0, G0 > 0 and nmax >= 2")
    nmax = int(nmax)
    c = _oracle_harmonic_coefficients(Vdc, a0, dV, iv_params, nmax)
    n = np.arange(2, nmax + 1, dtype=float)
    cn = c[2:]
    w = n**2/(n**2 - 1.0)
    b1 = np.sum(w*cn)/(dV*G0)
    bn = -(n/(n**2 - 1.0))*cn/(dV*G0)
    kappa1 = -np.sum(w*cn**2)/(2.0*(a0*dV*G0)**2)
    return np.concatenate(([b1], bn, [kappa1]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])",
            "call": "first_order_coefficients(0.44, 0.85, 0.32, 104.0, p, 12)",
            "gold_call": "_oracle_first_order_coefficients(0.44, 0.85, 0.32, 104.0, p, 12)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])",
            "call": "first_order_coefficients(0.64, 1.07, 0.32, 104.0, p, 12)",
            "gold_call": "_oracle_first_order_coefficients(0.64, 1.07, 0.32, 104.0, p, 12)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\np = np.array([6.2, 0.9, 0.3, 0.0, 1.60, 0.10, 0.8])",
            "call": "first_order_coefficients(1.1, 0.9, 0.48, 14.8, p, 2)",
            "gold_call": "_oracle_first_order_coefficients(1.1, 0.9, 0.48, 14.8, p, 2)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\np = np.array([24.0, 0.35, 0.09, 0.0, 1.10, 0.10, 15.0])",
            "call": "first_order_coefficients(0.44, 1.35, 0.188961, 221.0437575196, p, 12)",
            "gold_call": "_oracle_first_order_coefficients(0.44, 1.35, 0.188961, 221.0437575196, p, 12)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\np = np.array([20.0, 0.30, 0.07, 12.0, 0.95, 0.08, 9.0])",
            "call": "first_order_coefficients(0.40, 1.10, 0.162939, 241.7782759980, p, 16)",
            "gold_call": "_oracle_first_order_coefficients(0.40, 1.10, 0.162939, 241.7782759980, p, 16)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\np = np.array([18.0, 0.35, 0.18, 26.0, 1.05, 0.12, 6.0])",
            "call": "first_order_coefficients(0.60, 0.95, 0.346976, 81.6899045066, p, 8)",
            "gold_call": "_oracle_first_order_coefficients(0.60, 0.95, 0.346976, 81.6899045066, p, 8)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])\ndef run_model():\n    try:\n        first_order_coefficients(0.44, 0.0, 0.32, 104.0, p, 12); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try:\n        _oracle_first_order_coefficients(0.44, 0.0, 0.32, 104.0, p, 12); return 0\n    except ValueError: return 1\n    except Exception: return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
