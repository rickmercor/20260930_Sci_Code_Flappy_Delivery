"""
Evaluate the first-order oscillation frequency, the small parameter and the applicability criterion for a given resonator.

At a given bias the resonator sees the total capacitance C = C_r + C_LS, where C_r is the bias-independent capacitance that the antenna and resonator place in parallel with the diode and C_LS is the diode's large-signal capacitance from step 08, both per unit diode area. With the inductance-area product L of the resonator, the eigenfrequency is omega0 = 1/sqrt(L C), the small parameter is eps = G0/(C omega0), and to first order the oscillator runs at

    omega = omega0 (1 + eps^2 kappa1),

with kappa1 the frequency coefficient of step 07. The expansion is trusted only while every first-order term stays small, so the applicability criterion is the largest of |eps b1|, |eps b1^(n)| over the retained harmonics and |eps^2 kappa1|; the expansion is accepted when this number is at or below 0.3. Units: C_r and C_LS in fF/um^2, L in pH um^2, G0 in mS/um^2, and the frequency in GHz.

Returns
-------
numpy.ndarray [f, eps, crit]: oscillation frequency in GHz, small parameter and applicability criterion
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def bias_frequency(Cr: float, ind: float, c_ls: float, coeffs: np.ndarray, G0: float) -> np.ndarray:
    '''First-order oscillation frequency, small parameter and applicability criterion.

    Parameters
    ----------
    Cr : float
        Resonator capacitance per unit diode area in fF/um^2.
    ind : float
        Inductance-area product of the resonator in pH um^2.
    c_ls : float
        Large-signal diode capacitance in fF/um^2.
    coeffs : numpy.ndarray
        First-order coefficients [b1, b1^(2), ..., b1^(nmax), kappa1] from
        step 07.
    G0 : float
        Maximum negative differential conductance in mS/um^2.

    Returns
    -------
    out : numpy.ndarray
        Array [f, eps, crit]: the oscillation frequency omega/(2 pi) in GHz,
        the small parameter eps and the applicability criterion.

    Raises
    ------
    ValueError
        If Cr < 0, c_ls <= 0, ind <= 0 or G0 <= 0, or if coeffs has fewer than
        three entries.
    '''
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_bias_frequency(Cr: float, ind: float, c_ls: float, coeffs: np.ndarray, G0: float) -> np.ndarray:
    """Reference implementation."""
    co = np.asarray(coeffs, dtype=float).ravel()
    if Cr < 0.0 or c_ls <= 0.0 or ind <= 0.0 or G0 <= 0.0 or co.size < 3:
        raise ValueError("need Cr >= 0, c_ls > 0, ind > 0, G0 > 0 and at least three coefficients")
    C = (Cr + c_ls)*1e-15
    w0 = 1.0/np.sqrt(ind*1e-12*C)
    eps = G0*1e-3/(C*w0)
    kappa1 = co[-1]
    f = w0*(1.0 + eps**2*kappa1)/(2.0*np.pi)*1e-9
    crit = max(abs(eps*co[0]), float(np.max(np.abs(eps*co[1:-1]))), abs(eps**2*kappa1))
    return np.array([f, eps, crit])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nco = np.array([-0.10, 0.13, -0.05, -0.007, 0.004, -0.05])",
            "call": "bias_frequency(17.0, 2.1, 9.0, co, 104.0)",
            "gold_call": "_oracle_bias_frequency(17.0, 2.1, 9.0, co, 104.0)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nco = np.array([-0.26, 0.13, -0.02, -0.03])",
            "call": "bias_frequency(0.0, 4.0, 8.0, co, 104.0)",
            "gold_call": "_oracle_bias_frequency(0.0, 4.0, 8.0, co, 104.0)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nco = np.array([0.08, -0.21, 0.11, -0.012])",
            "call": "bias_frequency(40.0, 25.0, 6.0, co, 14.8)",
            "gold_call": "_oracle_bias_frequency(40.0, 25.0, 6.0, co, 14.8)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nco = np.array([-0.31, 0.24, -0.17, 0.09, -0.041])",
            "call": "bias_frequency(3.0, 1.2, 12.0, co, 240.0)",
            "gold_call": "_oracle_bias_frequency(3.0, 1.2, 12.0, co, 240.0)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nco = np.array([0.02, -0.01, 0.004, -0.0007])",
            "call": "bias_frequency(60.0, 8.0, 5.0, co, 14.8)",
            "gold_call": "_oracle_bias_frequency(60.0, 8.0, 5.0, co, 14.8)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nco = np.array([-0.10, 0.13, -0.05])\ndef run_model():\n    try:\n        bias_frequency(17.0, 0.0, 9.0, co, 104.0); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try:\n        _oracle_bias_frequency(17.0, 0.0, 9.0, co, 104.0); return 0\n    except ValueError: return 1\n    except Exception: return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
