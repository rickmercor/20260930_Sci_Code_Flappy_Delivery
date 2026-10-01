"""
Compute the large-signal diode capacitance seen by the resonator under the zero-order swing.

The diode also contributes a bias-dependent capacitance. Its small-signal value is modelled as

    C0(V) = Cb + Cq exp(-((V - Vq) / wq)^2),

with cap_params = [Cb, Cq, Vq, wq] in fF/um^2, fF/um^2, V and V: a constant background plus a peak from the charge stored in the quantum well in the negative-differential-conductance region. Under the zero-order swing v = a0 dV cos(tau) the diode charge follows C0 at every instant, so what the resonator sees at the oscillation frequency is an effective large-signal capacitance C_LS, defined as the capacitance of the linear capacitor that, driven by the same swing, carries the same fundamental-frequency displacement current as the diode. With no swing it reduces to C0(Vdc).

Returns
-------
float, the large-signal diode capacitance C_LS in fF/um^2
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def large_signal_capacitance(Vdc: float, a0: float, dV: float, cap_params: np.ndarray) -> float:
    '''Large-signal diode capacitance under the zero-order swing.

    Parameters
    ----------
    Vdc : float
        Operating bias in volts.
    a0 : float
        Zero-order amplitude in units of dV (a0 = 0 means no swing).
    dV : float
        Peak-to-valley voltage separation in volts.
    cap_params : numpy.ndarray
        Small-signal capacitance parameters [Cb, Cq, Vq, wq] in fF/um^2,
        fF/um^2, V and V.

    Returns
    -------
    C_LS : float
        Large-signal capacitance in fF/um^2.

    Raises
    ------
    ValueError
        If a0 < 0 or dV <= 0, or if cap_params does not hold exactly four
        values or its width wq is not positive.
    '''
    return C_LS

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_large_signal_capacitance(Vdc: float, a0: float, dV: float, cap_params: np.ndarray) -> float:
    """Reference implementation."""
    q = np.asarray(cap_params, dtype=float).ravel()
    if a0 < 0.0 or dV <= 0.0 or q.size != 4 or not q[3] > 0.0:
        raise ValueError("need a0 >= 0, dV > 0 and cap_params = [Cb, Cq, Vq, wq] with wq > 0")
    Cb, Cq, Vq, wq = q
    M = 2048
    t = (np.arange(M) + 0.5)*np.pi/M
    C0 = Cb + Cq*np.exp(-((Vdc + a0*dV*np.cos(t) - Vq)/wq)**2)
    return float(2.0*np.mean(C0*np.sin(t)**2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nq = np.array([7.0, 4.5, 0.40, 0.12])",
            "call": "large_signal_capacitance(0.44, 0.85, 0.32, q)",
            "gold_call": "_oracle_large_signal_capacitance(0.44, 0.85, 0.32, q)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nq = np.array([7.0, 4.5, 0.40, 0.12])",
            "call": "large_signal_capacitance(0.64, 1.07, 0.32, q)",
            "gold_call": "_oracle_large_signal_capacitance(0.64, 1.07, 0.32, q)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nq = np.array([5.0, 2.0, 0.85, 0.2])",
            "call": "large_signal_capacitance(0.95, 0.0, 0.48, q)",
            "gold_call": "_oracle_large_signal_capacitance(0.95, 0.0, 0.48, q)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nq = np.array([7.0, 4.5, 0.40, 0.06])",
            "call": "large_signal_capacitance(0.44, 1.60, 0.19, q)",
            "gold_call": "_oracle_large_signal_capacitance(0.44, 1.60, 0.19, q)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nq = np.array([5.5, 8.0, 0.55, 0.08])",
            "call": "large_signal_capacitance(0.40, 2.40, 0.163, q)",
            "gold_call": "_oracle_large_signal_capacitance(0.40, 2.40, 0.163, q)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nq = np.array([9.0, 0.0, 0.40, 0.12])",
            "call": "large_signal_capacitance(0.50, 1.00, 0.32, q)",
            "gold_call": "_oracle_large_signal_capacitance(0.50, 1.00, 0.32, q)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nq = np.array([7.0, 4.5, 0.40, 0.0])\ndef run_model():\n    try:\n        large_signal_capacitance(0.44, 0.85, 0.32, q); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try:\n        _oracle_large_signal_capacitance(0.44, 0.85, 0.32, q); return 0\n    except ValueError: return 1\n    except Exception: return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
