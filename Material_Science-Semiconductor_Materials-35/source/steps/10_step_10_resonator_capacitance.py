"""
Recover the resonator capacitance, its inductance and the quantum-well capacitance peak from a dc current and three bias-dependent oscillation frequencies (final orchestrator).

The last step recovers the capacitance that the antenna and resonator add in parallel with the diode, using only measurements made on the running oscillator. At bias[0] the oscillating diode draws the dc current density J_dc and oscillates at freq[0]; at bias[1] it oscillates at freq[1]; and when the bias is raised to the upper edge of the oscillation region, the last frequency seen before the oscillation collapses is freq[2] (all frequencies in GHz). The bias at that edge is not measured: it follows from the load. Four quantities are unknown and none of them changes with bias: the load conductance G_l, the resonator capacitance C_r, its inductance-area product L, and the height Cq of the quantum-well peak in the diode's small-signal capacitance, whose background Cb and position and width Vq, wq are given in cap_fixed.

At each bias the oscillator is described by the first-order expansion of the earlier steps: the zero-order amplitude obeys the gain balance with G_l (the stable, largest root, and the turning-point amplitude at the edge), the dc current of the oscillating diode is I(Vdc) + c_0(a0) (the smallest amplitude that reproduces a measured value), the diode contributes its large-signal capacitance built from [Cb, Cq, Vq, wq], and the frequency follows from the total capacitance C_r + C_LS and from L. The values of G_l, Cq >= 0, C_r >= 0 and L > 0 for which this description reproduces J_dc and all three measured frequencies are unique, and the result is accepted only if the applicability criterion is at or below 0.3 at every bias. Harmonics up to nmax are retained, and the characteristic is that of step 01.

Returns
-------
float, the resonator capacitance C_r per unit diode area in fF/um^2
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def resonator_capacitance(iv_params: np.ndarray, cap_fixed: np.ndarray, bias: np.ndarray, J_dc: float, freq: np.ndarray, nmax: int) -> float:
    '''Resonator capacitance recovered from bias-dependent oscillator data.

    Parameters
    ----------
    iv_params : numpy.ndarray
        The characteristic parameters [A1, V1, w1, A2, V2, w2, B] in mA/um^2,
        V, V, mA/um^2, V, V and mA/(um^2 V^3).
    cap_fixed : numpy.ndarray
        The known part of the small-signal capacitance model, [Cb, Vq, wq] in
        fF/um^2, V and V; the peak height Cq is unknown.
    bias : numpy.ndarray
        The two measured biases [V1, V2] in volts; the dc current is measured
        at V1. The third measurement is at the upper edge of the oscillation
        region, whose bias follows from the load.
    J_dc : float
        dc current density of the oscillating diode at V1 in mA/um^2.
    freq : numpy.ndarray
        Measured oscillation frequencies [f1, f2, f_edge] in GHz at V1, at V2
        and at the upper edge of the oscillation region.
    nmax : int
        Highest harmonic retained.

    Returns
    -------
    Cr : float
        Resonator capacitance per unit diode area in fF/um^2.

    Raises
    ------
    ValueError
        If bias does not hold two values, freq does not hold three, a frequency
        is not positive or nmax < 2; if the dc current implies a negative load
        conductance; if a bias supports no oscillation at that load; if no
        Cq >= 0, C_r >= 0 and L > 0 reproduce the three frequencies; if the
        applicability criterion exceeds 0.3 at any bias; or if an earlier step
        raises.
    '''
    return Cr

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _oracle_resonator_capacitance(iv_params: np.ndarray, cap_fixed: np.ndarray, bias: np.ndarray, J_dc: float, freq: np.ndarray, nmax: int) -> float:
    """Reference implementation. Chains steps 02 to 09."""
    V = np.asarray(bias, dtype=float).ravel()
    F = np.asarray(freq, dtype=float).ravel()
    q = np.asarray(cap_fixed, dtype=float).ravel()
    if V.size != 2 or F.size != 3 or np.any(F <= 0.0) or int(nmax) < 2 or q.size != 3:
        raise ValueError("need two biases, three positive frequencies, cap_fixed = [Cb, Vq, wq] and nmax >= 2")
    nmax = int(nmax)
    dV, G0 = _oracle_characteristic_scales(iv_params)[2:]
    a_first = _oracle_rectified_amplitude(V[0], J_dc, dV, iv_params)
    Gl = -_oracle_harmonic_coefficients(V[0], a_first, dV, iv_params, 1)[1]/(a_first*dV)
    if Gl < 0.0:
        raise ValueError("the measured dc current implies a negative load conductance")
    edge = _oracle_oscillation_edge(Gl, dV, iv_params)
    V = np.array([V[0], V[1], edge[0]])
    amps = [a_first, _oracle_oscillation_amplitude(V[1], Gl, dV, iv_params), float(edge[1])]
    if min(amps) <= 0.0:
        raise ValueError("a bias supports no oscillation at this load")
    coeffs = [_oracle_first_order_coefficients(v, a, dV, G0, iv_params, nmax) for v, a in zip(V, amps)]

    def _caps(Cq):
        pars = np.array([q[0], Cq, q[1], q[2]])
        return [_oracle_large_signal_capacitance(v, a, dV, pars) for v, a in zip(V, amps)]

    def _ind(Cr, cls0):
        g = lambda u: _oracle_bias_frequency(Cr, np.exp(u), cls0, coeffs[0], G0)[0] - F[0]
        return float(np.exp(brentq(g, np.log(1e-6), np.log(1e6), xtol=1e-14, rtol=1e-15)))

    def _pair(Cq):
        """(Cr, L) reproducing the first and last frequency for this Cq, or None."""
        cls = _caps(Cq)
        g = lambda Cr: _oracle_bias_frequency(Cr, _ind(Cr, cls[0]), cls[2], coeffs[2], G0)[0] - F[2]
        if g(0.0)*g(1e4) > 0.0:
            return None
        Cr = float(brentq(g, 0.0, 1e4, xtol=1e-12, rtol=1e-15))
        return Cr, _ind(Cr, cls[0]), cls

    def _middle(Cq):
        got = _pair(Cq)
        if got is None:
            return np.nan
        Cr, ind, cls = got
        return _oracle_bias_frequency(Cr, ind, cls[1], coeffs[1], G0)[0] - F[1]

    xs = np.linspace(0.02, 30.0, 60)
    vs = np.array([_middle(x) for x in xs])
    hits = [k for k in range(xs.size - 1)
            if np.isfinite(vs[k]) and np.isfinite(vs[k + 1]) and vs[k]*vs[k + 1] <= 0.0]
    if not hits:
        raise ValueError("no Cq >= 0 with C_r >= 0 and L > 0 reproduces the three frequencies")
    k = hits[0]
    Cq = float(brentq(_middle, xs[k], xs[k + 1], xtol=1e-12, rtol=1e-15))
    Cr, ind, cls = _pair(Cq)
    crit = max(_oracle_bias_frequency(Cr, ind, c, co, G0)[2] for c, co in zip(cls, coeffs))
    if crit > 0.3:
        raise ValueError("the expansion is not applicable at the recovered resonator")
    return Cr

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])\nq = np.array([7.0, 0.40, 0.12])",
            "call": "resonator_capacitance(p, q, np.array([0.44, 0.54]), 12.6, np.array([640.0, 664.0, 666.0]), 12)",
            "gold_call": "_oracle_resonator_capacitance(p, q, np.array([0.44, 0.54]), 12.6, np.array([640.0, 664.0, 666.0]), 12)",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])\nq = np.array([6.0, 0.42, 0.15])",
            "call": "resonator_capacitance(p, q, np.array([0.46, 0.56]), 11.0, np.array([700.0, 716.0, 718.0]), 8)",
            "gold_call": "_oracle_resonator_capacitance(p, q, np.array([0.46, 0.56]), 11.0, np.array([700.0, 716.0, 718.0]), 8)",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])\nq = np.array([7.0, 0.40, 0.12])",
            "call": "resonator_capacitance(p, q, np.array([0.44, 0.54]), 12.8, np.array([596.0, 626.0, 641.0]), 12)",
            "gold_call": "_oracle_resonator_capacitance(p, q, np.array([0.44, 0.54]), 12.8, np.array([596.0, 626.0, 641.0]), 12)",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([20.0, 0.30, 0.07, 12.0, 0.95, 0.08, 9.0])\nq = np.array([7.0, 0.36, 0.10])",
            "call": "resonator_capacitance(p, q, np.array([0.36, 0.44]), 7.711851, np.array([592.0156, 661.5101, 716.179]), 12)",
            "gold_call": "_oracle_resonator_capacitance(p, q, np.array([0.36, 0.44]), 7.711851, np.array([592.0156, 661.5101, 716.179]), 12)",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([6.2, 0.9, 0.3, 0.0, 1.60, 0.10, 0.8])\nq = np.array([5.0, 1.00, 0.25])",
            "call": "resonator_capacitance(p, q, np.array([1.00, 1.20]), 4.446915, np.array([204.7886, 212.7705, 214.5667]), 12)",
            "gold_call": "_oracle_resonator_capacitance(p, q, np.array([1.00, 1.20]), 4.446915, np.array([204.7886, 212.7705, 214.5667]), 12)",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])\nq = np.array([7.0, 0.40, 0.12])\ndef run_model():\n    try:\n        resonator_capacitance(p, q, np.array([0.44, 0.54]), 12.8, np.array([600.0, 660.0, 680.0]), 12); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try:\n        _oracle_resonator_capacitance(p, q, np.array([0.44, 0.54]), 12.8, np.array([600.0, 660.0, 680.0]), 12); return 0\n    except ValueError: return 1\n    except Exception: return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])\nq = np.array([7.0, 0.40, 0.12])\ndef run_model():\n    try:\n        resonator_capacitance(p, q, np.array([0.44, 0.54]), 12.8, np.array([648.0, 640.0, 630.0]), 12); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try:\n        _oracle_resonator_capacitance(p, q, np.array([0.44, 0.54]), 12.8, np.array([648.0, 640.0, 630.0]), 12); return 0\n    except ValueError: return 1\n    except Exception: return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
