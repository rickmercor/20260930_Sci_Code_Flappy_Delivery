"""
Evaluate the literature's underdetermined inverse-trace prediction for a fitted Lorentzian correlation.

The prediction is the primary literature's ensemble estimate of the expected whole-comb trace $$\mathrm{Tr}[(A^{\mathsf T}A)^+]$$ over all $$N$$ frequency channels, for a device with fewer measured ports than frequency channels ($$M<N$$) whose spectral correlations are Lorentzian. Use the source's underdetermined form, including its finite-window correction as the source fixes it. The throughput is the source's total transmittance from the input to all $$M$$ measured outputs, averaged over frequency, and the width is the correlation half-width at half maximum in units of the channel spacing. That width enters as the source's $$a=\Gamma_{\mathrm{corr}}/\Delta\omega$$ itself. Correction: the source's text calls $$\Gamma_{\mathrm{corr}}$$ a full width at half maximum, but its Lorentzian $$a^2/(k^2+a^2)$$ makes $$a$$ the half-width ratio.

Returns
-------
float: Finite positive inverse-Gram trace prediction in the stated convention.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def asymptotic_trace(normalized_width: float, throughput: float, n_ports: int, n_channels: int) -> float:
    r"""Compute the source model's trace prediction.
    
    Parameters
    ----------
    normalized_width : float
        Positive fitted Lorentzian half-width divided by channel spacing;
        0 < normalized_width <= 30.
    throughput : float
        Finite positive frequency-averaged total measured transmittance T0.
    n_ports : int
        Measured port count M >= 1.
    n_channels : int
        Frequency count N > M.
    
    Returns
    -------
    result : float
        Finite positive inverse-Gram trace prediction in the stated convention.
    
    Notes
    -----
    Inputs satisfy the stated domain. The source's closed form is evaluated
    throughout that domain, even where a particular cavity violates its
    statistical assumptions.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_asymptotic_trace(normalized_width: float, throughput: float, n_ports: int, n_channels: int) -> float:
    a = normalized_width
    beta = n_ports/n_channels
    j = 2*np.sinh(np.sqrt(2)*beta*np.pi*a)*np.arctan(np.tanh(np.pi*a/2))/(np.pi**2*a*a)
    return float(n_ports*n_ports*n_channels*j/((n_channels-n_ports)*throughput*throughput))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit differential cases for Studio contract validation."""
    return [
        {
            "setup": "args=(3.04, 0.79, 6, 48)",
            "call": "asymptotic_trace(*args)",
            "gold_call": "_oracle_asymptotic_trace(*args)",
            "tol": 1e-10
        },
        {
            "setup": "args=(0.05, 0.3, 6, 48)",
            "call": "asymptotic_trace(*args)",
            "gold_call": "_oracle_asymptotic_trace(*args)",
            "tol": 1e-10
        },
        {
            "setup": "args=(8.0, 0.7, 4, 60)",
            "call": "asymptotic_trace(*args)",
            "gold_call": "_oracle_asymptotic_trace(*args)",
            "tol": 1e-10
        },
        {
            "setup": "args=(1.2, 0.5, 19, 20)\n# Fixed output scaling: this configuration predicts a trace near 1e5, so both sides are divided by the same constant.\ndef measure(fn):\n    return fn(*args)/1e5\n",
            "call": "measure(asymptotic_trace)",
            "gold_call": "measure(_oracle_asymptotic_trace)",
            "tol": 1e-10
        },
        {
            "setup": "args=(3.0, 0.5, 40, 100)\n# Fixed output scaling: this configuration predicts a trace near 1e4, so both sides are divided by the same constant.\ndef measure(fn):\n    return fn(*args)/1e4\n",
            "call": "measure(asymptotic_trace)",
            "gold_call": "measure(_oracle_asymptotic_trace)",
            "tol": 1e-10
        }
    ]
