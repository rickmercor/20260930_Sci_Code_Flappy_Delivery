"""
Construct the fixed Hamiltonian and channel-coupling realization of an open optical cavity.

A Hermitian internal Hamiltonian models closed-cavity resonances, while a channel-coupling matrix opens the input and measured output ports. The intensity transmission obtained from this pair determines the noise amplification of spectral reconstruction. This task uses one deterministic realization, rather than an ensemble average.



For a mode count $$P$$, draw independent real standard-normal matrices $$X$$ and $$Y$$ of shape $$(P,P)$$ and $$V$$ of shape $$(P,M+1)$$, in that order, and define $$H=(X+X^{\mathsf T}+i(Y-Y^{\mathsf T}))/(2\sqrt P)$$ and $$W_0=V/\sqrt P$$. Channel zero is the input and the remaining $$M$$ channels are measured outputs. This normalization and seeded realization are benchmark conventions.

Returns
-------
np.ndarray: Complex array of shape (P, P + M + 1), containing H followed by W0 horizontally.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cavity_realization(n_modes: int, n_ports: int, seed: int) -> "np.ndarray":
    r"""Create one reproducible cavity in dimensionless frequency units.
    
    Parameters
    ----------
    n_modes : int
        Mode count P >= 2.
    n_ports : int
        Measured output count M >= 1 with M + 1 <= P.
    seed : int
        Nonnegative seed for numpy.random.default_rng. Draw X, then Y, then V
        using standard_normal with the shapes in the background; draw order
        defines this fixed realization, rather than a distribution-only test.
    
    Returns
    -------
    result : np.ndarray
        Complex array of shape (P, P + M + 1), containing H followed by W0
        horizontally. The final M + 1 columns are real-valued couplings.
    
    Notes
    -----
    Inputs satisfy the stated domain. No global random state is read or changed.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_cavity_realization(n_modes: int, n_ports: int, seed: int) -> "np.ndarray":
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_modes, n_modes))
    y = rng.standard_normal((n_modes, n_modes))
    v = rng.standard_normal((n_modes, n_ports + 1))
    h = (x + x.T + 1j*(y-y.T))/(2*np.sqrt(n_modes))
    return np.concatenate((h, v/np.sqrt(n_modes)), axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit differential cases for Studio contract validation."""
    return [
        {
            "setup": "import numpy as np\np=32\nm=6\nseed=17\ndef measure(fn):\n    value = np.asarray(fn(p,m,seed))\n    assert value.shape == (p,p+m+1)\n    return np.stack((value.real,value.imag))\n",
            "call": "measure(cavity_realization)",
            "gold_call": "measure(_oracle_cavity_realization)",
            "tol": 1e-11
        },
        {
            "setup": "import numpy as np\np=2\nm=1\nseed=0\ndef measure(fn):\n    value = np.asarray(fn(p,m,seed))\n    assert value.shape == (p,p+m+1)\n    return np.stack((value.real,value.imag))\n",
            "call": "measure(cavity_realization)",
            "gold_call": "measure(_oracle_cavity_realization)",
            "tol": 1e-11
        },
        {
            "setup": "import numpy as np\np=9\nm=8\nseed=41\ndef measure(fn):\n    value = np.asarray(fn(p,m,seed))\n    assert value.shape == (p,p+m+1)\n    return np.stack((value.real,value.imag))\n",
            "call": "measure(cavity_realization)",
            "gold_call": "measure(_oracle_cavity_realization)",
            "tol": 1e-11
        },
        {
            "setup": "import numpy as np\np=12\nm=3\nseed=8\ndef measure(fn):\n    value = np.asarray(fn(p,m,seed))\n    assert value.shape == (p,p+m+1)\n    return np.stack((value.real,value.imag))\n",
            "call": "measure(cavity_realization)",
            "gold_call": "measure(_oracle_cavity_realization)",
            "tol": 1e-11
        }
    ]
