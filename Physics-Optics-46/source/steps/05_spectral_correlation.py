"""
Measure throughput and a lag-dependent spectral correlation from a finite transmission matrix.

For each frequency channel subtract its mean over measured ports, then normalize that centered column to unit Euclidean norm. The correlation at lag $$k$$ is the average dot product of all normalized column pairs separated by $$k$$ channels, using $$N-k$$ pairs.

The total transmittance $$T_0$$ is the average, over frequency channels, of the sum of measured intensities.

Returns
-------
np.ndarray: Real vector of length L + 2, [T0, rho_0, rho_1, ..., rho_L].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectral_correlation(transmission: "np.ndarray", max_lag: int) -> "np.ndarray":
    r"""Return throughput and the normalized finite-record correlation profile.
    
    Parameters
    ----------
    transmission : np.ndarray
        Finite nonnegative real matrix A, shape (M,N), M >= 2, N >= 2.
        Every port-centered frequency column has positive Euclidean norm.
    max_lag : int
        Largest lag L, with 1 <= L < N.
    
    Returns
    -------
    result : np.ndarray
        Real vector of length L + 2, [T0, rho_0, rho_1, ..., rho_L].
        rho_0 equals one; lag k averages exactly N-k normalized dot products.
    
    Notes
    -----
    Inputs satisfy the stated domain and must remain unchanged.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_spectral_correlation(transmission: "np.ndarray", max_lag: int) -> "np.ndarray":
    centered = transmission-transmission.mean(axis=0,keepdims=True)
    unit = centered/np.linalg.norm(centered,axis=0,keepdims=True)
    n = transmission.shape[1]
    rho = [np.mean(np.sum(unit[:,:n-k]*unit[:,k:],axis=0)) for k in range(max_lag+1)]
    return np.r_[transmission.sum(axis=0).mean(),rho]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit differential cases for Studio contract validation."""
    return [
        {
            "setup": "import numpy as np\nr=np.random.default_rng(3)\na=r.uniform(.1,1,(5,12)); lag=5\ndef measure(fn):\n    value=a.copy(); out=np.asarray(fn(value,lag))\n    assert out.shape==(lag+2,) and np.array_equal(value,a)\n    return out\n",
            "call": "measure(spectral_correlation)",
            "gold_call": "measure(_oracle_spectral_correlation)",
            "tol": 1e-10
        },
        {
            "setup": "import numpy as np\nr=np.random.default_rng(3)\na=r.uniform(.1,1,(5,12)); lag=5\ndef measure(fn):\n    value=a.copy(); out=np.asarray(fn(value,lag))\n    assert out.shape==(lag+2,) and np.array_equal(value,a)\n    return out\nlag=1\n",
            "call": "measure(spectral_correlation)",
            "gold_call": "measure(_oracle_spectral_correlation)",
            "tol": 1e-10
        },
        {
            "setup": "import numpy as np\nr=np.random.default_rng(3)\na=r.uniform(.1,1,(5,12)); lag=5\ndef measure(fn):\n    value=a.copy(); out=np.asarray(fn(value,lag))\n    assert out.shape==(lag+2,) and np.array_equal(value,a)\n    return out\nlag=11\n",
            "call": "measure(spectral_correlation)",
            "gold_call": "measure(_oracle_spectral_correlation)",
            "tol": 1e-10
        },
        {
            "setup": "import numpy as np\nr=np.random.default_rng(3)\na=r.uniform(.1,1,(5,12)); lag=5\ndef measure(fn):\n    value=a.copy(); out=np.asarray(fn(value,lag))\n    assert out.shape==(lag+2,) and np.array_equal(value,a)\n    return out\na=np.tile(np.array([[1.],[2.],[4.]]),(1,7)); lag=6\n",
            "call": "measure(spectral_correlation)",
            "gold_call": "measure(_oracle_spectral_correlation)",
            "tol": 1e-10
        },
        {
            "setup": "import numpy as np\nr=np.random.default_rng(3)\na=r.uniform(.1,1,(5,12)); lag=5\ndef measure(fn):\n    value=a.copy(); out=np.asarray(fn(value,lag))\n    assert out.shape==(lag+2,) and np.array_equal(value,a)\n    return out\na=np.array([[1.,3.,1.,3.],[3.,1.,3.,1.]]); lag=3\n",
            "call": "measure(spectral_correlation)",
            "gold_call": "measure(_oracle_spectral_correlation)",
            "tol": 1e-10
        }
    ]
