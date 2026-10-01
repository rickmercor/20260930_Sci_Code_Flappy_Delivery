"""
Convert the two-parameter scattering jet into the intensity-transmission jet seen by the measured ports.

The reconstructive spectrometer uses the nonnegative transmission matrix $$A_{mn}=|S_{m0}(\omega_n)|^2$$, where incoming channel zero is illuminated and outgoing channel zero is the unmeasured reflection. Intensities are differentiated, as ordinary partial derivatives, with respect to the same two design parameters as the amplitude jet, and the returned orders follow the same convention: value, first derivative in the coupling, first derivative in the detuning, then the three second derivatives in the order $$qq$$, $$qs$$, $$ss$$.

Returns
-------
np.ndarray: Real array (6, C - 1, N) containing A, dA/dq, dA/ds, d2A/dq2, d2A/dqds, d2A/ds2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transmission_jet(scattering: "np.ndarray") -> "np.ndarray":
    r"""Extract the measured intensity matrix and its design derivatives.
    
    Parameters
    ----------
    scattering : np.ndarray
        Finite complex array (6, N, C, C), with orders S, dS/dq, dS/ds,
        d2S/dq2, d2S/dqds, d2S/ds2 and channel indices outgoing then
        incoming; C >= 2 and N >= 1.
    
    Returns
    -------
    result : np.ndarray
        Real array (6, C - 1, N) containing A, dA/dq, dA/ds, d2A/dq2,
        d2A/dqds, d2A/ds2. Only outgoing ports 1 through C - 1 for incoming
        port zero are used.
    
    Notes
    -----
    Inputs satisfy the stated domain and must remain unchanged. Jets are ordinary
    real-parameter partial derivatives, with no factorial normalization.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_transmission_jet(scattering: "np.ndarray") -> "np.ndarray":
    s, sq, ss, sqq, sqs, sss = scattering[:, :, 1:, 0]
    a = np.abs(s)**2
    aq = 2*np.real(np.conj(s)*sq)
    as_ = 2*np.real(np.conj(s)*ss)
    aqq = 2*(np.abs(sq)**2+np.real(np.conj(s)*sqq))
    aqs = 2*(np.real(np.conj(sq)*ss)+np.real(np.conj(s)*sqs))
    ass = 2*(np.abs(ss)**2+np.real(np.conj(s)*sss))
    return np.stack((a.T, aq.T, as_.T, aqq.T, aqs.T, ass.T))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit differential cases for Studio contract validation."""
    return [
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(9)\nx=r.standard_normal((6,5,4,4))+1j*r.standard_normal((6,5,4,4))\ndef measure(fn):\n    value=x.copy(); out=np.asarray(fn(value))\n    assert out.shape==(6,x.shape[2]-1,x.shape[1])\n    assert np.array_equal(value,x)\n    return out\n',
            "call": "measure(transmission_jet)",
            "gold_call": "measure(_oracle_transmission_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(9)\nx=r.standard_normal((6,5,4,4))+1j*r.standard_normal((6,5,4,4))\ndef measure(fn):\n    value=x.copy(); out=np.asarray(fn(value))\n    assert out.shape==(6,x.shape[2]-1,x.shape[1])\n    assert np.array_equal(value,x)\n    return out\nx=x[:,:1,:2,:2].copy()\n',
            "call": "measure(transmission_jet)",
            "gold_call": "measure(_oracle_transmission_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(9)\nx=r.standard_normal((6,5,4,4))+1j*r.standard_normal((6,5,4,4))\ndef measure(fn):\n    value=x.copy(); out=np.asarray(fn(value))\n    assert out.shape==(6,x.shape[2]-1,x.shape[1])\n    assert np.array_equal(value,x)\n    return out\nx[0]=0\n',
            "call": "measure(transmission_jet)",
            "gold_call": "measure(_oracle_transmission_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(9)\nx=r.standard_normal((6,5,4,4))+1j*r.standard_normal((6,5,4,4))\ndef measure(fn):\n    value=x.copy(); out=np.asarray(fn(value))\n    assert out.shape==(6,x.shape[2]-1,x.shape[1])\n    assert np.array_equal(value,x)\n    return out\nx[1:]=0\n',
            "call": "measure(transmission_jet)",
            "gold_call": "measure(_oracle_transmission_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(9)\nx=r.standard_normal((6,5,4,4))+1j*r.standard_normal((6,5,4,4))\ndef measure(fn):\n    value=x.copy(); out=np.asarray(fn(value))\n    assert out.shape==(6,x.shape[2]-1,x.shape[1])\n    assert np.array_equal(value,x)\n    return out\nx[1:3]=0\n',
            "call": "measure(transmission_jet)",
            "gold_call": "measure(_oracle_transmission_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(9)\nx=r.standard_normal((6,5,4,4))+1j*r.standard_normal((6,5,4,4))\ndef measure(fn):\n    value=x.copy(); out=np.asarray(fn(value))\n    assert out.shape==(6,x.shape[2]-1,x.shape[1])\n    assert np.array_equal(value,x)\n    return out\nx[:,:,:,0]*=np.exp(.73j)\n',
            "call": "measure(transmission_jet)",
            "gold_call": "measure(_oracle_transmission_jet)",
            "tol": 1e-10
        }
    ]
