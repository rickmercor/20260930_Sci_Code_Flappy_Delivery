"""
Evaluate the scattering matrix and its first and second derivatives in the two design parameters.

The device has two design parameters: the logarithmic channel coupling $$q$$ and the comb detuning $$s$$, which rigidly shifts the sampled frequencies. For coupling $$q$$ set $$W(q)=e^qW_0$$ and $$H_{\mathrm{eff}}(q)=H-iW(q)W(q)^\dagger/2$$, and for detuning $$s$$ evaluate the scattering matrix at the shifted frequency: $$S(\omega,q,s)=I-iW(q)^\dagger[(\omega+s)I-H_{\mathrm{eff}}(q)]^{-1}W(q)$$. The dagger denotes conjugate transpose. With Hermitian $$H$$ and no absorption, this construction conserves flux; the reflection amplitude is part of the full scattering matrix.



The returned orders are ordinary first and second partial derivatives of $$S$$ with respect to $$q$$ and $$s$$, including the mixed partial, in the order $$S$$, $$\partial_qS$$, $$\partial_sS$$, $$\partial_q^2S$$, $$\partial_q\partial_sS$$, $$\partial_s^2S$$. They are not Taylor coefficients divided by factorials. These derivatives support the search for a constrained noise-amplification minimum over the two-parameter design space.

Returns
-------
np.ndarray: Complex array (6, N, C, C) containing S, dS/dq, dS/ds, d2S/dq2, d2S/dqds, d2S/ds2, in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def scattering_jet(cavity: "np.ndarray", frequencies: "np.ndarray", log_coupling: float, detuning: float) -> "np.ndarray":
    r"""Compute a two-parameter scattering jet at the supplied design point.
    
    Parameters
    ----------
    cavity : np.ndarray
        Complex array (P, P + C), with Hermitian H in the first P columns
        and arbitrary complex W0 in the remaining C >= 2 columns.
    frequencies : np.ndarray
        Finite real vector of length N >= 1; its ordering is preserved.
    log_coupling : float
        Finite real q. All shifted-frequency resolvents are nonsingular.
    detuning : float
        Finite real s, added to every sampled frequency.
    
    Returns
    -------
    result : np.ndarray
        Complex array (6, N, C, C) containing S, dS/dq, dS/ds, d2S/dq2,
        d2S/dqds, d2S/ds2, in that order. Matrix indices are outgoing,
        incoming channels.
    
    Notes
    -----
    Inputs satisfy the stated domain and must remain unchanged. Results must
    resolve the ordinary partial derivatives, including near zero; analytic
    differentiation or an equivalently accurate method is accepted.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import lu_factor, lu_solve

def _oracle_scattering_jet(cavity: "np.ndarray", frequencies: "np.ndarray", log_coupling: float, detuning: float) -> "np.ndarray":
    p = cavity.shape[0]
    h = cavity[:, :p]
    w0 = cavity[:, p:]
    k = w0.shape[1]
    scale = np.exp(2*log_coupling)
    d = scale*(w0@w0.conj().T)
    adjoint = w0.conj().T
    eye = np.eye(p)
    result = np.empty((6, len(frequencies), k, k), dtype=complex)
    for n, frequency in enumerate(frequencies):
        factors = lu_factor((frequency+detuning)*eye-h+0.5j*d)
        # Powers of the resolvent applied to the coupling columns only.
        x1 = lu_solve(factors, w0)
        x2 = lu_solve(factors, x1)
        x3 = lu_solve(factors, x2)
        dx = lu_solve(factors, d@x1)
        dxd = lu_solve(factors, d@dx)
        mixed = lu_solve(factors, d@x2)+lu_solve(factors, dx)
        g = adjoint@x1
        gq = -1j*(adjoint@dx)
        gs = -(adjoint@x2)
        gqq = -2*(adjoint@dxd)-2j*(adjoint@dx)
        gqs = 1j*(adjoint@mixed)
        gss = 2*(adjoint@x3)
        result[0, n] = np.eye(k)-1j*scale*g
        result[1, n] = -1j*scale*(2*g+gq)
        result[2, n] = -1j*scale*gs
        result[3, n] = -1j*scale*(4*g+4*gq+gqq)
        result[4, n] = -1j*scale*(2*gs+gqs)
        result[5, n] = -1j*scale*gss
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit differential cases for Studio contract validation."""
    return [
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(21)\nx=r.standard_normal((7,7))+1j*r.standard_normal((7,7))\nh=(x+x.conj().T)/4\nw=(r.standard_normal((7,4))+1j*r.standard_normal((7,4)))/3\ncavity=np.concatenate((h,w),axis=1)\nfreq=np.array([-.7,-.13,.22,.9]); q=.2; s=.05\ndef measure(fn):\n    c=cavity.copy(); f=freq.copy()\n    v=np.asarray(fn(c,f,q,s))\n    assert v.shape==(6,len(freq),cavity.shape[1]-cavity.shape[0],cavity.shape[1]-cavity.shape[0])\n    assert np.array_equal(c,cavity) and np.array_equal(f,freq)\n    return np.stack((v.real,v.imag))\n',
            "call": "measure(scattering_jet)",
            "gold_call": "measure(_oracle_scattering_jet)",
            "tol": 2e-08
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(21)\nx=r.standard_normal((7,7))+1j*r.standard_normal((7,7))\nh=(x+x.conj().T)/4\nw=(r.standard_normal((7,4))+1j*r.standard_normal((7,4)))/3\ncavity=np.concatenate((h,w),axis=1)\nfreq=np.array([-.7,-.13,.22,.9]); q=.2; s=.05\ndef measure(fn):\n    c=cavity.copy(); f=freq.copy()\n    v=np.asarray(fn(c,f,q,s))\n    assert v.shape==(6,len(freq),cavity.shape[1]-cavity.shape[0],cavity.shape[1]-cavity.shape[0])\n    assert np.array_equal(c,cavity) and np.array_equal(f,freq)\n    return np.stack((v.real,v.imag))\nq=-1.3\n',
            "call": "measure(scattering_jet)",
            "gold_call": "measure(_oracle_scattering_jet)",
            "tol": 2e-08
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(21)\nx=r.standard_normal((7,7))+1j*r.standard_normal((7,7))\nh=(x+x.conj().T)/4\nw=(r.standard_normal((7,4))+1j*r.standard_normal((7,4)))/3\ncavity=np.concatenate((h,w),axis=1)\nfreq=np.array([-.7,-.13,.22,.9]); q=.2; s=.05\ndef measure(fn):\n    c=cavity.copy(); f=freq.copy()\n    v=np.asarray(fn(c,f,q,s))\n    assert v.shape==(6,len(freq),cavity.shape[1]-cavity.shape[0],cavity.shape[1]-cavity.shape[0])\n    assert np.array_equal(c,cavity) and np.array_equal(f,freq)\n    return np.stack((v.real,v.imag))\ns=-.4\n',
            "call": "measure(scattering_jet)",
            "gold_call": "measure(_oracle_scattering_jet)",
            "tol": 2e-08
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(21)\nx=r.standard_normal((7,7))+1j*r.standard_normal((7,7))\nh=(x+x.conj().T)/4\nw=(r.standard_normal((7,4))+1j*r.standard_normal((7,4)))/3\ncavity=np.concatenate((h,w),axis=1)\nfreq=np.array([-.7,-.13,.22,.9]); q=.2; s=.05\ndef measure(fn):\n    c=cavity.copy(); f=freq.copy()\n    v=np.asarray(fn(c,f,q,s))\n    assert v.shape==(6,len(freq),cavity.shape[1]-cavity.shape[0],cavity.shape[1]-cavity.shape[0])\n    assert np.array_equal(c,cavity) and np.array_equal(f,freq)\n    return np.stack((v.real,v.imag))\nq=1.2; s=.3\n',
            "call": "measure(scattering_jet)",
            "gold_call": "measure(_oracle_scattering_jet)",
            "tol": 2e-08
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(21)\nx=r.standard_normal((7,7))+1j*r.standard_normal((7,7))\nh=(x+x.conj().T)/4\nw=(r.standard_normal((7,4))+1j*r.standard_normal((7,4)))/3\ncavity=np.concatenate((h,w),axis=1)\nfreq=np.array([-.7,-.13,.22,.9]); q=.2; s=.05\ndef measure(fn):\n    c=cavity.copy(); f=freq.copy()\n    v=np.asarray(fn(c,f,q,s))\n    assert v.shape==(6,len(freq),cavity.shape[1]-cavity.shape[0],cavity.shape[1]-cavity.shape[0])\n    assert np.array_equal(c,cavity) and np.array_equal(f,freq)\n    return np.stack((v.real,v.imag))\ns=0.\n',
            "call": "measure(scattering_jet)",
            "gold_call": "measure(_oracle_scattering_jet)",
            "tol": 2e-08
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(21)\nx=r.standard_normal((7,7))+1j*r.standard_normal((7,7))\nh=(x+x.conj().T)/4\nw=(r.standard_normal((7,4))+1j*r.standard_normal((7,4)))/3\ncavity=np.concatenate((h,w),axis=1)\nfreq=np.array([-.7,-.13,.22,.9]); q=.2; s=.05\ndef measure(fn):\n    c=cavity.copy(); f=freq.copy()\n    v=np.asarray(fn(c,f,q,s))\n    assert v.shape==(6,len(freq),cavity.shape[1]-cavity.shape[0],cavity.shape[1]-cavity.shape[0])\n    assert np.array_equal(c,cavity) and np.array_equal(f,freq)\n    return np.stack((v.real,v.imag))\nfreq=np.array([.22])\n',
            "call": "measure(scattering_jet)",
            "gold_call": "measure(_oracle_scattering_jet)",
            "tol": 2e-08
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(21)\nx=r.standard_normal((7,7))+1j*r.standard_normal((7,7))\nh=(x+x.conj().T)/4\nw=(r.standard_normal((7,4))+1j*r.standard_normal((7,4)))/3\ncavity=np.concatenate((h,w),axis=1)\nfreq=np.array([-.7,-.13,.22,.9]); q=.2; s=.05\ndef measure(fn):\n    c=cavity.copy(); f=freq.copy()\n    v=np.asarray(fn(c,f,q,s))\n    assert v.shape==(6,len(freq),cavity.shape[1]-cavity.shape[0],cavity.shape[1]-cavity.shape[0])\n    assert np.array_equal(c,cavity) and np.array_equal(f,freq)\n    return np.stack((v.real,v.imag))\nw=np.zeros((7,2),complex)\ncavity=np.concatenate((h,w),axis=1)\n',
            "call": "measure(scattering_jet)",
            "gold_call": "measure(_oracle_scattering_jet)",
            "tol": 2e-08
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(21)\nx=r.standard_normal((7,7))+1j*r.standard_normal((7,7))\nh=(x+x.conj().T)/4\nw=(r.standard_normal((7,4))+1j*r.standard_normal((7,4)))/3\ncavity=np.concatenate((h,w),axis=1)\nfreq=np.array([-.7,-.13,.22,.9]); q=.2; s=.05\ndef measure(fn):\n    c=cavity.copy(); f=freq.copy()\n    v=np.asarray(fn(c,f,q,s))\n    assert v.shape==(6,len(freq),cavity.shape[1]-cavity.shape[0],cavity.shape[1]-cavity.shape[0])\n    assert np.array_equal(c,cavity) and np.array_equal(f,freq)\n    return np.stack((v.real,v.imag))\nfreq=freq[::-1].copy()\nw[:,0]*=1j\ncavity=np.concatenate((h,w),axis=1)\n',
            "call": "measure(scattering_jet)",
            "gold_call": "measure(_oracle_scattering_jet)",
            "tol": 2e-08
        }
    ]
