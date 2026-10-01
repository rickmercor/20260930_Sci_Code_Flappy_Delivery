"""
Evaluate the source's bond-length potential of mean force for a bond whose angular weight is given, at the requested bond length(s) (the paper's Eq. S82). Besides the bare stretching energy the PMF contains an entropic contribution: an angular average over the grid, weighted by that bond's angular weight, of the force coupling, together with the configurational measure the source uses. An l-independent additive constant is irrelevant and may be dropped. Recover the exact expression from the paper.

This PMF is the effective free-energy landscape for stretching the selected bond while the rest of the chain stays intact.

Returns
-------
return array like l (float64): the bond-length PMF of the selected bond
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def tm_pmf(l, w_i, f, beta, De, a, le):
    """l: bond length(s); w_i: (121,) angular weight of the selected bond;
    f: applied force; beta: inverse temperature; De, a, le: Morse parameters.
    Returns a float64 array shaped like l with the source's bond-length PMF
    (paper Eq. S82)."""
    return np.zeros_like(np.atleast_1d(np.asarray(l, dtype=np.float64)))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 5: bond-length PMF from the angular weight (Eq. S82)."""

import numpy as np

_NTH = 121


def _sw(n, h):
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-2:2] = 2.0
    return w * h / 3.0


_TH = np.linspace(0.0, np.pi, _NTH)
_WTH = _sw(_NTH, _TH[1] - _TH[0])


def _oracle_tm_pmf(l, w_i, f, beta, De, a, le):
    with np.errstate(all="ignore"):
        l = np.atleast_1d(np.asarray(l, dtype=np.float64))
        ex = beta * f * l[:, None] * np.cos(_TH)[None, :]
        m = ex.max(axis=1, keepdims=True)
        ang = np.log(((np.exp(ex - m) * (np.asarray(w_i) * np.sin(_TH))[None, :]) * _WTH[None, :]).sum(axis=1)) + m[:, 0]
        return _oracle_morse_potential(l, De, a, le) - (1.0 / beta) * (np.log(l ** 2) + ang)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nDe=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0\nf=0.20;Q=bending_kernel(beta,kphi,phi_e)\nw_i=tm_angular_weights(np.full(5,2.4),f,beta,De,a,le,Q)[0]\nl=np.array([1.0,1.5,2.0,2.4])', "call": 'tm_pmf(l, w_i, f, beta, De, a, le)', "gold_call": '_oracle_tm_pmf(l, w_i, f, beta, De, a, le)', "tol": 1e-08},
        {"setup": 'import numpy as np\nDe=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0\nf=0.15;Q=bending_kernel(beta,kphi,phi_e)\nw_i=tm_angular_weights(np.full(4,2.55),f,beta,De,a,le,Q)[1]\nl=np.linspace(0.9,2.5,6)', "call": 'tm_pmf(l, w_i, f, beta, De, a, le)', "gold_call": '_oracle_tm_pmf(l, w_i, f, beta, De, a, le)', "tol": 1e-08},
        {"setup": 'import numpy as np\nDe=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0\nf=0.25;Q=bending_kernel(beta,kphi,phi_e)\nw_i=tm_angular_weights(np.array([2.3,2.35,2.31]),f,beta,De,a,le,Q)[2]\nl=np.array([1.1,1.8,2.3])', "call": 'tm_pmf(l, w_i, f, beta, De, a, le)', "gold_call": '_oracle_tm_pmf(l, w_i, f, beta, De, a, le)', "tol": 1e-08},
    ]
