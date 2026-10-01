"""
Propagate the source's transfer-matrix messages along the chain and return the angular weight of every bond on the declared polar-angle grid (the paper's Eqs. S72 to S79). One message sweeps from the first bond toward the last and the other sweeps back, each step folding in the local intact weight of the bond just passed and the angular coupling kernel; the two boundary messages are uniform in the polar angle and normalised. Only the NORMALISED message shapes are needed, so renormalise each message to unit integral over the polar angle after every step and the accumulated magnitudes never have to be carried. The angular weight of a bond is then formed from the two messages meeting at that bond, in the source's way. Recover the propagation, the normalisation and the combination rule from the paper.

The angular weight encodes how the two adjacent subchains constrain the orientation of the selected bond; truncation at the chain ends is what makes it depend on bond position.

Returns
-------
return (N, 121) float64: the source's angular weight of every bond
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def tm_angular_weights(thresholds, f, beta, De, a, le, Q):
    """thresholds: (N,) rupture thresholds; f: applied force; beta: inverse
    temperature; De, a, le: Morse parameters; Q: (121, 121) angular coupling
    kernel. Returns (N, 121) float64 with the source's angular weight of each
    bond on the grid (paper Eqs. S72-S79)."""
    n = np.asarray(thresholds).size
    return np.zeros((n, 121))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 4: transfer-matrix angular weights (Eqs. S72-S79)."""

import numpy as np

_NTH = 121


def _sw(n, h):
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-2:2] = 2.0
    return w * h / 3.0


_TH = np.linspace(0.0, np.pi, _NTH)
_WTH = _sw(_NTH, _TH[1] - _TH[0])


def _oracle_tm_angular_weights(thresholds, f, beta, De, a, le, Q):
    with np.errstate(all="ignore"):
        thr = np.asarray(thresholds, dtype=np.float64)
        N = thr.size
        logI = np.array([_oracle_local_intact_weight(thr[i], f, beta, De, a, le) for i in range(N)])
        mx = np.max(np.where(np.isfinite(logI), logI, -np.inf), axis=1, keepdims=True)
        I = np.where(np.isfinite(logI), np.exp(np.minimum(logI - mx, 0.0)), 0.0)
        PL = np.zeros((N, _NTH))
        PR = np.zeros((N, _NTH))
        PL[0] = 1.0 / np.pi
        PR[N - 1] = 1.0 / np.pi
        for i in range(N - 1):
            v = Q @ (PL[i] * I[i] * _WTH)
            PL[i + 1] = v / np.sum(v * _WTH)
        for i in range(N - 1, 0, -1):
            v = Q.T @ (PR[i] * I[i] * _WTH)
            PR[i - 1] = v / np.sum(v * _WTH)
        return PL * PR

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nDe=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0\nf=0.20;thresholds=np.full(5,2.4);Q=bending_kernel(beta,kphi,phi_e)', "call": 'tm_angular_weights(thresholds, f, beta, De, a, le, Q)', "gold_call": '_oracle_tm_angular_weights(thresholds, f, beta, De, a, le, Q)', "tol": 1e-08},
        {"setup": 'import numpy as np\nDe=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0\nf=0.15;thresholds=np.full(4,2.55);Q=bending_kernel(beta,kphi,phi_e)', "call": 'tm_angular_weights(thresholds, f, beta, De, a, le, Q)', "gold_call": '_oracle_tm_angular_weights(thresholds, f, beta, De, a, le, Q)', "tol": 1e-08},
        {"setup": 'import numpy as np\nDe=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0\nf=0.25;thresholds=np.array([2.3,2.35,2.31]);Q=bending_kernel(beta,kphi,phi_e)', "call": 'tm_angular_weights(thresholds, f, beta, De, a, le, Q)', "gold_call": '_oracle_tm_angular_weights(thresholds, f, beta, De, a, le, Q)', "tol": 1e-08},
    ]
