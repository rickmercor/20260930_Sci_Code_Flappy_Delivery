"""
Return the natural logarithm of the source's local intact weight for one bond held at each polar angle of the declared grid, given that bond's rupture threshold (the paper's Eq. S60). The weight integrates the Boltzmann factor of the bond over the allowed bond-length interval, carrying the configurational measure the source uses. Because beta is large the raw weight spans hundreds of orders of magnitude, so return its logarithm and build it stably. Integrate over bond length with composite Simpson quadrature on 1201 uniform points from zero to the threshold. Where the measure vanishes at the two grid endpoints the log is returned as the finite floor -1e300 rather than minus infinity, so the array stays comparable. Raise ValueError on a non-positive threshold.

Separating this purely local bond-length integral from the orientational coupling is what makes the transfer-matrix propagation possible.

Returns
-------
return (121,) float64: natural log of the source's local intact weight per polar angle
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def local_intact_weight(lt, f, beta, De, a, le):
    """lt: rupture threshold of this bond; f: applied force; beta: inverse
    temperature; De, a, le: Morse parameters. Returns (121,) float64 with the
    natural log of the source's local intact weight at each grid polar angle
    (paper Eq. S60); wherever the configurational measure vanishes -- which on
    this grid means BOTH polar-angle endpoints, judged by sin(theta) <= 1e-12 so
    that theta = pi is treated exactly like theta = 0 -- the returned value is the
    finite floor -1e300 rather than -inf. Raises ValueError on a non-positive
    threshold."""
    return np.zeros(121)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 3: local intact weight (Eq. S60), returned as logs."""

import numpy as np

_NTH = 121


def _sw(n, h):
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-2:2] = 2.0
    return w * h / 3.0


_TH = np.linspace(0.0, np.pi, _NTH)
_WTH = _sw(_NTH, _TH[1] - _TH[0])

_NL = 1201


def _oracle_local_intact_weight(lt, f, beta, De, a, le):
    if lt <= 0:
        raise ValueError("threshold must be positive")
    with np.errstate(all="ignore"):
        l = np.linspace(0.0, lt, _NL)
        wl = _sw(_NL, l[1] - l[0])
        ex = -beta * (_oracle_morse_potential(l, De, a, le)[None, :] - f * l[None, :] * np.cos(_TH)[:, None])
        ex = ex + np.where(l > 0, np.log(np.where(l > 0, l, 1.0) ** 2), -np.inf)[None, :]
        m = np.max(np.where(np.isfinite(ex), ex, -np.inf), axis=1, keepdims=True)
        val = np.log((np.exp(ex - m) * wl[None, :]).sum(axis=1)) + m[:, 0]
        s = np.sin(_TH)
        ok = s > 1e-12
        out = np.where(ok, np.log(np.where(ok, s, 1.0)), -1e300) + val
        return np.where(np.isfinite(out), np.maximum(out, -1e300), -1e300)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nlt=2.4;f=0.20;De=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0', "call": 'local_intact_weight(lt, f, beta, De, a, le)', "gold_call": '_oracle_local_intact_weight(lt, f, beta, De, a, le)', "tol": 1e-08},
        {"setup": 'import numpy as np\nlt=2.55;f=0.15;De=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0', "call": 'local_intact_weight(lt, f, beta, De, a, le)', "gold_call": '_oracle_local_intact_weight(lt, f, beta, De, a, le)', "tol": 1e-08},
        {"setup": 'import numpy as np\nlt=2.3;f=0.25;De=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0', "call": 'local_intact_weight(lt, f, beta, De, a, le)', "gold_call": '_oracle_local_intact_weight(lt, f, beta, De, a, le)', "tol": 1e-08},
        {"setup": 'import numpy as np\nlt=1.05;f=0.20;De=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0', "call": 'local_intact_weight(lt, f, beta, De, a, le)', "gold_call": '_oracle_local_intact_weight(lt, f, beta, De, a, le)', "tol": 1e-08},
        {"setup": 'import numpy as np\nlt=5.5;f=0.05;De=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0', "call": 'local_intact_weight(lt, f, beta, De, a, le)', "gold_call": '_oracle_local_intact_weight(lt, f, beta, De, a, le)', "tol": 1e-08},
    ]
