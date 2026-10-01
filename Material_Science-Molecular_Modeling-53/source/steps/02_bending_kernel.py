"""
Build the source's angular coupling kernel between the polar angles of two successive bonds (the paper's Eq. S61). The bending energy penalises departures of the local bond angle from its equilibrium value with stiffness kphi (Eq. S9); because only the RELATIVE azimuth of the two bonds is unconstrained, the source removes it by integrating over it, which is what turns a bending energy of the bond angle into a kernel in the two polar angles. Recover from the paper both how the bond angle follows from the two polar angles and that relative azimuth, and what is integrated. Use a uniform grid of 121 polar angles spanning 0 to pi and a uniform grid of 121 azimuths spanning 0 to 2*pi, both with composite Simpson weights.

Nearest-neighbour bending is the only interaction coupling bond orientations, so once the relative azimuth is integrated out the whole chain can be propagated with a transfer matrix in the polar angle alone.

Returns
-------
return (121, 121) float64: the source's angular coupling kernel
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bending_kernel(beta, kphi, phi_e):
    """beta: inverse temperature; kphi: bending stiffness; phi_e: equilibrium
    bond angle. Returns (121, 121) float64: the source's angular coupling kernel
    on the declared polar-angle grid (paper Eq. S61)."""
    return np.zeros((121, 121))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 2: angular coupling kernel (Eq. S61, with Eq. S9)."""

import numpy as np

_NTH = 121


def _sw(n, h):
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-2:2] = 2.0
    return w * h / 3.0


_TH = np.linspace(0.0, np.pi, _NTH)
_WTH = _sw(_NTH, _TH[1] - _TH[0])

_NW = 121


def _oracle_bending_kernel(beta, kphi, phi_e):
    with np.errstate(all="ignore"):
        w = np.linspace(0.0, 2.0 * np.pi, _NW)
        ww = _sw(_NW, w[1] - w[0])
        ct, st = np.cos(_TH), np.sin(_TH)
        cphi = (ct[:, None, None] * ct[None, :, None]
                + st[:, None, None] * st[None, :, None] * np.cos(w)[None, None, :])
        phi = np.arccos(np.clip(cphi, -1.0, 1.0))
        return np.einsum('ijk,k->ij', np.exp(-0.5 * beta * kphi * (phi - phi_e) ** 2), ww)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nDe=1.0;a=2.15;le=1.0;beta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0', "call": 'bending_kernel(beta, kphi, phi_e)', "gold_call": '_oracle_bending_kernel(beta, kphi, phi_e)', "tol": 1e-09},
        {"setup": 'import numpy as np\nbeta=279.0;kphi=1000.0/(np.pi**2*279.0);phi_e=60.0*np.pi/180.0', "call": 'bending_kernel(beta, kphi, phi_e)', "gold_call": '_oracle_bending_kernel(beta, kphi, phi_e)', "tol": 1e-09},
        {"setup": 'import numpy as np\nbeta=150.0;kphi=2500.0/(np.pi**2*150.0);phi_e=75.0*np.pi/180.0', "call": 'bending_kernel(beta, kphi, phi_e)', "gold_call": '_oracle_bending_kernel(beta, kphi, phi_e)', "tol": 1e-09},
        {"setup": 'import numpy as np\nbeta=279.0;kphi=18.2/(np.pi**2*279.0);phi_e=69.0*np.pi/180.0', "call": 'bending_kernel(beta, kphi, phi_e)', "gold_call": '_oracle_bending_kernel(beta, kphi, phi_e)', "tol": 1e-09},
        {"setup": 'import numpy as np\nbeta=279.0;kphi=1820.0/(np.pi**2*279.0);phi_e=0.05', "call": 'bending_kernel(beta, kphi, phi_e)', "gold_call": '_oracle_bending_kernel(beta, kphi, phi_e)', "tol": 1e-09},
    ]
