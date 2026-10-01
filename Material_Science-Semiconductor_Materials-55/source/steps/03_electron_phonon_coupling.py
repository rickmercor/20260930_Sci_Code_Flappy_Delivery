"""
Convert the coordinate offset and the vibrational quantum into the dimensionless electron-phonon coupling strength and the Debye-Waller factor.

The offset between the two parabolas decides how much of the emission stays in
the sharp zero-phonon line. The dimensionless coupling strength S is the average
number of vibrational quanta created when the defect makes a vertical electronic
transition, that is, the relaxation energy the lattice releases after that
transition measured in units of the vibrational quantum, and it follows from the
offset and the shared curvature alone. With the excited state in its zero-point
level, as it is at low temperature, the overlap with the ground-state level of the same
index gives a Poisson distribution of vibrational final states with mean S, and
the weight of the zero-phonon term is the Poisson probability of emitting no
quanta at all. That weight is the Debye-Waller factor W, and it falls off
exponentially with the coupling
strength, so a defect whose lattice relaxes strongly on excitation loses almost
all of its emission into the sideband even when the electronic transition itself
is fast. Again delta_Q arrives in sqrt(amu) Angstrom and hbar Omega in eV, so
both must be converted before the ratio is formed.

Use these constant values exactly, so that the result is reproducible to the
precision this step is checked at:

hbar        = 1.054571817e-34 J s
1 eV        = 1.602176634e-19 J
1 amu       = 1.66053906660e-27 kg
1 Angstrom  = 1e-10 m

Returns
-------
numpy array of two floats, [S, W]: the coupling strength and the Debye-Waller factor.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def electron_phonon_coupling(delta_q: float, hbar_omega: float) -> np.ndarray:
    '''Dimensionless coupling strength and Debye-Waller factor.

    Parameters
    ----------
    delta_q : float
        Mass-weighted coordinate offset in sqrt(amu) Angstrom. Must be positive.
    hbar_omega : float
        Effective vibrational quantum in eV. Must be positive.

    Returns
    -------
    coupling : numpy.ndarray, shape (2,)
        [S, W], the dimensionless electron-phonon coupling strength and the
        Debye-Waller factor exp(-S). Raises ValueError on non-finite input or
        on a non-positive delta_q or hbar_omega.
    '''
    return coupling

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_electron_phonon_coupling(delta_q: float, hbar_omega: float) -> np.ndarray:
    hbar_si = 1.054571817e-34
    ev_j = 1.602176634e-19
    amu_kg = 1.66053906660e-27
    ang_m = 1.0e-10

    dq = float(delta_q)
    hw = float(hbar_omega)
    if not (np.isfinite(dq) and np.isfinite(hw)):
        raise ValueError("delta_q and hbar_omega must be finite")
    if dq <= 0.0 or hw <= 0.0:
        raise ValueError("delta_q and hbar_omega must be positive")
    omega = hw * ev_j / hbar_si
    dq2_si = dq * dq * amu_kg * ang_m * ang_m
    s = dq2_si * omega / (2.0 * hbar_si)
    return np.array([s, np.exp(-s)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {   # normal: the antimony-vacancy centre of the task
            "setup": "import numpy as np\n",
            "call": "np.round(electron_phonon_coupling(1.020205375219, 0.031348254831), 12).tolist()",
            "gold_call": "np.round(_oracle_electron_phonon_coupling(1.020205375219, 0.031348254831), 12).tolist()",
        },
        {   # normal: the oxygen-vacancy centre in the same host
            "setup": "import numpy as np\n",
            "call": "round(100.0 * float(electron_phonon_coupling(0.66, 0.044)[1]), 6)",
            "gold_call": "round(100.0 * float(_oracle_electron_phonon_coupling(0.66, 0.044)[1]), 6)",
        },
        {   # normal: the chlorine-vacancy centre in the same host
            "setup": "import numpy as np\n",
            "call": "round(100.0 * float(electron_phonon_coupling(0.93, 0.034)[1]), 6)",
            "gold_call": "round(100.0 * float(_oracle_electron_phonon_coupling(0.93, 0.034)[1]), 6)",
        },
        {   # boundary: a vanishing offset keeps essentially all emission in the line
            "setup": "import numpy as np\n",
            "call": "np.round(electron_phonon_coupling(1e-6, 0.03), 12).tolist()",
            "gold_call": "np.round(_oracle_electron_phonon_coupling(1e-6, 0.03), 12).tolist()",
        },
        {   # boundary: very strong coupling drives the line weight far into the tail
            "setup": "import numpy as np\n",
            "call": "float(np.round(np.log10(electron_phonon_coupling(3.4, 0.052)[1]), 10))",
            "gold_call": "float(np.round(np.log10(_oracle_electron_phonon_coupling(3.4, 0.052)[1]), 10))",
        },
        {   # edge: a non-positive vibrational quantum must raise ValueError
            "setup": """import numpy as np
def probe(fn):
    try:
        fn(1.0, 0.0)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "probe(electron_phonon_coupling)",
            "gold_call": "probe(_oracle_electron_phonon_coupling)",
        },
    ]
