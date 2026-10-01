"""
Split the electronic dipole rate into the phonon-corrected total radiative rate, the zero-phonon line rate and the sideband rate.

Two distinct corrections separate the bare dipole rate from what an experiment
actually measures. First, every emission event on average carries away S
vibrational quanta, so the mean emitted photon energy is E_ZPL - S hbar Omega
rather than E_ZPL. The total radiative rate is reduced by the ratio of that mean
photon energy to the zero-phonon line energy, taken to the first power: the cubic
density-of-states factor carried by the bare dipole rate is not re-evaluated at
the reduced energy. This correction is routinely dropped because S hbar Omega is small
against the transition energy in visible emitters, but for a near-infrared centre
with strong coupling it is a double-digit effect and cannot be ignored. Second,
only the zero-phonon fraction of the emission lands in the sharp line, and that
fraction is the Debye-Waller factor. The zero-phonon rate is referenced to the
uncorrected dipole rate, because the zero-phonon transition by construction
releases no vibrational energy at all, so applying the first correction to it as
well would count that energy twice. Everything that is radiated and does not land
in the line forms the phonon sideband. The three rates together describe how the
radiative budget of the defect is divided. The radiative quantum efficiency
depends on the total alone, while the zero-phonon figure of merit uses the line
rate.

Returns
-------
numpy array of three floats, [Gamma_R, Gamma_ZPL, Gamma_PSB] in s^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def radiative_rate_partition(gamma_r0: float, huang_rhys: float,
                             hbar_omega: float, e_zpl: float) -> np.ndarray:
    '''Split the electronic dipole rate into total, zero-phonon and sideband parts.

    Parameters
    ----------
    gamma_r0 : float
        Purely electronic electric-dipole emission rate in s^-1. Must be positive.
    huang_rhys : float
        Dimensionless electron-phonon coupling strength. Must be positive.
    hbar_omega : float
        Effective vibrational quantum in eV. Must be positive.
    e_zpl : float
        Zero-phonon line energy in eV. Must be positive.

    Returns
    -------
    rates : numpy.ndarray, shape (3,)
        [Gamma_R, Gamma_ZPL, Gamma_PSB] in s^-1: the phonon-corrected total
        radiative rate, the zero-phonon line rate, and the sideband rate.
        Raises ValueError on non-finite input, on a non-positive argument, or
        when the mean vibrational energy released reaches the transition energy,
        or when the corrected total rate would be smaller than the zero-phonon
        line rate and therefore imply a negative sideband rate.
    '''
    return rates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_radiative_rate_partition(gamma_r0: float, huang_rhys: float,
                                     hbar_omega: float, e_zpl: float) -> np.ndarray:
    g0 = float(gamma_r0)
    s = float(huang_rhys)
    hw = float(hbar_omega)
    ez = float(e_zpl)
    if not all(np.isfinite(v) for v in (g0, s, hw, ez)):
        raise ValueError("all arguments must be finite")
    if g0 <= 0.0 or s <= 0.0 or hw <= 0.0 or ez <= 0.0:
        raise ValueError("all arguments must be positive")
    reduction = 1.0 - s * hw / ez
    if reduction <= 0.0:
        raise ValueError("the mean vibrational energy released must stay below e_zpl")
    zpl_fraction = np.exp(-s)
    if reduction < zpl_fraction:
        raise ValueError("the corrected total radiative rate must not be below the zero-phonon line rate")
    gamma_r = g0 * reduction
    gamma_zpl = g0 * zpl_fraction
    return np.array([gamma_r, gamma_zpl, gamma_r - gamma_zpl], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {   # normal: the antimony-vacancy centre of the task
            "setup": "import numpy as np\nG0 = 1518331.655987\n",
            "call": "np.round(radiative_rate_partition(G0, 3.902705291196, 0.031348254831, 0.761436000001), 3).tolist()",
            "gold_call": "np.round(_oracle_radiative_rate_partition(G0, 3.902705291196, 0.031348254831, 0.761436000001), 3).tolist()",
        },
        {   # normal: weak coupling leaves the total rate close to the bare dipole rate
            "setup": "import numpy as np\n",
            "call": "np.round(radiative_rate_partition(1.0e8, 0.35, 0.065, 1.945), 3).tolist()",
            "gold_call": "np.round(_oracle_radiative_rate_partition(1.0e8, 0.35, 0.065, 1.945), 3).tolist()",
        },
        {   # normal: a visible centre in the radiative-only limit, reported as a percentage
            "setup": """import numpy as np
def line_share(fn):
    r = fn(1.0, 2.5631155, 0.045, 1.05)
    return round(100.0 * float(r[1] / r[0]), 8)
""",
            "call": "line_share(radiative_rate_partition)",
            "gold_call": "line_share(_oracle_radiative_rate_partition)",
        },
        {   # boundary: the three rates must balance, and the line rate is reported alongside
            "setup": """import numpy as np
def balance(fn):
    r = fn(4.2e6, 2.9, 0.028, 0.83)
    return [round(float(r[0] - r[1] - r[2]), 9), round(float(r[1]), 6)]
""",
            "call": "balance(radiative_rate_partition)",
            "gold_call": "balance(_oracle_radiative_rate_partition)",
        },
        {   # boundary: a positive total that would imply a negative sideband must raise ValueError
            "setup": """import numpy as np
def probe(fn):
    try:
        fn(1.0e6, 2.0, 0.4, 0.81)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "probe(radiative_rate_partition)",
            "gold_call": "probe(_oracle_radiative_rate_partition)",
        },
        {   # edge: a released vibrational energy exceeding the transition energy must raise ValueError
            "setup": """import numpy as np
def probe(fn):
    try:
        fn(1e6, 30.0, 0.03, 0.8)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "probe(radiative_rate_partition)",
            "gold_call": "probe(_oracle_radiative_rate_partition)",
        },
    ]
