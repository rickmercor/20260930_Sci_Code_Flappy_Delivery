"""
Collapse the thermally averaged golden-rule sum at the energy-conserving levels and return the nonradiative multiphonon transition rate at a given lattice temperature.

Nonradiative decay dumps the whole electronic energy into the vibrational ladder of
the electronic ground state. From a single excited-state vibrational level m, Fermi's
golden rule gives

Gamma_m = (2 pi / hbar) g w^2 sum_n |<chi_e,m| Q |chi_g,n>|^2 delta(E_ZPL + m hbar Omega - n hbar Omega),

where g is the configurational degeneracy of the defect and w is the electron-phonon
coupling matrix element. The delta function is what usually forces a Gaussian
broadening to be introduced by hand, with the answer depending on the width chosen.
Continuing the ground-state level index to real values removes that arbitrariness:
the sum becomes an integral over n and the delta function collapses it exactly at
the one level where the vibrational energy absorbed matches the energy released,
leaving behind the Jacobian of that change of variable.

Vibrational relaxation inside the excited electronic state is taken to be much faster
than either decay channel, so the excited-state ladder is in thermal equilibrium at
the lattice temperature and each level carries the Boltzmann probability of a single
harmonic mode. The measured rate is the population-weighted sum of the rates out of
every level. At zero temperature only the zero-point level contributes. Carry the sum
over excited levels until the neglected remainder is below one part in 10^12 of the
total. Units have to be handled with care: with w in eV amu^-1/2 Angstrom^-1 and the
squared matrix element in amu Angstrom^2 their product is an energy squared, which
divided by hbar Omega and multiplied by 2 pi / hbar leaves a rate in s^-1.

Use these constant values exactly, so that the result is reproducible to the
precision this step is checked at:

hbar        = 6.582119569e-16 eV s
k_B         = 8.617333262e-5 eV/K

Returns
-------
float, the thermally averaged nonradiative multiphonon transition rate in s^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def multiphonon_rate(delta_q: float, huang_rhys: float, hbar_omega: float,
                     e_zpl: float, coupling: float, degeneracy: float,
                     temperature: float) -> float:
    '''Thermally averaged nonradiative multiphonon transition rate.

    Parameters
    ----------
    delta_q : float
        Mass-weighted coordinate offset in sqrt(amu) Angstrom. Must be positive.
    huang_rhys : float
        Dimensionless electron-phonon coupling strength S. Must be positive.
    hbar_omega : float
        Effective vibrational quantum in eV. Must be positive.
    e_zpl : float
        Zero-phonon line energy in eV. Must be positive.
    coupling : float
        Electron-phonon coupling matrix element in eV amu^-1/2 Angstrom^-1.
        Must be positive.
    degeneracy : float
        Configurational degeneracy of the point defect. Must be positive.
    temperature : float
        Lattice temperature in K. Must lie between 0 and 3000 K, inclusive;
        zero selects the zero-point level alone.

    Returns
    -------
    gamma_nr : float
        The nonradiative multiphonon transition rate in s^-1. Raises ValueError
        on non-finite input, on a non-positive delta_q, huang_rhys, hbar_omega,
        e_zpl, coupling or degeneracy, or when temperature lies outside
        the supported interval from 0 to 3000 K.
    '''
    return gamma_nr

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_multiphonon_rate(delta_q: float, huang_rhys: float, hbar_omega: float,
                             e_zpl: float, coupling: float, degeneracy: float,
                             temperature: float) -> float:
    hbar_evs = 6.582119569e-16        # hbar in eV s
    kb_ev = 8.617333262e-5            # Boltzmann constant in eV / K

    dq = float(delta_q)
    s = float(huang_rhys)
    hw = float(hbar_omega)
    ez = float(e_zpl)
    w = float(coupling)
    g = float(degeneracy)
    t = float(temperature)
    if not all(np.isfinite(v) for v in (dq, s, hw, ez, w, g, t)):
        raise ValueError("all arguments must be finite")
    if dq <= 0.0 or s <= 0.0 or hw <= 0.0 or ez <= 0.0 or w <= 0.0 or g <= 0.0:
        raise ValueError("delta_q, huang_rhys, hbar_omega, e_zpl, coupling and degeneracy must be positive")
    if t < 0.0 or t > 3000.0:
        raise ValueError("temperature must lie between 0 and 3000 K")
    p = ez / hw
    prefactor = 2.0 * np.pi / hbar_evs * g * w * w / hw
    if t == 0.0:
        return float(prefactor * _oracle_coordinate_matrix_element(dq, s, np.array([0]), p)[0])
    x = np.exp(-hw / (kb_ev * t))
    # Each squared element out of level m is bounded by the second moment of the coordinate
    # in that level, dq^2 + dq^2 (2m + 1) / (4 S), so the Boltzmann tail beyond the last
    # level kept has a closed-form bound; double the ladder until that bound is negligible.
    q0_sq = dq * dq / (4.0 * s)
    top = 8
    while True:
        m = np.arange(top + 1)
        weights = (1.0 - x) * x**m
        total = float(np.sum(weights * _oracle_coordinate_matrix_element(dq, s, m, p)))
        k = top + 1
        tail = x**k * (dq * dq + q0_sq * (2.0 * k + 1.0 + 2.0 * x / (1.0 - x)))
        if total > 0.0 and tail < 1e-15 * total:
            return float(prefactor * total)
        top *= 2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {   # boundary: the task centre at zero temperature, the zero-point level alone
            "setup": "",
            "call": "multiphonon_rate(1.020205375219, 3.902705291187, 0.031348254831, 0.761436000001, 0.128, 4.0, 0.0)",
            "gold_call": "_oracle_multiphonon_rate(1.020205375219, 3.902705291187, 0.031348254831, 0.761436000001, 0.128, 4.0, 0.0)",
        },
        {   # normal: the task centre near the temperature where its efficiency halves
            "setup": "",
            "call": "multiphonon_rate(1.020205375219, 3.902705291187, 0.031348254831, 0.761436000001, 0.128, 4.0, 140.0)",
            "gold_call": "_oracle_multiphonon_rate(1.020205375219, 3.902705291187, 0.031348254831, 0.761436000001, 0.128, 4.0, 140.0)",
        },
        {   # normal: a soft, strongly coupled centre at room temperature
            "setup": "",
            "call": "multiphonon_rate(1.3, 6.2, 0.022, 0.41, 0.1, 1.0, 300.0)",
            "gold_call": "_oracle_multiphonon_rate(1.3, 6.2, 0.022, 0.41, 0.1, 1.0, 300.0)",
        },
        {   # edge: very soft mode and strong coupling at 700 K, hundreds of levels populated
            "setup": "",
            "call": "multiphonon_rate(1.45, 8.5, 0.015, 0.45, 0.1, 2.0, 700.0)",
            "gold_call": "_oracle_multiphonon_rate(1.45, 8.5, 0.015, 0.45, 0.1, 2.0, 700.0)",
        },
        {   # edge: a softer mode at 1000 K needs nearly the full 300-level numerical range and a controlled tail
            "setup": "",
            "call": "multiphonon_rate(1.0, 3.9, 0.008, 0.25, 0.1, 4.0, 1000.0) / 1.0e15",
            "gold_call": "_oracle_multiphonon_rate(1.0, 3.9, 0.008, 0.25, 0.1, 4.0, 1000.0) / 1.0e15",
        },
        {   # edge: a small gap close to the relaxation energy, where warming first slows the rate
            "setup": "",
            "call": "multiphonon_rate(0.9, 4.1, 0.03, 0.25, 0.1, 1.0, 60.0)",
            "gold_call": "_oracle_multiphonon_rate(0.9, 4.1, 0.03, 0.25, 0.1, 1.0, 60.0)",
        },
        {   # boundary: at 4 K no excited level above the zero-point one is populated
            "setup": "",
            "call": "multiphonon_rate(1.020205375219, 3.902705291187, 0.031348254831, 0.761436000001, 0.128, 4.0, 4.0)",
            "gold_call": "_oracle_multiphonon_rate(1.020205375219, 3.902705291187, 0.031348254831, 0.761436000001, 0.128, 4.0, 4.0)",
        },
        {   # edge: a negative temperature must raise ValueError
            "setup": """def probe(fn):
    try:
        fn(1.02, 3.9, 0.031, 0.76, 0.128, 4.0, -10.0)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "probe(multiphonon_rate)",
            "gold_call": "probe(_oracle_multiphonon_rate)",
        },
        {   # edge: temperatures above the declared numerical domain must raise ValueError
            "setup": """def probe(fn):
    try:
        fn(1.02, 3.9, 0.031, 0.76, 0.128, 4.0, 3000.1)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "probe(multiphonon_rate)",
            "gold_call": "probe(_oracle_multiphonon_rate)",
        },
    ]
