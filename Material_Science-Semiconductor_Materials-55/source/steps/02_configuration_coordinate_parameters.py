"""
Turn three total energies and the coordinate offset into the zero-phonon line energy, the relaxation energy and the effective vibrational quantum.

Within the single effective mode picture the two electronic states share one
harmonic force constant and differ only by an offset along the mass-weighted
coordinate and by the purely electronic energy,

E_GS(Q) = (1/2) Omega^2 Q^2,
E_ES(Q) = (1/2) Omega^2 (Q - delta_Q)^2 + E_ZPL.

Fitting a curvature normally means sampling the potential at many displaced
geometries, which is the expensive part of screening. Two parabolas that share
a curvature are instead pinned by their vertices plus a single extra point, so
three total energies suffice: the relaxed ground state, the relaxed excited
state, and the ground-state electronic configuration evaluated at the relaxed
excited-state geometry. The first two vertices fix the zero-phonon line energy.
The third point sits on the ground-state parabola at Q = delta_Q, so its
elevation above the ground-state minimum is the relaxation energy that the
lattice releases after a vertical transition. That elevation, together with the
offset delta_Q, is what fixes the curvature the two wells share, and therefore
the effective vibrational quantum hbar Omega. Consistent units matter here:
delta_Q arrives in
sqrt(amu) Angstrom while the energies are in eV, so the conversion to SI has to
be carried through before hbar Omega is read back out in eV.

Use these constant values exactly, so that the result is reproducible to the
precision this step is checked at:

hbar        = 1.054571817e-34 J s
1 eV        = 1.602176634e-19 J
1 amu       = 1.66053906660e-27 kg
1 Angstrom  = 1e-10 m

Returns
-------
numpy array of three floats, [E_ZPL, hbar_Omega, E_rel] in eV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def configuration_coordinate_parameters(e_ground_relaxed: float, e_excited_relaxed: float,
                                        e_ground_at_excited: float, delta_q: float) -> np.ndarray:
    '''Zero-phonon line energy, effective phonon quantum and relaxation energy.

    Parameters
    ----------
    e_ground_relaxed : float
        Total energy of the relaxed electronic ground state in eV.
    e_excited_relaxed : float
        Total energy of the relaxed electronic excited state in eV.
    e_ground_at_excited : float
        Total energy of the ground-state electronic configuration evaluated at
        the relaxed excited-state geometry, in eV.
    delta_q : float
        Mass-weighted coordinate offset between the two relaxed geometries in
        sqrt(amu) Angstrom. Must be positive.

    Returns
    -------
    parameters : numpy.ndarray, shape (3,)
        [E_ZPL, hbar_Omega, E_rel] in eV, where E_ZPL is the zero-phonon line
        energy, hbar_Omega the effective vibrational quantum obtained from the
        harmonic ground-state curve, and E_rel the relaxation energy released
        along that curve. Raises ValueError on non-finite input, on a
        non-positive delta_q, or when the energies do not describe an excited
        state above the ground state with a positive relaxation energy.
    '''
    return parameters

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_configuration_coordinate_parameters(e_ground_relaxed: float, e_excited_relaxed: float,
                                                e_ground_at_excited: float, delta_q: float) -> np.ndarray:
    hbar_si = 1.054571817e-34        # J s
    ev_j = 1.602176634e-19           # J per eV
    amu_kg = 1.66053906660e-27       # kg per amu
    ang_m = 1.0e-10                  # m per Angstrom

    eg = float(e_ground_relaxed)
    ee = float(e_excited_relaxed)
    ev = float(e_ground_at_excited)
    dq = float(delta_q)
    if not all(np.isfinite(v) for v in (eg, ee, ev, dq)):
        raise ValueError("all energies and delta_q must be finite")
    if dq <= 0.0:
        raise ValueError("delta_q must be positive")
    e_zpl = ee - eg
    e_rel = ev - eg
    if e_zpl <= 0.0:
        raise ValueError("the excited state must lie above the relaxed ground state")
    if e_rel <= 0.0:
        raise ValueError("the relaxation energy must be positive")
    dq_si = dq * np.sqrt(amu_kg) * ang_m
    omega = np.sqrt(2.0 * e_rel * ev_j) / dq_si
    hbar_omega = hbar_si * omega / ev_j
    return np.array([e_zpl, hbar_omega, e_rel], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {   # normal: the antimony-vacancy centre of the task
            "setup": "import numpy as np\nEG, EE, EV_ = -8912.475160, -8911.713724, -8912.352817\ndq = 1.020205375219\n",
            "call": "np.round(configuration_coordinate_parameters(EG, EE, EV_, dq), 12).tolist()",
            "gold_call": "np.round(_oracle_configuration_coordinate_parameters(EG, EE, EV_, dq), 12).tolist()",
        },
        {   # normal: a stiffer, more weakly coupled centre
            "setup": "import numpy as np\n",
            "call": "np.round(configuration_coordinate_parameters(-1204.0, -1202.05, -1203.94, 0.66), 12).tolist()",
            "gold_call": "np.round(_oracle_configuration_coordinate_parameters(-1204.0, -1202.05, -1203.94, 0.66), 12).tolist()",
        },
        {   # boundary: an absolute shift of the energy zero must change nothing
            "setup": "import numpy as np\nSH = 5000.0\n",
            "call": "np.round(configuration_coordinate_parameters(-8912.475160 - SH, -8911.713724 - SH, -8912.352817 - SH, 1.020205375219), 12).tolist()",
            "gold_call": "np.round(_oracle_configuration_coordinate_parameters(-8912.475160, -8911.713724, -8912.352817, 1.020205375219), 12).tolist()",
        },
        {   # boundary: an almost rigid lattice leaves a very soft effective mode
            "setup": "import numpy as np\n",
            "call": "float(np.round(np.log10(configuration_coordinate_parameters(0.0, 0.8, 1.0e-8, 1.0)[1]), 10))",
            "gold_call": "float(np.round(np.log10(_oracle_configuration_coordinate_parameters(0.0, 0.8, 1.0e-8, 1.0)[1]), 10))",
        },
        {   # edge: an excited state below the ground state must raise ValueError
            "setup": """import numpy as np
def probe(fn):
    try:
        fn(-100.0, -101.0, -99.9, 1.0)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "probe(configuration_coordinate_parameters)",
            "gold_call": "probe(_oracle_configuration_coordinate_parameters)",
        },
        {   # edge: a third energy below the relaxed ground state must raise ValueError
            "setup": """import numpy as np
def probe(fn):
    try:
        fn(-100.0, -99.0, -100.5, 1.0)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "probe(configuration_coordinate_parameters)",
            "gold_call": "probe(_oracle_configuration_coordinate_parameters)",
        },
    ]
