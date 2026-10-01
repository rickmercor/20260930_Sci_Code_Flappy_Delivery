"""
Evaluate the purely electronic electric-dipole emission rate of the defect radiating inside the dielectric host.

Stripped of every vibrational effect, the optical activity of a point defect is
the spontaneous emission rate of a pure electric dipole radiating inside a
dielectric,

Gamma_R^0 = (E_eff / E_0)^2  n E_ZPL^3 |mu|^2 / (3 pi eps_0 hbar^4 c^3),

evaluated at the zero-phonon line energy and therefore ignoring the phonon
induced lowering of the emitted photon energy. The cubic dependence on the
transition energy is the usual density-of-photon-states factor, and the linear
refractive index n accounts for the reduced phase velocity in the host. The
prefactor (E_eff / E_0) is the ratio of the electric field felt at the defect
to the field in the bulk, which is the local field correction; screening studies
of colour centres conventionally set it to unity rather than adopting a cavity
model, and the same convention is used here. The transition dipole moment
arrives in debye, so it must be converted to coulomb metre, and the transition
energy in eV must be converted to joule, before the SI expression is evaluated.
Getting this rate right matters twice over, since it sets both the zero-phonon
emission rate and the scale against which the nonradiative channel competes.

Use these constant values exactly, so that the result is reproducible to the
precision this step is checked at:

hbar        = 1.054571817e-34 J s
1 eV        = 1.602176634e-19 J
eps_0       = 8.8541878128e-12 F/m
c           = 2.99792458e8 m/s
1 debye     = 3.33564095198e-30 C m

Returns
-------
float, the purely electronic electric-dipole emission rate in s^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dipole_transition_rate(dipole_debye: float, refractive_index: float,
                           e_zpl: float, field_ratio: float = 1.0) -> float:
    '''Purely electronic electric-dipole emission rate of a point defect.

    Parameters
    ----------
    dipole_debye : float
        Magnitude of the transition dipole moment in debye. Must be positive.
    refractive_index : float
        Refractive index of the host at the emission wavelength. Must be positive.
    e_zpl : float
        Zero-phonon line energy in eV. Must be positive.
    field_ratio : float, optional
        Ratio of the effective electric field at the defect to the field in the
        bulk. Defaults to unity.

    Returns
    -------
    gamma_r0 : float
        The purely electronic radiative transition rate in s^-1. Raises
        ValueError on non-finite input or on a non-positive dipole moment,
        refractive index or transition energy.
    '''
    return gamma_r0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_dipole_transition_rate(dipole_debye: float, refractive_index: float,
                                   e_zpl: float, field_ratio: float = 1.0) -> float:
    hbar_si = 1.054571817e-34         # J s
    ev_j = 1.602176634e-19            # J per eV
    eps0_si = 8.8541878128e-12        # F / m
    c_si = 2.99792458e8               # m / s
    debye_cm = 3.33564095198e-30      # C m per debye

    mu_d = float(dipole_debye)
    n_r = float(refractive_index)
    e_z = float(e_zpl)
    f_r = float(field_ratio)
    if not all(np.isfinite(v) for v in (mu_d, n_r, e_z, f_r)):
        raise ValueError("all arguments must be finite")
    if mu_d <= 0.0 or n_r <= 0.0 or e_z <= 0.0:
        raise ValueError("dipole moment, refractive index and e_zpl must be positive")
    mu_si = mu_d * debye_cm
    energy = e_z * ev_j
    numerator = f_r * f_r * n_r * energy**3 * mu_si**2
    denominator = 3.0 * np.pi * eps0_si * hbar_si**4 * c_si**3
    return float(numerator / denominator)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {   # normal: the antimony-vacancy centre of the task
            "setup": "import numpy as np\n",
            "call": "round(dipole_transition_rate(2.85, 2.5732, 0.761436000001), 3)",
            "gold_call": "round(_oracle_dipole_transition_rate(2.85, 2.5732, 0.761436000001), 3)",
        },
        {   # normal: a visible emitter in diamond
            "setup": "import numpy as np\n",
            "call": "round(dipole_transition_rate(5.2, 2.4, 1.945), 3)",
            "gold_call": "round(_oracle_dipole_transition_rate(5.2, 2.4, 1.945), 3)",
        },
        {   # boundary: the local field correction enters squared
            "setup": "import numpy as np\n",
            "call": "round(dipole_transition_rate(3.0, 2.6, 0.9, 1.5), 3)",
            "gold_call": "round(_oracle_dipole_transition_rate(3.0, 2.6, 0.9, 1.5), 3)",
        },
        {   # boundary: a very weak dipole in the far infrared
            "setup": "import numpy as np\n",
            "call": "float(np.round(np.log10(dipole_transition_rate(0.02, 1.0, 0.15)), 10))",
            "gold_call": "float(np.round(np.log10(_oracle_dipole_transition_rate(0.02, 1.0, 0.15)), 10))",
        },
        {   # edge: a non-positive transition energy must raise ValueError
            "setup": """import numpy as np
def probe(fn):
    try:
        fn(2.85, 2.6, 0.0)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "probe(dipole_transition_rate)",
            "gold_call": "probe(_oracle_dipole_transition_rate)",
        },
        {   # edge: a non-finite refractive index must raise ValueError
            "setup": """import numpy as np
def probe(fn):
    try:
        fn(2.85, float("nan"), 0.9)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "probe(dipole_transition_rate)",
            "gold_call": "probe(_oracle_dipole_transition_rate)",
        },
    ]
