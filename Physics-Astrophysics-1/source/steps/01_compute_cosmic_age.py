"""
Cosmic age at a given redshift

The convolution between the progenitor formation history and the merger-rate history of binary black holes acts in cosmic time, while gravitational-wave catalogs report rates as functions of redshift. The first ingredient of the pipeline is therefore the age of the universe $$t(z)$$ for the adopted flat $$\Lambda$$CDM cosmology, with $$H_0 = 67.7\ \mathrm{km\,s^{-1}\,Mpc^{-1}}$$, $$\Omega_m = 0.31$$, and radiation neglected:




$$t(z) = \frac{1}{H_0}\int_z^{\infty}\frac{dz'}{(1+z')E(z')}$$,




$$E(z) = \sqrt{\Omega_m(1+z)^3 + (1-\Omega_m)}$$.




The result is expressed in Gyr using $$\frac{1}{\mathrm{km\,s^{-1}\,Mpc^{-1}}} = 977.79222168\ \mathrm{Gyr}$$.

Returns
-------
float, the cosmic age $$t(z)$$ in Gyr as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_cosmic_age(z: float) -> float:
    '''Age of the universe at redshift z for flat LCDM.

    Parameters
    ----------
    z : float
        Redshift, must be a finite scalar with z >= 0.

    Returns
    -------
    t : float
        Cosmic age t(z) in Gyr for H0 = 67.7 km/s/Mpc, Omega_m = 0.31,
        flat geometry, radiation neglected.

    Raises
    ------
    ValueError
        If z is not a finite scalar, or z < 0.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad


def _oracle_compute_cosmic_age(z: float) -> float:
    if not np.isscalar(z) or isinstance(z, (bool, np.bool_, str)) or not np.isrealobj(z):
        raise ValueError("z must be a finite scalar with z >= 0")
    z = float(z)
    if not np.isfinite(z) or z < 0.0:
        raise ValueError("z must be a finite scalar with z >= 0")
    _H0_KM_S_MPC, _OMEGA_M, _GYR_PER_INV_KM_S_MPC = 67.7, 0.31, 977.79222168
    omega_l = 1.0 - _OMEGA_M

    inv_power = (1.0 + z) ** -1.5
    age = 2.0 * np.arcsinh(np.sqrt(omega_l / _OMEGA_M) * inv_power)
    age /= 3.0 * np.sqrt(omega_l)
    return float(age * _GYR_PER_INV_KM_S_MPC / _H0_KM_S_MPC)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: age today (z = 0) ---
        {
            "setup": "import numpy as np\nz = 0.0\n",
            "call": 'compute_cosmic_age(z)',
            "gold_call": '_oracle_compute_cosmic_age(z)',
        },
        # --- Normal: interior redshift used by the pipeline window ---
        {
            "setup": "import numpy as np\nz = 1.5\n",
            "call": 'compute_cosmic_age(z)',
            "gold_call": '_oracle_compute_cosmic_age(z)',
        },
        # --- Boundary: high redshift, age approaches zero ---
        {
            "setup": "import numpy as np\nz = 20.0\n",
            "call": 'compute_cosmic_age(z)',
            "gold_call": '_oracle_compute_cosmic_age(z)',
        },
        # --- Edge: negative redshift must raise ValueError ---
        {
            "setup": """import numpy as np
z = -0.5
def run_model():
    try:
        compute_cosmic_age(z)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_compute_cosmic_age(z)
        return 0
    except ValueError:
        return 1
""",
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        # --- Edge: non-finite redshift must raise ValueError ---
        {
            "setup": """import numpy as np
z = np.nan
def run_model():
    try:
        compute_cosmic_age(z)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_compute_cosmic_age(z)
        return 0
    except ValueError:
        return 1
""",
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
