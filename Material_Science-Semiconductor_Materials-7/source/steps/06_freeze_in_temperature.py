"""
Return the temperature at which a defect species stops equilibrating during linear cooling of a polycrystalline film, given its migration energy, its diffusion prefactor, the cooling rate and the grain size.

Grain boundaries act as the sources and sinks through which a defect population adjusts its size. Once the distance a defect can still diffuse during the rest of the cooldown becomes too short to reach them, its total concentration can no longer change and is frozen in. Species with large migration barriers freeze in near the start of cooling, while fast interstitials stay mobile to much lower temperatures.

Returns
-------
float, the freeze-in temperature (K), within [t_min, t_max]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def freeze_in_temperature(em: float, d0: float, gamma: float, t_max: float, t_min: float, grain_size: float) -> float:
    '''Freeze-in temperature of one defect species under linear cooling.

    Parameters
    ----------
    em : float
        Migration energy (eV) of the defect, strictly positive.
    d0 : float
        Arrhenius diffusion prefactor (cm^2/s), strictly positive.
    gamma : float
        Constant cooling rate (K/s), strictly positive.
    t_max : float
        Starting temperature of the cooling path (K).
    t_min : float
        Final temperature of the cooling path (K), with 0 < t_min < t_max.
    grain_size : float
        Grain size of the film (cm), strictly positive.

    Returns
    -------
    t_freeze : float
        Freeze-in temperature (K) under the source framework's freeze-in criterion, using its
        convention relating the source/sink distance to the grain size. The value lies in
        [t_min, t_max]; it equals t_max when the criterion is already met at the start of cooling.
        Temperatures convert to energies with k_B = 8.617333262e-5 eV/K (CODATA 2018).

    Raises
    ------
    ValueError
        If em, d0, gamma or grain_size is not positive, or the temperatures do not satisfy
        0 < t_min < t_max.
    '''
    return t_freeze

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _oracle_freeze_in_temperature(em: float, d0: float, gamma: float, t_max: float, t_min: float, grain_size: float) -> float:
    if not (float(em) > 0.0 and float(d0) > 0.0 and float(gamma) > 0.0 and float(grain_size) > 0.0):
        raise ValueError("em, d0, gamma and grain_size must be positive")
    t_hi = float(t_max)
    t_lo = float(t_min)
    if not 0.0 < t_lo < t_hi:
        raise ValueError("temperatures must satisfy 0 < t_min < t_max")
    sink_distance = 0.5 * float(grain_size)

    def _excess(t):
        return _oracle_remaining_diffusion_length(t, em, d0, gamma, t_lo) - sink_distance

    if _excess(t_hi) <= 0.0:
        return t_hi
    return float(brentq(_excess, t_lo, t_hi, xtol=1e-10, rtol=1e-15, maxiter=500))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    raise_setup = """import numpy as np
def run(fn, *args):
    try:
        fn(*args)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Valid: fast interstitial in a few-micron film ---
        {
            "setup": "import numpy as np\n",
            "call": "freeze_in_temperature(0.97, 0.25, 0.73, 1123.0, 296.0, 3.8e-4)",
            "gold_call": "_oracle_freeze_in_temperature(0.97, 0.25, 0.73, 1123.0, 296.0, 3.8e-4)",
            "tol": 1e-6,
        },
        # --- Valid: slow complex in the same film ---
        {
            "setup": "import numpy as np\n",
            "call": "freeze_in_temperature(1.86, 0.25, 0.73, 1123.0, 296.0, 3.8e-4)",
            "gold_call": "_oracle_freeze_in_temperature(1.86, 0.25, 0.73, 1123.0, 296.0, 3.8e-4)",
            "tol": 1e-6,
        },
        # --- Valid: slow cooling of a coarse-grained sample ---
        {
            "setup": "import numpy as np\n",
            "call": "freeze_in_temperature(1.43, 0.37, 0.0062, 1150.0, 310.0, 2.7e-2)",
            "gold_call": "_oracle_freeze_in_temperature(1.43, 0.37, 0.0062, 1150.0, 310.0, 2.7e-2)",
            "tol": 1e-6,
        },
        # --- Boundary: criterion already met at the start of cooling ---
        {
            "setup": "import numpy as np\n",
            "call": "freeze_in_temperature(2.64, 0.25, 0.73, 1123.0, 296.0, 3.8e-4)",
            "gold_call": "_oracle_freeze_in_temperature(2.64, 0.25, 0.73, 1123.0, 296.0, 3.8e-4)",
            "tol": 1e-6,
        },
        # --- Edge: very mobile species that freezes just above the end of the path ---
        {
            "setup": "import numpy as np\n",
            "call": "freeze_in_temperature(0.45, 0.25, 0.73, 1123.0, 296.0, 3.8e-4)",
            "gold_call": "_oracle_freeze_in_temperature(0.45, 0.25, 0.73, 1123.0, 296.0, 3.8e-4)",
            "tol": 1e-6,
        },
        # --- Invalid: non-positive grain size ---
        {
            "setup": raise_setup,
            "call": "run(freeze_in_temperature, 0.97, 0.25, 0.73, 1123.0, 296.0, 0.0)",
            "gold_call": "run(_oracle_freeze_in_temperature, 0.97, 0.25, 0.73, 1123.0, 296.0, 0.0)",
            "tol": 1e-6,
        },
    ]
