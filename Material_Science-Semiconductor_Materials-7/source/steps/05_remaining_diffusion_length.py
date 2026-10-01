"""
Return the diffusion length a defect can still accumulate while the sample cools at a constant rate from a given temperature down to the end of the cooling path.

A defect with Arrhenius diffusivity keeps exchanging with sources and sinks only while it can still travel the distance separating them. During continuous cooling the diffusivity falls steeply, so the distance a defect can still cover from the current temperature onward decreases monotonically as the sample cools. This remaining diffusion length is the kinetic quantity that decides when a defect species stops equilibrating.

Returns
-------
float, the root-mean-square diffusion length (cm) still available between temperature and t_min
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def remaining_diffusion_length(temperature: float, em: float, d0: float, gamma: float, t_min: float) -> float:
    '''Diffusion length still available during linear cooling from temperature to t_min.

    Parameters
    ----------
    temperature : float
        Current temperature (K), with temperature >= t_min.
    em : float
        Migration energy (eV) of the defect's Arrhenius diffusivity, strictly positive.
    d0 : float
        Arrhenius diffusion prefactor (cm^2/s), strictly positive.
    gamma : float
        Constant cooling rate (K/s), strictly positive; the temperature falls linearly in time.
    t_min : float
        Final temperature of the cooling path (K), strictly positive.

    Returns
    -------
    length : float
        Root-mean-square diffusion length (cm), as defined in the source framework,
        accumulated between the current temperature and t_min. It is zero when
        temperature equals t_min.
        Temperatures convert to energies with k_B = 8.617333262e-5 eV/K (CODATA 2018).

    Raises
    ------
    ValueError
        If em, d0, gamma or t_min is not positive, or temperature < t_min.
    '''
    return length

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import expi

def _oracle_remaining_diffusion_length(temperature: float, em: float, d0: float, gamma: float, t_min: float) -> float:
    kb_ev = 8.617333262e-5  # Boltzmann constant (eV/K)
    t = float(temperature)
    t_lo = float(t_min)
    if not (float(em) > 0.0 and float(d0) > 0.0 and float(gamma) > 0.0 and t_lo > 0.0):
        raise ValueError("em, d0, gamma and t_min must be positive")
    if t < t_lo:
        raise ValueError("temperature must not be below t_min")
    a = float(em) / kb_ev

    def _antiderivative(tt):
        return tt * np.exp(-a / tt) + a * expi(-a / tt)

    x2 = float(d0) / float(gamma) * (_antiderivative(t) - _antiderivative(t_lo))
    return float(np.sqrt(max(x2, 0.0)))

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
        # --- Valid: fast interstitial from an intermediate temperature ---
        {
            "setup": "import numpy as np\n",
            "call": "remaining_diffusion_length(640.0, 0.97, 0.37, 0.73, 296.0) * 1e4",
            "gold_call": "_oracle_remaining_diffusion_length(640.0, 0.97, 0.37, 0.73, 296.0) * 1e4",
            "tol": 1e-7,
        },
        # --- Valid: slow complex from the top of the cooling path ---
        {
            "setup": "import numpy as np\n",
            "call": "remaining_diffusion_length(1123.0, 1.86, 0.12, 0.73, 296.0) * 1e4",
            "gold_call": "_oracle_remaining_diffusion_length(1123.0, 1.86, 0.12, 0.73, 296.0) * 1e4",
            "tol": 1e-7,
        },
        # --- Valid: slow cooling and small prefactor ---
        {
            "setup": "import numpy as np\n",
            "call": "remaining_diffusion_length(905.0, 1.43, 3.1e-3, 0.0062, 310.0) * 1e4",
            "gold_call": "_oracle_remaining_diffusion_length(905.0, 1.43, 3.1e-3, 0.0062, 310.0) * 1e4",
            "tol": 1e-7,
        },
        # --- Boundary: current temperature equals the end of the path ---
        {
            "setup": "import numpy as np\n",
            "call": "remaining_diffusion_length(296.0, 0.97, 0.37, 0.73, 296.0)",
            "gold_call": "_oracle_remaining_diffusion_length(296.0, 0.97, 0.37, 0.73, 296.0)",
            "tol": 1e-7,
        },
        # --- Edge: just above the end of the path (length small but finite) ---
        {
            "setup": "import numpy as np\n",
            "call": "remaining_diffusion_length(301.5, 0.62, 0.61, 0.73, 296.0) * 1e4",
            "gold_call": "_oracle_remaining_diffusion_length(301.5, 0.62, 0.61, 0.73, 296.0) * 1e4",
            "tol": 1e-7,
        },
        # --- Invalid: temperature below the end of the path ---
        {
            "setup": raise_setup,
            "call": "run(remaining_diffusion_length, 250.0, 0.97, 0.37, 0.73, 296.0)",
            "gold_call": "run(_oracle_remaining_diffusion_length, 250.0, 0.97, 0.37, 0.73, 296.0)",
            "tol": 1e-7,
        },
    ]
