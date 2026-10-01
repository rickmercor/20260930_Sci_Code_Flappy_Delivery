"""
Uniform cosmic time grid and its redshift mapping

The deconvolution operates on histories sampled uniformly in cosmic time, because the delay time convolution is a time domain operation. This step builds $$n_{\rm samples}$$ uniformly spaced cosmic-time samples spanning $$[0,t_0]$$, where $$t_0$$ is the age of the universe today, and maps every sample to redshift by inverting the age-redshift relation of the adopted flat $$\Lambda$$CDM cosmology, with $$H_0 = 67.7\ \mathrm{km\,s^{-1}\,Mpc^{-1}}$$, $$\Omega_m = 0.31$$. The inversion must be numerically accurate to better than $$10^{-6}$$ in $$z$$ over the table range; any sufficiently accurate method is acceptable. Samples earlier than the age at $$z = 60$$ map to $$z = 60$$. The merger rate model is constant there, so this cap does not affect the result.

Returns
-------
np.ndarray of shape (2, n_samples): row 0 the cosmic-time samples in Gyr, row 1 the corresponding redshifts
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_time_redshift_grid(n_samples: int) -> np.ndarray:
    '''Uniform cosmic time grid on [0, t0] with its redshift mapping.

    Parameters
    ----------
    n_samples : int
        Number of uniformly spaced cosmic time samples, must be an
        integer >= 16.

    Returns
    -------
    grid : np.ndarray
        Array of shape (2, n_samples): row 0 holds the cosmic time samples
        t in Gyr, uniformly spaced from 0 to t0 inclusive; row 1 holds the
        corresponding redshifts z(t), capped at z = 60.

    Raises
    ------
    ValueError
        If n_samples is not an integer (booleans and floats are rejected),
        or n_samples < 16.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad, cumulative_simpson


def _oracle_build_time_redshift_grid(n_samples: int) -> np.ndarray:
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise ValueError("n_samples must be an integer >= 16")
    if n_samples < 16:
        raise ValueError("n_samples must be an integer >= 16")
    t0 = _oracle_compute_cosmic_age(0.0)
    t_cap = _oracle_compute_cosmic_age(60.0)
    t_grid = np.linspace(0.0, t0, int(n_samples))
    omega_m = 0.31
    omega_l = 1.0 - omega_m
    hubble_time = 977.79222168 / 67.7
    active = t_grid > t_cap
    z_grid = np.full(t_grid.shape, 60.0)
    scaled_time = 1.5 * np.sqrt(omega_l) * t_grid[active] / hubble_time
    z_grid[active] = (np.sqrt(omega_l / omega_m) / np.sinh(scaled_time)) ** (2.0 / 3.0) - 1.0
    z_grid[-1] = 0.0
    return np.vstack([t_grid, z_grid])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: moderate grid, full array comparison ---
        {
            "setup": "import numpy as np\nn = 64\n",
            "tol": 1e-6,
            "call": 'build_time_redshift_grid(n)',
            "gold_call": '_oracle_build_time_redshift_grid(n)',
        },
        # --- Normal: endpoints (t starts at 0, ends at the age today) ---
        {
            "setup": "import numpy as np\nn = 128\n",
            "tol": 1e-6,
            "call": 'build_time_redshift_grid(n)[:, [0, -1]]',
            "gold_call": '_oracle_build_time_redshift_grid(n)[:, [0, -1]]',
        },
        # --- Boundary: smallest allowed grid ---
        {
            "setup": "import numpy as np\nn = 16\n",
            "tol": 1e-6,
            "call": 'build_time_redshift_grid(n)',
            "gold_call": '_oracle_build_time_redshift_grid(n)',
        },
        # --- Edge: too-small grid must raise ValueError ---
        {
            "setup": """import numpy as np
n = 8
def run_model():
    try:
        build_time_redshift_grid(n)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_build_time_redshift_grid(n)
        return 0
    except ValueError:
        return 1
""",
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        # --- Edge: non-integer grid size must raise ValueError ---
        {
            "setup": """import numpy as np
n = 64.5
def run_model():
    try:
        build_time_redshift_grid(n)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_build_time_redshift_grid(n)
        return 0
    except ValueError:
        return 1
""",
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
