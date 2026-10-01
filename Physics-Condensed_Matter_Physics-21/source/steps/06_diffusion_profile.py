"""
Evaluate the density-dependent diffusion coefficient that imposes a fixed fractional correlation length, together with the margin by which it satisfies the causality-preserving condition.

Smoothing the bridge by a diffusion process in an unphysical flow time controls its correlation length, and the choice of diffusion coefficient controls how that correlation length varies with density. Because the sound-speed fluctuations evolve towards a Gaussian correlation of width $$\sigma=\sqrt{4D(n)\tau}$$, requiring the correlation length to be a fixed fraction $$c$$ of the density at every density fixes the product $$D(n)\tau=n^2c^2/4$$, so evolving the flow time from zero to one calls for $$D(n)=n^2c^2/4$$.




Smoothing must not push an initially admissible equation of state out of the allowed region. A strictly positive diffusion coefficient is enough to preserve mechanical stability, since a monotonic profile cannot develop a non-monotonicity under the heat equation. Causality is more delicate: recasting the diffusion in terms of the sound speed and asking what happens where the sound speed first touches the speed of light gives the condition $$D''-D'/n\le0$$ across the domain.




This step returns the diffusion coefficient, its first derivative and the value of that causality margin, so that the prescription can be checked rather than assumed.

Returns
-------
np.ndarray, shape (3, N), rows [D, dD/dn, d2D/dn2 - (dD/dn)/n] on density_grid
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def diffusion_profile(density_grid: "np.ndarray", correlation_fraction: float) -> "np.ndarray":
    '''Evaluate the diffusion coefficient and its causality margin on a density grid.

    Parameters
    ----------
    density_grid : np.ndarray
        Finite strictly increasing shape (N,) densities in fm^-3, N >= 3, all
        strictly positive.
    correlation_fraction : float
        Finite positive fractional correlation length, the ratio of the
        correlation length to the density.

    Returns
    -------
    profile : np.ndarray
        Finite shape (3, N). Row 0 is the diffusion coefficient
        density_grid**2 * correlation_fraction**2 / 4, which makes the
        correlation length after unit flow time equal to
        correlation_fraction times the density. Row 1 is its first derivative
        with respect to density. Row 2 is the causality margin, the second
        derivative less the first derivative divided by the density, which must
        not be positive anywhere for the smoothing to preserve causality.

    Raises
    ------
    ValueError
        If the grid is not a finite strictly increasing array of at least three
        positive densities, if the correlation fraction is not a finite
        positive real scalar, or if the evaluated profile is not finite.
    '''
    return profile

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validated_density_grid(density_grid):
    """Validate a density grid and return it as a float array."""
    if not np.isrealobj(density_grid):
        raise ValueError("the density grid must be real")
    try:
        grid = np.asarray(density_grid, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("the density grid must be a finite real array") from exc
    if grid.ndim != 1 or grid.size < 3:
        raise ValueError("the density grid must be one-dimensional with at least three points")
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValueError("the density grid must be finite and strictly positive")
    if not np.all(np.diff(grid) > 0.0):
        raise ValueError("the density grid must be strictly increasing")
    return grid


def _oracle_diffusion_profile(density_grid: "np.ndarray",
                              correlation_fraction: float) -> "np.ndarray":
    grid = _validated_density_grid(density_grid)
    if not np.isscalar(correlation_fraction) or not np.isrealobj(correlation_fraction):
        raise ValueError("the correlation fraction must be a real scalar")
    fraction = float(correlation_fraction)
    if not np.isfinite(fraction) or fraction <= 0.0:
        raise ValueError("the correlation fraction must be finite and positive")
    coefficient = grid**2 * fraction**2 / 4.0
    first = grid * fraction**2 / 2.0
    second = np.full_like(grid, fraction**2 / 2.0)
    profile = np.stack((coefficient, first, second - first / grid))
    if not np.all(np.isfinite(profile)):
        raise ValueError("the evaluated diffusion profile must be finite")
    return profile

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    guard = ('def run_model():\n'
             '    try:\n'
             '        diffusion_profile(grid.copy(), c)\n'
             '        return 0\n'
             '    except ValueError:\n'
             '        return 1\n'
             'def run_oracle():\n'
             '    try:\n'
             '        _oracle_diffusion_profile(grid.copy(), c)\n'
             '        return 0\n'
             '    except ValueError:\n'
             '        return 1\n')
    return [
        # --- Normal: the benchmark grid and correlation fraction ---
        {"setup": "import numpy as np\ngrid = np.linspace(0.32, 4.80, 201)\nc = 0.20\n",
         "call": "diffusion_profile(grid.copy(), c)",
         "gold_call": "_oracle_diffusion_profile(grid.copy(), c)", "tol": 1e-12},
        # --- Normal: a weaker correlation fraction scales the coefficient quadratically ---
        {"setup": "import numpy as np\ngrid = np.linspace(0.32, 4.80, 201)\nc = 0.05\n",
         "call": "diffusion_profile(grid.copy(), c)",
         "gold_call": "_oracle_diffusion_profile(grid.copy(), c)", "tol": 1e-12},
        # --- Boundary: the smallest admissible grid ---
        {"setup": "import numpy as np\ngrid = np.array([0.32, 2.50, 4.80])\nc = 0.20\n",
         "call": "diffusion_profile(grid.copy(), c)",
         "gold_call": "_oracle_diffusion_profile(grid.copy(), c)", "tol": 1e-12},
        # --- Normal: a non-uniform grid, where the derivative rows still follow the form ---
        {"setup": "import numpy as np\ngrid = np.geomspace(0.32, 4.80, 64)\nc = 0.40\n",
         "call": "diffusion_profile(grid.copy(), c)",
         "gold_call": "_oracle_diffusion_profile(grid.copy(), c)", "tol": 1e-12},
        # --- Invalid: a non-positive correlation fraction must raise ValueError ---
        {"setup": "import numpy as np\ngrid = np.linspace(0.32, 4.80, 51)\nc = 0.0\n" + guard,
         "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: a grid that is not strictly increasing must raise ValueError ---
        {"setup": "import numpy as np\ngrid = np.array([0.32, 2.50, 2.50, 4.80])\nc = 0.20\n" + guard,
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
