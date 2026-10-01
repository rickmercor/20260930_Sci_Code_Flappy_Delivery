"""
Evaluate the squared speed of sound of a zero-temperature equation of state given as chemical potential against number density.

For cold matter at zero temperature the thermodynamic relations reduce to a pair of differentials: the pressure changes with the chemical potential as $$dp=n\,d\mu$$, and the energy density changes with the density as $$d\varepsilon=\mu\,dn$$, consistent with $$\varepsilon=-p+\mu n$$. The squared speed of sound is the derivative of pressure with respect to energy density, so dividing one relation by the other leaves it expressed entirely through the chemical potential and its density derivative.




The result, $$c_s^2=\frac{n}{\mu}\frac{d\mu}{dn}$$, is the reciprocal of the combination appearing in the causality condition, so demanding a subluminal sound speed is the same as demanding that the chemical potential not rise too steeply with density. A vanishing derivative corresponds to a vanishing sound speed, which marks a first-order phase transition in this representation.




Evaluating it numerically requires only a derivative on the given grid, but the result is sensitive to structure in the chemical potential at the scale of the grid spacing, which is why a smoothed bridge is differentiated rather than a raw one.

Returns
-------
np.ndarray, shape (N,), dimensionless c_s**2 on density_grid
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sound_speed_squared(density_grid: "np.ndarray",
                        chemical_potential: "np.ndarray") -> "np.ndarray":
    '''Evaluate the squared speed of sound on a density grid.

    Parameters
    ----------
    density_grid : np.ndarray
        Finite strictly increasing shape (N,) densities in fm^-3, N >= 3, all
        strictly positive.
    chemical_potential : np.ndarray
        Finite strictly positive shape (N,) chemical potentials in GeV on that
        grid.

    Returns
    -------
    speeds : np.ndarray
        Finite dimensionless shape (N,), the squared speed of sound, evaluated
        as the density divided by the chemical potential, times the derivative
        of the chemical potential with respect to density. The derivative uses
        second-order central differences in the interior and second-order
        one-sided differences at the two ends.

    Raises
    ------
    ValueError
        If the grid is not a finite strictly increasing array of at least three
        positive densities, if the chemical potential is not a finite positive
        array of matching shape, or if the evaluated speeds are not finite.
    '''
    return speeds

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sound_speed_squared(density_grid: "np.ndarray",
                                chemical_potential: "np.ndarray") -> "np.ndarray":
    grid = _validated_density_grid(density_grid)
    if not np.isrealobj(chemical_potential):
        raise ValueError("the chemical potential must be real")
    try:
        mu = np.asarray(chemical_potential, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("the chemical potential must be a finite real array") from exc
    if mu.shape != grid.shape or not np.all(np.isfinite(mu)) or np.any(mu <= 0.0):
        raise ValueError("the chemical potential must be finite, positive and match the grid")
    speeds = grid / mu * np.gradient(mu, grid, edge_order=2)
    if not np.all(np.isfinite(speeds)):
        raise ValueError("the evaluated sound speeds must be finite")
    return speeds

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    guard = ('def run_model():\n'
             '    try:\n'
             '        sound_speed_squared(grid.copy(), mu.copy())\n'
             '        return 0\n'
             '    except ValueError:\n'
             '        return 1\n'
             'def run_oracle():\n'
             '    try:\n'
             '        _oracle_sound_speed_squared(grid.copy(), mu.copy())\n'
             '        return 0\n'
             '    except ValueError:\n'
             '        return 1\n')
    return [
        # --- Normal: a straight bridge, where the sound speed varies smoothly ---
        {"setup": ("import numpy as np\ngrid = np.linspace(0.32, 4.80, 201)\n"
                   "mu = 1.00 + 1.60*(grid - 0.32)/(4.80 - 0.32)\n"),
         "call": "sound_speed_squared(grid.copy(), mu.copy())",
         "gold_call": "_oracle_sound_speed_squared(grid.copy(), mu.copy())", "tol": 1e-12},
        # --- Normal: a rippled bridge, where structure feeds straight into the derivative ---
        {"setup": ("import numpy as np\ngrid = np.linspace(0.32, 4.80, 201)\n"
                   "mu = 1.00 + 1.60*(grid - 0.32)/(4.80 - 0.32) + 0.02*np.sin(11.0*(grid - 0.32))\n"),
         "call": "sound_speed_squared(grid.copy(), mu.copy())",
         "gold_call": "_oracle_sound_speed_squared(grid.copy(), mu.copy())", "tol": 1e-12},
        # --- Edge: a locally flat chemical potential, giving a vanishing sound speed ---
        {"setup": ("import numpy as np\ngrid = np.linspace(0.32, 4.80, 201)\n"
                   "mu = 1.00 + 1.60*np.clip((grid - 1.50)/(4.80 - 1.50), 0.0, None)\n"),
         "call": "sound_speed_squared(grid.copy(), mu.copy())",
         "gold_call": "_oracle_sound_speed_squared(grid.copy(), mu.copy())", "tol": 1e-12},
        # --- Boundary: the smallest admissible grid, exercising the one-sided ends ---
        {"setup": ("import numpy as np\ngrid = np.array([0.32, 2.50, 4.80])\n"
                   "mu = np.array([1.00, 1.90, 2.60])\n"),
         "call": "sound_speed_squared(grid.copy(), mu.copy())",
         "gold_call": "_oracle_sound_speed_squared(grid.copy(), mu.copy())", "tol": 1e-12},
        # --- Invalid: a non-positive chemical potential must raise ValueError ---
        {"setup": ("import numpy as np\ngrid = np.linspace(0.32, 4.80, 51)\n"
                   "mu = np.linspace(-0.1, 2.6, 51)\n") + guard,
         "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: a mismatched array length must raise ValueError ---
        {"setup": ("import numpy as np\ngrid = np.linspace(0.32, 4.80, 51)\n"
                   "mu = np.linspace(1.0, 2.6, 50)\n") + guard,
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
