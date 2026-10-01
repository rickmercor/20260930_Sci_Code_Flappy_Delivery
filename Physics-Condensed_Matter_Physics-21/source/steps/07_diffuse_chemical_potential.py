"""
Smooth a bridge by evolving the chemical potential under a conservative diffusion in flow time, and report the energy density transported across the domain boundaries.

A bridge built by recursive subdivision carries structure at every resolved scale, which is neither numerically convenient nor physically expected. Imposing a finite correlation length is achieved by treating the chemical potential as a function of density and an unphysical flow time, and evolving it under a heat equation in which the flux is the diffusion coefficient times the density gradient of the chemical potential.




Holding the chemical potential fixed at both ends of the density interval keeps the endpoints of the equation of state where the underlying theory put them. Inside the interval the evolution conserves the integral of the chemical potential over density, which is the energy-density difference, so thermodynamic consistency is preserved there. At the two ends it is not conserved: energy density flows through the edges at a rate given by the flux evaluated at the boundaries, and the accumulated flow over the whole evolution is a direct measure of how far the smoothed bridge has drifted from exact thermodynamic consistency.




That accumulated boundary flow is small but not negligible, and it is the quantity this step reports alongside the smoothed chemical potential.

Returns
-------
np.ndarray, shape (N + 1,), [flux, mu[0], ..., mu[N-1]] with flux in GeV fm^-3
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def diffuse_chemical_potential(density_grid: "np.ndarray", chemical_potential: "np.ndarray",
                               correlation_fraction: float, n_tau_steps: int) -> "np.ndarray":
    '''Diffuse the chemical potential and accumulate the boundary energy flux.

    Parameters
    ----------
    density_grid : np.ndarray
        Finite strictly increasing shape (N,) densities in fm^-3, N >= 3, all
        strictly positive and uniformly spaced.
    chemical_potential : np.ndarray
        Finite shape (N,) chemical potentials in GeV on that grid.
    correlation_fraction : float
        Finite positive fractional correlation length.
    n_tau_steps : int
        Integer number of uniform flow-time steps, at least 1, covering the
        flow time from zero to one.

    Returns
    -------
    result : np.ndarray
        Finite shape (N + 1,). Element 0 is the accumulated boundary energy
        flux in GeV fm^-3, and elements 1 to N are the chemical potential in GeV
        after unit flow time.

        The evolution uses the conservative explicit update in which the flux
        between neighbouring grid points is the diffusion coefficient averaged
        over the two points, times their chemical-potential difference, divided
        by the grid spacing. Interior values advance by the flow-time step times
        the difference of the neighbouring fluxes divided by the grid spacing,
        and the first and last values are held fixed. The accumulated boundary
        flux adds, at every step, the flow-time step times the flux at the last
        interval less the flux at the first interval.

    Raises
    ------
    ValueError
        If the arrays are not finite real values of matching shape, if the grid
        is not uniformly spaced, if a preceding step rejects its inputs, if
        n_tau_steps is not an integer of at least 1, or if the evolved result is
        not finite.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_diffuse_chemical_potential(density_grid: "np.ndarray",
                                       chemical_potential: "np.ndarray",
                                       correlation_fraction: float,
                                       n_tau_steps: int) -> "np.ndarray":
    profile = _oracle_diffusion_profile(density_grid, correlation_fraction)
    grid = _validated_density_grid(density_grid)
    if not np.isrealobj(chemical_potential):
        raise ValueError("the chemical potential must be real")
    try:
        mu = np.array(chemical_potential, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("the chemical potential must be a finite real array") from exc
    if mu.shape != grid.shape or not np.all(np.isfinite(mu)):
        raise ValueError("the chemical potential must be finite and match the density grid")
    spacing = grid[1] - grid[0]
    if not np.allclose(np.diff(grid), spacing, rtol=0.0, atol=1e-12):
        raise ValueError("the density grid must be uniformly spaced")
    if isinstance(n_tau_steps, bool) or not isinstance(n_tau_steps, (int, np.integer)):
        raise ValueError("n_tau_steps must be an integer")
    steps = int(n_tau_steps)
    if steps < 1:
        raise ValueError("n_tau_steps must be at least 1")
    half = 0.5 * (profile[0][1:] + profile[0][:-1])
    step = 1.0 / steps
    accumulated = 0.0
    for _ in range(steps):
        flux = half * np.diff(mu) / spacing
        accumulated += step * (flux[-1] - flux[0])
        mu[1:-1] += step * np.diff(flux) / spacing
    result = np.concatenate(([accumulated], mu))
    if not np.all(np.isfinite(result)):
        raise ValueError("the diffused result must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    bridge = ("import numpy as np\n"
              "grid = np.linspace(0.32, 4.80, 201)\n"
              "mu = 1.00 + 1.60*(grid - 0.32)/(4.80 - 0.32)\n"
              "mu = mu + 0.02*np.sin(11.0*(grid - 0.32))\n"
              "c = 0.20\nsteps = 4000\n")
    guard = ('def run_model():\n'
             '    try:\n'
             '        diffuse_chemical_potential(grid.copy(), mu.copy(), c, steps)\n'
             '        return 0\n'
             '    except ValueError:\n'
             '        return 1\n'
             'def run_oracle():\n'
             '    try:\n'
             '        _oracle_diffuse_chemical_potential(grid.copy(), mu.copy(), c, steps)\n'
             '        return 0\n'
             '    except ValueError:\n'
             '        return 1\n')
    return [
        # --- Normal: a rippled bridge at the benchmark discretisation ---
        {"setup": bridge,
         "call": "diffuse_chemical_potential(grid.copy(), mu.copy(), c, steps)",
         "gold_call": "_oracle_diffuse_chemical_potential(grid.copy(), mu.copy(), c, steps)", "tol": 1e-10},
        # --- Normal: weaker smoothing leaves more structure and moves less energy ---
        {"setup": bridge.replace("c = 0.20", "c = 0.08"),
         "call": "diffuse_chemical_potential(grid.copy(), mu.copy(), c, steps)",
         "gold_call": "_oracle_diffuse_chemical_potential(grid.copy(), mu.copy(), c, steps)", "tol": 1e-10},
        # --- Boundary: a single flow-time step ---
        {"setup": bridge.replace("steps = 4000", "steps = 1"),
         "call": "diffuse_chemical_potential(grid.copy(), mu.copy(), c, steps)",
         "gold_call": "_oracle_diffuse_chemical_potential(grid.copy(), mu.copy(), c, steps)", "tol": 1e-10},
        # --- Edge: an initially straight bridge, where smoothing has little to remove ---
        {"setup": ("import numpy as np\n"
                   "grid = np.linspace(0.32, 4.80, 201)\n"
                   "mu = 1.00 + 1.60*(grid - 0.32)/(4.80 - 0.32)\n"
                   "c = 0.20\nsteps = 4000\n"),
         "call": "diffuse_chemical_potential(grid.copy(), mu.copy(), c, steps)",
         "gold_call": "_oracle_diffuse_chemical_potential(grid.copy(), mu.copy(), c, steps)", "tol": 1e-10},
        # --- Normal: a coarser grid with a correspondingly stable step count ---
        {"setup": ("import numpy as np\n"
                   "grid = np.linspace(0.32, 4.80, 101)\n"
                   "mu = 1.00 + 1.60*(grid - 0.32)/(4.80 - 0.32) + 0.02*np.sin(11.0*(grid - 0.32))\n"
                   "c = 0.20\nsteps = 2000\n"),
         "call": "diffuse_chemical_potential(grid.copy(), mu.copy(), c, steps)",
         "gold_call": "_oracle_diffuse_chemical_potential(grid.copy(), mu.copy(), c, steps)", "tol": 1e-10},
        # --- Invalid: a non-uniform grid must raise ValueError ---
        {"setup": ("import numpy as np\n"
                   "grid = np.geomspace(0.32, 4.80, 201)\n"
                   "mu = 1.00 + 1.60*(grid - 0.32)/(4.80 - 0.32)\n"
                   "c = 0.20\nsteps = 10\n") + guard,
         "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: a non-positive step count must raise ValueError ---
        {"setup": bridge.replace("steps = 4000", "steps = 0") + guard,
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
