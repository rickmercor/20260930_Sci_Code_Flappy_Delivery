"""
Solve the macroscopic diffusion equation for the temperature field driven by the dissipation and by the divergence of the non-Fourier part of the heat flux, between isothermal walls.

Updating the temperature only from the local collision balance couples cells to one another solely through the transport sweep, so information about a change in the temperature at one end of the film reaches the other end at the rate at which the sweep propagates it. That is fast when phonons stream freely and slow when they scatter often, which is why the plain scheme converges quickly in the ballistic corner and stalls as the diffusive limit is approached. The synthetic scheme repairs this by carrying a macroscopic equation alongside the kinetic one. The heat flux is split into the part a Fourier law with the bulk conductivity would produce and the remainder, which contains everything non-local and ballistic; substituting that split into the steady energy balance turns the balance into a diffusion equation for the temperature whose source is the deposited power minus the divergence of the non-Fourier remainder. Because the diffusion operator is global, one solve propagates temperature information across the whole film, which is exactly the error mode the kinetic sweep damps slowly.




Two properties of this update deserve care. The first is that the non-Fourier remainder must be evaluated with the temperature field of the previous iterate while the diffusion operator acts on the new one; the two Fourier-law contributions therefore do not cancel identically, and what is being iterated is a correction driven by the mismatch between the kinetic flux divergence and the deposited power. At a fixed point that mismatch vanishes, and the resulting statement is the discrete energy balance, which is algebraically the same condition as the collision closure. The two updates therefore converge to the same field, and the choice between them is a choice of convergence rate rather than of answer.




The second is that the boundary condition of the macroscopic equation appears in both the operator and the source with opposite signs and so drops out of the fixed point, affecting only the rate. Taking the wall temperature to be the reference temperature is therefore admissible even in the quasi-ballistic regime, where the true lattice temperature adjacent to a thermalising contact is far from it; the converged interior field still carries the full temperature jump, because it is fixed by the kinetic equations and not by the macroscopic boundary condition. A useful consequence of the same structure is that with no kinetic input at all, that is with a vanishing flux divergence and a vanishing previous field, the update returns exactly the Fourier solution of the same problem, which is the reference the whole calculation is measured against.




The spatial operator is the standard cell-centred finite-volume Laplacian on a uniform mesh, with the Dirichlet condition imposed at the wall face rather than at a cell centre, so that the boundary face gradient is taken over half a cell width.

Returns
-------
np.ndarray of shape (n_cells,), float: the updated temperature rise above the wall temperature in K.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def update_macroscopic_temperature(kappa_bulk: float, heat_source: np.ndarray,
                                   flux_divergence: np.ndarray,
                                   previous_temperature: np.ndarray,
                                   cell_size: float) -> np.ndarray:
    """Solve the macroscopic diffusion equation for the temperature rise.

    Parameters
    ----------
    kappa_bulk : float
        Bulk thermal conductivity in W/(m K) (kappa_bulk > 0).
    heat_source : np.ndarray
        Array of shape (n_cells,) holding the volumetric dissipation in W/m^3.
    flux_divergence : np.ndarray
        Array of shape (n_cells,) holding the divergence of the kinetic heat
        flux in W/m^3.
    previous_temperature : np.ndarray
        Array of shape (n_cells,) holding the temperature rise of the previous
        iterate in K.
    cell_size : float
        Width of a finite volume in m (cell_size > 0).

    Returns
    -------
    temperature : np.ndarray
        Array of shape (n_cells,) holding the updated temperature rise above
        the wall temperature, in K.

    Raises
    ------
    ValueError
        If the field arrays are empty, non-finite or do not have identical
        one-dimensional shapes, or if ``kappa_bulk`` or ``cell_size`` is not
        finite and strictly positive.
    """
    return temperature  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_update_macroscopic_temperature(kappa_bulk: float,
                                           heat_source: np.ndarray,
                                           flux_divergence: np.ndarray,
                                           previous_temperature: np.ndarray,
                                           cell_size: float) -> np.ndarray:
    # Local imports and nested helpers keep the oracle self-contained when the
    # harness executes it in isolation.
    import numpy as np

    def _laplacian(n_cells, dx):
        """Cell-centred Laplacian with the walls held at zero rise."""
        matrix = np.zeros((n_cells, n_cells), dtype=float)
        inverse = 1.0 / dx ** 2
        for cell in range(n_cells):
            # Left face: a wall face has a half-cell gradient arm.
            matrix[cell, cell] -= 2.0 * inverse if cell == 0 else inverse
            if cell > 0:
                matrix[cell, cell - 1] += inverse
            # Right face.
            matrix[cell, cell] -= 2.0 * inverse if cell == n_cells - 1 else inverse
            if cell < n_cells - 1:
                matrix[cell, cell + 1] += inverse
        return matrix

    source = np.asarray(heat_source, dtype=float)
    divergence = np.asarray(flux_divergence, dtype=float)
    previous = np.asarray(previous_temperature, dtype=float)

    if source.ndim != 1 or source.size < 1:
        raise ValueError("heat_source must be a 1D array of length >= 1")
    if divergence.shape != source.shape:
        raise ValueError("flux_divergence must have the same shape as heat_source")
    if previous.shape != source.shape:
        raise ValueError("previous_temperature must have the same shape as heat_source")
    if not (np.all(np.isfinite(source)) and np.all(np.isfinite(divergence))
            and np.all(np.isfinite(previous))):
        raise ValueError("all field inputs must be finite")
    if not (isinstance(kappa_bulk, (int, float, np.floating, np.integer))
            and not isinstance(kappa_bulk, bool)
            and np.isfinite(kappa_bulk) and float(kappa_bulk) > 0.0):
        raise ValueError("kappa_bulk must be a finite number > 0")
    if not (isinstance(cell_size, (int, float, np.floating, np.integer))
            and not isinstance(cell_size, bool)
            and np.isfinite(cell_size) and float(cell_size) > 0.0):
        raise ValueError("cell_size must be a finite number > 0")

    kappa_bulk = float(kappa_bulk)
    laplacian = _laplacian(source.size, float(cell_size))

    # Diffusion driven by the dissipation minus the divergence of the
    # non-Fourier flux, the latter evaluated at the previous iterate.
    right_hand = source - divergence - kappa_bulk * (laplacian @ previous)

    return np.linalg.solve(-kappa_bulk * laplacian, right_hand)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: pure Fourier reference of the benchmark hot spot (normal scenario) ---
        {
            "setup": """import numpy as np
n_cells, length = 200, 20.0e-9
cell_size = length / n_cells
xc = (np.arange(n_cells) + 0.5) * cell_size
heat_source = np.where(np.abs(xc - 0.5 * length) <= 0.1 * length, 1.5e19, 0.0)
flux_divergence = np.zeros(n_cells)
previous_temperature = np.zeros(n_cells)
kappa_bulk = 148.0
""",
            "call": "update_macroscopic_temperature(kappa_bulk, heat_source, flux_divergence, previous_temperature, cell_size)",
            "gold_call": "_oracle_update_macroscopic_temperature(kappa_bulk, heat_source, flux_divergence, previous_temperature, cell_size)",
        },
        # --- Valid: synthetic update with a genuine non-Fourier correction ---
        {
            "setup": """import numpy as np
n_cells, length = 60, 1.0e-7
cell_size = length / n_cells
xc = (np.arange(n_cells) + 0.5) * cell_size
heat_source = np.where(np.abs(xc - 0.5 * length) <= 0.1 * length, 1.5e17, 0.0)
previous_temperature = 5.0 * np.sin(np.pi * xc / length)
flux_divergence = 0.7 * heat_source + 3.0e15 * np.cos(2.0 * np.pi * xc / length)
kappa_bulk = 148.0
""",
            "call": "update_macroscopic_temperature(kappa_bulk, heat_source, flux_divergence, previous_temperature, cell_size)",
            "gold_call": "_oracle_update_macroscopic_temperature(kappa_bulk, heat_source, flux_divergence, previous_temperature, cell_size)",
        },
        # --- Boundary: a converged state, where the update must reproduce its input ---
        {
            "setup": """import numpy as np
n_cells = 40
cell_size = 5.0e-10
xc = (np.arange(n_cells) + 0.5) * cell_size
heat_source = np.full(n_cells, 2.0e18)
previous_temperature = np.linspace(1.0, 2.0, n_cells)
flux_divergence = heat_source.copy()
kappa_bulk = 148.0
""",
            "call": "update_macroscopic_temperature(kappa_bulk, heat_source, flux_divergence, previous_temperature, cell_size)",
            "gold_call": "_oracle_update_macroscopic_temperature(kappa_bulk, heat_source, flux_divergence, previous_temperature, cell_size)",
        },
        # --- Edge: a single cell clamped between two walls ---
        {
            "setup": """import numpy as np
heat_source = np.array([1.5e19])
flux_divergence = np.array([0.0])
previous_temperature = np.array([0.0])
kappa_bulk = 148.0
cell_size = 2.0e-9
""",
            "call": "update_macroscopic_temperature(kappa_bulk, heat_source, flux_divergence, previous_temperature, cell_size)",
            "gold_call": "_oracle_update_macroscopic_temperature(kappa_bulk, heat_source, flux_divergence, previous_temperature, cell_size)",
        },
        # --- Invalid: field arrays of inconsistent length ---
        {
            "setup": """import numpy as np
heat_source = np.zeros(10)
flux_divergence = np.zeros(8)
previous_temperature = np.zeros(10)
def run_model():
    try:
        update_macroscopic_temperature(148.0, heat_source, flux_divergence, previous_temperature, 1.0e-10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_update_macroscopic_temperature(148.0, heat_source, flux_divergence, previous_temperature, 1.0e-10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive bulk conductivity ---
        {
            "setup": """import numpy as np
heat_source = np.zeros(10)
flux_divergence = np.zeros(10)
previous_temperature = np.zeros(10)
def run_model():
    try:
        update_macroscopic_temperature(0.0, heat_source, flux_divergence, previous_temperature, 1.0e-10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_update_macroscopic_temperature(0.0, heat_source, flux_divergence, previous_temperature, 1.0e-10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
