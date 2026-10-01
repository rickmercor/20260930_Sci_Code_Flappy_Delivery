"""
Reuse the preassembled band-direction transport operators to solve every transport equation once for a frozen lattice-temperature field and return the resulting deviational energy-density tensor.

One outer iteration of a deterministic Boltzmann solver is a sweep: the lattice temperature is held fixed, the local equilibrium distribution it implies is built, and the transport equation of every band-direction pair is solved to convergence in space. The pairs do not couple to one another within a sweep — they communicate only through the temperature field that generated their equilibrium term and through the moments taken of their solutions afterwards — which is exactly the property that lets a production solver distribute them across processes with no spatial domain decomposition at all.




Two ingredients enter the right-hand side of each pair. The first is the relaxation term, the equilibrium energy density of the band divided by its relaxation time. The equilibrium energy density itself is the band heat capacity times the local temperature rise divided by the total angular quadrature weight, the division being what makes the angular integral of the equilibrium distribution return the heat capacity times the temperature rather than some multiple of it; getting this factor wrong rescales the collision term relative to the streaming term and therefore rescales every mean free path in the problem. The second ingredient is the isotropic source density that the dissipation contributes to that band, which is independent of direction and therefore identical for every ordinate of the same band.




Because the matrix of each pair is bidiagonal under first-order upwinding, and because it is independent of the temperature field, the matrices are supplied by sub-problem 05 after being assembled once outside the outer iteration. Each solve here is only a substitution sweep along the mesh in the direction the phonons travel. The cost of one sweep is therefore linear in the number of cells, bands and directions, and no matrix is rebuilt or refactorised as the temperature changes.




The result is a three-index tensor of deviational energy densities, indexed by band, by ordinate and by cell. It is not itself an observable: only its angular and spectral moments are.

Returns
-------
np.ndarray of shape (n_bands, n_dirs, n_cells), float: deviational phonon energy density per unit solid angle in J m^-3 sr^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solve_transport_sweep(bands: np.ndarray, quadrature: np.ndarray,
                          transport_operators: np.ndarray,
                          mode_source: np.ndarray,
                          lattice_temperature: np.ndarray) -> np.ndarray:
    """Solve every band-direction transport equation for a frozen temperature.

    Parameters
    ----------
    bands : np.ndarray
        Array of shape (n_bands, 3) holding the heat capacity, group velocity
        and relaxation time of each representative band.
    quadrature : np.ndarray
        Array of shape (n_dirs, 2) holding the direction cosines and the
        solid-angle weights of the ordinate set.
    transport_operators : np.ndarray
        Array of shape (n_bands, n_dirs, n_cells, n_cells) holding the
        temperature-independent matrices assembled by
        ``assemble_transport_operator`` before the outer iteration.
    mode_source : np.ndarray
        Array of shape (n_bands, n_cells) holding the isotropic per-band
        source density in W/(m^3 sr).
    lattice_temperature : np.ndarray
        Array of shape (n_cells,) holding the current temperature rise above
        the reference temperature, in K.
    Returns
    -------
    energy : np.ndarray
        Array of shape (n_bands, n_dirs, n_cells) holding the deviational
        energy density per unit solid angle, in J/(m^3 sr).

    Raises
    ------
    ValueError
        If the arrays have incompatible shapes or contain non-finite values,
        if any band velocity or relaxation time is not strictly positive, if
        a direction cosine lies outside ``[-1, 1]``, if the quadrature weights
        do not sum to a positive solid angle, or if a supplied transport
        operator has a non-positive diagonal.
    """
    return energy  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_transport_sweep(bands: np.ndarray, quadrature: np.ndarray,
                                  transport_operators: np.ndarray,
                                  mode_source: np.ndarray,
                                  lattice_temperature: np.ndarray) -> np.ndarray:
    # Local imports and nested helpers keep the oracle self-contained when the
    # harness executes it in isolation.
    import numpy as np

    band_table = np.asarray(bands, dtype=float)
    quad = np.asarray(quadrature, dtype=float)
    operators = np.asarray(transport_operators, dtype=float)
    source = np.asarray(mode_source, dtype=float)
    temperature = np.asarray(lattice_temperature, dtype=float)

    if band_table.ndim != 2 or band_table.shape[1] != 3 or band_table.shape[0] < 1:
        raise ValueError("bands must be a 2D array of shape (n_bands, 3)")
    if quad.ndim != 2 or quad.shape[1] != 2 or quad.shape[0] < 1:
        raise ValueError("quadrature must be a 2D array of shape (n_dirs, 2)")
    if temperature.ndim != 1 or temperature.size < 1:
        raise ValueError("lattice_temperature must be a 1D array of length >= 1")
    if source.shape != (band_table.shape[0], temperature.size):
        raise ValueError("mode_source must have shape (n_bands, n_cells)")
    expected_operator_shape = (band_table.shape[0], quad.shape[0],
                               temperature.size, temperature.size)
    if operators.shape != expected_operator_shape:
        raise ValueError(
            "transport_operators must have shape "
            "(n_bands, n_dirs, n_cells, n_cells)")
    if not (np.all(np.isfinite(band_table)) and np.all(np.isfinite(quad))
            and np.all(np.isfinite(operators)) and np.all(np.isfinite(source))
            and np.all(np.isfinite(temperature))):
        raise ValueError("all inputs must be finite")
    if np.any(band_table[:, 1] <= 0.0) or np.any(band_table[:, 2] <= 0.0):
        raise ValueError("band velocities and relaxation times must be > 0")
    if np.any(np.abs(quad[:, 0]) > 1.0):
        raise ValueError("direction cosines must lie in [-1, 1]")
    n_bands = band_table.shape[0]
    n_dirs = quad.shape[0]
    n_cells = temperature.size
    solid_angle = float(quad[:, 1].sum())
    if not solid_angle > 0.0:
        raise ValueError("the quadrature weights must sum to a positive solid angle")

    diagonal = np.diagonal(operators, axis1=2, axis2=3)
    if np.any(diagonal <= 0.0):
        raise ValueError("transport operator diagonals must be > 0")

    solution = np.empty((n_bands, n_dirs, n_cells), dtype=float)
    for b in range(n_bands):
        capacity, velocity, relaxation_time = band_table[b]
        # Relaxation towards the local equilibrium plus the isotropic source.
        rhs = capacity * temperature / solid_angle / relaxation_time + source[b]
        for i in range(n_dirs):
            matrix = operators[b, i]
            values = np.empty(n_cells, dtype=float)
            if quad[i, 0] > 0.0:
                values[0] = rhs[0] / matrix[0, 0]
                for cell in range(1, n_cells):
                    values[cell] = (rhs[cell]
                                    - matrix[cell, cell - 1] * values[cell - 1]) \
                                   / matrix[cell, cell]
            elif quad[i, 0] < 0.0:
                values[-1] = rhs[-1] / matrix[-1, -1]
                for cell in range(n_cells - 2, -1, -1):
                    values[cell] = (rhs[cell]
                                    - matrix[cell, cell + 1] * values[cell + 1]) \
                                   / matrix[cell, cell]
            else:
                values[:] = rhs / diagonal[b, i]
            solution[b, i] = values

    return solution

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: quasi-ballistic film with a central hot spot (normal scenario) ---
        {
            "setup": """import numpy as np
def _fx_assemble_transport_operator(bands, quadrature, n_cells, cell_size):
    band_table = np.asarray(bands, dtype=float)
    quad = np.asarray(quadrature, dtype=float)
    n_cells = int(n_cells)
    operators = np.zeros(
        (band_table.shape[0], quad.shape[0], n_cells, n_cells), dtype=float)
    index = np.arange(n_cells)
    for b in range(band_table.shape[0]):
        velocity = band_table[b, 1]
        relaxation_time = band_table[b, 2]
        for i in range(quad.shape[0]):
            advection = velocity * quad[i, 0] / float(cell_size)
            operator = operators[b, i]
            operator[index, index] = 1.0 / relaxation_time + abs(advection)
            if advection > 0.0:
                operator[index[1:], index[1:] - 1] = -advection
            elif advection < 0.0:
                operator[index[:-1], index[:-1] + 1] = advection
    return operators
bands = np.array([
    [1.2896e+06, 1451.03, 1.3540e-11],
    [2.1132e+05, 3657.21, 1.2908e-11],
    [1.1959e+03, 6146.84, 7.9310e-10],
])
mu, w = np.polynomial.legendre.leggauss(8)
quadrature = np.column_stack([mu, 2.0 * np.pi * w])
n_cells, length = 60, 20.0e-9
cell_size = length / n_cells
xc = (np.arange(n_cells) + 0.5) * cell_size
heat = np.where(np.abs(xc - 0.5 * length) <= 0.1 * length, 1.5e19, 0.0)
share = bands[:, 0] / bands[:, 0].sum() / (4.0 * np.pi)
mode_source = np.outer(share, heat)
lattice_temperature = np.full(n_cells, 5.0)
transport_operators = _fx_assemble_transport_operator(
    bands, quadrature, n_cells, cell_size)
""",
            "call": "solve_transport_sweep(bands, quadrature, transport_operators, mode_source, lattice_temperature)",
            "gold_call": "_oracle_solve_transport_sweep(bands, quadrature, transport_operators, mode_source, lattice_temperature)",
        },
        # --- Valid: no source at all, so the sweep only relaxes towards equilibrium ---
        {
            "setup": """import numpy as np
def _fx_assemble_transport_operator(bands, quadrature, n_cells, cell_size):
    band_table = np.asarray(bands, dtype=float)
    quad = np.asarray(quadrature, dtype=float)
    n_cells = int(n_cells)
    operators = np.zeros(
        (band_table.shape[0], quad.shape[0], n_cells, n_cells), dtype=float)
    index = np.arange(n_cells)
    for b in range(band_table.shape[0]):
        velocity = band_table[b, 1]
        relaxation_time = band_table[b, 2]
        for i in range(quad.shape[0]):
            advection = velocity * quad[i, 0] / float(cell_size)
            operator = operators[b, i]
            operator[index, index] = 1.0 / relaxation_time + abs(advection)
            if advection > 0.0:
                operator[index[1:], index[1:] - 1] = -advection
            elif advection < 0.0:
                operator[index[:-1], index[:-1] + 1] = advection
    return operators
bands = np.array([[1.0e6, 2000.0, 2.0e-11], [3.0e5, 5000.0, 1.0e-10]])
mu, w = np.polynomial.legendre.leggauss(4)
quadrature = np.column_stack([mu, 2.0 * np.pi * w])
n_cells = 25
cell_size = 4.0e-10
mode_source = np.zeros((2, n_cells))
lattice_temperature = np.linspace(0.0, 10.0, n_cells)
transport_operators = _fx_assemble_transport_operator(
    bands, quadrature, n_cells, cell_size)
""",
            "call": "solve_transport_sweep(bands, quadrature, transport_operators, mode_source, lattice_temperature)",
            "gold_call": "_oracle_solve_transport_sweep(bands, quadrature, transport_operators, mode_source, lattice_temperature)",
        },
        # --- Boundary: cells far thicker than the mean free path, the diffusive corner ---
        {
            "setup": """import numpy as np
def _fx_assemble_transport_operator(bands, quadrature, n_cells, cell_size):
    band_table = np.asarray(bands, dtype=float)
    quad = np.asarray(quadrature, dtype=float)
    n_cells = int(n_cells)
    operators = np.zeros(
        (band_table.shape[0], quad.shape[0], n_cells, n_cells), dtype=float)
    index = np.arange(n_cells)
    for b in range(band_table.shape[0]):
        velocity = band_table[b, 1]
        relaxation_time = band_table[b, 2]
        for i in range(quad.shape[0]):
            advection = velocity * quad[i, 0] / float(cell_size)
            operator = operators[b, i]
            operator[index, index] = 1.0 / relaxation_time + abs(advection)
            if advection > 0.0:
                operator[index[1:], index[1:] - 1] = -advection
            elif advection < 0.0:
                operator[index[:-1], index[:-1] + 1] = advection
    return operators
bands = np.array([[1.5e6, 1200.0, 5.0e-13]])
mu, w = np.polynomial.legendre.leggauss(6)
quadrature = np.column_stack([mu, 2.0 * np.pi * w])
n_cells = 20
cell_size = 1.0e-6
mode_source = np.full((1, n_cells), 1.0e14)
lattice_temperature = np.full(n_cells, 1.0)
transport_operators = _fx_assemble_transport_operator(
    bands, quadrature, n_cells, cell_size)
""",
            "call": "solve_transport_sweep(bands, quadrature, transport_operators, mode_source, lattice_temperature)",
            "gold_call": "_oracle_solve_transport_sweep(bands, quadrature, transport_operators, mode_source, lattice_temperature)",
        },
        # --- Edge: single cell and a single hemisphere pair, wholly ballistic ---
        {
            "setup": """import numpy as np
def _fx_assemble_transport_operator(bands, quadrature, n_cells, cell_size):
    band_table = np.asarray(bands, dtype=float)
    quad = np.asarray(quadrature, dtype=float)
    n_cells = int(n_cells)
    operators = np.zeros(
        (band_table.shape[0], quad.shape[0], n_cells, n_cells), dtype=float)
    index = np.arange(n_cells)
    for b in range(band_table.shape[0]):
        velocity = band_table[b, 1]
        relaxation_time = band_table[b, 2]
        for i in range(quad.shape[0]):
            advection = velocity * quad[i, 0] / float(cell_size)
            operator = operators[b, i]
            operator[index, index] = 1.0 / relaxation_time + abs(advection)
            if advection > 0.0:
                operator[index[1:], index[1:] - 1] = -advection
            elif advection < 0.0:
                operator[index[:-1], index[:-1] + 1] = advection
    return operators
bands = np.array([[1.8e6, 6000.0, 1.0e-8]])
mu, w = np.polynomial.legendre.leggauss(2)
quadrature = np.column_stack([mu, 2.0 * np.pi * w])
mode_source = np.array([[1.5e19]])
lattice_temperature = np.array([2.0])
cell_size = 1.0e-9
transport_operators = _fx_assemble_transport_operator(
    bands, quadrature, 1, cell_size)
""",
            "call": "solve_transport_sweep(bands, quadrature, transport_operators, mode_source, lattice_temperature)",
            "gold_call": "_oracle_solve_transport_sweep(bands, quadrature, transport_operators, mode_source, lattice_temperature)",
        },
        # --- Invalid: source table inconsistent with the band and cell counts ---
        {
            "setup": """import numpy as np
bands = np.array([[1.0e6, 2000.0, 2.0e-11], [3.0e5, 5000.0, 1.0e-10]])
mu, w = np.polynomial.legendre.leggauss(4)
quadrature = np.column_stack([mu, 2.0 * np.pi * w])
mode_source = np.zeros((3, 10))
lattice_temperature = np.zeros(10)
transport_operators = np.ones((2, 4, 10, 10))
def run_model():
    try:
        solve_transport_sweep(bands, quadrature, transport_operators, mode_source, lattice_temperature)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_transport_sweep(bands, quadrature, transport_operators, mode_source, lattice_temperature)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: direction cosine outside the unit interval ---
        {
            "setup": """import numpy as np
bands = np.array([[1.0e6, 2000.0, 2.0e-11]])
quadrature = np.array([[1.4, 6.283185307179586], [-1.4, 6.283185307179586]])
mode_source = np.zeros((1, 5))
lattice_temperature = np.zeros(5)
transport_operators = np.ones((1, 2, 5, 5))
def run_model():
    try:
        solve_transport_sweep(bands, quadrature, transport_operators, mode_source, lattice_temperature)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_transport_sweep(bands, quadrature, transport_operators, mode_source, lattice_temperature)
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
