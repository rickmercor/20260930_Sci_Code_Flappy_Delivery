"""
Assemble the reusable finite-volume transport matrices for every band-direction pair on the one-dimensional mesh, combining the collision term with the upwinded advection term and the thermalising wall condition.

After the spectral and angular discretisations, the steady Boltzmann equation separates into one independent transport problem for every pair of a phonon band and a propagation direction. Each such problem is linear in the energy density on the spatial mesh and has the same algebraic shape: the inverse relaxation time on the diagonal, coming from the collision term that relaxes the distribution towards local equilibrium, plus the band group velocity times an operator that depends only on the mesh and on the direction and advects the distribution along that direction. Writing the matrix in that factored form is what makes the whole solver affordable, because the matrix does not depend on the lattice temperature and therefore does not change from one outer iteration to the next; it can be assembled and factorised once before the iteration loop, and every subsequent iteration costs only a right-hand-side assembly and a triangular substitution.




The advection operator is built by upwinding the face fluxes. In a finite-volume cell the advection term is the difference of the fluxes through the two faces divided by the cell width, and stability requires the face value to be taken from the cell the phonons are coming from, which for a positive velocity component along the coordinate is the left neighbour and for a negative component the right neighbour. First order upwinding therefore produces a bidiagonal matrix, lower for one hemisphere and upper for the other, with the magnitude of the velocity component over the cell width on the diagonal and its negative on the off-diagonal.




The boundary faces are where the physics of the contact enters. A thermalising wall absorbs everything that reaches it and re-emits an equilibrium distribution at the wall temperature, independently of what arrived; in a formulation written for the deviation of the energy density from equilibrium at the reference temperature, and with the walls held at that reference temperature, the incoming deviational energy at such a wall is exactly zero. The inflow face therefore contributes nothing to the matrix and nothing to the right-hand side, which is why the boundary rows differ from the interior rows only by the absence of an off-diagonal entry. The outflow face contributes the ordinary upwind flux from the last interior cell, so no condition is imposed there, as it should be for a hyperbolic problem. What this arrangement produces, and what a Fourier description cannot, is a finite jump between the wall temperature and the lattice temperature of the adjacent material whenever the mean free paths exceed the film thickness.

Returns
-------
np.ndarray of shape (n_bands, n_dirs, n_cells, n_cells), float: every reusable band-direction transport matrix in s^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_transport_operator(bands: np.ndarray, quadrature: np.ndarray,
                                n_cells: int,
                                cell_size: float) -> np.ndarray:
    """Assemble the reusable transport matrices for all band-direction pairs.

    Parameters
    ----------
    bands : np.ndarray
        Array of shape (n_bands, 3) holding the heat capacity, group velocity
        and relaxation time of each representative band.
    quadrature : np.ndarray
        Array of shape (n_dirs, 2) holding the direction cosines and the
        solid-angle weights of the ordinate set.
    n_cells : int
        Number of finite volumes across the film (n_cells >= 1).
    cell_size : float
        Width of a finite volume in m (cell_size > 0).

    Returns
    -------
    operators : np.ndarray
        Array of shape (n_bands, n_dirs, n_cells, n_cells) holding every
        temperature-independent transport matrix in units of s^-1.

    Raises
    ------
    ValueError
        If ``bands`` or ``quadrature`` has an invalid shape or contains
        non-finite values, if a band velocity or relaxation time is not
        strictly positive, if a direction cosine lies outside ``[-1, 1]``,
        if ``n_cells`` is not an integer at least 1, or if ``cell_size`` is
        not finite and strictly positive.
    """
    return operator  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_assemble_transport_operator(bands: np.ndarray,
                                        quadrature: np.ndarray,
                                        n_cells: int,
                                        cell_size: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    band_table = np.asarray(bands, dtype=float)
    quad = np.asarray(quadrature, dtype=float)
    if band_table.ndim != 2 or band_table.shape[1] != 3 or band_table.shape[0] < 1:
        raise ValueError("bands must be a 2D array of shape (n_bands, 3)")
    if quad.ndim != 2 or quad.shape[1] != 2 or quad.shape[0] < 1:
        raise ValueError("quadrature must be a 2D array of shape (n_dirs, 2)")
    if not (np.all(np.isfinite(band_table)) and np.all(np.isfinite(quad))):
        raise ValueError("bands and quadrature must be finite")
    if np.any(band_table[:, 1] <= 0.0) or np.any(band_table[:, 2] <= 0.0):
        raise ValueError("band velocities and relaxation times must be > 0")
    if np.any(np.abs(quad[:, 0]) > 1.0):
        raise ValueError("direction cosines must lie in [-1, 1]")
    if not (isinstance(n_cells, (int, np.integer)) and not isinstance(n_cells, bool)
            and int(n_cells) >= 1):
        raise ValueError("n_cells must be an integer >= 1")
    if not (isinstance(cell_size, (int, float, np.floating, np.integer))
            and not isinstance(cell_size, bool)
            and np.isfinite(cell_size) and float(cell_size) > 0.0):
        raise ValueError("cell_size must be a finite number > 0")

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

            # Upwinded face flux: the inflow neighbour is on the left for
            # positive velocity components and on the right otherwise. The
            # incoming deviational energy at a thermalising wall is zero.
            if advection > 0.0:
                operator[index[1:], index[1:] - 1] = -advection
            elif advection < 0.0:
                operator[index[:-1], index[:-1] + 1] = advection

    return operators

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: several bands and both hemispheres (normal scenario) ---
        {
            "setup": """import numpy as np
bands = np.array([
    [1.2896e6, 1451.0335479, 1.354e-11],
    [2.1132e5, 3657.21, 1.2908e-11],
])
mu, w = np.polynomial.legendre.leggauss(8)
quadrature = np.column_stack([mu, 2.0 * np.pi * w])
n_cells = 40
cell_size = 5.0e-10
""",
            "call": "assemble_transport_operator(bands, quadrature, n_cells, cell_size)",
            "gold_call": "_oracle_assemble_transport_operator(bands, quadrature, n_cells, cell_size)",
        },
        # --- Valid: an asymmetric custom ordinate set ---
        {
            "setup": """import numpy as np
bands = np.array([[3.0e4, 6176.63416479, 5.3257e-09]])
quadrature = np.array([[-0.9894009349916499, 3.0], [0.35, 2.0]])
n_cells = 40
cell_size = 5.0e-10
""",
            "call": "assemble_transport_operator(bands, quadrature, n_cells, cell_size)",
            "gold_call": "_oracle_assemble_transport_operator(bands, quadrature, n_cells, cell_size)",
        },
        # --- Boundary: grazing ordinate, so advection nearly vanishes ---
        {
            "setup": """import numpy as np
bands = np.array([[1.0e6, 5000.0, 1.0e-11]])
quadrature = np.array([[0.0, 4.0 * np.pi]])
n_cells = 12
cell_size = 2.0e-10
""",
            "call": "assemble_transport_operator(bands, quadrature, n_cells, cell_size)",
            "gold_call": "_oracle_assemble_transport_operator(bands, quadrature, n_cells, cell_size)",
        },
        # --- Edge: a single cell, so both faces are walls ---
        {
            "setup": """import numpy as np
bands = np.array([[1.8e6, 4000.0, 2.0e-11]])
quadrature = np.array([[-1.0, 2.0 * np.pi], [1.0, 2.0 * np.pi]])
n_cells = 1
cell_size = 1.0e-9
""",
            "call": "assemble_transport_operator(bands, quadrature, n_cells, cell_size)",
            "gold_call": "_oracle_assemble_transport_operator(bands, quadrature, n_cells, cell_size)",
        },
        # --- Invalid: direction cosine outside the unit interval ---
        {
            "setup": """import numpy as np
bands = np.array([[1.0e6, 5000.0, 1.0e-11]])
quadrature = np.array([[1.5, 1.0]])
def run_model():
    try:
        assemble_transport_operator(bands, quadrature, 10, 1.0e-10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_transport_operator(bands, quadrature, 10, 1.0e-10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a band with vanishing relaxation time has no inverse ---
        {
            "setup": """import numpy as np
bands = np.array([[1.0e6, 5000.0, 0.0]])
quadrature = np.array([[-0.5, 2.0 * np.pi], [0.5, 2.0 * np.pi]])
def run_model():
    try:
        assemble_transport_operator(bands, quadrature, 10, 1.0e-10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_transport_operator(bands, quadrature, 10, 1.0e-10)
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
