"""
Reduce the band-direction energy tensor to the three macroscopic fields the outer iteration needs: the lattice temperature implied by the collision balance, the heat flux, and the divergence of that flux.

The kinetic solution is not itself an observable; three moments of it are. The first follows from the fact that the relaxation-time collision operator must conserve energy: summing the collision term over all ordinates and all bands has to give zero, and since each band relaxes at its own rate, that condition does not simply say that the total energy equals the total equilibrium energy but that the inverse-lifetime-weighted sums agree. Solved for the temperature, it gives the lattice temperature as the ratio of the inverse-lifetime-weighted angular moment of the energy density to the sum of the band heat capacities divided by their lifetimes. Weighting by the lifetimes is what makes a non-gray solver different from a grey one here: the short-lived, high-heat-capacity bands dominate the definition of the local temperature, while the long-lived ballistic bands, which carry the size effect, barely enter it.




The second moment is the heat flux, the quadrature-weighted sum over ordinates and bands of the energy density times the component of the group velocity along the transport coordinate. In equilibrium it vanishes by symmetry of the ordinate set, so it is a direct measure of how far from equilibrium the distribution is; in the quasi-ballistic regime it stays almost constant across the film outside the source region instead of tracking a local temperature gradient, which is exactly the failure of Fourier's law.




The third quantity is the divergence of that flux, and the way it is evaluated is not a matter of taste. Differencing the cell-centred flux by a central formula gives a divergence that is not the one the discrete transport equations actually satisfy. Summing the discrete transport equation over ordinates and bands shows that the exact discrete statement involves the same upwinded face fluxes used to build the transport matrices, so the divergence must be assembled from those face fluxes: the flux through a face is the quadrature-weighted sum over the pairs of the velocity component times the energy density taken from the upwind side of that face, with zero taken on the inflow side of a thermalising wall. Assembled this way, the divergence of the flux equals the deposited power density exactly when, and only when, the collision balance is satisfied, so the collision-moment update and any macroscopic energy-balance update share a single fixed point. Assembled the other way they do not, and the converged answer depends on which update the solver happens to use.

Returns
-------
np.ndarray of shape (3, n_cells), float: row 0 the lattice temperature rise in K, row 1 the heat flux in W m^-2, row 2 the flux divergence in W m^-3.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_macroscopic_fields(bands: np.ndarray, quadrature: np.ndarray,
                               energy: np.ndarray,
                               cell_size: float) -> np.ndarray:
    """Reduce the energy tensor to the macroscopic fields of the iteration.

    Parameters
    ----------
    bands : np.ndarray
        Array of shape (n_bands, 3) holding the heat capacity, group velocity
        and relaxation time of each representative band.
    quadrature : np.ndarray
        Array of shape (n_dirs, 2) holding the direction cosines and the
        solid-angle weights of the ordinate set.
    energy : np.ndarray
        Array of shape (n_bands, n_dirs, n_cells) holding the deviational
        energy density per unit solid angle.
    cell_size : float
        Width of a finite volume in m (cell_size > 0).

    Returns
    -------
    fields : np.ndarray
        Array of shape (3, n_cells) whose rows are the lattice temperature
        rise in K, the heat flux along the transport coordinate in W/m^2 and
        the divergence of that flux in W/m^3.

    Raises
    ------
    ValueError
        If the band, quadrature or energy arrays have incompatible shapes or
        contain non-finite values, if any band heat capacity or relaxation
        time is not strictly positive, if ``cell_size`` is not finite and
        strictly positive, or if the collision closure denominator is not
        positive.
    """
    return fields  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_macroscopic_fields(bands: np.ndarray, quadrature: np.ndarray,
                                       energy: np.ndarray,
                                       cell_size: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    band_table = np.asarray(bands, dtype=float)
    quad = np.asarray(quadrature, dtype=float)
    field = np.asarray(energy, dtype=float)

    if band_table.ndim != 2 or band_table.shape[1] != 3 or band_table.shape[0] < 1:
        raise ValueError("bands must be a 2D array of shape (n_bands, 3)")
    if quad.ndim != 2 or quad.shape[1] != 2 or quad.shape[0] < 1:
        raise ValueError("quadrature must be a 2D array of shape (n_dirs, 2)")
    if field.ndim != 3 or field.shape[:2] != (band_table.shape[0], quad.shape[0]) \
            or field.shape[2] < 1:
        raise ValueError("energy must have shape (n_bands, n_dirs, n_cells)")
    if not (np.all(np.isfinite(band_table)) and np.all(np.isfinite(quad))
            and np.all(np.isfinite(field))):
        raise ValueError("all inputs must be finite")
    if np.any(band_table[:, 0] <= 0.0) or np.any(band_table[:, 2] <= 0.0):
        raise ValueError("band heat capacities and relaxation times must be > 0")
    if not (isinstance(cell_size, (int, float, np.floating, np.integer))
            and not isinstance(cell_size, bool)
            and np.isfinite(cell_size) and float(cell_size) > 0.0):
        raise ValueError("cell_size must be a finite number > 0")

    capacity, velocity, relaxation = band_table[:, 0], band_table[:, 1], band_table[:, 2]
    cosine, weight = quad[:, 0], quad[:, 1]
    n_cells = field.shape[2]

    # Collision-moment closure for the lattice temperature.
    angular = np.einsum("i,bic->bc", weight, field)
    denominator = float(np.sum(capacity / relaxation))
    if denominator <= 0.0:
        raise ValueError("the collision closure has a non-positive denominator")
    lattice = np.einsum("b,bc->c", 1.0 / relaxation, angular) / denominator

    # First angular moment: the heat flux along the transport coordinate.
    flux = np.einsum("i,b,bic->c", weight * cosine, velocity, field)

    # Divergence from the same upwinded face fluxes the transport uses, with
    # zero deviational energy entering through a thermalising wall.
    coefficient = np.outer(velocity, cosine) * weight[None, :]
    face = np.zeros(n_cells + 1, dtype=float)
    face[1:] += np.einsum("bi,bic->c", np.where(coefficient > 0.0, coefficient, 0.0),
                          field)
    face[:-1] += np.einsum("bi,bic->c", np.where(coefficient < 0.0, coefficient, 0.0),
                           field)
    divergence = (face[1:] - face[:-1]) / float(cell_size)

    return np.vstack([lattice, flux, divergence])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: energy tensor from a quasi-ballistic sweep (normal scenario) ---
        {
            "setup": """import numpy as np
bands = np.array([
    [1.2896e+06, 1451.03, 1.3540e-11],
    [2.1132e+05, 3657.21, 1.2908e-11],
    [1.1959e+03, 6146.84, 7.9310e-10],
])
mu, w = np.polynomial.legendre.leggauss(8)
quadrature = np.column_stack([mu, 2.0 * np.pi * w])
n_cells = 40
cell_size = 5.0e-10
xc = (np.arange(n_cells) + 0.5) / n_cells
base = np.exp(-((xc - 0.5) / 0.2) ** 2)
energy = np.einsum("b,i,c->bic", bands[:, 0] / bands[:, 0].sum(),
                   1.0 + 0.3 * mu, 4.0e3 * base)
""",
            "call": "compute_macroscopic_fields(bands, quadrature, energy, cell_size)",
            "gold_call": "_oracle_compute_macroscopic_fields(bands, quadrature, energy, cell_size)",
        },
        # --- Valid: isotropic tensor, so the flux must vanish identically ---
        {
            "setup": """import numpy as np
bands = np.array([[1.0e6, 2000.0, 2.0e-11], [3.0e5, 5000.0, 1.0e-10]])
mu, w = np.polynomial.legendre.leggauss(6)
quadrature = np.column_stack([mu, 2.0 * np.pi * w])
n_cells = 15
cell_size = 2.0e-10
energy = np.ones((2, 6, n_cells)) * 1.0e3
""",
            "call": "compute_macroscopic_fields(bands, quadrature, energy, cell_size)",
            "gold_call": "_oracle_compute_macroscopic_fields(bands, quadrature, energy, cell_size)",
        },
        # --- Boundary: purely one-sided occupation, the free-streaming corner ---
        {
            "setup": """import numpy as np
bands = np.array([[8.0e5, 4000.0, 5.0e-11]])
mu, w = np.polynomial.legendre.leggauss(4)
quadrature = np.column_stack([mu, 2.0 * np.pi * w])
n_cells = 10
cell_size = 1.0e-9
energy = np.zeros((1, 4, n_cells))
energy[0, mu > 0.0, :] = 2.5e3
""",
            "call": "compute_macroscopic_fields(bands, quadrature, energy, cell_size)",
            "gold_call": "_oracle_compute_macroscopic_fields(bands, quadrature, energy, cell_size)",
        },
        # --- Edge: one cell, so both faces are walls and the divergence is a pure outflow ---
        {
            "setup": """import numpy as np
bands = np.array([[1.8e6, 6000.0, 1.0e-8]])
mu, w = np.polynomial.legendre.leggauss(2)
quadrature = np.column_stack([mu, 2.0 * np.pi * w])
energy = np.array([[[3.0e3], [1.0e3]]])
cell_size = 1.0e-9
""",
            "call": "compute_macroscopic_fields(bands, quadrature, energy, cell_size)",
            "gold_call": "_oracle_compute_macroscopic_fields(bands, quadrature, energy, cell_size)",
        },
        # --- Invalid: energy tensor inconsistent with the ordinate count ---
        {
            "setup": """import numpy as np
bands = np.array([[1.0e6, 2000.0, 2.0e-11]])
mu, w = np.polynomial.legendre.leggauss(4)
quadrature = np.column_stack([mu, 2.0 * np.pi * w])
energy = np.zeros((1, 6, 10))
def run_model():
    try:
        compute_macroscopic_fields(bands, quadrature, energy, 1.0e-10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_macroscopic_fields(bands, quadrature, energy, 1.0e-10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive cell width ---
        {
            "setup": """import numpy as np
bands = np.array([[1.0e6, 2000.0, 2.0e-11]])
mu, w = np.polynomial.legendre.leggauss(4)
quadrature = np.column_stack([mu, 2.0 * np.pi * w])
energy = np.zeros((1, 4, 10))
def run_model():
    try:
        compute_macroscopic_fields(bands, quadrature, energy, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_macroscopic_fields(bands, quadrature, energy, 0.0)
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
