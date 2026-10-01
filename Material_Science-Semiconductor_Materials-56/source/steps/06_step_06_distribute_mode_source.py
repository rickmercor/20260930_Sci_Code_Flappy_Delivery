"""
Convert a volumetric dissipation profile into the per-band, per-direction source density that enters each band-direction transport equation.

The power that a transistor dissipates does not enter the phonon system as a phonon distribution of its own choosing. Electrons accelerated by the drain field lose their energy to the lattice through electron-phonon scattering on a timescale short compared with the phonon transport timescale, and the phonons so created are re-distributed by anharmonic processes before they travel any appreciable distance. The consequence, and the modelling choice the solver makes, is that a volumetric heat source is an equilibrium phonon source: it is isotropic, carrying no net momentum and therefore no net heat flux of its own, and it is shared among the phonon bands in proportion to the heat capacity each band carries, exactly as an increment of equilibrium energy at a slightly raised temperature would be. Partitioning the same power in proportion to band conductivity, or in proportion to band population, would inject a different amount of energy into the ballistic long-mean-free-path bands and change the predicted hot-spot temperature; nothing in the transport equation itself decides between these, so it has to be decided by the physics of how the energy arrives.




The bookkeeping that makes this consistent has two factors. Dividing by the total heat capacity of all bands and multiplying by the heat capacity of one band gives the fraction of the power that band receives, so that summing over bands returns the deposited power exactly. Dividing by the total angular weight spreads that fraction isotropically over the propagation directions, so that integrating the source over the sphere with the quadrature weights returns the band's share and not some multiple of it. The two normalisations together guarantee global energy conservation of the discrete system: the angular and spectral moment of the source is the deposited power density in every cell, which is what allows the divergence of the heat flux to equal the dissipation at convergence and what makes the comparison with a Fourier solution driven by the same dissipation meaningful.




Because the source is isotropic, its value does not depend on the propagation direction, and the array returned carries only a band index and a cell index; the direction dependence is the constant factor already divided out.

Returns
-------
np.ndarray of shape (n_bands, n_cells), float: the isotropic per-band source density in W m^-3 sr^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def distribute_mode_source(bands: np.ndarray, heat_source: np.ndarray,
                           solid_angle: float) -> np.ndarray:
    """Split a volumetric dissipation profile among bands and directions.

    Parameters
    ----------
    bands : np.ndarray
        Array of shape (n_bands, 3) holding the heat capacity, group velocity
        and relaxation time of each representative band.
    heat_source : np.ndarray
        Array of shape (n_cells,) holding the volumetric dissipation in each
        finite volume, in W/m^3.
    solid_angle : float
        Total angular quadrature weight in steradians (solid_angle > 0).

    Returns
    -------
    mode_source : np.ndarray
        Array of shape (n_bands, n_cells) holding the source density per unit
        solid angle of every band in every cell, in W/(m^3 sr).

    Raises
    ------
    ValueError
        If ``bands`` or ``heat_source`` has an invalid shape or contains
        non-finite values, if any band heat capacity is not strictly positive,
        or if ``solid_angle`` is not finite and strictly positive.
    """
    return mode_source  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_distribute_mode_source(bands: np.ndarray, heat_source: np.ndarray,
                                   solid_angle: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    band_table = np.asarray(bands, dtype=float)
    source = np.asarray(heat_source, dtype=float)
    if band_table.ndim != 2 or band_table.shape[1] != 3 or band_table.shape[0] < 1:
        raise ValueError("bands must be a 2D array of shape (n_bands, 3)")
    if source.ndim != 1 or source.size < 1:
        raise ValueError("heat_source must be a 1D array of length >= 1")
    if not (np.all(np.isfinite(band_table)) and np.all(np.isfinite(source))):
        raise ValueError("bands and heat_source must be finite")
    if np.any(band_table[:, 0] <= 0.0):
        raise ValueError("band heat capacities must be > 0")
    if not (isinstance(solid_angle, (int, float, np.floating, np.integer))
            and not isinstance(solid_angle, bool)
            and np.isfinite(solid_angle) and float(solid_angle) > 0.0):
        raise ValueError("solid_angle must be a finite number > 0")

    capacity = band_table[:, 0]

    # Equilibrium source: shared among bands by heat capacity, spread
    # isotropically over the ordinates by the total angular weight.
    share = capacity / capacity.sum() / float(solid_angle)

    return np.outer(share, source)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark hot spot over the central fifth (normal scenario) ---
        {
            "setup": """import numpy as np
bands = np.array([
    [1.2896e+06, 1451.03, 1.3540e-11],
    [2.1132e+05, 3657.21, 1.2908e-11],
    [1.1727e+05, 4365.08, 1.6869e-11],
    [7.0648e+04, 4852.99, 2.2256e-11],
])
n_cells, length = 200, 20.0e-9
dx = length / n_cells
xc = (np.arange(n_cells) + 0.5) * dx
heat_source = np.where(np.abs(xc - 0.5 * length) <= 0.1 * length, 1.5e19, 0.0)
solid_angle = 4.0 * np.pi
""",
            "call": "distribute_mode_source(bands, heat_source, solid_angle)",
            "gold_call": "_oracle_distribute_mode_source(bands, heat_source, solid_angle)",
        },
        # --- Valid: smoothly varying dissipation profile ---
        {
            "setup": """import numpy as np
bands = np.array([[5.0e5, 2000.0, 1.0e-11], [3.0e5, 4000.0, 5.0e-11]])
xc = np.linspace(0.0, 1.0, 50)
heat_source = 1.0e18 * np.exp(-((xc - 0.5) / 0.1) ** 2)
solid_angle = 4.0 * np.pi
""",
            "call": "distribute_mode_source(bands, heat_source, solid_angle)",
            "gold_call": "_oracle_distribute_mode_source(bands, heat_source, solid_angle)",
        },
        # --- Boundary: no dissipation anywhere ---
        {
            "setup": """import numpy as np
bands = np.array([[1.0e6, 3000.0, 2.0e-11], [2.0e5, 5000.0, 1.0e-10]])
heat_source = np.zeros(16)
solid_angle = 4.0 * np.pi
""",
            "call": "distribute_mode_source(bands, heat_source, solid_angle)",
            "gold_call": "_oracle_distribute_mode_source(bands, heat_source, solid_angle)",
        },
        # --- Edge: one band and one cell ---
        {
            "setup": """import numpy as np
bands = np.array([[1.8e6, 1500.0, 1.4e-11]])
heat_source = np.array([1.5e19])
solid_angle = 4.0 * np.pi
""",
            "call": "distribute_mode_source(bands, heat_source, solid_angle)",
            "gold_call": "_oracle_distribute_mode_source(bands, heat_source, solid_angle)",
        },
        # --- Invalid: a band with vanishing heat capacity cannot take a share ---
        {
            "setup": """import numpy as np
bands = np.array([[0.0, 3000.0, 1.0e-11], [1.0e6, 5000.0, 1.0e-10]])
heat_source = np.ones(8) * 1.0e18
def run_model():
    try:
        distribute_mode_source(bands, heat_source, 4.0 * np.pi)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_distribute_mode_source(bands, heat_source, 4.0 * np.pi)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive total angular weight ---
        {
            "setup": """import numpy as np
bands = np.array([[1.0e6, 3000.0, 1.0e-11]])
heat_source = np.ones(8) * 1.0e18
def run_model():
    try:
        distribute_mode_source(bands, heat_source, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_distribute_mode_source(bands, heat_source, 0.0)
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
