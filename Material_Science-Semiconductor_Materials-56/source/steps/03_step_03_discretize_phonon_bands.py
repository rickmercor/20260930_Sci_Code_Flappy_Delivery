"""
Compress the sampled spectrum into a small set of representative phonon bands of equal contribution to the bulk conductivity, each carrying one heat capacity, one group velocity and one relaxation time.

A deterministic Boltzmann solve carries one transport problem per band and per propagation direction, so the number of bands multiplies the cost of everything downstream and must be kept to a dozen or so. The question is which dozen. Slicing the frequency axis uniformly is the obvious choice and the wrong one, because it spends most of its bands on the zone-edge modes, which are numerous and carry heat capacity but travel almost nowhere, and lumps together the long-wavelength modes whose mean free paths differ by orders of magnitude and whose spread is the entire origin of the size effect. The variable that organises the spectrum correctly is the mean free path, and the natural measure on it is the thermal conductivity accumulated as the mean free path grows: bands are formed by cutting the mean-free-path-ordered spectrum at equally spaced levels of that accumulated conductivity, so that every band is equally important to bulk transport while spanning whatever range of mean free paths it takes to be so.




Reducing a band to a single mode is a second choice with consequences. A band has three degrees of freedom available, a heat capacity, a group velocity and a relaxation time, and three natural quantities to preserve. Its heat capacity must be preserved because the total heat capacity fixes both the local equilibrium distribution and the way a volumetric source is partitioned among bands. Its contribution to the bulk conductivity must be preserved because otherwise the reduced model does not reproduce the diffusive limit, and a comparison against a Fourier prediction that uses the true bulk conductivity would then be measuring a discretisation error rather than a size effect. The remaining freedom is fixed by taking the band group velocity to be the heat-capacity-weighted mean of the group velocities it contains, which is the weighting under which the band reproduces the correct ballistic energy flux carried per unit temperature difference. The relaxation time is then not free: it is whatever value makes the kinetic-theory conductivity of the representative mode equal the conductivity of the shells it replaces, which for a given heat capacity and group velocity is three times that conductivity divided by the heat capacity and the squared velocity.




The resulting band mean free paths span the whole physically relevant range, from a couple of tens of nanometres for the band that holds most of the heat capacity to tens of micrometres for the ballistic tail that holds almost none. Every band is required to be non-empty; a partition that leaves a band with no shells means the spectral sampling is too coarse for the requested number of bands and the reduced model is not defined.

Returns
-------
np.ndarray of shape (n_bands, 3), float: heat capacity (J m^-3 K^-1), group velocity (m/s) and relaxation time (s) of each representative band, ordered by increasing mean free path.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def discretize_phonon_bands(spectral: np.ndarray, impurity: float,
                            umklapp: float, n_bands: int) -> np.ndarray:
    """Reduce the sampled spectrum to representative phonon bands.

    Parameters
    ----------
    spectral : np.ndarray
        Array of shape (n_shells, 3) holding the angular frequency, group
        velocity and volumetric heat capacity of every sampled shell.
    impurity : float
        Coefficient of the fourth-power scattering channel in s^3
        (impurity >= 0).
    umklapp : float
        Coefficient of the second-power scattering channel in s
        (umklapp > 0).
    n_bands : int
        Number of representative bands (1 <= n_bands <= n_shells).

    Returns
    -------
    bands : np.ndarray
        Array of shape (n_bands, 3) whose columns are the volumetric heat
        capacity in J/(m^3 K), the group velocity in m/s and the relaxation
        time in s of each representative band, ordered by increasing mean
        free path.

    Raises
    ------
    ValueError
        If ``spectral`` has an invalid shape or contains non-finite or
        non-physical values, if either scattering coefficient is outside its
        stated domain, if ``n_bands`` is not an integer in
        ``[1, n_shells]``, or if the requested partition contains an empty or
        zero-heat-capacity band.
    """
    return bands  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_discretize_phonon_bands(spectral: np.ndarray, impurity: float,
                                    umklapp: float, n_bands: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    table = np.asarray(spectral, dtype=float)
    if table.ndim != 2 or table.shape[1] != 3 or table.shape[0] < 1:
        raise ValueError("spectral must be a 2D array of shape (n_shells, 3)")
    if not np.all(np.isfinite(table)):
        raise ValueError("spectral must be finite")
    if np.any(table[:, 0] <= 0.0):
        raise ValueError("spectral frequencies must be > 0")
    if not (isinstance(impurity, (int, float, np.floating, np.integer))
            and not isinstance(impurity, bool)
            and np.isfinite(impurity) and float(impurity) >= 0.0):
        raise ValueError("impurity must be a finite number >= 0")
    if not (isinstance(umklapp, (int, float, np.floating, np.integer))
            and not isinstance(umklapp, bool)
            and np.isfinite(umklapp) and float(umklapp) > 0.0):
        raise ValueError("umklapp must be a finite number > 0")
    if not (isinstance(n_bands, (int, np.integer)) and not isinstance(n_bands, bool)
            and 1 <= int(n_bands) <= table.shape[0]):
        raise ValueError("n_bands must be an integer in [1, n_shells]")

    omega, velocity, capacity = table[:, 0], table[:, 1], table[:, 2]
    n_bands = int(n_bands)
    lifetime = 1.0 / (float(impurity) * omega ** 4 + float(umklapp) * omega ** 2)
    mean_free_path = velocity * lifetime
    conductivity = capacity * velocity ** 2 * lifetime / 3.0

    if not np.all(conductivity > 0.0):
        raise ValueError("every shell must carry a strictly positive conductivity")

    # Cut the mean-free-path-ordered spectrum at equally spaced levels of the
    # accumulated conductivity; a shell belongs to the band that contains the
    # accumulated fraction reached once that shell is included.
    order = np.argsort(mean_free_path, kind="stable")
    accumulated = np.cumsum(conductivity[order])
    label = np.minimum((n_bands * accumulated / accumulated[-1]).astype(int),
                       n_bands - 1)

    bands = np.zeros((n_bands, 3), dtype=float)
    for index in range(n_bands):
        members = order[label == index]
        if members.size == 0:
            raise ValueError("band discretisation produced an empty band")
        band_capacity = float(capacity[members].sum())
        band_conductivity = float(conductivity[members].sum())
        if band_capacity <= 0.0:
            raise ValueError("band discretisation produced a band of zero heat capacity")
        band_velocity = float(np.sum(capacity[members] * velocity[members])
                              / band_capacity)
        band_lifetime = 3.0 * band_conductivity / (band_capacity * band_velocity ** 2)
        bands[index] = (band_capacity, band_velocity, band_lifetime)

    return bands

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark twelve-band reduction (normal scenario) ---
        {
            "setup": """import numpy as np
hbar, kb = 1.054571817e-34, 1.380649e-23
n_shells, omega_max, density, temperature = 400, 2.0 * np.pi * 9.0e12, 5.0e28, 300.0
kd = (6.0 * np.pi ** 2 * density) ** (1.0 / 3.0)
dk = kd / n_shells
k = (np.arange(n_shells) + 0.5) * dk
phase = 0.5 * np.pi * k / kd
omega = omega_max * np.sin(phase)
velocity = omega_max * (0.5 * np.pi / kd) * np.cos(phase)
x = hbar * omega / (kb * temperature)
capacity = (3.0 * k ** 2 * dk / (2.0 * np.pi ** 2)) * kb * x ** 2 * np.exp(x) / (np.exp(x) - 1.0) ** 2
spectral = np.column_stack([omega, velocity, capacity])
impurity, umklapp, n_bands = 1.32e-45, 3.5350977862996176e-17, 12
band_scale = np.array([1.0e-6, 1.0e-3, 1.0e12])
""",
            "call": "discretize_phonon_bands(spectral, impurity, umklapp, n_bands) * band_scale",
            "gold_call": "_oracle_discretize_phonon_bands(spectral, impurity, umklapp, n_bands) * band_scale",
        },
        # --- Valid: same spectrum reduced far more aggressively ---
        {
            "setup": """import numpy as np
omega = np.linspace(1.0e12, 5.0e13, 300)
velocity = np.linspace(6200.0, 400.0, 300)
capacity = np.linspace(500.0, 3.0e4, 300)
spectral = np.column_stack([omega, velocity, capacity])
impurity, umklapp, n_bands = 1.32e-45, 3.0e-17, 4
band_scale = np.array([1.0e-6, 1.0e-3, 1.0e12])
""",
            "call": "discretize_phonon_bands(spectral, impurity, umklapp, n_bands) * band_scale",
            "gold_call": "_oracle_discretize_phonon_bands(spectral, impurity, umklapp, n_bands) * band_scale",
        },
        # --- Boundary: one band, which must reproduce the grey model exactly ---
        {
            "setup": """import numpy as np
omega = np.linspace(2.0e12, 4.0e13, 150)
velocity = np.linspace(6000.0, 1200.0, 150)
capacity = np.linspace(800.0, 1.5e4, 150)
spectral = np.column_stack([omega, velocity, capacity])
impurity, umklapp, n_bands = 1.0e-45, 2.0e-17, 1
band_scale = np.array([1.0e-6, 1.0e-3, 1.0e12])
""",
            "call": "discretize_phonon_bands(spectral, impurity, umklapp, n_bands) * band_scale",
            "gold_call": "_oracle_discretize_phonon_bands(spectral, impurity, umklapp, n_bands) * band_scale",
        },
        # --- Edge: mean free paths spanning six orders of magnitude ---
        {
            "setup": """import numpy as np
omega = np.logspace(np.log10(1.0e11), np.log10(6.0e13), 200)
velocity = 6000.0 * np.exp(-omega / 6.0e13)
capacity = 1.0e2 + 2.0e4 * (omega / 6.0e13) ** 2
spectral = np.column_stack([omega, velocity, capacity])
impurity, umklapp, n_bands = 1.32e-45, 4.0e-17, 3
band_scale = np.array([1.0e-6, 1.0e-3, 1.0e12])
""",
            "call": "discretize_phonon_bands(spectral, impurity, umklapp, n_bands) * band_scale",
            "gold_call": "_oracle_discretize_phonon_bands(spectral, impurity, umklapp, n_bands) * band_scale",
        },
        # --- Invalid: more bands requested than sampled shells ---
        {
            "setup": """import numpy as np
omega = np.linspace(5.0e12, 3.0e13, 8)
spectral = np.column_stack([omega, np.full(8, 5000.0), np.full(8, 1.0e3)])
def run_model():
    try:
        discretize_phonon_bands(spectral, 1.32e-45, 4.0e-17, 12)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_discretize_phonon_bands(spectral, 1.32e-45, 4.0e-17, 12)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: vanishing anharmonic coefficient leaves the lifetimes undefined ---
        {
            "setup": """import numpy as np
omega = np.linspace(5.0e12, 3.0e13, 40)
spectral = np.column_stack([omega, np.full(40, 5000.0), np.full(40, 1.0e3)])
def run_model():
    try:
        discretize_phonon_bands(spectral, 1.32e-45, 0.0, 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_discretize_phonon_bands(spectral, 1.32e-45, 0.0, 5)
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
