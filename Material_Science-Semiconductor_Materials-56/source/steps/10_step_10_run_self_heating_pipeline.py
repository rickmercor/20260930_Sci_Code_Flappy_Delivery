"""
Chain the sub-problem functions 01-09 end to end on the film testbed and return the self-heating amplification factor, the peak kinetic temperature rise divided by the peak Fourier temperature rise.

This step runs the whole measurement. It (i) samples the phonon spectrum with sub-problem 01 and fixes the anharmonic scattering strength against the bulk conductivity with sub-problem 02, (ii) compresses the spectrum into representative bands of equal conductivity contribution with sub-problem 03 and builds the ordinate set with sub-problem 04, (iii) assembles every temperature-independent band-direction operator with sub-problem 05 once before the outer loop, (iv) lays the hot spot on the mesh and shares it among the bands with sub-problem 06, (v) obtains the Fourier reference by calling the macroscopic solve of sub-problem 09 with no kinetic input, (vi) iterates the kinetic solution, passing the preassembled operators to sub-problem 07 and taking the moments of the result with sub-problem 08 until the temperature field stops moving, and (vii) divides the peak kinetic temperature rise by the peak Fourier rise.




The starting guess for the iteration is the Fourier field itself, which costs nothing extra since it is needed as the reference anyway and which is closer to the answer than a cold start in every regime. The temperature update is the collision-moment closure of sub-problem 08; the macroscopic update of sub-problem 09 reaches the same fixed point, since the collision balance and the discrete energy balance are the same statement once the flux divergence is assembled from the upwinded transport faces, so the reported number does not depend on which of the two drives the iteration.




The returned scalar is what a compact thermal model needs and Fourier's law cannot supply: the factor by which the true hot-spot temperature rise exceeds the diffusive estimate at this device size. A value near one would mean that a Fourier description with the bulk conductivity is adequate. A value far above one localises the discrepancy in the ballistic transport of the long-mean-free-path bands, and its reciprocal is the effective conductivity, relative to the bulk value, that a Fourier model would have to be given to reproduce the correct peak temperature for this geometry and this heat-source size.

Returns
-------
float: the self-heating amplification factor of the film, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_self_heating_pipeline(length: float = 2.0e-8, n_cells: int = 200,
                              n_bands: int = 12, n_dirs: int = 16,
                              n_shells: int = 400,
                              omega_max: float = 56548667764616.27,
                              number_density: float = 5.0e28,
                              temperature: float = 300.0,
                              impurity: float = 1.32e-45,
                              kappa_bulk: float = 148.0,
                              power_density: float = 1.5e19,
                              hotspot_fraction: float = 0.2,
                              tol: float = 1.0e-12, max_iter: int = 5000,
                              scheme: str = "sequential") -> float:
    """Run the full self-heating amplification measurement on the film testbed.

    Parameters
    ----------
    length : float
        Film thickness in m (length > 0).
    n_cells : int
        Number of finite volumes across the film (n_cells >= 1).
    n_bands : int
        Number of representative phonon bands
        (1 <= n_bands <= n_shells).
    n_dirs : int
        Number of ordinates; must be even and at least 2.
    n_shells : int
        Number of spectral shells (n_shells >= 1).
    omega_max : float
        Maximum angular frequency of the dispersion in rad/s (omega_max > 0).
    number_density : float
        Atomic number density in m^-3 (number_density > 0).
    temperature : float
        Reference temperature in K (temperature > 0).
    impurity : float
        Coefficient of the fourth-power scattering channel in s^3
        (impurity >= 0).
    kappa_bulk : float
        Bulk thermal conductivity in W/(m K) (kappa_bulk > 0).
    power_density : float
        Volumetric dissipation inside the hot spot in W/m^3
        (power_density > 0).
    hotspot_fraction : float
        Fraction of the film thickness occupied by the centred hot spot,
        0 < hotspot_fraction <= 1.
    tol : float
        Relative max-norm convergence tolerance on the temperature field
        (tol > 0).
    max_iter : int
        Maximum number of outer iterations (max_iter >= 1).
    scheme : str
        Temperature update driving the iteration, either "sequential" or
        "synthetic".

    Returns
    -------
    amplification : float
        Peak kinetic temperature rise divided by the peak Fourier temperature
        rise, as a native Python float.

    Raises
    ------
    ValueError
        If an integer resolution or iteration count is outside its stated
        domain, if ``n_dirs`` is odd, if a physical scalar or tolerance is
        non-finite or outside its stated domain, if ``n_bands`` exceeds
        ``n_shells``, if ``scheme`` is not ``"sequential"`` or
        ``"synthetic"``, or if the discretised source yields no positive
        Fourier temperature rise.
    """
    return amplification  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_self_heating_pipeline(length: float = 2.0e-8, n_cells: int = 200,
                                      n_bands: int = 12, n_dirs: int = 16,
                                      n_shells: int = 400,
                                      omega_max: float = 56548667764616.27,
                                      number_density: float = 5.0e28,
                                      temperature: float = 300.0,
                                      impurity: float = 1.32e-45,
                                      kappa_bulk: float = 148.0,
                                      power_density: float = 1.5e19,
                                      hotspot_fraction: float = 0.2,
                                      tol: float = 1.0e-12,
                                      max_iter: int = 5000,
                                      scheme: str = "sequential") -> float:
    import numpy as np

    # The grading harness concatenates the sub-problems into one namespace.
    # Bind the preceding oracles directly so the final reference pipeline
    # cannot fall back to candidate code or depend on the filesystem.
    build_spectrum = _oracle_build_spectral_model
    calibrate = _oracle_calibrate_umklapp_coefficient
    discretize = _oracle_discretize_phonon_bands
    build_quadrature = _oracle_build_angular_quadrature
    assemble = _oracle_assemble_transport_operator
    split_source = _oracle_distribute_mode_source
    sweep = _oracle_solve_transport_sweep
    moments = _oracle_compute_macroscopic_fields
    macroscopic = _oracle_update_macroscopic_temperature

    # -- Validate the orchestrator inputs.
    for name, value, floor in (("n_cells", n_cells, 1), ("n_bands", n_bands, 1),
                               ("n_dirs", n_dirs, 2), ("n_shells", n_shells, 1),
                               ("max_iter", max_iter, 1)):
        if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                and int(value) >= floor):
            raise ValueError(f"{name} must be an integer >= {floor}")
    if int(n_dirs) % 2 != 0:
        raise ValueError("n_dirs must be an even integer >= 2")
    if int(n_bands) > int(n_shells):
        raise ValueError("n_bands must not exceed n_shells")
    for name, value in (("length", length), ("omega_max", omega_max),
                        ("number_density", number_density),
                        ("temperature", temperature), ("kappa_bulk", kappa_bulk),
                        ("power_density", power_density), ("tol", tol)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool)
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(impurity, (int, float, np.floating, np.integer))
            and not isinstance(impurity, bool)
            and np.isfinite(impurity) and float(impurity) >= 0.0):
        raise ValueError("impurity must be a finite number >= 0")
    if not (isinstance(hotspot_fraction, (int, float, np.floating, np.integer))
            and not isinstance(hotspot_fraction, bool)
            and np.isfinite(hotspot_fraction)
            and 0.0 < float(hotspot_fraction) <= 1.0):
        raise ValueError("hotspot_fraction must be a finite number in (0, 1]")
    if scheme not in ("sequential", "synthetic"):
        raise ValueError("scheme must be either 'sequential' or 'synthetic'")

    n_cells = int(n_cells)
    length = float(length)

    # -- Sub-problems 01-04: material model, band reduction and ordinate set.
    spectral = build_spectrum(int(n_shells), float(omega_max),
                              float(number_density), float(temperature))
    umklapp = calibrate(spectral, float(impurity), float(kappa_bulk))
    bands = discretize(spectral, float(impurity), umklapp, int(n_bands))
    quadrature = build_quadrature(int(n_dirs))
    solid_angle = float(np.asarray(quadrature, dtype=float)[:, 1].sum())

    # -- Geometry: a centred hot spot on a uniform mesh.
    cell_size = length / n_cells
    centres = (np.arange(n_cells) + 0.5) * cell_size
    inside = (np.abs(centres - 0.5 * length)
              <= 0.5 * float(hotspot_fraction) * length + 1.0e-12 * length)
    heat_source = np.where(inside, float(power_density), 0.0)

    # -- Sub-problem 05: assemble each temperature-independent matrix once.
    #    Step 07 reuses this bank for every right-hand-side substitution.
    operators = assemble(bands, quadrature, n_cells, cell_size)

    # -- Sub-problem 06: share the dissipation among bands and directions.
    mode_source = split_source(bands, heat_source, solid_angle)

    # -- Sub-problem 09 with no kinetic input: the Fourier reference.
    zeros = np.zeros(n_cells)
    fourier = macroscopic(float(kappa_bulk), heat_source, zeros, zeros, cell_size)
    peak_fourier = float(np.max(fourier))
    if not peak_fourier > 0.0:
        raise ValueError("the Fourier reference has no positive temperature rise")

    # -- Sub-problems 05, 07, 08: outer iteration on the kinetic solution.
    field = fourier.copy()
    for _ in range(int(max_iter)):
        energy = sweep(bands, quadrature, operators, mode_source, field)
        fields = moments(bands, quadrature, energy, cell_size)
        if scheme == "sequential":
            updated = fields[0]
        else:
            updated = macroscopic(float(kappa_bulk), heat_source, fields[2],
                                  field, cell_size)
        scale = float(np.max(np.abs(updated)))
        residual = float(np.max(np.abs(updated - field))) / max(scale, 1.0e-300)
        field = updated
        if residual <= float(tol):
            break

    return float(np.max(field) / peak_fourier)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Integration: small film, whole pipeline at reduced resolution ---
        {
            "setup": """import numpy as np
length = 20.0e-9
n_cells = 40
n_bands = 6
n_dirs = 8
""",
            "call": "run_self_heating_pipeline(length, n_cells, n_bands, n_dirs)",
            "gold_call": "_oracle_run_self_heating_pipeline(length, n_cells, n_bands, n_dirs)",
        },
        # --- Valid: distinct medium-resolution central-hotspot configuration ---
        {
            "setup": """import numpy as np
length = 30.0e-9
n_cells = 36
n_bands = 5
n_dirs = 6
hotspot_fraction = 0.5
""",
            "call": "run_self_heating_pipeline(length, n_cells, n_bands, n_dirs, hotspot_fraction=hotspot_fraction)",
            "gold_call": "_oracle_run_self_heating_pipeline(length, n_cells, n_bands, n_dirs, hotspot_fraction=hotspot_fraction)",
        },
        # --- Integration (boundary): the whole film is the hot spot ---
        {
            "setup": """import numpy as np
length = 50.0e-9
n_cells = 30
n_bands = 4
n_dirs = 6
""",
            "call": "run_self_heating_pipeline(length, n_cells, n_bands, n_dirs, hotspot_fraction=1.0)",
            "gold_call": "_oracle_run_self_heating_pipeline(length, n_cells, n_bands, n_dirs, hotspot_fraction=1.0)",
        },
        # --- Integration (edge): thicker film driven towards the diffusive limit ---
        {
            "setup": """import numpy as np
length = 1.0e-6
n_cells = 30
n_bands = 5
n_dirs = 6
""",
            "call": "run_self_heating_pipeline(length, n_cells, n_bands, n_dirs, power_density=1.5e15, scheme='synthetic')",
            "gold_call": "_oracle_run_self_heating_pipeline(length, n_cells, n_bands, n_dirs, power_density=1.5e15, scheme='synthetic')",
        },
        # --- Invalid: odd ordinate count ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_self_heating_pipeline(n_dirs=7)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_self_heating_pipeline(n_dirs=7)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: unknown iteration scheme ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_self_heating_pipeline(scheme="newton")
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_self_heating_pipeline(scheme="newton")
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
