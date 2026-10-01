"""
Chain the sub-problem functions 01-09 end to end on the layered via and return the largest von Mises stress found on the requested cross-section at the requested instant, expressed in megapascals.

This step runs the whole measurement. It (i) derives the layer thermal diffusivities, volumetric heat capacities and shear moduli from the tabulated material data, (ii) builds the axial modes with sub-problem 01 and, for each of them, the layered radial spectrum with sub-problems 02 and 03 and the expansion amplitudes with sub-problem 04, (iii) sums the temperature of each axial mode at each evaluation radius with sub-problem 05, feeds it to sub-problem 06 to obtain the thermally generated fields at the interfaces and solves for the complementary amplitudes of every axial mode with sub-problems 07 and 08, and (iv) sweeps the requested cross-section layer by layer, again taking the temperature from sub-problem 05, summing the modal stresses of sub-problem 09 and reducing them to an equivalent scalar.




The temperature enters the elastic problem twice over and by two different routes. Differentiated, it generates the thermoelastic displacement potential, which sub-problem 06 obtains from the modal data directly; undifferentiated, it is subtracted from each of the three normal stresses by the constitutive law, and that is the value sub-problem 05 supplies. Passing it explicitly is what keeps a single temperature field behind both the thermal and the mechanical halves of the calculation: an error in the temperature series cannot cancel itself between the two.




The sweep is organised by layer rather than by radius because the stress state is not a single-valued function of position at an interface: the traction components are continuous there but the hoop and axial components are not, so a point on an interface carries two distinct stress states and both must be examined. Sampling each layer between its own inner and outer radius, endpoints included, makes both of them available and is why the peak reported here is a genuine peak rather than an average of the two sides.




The scalar returned is the equivalent stress that a ductile yield or interfacial-failure criterion consumes, and its magnitude relative to the strength of the interface is what decides whether a via design is viable. Its usefulness depends on being a transient quantity: the equivalent stress tracks the temperature difference that produced it, so the number reported is meaningful only together with the instant at which it is evaluated, and repeating the calculation over a sequence of instants traces the load history that drives fatigue at the metal-liner and liner-substrate interfaces.

Returns
-------
float: the largest von Mises stress found on the requested cross-section at the requested instant, in MPa, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_tsv_thermal_stress_pipeline(radii: tuple = (15.0e-6, 16.0e-6, 25.0e-6),
                                    conductivities: tuple = (400.0, 1.4, 130.0),
                                    densities: tuple = (8960.0, 2200.0, 2329.0),
                                    specific_heats: tuple = (385.0, 730.0, 700.0),
                                    expansions: tuple = (17.0e-6, 0.5e-6, 2.6e-6),
                                    youngs_moduli: tuple = (110.0e9, 70.0e9, 170.0e9),
                                    poisson: tuple = (0.35, 0.17, 0.28),
                                    height: float = 200.0e-6,
                                    initial_rise: float = 100.0,
                                    plane_fraction: float = 0.9,
                                    time: float = 5.0e-5,
                                    n_axial: int = 12, n_radial: int = 6,
                                    n_samples: int = 401) -> float:
    """Return the peak von Mises stress on a cross-section of the layered via.

    Parameters
    ----------
    radii : tuple
        Outer radius of each layer in m, strictly increasing and positive.
    conductivities : tuple
        Thermal conductivity of each layer in W/(m K), all positive.
    densities : tuple
        Density of each layer in kg/m^3, all positive.
    specific_heats : tuple
        Specific heat capacity of each layer in J/(kg K), all positive.
    expansions : tuple
        Coefficient of thermal expansion of each layer in 1/K.
    youngs_moduli : tuple
        Young's modulus of each layer in Pa, all positive.
    poisson : tuple
        Poisson's ratio of each layer, each in (-1, 0.5).
    height : float
        Height of the via in m (height > 0).
    initial_rise : float
        Uniform initial temperature of the via above ambient in K.
    plane_fraction : float
        Position of the examined cross-section as a fraction of the height,
        0 <= plane_fraction <= 1.
    time : float
        Instant at which the stress is evaluated in s (time >= 0).
    n_axial : int
        Number of axial modes retained (n_axial >= 1).
    n_radial : int
        Number of radial modes retained per axial mode (n_radial >= 1).
    n_samples : int
        Number of radii sampled within each layer, endpoints included
        (n_samples >= 2).

    Returns
    -------
    peak_stress : float
        Largest von Mises stress on the cross-section in MPa, as a native
        Python float.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above; invalid input
        must raise ValueError rather than return a sentinel value.
    """
    return peak_stress  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_tsv_thermal_stress_pipeline(
        radii: tuple = (15.0e-6, 16.0e-6, 25.0e-6),
        conductivities: tuple = (400.0, 1.4, 130.0),
        densities: tuple = (8960.0, 2200.0, 2329.0),
        specific_heats: tuple = (385.0, 730.0, 700.0),
        expansions: tuple = (17.0e-6, 0.5e-6, 2.6e-6),
        youngs_moduli: tuple = (110.0e9, 70.0e9, 170.0e9),
        poisson: tuple = (0.35, 0.17, 0.28),
        height: float = 200.0e-6, initial_rise: float = 100.0,
        plane_fraction: float = 0.9, time: float = 5.0e-5,
        n_axial: int = 12, n_radial: int = 6, n_samples: int = 401) -> float:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import glob
    import importlib.util
    import os
    import sys

    import numpy as np

    # -- Resolve the oracle functions of sub-problems 01-09. Preference order:
    #    (1) already present in the executing namespace (shared-namespace
    #    harness), (2) loaded from a sibling sub-problem file matched by name
    #    pattern (standalone execution; file prefixes may vary), (3) the
    #    public function of the same step if the harness injected it.
    def _resolve_step(oracle_name, pattern):
        namespace = globals()
        candidate = namespace.get(oracle_name)
        if callable(candidate):
            return candidate
        search_dirs = []
        if "__file__" in namespace:
            search_dirs.append(os.path.dirname(os.path.abspath(namespace["__file__"])))
        cwd = os.getcwd()
        search_dirs += [cwd, os.path.join(cwd, "sub_problems")]
        if sys.argv and sys.argv[0]:
            search_dirs.append(os.path.dirname(os.path.abspath(sys.argv[0])))
        seen = set()
        search_dirs = [d for d in search_dirs if not (d in seen or seen.add(d))]
        for directory in search_dirs:
            for path in sorted(glob.glob(os.path.join(directory, pattern))):
                spec = importlib.util.spec_from_file_location(
                    os.path.basename(path)[:-3], path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, oracle_name):
                    return getattr(module, oracle_name)
        public = namespace.get(oracle_name.replace("_oracle_", "", 1))
        if callable(public):
            return public
        raise RuntimeError(f"cannot resolve required step function {oracle_name}")

    axial_modes_of = _resolve_step(
        "_oracle_compute_axial_modes", "*compute_axial_modes*.py")
    eigenvalues_of = _resolve_step(
        "_oracle_solve_radial_eigenvalues", "*solve_radial_eigenvalues*.py")
    eigenfunction_of = _resolve_step(
        "_oracle_build_radial_eigenfunction", "*build_radial_eigenfunction*.py")
    amplitude_of = _resolve_step(
        "_oracle_compute_expansion_coefficient", "*compute_expansion_coefficient*.py")
    temperature_of = _resolve_step(
        "_oracle_evaluate_temperature_rise", "*evaluate_temperature_rise*.py")
    thermal_of = _resolve_step(
        "_oracle_compute_thermal_source_terms", "*compute_thermal_source_terms*.py")
    system_of = _resolve_step(
        "_oracle_assemble_love_system", "*assemble_love_system*.py")
    solve_of = _resolve_step(
        "_oracle_solve_love_coefficients", "*solve_love_coefficients*.py")
    stress_of = _resolve_step(
        "_oracle_evaluate_stress_components", "*evaluate_stress_components*.py")

    # -- Validate the orchestrator inputs.
    for name, value, floor in (("n_axial", n_axial, 1), ("n_radial", n_radial, 1),
                               ("n_samples", n_samples, 2)):
        if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                and int(value) >= floor):
            raise ValueError(f"{name} must be an integer >= {floor}")
    for name, value in (("height", height),):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool)
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(time, (int, float, np.floating, np.integer))
            and not isinstance(time, bool) and np.isfinite(time) and float(time) >= 0.0):
        raise ValueError("time must be a finite number >= 0")
    if not (isinstance(initial_rise, (int, float, np.floating, np.integer))
            and not isinstance(initial_rise, bool) and np.isfinite(initial_rise)):
        raise ValueError("initial_rise must be a finite number")
    if not (isinstance(plane_fraction, (int, float, np.floating, np.integer))
            and not isinstance(plane_fraction, bool) and np.isfinite(plane_fraction)
            and 0.0 <= float(plane_fraction) <= 1.0):
        raise ValueError("plane_fraction must be a finite number in [0, 1]")

    radii = np.asarray(radii, dtype=float).ravel()
    conductivities = np.asarray(conductivities, dtype=float).ravel()
    densities = np.asarray(densities, dtype=float).ravel()
    specific_heats = np.asarray(specific_heats, dtype=float).ravel()
    expansions = np.asarray(expansions, dtype=float).ravel()
    youngs_moduli = np.asarray(youngs_moduli, dtype=float).ravel()
    poisson = np.asarray(poisson, dtype=float).ravel()
    n_layers = radii.size
    if n_layers < 1:
        raise ValueError("radii must hold at least one layer")
    for name, array in (("conductivities", conductivities), ("densities", densities),
                        ("specific_heats", specific_heats), ("expansions", expansions),
                        ("youngs_moduli", youngs_moduli), ("poisson", poisson)):
        if array.size != n_layers:
            raise ValueError(f"{name} must have one entry per layer")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
    if radii[0] <= 0.0 or np.any(np.diff(radii) <= 0.0):
        raise ValueError("radii must be positive and strictly increasing")
    for name, array in (("conductivities", conductivities), ("densities", densities),
                        ("specific_heats", specific_heats),
                        ("youngs_moduli", youngs_moduli)):
        if np.any(array <= 0.0):
            raise ValueError(f"{name} must be positive")
    if np.any(poisson <= -1.0) or np.any(poisson >= 0.5):
        raise ValueError("every Poisson's ratio must lie in (-1, 0.5)")

    height = float(height)
    time = float(time)
    n_axial = int(n_axial)
    n_radial = int(n_radial)
    n_samples = int(n_samples)
    inner = np.concatenate([[0.0], radii[:-1]])
    plane = float(plane_fraction) * height

    # -- Derived layer properties.
    capacities = densities * specific_heats
    diffusivities = conductivities / capacities
    shear_moduli = youngs_moduli / (2.0 * (1.0 + poisson))

    # -- Sub-problems 01-04: the modal description of the temperature field.
    axial = np.asarray(axial_modes_of(height, n_axial), dtype=float)
    decays = np.zeros((n_axial, n_radial))
    eigen = np.zeros((n_axial, n_radial, n_layers, 3))
    amplitudes = np.zeros((n_axial, n_radial))
    for m in range(n_axial):
        eta = float(axial[m, 0])
        decays[m] = np.asarray(eigenvalues_of(radii, conductivities, diffusivities,
                                              eta, n_radial), dtype=float)
        for n in range(n_radial):
            eigen[m, n] = np.asarray(
                eigenfunction_of(radii, conductivities, diffusivities, eta,
                                 float(decays[m, n])), dtype=float)
            amplitudes[m, n] = float(
                amplitude_of(radii, capacities, eigen[m, n], axial[m], initial_rise))

    # -- Sub-problem 05: the modal temperature at a radius, stripped of its
    #    axial factor, which is what an axial position of zero returns.
    def _temperature(m, layer, radius):
        return float(temperature_of(axial[m:m + 1], decays[m:m + 1],
                                    eigen[m:m + 1], amplitudes[m:m + 1],
                                    layer, float(radius), 0.0, time))

    # -- Sub-problems 05-08: the complementary amplitudes of every axial mode.
    love = np.zeros((n_axial, n_layers, 4))
    for m in range(n_axial):
        eta = float(axial[m, 0])
        terms = np.zeros((n_layers, 2, 6))
        for i in range(n_layers):
            for j, radius in enumerate((inner[i], radii[i])):
                terms[i, j] = np.asarray(
                    thermal_of(decays[m], eigen[m], amplitudes[m], eta,
                               _temperature(m, i, radius),
                               float(expansions[i]), float(poisson[i]),
                               float(shear_moduli[i]), i, float(radius), time),
                    dtype=float)
        system = system_of(radii, poisson, shear_moduli, eta, terms)
        love[m] = np.asarray(solve_of(system, n_layers), dtype=float)

    # -- Sub-problem 09: sweep the cross-section layer by layer.
    peak = 0.0
    for i in range(n_layers):
        for radius in np.linspace(inner[i], radii[i], n_samples):
            total = np.zeros(4)
            for m in range(n_axial):
                eta = float(axial[m, 0])
                terms = np.asarray(
                    thermal_of(decays[m], eigen[m], amplitudes[m], eta,
                               _temperature(m, i, radius),
                               float(expansions[i]), float(poisson[i]),
                               float(shear_moduli[i]), i, float(radius), time),
                    dtype=float)
                total = total + np.asarray(
                    stress_of(love[m, i], terms, eta, float(poisson[i]),
                              float(shear_moduli[i]), float(radius), plane),
                    dtype=float)
            equivalent = np.sqrt(
                0.5 * ((total[0] - total[1]) ** 2 + (total[1] - total[2]) ** 2
                       + (total[2] - total[0]) ** 2) + 3.0 * total[3] ** 2)
            peak = max(peak, float(equivalent))

    return peak / 1.0e6

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Integration: radial truncation is the only thing varied ---
        # Two radial modes per axial mode must reproduce six, with the axial
        # truncation and the sampling held fixed, because the third radial
        # eigenvalue of this via is three orders of magnitude above the second
        # and has already decayed away at the requested instant. Comparing the
        # candidate at n_radial = 2 against the oracle at n_radial = 6 isolates
        # that insensitivity from every other resolution parameter.
        #
        # Margin note: this reduced configuration returns 44.95277459934362
        # against the benchmark configuration's 44.95277460205567, a relative
        # difference of 6e-11. That is the residual axial-truncation error, not
        # a radial one, and it is six orders of magnitude inside the 0.1%
        # tolerance the final-answer rubric item allows.
        {
            "setup": """import numpy as np
n_axial = 4
n_samples = 41
""",
            "call": "run_tsv_thermal_stress_pipeline(n_axial=n_axial, n_radial=2, n_samples=n_samples)",
            "gold_call": "_oracle_run_tsv_thermal_stress_pipeline(n_axial=n_axial, n_radial=6, n_samples=n_samples)",
        },
        # --- Integration: final-answer configuration ---
        {
            "setup": """import numpy as np
""",
            "call": "run_tsv_thermal_stress_pipeline()",
            "gold_call": "_oracle_run_tsv_thermal_stress_pipeline()",
        },
        # --- Integration (boundary): the insulated face of the via ---
        {
            "setup": """import numpy as np
""",
            "call": "run_tsv_thermal_stress_pipeline(plane_fraction=0.0, n_axial=6, n_radial=3, n_samples=51)",
            "gold_call": "_oracle_run_tsv_thermal_stress_pipeline(plane_fraction=0.0, n_axial=6, n_radial=3, n_samples=51)",
        },
        # --- Integration (edge): a tungsten via with a benzocyclobutene liner ---
        {
            "setup": """import numpy as np
radii = (2.5e-6, 2.8e-6, 10.0e-6)
conductivities = (173.0, 0.29, 130.0)
densities = (19250.0, 1050.0, 2329.0)
specific_heats = (132.0, 1200.0, 700.0)
expansions = (4.5e-6, 42.0e-6, 2.6e-6)
youngs_moduli = (411.0e9, 2.9e9, 170.0e9)
poisson = (0.28, 0.34, 0.28)
""",
            "call": "run_tsv_thermal_stress_pipeline(radii, conductivities, densities, specific_heats, expansions, youngs_moduli, poisson, height=50.0e-6, time=1.0e-5, n_axial=6, n_radial=3, n_samples=51)",
            "gold_call": "_oracle_run_tsv_thermal_stress_pipeline(radii, conductivities, densities, specific_heats, expansions, youngs_moduli, poisson, height=50.0e-6, time=1.0e-5, n_axial=6, n_radial=3, n_samples=51)",
        },
        # --- Invalid: property tuple with the wrong number of layers ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_tsv_thermal_stress_pipeline(conductivities=(400.0, 1.4))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_tsv_thermal_stress_pipeline(conductivities=(400.0, 1.4))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: examined plane outside the via ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_tsv_thermal_stress_pipeline(plane_fraction=1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_tsv_thermal_stress_pipeline(plane_fraction=1.5)
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
