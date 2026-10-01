"""
Chain the sub-problem functions 01-11 end to end on the cyclic water tetramer and return the ratio of its quantum to its classical dimensionless rigidification penalty.

This step runs the whole measurement end to end. It (i) minimizes the isolated monomer to fix the reference internal geometry with sub-problem 01, (ii) rebuilds every monomer of the input cluster at that geometry in its own position and orientation with sub-problem 04, giving the zeroth-order configuration that the conventional frozen treatment stops at, (iii) builds the mass-scaled block-diagonal curvature there with sub-problem 05, differentiating the analytic gradient of sub-problem 03, (iv) converts it into the stiff-subspace pseudoinverse with sub-problem 06, (v) takes the prescribed number of preconditioned Newton steps with sub-problem 07, following that same gradient while holding the pseudoinverse fixed, (vi) rebuilds the curvature at the relaxed point and extracts both sets of stiff frequencies with sub-problems 05, 08 and 09, (vii) evaluates the potential energy at both configurations with sub-problem 02, (viii) charges each configuration its own harmonic free energy in both the classical and the quantum regime with sub-problem 10, and (ix) reduces the four free energies to the reported dimensionless ratio.

Two configurations therefore appear, and each is priced twice, once in the classical regime and once in the quantum one, giving four free energies and two penalties. The frozen treatment is not a separate algorithm requiring separate code: it is this same pipeline stopped after zero relaxation steps, which is why the free energies can be assembled from the same components without any risk that a difference in convention contaminates their difference. Normalizing by the total number of stiff degrees of freedom and by the thermal energy turns each difference into an intensive quantity, the free-energy error per eliminated mode in units of the thermal energy.

The returned ratio answers a question the rigidification theory does not address. The theory establishes that relaxing the fast coordinates and charging their harmonic free energy reproduces all-atom thermodynamics almost exactly, and separately that freezing them does not; it says nothing about how that error depends on the statistics assigned to the eliminated modes. The two regimes weight the frequency shift quite differently. Classically the vibrational term is a sum of logarithms of frequency ratios, so a mode that softens and a mode that stiffens partly cancel, and the term is small beside the potential-energy change. In the quantum regime the leading vibrational term is the zero-point energy, which is linear in the frequency itself rather than logarithmic, so the large red shift of the hydrogen-bond-donating stretch is not offset and the vibrational contribution becomes comparable to the energetic one.

Because the ratio is built from two penalties that share the same potential-energy change and the same frequencies, it is exactly one when the relaxation leaves the frequencies unchanged. Any departure from unity is therefore attributable entirely to the curvature shift, which makes the returned number a direct measure of how much the neglected intramolecular relaxation matters once nuclear quantum effects are admitted; the all-atom potential used here was itself parameterized for path-integral simulation, so that is the regime in which it is meant to be applied.

The all-atom reconstruction of sub-problem 11 is exercised alongside the reduction, since it is what makes all-atom observables available from a coarse-grained trajectory, but it does not enter the reported ratio: the ratio is a property of the free-energy surface and must not depend on a random draw.

Returns
-------
float: the ratio of the quantum to the classical dimensionless rigidification penalty of the cluster, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_shr_pipeline(coords: np.ndarray = None, temperature: float = 250.0,
                     n_iterations: int = 2, n_stiff: int = 3,
                     mass_o: float = 15.9994, mass_h: float = 1.00794,
                     hessian_step: float = 1.0e-5) -> float:
    """Run the full rigidification-penalty measurement on a water cluster.

    Parameters
    ----------
    coords : np.ndarray or None
        Array of shape (3 * n_monomers, 3) in angstrom holding the all-atom
        cluster, ordered O, H, H within each monomer. ``None`` selects the
        benchmark cyclic tetramer of the problem statement.
    temperature : float
        Absolute temperature in kelvin (temperature > 0).
    n_iterations : int
        Number of subspace Newton-Raphson iterations used by the relaxed
        treatment (n_iterations >= 1; zero would make the two treatments
        identical and the reported ratio undefined).
    n_stiff : int
        Number of stiff intramolecular degrees of freedom per monomer, at
        least 1 and at most 9.
    mass_o : float
        Oxygen mass in unified atomic mass units (mass_o > 0).
    mass_h : float
        Hydrogen mass in unified atomic mass units (mass_h > 0).
    hessian_step : float
        Central-difference displacement in angstrom used for the block
        Hessians (hessian_step > 0).

    Returns
    -------
    amplification : float
        Dimensionless ratio of the quantum rigidification penalty to the
        classical one, each penalty being the free-energy excess of the frozen
        treatment over the relaxed one per stiff degree of freedom in units of
        the thermal energy, as a native Python float.

    Raises
    ------
    ValueError
        If ``coords`` has the wrong shape, atom count, or non-finite entries;
        if the temperature, masses, or Hessian step are non-finite or
        non-positive; if ``n_iterations`` is negative or ``n_stiff`` is not an
        integer from 1 through 9; if reconstruction violates its contract; or
        if zero iterations makes the classical penalty and ratio undefined.
    """
    return amplification  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_shr_pipeline(coords: np.ndarray = None, temperature: float = 250.0,
                             n_iterations: int = 2, n_stiff: int = 3,
                             mass_o: float = 15.9994, mass_h: float = 1.00794,
                             hessian_step: float = 1.0e-5) -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import glob
    import importlib.util
    import os
    import sys

    import numpy as np

    # -- Resolve the oracle functions of sub-problems 01-11. Preference order:
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

    monomer_reference_geometry = _resolve_step(
        "_oracle_compute_monomer_reference_geometry", "*compute_monomer_reference_geometry*.py")
    potential_energy = _resolve_step(
        "_oracle_compute_potential_energy", "*compute_potential_energy*.py")
    potential_gradient = _resolve_step(
        "_oracle_compute_potential_gradient", "*compute_potential_gradient*.py")
    reference_configuration = _resolve_step(
        "_oracle_build_reference_configuration", "*build_reference_configuration*.py")
    block_hessian = _resolve_step(
        "_oracle_compute_block_hessian", "*compute_block_hessian*.py")
    subspace_pseudoinverse = _resolve_step(
        "_oracle_build_subspace_pseudoinverse", "*build_subspace_pseudoinverse*.py")
    relax_newton_raphson = _resolve_step(
        "_oracle_relax_subspace_newton_raphson", "*relax_subspace_newton_raphson*.py")
    stiff_mode_frequencies = _resolve_step(
        "_oracle_compute_stiff_mode_frequencies", "*compute_stiff_mode_frequencies*.py")
    stiff_mode_vectors = _resolve_step(
        "_oracle_compute_stiff_mode_vectors", "*compute_stiff_mode_vectors*.py")
    cg_free_energy = _resolve_step(
        "_oracle_compute_cg_free_energy", "*compute_cg_free_energy*.py")
    backmap_configuration = _resolve_step(
        "_oracle_backmap_thermal_configuration", "*backmap_thermal_configuration*.py")

    # -- The benchmark cyclic tetramer of the problem statement.
    if coords is None:
        coords = np.array([
            [1.965757, 0.000000, 0.000000], [1.276254, 0.689503, 0.097837],
            [2.425271, 0.000868, 0.837181], [0.000000, 1.965757, 0.000000],
            [-0.680176, 1.285581, -0.077118], [0.004437, 2.439238, -0.852304],
            [-1.965757, 0.000000, 0.000000], [-1.270755, -0.695002, 0.118515],
            [-2.428412, 0.000015, 0.841160], [0.000000, -1.965757, 0.000000],
            [0.684659, -1.281098, -0.058165], [0.015394, -2.427969, -0.869683]],
            dtype=float)
    coords = np.asarray(coords, dtype=float)

    # -- Validate the orchestrator inputs.
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if coords.shape[0] < 3 or coords.shape[0] % 3 != 0:
        raise ValueError("coords must hold a whole number of three-atom monomers")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite entries")
    for name, value in (("temperature", temperature), ("mass_o", mass_o),
                        ("mass_h", mass_h), ("hessian_step", hessian_step)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(n_iterations, (int, np.integer)) and not isinstance(n_iterations, bool)
            and int(n_iterations) >= 0):
        raise ValueError("n_iterations must be an integer >= 0")
    if not (isinstance(n_stiff, (int, np.integer)) and not isinstance(n_stiff, bool)
            and 1 <= int(n_stiff) <= 9):
        raise ValueError("n_stiff must be an integer between 1 and 9")

    temperature = float(temperature)
    n_iterations, n_stiff = int(n_iterations), int(n_stiff)
    hessian_step = float(hessian_step)
    n_monomers = coords.shape[0] // 3
    atom_masses = np.tile(np.array([float(mass_o), float(mass_h), float(mass_h)]), n_monomers)

    # -- Sub-problems 01 and 04: the zeroth-order reference configuration.
    reference_geometry = monomer_reference_geometry(116.09, 2.287, 0.9419, 87.85, 107.4)
    frozen_coords = reference_configuration(coords, reference_geometry)

    # -- Sub-problem 03: the one analytic gradient that both the curvature and
    #    the relaxation are built on. Passing it explicitly is what keeps the
    #    Newton step and the Hessian it is preconditioned by on one surface.
    def gradient_fn(positions):
        return potential_gradient(positions)

    # -- Sub-problems 05 and 06: curvature at the frozen point and the operator
    #    that projects a gradient onto the stiff subspace.
    frozen_blocks = block_hessian(frozen_coords, atom_masses, gradient_fn, hessian_step)
    pseudoinverse = subspace_pseudoinverse(frozen_blocks, n_stiff)

    # -- Sub-problem 07: relaxation with the pseudoinverse held fixed.
    relaxed_coords = relax_newton_raphson(frozen_coords, pseudoinverse, atom_masses,
                                          gradient_fn, n_iterations)

    # -- Sub-problems 05, 08 and 09: the two sets of stiff frequencies. The block
    #    Hessian is genuinely rebuilt at the relaxed point.
    relaxed_blocks = block_hessian(relaxed_coords, atom_masses, gradient_fn, hessian_step)
    frozen_frequencies = stiff_mode_frequencies(frozen_blocks, n_stiff)
    relaxed_frequencies = stiff_mode_frequencies(relaxed_blocks, n_stiff)
    relaxed_modes = stiff_mode_vectors(relaxed_blocks, n_stiff)

    # -- Sub-problems 02 and 10: four coarse-grained free energies, the two
    #    treatments evaluated in each of the two vibrational regimes.
    frozen_energy = potential_energy(frozen_coords)
    relaxed_energy = potential_energy(relaxed_coords)
    frozen_classical = cg_free_energy(frozen_energy, frozen_frequencies, temperature, False)
    relaxed_classical = cg_free_energy(relaxed_energy, relaxed_frequencies, temperature, False)
    frozen_quantum = cg_free_energy(frozen_energy, frozen_frequencies, temperature, True)
    relaxed_quantum = cg_free_energy(relaxed_energy, relaxed_frequencies, temperature, True)

    # -- Sub-problem 11: the all-atom reconstruction. Its return value cannot
    #    enter a free-energy ratio, which must not depend on a random draw, so
    #    it is consumed as a check instead. Only the stiff directions are
    #    displaced, so mass-weighting the displacement of each monomer must
    #    return a vector lying wholly in the span of that monomer's stiff mode
    #    vectors; a reconstruction that forgets the inverse-square-root mass
    #    factor, or that displaces along the rigid-body directions too, leaves a
    #    residual outside that span.
    reconstructed = backmap_configuration(relaxed_coords, relaxed_frequencies, relaxed_modes,
                                          atom_masses, temperature, 0)
    reconstructed = np.asarray(reconstructed, dtype=float)
    if reconstructed.shape != relaxed_coords.shape or not np.all(np.isfinite(reconstructed)):
        raise ValueError("the all-atom reconstruction must have the shape of the cluster")
    for mono in range(n_monomers):
        rows = np.arange(3 * mono, 3 * mono + 3)
        root = np.sqrt(np.repeat(atom_masses[rows], 3))
        scaled = root * (reconstructed[rows] - relaxed_coords[rows]).reshape(9)
        basis = relaxed_modes[mono]
        residual = scaled - basis @ (basis.T @ scaled)
        if np.linalg.norm(residual) > 1.0e-8 * max(np.linalg.norm(scaled), 1.0):
            raise ValueError("the reconstruction displaced outside the stiff subspace")
        # At a finite temperature the draw is non-degenerate, so a reconstruction
        # that returns the relaxed configuration untouched has not sampled at all.
        if np.linalg.norm(scaled) <= 0.0:
            raise ValueError("the reconstruction left the configuration unchanged")

    # -- The two intensive penalties and the reported amplification factor.
    boltzmann_kcal = 0.0019872042586408316  # kcal/(mol K), from CODATA 2018
    scale = n_monomers * n_stiff * boltzmann_kcal * temperature
    penalty_classical = (frozen_classical - relaxed_classical) / scale
    penalty_quantum = (frozen_quantum - relaxed_quantum) / scale
    if penalty_classical == 0.0:
        raise ValueError("the classical rigidification penalty vanishes; the ratio is undefined")

    return float(penalty_quantum / penalty_classical)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Integration: the benchmark configuration whose value is the
        #     task's final answer (normal scenario) ---
        {
            "setup": """import numpy as np
""",
            "call": "run_shr_pipeline()",
            "gold_call": "_oracle_run_shr_pipeline()",
        },
        # --- Integration (boundary): a single relaxation step, which is exact
        #     for a harmonic surface and isolates the anharmonic remainder ---
        {
            "setup": """import numpy as np
""",
            "call": "run_shr_pipeline(None, 250.0, 1)",
            "gold_call": "_oracle_run_shr_pipeline(None, 250.0, 1)",
        },
        # --- Invalid: zero relaxation steps make the two treatments identical,
        #     so the classical penalty vanishes and the ratio is undefined ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_shr_pipeline(None, 250.0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_shr_pipeline(None, 250.0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Integration: a water dimer at a different temperature ---
        {
            "setup": """import numpy as np
coords = np.array([
    [0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0],
    [2.85, 0.15, 0.10], [3.55, 0.62, 0.55], [2.70, -0.55, 0.60]])
""",
            "call": "run_shr_pipeline(coords, 320.0, 2)",
            "gold_call": "_oracle_run_shr_pipeline(coords, 320.0, 2)",
        },
        # --- Integration (edge): deuterated tetramer with a longer relaxation ---
        {
            "setup": """import numpy as np
""",
            "call": "run_shr_pipeline(None, 200.0, 4, 3, 15.9994, 2.0141)",
            "gold_call": "_oracle_run_shr_pipeline(None, 200.0, 4, 3, 15.9994, 2.0141)",
        },
        # --- Invalid: negative iteration count ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_shr_pipeline(None, 250.0, -1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_shr_pipeline(None, 250.0, -1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive temperature ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_shr_pipeline(None, 0.0, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_shr_pipeline(None, 0.0, 2)
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
