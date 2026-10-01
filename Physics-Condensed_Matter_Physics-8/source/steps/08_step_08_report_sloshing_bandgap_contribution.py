"""
This is the final step and it runs the whole chain from the top-level configuration, twice, once at the graded liquid depth and once at the reference depth, on the same lattice and with the same liquid. Step one assembles the bending stiffness and the section inertia of the two-phase moderately thick plate cell, with the transverse shear energy integrated at the element centre. Step two assembles the Dirichlet form of the liquid displacement potential over the layer, which carries no material constant at all. Step three assembles the two interface couplings and the gravitational restoring operator of the free surface, all three from the same consistent surface integral of the bilinear shape functions. None of those depends on the wave vector, so the plate and the interface operators are built once and the liquid operator once per depth. Step four then imposes the Bloch phase condition separately on the plate, on the liquid and on the free surface, and reduces every operator by the conjugated congruence at one wave vector. Step five eliminates the liquid potential from those already-reduced operators and returns the added mass, the coupling mass and the surface mass. Step six solves the dry, the added-mass and the hydro-elastic cells from the same operators and splits the hydro-elastic branches into the free-surface family and the structural family. Steps four to six repeat at every wave vector of the locus, filling one branch table per description, and step seven reads the signed gap width and the positions of its edges off each table.

The locus is the zone-boundary segment of the square lattice, running from the edge centre, whose components are pi over the lattice constant along the first in-plane axis and zero along the second, to the corner, whose components are pi over the lattice constant along both, divided into equal steps with both endpoints retained. Every wave vector on it has an in-plane magnitude of at least pi over the lattice constant, which keeps the assignment of a branch to a family determinate and the branch numbering meaningful. Determinate is not the same as separated in frequency: at the graded depth the free-surface family also lies wholly below the structural family, while at the reference depth the two overlap, so the split is made by elevation participation and by the count of elevation freedoms and never by a frequency threshold, which would misclassify at the deeper layer. Approaching the zone centre the lowest branch of each family falls towards zero and the two hybridise, so that the assignment of a branch to a family, and with it the numbering of the structural branches, stops being determinate there.

Only the depth changes between the two runs. The dry cell is therefore unaffected by it and must return the same width in both, which is a free consistency check on the chain and is enforced rather than merely reported. The graded number is the difference between the hydro-elastic and the added-mass widths at the graded depth. The same difference at the reference depth is what establishes the regime: at ten plate thicknesses the two descriptions of the liquid have converged and the difference has collapsed, while at half a plate thickness it is of the order of the whole dry gap.

Returns
-------
dict holding the native ints n_wave_vector, n_surface_branch and edges_on_endpoints; the floats corner_surface_low, corner_surface_high and corner_structural_low in hertz at the graded depth; the floats dry_gap, graded_hydro_gap, graded_added_mass_gap, reference_hydro_gap, reference_added_mass_gap and reference_contribution, all signed gap widths in hertz; and the float sloshing_contribution, the graded answer in hertz.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def report_sloshing_bandgap_contribution(
    graded_depth: float,
    reference_depth: float,
    lower_branch: int,
    n_step: int,
    n_elem: int,
    n_layer: int,
    cell_side: float,
    inclusion_side: float,
    plate_thickness: float,
    shear_factor: float,
    matrix_constants: np.ndarray,
    inclusion_constants: np.ndarray,
    fluid_density: float,
    gravity: float,
) -> dict:
    """Run the chain at both depths and report the contribution the free surface makes to the gap.

    Parameters
    ----------
    graded_depth : float
        Depth of the liquid layer at which the answer is reported, in metres.
    reference_depth : float
        Deeper layer at which the same difference is evaluated, in metres.
    lower_branch : int
        One-based index of the lower of the two structural branches bounding the gap.
    n_step : int
        Steps into which the zone-boundary segment is divided.
    n_elem : int
        Elements per side of the square cell in plane.
    n_layer : int
        Slices through the depth of the liquid.
    cell_side : float
        Lattice constant of the square cell in metres.
    inclusion_side : float
        Side of the centred square inclusion in metres.
    plate_thickness : float
        Uniform plate thickness in metres.
    shear_factor : float
        Transverse shear correction factor.
    matrix_constants : np.ndarray
        Young modulus, Poisson ratio and density of the matrix phase, shape (3,).
    inclusion_constants : np.ndarray
        The same three constants for the inclusion phase, shape (3,).
    fluid_density : float
        Liquid density in kilogram per cubic metre.
    gravity : float
        Gravitational acceleration in metre per second squared.

    Returns
    -------
    dict
        Under the keys n_wave_vector, n_surface_branch, edges_on_endpoints, corner_surface_low, corner_surface_high, corner_structural_low, dry_gap, graded_hydro_gap, graded_added_mass_gap, reference_hydro_gap, reference_added_mass_gap, reference_contribution and sloshing_contribution.

    Raises
    ------
    ValueError
        When either depth is not finite and above zero, when reference_depth does not exceed graded_depth, when lower_branch or n_step is not an integer of one or more, when any configuration value rejected by an earlier step is passed through, when the cell returns fewer structural branches than the pair needs, or when the dry cell does not return the same gap width at the two depths.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _positive_depth_constant(value, label):
    """Return a length as a float once it is known to be finite and above zero."""
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError("%s wants a finite value above zero" % label)
    return number


def _counted_report_integer(value, label, floor):
    """Return a count as a native int once it is known to be an integer at or above a floor."""
    if not isinstance(value, (int, np.integer)) or int(value) < floor:
        raise ValueError("%s wants an integer of %d or more" % (label, floor))
    return int(value)


def _oracle_report_sloshing_bandgap_contribution(
    graded_depth: float,
    reference_depth: float,
    lower_branch: int,
    n_step: int,
    n_elem: int,
    n_layer: int,
    cell_side: float,
    inclusion_side: float,
    plate_thickness: float,
    shear_factor: float,
    matrix_constants: np.ndarray,
    inclusion_constants: np.ndarray,
    fluid_density: float,
    gravity: float,
) -> dict:
    """Reference implementation."""
    graded_depth = _positive_depth_constant(graded_depth, "graded_depth")
    reference_depth = _positive_depth_constant(reference_depth, "reference_depth")
    if reference_depth <= graded_depth:
        raise ValueError("reference_depth must sit above graded_depth")
    lower_branch = _counted_report_integer(lower_branch, "lower_branch", 1)
    n_step = _counted_report_integer(n_step, "n_step", 1)

    plate = _oracle_assemble_plate_bending_cell(  # noqa: F821
        n_elem, cell_side, inclusion_side, plate_thickness, shear_factor,
        matrix_constants, inclusion_constants)
    faces = _oracle_assemble_interface_operators(  # noqa: F821
        n_elem, cell_side, n_layer, fluid_density, gravity)
    edge = np.pi / float(cell_side)
    start = np.array([edge, 0.0])
    finish = np.array([edge, edge])
    models = ("dry", "added_mass", "hydro_elastic")
    wanted = lower_branch + 1
    outcome = {}
    for label, depth in (("graded", graded_depth), ("reference", reference_depth)):
        liquid = _oracle_assemble_fluid_potential_stiffness(  # noqa: F821
            n_elem, cell_side, depth, n_layer)
        tables = {name: [] for name in models}
        corner = None
        for step in range(n_step + 1):
            vector = start + (finish - start) * (step / n_step)
            reduced = _oracle_reduce_cell_by_bloch_phase(  # noqa: F821
                plate["stiffness"], plate["mass"], liquid["stiffness"], faces["fsi_coupling"],
                faces["surface_coupling"], faces["surface_stiffness"],
                n_elem, n_layer, cell_side, vector)
            masses = _oracle_condense_potential_to_mass_operators(  # noqa: F821
                reduced["fluid_stiffness"], reduced["fsi_coupling"],
                reduced["surface_coupling"], fluid_density)
            for name in models:
                branches = _oracle_solve_and_index_branches(  # noqa: F821
                    reduced["plate_stiffness"], reduced["plate_mass"], masses["added_mass"],
                    masses["coupling_mass"], masses["surface_mass"],
                    reduced["surface_stiffness"], name, 1.0)
                structural = branches["structural_frequencies"]
                if structural.size < wanted:
                    raise ValueError("the cell returns fewer structural branches than the pair needs")
                tables[name].append(structural)
                if name == "hydro_elastic" and step == n_step:
                    corner = branches
        edges = {}
        for name in models:
            edges[name] = _oracle_locate_gap_edges(np.array(tables[name]), lower_branch)  # noqa: F821
        outcome[label] = (edges, corner)

    graded_edges, graded_corner = outcome["graded"]
    reference_edges = outcome["reference"][0]
    dry_gap = graded_edges["dry"]["signed_width"]
    if abs(dry_gap - reference_edges["dry"]["signed_width"]) > 1.0e-9 * max(1.0, abs(dry_gap)):
        raise ValueError("the dry cell must not depend on the depth of the liquid")
    graded_hydro = graded_edges["hydro_elastic"]["signed_width"]
    graded_added = graded_edges["added_mass"]["signed_width"]
    reference_hydro = reference_edges["hydro_elastic"]["signed_width"]
    reference_added = reference_edges["added_mass"]["signed_width"]
    return {
        "n_wave_vector": n_step + 1,
        "n_surface_branch": int(graded_corner["n_surface_branch"]),
        "edges_on_endpoints": int(graded_edges["hydro_elastic"]["edges_on_endpoints"]),
        "corner_surface_low": float(graded_corner["surface_frequencies"][0]),
        "corner_surface_high": float(graded_corner["surface_frequencies"][-1]),
        "corner_structural_low": float(graded_corner["structural_frequencies"][0]),
        "dry_gap": float(dry_gap),
        "graded_hydro_gap": float(graded_hydro),
        "graded_added_mass_gap": float(graded_added),
        "reference_hydro_gap": float(reference_hydro),
        "reference_added_mass_gap": float(reference_added),
        "reference_contribution": float(reference_hydro - reference_added),
        "sloshing_contribution": float(graded_hydro - graded_added),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [{'setup': 'import numpy as np\n'
           'MATRIX = np.array([6.0e6, 0.47, 1290.0])\n'
           'TUNGSTEN = np.array([411.0e9, 0.28, 19300.0])\n'
           'SMALL = (5, 4, 4, 3, 0.2, 0.1, 0.02, 5.0/6.0, MATRIX, TUNGSTEN, 1000.0, 9.8)\n'
           '# a reduced cell exercising the whole chain: the free-surface family has one branch per\n'
           '# elevation freedom, the two families stay apart at the corner, both gap edges fall on\n'
           '# the endpoints of the locus, and the contribution collapses between the shallow layer\n'
           '# and ten plate thicknesses\n'
           'def digest(out):\n'
           '    return (out["n_surface_branch"], out["n_wave_vector"], out["edges_on_endpoints"],\n'
           '            round(out["dry_gap"], 6), round(out["graded_hydro_gap"], 6),\n'
           '            round(out["graded_added_mass_gap"], 6), round(out["reference_contribution"], 6),\n'
           '            round(out["sloshing_contribution"], 6),\n'
           '            int(out["corner_structural_low"] > 2.0 * out["corner_surface_high"]),\n'
           '            int(abs(out["sloshing_contribution"]) > 100.0 * '
           'abs(out["reference_contribution"])))\n'
           '\n'
           '\n'
           '# Each candidate/oracle invocation receives equivalent independent inputs.\n'
           '# Copy the whole arguments object to retain aliases within one invocation.\n'
           'def _scicode_fresh_inputs(function):\n'
           '    def invoke(*args, **kwargs):\n'
           '        from copy import deepcopy\n'
           '        call_args, call_kwargs = deepcopy((args, kwargs))\n'
           '        return function(*call_args, **call_kwargs)\n'
           '    return invoke\n',
  'call': 'digest(_scicode_fresh_inputs(report_sloshing_bandgap_contribution)(0.01, 0.2, *SMALL))',
  'gold_call': 'digest(_scicode_fresh_inputs(_oracle_report_sloshing_bandgap_contribution)(0.01, 0.2, '
               '*SMALL))'},
 {'setup': 'import numpy as np\n'
           'MATRIX = np.array([6.0e6, 0.47, 1290.0])\n'
           'STEEL = np.array([2.0e11, 0.3, 7800.0])\n'
           '# a different contrast on the coarsest cell that hosts an inclusion, and the lowest\n'
           '# branch pair rather than the graded one\n'
           'SMALL = (1, 3, 4, 2, 0.2, 0.1, 0.02, 5.0/6.0, MATRIX, STEEL, 1000.0, 9.8)\n'
           'def digest(out):\n'
           '    return (out["n_surface_branch"], round(out["dry_gap"], 6),\n'
           '            round(out["graded_hydro_gap"], 6), round(out["graded_added_mass_gap"], 6),\n'
           '            round(out["sloshing_contribution"], 6),\n'
           '            # five decimals, not six: on this small, strongly contrasted cell the pencil is\n'
           '            # ill-conditioned enough that a relative perturbation of four machine epsilon in '
           'the\n'
           '            # assembled operators moves this eigenvalue by about one part in ten million, and '
           'at\n'
           '            # six decimals the value sits only 1.3e-7 relatively above its rounding boundary, '
           'so\n'
           '            # an equivalent arithmetic ordering flips the last digit. At five decimals the '
           'margin\n'
           '            # is an order of magnitude larger than the amplified noise.\n'
           '            round(out["corner_surface_low"], 5),\n'
           '            int(out["dry_gap"] < 0.0))\n'
           '\n'
           '\n'
           '# Each candidate/oracle invocation receives equivalent independent inputs.\n'
           '# Copy the whole arguments object to retain aliases within one invocation.\n'
           'def _scicode_fresh_inputs(function):\n'
           '    def invoke(*args, **kwargs):\n'
           '        from copy import deepcopy\n'
           '        call_args, call_kwargs = deepcopy((args, kwargs))\n'
           '        return function(*call_args, **call_kwargs)\n'
           '    return invoke\n',
  'call': 'digest(_scicode_fresh_inputs(report_sloshing_bandgap_contribution)(0.02, 0.5, *SMALL))',
  'gold_call': 'digest(_scicode_fresh_inputs(_oracle_report_sloshing_bandgap_contribution)(0.02, 0.5, '
               '*SMALL))'},
 {'setup': 'import numpy as np\n'
           'MATRIX = np.array([6.0e6, 0.47, 1290.0])\n'
           'TUNGSTEN = np.array([411.0e9, 0.28, 19300.0])\n'
           '# doubling the liquid density leaves the dry cell alone and deepens the disagreement\n'
           '# between the two descriptions of the liquid\n'
           'def digest(pair):\n'
           '    light, heavy = pair\n'
           '    return (round(light["dry_gap"], 6), round(heavy["dry_gap"], 6),\n'
           '            int(abs(light["dry_gap"] - heavy["dry_gap"]) < 1.0e-9),\n'
           '            round(light["sloshing_contribution"], 6),\n'
           '            round(heavy["sloshing_contribution"], 6),\n'
           '            int(abs(heavy["graded_added_mass_gap"] - heavy["dry_gap"])\n'
           '                > abs(light["graded_added_mass_gap"] - light["dry_gap"])))\n'
           '\n'
           '\n'
           '# Each candidate/oracle invocation receives equivalent independent inputs.\n'
           '# Copy the whole arguments object to retain aliases within one invocation.\n'
           'def _scicode_fresh_inputs(function):\n'
           '    def invoke(*args, **kwargs):\n'
           '        from copy import deepcopy\n'
           '        call_args, call_kwargs = deepcopy((args, kwargs))\n'
           '        return function(*call_args, **call_kwargs)\n'
           '    return invoke\n',
  'call': 'digest((_scicode_fresh_inputs(report_sloshing_bandgap_contribution)(0.01, 0.2, 5, 3, 4, 3, 0.2, '
          '0.1, 0.02, 5.0 / 6.0, MATRIX, TUNGSTEN, 1000.0, 9.8), '
          '_scicode_fresh_inputs(report_sloshing_bandgap_contribution)(0.01, 0.2, 5, 3, 4, 3, 0.2, 0.1, '
          '0.02, 5.0 / 6.0, MATRIX, TUNGSTEN, 2000.0, 9.8)))',
  'gold_call': 'digest((_scicode_fresh_inputs(_oracle_report_sloshing_bandgap_contribution)(0.01, 0.2, 5, 3, '
               '4, 3, 0.2, 0.1, 0.02, 5.0 / 6.0, MATRIX, TUNGSTEN, 1000.0, 9.8), '
               '_scicode_fresh_inputs(_oracle_report_sloshing_bandgap_contribution)(0.01, 0.2, 5, 3, 4, 3, '
               '0.2, 0.1, 0.02, 5.0 / 6.0, MATRIX, TUNGSTEN, 2000.0, 9.8)))'},
 {'setup': 'import numpy as np\n'
           'MATRIX = np.array([6.0e6, 0.47, 1290.0])\n'
           'TUNGSTEN = np.array([411.0e9, 0.28, 19300.0])\n'
           'SMALL = (5, 3, 4, 2, 0.2, 0.1, 0.02, 5.0/6.0, MATRIX, TUNGSTEN, 1000.0, 9.8)\n'
           'def sentinel(fn):\n'
           '    codes = []\n'
           '    trials = [(0.0, 0.2) + SMALL, (0.2, 0.01) + SMALL, (0.01, 0.01) + SMALL,\n'
           '              (np.nan, 0.2) + SMALL,\n'
           '              (0.01, 0.2, 0, 3, 4, 2, 0.2, 0.1, 0.02, 5.0/6.0, MATRIX, TUNGSTEN, 1000.0, 9.8),\n'
           '              (0.01, 0.2, 5, 0, 4, 2, 0.2, 0.1, 0.02, 5.0/6.0, MATRIX, TUNGSTEN, 1000.0, 9.8),\n'
           '              (0.01, 0.2, 5000, 3, 4, 2, 0.2, 0.1, 0.02, 5.0/6.0, MATRIX, TUNGSTEN, 1000.0, '
           '9.8),\n'
           '              (0.01, 0.2, 5, 3, 4, 2, 0.2, 0.1, 0.02, 5.0/6.0, MATRIX, TUNGSTEN, 0.0, 9.8)]\n'
           '    for args in trials:\n'
           '        try:\n'
           '            fn(*args)\n'
           '            codes.append(0)\n'
           '        except ValueError:\n'
           '            codes.append(1)\n'
           '        except Exception:\n'
           '            codes.append(2)\n'
           '    return tuple(codes)\n'
           '\n'
           '\n'
           '# Each candidate/oracle invocation receives equivalent independent inputs.\n'
           '# Copy the whole arguments object to retain aliases within one invocation.\n'
           'def _scicode_fresh_inputs(function):\n'
           '    def invoke(*args, **kwargs):\n'
           '        from copy import deepcopy\n'
           '        call_args, call_kwargs = deepcopy((args, kwargs))\n'
           '        return function(*call_args, **call_kwargs)\n'
           '    return invoke\n',
  'call': 'sentinel(_scicode_fresh_inputs(report_sloshing_bandgap_contribution))',
  'gold_call': 'sentinel(_scicode_fresh_inputs(_oracle_report_sloshing_bandgap_contribution))'}]
