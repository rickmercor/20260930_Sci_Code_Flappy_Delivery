"""
Complete workflow: optimize geometries, sample, train a workspace program and score it on a hold-out molecule.

For every training molecule the equilibrium bond lengths and normal modes are computed from the model, the listed normal-mode samples are turned into geometries, their reference energies are converged, and geometries whose reference energy lies more than the threshold above the equilibrium reference energy are dropped. The retained geometries train a program by simulated annealing. The per-element shift contributions fitted for that program on the training set are then carried over to a hold-out molecule of the same elements, whose listed geometries are predicted with the program in a single diagonalization each, and the root-mean-square deviation between shifted predictions and shifted references is reported.

Returns
-------
float, the hold-out root-mean-square difference between shifted program predictions and shifted reference energies, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pips_holdout_rmse(train_specs: list, holdout_spec: dict, params: dict, e_max: float, n_functions: int, n_iter: int, t_init: float, seed: int) -> float:
    """Return the hold-out RMSE of the annealed program with transferred shifts.

    Parameters
    ----------
    train_specs : list
        Training molecules; each entry is a dict with keys ``types`` (shape
        (n,) element types 0 or 1), ``start_bonds`` (shape (n - 1,) starting
        bond lengths for the geometry optimization) and ``samples`` (list of
        (eps, direction) pairs for the normal-mode sampler, directions given
        in the frequency-scaled coordinates of the computed modes, ordered
        by increasing frequency).
    holdout_spec : dict
        Same keys for the hold-out molecule; all of its samples are used.
    params : dict
        Model constants, including ``onsite_energy`` whose entries are the
        isolated-atom energies per element type.
    e_max : float
        A training geometry is retained only if its reference energy minus
        the reference energy of the equilibrium geometry is at most e_max.
    n_functions : int
        Program length for the search over the full seven-primitive library.
    n_iter : int
        Number of annealing iterations.
    t_init : float
        Initial annealing temperature.
    seed : int
        Seed of the annealing generator.

    Returns
    -------
    rmse : float
        Root of the mean squared difference, over the hold-out geometries,
        between the program prediction minus the hold-out shift and the
        reference SCF energy minus the hold-out independent-atom energy. The
        hold-out shift is assembled from the per-element contributions fitted
        on the retained training geometries for the best annealed program.
        Native Python float.

    Raises
    ------
    ValueError
        If no training geometry is retained, if the training molecules cannot
        determine every per-element contribution, or if any molecule has an
        odd atom count.
    """
    return rmse

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_pips_holdout_rmse(train_specs: list, holdout_spec: dict, params: dict, e_max: float, n_functions: int, n_iter: int, t_init: float, seed: int) -> float:
    alpha = np.asarray(params["onsite_energy"], dtype=float)
    n_types = alpha.shape[0]
    molecules = []
    for spec in train_specs:
        types = np.asarray(spec["types"], dtype=int)
        positions, energies, e_eq = _sampled_geometries(spec, params)
        keep = energies - e_eq <= e_max
        if not np.any(keep):
            continue
        molecules.append({"types": types, "positions": positions[keep], "ref_energies": energies[keep],
                          "ref_shift": float(np.sum(alpha[types]))})
    if not molecules:
        raise ValueError("no training geometry lies within e_max of equilibrium")
    result = _oracle_anneal_program(molecules, params, 7, n_functions, n_iter, t_init, seed)
    program = result[1:].astype(int)
    shifts = _training_loss(program, molecules, params)[1:]
    types = np.asarray(holdout_spec["types"], dtype=int)
    positions, energies, _ = _sampled_geometries(holdout_spec, params)
    predicted = np.array([_oracle_program_energy(program, pos, types, params) for pos in positions])
    d_bar = float(np.bincount(types, minlength=n_types) @ shifts)
    d_ref = float(np.sum(alpha[types]))
    residual = (predicted - d_bar) - (energies - d_ref)
    return float(np.sqrt(np.mean(residual ** 2)))

def _sampled_geometries(spec: dict, params: dict) -> tuple:
    """Sampled positions and their SCF energies, plus the equilibrium SCF energy."""
    types = np.asarray(spec["types"], dtype=int)
    bonds = _oracle_equilibrium_bonds(types, spec["start_bonds"], params)
    modes = _oracle_normal_modes(types, bonds, params)
    omegas, vectors = modes[0], modes[1:]
    positions, energies = [], []
    for eps, direction in spec["samples"]:
        pos = _oracle_displace_along_modes(bonds, vectors, omegas, eps, direction)
        positions.append(pos)
        energies.append(_oracle_scf_reference_energy(pos, types, params))
    e_eq = _oracle_scf_reference_energy(_bonds_to_positions(bonds), types, params)
    return np.array(positions), np.array(energies), e_eq

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test cases for this step."""
    params = ("params = {'onsite_energy': [-1.0, -1.6], 'onsite_repulsion': [0.9, 1.3], "
              "'kappa': 1.75, 'sigma': 1.1, 'core_repulsion_strength': 6.0, 'core_repulsion_range': 0.45}\n")
    specs = """aa = {'types': np.array([0, 0]), 'start_bonds': np.array([1.2]),
      'samples': [(0.0, np.array([1.0])), (0.005, np.array([1.0])), (0.012, np.array([-1.0])), (0.019, np.array([1.0])), (0.019, np.array([-1.0]))]}
bb = {'types': np.array([1, 1]), 'start_bonds': np.array([1.2]),
      'samples': [(0.0, np.array([1.0])), (0.008, np.array([-1.0])), (0.015, np.array([1.0])), (0.019, np.array([-1.0])), (0.004, np.array([1.0]))]}
abab = {'types': np.array([0, 1, 0, 1]), 'start_bonds': np.array([1.2, 1.2, 1.2]),
        'samples': [(0.0, np.array([1.0, 0.0, 0.0])), (0.006, np.array([1.0, 1.0, 1.0])), (0.012, np.array([2.0, -1.0, 0.5])),
                    (0.018, np.array([-1.0, 0.0, 1.0])), (0.019, np.array([0.0, -3.0, -4.0]))]}
ab = {'types': np.array([0, 1]), 'start_bonds': np.array([1.2]),
      'samples': [(0.0, np.array([1.0])), (0.006, np.array([1.0])), (0.012, np.array([-1.0])), (0.019, np.array([1.0])), (0.019, np.array([-1.0]))]}
"""
    common = "import numpy as np\nimport copy\n" + params + specs
    return [
        # the benchmark run
        {
            "setup": common,
            "call": "pips_holdout_rmse(*copy.deepcopy(([aa, bb, abab], ab, params, 0.02, 4, 200, 0.05, 7)))",
            "gold_call": "_oracle_pips_holdout_rmse([aa, bb, abab], ab, params, 0.02, 4, 200, 0.05, 7)",
        },
        # looser threshold keeps every training geometry
        {
            "setup": common,
            "call": "pips_holdout_rmse(*copy.deepcopy(([aa, bb, abab], ab, params, 0.03, 4, 200, 0.05, 7)))",
            "gold_call": "_oracle_pips_holdout_rmse([aa, bb, abab], ab, params, 0.03, 4, 200, 0.05, 7)",
        },
        # diatomics only in training, the chain held out, shorter programs
        {
            "setup": common,
            "call": "pips_holdout_rmse(*copy.deepcopy(([aa, bb, ab], abab, params, 0.02, 3, 150, 0.05, 11)))",
            "gold_call": "_oracle_pips_holdout_rmse([aa, bb, ab], abab, params, 0.02, 3, 150, 0.05, 11)",
        },
        # tight threshold keeps only the near-equilibrium geometries
        {
            "setup": common,
            "call": "pips_holdout_rmse(*copy.deepcopy(([aa, bb, abab], ab, params, 0.007, 4, 100, 0.05, 7)))",
            "gold_call": "_oracle_pips_holdout_rmse([aa, bb, abab], ab, params, 0.007, 4, 100, 0.05, 7)",
        },
        # one mixed molecule cannot pin down both contributions
        {
            "setup": common + """def run_model():
    try:
        pips_holdout_rmse([abab], ab, params, 0.02, 4, 20, 0.05, 7)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_pips_holdout_rmse([abab], ab, params, 0.02, 4, 20, 0.05, 7)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
