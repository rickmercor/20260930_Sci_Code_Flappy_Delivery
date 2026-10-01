"""
Search the program space by simulated annealing to minimize the shifted training loss.

A program of fixed length is initialized at random from the primitive library and then perturbed for a fixed number of iterations. At each iteration a random number of positions (up to three) is overwritten with random primitives, the shifted loss of the candidate program is evaluated with freshly optimized per-element shifts, and the candidate is accepted with the Metropolis criterion at an artificial temperature that decreases linearly to zero over the run. The best program seen during the run is returned together with its loss.

Returns
-------
np.ndarray, shape (1 + n_functions,) float array with the best loss first and the primitive indices of the program that reached it
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def anneal_program(molecules: list, params: dict, n_ops: int, n_functions: int, n_iter: int, t_init: float, seed: int) -> "np.ndarray":
    """Return the best loss and program found by simulated annealing.

    Parameters
    ----------
    molecules : list
        Training molecules; each entry is a dict with keys ``positions``
        (shape (N_i, n_i) array, one retained configuration per row),
        ``types`` (shape (n_i,) element types 0 or 1), ``ref_energies``
        (shape (N_i,) reference total energies of those rows) and
        ``ref_shift`` (float independent-atom reference energy D_i). The
        atom-type counts n_{it} are derived from ``types`` for the T element
        types listed in ``params['onsite_energy']``.
    params : dict
        Model constants.
    n_ops : int
        Number of usable primitives: candidate entries are drawn from
        0 .. n_ops - 1 of the library (n_ops <= 7).
    n_functions : int
        Program length N_f >= 1.
    n_iter : int
        Number of annealing iterations.
    t_init : float
        Initial artificial temperature; iteration k (0-based) uses
        T_k = t_init * (1 - k / n_iter).
    seed : int
        Seed of the NumPy generator ``np.random.default_rng(seed)`` that
        drives every random draw, in this order: the initial program
        ``rng.integers(0, n_ops, size=n_functions)``; then per iteration
        ``r = rng.integers(1, 4)`` capped at n_functions, the positions
        ``rng.choice(n_functions, size=r, replace=False)``, the replacement
        primitives ``rng.integers(0, n_ops, size=r)``, and, only when the
        candidate loss exceeds the current loss, one ``rng.random()`` that
        accepts the candidate if it is below exp(-(loss_cand - loss_cur) / T_k).
        A candidate whose loss does not exceed the current loss is accepted
        without a draw.

    Returns
    -------
    result : np.ndarray
        Shape (1 + n_functions,) float array [best_loss, op_1, ..., op_Nf]
        where best_loss is the smallest shifted RMSE loss encountered (initial
        program included) and op_k are the primitive indices of the first
        program that reached it.

    Raises
    ------
    ValueError
        If n_functions < 1, n_ops is outside 1..7, or the training molecules
        cannot determine every per-element shift.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_anneal_program(molecules: list, params: dict, n_ops: int, n_functions: int, n_iter: int, t_init: float, seed: int) -> "np.ndarray":
    n_ops, n_functions, n_iter = int(n_ops), int(n_functions), int(n_iter)
    if n_functions < 1:
        raise ValueError("n_functions must be at least 1")
    if n_ops < 1 or n_ops > 7:
        raise ValueError("n_ops must be in 1..7")
    rng = np.random.default_rng(int(seed))
    program = rng.integers(0, n_ops, size=n_functions)
    current = float(_training_loss(program, molecules, params)[0])
    best_loss, best_program = current, program.copy()
    for k in range(n_iter):
        temperature = float(t_init) * (1.0 - k / n_iter)
        r = min(int(rng.integers(1, 4)), n_functions)
        positions = rng.choice(n_functions, size=r, replace=False)
        candidate = program.copy()
        candidate[positions] = rng.integers(0, n_ops, size=r)
        cand_loss = float(_training_loss(candidate, molecules, params)[0])
        if cand_loss <= current or rng.random() < np.exp(-(cand_loss - current) / temperature):
            program, current = candidate, cand_loss
            if current < best_loss:
                best_loss, best_program = current, program.copy()
    return np.concatenate([[best_loss], best_program.astype(float)])

def _training_loss(program: "np.ndarray", molecules: list, params: dict) -> "np.ndarray":
    """Shifted loss and fitted shifts of one program over the training molecules."""
    n_types = len(params["onsite_energy"])
    pred, ref, shifts, counts = [], [], [], []
    for mol in molecules:
        types = np.asarray(mol["types"], dtype=int)
        pred.append(np.array([_oracle_program_energy(program, pos, types, params) for pos in mol["positions"]]))
        ref.append(np.asarray(mol["ref_energies"], dtype=float))
        shifts.append(float(mol["ref_shift"]))
        counts.append(np.bincount(types, minlength=n_types))
    return _oracle_fit_shifted_loss(pred, ref, np.array(shifts), np.array(counts))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test cases for this step."""
    params = ("params = {'onsite_energy': [-1.0, -1.6], 'onsite_repulsion': [0.9, 1.3], "
              "'kappa': 1.75, 'sigma': 1.1, 'core_repulsion_strength': 6.0, 'core_repulsion_range': 0.45}\n")
    aa = ("aa = {'types': np.array([0, 0]), 'ref_shift': -2.0,\n"
          "      'positions': np.array([[0.0, 1.2752], [0.0, 1.3379549419516785], [0.0, 1.177980461971574], [0.0, 1.3975319026646873]]),\n"
          "      'ref_energies': np.array([-2.5257404487356467, -2.521028718273958, -2.512580511716446, -2.5087707080009367])}\n")
    bb = ("bb = {'types': np.array([1, 1]), 'ref_shift': -3.2,\n"
          "      'positions': np.array([[0.0, 1.0672], [0.0, 1.0023227848352385], [0.0, 1.15603678553464], [0.0, 1.1130751187875014]]),\n"
          "      'ref_energies': np.array([-4.194077877073372, -4.185566902368195, -4.180257893557647, -4.1902412315112985])}\n")
    abab = ("abab = {'types': np.array([0, 1, 0, 1]), 'ref_shift': -5.2,\n"
            "        'positions': np.array([[0.0, 1.0399, 2.7489, 3.7827], [0.0, 0.9987486251936643, 2.7677931895855523, 3.806028217413664],\n"
            "                               [0.0, 1.0554250258066398, 2.895013980597493, 3.9536680044190904], [0.0, 1.0080868874060833, 2.590532316303852, 3.6792521024511045],\n"
            "                               [0.0, 1.1241143627534043, 2.833283548553013, 3.839867528518033]]),\n"
            "        'ref_energies': np.array([-8.196836841389477, -8.190802538418067, -8.186326831487744, -8.177920195731193, -8.179024111740897])}\n")
    common = "import numpy as np\nimport copy\n" + params + aa + bb + abab
    return [
        # full library, four primitives, the benchmark annealing schedule
        {
            "setup": common + "molecules = [aa, bb, abab]\n",
            "call": "anneal_program(*copy.deepcopy((molecules, params, 7, 4, 200, 0.05, 7)))[0]",
            "gold_call": "_oracle_anneal_program(molecules, params, 7, 4, 200, 0.05, 7)[0]",
        },
        # two-primitive programs, perturbation count capped at the program length
        {
            "setup": common + "molecules = [aa, bb, abab]\n",
            "call": "anneal_program(*copy.deepcopy((molecules, params, 7, 2, 80, 0.05, 3)))[0]",
            "gold_call": "_oracle_anneal_program(molecules, params, 7, 2, 80, 0.05, 3)[0]",
        },
        # restricted library without the screen and half primitives, two training molecules
        {
            "setup": common + "molecules = [aa, abab]\n",
            "call": "anneal_program(*copy.deepcopy((molecules, params, 5, 3, 120, 0.1, 1)))[0]",
            "gold_call": "_oracle_anneal_program(molecules, params, 5, 3, 120, 0.1, 1)[0]",
        },
        # a single-primitive program with a high initial temperature
        {
            "setup": common + "molecules = [aa, bb, abab]\n",
            "call": "anneal_program(*copy.deepcopy((molecules, params, 7, 1, 30, 0.5, 11)))[0]",
            "gold_call": "_oracle_anneal_program(molecules, params, 7, 1, 30, 0.5, 11)[0]",
        },
        # one mixed molecule cannot separate the two element shifts
        {
            "setup": common + """molecules = [abab]
def run_model():
    try:
        anneal_program(molecules, params, 7, 4, 20, 0.05, 7)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_anneal_program(molecules, params, 7, 4, 20, 0.05, 7)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
