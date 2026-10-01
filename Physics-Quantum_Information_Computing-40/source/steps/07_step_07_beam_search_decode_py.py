"""
Execute the complete source search for one syndrome, including initialization, masked branching, descending-score exploration, pruning, unique-result collection, and weighted selection.

This is the main code wall. It composes the paper-specific cumulative reliability metric, parent-message warm starts, masked syndrome transformation, iteration-normalized path scoring, finite-width pruning, and multi-result stopping logic.

Returns
-------
real `np.ndarray` equal to `[weight, rounds_used, unique_results_found, children_expanded, correction_bits...]`
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def beam_search_decode(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", max_rounds: int, beam_width: int, initial_iters: int, iters_per_round: int, num_results: int) -> "np.ndarray":
    '''Decode one syndrome with bounded BP-guided branching.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix of shape (M,N).
    p : np.ndarray
        Error probabilities of shape (N,).
    syndrome : np.ndarray
        Binary syndrome of shape (M,).
    max_rounds : int
        Positive maximum number of masked branching rounds.
    beam_width : int
        Positive number of active paths retained after each round.
    initial_iters : int
        Positive initial standard-BP iteration cap.
    iters_per_round : int
        Positive masked-BP iteration cap per child.
    num_results : int
        Positive target number of unique valid corrections.

    Returns
    -------
    result : np.ndarray
        Float vector [weight, rounds_used, unique_results_found,
        children_expanded, correction_bits...]. Paths are explored in stable
        decreasing-score order; each path branches value 0 then 1. Stop as soon
        as num_results unique valid corrections have been collected and choose
        their minimum-weight member. If max_rounds is exhausted first, choose
        the minimum-weight member among all unique valid corrections found.
        Unless collecting a correction triggers this overall return, retain
        every evaluated child for the next round's score-based pruning,
        including successful children and children yielding duplicate
        corrections. A successful child may therefore be expanded again.

    Raises
    ------
    ValueError
        If no valid correction is found within the search budget.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_beam_search_decode(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", max_rounds: int, beam_width: int, initial_iters: int, iters_per_round: int, num_results: int) -> "np.ndarray":
    H = np.asarray(H, dtype=int)
    p = np.asarray(p, dtype=float)
    s = np.asarray(syndrome, dtype=int)
    N = H.shape[1]
    seed = _oracle_seed_beam_state(H, p, s, int(initial_iters))
    results = []
    seen = set()
    children_expanded = 0

    def _add_unique(bits):
        key = tuple(int(v) for v in np.asarray(bits, dtype=int))
        if key not in seen:
            seen.add(key)
            results.append(np.asarray(bits, dtype=int).copy())

    if int(round(seed[0])) == 1:
        _add_unique(seed[-N:])
        if int(num_results) == 1:
            selected = _oracle_minimum_weight_correction(np.vstack(results), p)
            return np.concatenate([selected[:1], [0.0, 1.0, 0.0], selected[1:]])

    active = seed.reshape(1, -1)
    for round_index in range(1, int(max_rounds) + 1):
        active = _oracle_prune_beam_states(active, active.shape[0])
        next_rows = []
        for path in active:
            children = _oracle_expand_beam_state(H, p, s, path, int(iters_per_round))
            for child in children:
                children_expanded += 1
                if int(round(child[0])) == 1:
                    _add_unique(child[-N:])
                    if len(results) >= int(num_results):
                        selected = _oracle_minimum_weight_correction(np.vstack(results), p)
                        return np.concatenate([
                            selected[:1],
                            [float(round_index), float(len(results)), float(children_expanded)],
                            selected[1:],
                        ])
                next_rows.append(child)
        if next_rows:
            active = _oracle_prune_beam_states(np.vstack(next_rows), int(beam_width))

    if not results:
        raise ValueError("no valid correction found within search budget")
    selected = _oracle_minimum_weight_correction(np.vstack(results), p)
    return np.concatenate([
        selected[:1],
        [float(max_rounds), float(len(results)), float(children_expanded)],
        selected[1:],
    ])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    '''Return list of test case specifications.'''
    common = '''import numpy as np\nH=np.array([[0,1,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0],[1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0],[0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0],[0,1,0,0,0,1,0,0,1,0,0,1,0,1,0,1,0,0,0,0],[0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,0,1],[1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,1,0,1,0],[0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1],[0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0],[0,1,0,0,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,0],[0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1]],dtype=int)\np=np.array([0.099321675569,0.055717560406,0.074255307877,0.070190994539,0.151229358583,0.145537983086,0.035252957104,0.085929973968,0.086129312934,0.159482638903,0.153475804526,0.075766077846,0.144920993860,0.135244812128,0.140972421803,0.131736174287,0.114588720979,0.098089955076,0.082017411514,0.155506630793])\n'''
    return [
        {
            "setup": common + '''s=np.array([1,0,1,0,1,1,0,1,0,1,1],dtype=int)\n''',
            "call": "beam_search_decode(H.copy(),p.copy(),s.copy(),6,4,3,4,3)",
            "gold_call": "_oracle_beam_search_decode(H.copy(),p.copy(),s.copy(),6,4,3,4,3)",
            "tol": 2e-11,
        },
        {
            "setup": common + '''s=np.array([1,0,0,0,0,0,1,0,1,1,1],dtype=int)\n''',
            "call": "beam_search_decode(H.copy(),p.copy(),s.copy(),6,8,3,4,3)",
            "gold_call": "_oracle_beam_search_decode(H.copy(),p.copy(),s.copy(),6,8,3,4,3)",
            "tol": 2e-11,
        },
        {
            "setup": common + '''s=np.zeros(H.shape[0],dtype=int)\n''',
            "call": "beam_search_decode(H.copy(),p.copy(),s.copy(),6,4,3,4,1)",
            "gold_call": "_oracle_beam_search_decode(H.copy(),p.copy(),s.copy(),6,4,3,4,1)",
            "tol": 2e-11,
        },
    ]
