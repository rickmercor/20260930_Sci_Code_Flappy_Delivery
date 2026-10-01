"""
Evaluate the full decoder across the ordered beam widths and summarize correction quality and search work.

Beam width controls the speed-accuracy tradeoff by changing how many high-score partial assignments survive. The profile retains both the selected correction statistics and execution diagnostics needed by the benchmark.

Returns
-------
real `np.ndarray` of shape `(B,5)` with rows `(selected_weight,hamming_weight,rounds_used,unique_results_found,children_expanded)`
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def beam_search_profile(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", beam_widths: "np.ndarray", max_rounds: int, initial_iters: int, iters_per_round: int, num_results: int) -> "np.ndarray":
    '''Return one five-component decoder profile per beam width.

    Parameters
    ----------
    H, p, syndrome : np.ndarray
        Detector model, error probabilities, and binary syndrome.
    beam_widths : np.ndarray
        Positive integer-like beam widths in the requested order.
    max_rounds, initial_iters, iters_per_round, num_results : int
        Decoder controls passed unchanged to beam_search_decode.

    Returns
    -------
    profile : np.ndarray
        Float array of shape (B,5). Each row is
        [selected_weight, correction_hamming_weight, rounds_used,
        unique_results_found, children_expanded].
    '''
    return profile

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_beam_search_profile(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", beam_widths: "np.ndarray", max_rounds: int, initial_iters: int, iters_per_round: int, num_results: int) -> "np.ndarray":
    H = np.asarray(H, dtype=int)
    N = H.shape[1]
    rows = []
    for width in np.asarray(beam_widths, dtype=int):
        result = _oracle_beam_search_decode(
            H, p, syndrome, int(max_rounds), int(width), int(initial_iters),
            int(iters_per_round), int(num_results)
        )
        correction = result[4:4 + N]
        rows.append([
            result[0], float(np.sum(correction)), result[1], result[2], result[3]
        ])
    return np.asarray(rows, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    '''Return list of test case specifications.'''
    common = '''import numpy as np\nH=np.array([[0,1,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0],[1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0],[0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0],[0,1,0,0,0,1,0,0,1,0,0,1,0,1,0,1,0,0,0,0],[0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,0,1],[1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,1,0,1,0],[0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1],[0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0],[0,1,0,0,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,0],[0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1]],dtype=int)\np=np.array([0.099321675569,0.055717560406,0.074255307877,0.070190994539,0.151229358583,0.145537983086,0.035252957104,0.085929973968,0.086129312934,0.159482638903,0.153475804526,0.075766077846,0.144920993860,0.135244812128,0.140972421803,0.131736174287,0.114588720979,0.098089955076,0.082017411514,0.155506630793])\nwidths=np.array([1,2,4,8])\n'''
    return [
        {
            "setup": common + '''s=np.array([1,0,1,0,1,1,0,1,0,1,1],dtype=int)\n''',
            "call": "beam_search_profile(H.copy(),p.copy(),s.copy(),widths.copy(),6,3,4,3)",
            "gold_call": "_oracle_beam_search_profile(H.copy(),p.copy(),s.copy(),widths.copy(),6,3,4,3)",
            "tol": 2e-11,
        },
        {
            "setup": common + '''s=np.array([1,0,0,0,0,0,1,0,1,1,1],dtype=int)\n''',
            "call": "beam_search_profile(H.copy(),p.copy(),s.copy(),widths.copy(),6,3,4,3)",
            "gold_call": "_oracle_beam_search_profile(H.copy(),p.copy(),s.copy(),widths.copy(),6,3,4,3)",
            "tol": 2e-11,
        },
        {
            "setup": common + '''s=np.array([1,1,1,0,1,0,1,1,1,0,0],dtype=int)\n''',
            "call": "beam_search_profile(H.copy(),p.copy(),s.copy(),widths.copy(),6,3,4,3)",
            "gold_call": "_oracle_beam_search_profile(H.copy(),p.copy(),s.copy(),widths.copy(),6,3,4,3)",
            "tol": 2e-11,
        },
    ]
