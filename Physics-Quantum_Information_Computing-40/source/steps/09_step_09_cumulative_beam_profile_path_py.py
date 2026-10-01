"""
Normalize the four beam-width profiles and sum the Euclidean distances between consecutive profiles.

The final orchestrator composes the full source-grounded decoder pipeline and measures how correction quality and bounded-search effort change as the retained beam broadens.

Returns
-------
native Python `float`, the cumulative normalized profile-path length
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cumulative_beam_profile_path(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", beam_widths: "np.ndarray", max_rounds: int, initial_iters: int, iters_per_round: int, num_results: int) -> float:
    '''Return the cumulative Euclidean path through normalized beam profiles.

    Parameters
    ----------
    H, p, syndrome : np.ndarray
        Detector model, probabilities, and binary syndrome.
    beam_widths : np.ndarray
        Ordered positive beam widths.
    max_rounds, initial_iters, iters_per_round, num_results : int
        Decoder parameters.

    Returns
    -------
    value : float
        For each width b and profile row (W,h,r,R,C), form
        g_b=(W/sum_j log((1-p_j)/p_j), h/N, r/max_rounds,
        R/num_results, C/(2*b*max_rounds)). Return the sum of Euclidean
        distances between consecutive g_b rows, without intermediate rounding.
    '''
    return value

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_cumulative_beam_profile_path(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", beam_widths: "np.ndarray", max_rounds: int, initial_iters: int, iters_per_round: int, num_results: int) -> float:
    H = np.asarray(H, dtype=int)
    p = np.asarray(p, dtype=float)
    widths = np.asarray(beam_widths, dtype=int)
    profile = _oracle_beam_search_profile(
        H, p, syndrome, widths, int(max_rounds), int(initial_iters),
        int(iters_per_round), int(num_results)
    )
    prior_sum = float(np.sum(np.log((1.0 - p) / p)))
    N = H.shape[1]
    normalized = profile.copy()
    normalized[:, 0] /= prior_sum
    normalized[:, 1] /= float(N)
    normalized[:, 2] /= float(max_rounds)
    normalized[:, 3] /= float(num_results)
    normalized[:, 4] /= (2.0 * widths.astype(float) * float(max_rounds))
    return float(np.sum(np.linalg.norm(np.diff(normalized, axis=0), axis=1)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    '''Return list of test case specifications.'''
    common = '''import numpy as np\nH=np.array([[0,1,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0],[1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0],[0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0],[0,1,0,0,0,1,0,0,1,0,0,1,0,1,0,1,0,0,0,0],[0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,0,1],[1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,1,0,1,0],[0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1],[0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0],[0,1,0,0,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,0],[0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1]],dtype=int)\np=np.array([0.099321675569,0.055717560406,0.074255307877,0.070190994539,0.151229358583,0.145537983086,0.035252957104,0.085929973968,0.086129312934,0.159482638903,0.153475804526,0.075766077846,0.144920993860,0.135244812128,0.140972421803,0.131736174287,0.114588720979,0.098089955076,0.082017411514,0.155506630793])\nwidths=np.array([1,2,4,8])\n'''
    return [
        {
            "setup": common + '''s=np.array([1,0,1,0,1,1,0,1,0,1,1],dtype=int)\n''',
            "call": "cumulative_beam_profile_path(H.copy(),p.copy(),s.copy(),widths.copy(),6,3,4,3)",
            "gold_call": "_oracle_cumulative_beam_profile_path(H.copy(),p.copy(),s.copy(),widths.copy(),6,3,4,3)",
            "tol": 5e-12,
        },
        {
            "setup": common + '''s=np.array([1,0,0,0,0,0,1,0,1,1,1],dtype=int)\n''',
            "call": "cumulative_beam_profile_path(H.copy(),p.copy(),s.copy(),widths.copy(),6,3,4,3)",
            "gold_call": "_oracle_cumulative_beam_profile_path(H.copy(),p.copy(),s.copy(),widths.copy(),6,3,4,3)",
            "tol": 5e-12,
        },
        {
            "setup": common + '''s=np.array([1,1,1,0,1,0,1,1,1,0,0],dtype=int)\n''',
            "call": "cumulative_beam_profile_path(H.copy(),p.copy(),s.copy(),widths.copy(),6,3,4,3)",
            "gold_call": "_oracle_cumulative_beam_profile_path(H.copy(),p.copy(),s.copy(),widths.copy(),6,3,4,3)",
            "tol": 5e-12,
        },
    ]
