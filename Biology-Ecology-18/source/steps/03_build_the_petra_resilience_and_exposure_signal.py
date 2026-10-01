"""
Build the PETRA resilience and exposure signal.

Forecast from the supplied pre-disturbance state with the selected PETRA setting. Return the four resilience measures followed by the signed six-species exposure obtained from the closest forecast state. Reject misaligned, non-finite, or out-of-contract inputs with ValueError.

Returns
-------
A float64 vector containing four resilience measures followed by six signed species exposures
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def petra_residual_signal(reference: "np.ndarray", disturbed: "np.ndarray", candidate: tuple[int, float, int, str, float], max_steps: int, indices: "np.ndarray", guild_map: "np.ndarray") -> "np.ndarray":
    """Build the PETRA resilience and exposure signal.

    Returns
    -------
    A float64 vector containing four resilience measures followed by six signed species exposures.

    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_petra_residual_signal(reference: "np.ndarray", disturbed: "np.ndarray", candidate: tuple[int, float, int, str, float], max_steps: int, indices: "np.ndarray", guild_map: "np.ndarray") -> "np.ndarray":
    r = np.asarray(reference, dtype=np.float64)
    obs = np.asarray(disturbed, dtype=np.float64)
    mapping = np.asarray(guild_map, dtype=np.float64)
    ids = np.asarray(indices, dtype=np.int64)
    if r.ndim != 3 or obs.ndim != 2 or mapping.ndim != 2 or mapping.shape[1] != obs.shape[1] or ids.shape != (3,) or not (0 <= ids[0] < ids[1] < ids[2] < len(obs)):
        raise ValueError("unaligned disturbance inputs")
    _, path, _, _ = _forecast(r, obs[ids[0]], candidate, int(max_steps))
    dref = _bc_rows(obs[ids], path).min(axis=1)
    direct = float(_bc_rows(obs[[ids[0]]], obs[[ids[1]]])[0, 0])
    profile = np.array([1.0 - direct, dref[1] - dref[0], dref[1] - dref[2], dref[2] - dref[0]], dtype=np.float64)
    closest = path[int(np.argmin(_bc_rows(obs[[ids[2]]], path)[0]))]
    denom = float((obs[ids[2]] + closest).sum())
    signed = (obs[ids[2]] - closest) / denom
    exposure = mapping @ signed
    return np.concatenate([profile, exposure]).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    return [

        {

            "setup": "import numpy as np\nreference=np.array([[[62.0, 26.0, 8.0, 3.0, 1.0], [56.0, 28.0, 10.0, 4.0, 2.0], [48.0, 31.0, 12.0, 6.0, 3.0], [38.0, 32.0, 17.0, 8.0, 5.0], [28.0, 30.0, 22.0, 12.0, 8.0], [21.0, 25.0, 25.0, 19.0, 10.0], [17.0, 20.0, 26.0, 23.0, 14.0], [13.0, 18.0, 23.0, 26.0, 20.0]], [[65.0, 24.0, 7.0, 3.0, 1.0], [57.0, 28.0, 9.0, 4.0, 2.0], [49.0, 31.0, 12.0, 5.0, 3.0], [39.0, 31.0, 17.0, 8.0, 5.0], [29.0, 30.0, 21.0, 12.0, 8.0], [21.0, 26.0, 25.0, 17.0, 11.0], [16.0, 21.0, 26.0, 22.0, 15.0], [12.0, 18.0, 24.0, 26.0, 20.0]], [[60.0, 28.0, 8.0, 3.0, 1.0], [53.0, 30.0, 11.0, 4.0, 2.0], [46.0, 32.0, 13.0, 6.0, 3.0], [37.0, 31.0, 18.0, 9.0, 5.0], [29.0, 28.0, 22.0, 13.0, 8.0], [22.0, 24.0, 24.0, 19.0, 11.0], [18.0, 20.0, 25.0, 23.0, 15.0], [14.0, 18.0, 23.0, 25.0, 20.0]], [[66.0, 23.0, 7.0, 3.0, 1.0], [58.0, 27.0, 9.0, 4.0, 2.0], [49.0, 30.0, 12.0, 6.0, 3.0], [39.0, 30.0, 17.0, 9.0, 5.0], [29.0, 29.0, 20.0, 14.0, 8.0], [21.0, 25.0, 24.0, 19.0, 11.0], [16.0, 21.0, 25.0, 22.0, 16.0], [11.0, 19.0, 24.0, 25.0, 21.0]], [[59.0, 27.0, 10.0, 3.0, 1.0], [53.0, 29.0, 12.0, 4.0, 2.0], [45.0, 32.0, 14.0, 6.0, 3.0], [37.0, 32.0, 17.0, 9.0, 5.0], [29.0, 29.0, 22.0, 12.0, 8.0], [23.0, 24.0, 24.0, 18.0, 11.0], [18.0, 20.0, 26.0, 21.0, 15.0], [15.0, 16.0, 24.0, 25.0, 20.0]], [[63.0, 27.0, 6.0, 3.0, 1.0], [56.0, 29.0, 9.0, 4.0, 2.0], [47.0, 32.0, 12.0, 6.0, 3.0], [38.0, 32.0, 17.0, 8.0, 5.0], [28.0, 30.0, 21.0, 13.0, 8.0], [21.0, 25.0, 25.0, 18.0, 11.0], [17.0, 20.0, 26.0, 22.0, 15.0], [13.0, 17.0, 24.0, 26.0, 20.0]]], dtype=float)\ncalibration_targets=np.array([[53.0, 30.0, 10.0, 5.0, 2.0], [36.0, 31.0, 18.0, 10.0, 5.0], [20.0, 24.0, 25.0, 19.0, 12.0]], dtype=float)\ndisturbed=np.array([[56.0, 28.0, 10.0, 4.0, 2.0], [46.0, 31.0, 14.0, 6.0, 3.0], [18.0, 17.0, 18.0, 25.0, 22.0], [20.0, 20.0, 20.0, 23.0, 17.0], [22.0, 22.0, 21.0, 20.0, 15.0], [20.0, 23.0, 23.0, 19.0, 15.0]], dtype=float)\ncandidates=[(4, 0.42, 3, 'linear', 1.0), (5, 0.42, 3, 'exponential', 2.0), (6, 0.42, 4, 'hyperbolic', 3.0), (7, 0.42, 4, 'spherical', 1.0), (8, 0.42, 5, 'power', 2.0)]\nmax_steps=4\nindices=np.array([1, 2, 4], dtype=int)\nguild_map=np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.25, -0.15, 0.35, -0.2, 0.4]], dtype=float)\ncandidate=candidates[4]",

            'call': 'petra_residual_signal(reference,disturbed,candidate,max_steps,indices,guild_map)',

            'gold_call': '_oracle_petra_residual_signal(reference,disturbed,candidate,max_steps,indices,guild_map)',

        },

        {

            "setup": "import numpy as np\nreference=np.array([[[62.0, 26.0, 8.0, 3.0, 1.0], [56.0, 28.0, 10.0, 4.0, 2.0], [48.0, 31.0, 12.0, 6.0, 3.0], [38.0, 32.0, 17.0, 8.0, 5.0], [28.0, 30.0, 22.0, 12.0, 8.0], [21.0, 25.0, 25.0, 19.0, 10.0], [17.0, 20.0, 26.0, 23.0, 14.0], [13.0, 18.0, 23.0, 26.0, 20.0]], [[65.0, 24.0, 7.0, 3.0, 1.0], [57.0, 28.0, 9.0, 4.0, 2.0], [49.0, 31.0, 12.0, 5.0, 3.0], [39.0, 31.0, 17.0, 8.0, 5.0], [29.0, 30.0, 21.0, 12.0, 8.0], [21.0, 26.0, 25.0, 17.0, 11.0], [16.0, 21.0, 26.0, 22.0, 15.0], [12.0, 18.0, 24.0, 26.0, 20.0]], [[60.0, 28.0, 8.0, 3.0, 1.0], [53.0, 30.0, 11.0, 4.0, 2.0], [46.0, 32.0, 13.0, 6.0, 3.0], [37.0, 31.0, 18.0, 9.0, 5.0], [29.0, 28.0, 22.0, 13.0, 8.0], [22.0, 24.0, 24.0, 19.0, 11.0], [18.0, 20.0, 25.0, 23.0, 15.0], [14.0, 18.0, 23.0, 25.0, 20.0]], [[66.0, 23.0, 7.0, 3.0, 1.0], [58.0, 27.0, 9.0, 4.0, 2.0], [49.0, 30.0, 12.0, 6.0, 3.0], [39.0, 30.0, 17.0, 9.0, 5.0], [29.0, 29.0, 20.0, 14.0, 8.0], [21.0, 25.0, 24.0, 19.0, 11.0], [16.0, 21.0, 25.0, 22.0, 16.0], [11.0, 19.0, 24.0, 25.0, 21.0]], [[59.0, 27.0, 10.0, 3.0, 1.0], [53.0, 29.0, 12.0, 4.0, 2.0], [45.0, 32.0, 14.0, 6.0, 3.0], [37.0, 32.0, 17.0, 9.0, 5.0], [29.0, 29.0, 22.0, 12.0, 8.0], [23.0, 24.0, 24.0, 18.0, 11.0], [18.0, 20.0, 26.0, 21.0, 15.0], [15.0, 16.0, 24.0, 25.0, 20.0]], [[63.0, 27.0, 6.0, 3.0, 1.0], [56.0, 29.0, 9.0, 4.0, 2.0], [47.0, 32.0, 12.0, 6.0, 3.0], [38.0, 32.0, 17.0, 8.0, 5.0], [28.0, 30.0, 21.0, 13.0, 8.0], [21.0, 25.0, 25.0, 18.0, 11.0], [17.0, 20.0, 26.0, 22.0, 15.0], [13.0, 17.0, 24.0, 26.0, 20.0]]], dtype=float)\ncalibration_targets=np.array([[53.0, 30.0, 10.0, 5.0, 2.0], [36.0, 31.0, 18.0, 10.0, 5.0], [20.0, 24.0, 25.0, 19.0, 12.0]], dtype=float)\ndisturbed=np.array([[56.0, 28.0, 10.0, 4.0, 2.0], [46.0, 31.0, 14.0, 6.0, 3.0], [18.0, 17.0, 18.0, 25.0, 22.0], [20.0, 20.0, 20.0, 23.0, 17.0], [22.0, 22.0, 21.0, 20.0, 15.0], [20.0, 23.0, 23.0, 19.0, 15.0]], dtype=float)\ncandidates=[(4, 0.42, 3, 'linear', 1.0), (5, 0.42, 3, 'exponential', 2.0), (6, 0.42, 4, 'hyperbolic', 3.0), (7, 0.42, 4, 'spherical', 1.0), (8, 0.42, 5, 'power', 2.0)]\nmax_steps=4\nindices=np.array([1, 2, 4], dtype=int)\nguild_map=np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.25, -0.15, 0.35, -0.2, 0.4]], dtype=float)\ncandidate=candidates[3]",

            'call': 'petra_residual_signal(reference,disturbed,candidate,max_steps,indices,guild_map)',

            'gold_call': '_oracle_petra_residual_signal(reference,disturbed,candidate,max_steps,indices,guild_map)',

        },

        {

            "setup": "import numpy as np\nreference=np.array([[[62.0, 26.0, 8.0, 3.0, 1.0], [56.0, 28.0, 10.0, 4.0, 2.0], [48.0, 31.0, 12.0, 6.0, 3.0], [38.0, 32.0, 17.0, 8.0, 5.0], [28.0, 30.0, 22.0, 12.0, 8.0], [21.0, 25.0, 25.0, 19.0, 10.0], [17.0, 20.0, 26.0, 23.0, 14.0], [13.0, 18.0, 23.0, 26.0, 20.0]], [[65.0, 24.0, 7.0, 3.0, 1.0], [57.0, 28.0, 9.0, 4.0, 2.0], [49.0, 31.0, 12.0, 5.0, 3.0], [39.0, 31.0, 17.0, 8.0, 5.0], [29.0, 30.0, 21.0, 12.0, 8.0], [21.0, 26.0, 25.0, 17.0, 11.0], [16.0, 21.0, 26.0, 22.0, 15.0], [12.0, 18.0, 24.0, 26.0, 20.0]], [[60.0, 28.0, 8.0, 3.0, 1.0], [53.0, 30.0, 11.0, 4.0, 2.0], [46.0, 32.0, 13.0, 6.0, 3.0], [37.0, 31.0, 18.0, 9.0, 5.0], [29.0, 28.0, 22.0, 13.0, 8.0], [22.0, 24.0, 24.0, 19.0, 11.0], [18.0, 20.0, 25.0, 23.0, 15.0], [14.0, 18.0, 23.0, 25.0, 20.0]], [[66.0, 23.0, 7.0, 3.0, 1.0], [58.0, 27.0, 9.0, 4.0, 2.0], [49.0, 30.0, 12.0, 6.0, 3.0], [39.0, 30.0, 17.0, 9.0, 5.0], [29.0, 29.0, 20.0, 14.0, 8.0], [21.0, 25.0, 24.0, 19.0, 11.0], [16.0, 21.0, 25.0, 22.0, 16.0], [11.0, 19.0, 24.0, 25.0, 21.0]], [[59.0, 27.0, 10.0, 3.0, 1.0], [53.0, 29.0, 12.0, 4.0, 2.0], [45.0, 32.0, 14.0, 6.0, 3.0], [37.0, 32.0, 17.0, 9.0, 5.0], [29.0, 29.0, 22.0, 12.0, 8.0], [23.0, 24.0, 24.0, 18.0, 11.0], [18.0, 20.0, 26.0, 21.0, 15.0], [15.0, 16.0, 24.0, 25.0, 20.0]], [[63.0, 27.0, 6.0, 3.0, 1.0], [56.0, 29.0, 9.0, 4.0, 2.0], [47.0, 32.0, 12.0, 6.0, 3.0], [38.0, 32.0, 17.0, 8.0, 5.0], [28.0, 30.0, 21.0, 13.0, 8.0], [21.0, 25.0, 25.0, 18.0, 11.0], [17.0, 20.0, 26.0, 22.0, 15.0], [13.0, 17.0, 24.0, 26.0, 20.0]]], dtype=float)\ncalibration_targets=np.array([[53.0, 30.0, 10.0, 5.0, 2.0], [36.0, 31.0, 18.0, 10.0, 5.0], [20.0, 24.0, 25.0, 19.0, 12.0]], dtype=float)\ndisturbed=np.array([[56.0, 28.0, 10.0, 4.0, 2.0], [46.0, 31.0, 14.0, 6.0, 3.0], [18.0, 17.0, 18.0, 25.0, 22.0], [20.0, 20.0, 20.0, 23.0, 17.0], [22.0, 22.0, 21.0, 20.0, 15.0], [20.0, 23.0, 23.0, 19.0, 15.0]], dtype=float)\ncandidates=[(4, 0.42, 3, 'linear', 1.0), (5, 0.42, 3, 'exponential', 2.0), (6, 0.42, 4, 'hyperbolic', 3.0), (7, 0.42, 4, 'spherical', 1.0), (8, 0.42, 5, 'power', 2.0)]\nmax_steps=4\nindices=np.array([1, 2, 4], dtype=int)\nguild_map=np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.25, -0.15, 0.35, -0.2, 0.4]], dtype=float)\ncandidate=candidates[4]\nindices=np.array([2,1,4])\ndef candidate_code():\n    try:\n        petra_residual_signal(reference,disturbed,candidate,max_steps,indices,guild_map)\n    except ValueError:\n        return 1.0\n    return 0.0\ndef oracle_code():\n    try:\n        _oracle_petra_residual_signal(reference,disturbed,candidate,max_steps,indices,guild_map)\n    except ValueError:\n        return 1.0\n    return 0.0",

            'call': 'candidate_code()',

            'gold_call': 'oracle_code()',

            'tol': 0.0,

        },

    ]
