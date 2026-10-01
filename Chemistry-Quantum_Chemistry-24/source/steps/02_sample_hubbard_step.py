"""
Advance one Hubbard occupation using one supplied inverse-CDF draw.

A stochastic propagation step selects between retention of the current occupation and one of the ordered hopping channels. Apply the source single-step convention to the supplied channel data, preserving its ordering and fermionic sign. The input occupation and channel arrays must not be mutated.

Returns
-------
Return a tuple containing the resulting binary occupation array and its signed scalar amplitude.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def sample_hubbard_step(
    occupation: np.ndarray,
    moves: np.ndarray,
    coefficients: np.ndarray,
    normalization: float,
    interaction: float,
    delta_beta: float,
    uniform: float,
) -> tuple[np.ndarray, float]:
    """Advance one occupation with one supplied inverse-CDF draw.

    Apply the source single-step convention to the ordered channel data and
    return the resulting binary occupation and signed amplitude. Do not mutate
    the supplied occupation or channel arrays.

    moves is an integer array of shape (n_moves, 3) whose rows are
    (spin index 0 or 1, source site, destination site), and coefficients is
    the aligned float array of shape (n_moves,); both are in the channel scan
    order defined by the task.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_sample_hubbard_step(
    occupation: np.ndarray,
    moves: np.ndarray,
    coefficients: np.ndarray,
    normalization: float,
    interaction: float,
    delta_beta: float,
    uniform: float,
) -> tuple[np.ndarray, float]:
    import numpy as np

    raw_state = np.asarray(occupation, dtype=float)
    if raw_state.ndim != 1 or len(raw_state) == 0 or len(raw_state) % 2:
        raise ValueError("occupation must contain two equal spin blocks")
    if np.any((raw_state != 0) & (raw_state != 1)):
        raise ValueError("occupation must be binary")
    state = raw_state.astype(int)
    n_sites = len(state) // 2
    move_array = np.asarray(moves, dtype=int)
    weights = np.asarray(coefficients, dtype=float)
    if move_array.ndim != 2 or move_array.shape[1:] != (3,):
        raise ValueError("moves must have shape (n_moves, 3)")
    if (
        weights.shape != (len(move_array),)
        or np.any(weights < 0)
        or np.any(~np.isfinite(weights))
    ):
        raise ValueError("invalid coefficients")
    if not np.isfinite(normalization) or not np.isclose(
        normalization, 1.0 + np.sum(weights), rtol=1e-12, atol=1e-14
    ):
        raise ValueError("normalization must equal 1 + sum(coefficients)")
    if interaction < 0 or delta_beta <= 0 or not 0 <= uniform < 1:
        raise ValueError("invalid scalar input")

    double_occupancy = int(np.dot(state[:n_sites], state[n_sites:]))
    diagonal = float(np.exp(-interaction * double_occupancy * delta_beta))
    scaled_uniform = float(uniform * normalization)
    if scaled_uniform < 1.0:
        return state.copy(), float(normalization * diagonal)

    index = int(
        np.searchsorted(
            1.0 + np.cumsum(weights), scaled_uniform, side="right"
        )
    )
    if index >= len(move_array):
        index = len(move_array) - 1
    spin, source, destination = map(int, move_array[index])
    if (
        spin not in (0, 1)
        or min(source, destination) < 0
        or max(source, destination) >= n_sites
    ):
        raise ValueError("invalid hop")
    source_orbital = spin * n_sites + source
    destination_orbital = spin * n_sites + destination
    if state[source_orbital] != 1 or state[destination_orbital] != 0:
        raise ValueError("hop is not allowed by the occupation")
    low, high = sorted((source_orbital, destination_orbital))
    fermion_sign = (
        -1.0 if int(np.sum(state[low + 1 : high])) % 2 else 1.0
    )
    result = state.copy()
    result[source_orbital] = 0
    result[destination_orbital] = 1
    amplitude = -float(normalization) * fermion_sign * diagonal
    return result, float(amplitude)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\ndef pack(x):\n s,a=x; return np.concatenate((s.astype(float),np.array([a])))\ns=np.array([1,0,1,0]); m=np.array([[0,0,1],[1,0,1]]); c=np.array([.1,.2]); l=1.3",
            "call": "pack(sample_hubbard_step(s,m,c,l,2.,.05,.2))",
            "gold_call": "pack(_oracle_sample_hubbard_step(s,m,c,l,2.,.05,.2))",
        },
        {
            "setup": "import numpy as np\ndef pack(x):\n s,a=x; return np.concatenate((s.astype(float),np.array([a])))\ns=np.array([1,0,1,0]); m=np.array([[0,0,1],[1,0,1]]); c=np.array([.1,.2]); l=1.3",
            "call": "pack(sample_hubbard_step(s,m,c,l,2.,.05,1/l))",
            "gold_call": "pack(_oracle_sample_hubbard_step(s,m,c,l,2.,.05,1/l))",
        },
        {
            "setup": "import numpy as np\ndef pack(x):\n s,a=x; return np.concatenate((s.astype(float),np.array([a])))\ns=np.array([1,1,0,0,0,0,0,0]); m=np.array([[0,0,3]]); c=np.array([.25]); l=1.25",
            "call": "pack(sample_hubbard_step(s,m,c,l,0.,.1,.9))",
            "gold_call": "pack(_oracle_sample_hubbard_step(s,m,c,l,0.,.1,.9))",
        },
    ]
