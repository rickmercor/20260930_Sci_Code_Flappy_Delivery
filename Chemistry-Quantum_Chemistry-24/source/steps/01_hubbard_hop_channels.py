"""
Construct ordered channel data for one fixed-spin occupation.

A stochastic Hubbard propagation step remains in the fixed-spin occupation sector by considering allowed occupied-to-empty hopping channels. The channel coefficients and normalization follow the source protocol. Preserve the supplied edge order and do not mutate the input arrays.

Returns
-------
Return a tuple (moves, coefficients, normalization): moves is an integer array of shape (n_moves, 3) whose rows are (spin index 0 or 1, source site, destination site) in channel scan order, coefficients is the aligned float array of shape (n_moves,), and normalization is the scalar float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def hubbard_hop_channels(
    occupation: np.ndarray,
    edges: np.ndarray,
    hopping: float,
    interaction: float,
    delta_beta: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Construct ordered channel data for one fixed-spin occupation.

    Preserve the supplied edge order and return the directed move table, its
    nonnegative source-protocol coefficients, and the scalar channel
    normalization. The input arrays must not be mutated.

    Returns
    -------
    moves : np.ndarray
        Integer array of shape (n_moves, 3). Each row is
        (spin index 0 or 1, source site, destination site), listed in the
        channel scan order defined by the task.
    coefficients : np.ndarray
        Float array of shape (n_moves,), aligned row by row with moves.
    normalization : float
        The scalar channel normalization. When the occupation admits no
        allowed hop, return arrays of shape (0, 3) and (0,) with
        normalization 1.0.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_hubbard_hop_channels(
    occupation: np.ndarray,
    edges: np.ndarray,
    hopping: float,
    interaction: float,
    delta_beta: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    import numpy as np

    raw_state = np.asarray(occupation, dtype=float)
    if raw_state.ndim != 1 or len(raw_state) == 0 or len(raw_state) % 2:
        raise ValueError("occupation must contain two equal spin blocks")
    if np.any(~np.isfinite(raw_state)) or np.any(
        (raw_state != 0) & (raw_state != 1)
    ):
        raise ValueError("occupation must be binary")
    state = raw_state.astype(int)
    n_sites = len(state) // 2

    raw_edges = np.asarray(edges, dtype=float)
    if raw_edges.ndim != 2 or raw_edges.shape[1:] != (2,):
        raise ValueError("edges must have shape (n_edges, 2)")
    if np.any(~np.isfinite(raw_edges)) or np.any(
        raw_edges != np.floor(raw_edges)
    ):
        raise ValueError("edge indices must be integers")
    edge_array = raw_edges.astype(int)
    seen = set()
    for i, j in edge_array:
        if i == j or min(i, j) < 0 or max(i, j) >= n_sites:
            raise ValueError("invalid edge")
        key = (min(int(i), int(j)), max(int(i), int(j)))
        if key in seen:
            raise ValueError("edges must be unique")
        seen.add(key)

    scalars = np.asarray([hopping, interaction, delta_beta], dtype=float)
    if (
        np.any(~np.isfinite(scalars))
        or hopping < 0
        or interaction < 0
        or delta_beta <= 0
    ):
        raise ValueError("invalid Hubbard parameters")

    moves = []
    coefficients = []
    for i, j in edge_array:
        for spin in range(2):
            for source, destination in ((int(i), int(j)), (int(j), int(i))):
                source_orbital = spin * n_sites + source
                destination_orbital = spin * n_sites + destination
                if (
                    state[source_orbital] == 1
                    and state[destination_orbital] == 0
                ):
                    opposite = 1 - spin
                    exponent = (
                        state[opposite * n_sites + source]
                        - state[opposite * n_sites + destination]
                    ) * interaction * delta_beta / 2.0
                    moves.append((spin, source, destination))
                    coefficients.append(
                        hopping * delta_beta * np.exp(exponent)
                    )

    move_array = np.asarray(moves, dtype=int).reshape(-1, 3)
    coefficient_array = np.asarray(coefficients, dtype=float)
    normalization = float(1.0 + np.sum(coefficient_array))
    return move_array, coefficient_array, normalization

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\ndef pack(x):\n m,c,l=x; return np.concatenate((m.astype(float).ravel(),c.ravel(),np.array([l])))\ns=np.array([1,1,0,0,1,0,0,0]); e=np.array([[0,1],[1,2],[2,3],[3,0]])",
            "call": "pack(hubbard_hop_channels(s,e,1.,2.,.05))",
            "gold_call": "pack(_oracle_hubbard_hop_channels(s,e,1.,2.,.05))",
        },
        {
            "setup": "import numpy as np\ndef pack(x):\n m,c,l=x; return np.concatenate((m.astype(float).ravel(),c.ravel(),np.array([l])))\ns=np.array([1,0,1,0,0,1,0,0]); e=np.array([[3,0],[2,3],[1,2],[0,1]])",
            "call": "pack(hubbard_hop_channels(s,e,.7,3.,.1))",
            "gold_call": "pack(_oracle_hubbard_hop_channels(s,e,.7,3.,.1))",
        },
        {
            "setup": "import numpy as np\ndef pack(x):\n m,c,l=x; return np.concatenate((m.astype(float).ravel(),c.ravel(),np.array([l])))\ns=np.ones(6,dtype=int); e=np.array([[0,1],[1,2]])",
            "call": "pack(hubbard_hop_channels(s,e,1.,4.,.02))",
            "gold_call": "pack(_oracle_hubbard_hop_channels(s,e,1.,4.,.02))",
        },
    ]
