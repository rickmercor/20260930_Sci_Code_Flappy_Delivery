"""
Propagate fixed-spin Hubbard walkers through a supplied matrix of random draws.

A complete stochastic trajectory is obtained by composing the channel-construction and single-step operations at every supplied draw. Each walker carries a binary occupation and a cumulative signed amplitude. The implementation must preserve walker order and must not mutate any input array.

Returns
-------
Return a tuple containing the final binary occupation rows and the cumulative signed amplitude of every walker.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def propagate_hubbard_walkers(
    initial_states: np.ndarray,
    edges: np.ndarray,
    hopping: float,
    interaction: float,
    delta_beta: float,
    uniforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate fixed-spin walkers through a supplied draw matrix.

    Compose hubbard_hop_channels and sample_hubbard_step at every draw. Return
    final occupation rows and cumulative signed amplitudes without mutating
    any input array.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_propagate_hubbard_walkers(
    initial_states: np.ndarray,
    edges: np.ndarray,
    hopping: float,
    interaction: float,
    delta_beta: float,
    uniforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    raw_states = np.asarray(initial_states, dtype=float)
    draws = np.asarray(uniforms, dtype=float)
    if (
        raw_states.ndim != 2
        or len(raw_states) == 0
        or raw_states.shape[1] % 2
    ):
        raise ValueError("initial_states must be a nonempty two-spin array")
    if np.any((raw_states != 0) & (raw_states != 1)):
        raise ValueError("states must be binary")
    if draws.ndim != 2 or draws.shape[0] != len(raw_states):
        raise ValueError("one uniform row is required per walker")
    if np.any(~np.isfinite(draws)) or np.any((draws < 0) | (draws >= 1)):
        raise ValueError("uniforms must lie in [0,1)")
    if hopping < 0 or interaction < 0 or delta_beta <= 0:
        raise ValueError("invalid Hubbard parameters")
    edge_array = np.asarray(edges, dtype=int)
    if edge_array.ndim != 2 or edge_array.shape[1:] != (2,):
        raise ValueError("edges must have shape (n_edges, 2)")

    states = raw_states.astype(int).copy()
    amplitudes = np.ones(len(states), dtype=float)
    channel_cache = {}

    for step_index in range(draws.shape[1]):
        for walker in range(len(states)):
            key = tuple(states[walker])
            if key not in channel_cache:
                channel_cache[key] = _oracle_hubbard_hop_channels(
                    states[walker],
                    edge_array,
                    hopping,
                    interaction,
                    delta_beta,
                )

            moves, coefficients, normalization = channel_cache[key]
            states[walker], factor = _oracle_sample_hubbard_step(
                states[walker],
                moves,
                coefficients,
                normalization,
                interaction,
                delta_beta,
                draws[walker, step_index],
            )
            amplitudes[walker] *= factor

    return states, amplitudes

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\ndef pack(x):\n s,a=x; return np.concatenate((s.astype(float).ravel(),a.ravel()))\ns=np.array([[1,1,0,0,1,0,0,0],[1,0,1,0,0,1,0,0],[0,1,0,1,0,0,1,0]]); e=np.array([[0,1],[1,2],[2,3],[3,0]]); u=np.array([[.02,.93,.41,.77,.11],[.88,.15,.62,.99,.34],[.51,.05,.95,.28,.83]])",
            "call": "pack(propagate_hubbard_walkers(s,e,1.,2.,.05,u))",
            "gold_call": "pack(_oracle_propagate_hubbard_walkers(s,e,1.,2.,.05,u))",
        },
        {
            "setup": "import numpy as np\ndef pack(x):\n s,a=x; return np.concatenate((s.astype(float).ravel(),a.ravel()))\ns=np.array([[1,0,1,0,0,1,0,0],[0,1,0,1,1,0,0,0]]); e=np.array([[3,0],[2,3],[1,2],[0,1]]); u=np.random.default_rng(913).random((2,7))",
            "call": "pack(propagate_hubbard_walkers(s,e,.8,1.5,.1,u))",
            "gold_call": "pack(_oracle_propagate_hubbard_walkers(s,e,.8,1.5,.1,u))",
        },
        {
            "setup": "import numpy as np\ndef pack(x):\n s,a=x; return np.concatenate((s.astype(float).ravel(),a.ravel()))\ns=np.array([[1,0,1,0]]); e=np.array([[0,1]]); u=np.empty((1,0))",
            "call": "pack(propagate_hubbard_walkers(s,e,1.,2.,.05,u))",
            "gold_call": "pack(_oracle_propagate_hubbard_walkers(s,e,1.,2.,.05,u))",
        },
    ]
