"""
List every copy-number state of the sequential processive n-site phosphorylation and dephosphorylation network that can be reached from a given initial state.

Conservation of total substrate, kinase and phosphatase makes the reachable state space of this network finite, so expectations under its stochastic mass-action dynamics can be evaluated exactly on that set.

Returns
-------
np.ndarray: integer array (M, 2n + 4) of reachable states in ascending lexicographic order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def enumerate_processive_states(n_sites: int, initial_state: "np.ndarray") -> "np.ndarray":
    """Return every state reachable from ``initial_state``, rows in ascending lexicographic order.

    With ``n = n_sites``, a state is an integer vector of length ``2n + 4``
    holding the copy numbers of ``(S_0, S_n, K, F, S_0K, ..., S_{n-1}K,
    S_1F, ..., S_nF)``. The ``4n + 2`` reaction channels, listed in the order
    of the rate vector ``(alpha_1, ..., alpha_{2n+1}, beta_1, ...,
    beta_{2n+1})``, are

    * ``alpha_1: S_0 + K -> S_0K`` and ``alpha_2: S_0K -> S_0 + K``;
    * for ``i = 1, ..., n - 1``, ``alpha_{2i+1}: S_{i-1}K -> S_iK`` and
      ``alpha_{2i+2}: S_iK -> S_{i-1}K``;
    * ``alpha_{2n+1}: S_{n-1}K -> S_n + K``;
    * ``beta_1: S_1F -> S_0 + F``;
    * for ``i = 1, ..., n - 1``, ``beta_{2i}: S_iF -> S_{i+1}F`` and
      ``beta_{2i+1}: S_{i+1}F -> S_iF``;
    * ``beta_{2n}: S_nF -> S_n + F`` and ``beta_{2n+1}: S_n + F -> S_nF``.

    A channel can fire from a state when every reactant copy number is at
    least one. The returned set is the closure of ``{initial_state}`` under
    firing channels, and it includes ``initial_state`` itself.

    Parameters
    ----------
    n_sites : int
        Number of phosphorylation sites, at least 1.
    initial_state : np.ndarray
        One-dimensional array of ``2n + 4`` non-negative integer copy numbers.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(M, 2n + 4)``, one reachable state per row.

    Raises
    ------
    ValueError
        If ``n_sites`` is not an integer of at least 1 (booleans are
        rejected), or if ``initial_state`` does not have ``2n + 4`` entries or
        holds a negative or non-integer entry.
    """
    return states

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _processive_reactions(n_sites):
    """Return reactant counts and state changes of the 4n + 2 channels, rows in rate-vector order."""
    import numpy as np

    n = int(n_sites)
    s_0, s_n, kinase, phosphatase = 0, 1, 2, 3

    def _bound_k(i):  # Column of S_iK, i = 0, ..., n - 1.
        return 4 + i

    def _bound_f(j):  # Column of S_jF, j = 1, ..., n.
        return 3 + n + j

    channels = [((s_0, kinase), (_bound_k(0),)), ((_bound_k(0),), (s_0, kinase))]
    for i in range(1, n):
        channels += [((_bound_k(i - 1),), (_bound_k(i),)), ((_bound_k(i),), (_bound_k(i - 1),))]
    channels.append(((_bound_k(n - 1),), (s_n, kinase)))
    channels.append(((_bound_f(1),), (s_0, phosphatase)))
    for i in range(1, n):
        channels += [((_bound_f(i),), (_bound_f(i + 1),)), ((_bound_f(i + 1),), (_bound_f(i),))]
    channels += [((_bound_f(n),), (s_n, phosphatase)), ((s_n, phosphatase), (_bound_f(n),))]
    reactants = np.zeros((len(channels), 2 * n + 4), dtype=np.int64)
    change = np.zeros_like(reactants)
    for row, (inputs, outputs) in enumerate(channels):
        for species in inputs:
            reactants[row, species] += 1
            change[row, species] -= 1
        for species in outputs:
            change[row, species] += 1
    return reactants, change

def _oracle_enumerate_processive_states(n_sites: int, initial_state: "np.ndarray") -> "np.ndarray":
    """Reference implementation: depth-first closure under the firing channels."""
    import numpy as np

    if isinstance(n_sites, bool) or not isinstance(n_sites, (int, np.integer)) or n_sites < 1:
        raise ValueError("n_sites must be an integer of at least 1")
    start = np.asarray(initial_state)
    if start.ndim != 1 or start.shape[0] != 2 * int(n_sites) + 4:
        raise ValueError("initial_state must have 2 * n_sites + 4 entries")
    if not np.all(np.isfinite(start.astype(float))) or np.any(start.astype(float) != np.round(start.astype(float))):
        raise ValueError("initial_state must hold integer copy numbers")
    if np.any(start < 0):
        raise ValueError("initial_state must be non-negative")
    reactants, change = _processive_reactions(n_sites)
    first = tuple(int(v) for v in start)
    seen, stack = {first}, [first]
    while stack:
        state = np.array(stack.pop(), dtype=np.int64)
        for row in range(reactants.shape[0]):
            if np.all(state >= reactants[row]):
                successor = tuple(int(v) for v in state + change[row])
                if successor not in seen:
                    seen.add(successor)
                    stack.append(successor)
    return np.array(sorted(seen), dtype=np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _digest(states):\n"
        "    a = np.asarray(states, dtype=float)\n"
        "    if a.ndim != 2:\n"
        "        return -1.0\n"
        "    w = np.sqrt(np.arange(1.0, a.size + 1.0))\n"
        "    return float(a.shape[0] + 1.0e-3 * a.shape[1] + 1.0e-4 * np.sum(w * a.ravel()) / np.sqrt(a.size))\n"
    )

    raises = (
        "import numpy as np\n"
        "def _candidate():\n    try:\n        enumerate_processive_states({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n"
        "def _reference():\n    try:\n        _oracle_enumerate_processive_states({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n"
    )
    return [
        {   # Normal: five sites, two kinases.
            "setup": digest + "x0 = np.array([3, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])\n",
            "call": "_digest(enumerate_processive_states(5, x0.copy()))",
            "gold_call": "_digest(_oracle_enumerate_processive_states(5, x0.copy()))",
        },
        {   # Boundary: a single site and one of each molecule.
            "setup": digest + "x0 = np.array([1, 0, 1, 1, 0, 0])\n",
            "call": "_digest(enumerate_processive_states(1, x0.copy()))",
            "gold_call": "_digest(_oracle_enumerate_processive_states(1, x0.copy()))",
        },
        {   # Edge: no enzyme, so no channel can fire.
            "setup": digest + "x0 = np.array([2, 0, 0, 0, 0, 0, 0, 0])\n",
            "call": "_digest(enumerate_processive_states(2, x0.copy()))",
            "gold_call": "_digest(_oracle_enumerate_processive_states(2, x0.copy()))",
        },
        {   # Edge: start from the bound complexes S_1K and S_2F.
            "setup": digest + "x0 = np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 0])\n",
            "call": "_digest(enumerate_processive_states(3, x0.copy()))",
            "gold_call": "_digest(_oracle_enumerate_processive_states(3, x0.copy()))",
        },
        {   # Invalid: a negative copy number.
            "setup": raises.replace("{args}", "2, np.array([1, 0, 1, 1, 0, 0, -1, 0])"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
        {   # Invalid: the wrong number of entries.
            "setup": raises.replace("{args}", "2, np.array([1, 0, 1, 1, 0, 0])"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
    ]
