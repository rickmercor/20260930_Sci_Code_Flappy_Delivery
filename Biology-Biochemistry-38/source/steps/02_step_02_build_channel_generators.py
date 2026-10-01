"""
Assemble, for every reaction channel of the processive network, the unit-rate generator matrix of the continuous-time Markov chain on a supplied closed set of states.

Under stochastic mass-action kinetics, each channel's intensity is its rate constant times a combinatorial factor of the reactant copy numbers, so the chain's generator is linear in the rate vector and splits into one matrix per channel.

Returns
-------
np.ndarray: float array (4n + 2, M, M) of unit-rate channel generators.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_channel_generators(n_sites: int, states: "np.ndarray") -> "np.ndarray":
    """Return the unit-rate generator of every channel on the supplied states.

    Channels, species order and firing rules are those of
    ``enumerate_processive_states``. For channel ``l`` with reactant counts
    ``nu_l`` and state change ``zeta_l``, the combinatorial factor is
    ``h_l(x) = prod_s x_s (x_s - 1) ... (x_s - nu_{s,l} + 1)``. In generator
    ``l``, row ``i`` holds ``h_l(x_i)`` in the column of the state
    ``x_i + zeta_l`` and ``-h_l(x_i)`` on the diagonal, and is zero when
    ``h_l(x_i) = 0``. Rows and columns follow the row order of ``states``, so
    the generator of the chain with rate vector ``theta`` is
    ``sum_l theta_l * G[l]``.

    Parameters
    ----------
    n_sites : int
        Number of phosphorylation sites, at least 1.
    states : np.ndarray
        Integer array of shape ``(M, 2n + 4)`` of distinct non-negative
        states, closed under every channel that can fire from them.

    Returns
    -------
    np.ndarray
        Float array ``G`` of shape ``(4n + 2, M, M)``.

    Raises
    ------
    ValueError
        If ``n_sites`` is not an integer of at least 1, if ``states`` is not
        a two-dimensional array of non-negative integers with ``2n + 4``
        columns and distinct rows, or if a channel that can fire leads out of
        the supplied set.
    """
    return generators

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_channel_generators(n_sites: int, states: "np.ndarray") -> "np.ndarray":
    """Reference implementation: one sparse row per state and channel."""
    import numpy as np

    if isinstance(n_sites, bool) or not isinstance(n_sites, (int, np.integer)) or n_sites < 1:
        raise ValueError("n_sites must be an integer of at least 1")
    table = np.asarray(states)
    if table.ndim != 2 or table.shape[1] != 2 * int(n_sites) + 4 or table.shape[0] < 1:
        raise ValueError("states must have shape (M, 2 * n_sites + 4)")
    as_float = table.astype(float)
    if not np.all(np.isfinite(as_float)) or np.any(as_float != np.round(as_float)) or np.any(as_float < 0):
        raise ValueError("states must hold non-negative integers")
    table = table.astype(np.int64)
    index = {tuple(int(v) for v in row): i for i, row in enumerate(table)}
    if len(index) != table.shape[0]:
        raise ValueError("states must be distinct")
    reactants, change = _processive_reactions(n_sites)
    size = table.shape[0]
    generators = np.zeros((reactants.shape[0], size, size))
    for channel in range(reactants.shape[0]):
        for i, state in enumerate(table):
            factor = 1.0
            for species in np.flatnonzero(reactants[channel]):
                for k in range(int(reactants[channel, species])):
                    factor *= float(max(int(state[species]) - k, 0))
            if factor == 0.0:
                continue
            target = index.get(tuple(int(v) for v in state + change[channel]))
            if target is None:
                raise ValueError("a firing channel leads out of the supplied states")
            generators[channel, i, target] += factor
            generators[channel, i, i] -= factor
    return generators

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    helpers = (
        "import itertools\n"
        "import numpy as np\n"
        "def _conserved(n, s_tot, k_tot, f_tot):\n"
        "    d = 2 * n + 4\n"
        "    rows = []\n"
        "    for x in itertools.product(range(max(s_tot, k_tot, f_tot) + 1), repeat=d):\n"
        "        s = x[0] + x[1] + sum(x[4:4 + n]) + sum(x[4 + n:])\n"
        "        if s == s_tot and x[2] + sum(x[4:4 + n]) == k_tot and x[3] + sum(x[4 + n:]) == f_tot:\n"
        "            rows.append(x)\n"
        "    return np.array(rows, dtype=np.int64)\n"
        "def _digest(stack):\n"
        "    g = np.asarray(stack, dtype=float)\n"
        "    if g.ndim != 3 or g.shape[1] != g.shape[2]:\n"
        "        return -1.0\n"
        "    m = g.shape[1]\n"
        "    w = np.outer(np.arange(1.0, m + 1.0), np.sqrt(np.arange(1.0, m + 1.0)))\n"
        "    per = np.array([np.sum(w * b) + 0.5 * np.sum(w * np.abs(b)) for b in g])\n"
        "    return float(g.shape[0] + 1.0e-3 * m + 1.0e-4 * np.sum(per * np.log(np.arange(2.0, g.shape[0] + 2.0))) / m)\n"
    )
    one_site = "np.array([[0, 0, 0, 1, 1, 0], [0, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0], [1, 0, 1, 1, 0, 0]])"

    raises = (
        "def _candidate():\n    try:\n        build_channel_generators({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n"
        "def _reference():\n    try:\n        _oracle_build_channel_generators({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n"
    )
    return [
        {   # Normal: two sites, bimolecular channels active.
            "setup": helpers + "states = _conserved(2, 2, 1, 1)\n",
            "call": "_digest(build_channel_generators(2, states.copy()))",
            "gold_call": "_digest(_oracle_build_channel_generators(2, states.copy()))",
        },
        {   # Boundary: one site, four states.
            "setup": helpers + "states = " + one_site + "\n",
            "call": "_digest(build_channel_generators(1, states.copy()))",
            "gold_call": "_digest(_oracle_build_channel_generators(1, states.copy()))",
        },
        {   # Edge: rows supplied in reverse order.
            "setup": helpers + "states = " + one_site + "[::-1]\n",
            "call": "_digest(build_channel_generators(1, states.copy()))",
            "gold_call": "_digest(_oracle_build_channel_generators(1, states.copy()))",
        },
        {   # Edge: no enzyme, so every mass-action factor is zero.
            "setup": helpers + "states = np.array([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0]])\n",
            "call": "_digest(build_channel_generators(3, states.copy()))",
            "gold_call": "_digest(_oracle_build_channel_generators(3, states.copy()))",
        },
        {   # Invalid: the supplied set is not closed under the channels.
            "setup": helpers + "states = " + one_site + "[:3]\n" + raises.replace("{args}", "1, states.copy()"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
        {   # Invalid: the wrong number of columns for n_sites.
            "setup": helpers + "states = " + one_site + "\n" + raises.replace("{args}", "2, states.copy()"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
    ]
