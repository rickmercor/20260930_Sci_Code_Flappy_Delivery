"""
Step 2 - contact energy change on opening per channel.

Every channel's closed-to-open transition is allosterically modified by the state of its neighbours: for each of its cross-class contacts, opening pays or gains the corresponding contact energy depending on whether the contact's far side is closed or open. The treatment tables a single pair of contact energies, one for the open-open configuration and one for the closed-closed configuration, and the mixed configurations carry zero energy (a gauge choice with no effect on rates or equilibria). This step computes, from the adjacency matrix and the current occupancy of the couplon, the net contact energy change that would follow from opening each currently closed channel. Inactivated channels count as closed for this accounting. The result is the per-channel energy vector consumed by the transition-rate law of the next step.

Returns
-------
np.ndarray float (n,): per-channel net contact energy change on opening, in kT
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def contact_energy_changes(adjacency: np.ndarray, open_mask: np.ndarray) -> np.ndarray:
    """Net contact energy change on opening for each channel of the couplon lattice.

    Parameters
    ----------
    adjacency : np.ndarray
        Shape (n, n), floating or boolean 0-1 matrix of the cross-class contacts, as built
        by `build_lattice_adjacency`. Must be square and symmetric within tolerance.
    open_mask : np.ndarray
        Shape (n,), 0.0/1.0 marks of which channels are currently open. Only open channels
        of either class contribute contact energy to a neighbour's opening barrier.

    Returns
    -------
    np.ndarray
        Shape (n,), native floats: the net contact energy change on opening, in kT, for
        each channel of the lattice.

    Raises
    ------
    ValueError
        If shapes are inconsistent, `adjacency` is not square/symmetric, `open_mask`
        contains non-binary marks, or values are not finite.
    """
    delta_energy_vector = np.empty(adjacency.shape[0], dtype=float)
    return delta_energy_vector  # placeholder to complete!

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_contact_energy_changes(adjacency, open_mask):
    """Reference implementation for contact_energy_changes."""
    import numpy as np

    EPS_OO, EPS_CC = -5.7, -0.7  # kT, the treatment's contact energies (Table 1)
    adjacency = np.asarray(adjacency, dtype=float)
    open_mask = np.asarray(open_mask, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.all(np.isfinite(adjacency)):
        raise ValueError("adjacency must be finite")
    if not np.allclose(adjacency, adjacency.T, rtol=0.0, atol=1e-12):
        raise ValueError("adjacency must be symmetric")
    if open_mask.ndim != 1 or open_mask.shape[0] != adjacency.shape[0]:
        raise ValueError("open_mask must align with adjacency")
    if not np.all(np.isfinite(open_mask)) or not np.all((open_mask == 0.0) | (open_mask == 1.0)):
        raise ValueError("open_mask must contain only 0.0/1.0")

    # Each open neighbour contributes EPS_OO to the opening energy change; each closed
    # neighbour's contact (energy EPS_CC) ceases to exist on opening and therefore enters
    # with a minus sign. Inactivated channels enter as closed because open_mask marks them 0.
    degree = adjacency.sum(axis=1)
    n_open = adjacency @ open_mask
    return n_open * EPS_OO - (degree - n_open) * EPS_CC

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = (
        "import numpy as np\n"
        "ncol = 4\n"
        "n = 2 * ncol\n"
        "adj = np.zeros((n, n), float)\n"
        "for r in range(2):\n"
        "    for c in range(ncol):\n"
        "        i = r * ncol + c\n"
        "        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):\n"
        "            rr, cc = r + dr, c + dc\n"
        "            if 0 <= rr < 2 and 0 <= cc < ncol:\n"
        "                adj[i, rr * ncol + cc] = 1.0\n"
        # The fixture mirrors the graded 4+4 checkerboard (degrees 2/2/3/3/2/2/3/3).
    )
    invalid = setup + (
        "def run_model(a, o):\n"
        "    try:\n"
        "        contact_energy_changes(a, o)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
        "def run_gold(a, o):\n"
        "    try:\n"
        "        _oracle_contact_energy_changes(a, o)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
    )
    patterns = [
        # Normal: all closed. D_i = -deg_i * eps_CC = +0.7 * deg_i for every channel.
        "om = np.zeros(adj.shape[0])\n",
        # Normal: only channel 0 (a degree-2 corner) open; its neighbours should each see
        # one EPS_OO against their remaining closed contacts.
        "om = np.zeros(adj.shape[0]); om[0] = 1.0\n",
        # Boundary: all open; D_i = deg_i * EPS_OO for every channel.
        "om = np.ones(adj.shape[0])\n",
        # Edge: an arbitrary mid-pulse pattern (open at indices 1, 4, 7).
        "om = np.zeros(adj.shape[0]); om[[1, 4, 7]] = 1.0\n",
    ]
    return [
        # Normal: all channels closed. D_i = -deg_i * eps_CC = +0.7 * deg_i.
        {"setup": setup + patterns[0],
         "call": "contact_energy_changes(adj, om)",
         "gold_call": "_oracle_contact_energy_changes(adj, om)"},
        # Normal: only channel 0 (a degree-2 corner) open; its neighbours each see one
        # EPS_OO against their remaining closed contacts.
        {"setup": setup + patterns[1],
         "call": "contact_energy_changes(adj, om)",
         "gold_call": "_oracle_contact_energy_changes(adj, om)"},
        # Boundary: all channels open; D_i = deg_i * EPS_OO for every channel.
        {"setup": setup + patterns[2],
         "call": "contact_energy_changes(adj, om)",
         "gold_call": "_oracle_contact_energy_changes(adj, om)"},
        # Edge: an arbitrary mid-pulse pattern (open at indices 1, 4, 7).
        {"setup": setup + patterns[3],
         "call": "contact_energy_changes(adj, om)",
         "gold_call": "_oracle_contact_energy_changes(adj, om)"},
    ] + [
        # Invalid: open_mask with a non-binary mark.
        {"setup": invalid,
         "call": "run_model(adj, np.array([0.5] + [0.0] * (adj.shape[0] - 1)))",
         "gold_call": "run_gold(adj, np.array([0.5] + [0.0] * (adj.shape[0] - 1)))"},
        # Invalid: asymmetric adjacency.
        {"setup": invalid + "bad = adj.copy(); bad[0, 1] = 1.0; bad[1, 0] = 0.0\n",
         "call": "run_model(bad, np.zeros(adj.shape[0]))",
         "gold_call": "run_gold(bad, np.zeros(adj.shape[0]))"},
        # Invalid: mask shape mismatch.
        {"setup": invalid,
         "call": "run_model(adj, np.zeros(adj.shape[0] - 1))",
         "gold_call": "run_gold(adj, np.zeros(adj.shape[0] - 1))"},
    ]
