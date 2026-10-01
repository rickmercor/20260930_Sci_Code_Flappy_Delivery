"""
Assemble the couplon's Markov generator from the per-channel transition law.

With the lattice in hand and the per-channel transition law fixed, the whole couplon is a single continuous-time Markov chain whose states are the joint configurations of all channels. Each V channel contributes 2 states (closed, open) and each C channel 4 (closed, open, first inactivated, second inactivated), so the graded 4+4 lattice has 4^4 x 2^4 = 4096 global states. The generator Q is the matrix of single-channel transition rates between global states: off-diagonal entry Q[dst, src] is the rate of the single-channel transition that turns src into dst (the per-channel law of step 03 evaluated on the neighbour configuration of src), and the diagonal entry Q[src, src] is minus the total exit rate of src, so every column sums to zero. Global states are enumerated in channel-index order with a mixed-radix encoding: the digit of a channel runs over its own local states and channel 0 is the slowest-varying. The step returns the generator in coordinate form (row indices, column indices, values), all three entries are plain numerical arrays, and the full sparse generator is recovered from them with a three-term construction.

Returns
-------
(rows, cols, values) : tuple of np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_generator(is_v, adjacency, eps_v, nu=1.0, nu_v=0.8):
    """Assemble the couplon generator in coordinate form at one pulse energy.

    Parameters
    ----------
    is_v : np.ndarray
        Shape (n,), 0.0/1.0 marks of the sensor-coupled class, from step 01.
    adjacency : np.ndarray
        Shape (n, n), float 0.0/1.0 cross-class contact matrix, from step 01.
    eps_v : float
        Electrical energy of the applied pulse, in kT, delivered to the sensor-coupled
        class.
    nu, nu_v : float
        Distribution coefficients of the treatment, applied to the contact and electrical
        energy changes respectively.

    Returns
    -------
    tuple (rows, cols, values)
        rows : np.ndarray of shape (nnz,), integer row indices (destination states)
        cols : np.ndarray of shape (nnz,), integer column indices (source states)
        values : np.ndarray of shape (nnz,), float transition rates in ms^-1
        The full generator is the 4^4 x 2^4 = 4096 (for the graded lattice) sparse matrix
        with entries values[k] at (rows[k], cols[k]); every column sums to zero.

    Raises
    ------
    ValueError
        If the inputs are inconsistent in shape, or if eps_v / nu / nu_v are not finite.
    """
    n_channels = np.asarray(is_v).size
    n_states = int(np.prod([2 if v else 4 for v in np.asarray(is_v)]))
    rows = np.empty(0, dtype=np.int64)
    cols = np.empty(0, dtype=np.int64)
    values = np.empty(0, dtype=float)
    return rows, cols, values  # placeholder to fill!

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_assemble_generator(is_v, adjacency, eps_v, nu=1.0, nu_v=0.8):
    """Reference implementation for assemble_generator."""
    import numpy as np

    def _global_state_table(is_v):
        is_v = np.asarray(is_v, dtype=bool)
        n = is_v.size
        dims = tuple(2 if v else 4 for v in is_v)
        n_states = int(np.prod(dims))
        digits = np.stack(np.unravel_index(np.arange(n_states), dims), axis=1)
        strides = np.array([int(np.prod(dims[c + 1:])) for c in range(n)])
        return digits, strides, n_states

    is_v = np.asarray(is_v, dtype=float)
    adjacency = np.asarray(adjacency, dtype=float)
    for name, value in (("eps_v", eps_v), ("nu", nu), ("nu_v", nu_v)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    n = is_v.size
    if adjacency.shape != (n, n):
        raise ValueError("shape mismatch between is_v and adjacency")

    digits, strides, n_states = _global_state_table(is_v != 0.0)
    rows, cols, data = [], [], []

    for src in range(n_states):
        local = digits[src]
        open_mask = (local == 1).astype(float)
        D = _oracle_contact_energy_changes(adjacency, open_mask)
        for c in range(n):
            c_local = int(local[c])
            rates = _oracle_channel_transition_rates(
                c_local, bool(is_v[c]), float(D[c]), float(eps_v),
                nu=float(nu), nu_v=float(nu_v))
            for dst_local in range(4):
                if dst_local == c_local or rates[dst_local] <= 0.0:
                    continue
                dst = int(src + (dst_local - c_local) * strides[c])
                rows.append(dst)
                cols.append(src)
                data.append(float(rates[dst_local]))

    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)
    data = np.array(data, dtype=float)

    # Conservative diagonal: Q[src, src] = -(total exit rate of src). Column sums of the
    # off-diagonal entries give the exit rates directly.
    exit_rate = np.zeros(n_states, dtype=float)
    np.add.at(exit_rate, cols, data)
    diag_rows = np.arange(n_states, dtype=np.int64)
    rows = np.concatenate([rows, diag_rows])
    cols = np.concatenate([cols, diag_rows])
    data = np.concatenate([data, -exit_rate])
    return rows, cols, data

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = (
        "import numpy as np\n"
        "def _lattice(n_per_class):\n"
        "    ncol = n_per_class\n"
        "    n = 2 * ncol\n"
        "    is_v = np.zeros(n, float)\n"
        "    adj = np.zeros((n, n), float)\n"
        "    for r in range(2):\n"
        "        for c in range(ncol):\n"
        "            i = r * ncol + c\n"
        "            is_v[i] = 1.0 if (r + c) % 2 == 0 else 0.0\n"
        "            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):\n"
        "                rr, cc = r + dr, c + dc\n"
        "                if 0 <= rr < 2 and 0 <= cc < ncol:\n"
        "                    adj[i, rr * ncol + cc] = 1.0\n"
        "    return is_v, adj\n"
        "is_v, adj = _lattice(4)\n"
    )
    summary = (
        "def summarize(rows, cols, values):\n"
        "    exit_ = -values[cols == rows] if np.any(cols == rows) else None\n"
        "    n_states = int(np.sqrt(len(exit_))) if exit_ is not None else 0\n"
        "    colsum = np.zeros(n_states ** 2); np.add.at(colsum, cols, values)\n"
        "    return np.array([\n"
        "        float(values.size),\n"
        "        float(values.max()),\n"
        "        float(exit_.min()),\n"
        "        float(exit_.max()),\n"
        "        float(np.abs(colsum).sum()),\n"
        "    ])\n"
    )
    invalid = setup + (
        "def run_model(**kw):\n"
        "    try:\n"
        "        assemble_generator(**kw)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_assemble_generator(**kw)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
    )
    # The summary collapses structurally interesting scalars: nnz count, biggest off-
    # diagonal rate, smallest and biggest exit rates, and total column imbalance (must be 0
    # for a conservative generator).
    return [
        # Normal: mid-family pulse on the graded lattice.
        {"setup": setup + summary,
         "call": "summarize(*assemble_generator(is_v, adj, -8.0))",
         "gold_call": "summarize(*_oracle_assemble_generator(is_v, adj, -8.0))"},
        # Boundary: resting energy; electrical factors are exactly 1.
        {"setup": setup + summary,
         "call": "summarize(*assemble_generator(is_v, adj, 0.0))",
         "gold_call": "summarize(*_oracle_assemble_generator(is_v, adj, 0.0))"},
        # Edge: saturating energy at the end of the family.
        {"setup": setup + summary,
         "call": "summarize(*assemble_generator(is_v, adj, -14.0))",
         "gold_call": "summarize(*_oracle_assemble_generator(is_v, adj, -14.0))"},
    ] + [
        # Edge: smaller 2+2 lattice (256 states), still exact arithmetic.
        {"setup": setup.replace("is_v, adj = _lattice(4)", "is_v, adj = _lattice(2)") + summary,
         "call": "summarize(*assemble_generator(is_v, adj, -8.0))",
         "gold_call": "summarize(*_oracle_assemble_generator(is_v, adj, -8.0))"},
        # Invalid: shape mismatch between is_v and adjacency.
        {"setup": invalid,
         "call": "run_model(is_v=is_v[:3], adjacency=adj, eps_v=-8.0)",
         "gold_call": "run_gold(is_v=is_v[:3], adjacency=adj, eps_v=-8.0)"},
        # Invalid: non-finite pulse energy.
        {"setup": invalid,
         "call": "run_model(is_v=is_v, adjacency=adj, eps_v=float('nan'))",
         "gold_call": "run_gold(is_v=is_v, adjacency=adj, eps_v=float('nan'))"},
    ]
