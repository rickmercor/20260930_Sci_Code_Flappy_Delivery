"""
Step 1 - the couplon lattice as a cross-class adjacency matrix.

The treatment arranges the channels of the couplon on a checkerboard double row: two rows of equal length, alternating classes on every step of either row, and every contact between channels of different classes only. Channels away from the row ends have three contacts; the two end channels of each row have two. This step returns the class membership mask and the symmetric adjacency matrix of that graph for a lattice of $n_per_class$ channels of each class. Boolean masks are returned as float arrays of 0.0/1.0 so that every step output is a numerical value.

Returns
-------
(is_v, adjacency) : tuple of np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_lattice_adjacency(n_per_class: int = 4) -> tuple:
    """Build the couplon's class mask and adjacency matrix for the checkerboard lattice.

    Parameters
    ----------
    n_per_class : int
        Number of channels of each class. The lattice has two rows of `n_per_class`
        channels, so the total channel count is 2 * n_per_class.

    Returns
    -------
    tuple (is_v, adjacency)
        is_v : np.ndarray
            Shape (2 * n_per_class,), float; entry 1.0 marks a channel of the class in
            contact with the voltage sensors, entry 0.0 the contact-free class.
            The checkerboard convention used elsewhere in the pipeline places the
            sensor-coupled class on lattice positions of even (row + column) parity.
        adjacency : np.ndarray
            Shape (2 * n_per_class, 2 * n_per_class), float; symmetric 0.0/1.0 matrix of
            the four-neighbourhood cross-class contacts.

    Raises
    ------
    ValueError
        If `n_per_class` is less than 2 or not an integer.
    """
    is_v = np.empty(2 * n_per_class, dtype=float)
    adjacency = np.empty((2 * n_per_class, 2 * n_per_class), dtype=float)
    return is_v, adjacency  # to complete!

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_lattice_adjacency(n_per_class=4):
    """Reference implementation for build_lattice_adjacency."""
    import numpy as np

    if isinstance(n_per_class, bool) or int(n_per_class) != n_per_class:
        raise ValueError("n_per_class must be an integer")
    n_per_class = int(n_per_class)
    if n_per_class < 2:
        raise ValueError("n_per_class must be at least 2")
    ncol = n_per_class
    n = 2 * ncol
    is_v = np.zeros(n, dtype=float)
    adj = np.zeros((n, n), dtype=float)
    for r in range(2):
        for c in range(ncol):
            i = r * ncol + c
            is_v[i] = 1.0 if (r + c) % 2 == 0 else 0.0
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 2 and 0 <= cc < ncol:
                    adj[i, rr * ncol + cc] = 1.0
    return is_v, adj

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    sized = (
        "import numpy as np\n"
        "def summarize(is_v, adj):\n"
        "    deg = adj.sum(axis=1)\n"
        "    return np.array([\n"
        "        adj.shape[0],\n"
        "        is_v.sum(),\n"
        "        deg.min(),\n"
        "        deg.max(),\n"
        "        (deg == 2).sum(),\n"
        "        adj.trace(),\n"
        "        float((adj == adj.T).all()),\n"
        "        float((adj * (is_v[:, None] == is_v[None, :])).sum()),\n"
        "    ])\n"
    )
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        build_lattice_adjacency(**kw)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_build_lattice_adjacency(**kw)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
    )
    return [
        # Normal: the graded 4+4 lattice. Cross-class-only check reads zero same-class links.
        {"setup": sized,
         "call": "summarize(*build_lattice_adjacency(4))",
         "gold_call": "summarize(*_oracle_build_lattice_adjacency(4))"},
        # Boundary: the smallest legal checkerboard (2+2, every channel a corner of degree 2).
        {"setup": sized,
         "call": "summarize(*build_lattice_adjacency(2))",
         "gold_call": "summarize(*_oracle_build_lattice_adjacency(2))"},
        # Edge: odd column count (3+3), so the two rows interleave differently.
        {"setup": sized,
         "call": "summarize(*build_lattice_adjacency(3))",
         "gold_call": "summarize(*_oracle_build_lattice_adjacency(3))"},
        # Edge: a long row (the paper's 30+30 scale), cheap to build, still exact.
        {"setup": sized,
         "call": "summarize(*build_lattice_adjacency(30))",
         "gold_call": "summarize(*_oracle_build_lattice_adjacency(30))"},
        # Invalid: too few channels per class.
        {"setup": invalid,
         "call": "run_model(n_per_class=1)",
         "gold_call": "run_gold(n_per_class=1)"},
        # Invalid: non-integer size.
        {"setup": invalid,
         "call": "run_model(n_per_class=2.5)",
         "gold_call": "run_gold(n_per_class=2.5)"},
    ]
