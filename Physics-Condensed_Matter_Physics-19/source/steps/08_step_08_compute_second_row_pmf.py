"""
Compose every earlier step to obtain the exact probability that the second row of the RSK shape of a symmetric geometric waiting-time matrix has a given length.

The second row of the shape plays the role of the second-largest eigenvalue of an orthogonal random-matrix ensemble.

Returns
-------
float: the probability that the second RSK row has the requested length.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_second_row_pmf(
    q: float = 0.2,
    n_rows: int = 20,
    row_length: int = 21,
    n_sites: int = 120,
) -> float:
    """Return P(lambda_2 = row_length) for a symmetric geometric waiting-time matrix.

    The ``n_rows x n_rows`` waiting-time matrix is symmetric with independent
    entries on and above the diagonal: an off-diagonal entry takes the value
    ``k = 0, 1, 2, ...`` with probability ``(1 - q) * q**k`` and a diagonal
    entry with probability ``(1 - sqrt(q)) * q**(k / 2)``. ``lambda`` is the
    shape that the Robinson-Schensted-Knuth correspondence assigns to the
    matrix, so ``lambda_1`` is the last-passage time from ``(1, 1)`` to
    ``(n_rows, n_rows)``. The shifted rows ``h_i = lambda_i + n_rows - i`` are
    treated exactly on the sites ``0, 1, ..., n_sites - 1``, which truncate
    the non-negative integers. Return the probability that the second row has
    length ``row_length``. The defaults reproduce the problem statement; at
    these defaults the 120-site truncation changes the result by less than
    1e-15.

    Parameters
    ----------
    q : float
        Geometric ratio of the off-diagonal entries, ``0 < q < 1``.
    n_rows : int
        Matrix size, an even integer of at least 2.
    row_length : int
        Positive integer length of the second row.
    n_sites : int
        Number of sites, with ``row_length + n_rows - 1 < n_sites``.

    Returns
    -------
    float
        The probability ``P(lambda_2 = row_length)``.

    Raises
    ------
    ValueError
        If ``q`` is not a finite real number in (0, 1), ``n_rows`` is not an
        even integer of at least 2, ``row_length`` is not a positive
        integer, ``n_sites`` is not an integer with
        ``row_length + n_rows - 1 < n_sites`` (booleans are rejected for all
        integers), or any stage rejects its input.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_second_row_pmf(
    q: float = 0.2,
    n_rows: int = 20,
    row_length: int = 21,
    n_sites: int = 120,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _is_integer(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    if not (_is_integer(n_rows) and n_rows >= 2 and n_rows % 2 == 0):
        raise ValueError("n_rows must be an even integer of at least 2")
    if not (_is_integer(row_length) and row_length >= 1):
        raise ValueError("row_length must be a positive integer")
    if not (_is_integer(n_sites) and row_length + n_rows - 1 < n_sites):
        raise ValueError("n_sites must exceed row_length + n_rows - 1")
    gram = _oracle_build_skew_inner_product(q, n_sites)
    nodes = np.arange(int(n_sites), dtype=float)
    polys, _ = _oracle_run_symplectic_arnoldi(gram, nodes, int(n_rows))
    # gram[0, x] = w(0) w(x) / 2 with w(0) = 1 recovers the site weights.
    weights = 2.0 * gram[0].copy()
    weights[0] = 1.0
    kernel = _oracle_assemble_pfaffian_kernel(polys, weights)
    # lambda_2 = row_length  <=>  the second-largest shifted row sits at `site`.
    site = int(row_length) + int(n_rows) - 2
    density = kernel[2 * site, 2 * site + 1]
    if not (np.isfinite(density) and density > 0.0):
        raise ValueError("the target site carries no probability")
    conditioned = _oracle_condition_kernel(kernel, [site], True)
    # After removing `site`, the sites above it are relabelled site, ..., n_sites - 2.
    above = list(range(site, int(n_sites) - 1))
    return float(density * _oracle_compute_single_occupancy_probability(conditioned, above))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only end-to-end test specifications."""
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": "import numpy as np\n",
            "call": "compute_second_row_pmf()",
            "gold_call": "_oracle_compute_second_row_pmf()",
        },
        {
            "setup": "import numpy as np\n",
            "call": "compute_second_row_pmf(0.3, 6, 7, 70)",
            "gold_call": "_oracle_compute_second_row_pmf(0.3, 6, 7, 70)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "compute_second_row_pmf(0.45, 2, 2, 60)",
            "gold_call": "_oracle_compute_second_row_pmf(0.45, 2, 2, 60)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "compute_second_row_pmf(0.1, 8, 3, 60)",
            "gold_call": "_oracle_compute_second_row_pmf(0.1, 8, 3, 60)",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_second_row_pmf(0.2, 5, 4, 60))",
            "gold_call": "_status(lambda: _oracle_compute_second_row_pmf(0.2, 5, 4, 60))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_second_row_pmf(0.2, 10, 30, 38))",
            "gold_call": "_status(lambda: _oracle_compute_second_row_pmf(0.2, 10, 30, 38))",
        },
    ]
