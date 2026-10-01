"""
Build the matrix of the skew-symmetric bilinear form under which the skew-orthogonal polynomials of the shifted RSK rows of a symmetric geometric waiting-time matrix are defined.

The RSK shape of a symmetric matrix of geometric waiting times, shifted to h_i = lambda_i + N - i, is an orthogonal-type (beta = 1) particle system on the non-negative integers, so its correlations follow from polynomials that are skew-orthogonal under a sign-kernel form weighted by the per-site weight of that system.

Returns
-------
np.ndarray: the (n_sites, n_sites) skew form matrix G.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def build_skew_inner_product(q: float, n_sites: int) -> np.ndarray:
    """Return the matrix of the beta = 1 skew form on the sites 0, 1, ..., n_sites - 1.

    The waiting-time matrix is symmetric with independent entries on and above
    the diagonal: an off-diagonal entry takes the value k = 0, 1, 2, ... with
    probability ``(1 - q) * q**k`` and a diagonal entry with probability
    ``(1 - sqrt(q)) * q**(k / 2)``. For the RSK shape ``lambda`` of an N x N
    such matrix, the shifted rows ``h_i = lambda_i + N - i`` are distinct
    non-negative integers with joint law proportional to
    ``prod_{i<j} |h_i - h_j| * prod_i w(h_i)`` for a per-site weight ``w``,
    normalised here to ``w(0) = 1``. Return the matrix ``G`` of the discrete
    beta = 1 skew form associated with this ensemble, represented so that the
    skew product of two sampled functions is ``<f, g> = f @ G @ g``.
    Use the sign-kernel convention ``epsilon(x, y) = 0.5 * sign(y - x)``:
    its value is +0.5 for x < y, zero for x = y, and -0.5 for x > y.

    Parameters
    ----------
    q : float
        Geometric ratio of the off-diagonal entries, ``0 < q < 1``.
    n_sites : int
        Number of sites ``0, 1, ..., n_sites - 1``; at least 2.

    Returns
    -------
    np.ndarray
        Real skew-symmetric array of shape ``(n_sites, n_sites)``.

    Raises
    ------
    ValueError
        If ``q`` is not a finite real number in the open interval (0, 1)
        (booleans are rejected), or if ``n_sites`` is not an integer of at
        least 2 (booleans are rejected).
    """
    return gram

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_skew_inner_product(q: float, n_sites: int) -> np.ndarray:
    """Reference implementation."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_integer(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    if not (_is_number(q) and 0.0 < q < 1.0):
        raise ValueError("q must be a finite real number in (0, 1)")
    if not (_is_integer(n_sites) and n_sites >= 2):
        raise ValueError("n_sites must be an integer of at least 2")
    sites = np.arange(int(n_sites), dtype=float)
    # Symmetric RSK: a matrix of shape lambda carries prod_{i<j} q^{w_ij}
    # prod_i q^{w_ii / 2} = q^{|lambda| / 2}, and the number of symmetric
    # matrices of shape lambda is s_lambda(1^N), proportional to
    # prod_{i<j} (h_i - h_j). With |lambda| = sum_i h_i - N(N - 1)/2 the
    # per-site weight is q^{h / 2}, normalised to 1 at h = 0.
    weight = float(q) ** (0.5 * sites)
    orientation = np.sign(sites[None, :] - sites[:, None])
    return 0.5 * orientation * np.outer(weight, weight)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
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
    probe = (
        "import numpy as np\n"
        "def _probe(G):\n"
        "    n = G.shape[0]\n"
        "    left = np.cos(0.8 * np.arange(n))\n"
        "    right = np.arange(1.0, n + 1.0) / n\n"
        "    return float(left @ G @ right + np.sum(np.abs(G)) + 10.0 * n)\n"
    )
    return [
        {
            "setup": probe,
            "call": "_probe(build_skew_inner_product(0.2, 12))",
            "gold_call": "_probe(_oracle_build_skew_inner_product(0.2, 12))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(build_skew_inner_product(0.36, 2)[0, 1])",
            "gold_call": "float(_oracle_build_skew_inner_product(0.36, 2)[0, 1])",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(np.sum(build_skew_inner_product(0.5, 7)[:, 6]))",
            "gold_call": "float(np.sum(_oracle_build_skew_inner_product(0.5, 7)[:, 6]))",
        },
        {
            "setup": probe,
            "call": "_probe(build_skew_inner_product(0.93, 25)) / 100.0",
            "gold_call": "_probe(_oracle_build_skew_inner_product(0.93, 25)) / 100.0",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_skew_inner_product(1.0, 10))",
            "gold_call": "_status(lambda: _oracle_build_skew_inner_product(1.0, 10))",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_skew_inner_product(0.3, 1))",
            "gold_call": "_status(lambda: _oracle_build_skew_inner_product(0.3, 1))",
        },
    ]
