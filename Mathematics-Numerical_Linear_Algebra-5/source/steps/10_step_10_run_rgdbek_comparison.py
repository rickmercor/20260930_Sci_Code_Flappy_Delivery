"""
Chains the instance construction, the full serial RGDBEK march, the full hybrid-parallel RGDBEK march, and the convergence diagnostics of the preceding steps, in order, at the requested configuration, and extracts the single requested observable: the ratio of the parallel march's relative solution error to the serial march's relative solution error.

Result extraction for the pinned benchmark.

Running the instance-construction step once, then the serial march and the parallel march independently against that same instance, each for its own fixed iteration budget starting from the same zero-vector initial solution estimate, and finally the diagnostics step against both marches' results, is the complete RGDBEK comparison experiment: build the pinned inconsistent least-squares instance, run the serial and parallel marches against it, and read off how far each march's final solution estimate landed from the instance's exact least-squares solution, then divide the parallel march's relative error by the serial march's. Because every step of the chain is fully deterministic given its inputs, and every default value reproduces the pinned benchmark configuration exactly, calling this function with no arguments reproduces the benchmark run and its relative-error ratio with no additional post-processing beyond the division already performed inside the diagnostics step.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_rgdbek_comparison(seed: int = 20260920, march_seed: int = 7, m: int = 600, n: int = 150, density: float = 0.20, noise_ratio: float = 0.10, eta: float = 0.2, iterations: int = 12, n_ranks: int = 4) -> float:
    """Run the full RGDBEK comparison experiment and return the error ratio.

    Parameters
    ----------
    seed : int
        Seed for the instance construction's random number generator
        (must be a non-negative integer).
    march_seed : int
        Seed for the serial march's single, threaded random number
        generator. The parallel march is seeded with this same integer
        value but on its own, freshly constructed, separate generator
        object (must be a non-negative integer).
    m : int
        Number of rows of the coefficient matrix (must be a positive
        integer with m >= n).
    n : int
        Number of columns of the coefficient matrix (must be a positive
        integer with n <= m).
    density : float
        Target fraction of nonzero entries of the coefficient matrix
        (must satisfy 0 < density <= 1).
    noise_ratio : float
        Ratio of the noise vector's Euclidean norm to the Euclidean norm
        of the noiseless signal (must be > 0).
    eta : float
        Block-size fraction for both marches (must satisfy 0 < eta <= 1).
    iterations : int
        Number of fixed iterations each march runs (must be a positive
        integer).
    n_ranks : int
        Number of ranks the parallel march's row dimension is
        partitioned into (must be a positive integer no greater than
        m; ranks own a fixed, contiguous partition of the rows,
        determined by the rule named in the configuration).

    Returns
    -------
    error_ratio : float
        The ratio of the parallel march's relative solution error to
        the serial march's relative solution error, after each march's
        fixed-budget run:
        (||x_T^par - x_star||_2 / ||x_star||_2)
        / (||x_T - x_star||_2 / ||x_star||_2).

    Raises
    ------
    ValueError
        If any parameter violates the constraints stated above, or if
        any chained step itself raises ValueError on the given
        configuration.
    """
    return error_ratio

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _row_block_bounds(A, n_ranks):
    """Contiguous, nonzero-count-balanced row partition of A's rows into
    n_ranks blocks, in index order (S22: same-named function redefined
    elsewhere must be byte-identical). Row i's nonzero count c_i is
    computed once over the whole matrix; C = sum(c_i) is the total
    nonzero count. The running prefix sum S(i) = c_0 + ... + c_i is a
    single monotone pass over the whole matrix, taken from row 0 and
    never reset at a block boundary. For p = 0, ..., n_ranks - 2 (in
    increasing p order), boundary p closes immediately after the first
    row index i such that S(i) >= (p + 1) * C / n_ranks (i.e. the first
    row whose global cumulative nonzero count reaches or exceeds a
    (p + 1) / n_ranks share of the whole matrix's total nonzero count);
    the final rank takes every remaining row. Non-degeneracy guard,
    applied left to right in increasing p order: if the scan would ever
    produce an empty block, the boundary is pulled forward by one row so
    every rank owns at least one row."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m = A.shape[0]
    if not isinstance(n_ranks, (int, np.integer)) or int(n_ranks) < 1:
        raise ValueError("n_ranks must be a positive integer")
    n_ranks = int(n_ranks)
    if m < n_ranks:
        raise ValueError(f"m={m} rows fewer than n_ranks={n_ranks}")
    c = np.count_nonzero(A, axis=1).astype(np.int64)
    C = int(c.sum())
    if C == 0:
        raise ValueError("A has no nonzero entries; cannot balance the row partition")
    S = np.cumsum(c)
    bounds = []
    lo = 0
    for p in range(n_ranks - 1):
        target = (p + 1) * C / n_ranks
        i = int(np.searchsorted(S, target, side="left"))
        i = min(i, m - 1)
        hi = i + 1
        if hi <= lo:
            hi = lo + 1
        bounds.append((lo, hi))
        lo = hi
    bounds.append((lo, m))
    return bounds


def _oracle_run_rgdbek_comparison(seed: int = 20260920, march_seed: int = 7, m: int = 600, n: int = 150, density: float = 0.20, noise_ratio: float = 0.10, eta: float = 0.2, iterations: int = 12, n_ranks: int = 4) -> float:
    """Reference implementation of run_rgdbek_comparison: chains
    _oracle_build_kaczmarz_instance, then runs the serial march both step by
    step (through _oracle_column_phase_update / _oracle_row_phase_update)
    and via the fused _oracle_march_rgdbek, cross-checking the two agree;
    runs the parallel march both step by step (through
    _oracle_parallel_column_phase / _oracle_parallel_row_phase) and via the
    fused _oracle_march_parallel_rgdbek, cross-checking the two agree; calls
    _oracle_sample_index_block on a fixed input as a standalone conformance
    check of the shared sampling primitive's own no-repeat/in-range
    contract; and finally calls _oracle_convergence_diagnostics on both
    marches' results."""
    # Standalone conformance check on the shared sampling primitive: every
    # step that samples (02's own function, and the locally-reproduced
    # copies inside 03/04/06/07) must draw distinct, in-range indices. A
    # broken sample_index_block oracle would violate this on this fixed
    # input regardless of which march consumes it.
    _sample_check = _oracle_sample_index_block(np.array([0.2, 0.3, 0.5]), 2, np.random.default_rng(0))
    if len(set(_sample_check.tolist())) != 2 or np.any(_sample_check < 0) or np.any(_sample_check >= 3):
        raise ValueError("sample_index_block oracle violated its own no-repeat/in-range contract")

    instance = _oracle_build_kaczmarz_instance(seed=seed, m=m, n=n, density=density, noise_ratio=noise_ratio)
    A, b = instance["A"], instance["b"]
    m_rows, n_cols = A.shape
    n_eta = max(1, round(eta * n_cols))
    m_eta = max(1, round(eta * m_rows))

    # --- serial march: step-by-step, then cross-checked against the fused oracle ---
    rng = np.random.default_rng(march_seed)
    col_sq_norms = np.sum(A ** 2, axis=0)
    row_sq_norms = np.sum(A ** 2, axis=1)
    x = np.zeros(n_cols)
    z = b.copy()
    for _k in range(iterations):
        z_next = _oracle_column_phase_update(A, z, col_sq_norms, n_eta, rng)
        x_next = _oracle_row_phase_update(A, b, z_next, x, row_sq_norms, m_eta, rng)
        x = x_next
        z = z_next

    fused_serial = _oracle_march_rgdbek(A, b, eta, iterations, march_seed)
    if not (np.allclose(x, fused_serial["x"], rtol=1e-9, atol=1e-12)
            and np.allclose(z, fused_serial["z"], rtol=1e-9, atol=1e-12)):
        raise ValueError("step-by-step serial march disagrees with the fused march (_oracle_march_rgdbek) beyond tolerance")

    # --- parallel march: step-by-step, then cross-checked against the fused oracle ---
    _row_block_bounds(A, n_ranks)  # validates n_ranks against A up front
    d_eta = max(1, round(eta * m_rows / n_ranks))
    rng_par = np.random.default_rng(march_seed)
    x_par = np.zeros(n_cols)
    z_par = b.copy()
    for _k in range(iterations):
        z_par_next = _oracle_parallel_column_phase(A, z_par, n_ranks, n_eta, rng_par)
        x_par_next = _oracle_parallel_row_phase(A, b, z_par_next, x_par, n_ranks, d_eta, rng_par)
        x_par = x_par_next
        z_par = z_par_next

    fused_parallel = _oracle_march_parallel_rgdbek(A, b, eta, iterations, march_seed, n_ranks=n_ranks)
    if not (np.allclose(x_par, fused_parallel["x"], rtol=1e-9, atol=1e-12)
            and np.allclose(z_par, fused_parallel["z"], rtol=1e-9, atol=1e-12)):
        raise ValueError("step-by-step parallel march disagrees with the fused march (_oracle_march_parallel_rgdbek) beyond tolerance")

    diagnostics = _oracle_convergence_diagnostics(
        A, b, x, x_par, z, z_par, instance["x_star"], instance["r_opt"]
    )
    return diagnostics["error_ratio"]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    Tiny configurations keep every case cheap for the mutation battery.
    >= 3 cases: a normal tiny-instance scenario, a boundary scenario at a
    different eta/iterations/n_ranks, and two invalid-input edge cases.
    Every m used here is divisible by the n_ranks in play.
    """
    shared_setup = (
        "import importlib\n"
        "import sys\n"
        "import numpy as np\n"
        "try:\n"
        "    _t5 = importlib.import_module('10_run_rgdbek_comparison')\n"
        "except ImportError:\n"
        "    _t5 = next((sys.modules[k] for k in sys.modules if '10_run_rgdbek_comparison' in k), None)\n"
        "if _t5 is None:\n"
        "    pass  # runner injected the module members directly into this namespace\n"
        "else:\n"
        "    run_rgdbek_comparison = _t5.run_rgdbek_comparison\n"
                        )
    guard_def = (
        "def _guard(thunk):\n"
        "    try:\n"
        "        thunk()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 2\n"
        "    except Exception:\n"
        "        return 1"
    )
    return [
        {
            # normal: tiny instance, few iterations, default n_ranks=4
            "setup": shared_setup,
            "call": "float(run_rgdbek_comparison(seed=1, march_seed=2, m=60, n=20, density=0.35, eta=0.3, iterations=4))",
            "gold_call": "float((_t5._oracle_run_rgdbek_comparison if _t5 is not None else _oracle_run_rgdbek_comparison)(seed=1, march_seed=2, m=60, n=20, density=0.35, eta=0.3, iterations=4))",
        },
        {
            # boundary: a different eta, a longer tiny run, and n_ranks=2
            "setup": shared_setup,
            "call": "float(run_rgdbek_comparison(seed=3, march_seed=4, m=80, n=25, density=0.35, eta=0.15, iterations=6, n_ranks=2))",
            "gold_call": "float((_t5._oracle_run_rgdbek_comparison if _t5 is not None else _oracle_run_rgdbek_comparison)(seed=3, march_seed=4, m=80, n=25, density=0.35, eta=0.15, iterations=6, n_ranks=2))",
        },
        {
            # edge: eta outside (0, 1] is invalid
            "setup": shared_setup + "\n" + guard_def,
            "call": "_guard(lambda: run_rgdbek_comparison(m=60, n=20, density=0.35, eta=0.0, iterations=4))",
            "gold_call": "_guard(lambda: (_t5._oracle_run_rgdbek_comparison if _t5 is not None else _oracle_run_rgdbek_comparison)(m=60, n=20, density=0.35, eta=0.0, iterations=4))",
        },
        {
            # edge: n_ranks greater than m
            "setup": shared_setup + "\n" + guard_def,
            "call": "_guard(lambda: run_rgdbek_comparison(m=60, n=20, density=0.35, eta=0.3, iterations=4, n_ranks=61))",
            "gold_call": "_guard(lambda: (_t5._oracle_run_rgdbek_comparison if _t5 is not None else _oracle_run_rgdbek_comparison)(m=60, n=20, density=0.35, eta=0.3, iterations=4, n_ranks=61))",
        },
    ]
