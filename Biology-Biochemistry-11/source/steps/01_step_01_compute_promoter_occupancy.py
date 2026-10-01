"""
Compute the stationary occupancy of each promoter state of a gene whose promoter switches silently and can also change state in the same event that releases a transcript.

The regulatory state of a promoter evolves as a finite continuous-time Markov chain, and in many genes, the act of transcribing moves the promoter to another state, so transcription events shape how long each state is occupied.

Returns
-------
np.ndarray: stationary probabilities of the N promoter states, shape (N,), summing to 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_promoter_occupancy(switch_rates: np.ndarray, transcription_rates: np.ndarray) -> np.ndarray:
    """Return the long-time probability of each promoter state.

    The promoter has ``N`` states labelled ``0..N-1``. For ``i != j``,
    ``switch_rates[i, j]`` is the rate of a transition from state ``i`` to
    state ``j`` that releases no transcript. For every ``i`` and ``j``,
    ``transcription_rates[i, j]`` is the rate at which a promoter in state
    ``i`` releases one transcript in an event that leaves it in state ``j``;
    the diagonal entry ``j = i`` is transcription without a change of state.
    All events are memoryless.

    Parameters
    ----------
    switch_rates : np.ndarray
        Array of shape ``(N, N)``, ``N >= 1``, with nonnegative finite
        entries and a zero diagonal.
    transcription_rates : np.ndarray
        Array of shape ``(N, N)`` with nonnegative finite entries.

    Returns
    -------
    np.ndarray
        Float array of shape ``(N,)`` holding the stationary probabilities of
        the promoter states; its entries sum to 1.

    Raises
    ------
    ValueError
        If either array is not square of the same shape ``(N, N)`` with
        ``N >= 1``, if an entry is negative or not finite, if
        ``switch_rates`` has a nonzero diagonal entry, or if the promoter
        chain does not have a unique stationary distribution.
    """
    return occupancy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_promoter_occupancy(switch_rates: np.ndarray, transcription_rates: np.ndarray) -> np.ndarray:
    """Reference implementation (left null vector of the promoter generator)."""
    import numpy as np

    silent = np.asarray(switch_rates, dtype=float)
    firing = np.asarray(transcription_rates, dtype=float)
    if silent.ndim != 2 or silent.shape[0] != silent.shape[1] or silent.shape[0] < 1:
        raise ValueError("switch_rates must be a square (N, N) array with N >= 1")
    if firing.shape != silent.shape:
        raise ValueError("transcription_rates must have the shape of switch_rates")
    if not (np.all(np.isfinite(silent)) and np.all(np.isfinite(firing))):
        raise ValueError("rates must be finite")
    if np.any(silent < 0.0) or np.any(firing < 0.0):
        raise ValueError("rates must be nonnegative")
    if np.any(np.diag(silent) != 0.0):
        raise ValueError("switch_rates must have a zero diagonal")
    n = silent.shape[0]
    # Only state-changing events enter the promoter generator.  Diagonal
    # transcription events release a transcript but leave the promoter state
    # unchanged, so they cancel from the promoter-state balance exactly.
    state_change_rates = silent + firing
    np.fill_diagonal(state_change_rates, 0.0)
    generator = state_change_rates.copy()
    np.fill_diagonal(generator, -state_change_rates.sum(axis=1))

    if n == 1:
        return np.ones(1, dtype=float)

    # A finite chain has one stationary distribution exactly when it has one
    # closed communicating class.  Check that graph property directly instead
    # of deciding it from an ill-conditioned numerical rank threshold.
    reach = (state_change_rates > 0.0) | np.eye(n, dtype=bool)
    for k in range(n):
        reach |= reach[:, k, None] & reach[None, k, :]
    assigned = np.zeros(n, dtype=bool)
    closed_classes = 0
    for i in range(n):
        if assigned[i]:
            continue
        component = reach[i] & reach[:, i]
        assigned |= component
        if not np.any((state_change_rates[component] > 0.0)[:, ~component]):
            closed_classes += 1
    if closed_classes != 1:
        raise ValueError("the promoter chain has no unique stationary distribution")

    # Uniform rate rescaling cannot change a stationary distribution.  Scale
    # the balance equations before replacing one dependent equation by the
    # normalization condition, so very slow but otherwise ordinary chains do
    # not lose their state-balance information next to the unit-sum row.
    rate_scale = float(np.max(np.abs(generator)))
    scaled_generator = generator / rate_scale
    system = scaled_generator.T.copy()
    system[-1, :] = 1.0
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    occupancy = np.linalg.solve(system, rhs)
    return occupancy.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _digest(x):\n"
        "    a = np.asarray(x, dtype=float)\n"
        "    if a.ndim != 1:\n"
        "        return -1.0\n"
        "    w = 2.0 + np.cos(np.arange(a.size, dtype=float))\n"
        "    r = np.sqrt(np.arange(1.0, a.size + 1.0))\n"
        "    return float(a.size + 1.0e3 * np.dot(w, a) + 1.0e2 * np.dot(r, a))\n"
    )
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
    invariant = (
        "import numpy as np\n"
        "def _occupancy_error(x, expected):\n"
        "    a = np.asarray(x, dtype=float)\n"
        "    target = np.asarray(expected, dtype=float)\n"
        "    if a.shape != target.shape:\n"
        "        return 1.0e6\n"
        "    return float(max(np.max(np.abs(a - target)), abs(a.sum() - 1.0)))\n"
    )
    return [
        {
            "setup": digest,
            "call": "_digest(compute_promoter_occupancy(np.array([[0.0, 0.3, 0.0], [0.7, 0.0, 1.2], [0.2, 0.9, 0.0]]), np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.8, 0.0, 2.6]])))",
            "gold_call": "_digest(_oracle_compute_promoter_occupancy(np.array([[0.0, 0.3, 0.0], [0.7, 0.0, 1.2], [0.2, 0.9, 0.0]]), np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.8, 0.0, 2.6]])))",
        },
        {
            "setup": digest,
            "call": "_digest(compute_promoter_occupancy(np.array([[0.0, 0.05, 0.0, 0.0], [0.4, 0.0, 0.6, 0.0], [0.0, 0.3, 0.0, 0.8], [0.0, 0.0, 0.25, 0.0]]), np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0], [2.0, 0.0, 1.5, 4.0]])))",
            "gold_call": "_digest(_oracle_compute_promoter_occupancy(np.array([[0.0, 0.05, 0.0, 0.0], [0.4, 0.0, 0.6, 0.0], [0.0, 0.3, 0.0, 0.8], [0.0, 0.0, 0.25, 0.0]]), np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0], [2.0, 0.0, 1.5, 4.0]])))",
        },
        {
            "setup": digest,
            "call": "_digest(compute_promoter_occupancy(np.array([[0.0]]), np.array([[3.0]])))",
            "gold_call": "_digest(_oracle_compute_promoter_occupancy(np.array([[0.0]]), np.array([[3.0]])))",
        },
        {
            "setup": invariant,
            "call": "_occupancy_error(compute_promoter_occupancy(np.array([[0.0, 1.0e-14], [1.0e-14, 0.0]]), np.eye(2)), np.array([0.5, 0.5]))",
            "gold_call": "_occupancy_error(_oracle_compute_promoter_occupancy(np.array([[0.0, 1.0e-14], [1.0e-14, 0.0]]), np.eye(2)), np.array([0.5, 0.5]))",
        },
        {
            "setup": invariant,
            "call": "_occupancy_error(compute_promoter_occupancy(np.array([[0.0, 2.0], [3.0, 0.0]]), np.array([[5.0, 0.0], [0.0, 7.0]])), np.array([0.6, 0.4]))",
            "gold_call": "_occupancy_error(_oracle_compute_promoter_occupancy(np.array([[0.0, 2.0], [3.0, 0.0]]), np.array([[5.0, 0.0], [0.0, 7.0]])), np.array([0.6, 0.4]))",
        },
        {
            "setup": invariant,
            "call": "_occupancy_error(compute_promoter_occupancy(np.array([[0.0, 2.0e-14], [3.0e-14, 0.0]]), np.array([[5.0, 0.0], [0.0, 7.0]])), np.array([0.6, 0.4]))",
            "gold_call": "_occupancy_error(_oracle_compute_promoter_occupancy(np.array([[0.0, 2.0e-14], [3.0e-14, 0.0]]), np.array([[5.0, 0.0], [0.0, 7.0]])), np.array([0.6, 0.4]))",
        },
        {
            "setup": digest,
            "call": "_digest(compute_promoter_occupancy(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]])))",
            "gold_call": "_digest(_oracle_compute_promoter_occupancy(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]])))",
        },
        {
            "setup": digest,
            "call": "_digest(compute_promoter_occupancy(np.zeros((3, 3)), np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.5], [0.7, 0.0, 3.0]])))",
            "gold_call": "_digest(_oracle_compute_promoter_occupancy(np.zeros((3, 3)), np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.5], [0.7, 0.0, 3.0]])))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_promoter_occupancy(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), np.eye(3)))",
            "gold_call": "_status(lambda: _oracle_compute_promoter_occupancy(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), np.eye(3)))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_promoter_occupancy(np.array([[0.5, 1.0], [1.0, 0.0]]), np.eye(2)))",
            "gold_call": "_status(lambda: _oracle_compute_promoter_occupancy(np.array([[0.5, 1.0], [1.0, 0.0]]), np.eye(2)))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_promoter_occupancy(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[1.0, -0.2], [0.0, 1.0]])))",
            "gold_call": "_status(lambda: _oracle_compute_promoter_occupancy(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[1.0, -0.2], [0.0, 1.0]])))",
        },
    ]
