"""
Compose every earlier step to estimate the drive and measurement parameters of a monitored qubit from repeated homodyne records and return the squared joint Studentized distance of the estimate from a nominal calibration.

Repeated runs from a known preparation give a many-trajectory regime in which the averaged-state contrast estimate is asymptotically normal with an empirically estimable sandwich covariance, so a nominal calibration can be tested against the records without filtering any of them.

Returns
-------
float: the squared joint Studentized distance D2 of the estimate from the nominal calibration.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assess_nominal_calibration(
    records: 'np.ndarray',
    delta: float = 0.4,
    bloch0: tuple = (0.0, -0.8, 0.6),
    theta_nominal: tuple = (-1.5, 0.55),
    alpha_bounds: tuple = (-5.0, 5.0),
    beta_bounds: tuple = (0.05, 3.0),
    grid_spacing: float = 0.05,
    tolerance: float = 1e-10,
) -> float:
    """Return the squared joint Studentized distance of the estimate from the nominal.

    ``records[i, j]`` is run ``i``'s integrated homodyne record ``Y_i`` at
    ``t = (j + 1) * delta`` (``Y_i(0) = 0``) for a qubit with Hamiltonian
    ``H = (alpha / 2) sigma_x``, measurement operator ``L = beta sigma_z``
    and initial Bloch vector ``bloch0``. Build the averaged-state drift of
    this model from the Lindblad generators of ``sigma_x / 2`` (no jump
    operators) and of the single jump operator ``sigma_z`` (zero
    Hamiltonian); maximize the contrast of the ensemble-mean record over the
    box ``alpha_bounds x beta_bounds``, starting from the grid candidates
    (spacing ``grid_spacing``, at most 8, separation 0.15) and refining them
    to ``tolerance``; estimate the sandwich covariance from the records at
    the maximizer; and return the squared joint Studentized distance of
    the maximizer from ``theta_nominal``. The defaults reproduce the problem
    statement apart from the records.

    Parameters
    ----------
    records : np.ndarray
        Finite array of shape ``(N, n)`` with ``N >= 2``, ``1 <= n <= 128``,
        and ``(n - 1) * delta <= 4``, as required by the drift step.
    delta : float
        Sampling interval in ``[1e-4, 1]``.
    bloch0 : tuple
        Initial Bloch vector ``(x, y, z)`` inside the Bloch ball.
    theta_nominal : tuple
        Nominal ``(alpha, beta)``.
    alpha_bounds, beta_bounds : tuple
        ``(low, high)`` search box contained in ``[-5, 5] x [0, 3]``.
    grid_spacing : float
        Candidate-grid spacing dividing both ranges, subject to the
        supported spacing and grid-size limits of the candidate step.
    tolerance : float
        Positive accuracy of the refined maximizers.

    Returns
    -------
    float
        The squared joint Studentized distance ``D2``.

    Raises
    ------
    ValueError
        If ``records`` is not a finite ``(N, n)`` array with ``N >= 2`` and
        ``1 <= n <= 128``, the search box exceeds the supported parameter
        domain, or any stage rejects its input (including the drift step's
        sampling-time and real Bloch-vector requirements).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_assess_nominal_calibration(
    records: 'np.ndarray',
    delta: float = 0.4,
    bloch0: tuple = (0.0, -0.8, 0.6),
    theta_nominal: tuple = (-1.5, 0.55),
    alpha_bounds: tuple = (-5.0, 5.0),
    beta_bounds: tuple = (0.05, 3.0),
    grid_spacing: float = 0.05,
    tolerance: float = 1e-10,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    try:
        data = np.asarray(records, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("records must be a numeric array") from None
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 1 or not np.all(np.isfinite(data)):
        raise ValueError("records must be a finite (N, n) array with N >= 2")
    def _bounded_pair(bounds, lower, upper):
        try:
            pair = np.asarray(bounds, dtype=complex)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("search bounds must be finite real pairs") from None
        if pair.shape != (2,) or not np.all(np.isfinite(pair)) or np.any(pair.imag != 0.0):
            raise ValueError("search bounds must be finite real pairs")
        low, high = pair.real
        if not lower <= low < high <= upper:
            raise ValueError("search box must lie within [-5, 5] x [0, 3]")
        return (float(low), float(high))

    alpha_bounds = _bounded_pair(alpha_bounds, -5.0, 5.0)
    beta_bounds = _bounded_pair(beta_bounds, 0.0, 3.0)
    n_records, n_intervals = data.shape
    increments = np.diff(np.concatenate([np.zeros((n_records, 1)), data], axis=1), axis=1)
    mean_increments = increments.mean(axis=0)
    start = bloch0

    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    drive = _oracle_build_bloch_generator(0.5 * sigma_x, np.zeros((0, 2, 2), dtype=complex))
    dephasing = _oracle_build_bloch_generator(np.zeros((2, 2), dtype=complex), np.array([sigma_z]))

    def _drift_fn(alpha, beta):
        return _oracle_propagate_averaged_drift(
            alpha, beta, start, delta, n_intervals, drive, dephasing
        )

    def _model_fn(alpha, beta):
        return _oracle_evaluate_contrast_model(alpha, beta, mean_increments, delta, _drift_fn)

    def _contrast_fn(alpha, beta):
        return float(_model_fn(alpha, beta)[0])

    starts = _oracle_locate_contrast_candidates(
        _contrast_fn, alpha_bounds, beta_bounds, grid_spacing, 8, 0.15
    )
    theta_hat = _oracle_maximize_contrast(_model_fn, starts, alpha_bounds, beta_bounds, tolerance)
    covariance = _oracle_estimate_sandwich_covariance(
        theta_hat[0], theta_hat[1], increments, delta, _drift_fn
    )
    statistics = _oracle_compute_studentized_statistics(
        theta_hat, np.asarray(theta_nominal, dtype=float), covariance, n_records
    )
    return float(statistics[4])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only tests on synthetic record sets."""
    maker = (
        "import numpy as np\n"
        "from scipy.linalg import expm\n"
        "def _records(alpha, beta, y0, z0, delta, n, runs, seed):\n"
        "    gen = np.array([[-2.0 * beta ** 2, -alpha], [alpha, 0.0]])\n"
        "    step = expm(0.5 * delta * gen)\n"
        "    v = np.array([y0, z0], dtype=float)\n"
        "    mids = []\n"
        "    for j in range(2 * n):\n"
        "        v = step @ v\n"
        "        if j % 2 == 0:\n"
        "            mids.append(2.0 * beta * v[1])\n"
        "    rng = np.random.default_rng(seed)\n"
        "    wobble = rng.normal(0.0, 0.15, size=(runs, 1)) * np.sin(np.arange(1, n + 1))\n"
        "    noise = rng.normal(0.0, np.sqrt(delta), size=(runs, n))\n"
        "    inc = np.array(mids) * delta + noise + wobble * delta\n"
        "    return np.round(np.cumsum(inc, axis=1), 3)\n"
    )
    status = (
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
            "setup": maker + "R = _records(-1.2, 0.6, -0.6, 0.8, 0.5, 8, 16, 7)\n",
            "call": "assess_nominal_calibration(R, 0.5, (0.0, -0.6, 0.8), (-1.0, 0.5), (-3.0, 3.0), (0.1, 2.0), 0.1)",
            "gold_call": "_oracle_assess_nominal_calibration(R, 0.5, (0.0, -0.6, 0.8), (-1.0, 0.5), (-3.0, 3.0), (0.1, 2.0), 0.1)",
        },
        {
            "setup": maker + "R = _records(0.9, 0.8, 0.5, 0.5, 0.4, 10, 20, 11)\n",
            "call": "assess_nominal_calibration(R, 0.4, (0.3, 0.5, 0.5), (0.9, 0.8), (-2.5, 2.5), (0.1, 1.9), 0.1)",
            "gold_call": "_oracle_assess_nominal_calibration(R, 0.4, (0.3, 0.5, 0.5), (0.9, 0.8), (-2.5, 2.5), (0.1, 1.9), 0.1)",
        },
        {
            "setup": maker + "R = _records(-1.6, 0.7, -0.8, 0.6, 0.4, 10, 24, 3)\n",
            "call": "assess_nominal_calibration(R)",
            "gold_call": "_oracle_assess_nominal_calibration(R)",
        },
        {
            "setup": maker + status + "R = _records(-1.2, 0.6, -0.6, 0.8, 0.5, 8, 16, 7)\n",
            "call": "_status(lambda: assess_nominal_calibration(R[:1]))",
            "gold_call": "_status(lambda: _oracle_assess_nominal_calibration(R[:1]))",
        },
        {
            "setup": maker + status + "R = _records(-1.2, 0.6, -0.6, 0.8, 0.5, 8, 16, 7)\n",
            "call": "_status(lambda: assess_nominal_calibration(R, 0.5, (0.0, -0.6, 0.8), (-1.0, 0.5), (-3.0, 3.0), (-0.1, 2.0), 0.1))",
            "gold_call": "_status(lambda: _oracle_assess_nominal_calibration(R, 0.5, (0.0, -0.6, 0.8), (-1.0, 0.5), (-3.0, 3.0), (-0.1, 2.0), 0.1))",
        },
        {
            "setup": maker + status,
            "call": "_status(lambda: assess_nominal_calibration(np.zeros((2, 10)), alpha_bounds=(-5.1, 5.)))",
            "gold_call": "_status(lambda: _oracle_assess_nominal_calibration(np.zeros((2, 10)), alpha_bounds=(-5.1, 5.)))",
        },
        {
            "setup": maker + status,
            "call": "_status(lambda: assess_nominal_calibration(np.zeros((2, 10)), beta_bounds=(0., 3.1)))",
            "gold_call": "_status(lambda: _oracle_assess_nominal_calibration(np.zeros((2, 10)), beta_bounds=(0., 3.1)))",
        },
        {
            "setup": maker + status,
            "call": "_status(lambda: assess_nominal_calibration(np.zeros((2, 10)), delta=1.1))",
            "gold_call": "_status(lambda: _oracle_assess_nominal_calibration(np.zeros((2, 10)), delta=1.1))",
        },
        {
            "setup": maker + status,
            "call": "_status(lambda: assess_nominal_calibration(np.zeros((2, 10)), delta=.5))",
            "gold_call": "_status(lambda: _oracle_assess_nominal_calibration(np.zeros((2, 10)), delta=.5))",
        },
        {
            "setup": maker + status,
            "call": "_status(lambda: assess_nominal_calibration(np.zeros((2, 10)), bloch0=(0, 1j, 0)))",
            "gold_call": "_status(lambda: _oracle_assess_nominal_calibration(np.zeros((2, 10)), bloch0=(0, 1j, 0)))",
        },
    ]
