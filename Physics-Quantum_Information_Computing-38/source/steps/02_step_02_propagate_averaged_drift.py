"""
Propagate the ensemble-averaged Bloch vector of the monitored qubit and return the averaged homodyne drift together with its exact parameter derivatives on the left ends of a uniform sampling grid.

Replacing each record's conditional state by the deterministic averaged state turns the record drift into a known function of time and of the parameters, whose sensitivities to the drive and to the measurement strength drive both the estimator and its information.

Returns
-------
np.ndarray: shape (n_intervals, 3), rows [h, dh/dalpha, dh/dbeta] at t_j = j * delta.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def propagate_averaged_drift(
    alpha: float,
    beta: float,
    bloch0: 'np.ndarray',
    delta: float,
    n_intervals: int,
    drive_generator: 'np.ndarray',
    dephasing_generator: 'np.ndarray',
) -> 'np.ndarray':
    """Return the averaged drift and its derivatives at the sampling-interval starts.

    The qubit has Hamiltonian ``H = (alpha / 2) sigma_x`` and measurement
    operator ``L = beta sigma_z``. Its ensemble-averaged state has Bloch
    vector ``u(t) = (1, x, y, z)`` obeying
    ``d u / d t = (alpha * drive_generator + beta**2 * dephasing_generator) u``
    with ``u(0) = (1, bloch0)``, and the averaged drift of the homodyne
    record is ``h(t) = Tr(L rho(t) + rho(t) L^dag)``. Row ``j`` of the
    result, ``j = 0, ..., n_intervals - 1``, holds
    ``[h(t_j), dh/dalpha (t_j), dh/dbeta (t_j)]`` at ``t_j = j * delta``,
    where both derivatives are total derivatives at fixed ``bloch0``.
    On the supported domain below, the accuracy target is a maximum-entry
    error of ``1e-11 * max(1, max(abs(exact_drift)))`` for the whole returned
    array. The derivatives are evaluated from the differentiated dynamics;
    this is a floating-point accuracy target, not exact arithmetic.

    Both generators must have a zero first row and entries of absolute
    value at most 2. For ``G = alpha * drive_generator + beta**2 *
    dephasing_generator``, the largest eigenvalue of the symmetric part of
    ``G[1:, 1:]`` must be at most ``1e-12``. This contract supports
    contractive Bloch dynamics, including an affine forcing column, and
    excludes exponentially growing homogeneous dynamics. The last returned
    sample time ``(n_intervals - 1) * delta`` must not exceed 4.

    Parameters
    ----------
    alpha : float
        Finite real drive parameter, ``-5 <= alpha <= 5``.
    beta : float
        Finite real measurement parameter, ``0 <= beta <= 3``.
    bloch0 : np.ndarray
        Initial Bloch vector ``(x, y, z)`` with Euclidean norm at most
        ``1 + 1e-12``.
    delta : float
        Sampling interval in ``[1e-4, 1]``.
    n_intervals : int
        Integer number of sampling intervals in ``[1, 128]``.
    drive_generator : np.ndarray
        Finite real ``(4, 4)`` matrix satisfying the bounds above.
    dephasing_generator : np.ndarray
        Finite real ``(4, 4)`` matrix satisfying the bounds above.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_intervals, 3)``.

    Raises
    ------
    ValueError
        If ``alpha``, ``beta`` or ``delta`` is not a finite real number,
        a parameter, sample count or time exceeds its supported range,
        ``bloch0`` is not a finite real length-3 vector of norm at most
        ``1 + 1e-12``, either generator fails its real-array, shape, entry,
        zero-first-row or contractivity requirements, or a numerical
        intermediate/result is nonfinite. Non-real entries are rejected.
        Boolean scalar parameters and sample counts are rejected.
    """
    return drift

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm
def _oracle_propagate_averaged_drift(
    alpha: float,
    beta: float,
    bloch0: 'np.ndarray',
    delta: float,
    n_intervals: int,
    drive_generator: 'np.ndarray',
    dephasing_generator: 'np.ndarray',
) -> 'np.ndarray':
    """Reference implementation (block matrix exponential for the Frechet derivatives)."""
    import numpy as np
    from scipy.linalg import expm

    def _is_number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            return False
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError, OverflowError):
            return False

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    def _as_real(value, shape, name):
        try:
            mat = np.asarray(value, dtype=complex)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be a finite real array") from None
        if mat.shape != shape or not np.all(np.isfinite(mat)) or np.any(mat.imag != 0.0):
            raise ValueError(f"{name} must be a finite real array of shape {shape}")
        return mat.real.copy()

    if not (_is_number(alpha) and _is_number(beta) and _is_number(delta)):
        raise ValueError("alpha, beta and delta must be finite real numbers")
    if not (-5.0 <= alpha <= 5.0 and 0.0 <= beta <= 3.0 and 1e-4 <= delta <= 1.0):
        raise ValueError("alpha, beta or delta is outside the supported numerical range")
    if not (_is_integer(n_intervals) and 1 <= n_intervals <= 128):
        raise ValueError("n_intervals must be an integer in [1, 128]")
    if (int(n_intervals) - 1) * float(delta) > 4.0:
        raise ValueError("the last returned sample time must be <= 4")
    start = _as_real(bloch0, (3,), "bloch0")
    if np.linalg.norm(start) > 1.0 + 1e-12:
        raise ValueError("bloch0 must lie in the Bloch ball")
    drive = _as_real(drive_generator, (4, 4), "drive_generator")
    damp = _as_real(dephasing_generator, (4, 4), "dephasing_generator")
    if any(np.any(np.abs(mat) > 2.0) or np.any(mat[0] != 0.0) for mat in (drive, damp)):
        raise ValueError("generators must have zero first rows and entries within [-2, 2]")

    a, b = float(alpha), float(beta)
    gen = a * drive + b * b * damp
    if not np.all(np.isfinite(gen)):
        raise ValueError("nonfinite averaged generator")
    symmetric = 0.5 * (gen[1:, 1:] + gen[1:, 1:].T)
    if np.linalg.eigvalsh(symmetric)[-1] > 1e-12:
        raise ValueError("the homogeneous Bloch generator must be contractive")
    # exp(t M) of the block upper-triangular M carries exp(t G) on the diagonal
    # and the Frechet derivatives of exp(t G) along dG/dalpha, dG/dbeta above it;
    # the same expression is regular when G is defective (critical damping).
    block = np.zeros((12, 12))
    block[0:4, 0:4] = gen
    block[0:4, 4:8] = drive
    block[0:4, 8:12] = 2.0 * b * damp
    block[4:8, 4:8] = gen
    block[8:12, 8:12] = gen
    if not np.all(np.isfinite(block)):
        raise ValueError("nonfinite sensitivity generator")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            step = expm(float(delta) * block)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError("failed finite propagation") from exc
    if not np.all(np.isfinite(step)):
        raise ValueError("nonfinite propagator")
    state = np.zeros((12, 2))
    u0 = np.concatenate([[1.0], start])
    state[4:8, 0] = u0
    state[8:12, 1] = u0
    drift = np.empty((int(n_intervals), 3))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for j in range(int(n_intervals)):
                z_value = state[7, 0]
                dz_alpha = state[3, 0]
                dz_beta = state[3, 1]
                # h = 2 beta z; the beta-derivative also acts on L itself.
                drift[j] = (2.0 * b * z_value, 2.0 * b * dz_alpha, 2.0 * z_value + 2.0 * b * dz_beta)
                if j + 1 < int(n_intervals):
                    state = step @ state
                    if not np.all(np.isfinite(state)):
                        raise ValueError("nonfinite propagated state or sensitivity")
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("nonfinite drift evaluation") from exc
    if not np.all(np.isfinite(drift)):
        raise ValueError("nonfinite averaged drift or derivative")
    return drift

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    helpers = (
        "import numpy as np\n"
        "D = np.zeros((4, 4)); D[2, 3] = -1.0; D[3, 2] = 1.0\n"
        "K = np.zeros((4, 4)); K[1, 1] = -2.0; K[2, 2] = -2.0\n"
        "D2 = np.array([[0.0, 0.0, 0.0, 0.0], [0.1, -0.2, 0.3, 0.0],"
        " [0.0, -0.4, 0.1, -0.9], [0.2, 0.0, 0.8, -0.3]])\n"
        "K2 = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, -1.1, 0.2, 0.1],"
        " [0.3, 0.0, -0.7, 0.0], [-0.5, 0.1, 0.0, -0.9]])\n"
        "def _dsig(a, n):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (n, 3):\n"
        "        return -1.0\n"
        "    flat = a.ravel()\n"
        "    weights = np.cos(np.arange(1, flat.size + 1, dtype=float))\n"
        "    return float(np.sum(np.abs(flat)) + np.sum(flat * weights))\n"
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
            "setup": helpers,
            "call": "_dsig(propagate_averaged_drift(-1.3, 0.7, np.array([0.0, -0.8, 0.6]), 0.4, 10, D, K), 10) / 10.0",
            "gold_call": "_dsig(_oracle_propagate_averaged_drift(-1.3, 0.7, np.array([0.0, -0.8, 0.6]), 0.4, 10, D, K), 10) / 10.0",
        },
        {
            "setup": helpers,
            "call": "_dsig(propagate_averaged_drift(0.25, 0.5, np.array([0.3, 0.5, 0.7]), 0.5, 8, D, K), 8) / 10.0",
            "gold_call": "_dsig(_oracle_propagate_averaged_drift(0.25, 0.5, np.array([0.3, 0.5, 0.7]), 0.5, 8, D, K), 8) / 10.0",
        },
        {
            "setup": helpers,
            "call": "_dsig(propagate_averaged_drift(1.1, 0.0, np.array([0.2, -0.6, 0.7]), 0.3, 9, D, K), 9) / 10.0",
            "gold_call": "_dsig(_oracle_propagate_averaged_drift(1.1, 0.0, np.array([0.2, -0.6, 0.7]), 0.3, 9, D, K), 9) / 10.0",
        },
        {
            "setup": helpers,
            "call": "_dsig(propagate_averaged_drift(0.0, 0.9, np.array([0.5, 0.6, -0.6]), 0.25, 12, D, K), 12) / 10.0",
            "gold_call": "_dsig(_oracle_propagate_averaged_drift(0.0, 0.9, np.array([0.5, 0.6, -0.6]), 0.25, 12, D, K), 12) / 10.0",
        },
        {
            "setup": helpers,
            "call": "_dsig(propagate_averaged_drift(0.8, 1.2, np.array([0.1, 0.2, -0.4]), 0.2, 6, D2, K2), 6)",
            "gold_call": "_dsig(_oracle_propagate_averaged_drift(0.8, 1.2, np.array([0.1, 0.2, -0.4]), 0.2, 6, D2, K2), 6)",
        },
        {
            "setup": helpers,
            "call": "float(propagate_averaged_drift(-2.2, 1.4, np.array([0.0, 0.6, 0.8]), 0.35, 11, D, K)[6, 2])",
            "gold_call": "float(_oracle_propagate_averaged_drift(-2.2, 1.4, np.array([0.0, 0.6, 0.8]), 0.35, 11, D, K)[6, 2])",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(1.0, -0.2, np.array([0.0, 0.0, 1.0]), 0.4, 5, D, K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(1.0, -0.2, np.array([0.0, 0.0, 1.0]), 0.4, 5, D, K))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(1.0, 0.5, np.array([0.0, 0.9, 0.8]), 0.4, 5, D, K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(1.0, 0.5, np.array([0.0, 0.9, 0.8]), 0.4, 5, D, K))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(1.0, 0.5, np.array([0.0, 0.0, 1.0]), 0.4, 0, D, K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(1.0, 0.5, np.array([0.0, 0.0, 1.0]), 0.4, 0, D, K))",
        },
        {
            "setup": helpers,
            "call": "_dsig(propagate_averaged_drift(5., 0., [0, 0, 1], 4/127, 128, D, K), 128) / 128",
            "gold_call": "_dsig(_oracle_propagate_averaged_drift(5., 0., [0, 0, 1], 4/127, 128, D, K), 128) / 128",
        },
        {
            "setup": helpers,
            "call": "_dsig(propagate_averaged_drift(-5., 0., [0, 0, 1], 1., 5, D, K), 5) / 5",
            "gold_call": "_dsig(_oracle_propagate_averaged_drift(-5., 0., [0, 0, 1], 1., 5, D, K), 5) / 5",
        },
        {
            "setup": helpers,
            "call": "_dsig(propagate_averaged_drift(5., np.sqrt(5.), [0, 0, 1], 4/127, 128, D, K), 128) / 128",
            "gold_call": "_dsig(_oracle_propagate_averaged_drift(5., np.sqrt(5.), [0, 0, 1], 4/127, 128, D, K), 128) / 128",
        },
        {
            "setup": helpers,
            "call": "_dsig(propagate_averaged_drift(-5., np.sqrt(5.), [0, .6, .8], 1., 5, D, K), 5) / 5",
            "gold_call": "_dsig(_oracle_propagate_averaged_drift(-5., np.sqrt(5.), [0, .6, .8], 1., 5, D, K), 5) / 5",
        },
        {
            "setup": helpers,
            "call": "_dsig(propagate_averaged_drift(5., 3., [0, .6, .8], 1., 5, D, K), 5) / 5",
            "gold_call": "_dsig(_oracle_propagate_averaged_drift(5., 3., [0, .6, .8], 1., 5, D, K), 5) / 5",
        },
        {
            "setup": helpers,
            "call": "_dsig(propagate_averaged_drift(0., 3., [0, 0, 1], 1e-4, 1, D, K), 1)",
            "gold_call": "_dsig(_oracle_propagate_averaged_drift(0., 3., [0, 0, 1], 1e-4, 1, D, K), 1)",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(1e7, 0., [0, 0, 1], .4, 2, D, K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(1e7, 0., [0, 0, 1], .4, 2, D, K))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(0., 1e155, [0, 0, 1], .4, 2, D, K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(0., 1e155, [0, 0, 1], .4, 2, D, K))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(0., 1., [0, 1j, 0], .4, 2, D, K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(0., 1., [0, 1j, 0], .4, 2, D, K))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(0., 1., np.array([0, 1j, 0]), .4, 2, D, K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(0., 1., np.array([0, 1j, 0]), .4, 2, D, K))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(np.nextafter(5., np.inf), 1., [0, 0, 1], .4, 2, D, K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(np.nextafter(5., np.inf), 1., [0, 0, 1], .4, 2, D, K))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(np.nextafter(-5., -np.inf), 1., [0, 0, 1], .4, 2, D, K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(np.nextafter(-5., -np.inf), 1., [0, 0, 1], .4, 2, D, K))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(1., np.nextafter(3., np.inf), [0, 0, 1], .4, 2, D, K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(1., np.nextafter(3., np.inf), [0, 0, 1], .4, 2, D, K))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(1., 1., [0, 0, 1], np.nextafter(1e-4, 0.), 2, D, K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(1., 1., [0, 0, 1], np.nextafter(1e-4, 0.), 2, D, K))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(1., 1., [0, 0, 1], np.nextafter(1., np.inf), 2, D, K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(1., 1., [0, 0, 1], np.nextafter(1., np.inf), 2, D, K))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(1., 1., [0, 0, 1], .01, 129, D, K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(1., 1., [0, 0, 1], .01, 129, D, K))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(1., 1., [0, 0, 1], np.nextafter(.5, np.inf), 9, D, K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(1., 1., [0, 0, 1], np.nextafter(.5, np.inf), 9, D, K))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(10**400, 1., [0, 0, 1], .4, 2, D, K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(10**400, 1., [0, 0, 1], .4, 2, D, K))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(1., 1., [0, 0, 1], .4, 2, D, -K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(1., 1., [0, 0, 1], .4, 2, D, -K))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(1., 1., [0, 0, 1], .4, 2, D, K * np.nextafter(1., np.inf)))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(1., 1., [0, 0, 1], .4, 2, D, K * np.nextafter(1., np.inf)))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(1., 1., [0, 0, 1], .4, 2, D + np.diag([.1, 0, 0, 0]), K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(1., 1., [0, 0, 1], .4, 2, D + np.diag([.1, 0, 0, 0]), K))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(1., 1., [0, 0, 1], .4, 2, D + 1j, K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(1., 1., [0, 0, 1], .4, 2, D + 1j, K))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: propagate_averaged_drift(1., 1., [0, 0, 1], .4, 2, D * np.nan, K))",
            "gold_call": "_status(lambda: _oracle_propagate_averaged_drift(1., 1., [0, 0, 1], .4, 2, D * np.nan, K))",
        },
    ]
