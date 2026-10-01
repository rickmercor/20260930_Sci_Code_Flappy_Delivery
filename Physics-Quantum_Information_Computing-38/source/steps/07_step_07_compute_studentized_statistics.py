"""
Studentize the estimation error against a reference parameter value with a covariance estimate, componentwise and jointly, and form the squared joint distance.

A consistent covariance estimate for the scaled estimation error turns the asymptotic normality of the estimator into standard errors, componentwise Studentized statistics and a joint statistic whose squared norm has a chi-squared reference law.

Returns
-------
np.ndarray: shape (5,), [T_alpha, T_beta, J_1, J_2, D2].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_studentized_statistics(
    theta_hat: 'np.ndarray',
    theta_ref: 'np.ndarray',
    covariance: 'np.ndarray',
    n_records: int,
) -> 'np.ndarray':
    """Return the componentwise and joint Studentized errors and the squared distance.

    With ``d = sqrt(n_records) * (theta_hat - theta_ref)`` and ``V =
    covariance`` (the covariance of ``sqrt(N)(theta_hat - theta)``), return
    ``[T_1, T_2, J_1, J_2, D2]`` where ``T_k = d_k / sqrt(V_kk)``,
    ``(J_1, J_2) = V^{-1/2} d`` with ``V^{-1/2}`` the symmetric
    positive-definite inverse square root of ``V``, and ``D2 = J_1**2 + J_2**2``.

    Parameters
    ----------
    theta_hat, theta_ref : np.ndarray
        Finite length-2 parameter vectors.
    covariance : np.ndarray
        Finite ``(2, 2)`` symmetric positive-definite matrix.
    n_records : int
        Number of records ``N >= 1``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(5,)``.

    Raises
    ------
    ValueError
        If either parameter vector is not a finite length-2 vector,
        ``covariance`` is not a finite ``(2, 2)`` matrix whose entries match
        its transpose to ``1e-12`` times its largest absolute entry, or it is
        not positive definite, or ``n_records`` is not an integer ``>= 1``
        (booleans are rejected).
    """
    return statistics

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_studentized_statistics(
    theta_hat: 'np.ndarray',
    theta_ref: 'np.ndarray',
    covariance: 'np.ndarray',
    n_records: int,
) -> 'np.ndarray':
    """Reference implementation (eigen-decomposition for the symmetric root)."""
    import numpy as np

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    def _vector(value, name):
        try:
            vec = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be numeric") from None
        if vec.shape != (2,) or not np.all(np.isfinite(vec)):
            raise ValueError(f"{name} must be a finite length-2 vector")
        return vec

    estimate = _vector(theta_hat, "theta_hat")
    reference = _vector(theta_ref, "theta_ref")
    if not (_is_integer(n_records) and n_records >= 1):
        raise ValueError("n_records must be an integer >= 1")
    try:
        cov = np.asarray(covariance, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("covariance must be numeric") from None
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        raise ValueError("covariance must be a finite (2, 2) matrix")
    scale = np.max(np.abs(cov))
    if np.max(np.abs(cov - cov.T)) > 1e-12 * scale:
        raise ValueError("covariance must be symmetric")
    cov = 0.5 * (cov + cov.T)
    values, vectors = np.linalg.eigh(cov)
    if not values[0] > 0.0:
        raise ValueError("covariance must be positive definite")

    scaled = np.sqrt(float(n_records)) * (estimate - reference)
    marginal = scaled / np.sqrt(np.diag(cov))
    inv_root = vectors @ np.diag(values ** -0.5) @ vectors.T
    joint = inv_root @ scaled
    distance = float(joint @ joint)
    return np.array([marginal[0], marginal[1], joint[0], joint[1], distance])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    helpers = (
        "import numpy as np\n"
        "def _ssig(s):\n"
        "    s = np.asarray(s, dtype=float)\n"
        "    if s.shape != (5,):\n"
        "        return -1.0\n"
        "    weights = np.array([1.0, -0.7, 0.4, 1.3, 0.25])\n"
        "    return float(np.sum(np.abs(s)) + s @ weights)\n"
        "V1 = np.array([[1.7, -0.6], [-0.6, 0.9]])\n"
        "V2 = np.array([[0.8, 0.5], [0.5, 1.4]])\n"
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
            "call": "_ssig(compute_studentized_statistics(np.array([-1.3, 0.8]), np.array([-1.5, 0.55]), V1, 30))",
            "gold_call": "_ssig(_oracle_compute_studentized_statistics(np.array([-1.3, 0.8]), np.array([-1.5, 0.55]), V1, 30))",
        },
        {
            "setup": helpers,
            "call": "float(compute_studentized_statistics(np.array([0.4, 1.9]), np.array([0.1, 2.2]), V2, 12)[4])",
            "gold_call": "float(_oracle_compute_studentized_statistics(np.array([0.4, 1.9]), np.array([0.1, 2.2]), V2, 12)[4])",
        },
        {
            "setup": helpers,
            "call": "_ssig(compute_studentized_statistics(np.array([0.4, 1.9]), np.array([0.1, 2.2]), V2, 1))",
            "gold_call": "_ssig(_oracle_compute_studentized_statistics(np.array([0.4, 1.9]), np.array([0.1, 2.2]), V2, 1))",
        },
        {
            "setup": helpers,
            "call": "float(compute_studentized_statistics(np.array([2.0, 0.3]), np.array([1.0, 0.1]), np.diag([0.25, 4.0]), 9)[3])",
            "gold_call": "float(_oracle_compute_studentized_statistics(np.array([2.0, 0.3]), np.array([1.0, 0.1]), np.diag([0.25, 4.0]), 9)[3])",
        },
        {
            "setup": helpers,
            "call": "_ssig(compute_studentized_statistics(np.array([0.7, 0.7]), np.array([0.7, 0.7]), V1, 50))",
            "gold_call": "_ssig(_oracle_compute_studentized_statistics(np.array([0.7, 0.7]), np.array([0.7, 0.7]), V1, 50))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: compute_studentized_statistics(np.zeros(2), np.ones(2), np.array([[1.0, 2.0], [2.0, 1.0]]), 5))",
            "gold_call": "_status(lambda: _oracle_compute_studentized_statistics(np.zeros(2), np.ones(2), np.array([[1.0, 2.0], [2.0, 1.0]]), 5))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: compute_studentized_statistics(np.zeros(2), np.ones(2), np.array([[1.0, 0.2], [0.1, 1.0]]), 5))",
            "gold_call": "_status(lambda: _oracle_compute_studentized_statistics(np.zeros(2), np.ones(2), np.array([[1.0, 0.2], [0.1, 1.0]]), 5))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: compute_studentized_statistics(np.zeros(2), np.ones(2), V1, 0))",
            "gold_call": "_status(lambda: _oracle_compute_studentized_statistics(np.zeros(2), np.ones(2), V1, 0))",
        },
    ]
