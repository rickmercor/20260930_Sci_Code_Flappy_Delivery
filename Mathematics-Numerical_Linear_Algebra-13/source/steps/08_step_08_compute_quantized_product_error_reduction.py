"""
Run the complete certified bounded error-reduction calculation.



The pipeline constructs the nonsmooth range partition, encodes every active

cell, reconstructs exact exponential branches, enumerates the finite global

candidate set, checks the original three-term identity, and returns the

certified percentage reduction from the identity transform.

Returns
-------
one finite float equal to 100 * (1 - minimum full error / identity full error)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_quantized_product_error_reduction(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float = 1e-12,
) -> float:
    """Return the full expected-error reduction from bounded shared scaling.

    Raises ``ValueError`` unless ``A`` and ``B`` are finite two-dimensional
    arrays with shapes ``(m, 2)`` and ``(2, n)`` for positive ``m`` and ``n``;
    both bit widths are non-Boolean integers of at least 2 producing finite
    positive coefficients; ``bound`` is finite and strictly positive;
    ``tolerance`` is finite, strictly positive, and smaller than ``bound``;
    and the identity-transform expected error is finite and strictly positive.
    This step is the orchestrator: it calls every earlier step in order, so
    all of them must be defined.

    Parameters
    ----------
    A : np.ndarray
        Left factor of shape ``(m, 2)``.
    B : np.ndarray
        Right factor of shape ``(2, n)``.
    bits_a : int
        Signed bit width for the left factor.
    bits_b : int
        Signed bit width for the right factor.
    bound : float
        Positive closed log-scale bound for ``x``.
    tolerance : float, optional
        Positive switch, tie, and certificate tolerance.

    Returns
    -------
    float
        Finite percentage ``100 * (1 - E_star / E_0)``.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _local_validate(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    left = np.asarray(A, dtype=float)
    right = np.asarray(B, dtype=float)
    if left.ndim != 2 or left.shape[0] < 1 or left.shape[1] != 2:
        raise ValueError("A must have shape (m, 2) with m positive")
    if right.ndim != 2 or right.shape[0] != 2 or right.shape[1] < 1:
        raise ValueError("B must have shape (2, n) with n positive")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("matrix entries must be finite")
    if isinstance(bound, (bool, np.bool_)) or not np.isscalar(bound):
        raise ValueError("bound must be finite and strictly positive")
    if isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance):
        raise ValueError("tolerance must be finite, positive, and smaller than bound")
    try:
        limit = float(bound)
        tol = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("bound and tolerance must be finite real scalars") from exc
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("bound must be finite and strictly positive")
    if not np.isfinite(tol) or tol <= 0.0 or tol >= limit:
        raise ValueError("tolerance must be finite, positive, and smaller than bound")
    return (
        left,
        right,
        _oracle_compute_dither_variance_coefficient(bits_a),  # noqa: F821 - earlier pipeline step
        _oracle_compute_dither_variance_coefficient(bits_b),  # noqa: F821 - earlier pipeline step
        limit,
        tol,
    )


def _local_transform(
    left: np.ndarray, right: np.ndarray, x: float
) -> tuple[np.ndarray, np.ndarray]:
    scales = np.exp(np.array([x, -x], dtype=float))
    return left * scales, right / scales[:, None]


def _local_ranges(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.max(np.abs(left), axis=1), np.max(np.abs(right), axis=0)


def _local_fields(
    ranges_a: np.ndarray,
    ranges_b: np.ndarray,
    coefficient_a: float,
    coefficient_b: float,
) -> tuple[np.ndarray, np.ndarray]:
    field_a = np.repeat((coefficient_a * ranges_a**2)[:, None], 2, axis=1)
    field_b = np.repeat((coefficient_b * ranges_b**2)[None, :], 2, axis=0)
    return field_a, field_b


def _local_error(
    left: np.ndarray,
    right: np.ndarray,
    coefficient_a: float,
    coefficient_b: float,
    x: float,
) -> float:
    transformed_left, transformed_right = _local_transform(left, right, x)
    ranges_a, ranges_b = _local_ranges(transformed_left, transformed_right)
    field_a, field_b = _local_fields(ranges_a, ranges_b, coefficient_a, coefficient_b)
    return float(
        _oracle_compute_expected_product_error(  # noqa: F821 - step 06
            transformed_left, transformed_right, field_a, field_b
        )[3]
    )


def _oracle_compute_quantized_product_error_reduction(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float = 1e-12,
) -> float:
    """Reference end-to-end reduction with all certificate stages checked."""
    left, right, coefficient_a, coefficient_b, limit, tol = _local_validate(
        A, B, bits_a, bits_b, bound, tolerance
    )

    identity_left, identity_right = _local_transform(left, right, 0.0)
    if not np.allclose(
        identity_left @ identity_right, left @ right, rtol=1e-13, atol=1e-13
    ):
        raise ValueError("inverse-pair transform must preserve the product")
    ranges_a, ranges_b = _local_ranges(identity_left, identity_right)
    field_a, field_b = _local_fields(ranges_a, ranges_b, coefficient_a, coefficient_b)
    baseline_parts = _oracle_compute_expected_product_error(  # noqa: F821 - earlier pipeline step
        identity_left, identity_right, field_a, field_b
    )
    baseline = float(baseline_parts[3])
    if not np.isfinite(baseline) or baseline <= 0.0:
        raise ValueError(
            "identity-transform expected error must be positive and finite"
        )

    partition = _oracle_construct_two_channel_switch_partition(left, right, limit, tol)  # noqa: F821 - earlier pipeline step
    active_table = _oracle_encode_partition_active_branches(left, right, partition, tol)  # noqa: F821 - earlier pipeline step
    switch_count = int(round(float(partition[0])))
    first_probe = float(partition[1 + switch_count])
    branch = _oracle_reconstruct_active_error_branch(  # noqa: F821 - earlier pipeline step
        left, right, bits_a, bits_b, first_probe, tol
    )
    candidates = _oracle_enumerate_bounded_error_candidates(  # noqa: F821 - earlier pipeline step
        left, right, bits_a, bits_b, limit, tol
    )
    certificate = _oracle_certify_bounded_shared_scaling(  # noqa: F821 - earlier pipeline step
        left, right, bits_a, bits_b, limit, tol
    )
    if int(round(float(active_table[0]))) != switch_count - 1:
        raise ValueError("active-cell count does not match the partition")
    if int(round(float(candidates[0]))) != int(round(float(certificate[6]))):
        raise ValueError("candidate count does not match the certificate")
    probe_error = _local_error(left, right, coefficient_a, coefficient_b, first_probe)
    if abs(float(branch[3]) - probe_error) > 50.0 * tol * max(1.0, abs(probe_error)):
        raise ValueError("branch reconstruction does not match direct error")
    if certificate[12] > 100.0 * tol * max(1.0, baseline):
        raise ValueError("branch reconstruction residual is too large")
    if certificate[13] > 100.0 * tol * max(1.0, float(np.max(np.abs(left @ right)))):
        raise ValueError("product-preservation residual is too large")

    x_star = float(certificate[0])
    optimized_left, optimized_right = _local_transform(left, right, x_star)
    if not np.allclose(
        optimized_left @ optimized_right, left @ right, rtol=1e-13, atol=1e-13
    ):
        raise ValueError("inverse-pair transform must preserve the product")
    ranges_a, ranges_b = _local_ranges(optimized_left, optimized_right)
    field_a, field_b = _local_fields(ranges_a, ranges_b, coefficient_a, coefficient_b)
    optimized_parts = _oracle_compute_expected_product_error(  # noqa: F821 - earlier pipeline step
        optimized_left, optimized_right, field_a, field_b
    )
    optimized = float(optimized_parts[3])
    if abs(optimized - float(certificate[3])) > 50.0 * tol * max(1.0, abs(optimized)):
        raise ValueError("certified and directly evaluated optima disagree")
    if optimized > baseline and optimized - baseline <= tol * max(1.0, baseline):
        optimized = baseline
    reduction = 100.0 * (1.0 - optimized / baseline)
    if not np.isfinite(reduction):
        raise ValueError("error reduction must be finite")
    return float(reduction)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return five whole-pipeline regimes and one invalid case."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[12.0,0.5],[7.0,-0.4],[0.6,9.0],[0.8,6.0],[1.2,0.7]])
B = np.array([[0.2,0.5,-0.3,0.4],[6.0,-4.0,5.0,3.0]])
bits_a = bits_b = 2
bound = np.log(4.0)
""",
            "call": "compute_quantized_product_error_reduction(A, B, bits_a, bits_b, bound)",
            "gold_call": "_oracle_compute_quantized_product_error_reduction(A, B, bits_a, bits_b, bound)",
            "tol": 1e-9,
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0,1.0],[1.0,2.0],[-2.0,1.0],[1.0,-2.0]])
B = np.array([[2.0,1.0,-2.0,1.0],[1.0,2.0,1.0,-2.0]])
bits_a = bits_b = 2
bound = np.log(4.0)
""",
            "call": "compute_quantized_product_error_reduction(A, B, bits_a, bits_b, bound)",
            "gold_call": "_oracle_compute_quantized_product_error_reduction(A, B, bits_a, bits_b, bound)",
            "tol": 1e-9,
        },
        {
            "setup": """import numpy as np
A = np.array([[100.0,1.0],[80.0,2.0],[60.0,1.0]])
B = np.array([[0.001,0.002],[4.0,5.0]])
bits_a = bits_b = 2
bound = np.log(4.0)
""",
            "call": "compute_quantized_product_error_reduction(A, B, bits_a, bits_b, bound)",
            "gold_call": "_oracle_compute_quantized_product_error_reduction(A, B, bits_a, bits_b, bound)",
            "tol": 1e-8,
        },
        {
            "setup": """import numpy as np
A = np.array([[-8.0,0.2],[0.4,7.0],[3.0,-2.0],[0.1,5.0]])
B = np.array([[5.0,-0.3,1.0,-2.0],[-0.2,6.0,4.0,0.5]])
bits_a, bits_b = 5, 3
bound = 0.83
""",
            "call": "compute_quantized_product_error_reduction(A, B, bits_a, bits_b, bound)",
            "gold_call": "_oracle_compute_quantized_product_error_reduction(A, B, bits_a, bits_b, bound)",
            "tol": 1e-9,
        },
        {
            "setup": """import numpy as np
A = np.array([[4.0,-4.0],[-4.0,-1.0],[2.0,4.0]])
B = np.array([[-5.0,1.0,-1.0],[-4.0,-4.0,-4.0]])
bits_a, bits_b = 2, 3
bound = 1.0
""",
            "call": "compute_quantized_product_error_reduction(A, B, bits_a, bits_b, bound)",
            "gold_call": "_oracle_compute_quantized_product_error_reduction(A, B, bits_a, bits_b, bound)",
            "tol": 1e-9,
        },
        {
            "setup": """import numpy as np
A = np.zeros((2, 2))
B = np.ones((2, 2))
bits_a = bits_b = 2
bound = np.log(4.0)
def run_model():
    try:
        compute_quantized_product_error_reduction(A, B, bits_a, bits_b, bound)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_quantized_product_error_reduction(A, B, bits_a, bits_b, bound)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 0.0,
        },
    ]
