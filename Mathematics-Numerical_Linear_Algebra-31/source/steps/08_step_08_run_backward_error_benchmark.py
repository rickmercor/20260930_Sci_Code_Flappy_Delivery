"""
Run the deterministic benchmark and return the final MINBERR-NE-over-direct ratio.

The final orchestrator constructs the controlled system, evaluates all three histories, and compares the two paper-specific minimizers after the same number of retained history entries. A ratio above one means direct MINBERR has the smaller final backward error.

Returns
-------
float, the positive final MINBERR-NE-over-direct backward-error ratio
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_backward_error_benchmark(
    n: int,
    condition_number: float,
    iterations: int,
    inverse_steps: int,
    seed: int,
) -> float:
    """Return the final MINBERR-NE error divided by the direct error.

    Construct the prescribed augmented positive-definite system, compute the
    fixed-update, direct-MINBERR, and MINBERR-NE histories, and return
    ``history[2, -1] / history[1, -1]``.

    Parameters
    ----------
    n : int
        System dimension, at least 4.
    condition_number : float
        Finite target condition number strictly greater than 1.
    iterations : int
        Number of history entries, with ``1 <= iterations < n``.
    inverse_steps : int
        Positive fixed iteration count for every reduced problem.
    seed : int
        Base integer seed for the reduced iterations.

    Returns
    -------
    float
        Positive final MINBERR-NE-over-direct error ratio.

    Raises
    ------
    ValueError
        If any earlier-stage validation fails for the constructed system or
        histories (invalid ``n``, ``condition_number``, ``iterations``,
        ``inverse_steps``, or ``seed``, per the earlier steps' contracts), or
        if the final minimum backward error or the final normal-equation
        backward error is not positive.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_backward_error_benchmark(
    n: int,
    condition_number: float,
    iterations: int,
    inverse_steps: int,
    seed: int,
) -> float:
    """Return the deterministic end-to-end improvement factor."""
    np = __import__("numpy")

    packed = _oracle_construct_controlled_psd_system(n, condition_number)
    A = packed[:, :-1]
    b = packed[:, -1]

    norm_A = float(np.linalg.norm(A, 2))
    norm_b = float(np.linalg.norm(b))

    Q, T = _oracle_build_orthogonal_recurrence(A, b, int(iterations))

    fixed_errors = []
    direct_errors = []
    x_fixed = np.zeros(A.shape[0], dtype=float)

    for j in range(1, iterations + 1):
        x_fixed = x_fixed + (b - A @ x_fixed) / norm_A
        fixed_errors.append(_oracle_compute_relative_backward_error(A, b, x_fixed))

        Qj = Q[:, : j + 1]
        Tj = T[: j + 1, :j]
        reduced, certificate, factor = _oracle_extract_reduced_operator(Tj, 1.0e-5)
        if certificate.shape != (j + 2,) or not np.all(np.isfinite(certificate)):
            raise ValueError("invalid shifted-Cholesky certificate")
        if factor.shape != (j, j) or not np.all(np.isfinite(factor)):
            raise ValueError("invalid shifted-Cholesky factor state")
        pair = _oracle_approximate_smallest_singular_pair(
            reduced,
            int(inverse_steps),
            int(seed) + j,
        )
        v = pair[1:]
        denom_dir = float(Tj[0, :] @ v)
        denom_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(1.0, float(np.linalg.norm(Tj[0, :])))
        )
        if abs(denom_dir) <= denom_tolerance:
            raise ValueError("the direct MINBERR direction cannot be scaled")
        x_direct = Qj[:, :j] @ ((norm_b / denom_dir) * v)
        direct_errors.append(_oracle_compute_relative_backward_error(A, b, x_direct))
        if int(certificate[0]) < j:
            break

    retained = len(direct_errors)
    normal_errors = _oracle_compute_normal_equation_history(
        A,
        b,
        retained,
        int(inverse_steps),
        int(seed) + 10000,
    )

    histories = np.vstack(
        (
            np.asarray(fixed_errors),
            np.asarray(direct_errors),
            np.asarray(normal_errors),
        )
    )
    # Keep reference results independent of candidate function bindings.
    histories_check = _oracle_compute_error_histories(
        A, b, int(iterations), int(inverse_steps), int(seed)
    )
    if not np.allclose(histories, histories_check, rtol=1e-9, atol=1e-9):
        raise ValueError("compute_error_histories is inconsistent with the per-step chain")

    denominator = float(histories[1, -1])
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("the final minimum backward error must be positive")
    numerator = float(histories[2, -1])
    if not np.isfinite(numerator) or numerator <= 0.0:
        raise ValueError("the final normal-equation backward error must be positive")
    return float(numerator / denominator)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return three end-to-end cases and one invalid configuration."""
    return [
        {
            "setup": "n = 8\ncondition_number = 1.0e4\niterations = 4\ninverse_steps = 14\nseed = 11",
            "call": "run_backward_error_benchmark(n, condition_number, iterations, inverse_steps, seed)",
            "gold_call": "_oracle_run_backward_error_benchmark(n, condition_number, iterations, inverse_steps, seed)",
        },
        {
            "setup": "n = 11\ncondition_number = 1.0e7\niterations = 5\ninverse_steps = 20\nseed = 97",
            "call": "run_backward_error_benchmark(n, condition_number, iterations, inverse_steps, seed)",
            "gold_call": "_oracle_run_backward_error_benchmark(n, condition_number, iterations, inverse_steps, seed)",
        },
        {
            "setup": "n = 16\ncondition_number = 1.0e12\niterations = 7\ninverse_steps = 30\nseed = 1729",
            "call": "run_backward_error_benchmark(n, condition_number, iterations, inverse_steps, seed)",
            "gold_call": "_oracle_run_backward_error_benchmark(n, condition_number, iterations, inverse_steps, seed)",
        },
        {
            "setup": """def run_model():
    try:
        run_backward_error_benchmark(3, 100.0, 2, 5, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_backward_error_benchmark(3, 100.0, 2, 5, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
