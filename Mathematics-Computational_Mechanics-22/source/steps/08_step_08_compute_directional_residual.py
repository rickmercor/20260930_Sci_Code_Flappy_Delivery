"""
Run flexible right preconditioning from compact retained-coordinate states.

Flexible GMRES must preserve the action used at each Arnoldi column.  For the

source method that action is not a dense inverse matrix: it is Algorithm 1's

core solve followed by the retained-coordinate Woodbury correction.  Each

iteration may use a different retained rank, so the compact row, response, and

reduced-matrix sequences are ragged and ordered.  The accepted iterate still

minimizes the complete-operator residual over the stored preconditioned

directions, and the reported residual is freshly recomputed from that iterate.

For the source's prospective pressure-space extension, that core action can

itself be a compact Woodbury action over a still lower base. Restart boundaries

then matter: every new cycle is initialized by the fresh complete-equation

residual, while the compact states remain globally ordered.

Returns
-------
tuple containing finite float arrays of shapes (n,), (max_iterations + 1,) and (max_iterations,), then an integer count of iterations taken
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_directional_residual(
    full_operator: np.ndarray,
    right_hand_side: np.ndarray,
    base_operator: np.ndarray,
    retained_row_sequence: tuple[np.ndarray, ...],
    core_response_sequence: tuple[np.ndarray, ...],
    reduced_matrix_sequence: tuple[np.ndarray, ...],
    max_iterations: int,
    cycle_lengths: np.ndarray | None = None,
    initial_guess: np.ndarray | None = None,
    lower_retained_rows: np.ndarray | None = None,
    lower_core_responses: np.ndarray | None = None,
    lower_reduced_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    r"""Run restarted FGMRES from ordered one- or two-level compact states.

    At column j, apply the source's compact retained-basis inverse action for
    the j-th row/response/reduced-matrix triple to the current Arnoldi vector.
    If the three lower-level arrays are supplied, first apply their fixed
    Woodbury action over base_operator and use that effective core beneath
    every top-level correction; otherwise solve base_operator directly.
    Store that preconditioned direction, use two passes of orthogonalization on
    its complete-operator image. Within each cycle, accept the
    least-Euclidean-norm coordinates minimizing the fresh residual at that
    cycle's start over the directions stored in that cycle. At a restart,
    discard the Arnoldi basis, keep the accumulated iterate, and normalize a
    freshly evaluated complete residual. Close the entire recurrence when a
    twice-orthogonalized image vanishes to working precision.

    cycle_lengths is a one-dimensional positive-integer array whose sum is
    max_iterations; None means one unrestarted cycle. initial_guess is zero
    when omitted. The residual history has length max_iterations + 1, starts at
    the fresh normalized residual of that guess, and holds
    the freshly evaluated normalized residual after every used column, with the
    terminal value repeated through unused entries.  The returned coordinate
    vector is zero-padded to max_iterations and concatenates the accepted local
    least-squares coordinates cycle by cycle.

    Raises ValueError unless full_operator and base_operator are finite
    nonempty square matrices of the same shape; base_operator is nonsingular;
    right_hand_side is a finite nonzero vector of shape (n,); max_iterations is
    a Python or NumPy integer, not a bool, in [1, n]; each supplied sequence is
    a tuple or list of exactly max_iterations arrays; and at iteration j the
    three arrays have shapes (r_j, n), (n, r_j), and (r_j, r_j), including
    r_j=0, are finite, satisfy the compact-state identities to rtol 1e-10 and
    atol 1e-12, and define a nonsingular reduced matrix.  A numerically zero
    compact inverse direction is also rejected. The three lower-level arrays
    must be supplied together or omitted together; when supplied they have
    shapes (p, n), (n, p), and (p, p), satisfy their two compact identities,
    and define a nonsingular reduced matrix. Every top-level response is
    checked against the resulting effective core. cycle_lengths must have the
    partition stated above, and initial_guess must be finite with shape (n,).

    Parameters
    ----------
    full_operator : np.ndarray
        Complete, potentially nonsymmetric operator A of shape (n, n).
    right_hand_side : np.ndarray
        Nonzero right-hand side of shape (n,).
    base_operator : np.ndarray
        Fixed nonsingular core M of shape (n, n).
    retained_row_sequence : tuple[np.ndarray, ...]
        Ordered retained row matrices R_j of shapes (r_j, n).
    core_response_sequence : tuple[np.ndarray, ...]
        Ordered core-response blocks Z_j of shapes (n, r_j).
    reduced_matrix_sequence : tuple[np.ndarray, ...]
        Ordered reduced matrices S_j of shapes (r_j, r_j).
    max_iterations : int
        Requested number of flexible Arnoldi columns.
    cycle_lengths : np.ndarray or None
        Optional positive-integer restart partition summing to max_iterations.
    initial_guess : np.ndarray or None
        Optional initial iterate of shape (n,); defaults to zero.
    lower_retained_rows, lower_core_responses, lower_reduced_matrix : np.ndarray or None
        Optional fixed lower-level compact Woodbury state.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, int]
        Final iterate of shape (n,), normalized residual history of shape
        (k + 1,), zero-padded final coordinates of shape (k,), and the number
        of iterations taken.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_directional_residual(
    full_operator: np.ndarray,
    right_hand_side: np.ndarray,
    base_operator: np.ndarray,
    retained_row_sequence: tuple[np.ndarray, ...],
    core_response_sequence: tuple[np.ndarray, ...],
    reduced_matrix_sequence: tuple[np.ndarray, ...],
    max_iterations: int,
    cycle_lengths: np.ndarray | None = None,
    initial_guess: np.ndarray | None = None,
    lower_retained_rows: np.ndarray | None = None,
    lower_core_responses: np.ndarray | None = None,
    lower_reduced_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Reference restarted, optionally nested compact-state FGMRES."""
    operator = np.asarray(full_operator, dtype=float)
    rhs = np.asarray(right_hand_side, dtype=float)
    base = np.asarray(base_operator, dtype=float)
    if (
        operator.ndim != 2
        or operator.shape[0] != operator.shape[1]
        or operator.shape[0] == 0
    ):
        raise ValueError("full_operator must be a nonempty square matrix")
    n_dof = operator.shape[0]
    if base.shape != (n_dof, n_dof):
        raise ValueError("base_operator must have shape (n, n)")
    if rhs.shape != (n_dof,):
        raise ValueError("right_hand_side must have shape (n,)")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(
        max_iterations, (int, np.integer)
    ):
        raise ValueError("max_iterations must be an integer")  # noqa: TRY004
    max_iterations = int(max_iterations)
    if max_iterations < 1 or max_iterations > n_dof:
        raise ValueError("max_iterations must lie in [1, n]")
    if cycle_lengths is None:
        cycles = np.array([max_iterations], dtype=int)
    else:
        raw_cycles = np.asarray(cycle_lengths)
        if (
            raw_cycles.ndim != 1
            or raw_cycles.size == 0
            or raw_cycles.dtype.kind not in {"i", "u"}
            or raw_cycles.dtype.kind == "b"
        ):
            raise ValueError("cycle_lengths must be a positive-integer vector")
        cycles = np.asarray(raw_cycles, dtype=int)
        if np.any(cycles <= 0) or int(np.sum(cycles)) != max_iterations:
            raise ValueError("cycle_lengths must be positive and sum to k")
    if not all(
        isinstance(sequence, (tuple, list)) and len(sequence) == max_iterations
        for sequence in (
            retained_row_sequence,
            core_response_sequence,
            reduced_matrix_sequence,
        )
    ):
        raise ValueError("each compact-state sequence must have length k")
    if not all(np.all(np.isfinite(x)) for x in (operator, rhs, base)):
        raise ValueError("all inputs must be finite")
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        raise ValueError("right_hand_side must have nonzero norm")
    try:
        np.linalg.solve(base, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("base_operator must be nonsingular") from exc

    lower_flags = (
        lower_retained_rows is not None,
        lower_core_responses is not None,
        lower_reduced_matrix is not None,
    )
    if any(lower_flags) and not all(lower_flags):
        raise ValueError("supply all three lower compact arrays or none")
    if all(lower_flags):
        lower_rows = np.asarray(lower_retained_rows, dtype=float)
        lower_responses = np.asarray(lower_core_responses, dtype=float)
        lower_reduced = np.asarray(lower_reduced_matrix, dtype=float)
        if lower_rows.ndim != 2 or lower_rows.shape[1] != n_dof:
            raise ValueError("lower_retained_rows must have shape (p, n)")
        lower_rank = lower_rows.shape[0]
        if lower_responses.shape != (n_dof, lower_rank):
            raise ValueError("lower_core_responses must have shape (n, p)")
        if lower_reduced.shape != (lower_rank, lower_rank):
            raise ValueError("lower_reduced_matrix must have shape (p, p)")
        if not all(
            np.all(np.isfinite(x)) for x in (lower_rows, lower_responses, lower_reduced)
        ):
            raise ValueError("the lower compact state must be finite")
        if not np.allclose(
            base @ lower_responses,
            lower_rows.T,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("the lower core responses are inconsistent")
        expected_lower = np.eye(lower_rank) + lower_rows @ lower_responses
        if not np.allclose(lower_reduced, expected_lower, rtol=1e-10, atol=1e-12):
            raise ValueError("the lower reduced matrix is inconsistent")
        if lower_rank:
            try:
                np.linalg.solve(lower_reduced, np.ones(lower_rank))
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "the lower reduced matrix must be nonsingular"
                ) from exc
        effective_core = base + lower_rows.T @ lower_rows
    else:
        lower_rows = np.zeros((0, n_dof), dtype=float)
        lower_responses = np.zeros((n_dof, 0), dtype=float)
        lower_reduced = np.zeros((0, 0), dtype=float)
        effective_core = base

    def apply_effective_core_inverse(vector: np.ndarray) -> np.ndarray:
        """Apply the direct or nested lower-level inverse."""
        action = np.linalg.solve(base, vector)
        if lower_rows.shape[0]:
            action = action - lower_responses @ np.linalg.solve(
                lower_reduced, lower_rows @ action
            )
        return action

    compact_states = []
    for column in range(max_iterations):
        rows = np.asarray(retained_row_sequence[column], dtype=float)
        responses = np.asarray(core_response_sequence[column], dtype=float)
        reduced = np.asarray(reduced_matrix_sequence[column], dtype=float)
        if rows.ndim != 2 or rows.shape[1] != n_dof:
            raise ValueError("each retained row matrix must have shape (r_j, n)")
        rank = rows.shape[0]
        if responses.shape != (n_dof, rank):
            raise ValueError("each core response block must have shape (n, r_j)")
        if reduced.shape != (rank, rank):
            raise ValueError("each reduced matrix must have shape (r_j, r_j)")
        if not all(np.all(np.isfinite(x)) for x in (rows, responses, reduced)):
            raise ValueError("all compact states must be finite")
        if not np.allclose(effective_core @ responses, rows.T, rtol=1e-10, atol=1e-12):
            raise ValueError("a core response block is inconsistent")
        expected_reduced = np.eye(rank, dtype=float) + rows @ responses
        if not np.allclose(reduced, expected_reduced, rtol=1e-10, atol=1e-12):
            raise ValueError("a reduced matrix is inconsistent")
        if rank:
            try:
                np.linalg.solve(reduced, np.ones(rank, dtype=float))
            except np.linalg.LinAlgError as exc:
                raise ValueError("each reduced matrix must be nonsingular") from exc
        compact_states.append((rows, responses, reduced))

    if initial_guess is None:
        iterate = np.zeros(n_dof, dtype=float)
    else:
        iterate = np.asarray(initial_guess, dtype=float)
        if iterate.shape != (n_dof,) or not np.all(np.isfinite(iterate)):
            raise ValueError("initial_guess must be a finite vector of shape (n,)")
        iterate = iterate.copy()

    history = np.zeros(max_iterations + 1, dtype=float)
    coefficients = np.zeros(max_iterations, dtype=float)
    initial_residual = rhs - operator @ iterate
    history[0] = float(np.linalg.norm(initial_residual) / rhs_norm)
    used = 0
    terminate = False
    operator_norm = float(np.linalg.norm(operator, ord=2))

    for cycle_length in cycles:
        cycle_start = iterate.copy()
        cycle_residual = rhs - operator @ cycle_start
        cycle_norm = float(np.linalg.norm(cycle_residual))
        zero_scale = max(
            1.0,
            rhs_norm,
            operator_norm * float(np.linalg.norm(cycle_start)),
        )
        if cycle_norm <= 100.0 * np.finfo(float).eps * zero_scale:
            terminate = True
            break

        basis = np.zeros((n_dof, int(cycle_length) + 1), dtype=float)
        preconditioned = np.zeros((n_dof, int(cycle_length)), dtype=float)
        basis[:, 0] = cycle_residual / cycle_norm
        cycle_offset = used

        for column in range(int(cycle_length)):
            rows, responses, reduced = compact_states[used]
            core_action = apply_effective_core_inverse(basis[:, column])
            if rows.shape[0]:
                correction = responses @ np.linalg.solve(reduced, rows @ core_action)
            else:
                correction = np.zeros(n_dof, dtype=float)
            direction = core_action - correction
            direction_scale = max(
                1.0,
                float(np.linalg.norm(core_action)),
                float(np.linalg.norm(correction)),
            )
            if (
                float(np.linalg.norm(direction))
                <= np.finfo(float).eps * direction_scale
            ):
                raise ValueError("a compact inverse action produced a zero direction")
            preconditioned[:, column] = direction

            raw_image = operator @ direction
            work = raw_image.copy()
            active_basis = basis[:, : column + 1]
            for _ in range(2):
                work -= active_basis @ (active_basis.T @ work)
            next_norm = float(np.linalg.norm(work))
            breakdown = next_norm <= 100.0 * np.finfo(float).eps * max(
                1.0, float(np.linalg.norm(raw_image))
            )
            if not breakdown:
                basis[:, column + 1] = work / next_norm

            images = operator @ preconditioned[:, : column + 1]
            local_coefficients = np.linalg.lstsq(images, cycle_residual, rcond=None)[0]
            coefficients[cycle_offset : cycle_offset + column + 1] = local_coefficients
            iterate = cycle_start + preconditioned[:, : column + 1] @ local_coefficients
            residual = float(np.linalg.norm(rhs - operator @ iterate) / rhs_norm)
            if not np.isfinite(residual):
                raise ValueError("the fresh residual must be finite")
            used += 1
            history[used] = residual
            if breakdown:
                terminate = True
                break
        if terminate:
            break

    history[used + 1 :] = history[used]

    if not all(np.all(np.isfinite(x)) for x in (iterate, history, coefficients)):
        raise ValueError("the cycle outputs must be finite")
    return iterate, history, coefficients, used

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return compact stationary/flexible, breakdown, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[104.58,-1.48,.2,0,.25],[-1.51,28.626666666666665,-.603333333333333,.433333333333333,-.25],[.2,-.653333333333333,7.226666666666666,-.306666666666667,.22],[0,.433333333333333,-.346666666666667,4.136666666666667,1.74],[.26,-.25,.22,1.68,2.395]])
M = np.array([[40.5,-1.5,.2,0,.25],[-1.5,12.666666666666666,-.633333333333333,.433333333333333,-.25],[.2,-.633333333333333,3.166666666666667,-.866666666666667,.1],[0,.433333333333333,-.866666666666667,1.866666666666667,-.1],[.25,-.25,.1,-.1,.925]])
R = np.array([[-.331419211787616,.086034102962501,.822780175278025,1.4677952822772,1.115852830670655],[7.607543403265093,1.165070606175298,.207739110938568,.012975632230369,-.00628049304442]])
Z = np.linalg.solve(M, R.T)
S = np.eye(2) + R @ Z
b = np.array([1.0, -.5, .75, .2, -1.1])
Rs, Zs, Ss, k = (R,), (Z,), (S,), 1
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in compute_directional_residual(A, b, M, Rs, Zs, Ss, k)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_compute_directional_residual(A, b, M, Rs, Zs, Ss, k)])",
        },
        {
            "setup": """import numpy as np
A = np.array([[4.0, 1.0, 0.0, -.2], [-.5, 3.0, .7, 0.0], [.2, -.4, 2.0, .5], [0.0, .1, -.8, 1.5]])
M = np.diag([4.0, 3.0, 2.0, 1.5])
b = np.array([1.0, -2.0, .5, 1.2])
Rs = (np.array([[1.0,-.2,0.0,.1]]), np.array([[0.0,1.0,.3,0.0],[.2,0.0,0.0,1.0]]), np.array([[.5,0.0,1.0,-.2]]))
Zs = tuple(np.linalg.solve(M, R.T) for R in Rs)
Ss = tuple(np.eye(R.shape[0]) + R @ Z for R, Z in zip(Rs, Zs))
k = 3
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in compute_directional_residual(A, b, M, Rs, Zs, Ss, k)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_compute_directional_residual(A, b, M, Rs, Zs, Ss, k)])",
        },
        {
            "setup": """import numpy as np
A = np.array([[3.0, -1.2, .4, 0.0], [.6, 2.5, -.3, .2], [0.0, .35, 1.8, -.5], [.15, 0.0, .45, 1.1]])
M = np.diag([3.0, 2.5, 1.8, 1.1])
b = np.array([.8, 1.4, -.6, .2])
R = np.array([[1.0, -.4, .2, 0.0]])
Z = np.linalg.solve(M, R.T)
S = np.eye(1) + R @ Z
Rs, Zs, Ss, k = (R, R, R, R), (Z, Z, Z, Z), (S, S, S, S), 4
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in compute_directional_residual(A, b, M, Rs, Zs, Ss, k)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_compute_directional_residual(A, b, M, Rs, Zs, Ss, k)])",
        },
        {
            "setup": """import numpy as np
A = np.array([[3.0, -1.2, .4, 0.0], [.6, 2.5, -.3, .2], [0.0, .35, 1.8, -.5], [.15, 0.0, .45, 1.1]])
M = np.array([[3.2,.2,0,0],[.2,2.4,.1,0],[0,.1,1.7,-.1],[0,0,-.1,1.2]])
b = np.array([.8, 1.4, -.6, .2])
Rs = (np.zeros((0,4)), np.array([[1.0,0.0,-.2,.1]]), np.array([[0.0,1.0,.3,0.0],[.2,0.0,0.0,1.0]]), np.array([[.5,-.1,0.0,.7]]))
Zs = tuple(np.linalg.solve(M, R.T) for R in Rs)
Ss = tuple(np.eye(R.shape[0]) + R @ Z for R, Z in zip(Rs, Zs))
k = 4
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in compute_directional_residual(A, b, M, Rs, Zs, Ss, k)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_compute_directional_residual(A, b, M, Rs, Zs, Ss, k)])",
        },
        {
            "setup": """import numpy as np
A = np.eye(3)
M = np.eye(3)
b = np.array([1.0, -2.0, .5])
R = np.zeros((0,3)); Z = np.zeros((3,0)); S = np.zeros((0,0))
Rs, Zs, Ss, k = (R,R,R), (Z,Z,Z), (S,S,S), 3
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in compute_directional_residual(A, b, M, Rs, Zs, Ss, k)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_compute_directional_residual(A, b, M, Rs, Zs, Ss, k)])",
        },
        {
            "setup": """import numpy as np
A = np.diag([2.0, 2.0, 5.0])
M = np.eye(3)
b = np.array([1.0, 1.0, 0.0])
R = np.zeros((0,3)); Z = np.zeros((3,0)); S = np.zeros((0,0))
Rs, Zs, Ss, k = (R,R,R), (Z,Z,Z), (S,S,S), 3
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in compute_directional_residual(A, b, M, Rs, Zs, Ss, k)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_compute_directional_residual(A, b, M, Rs, Zs, Ss, k)])",
        },
        {
            "setup": """import numpy as np
A = np.eye(2); M = np.eye(2); b = np.ones(2)
R = np.array([[1.0,0.0]]); Z = np.array([[0.0],[1.0]]); S = np.eye(1)
Rs, Zs, Ss, k = (R,), (Z,), (S,), 1
def run_model():
    try:
        compute_directional_residual(A, b, M, Rs, Zs, Ss, k)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_directional_residual(A, b, M, Rs, Zs, Ss, k)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A = np.eye(2); M = np.eye(2); b = np.ones(2)
R = np.zeros((0,2)); Z = np.zeros((2,0)); S = np.zeros((0,0))
Rs, Zs, Ss, k = (R,R,R), (Z,Z,Z), (S,S,S), 3
def run_model():
    try:
        compute_directional_residual(A, b, M, Rs, Zs, Ss, k)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_directional_residual(A, b, M, Rs, Zs, Ss, k)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A = np.eye(3); M = np.eye(3); b = np.ones(3)
R = np.zeros((0,3)); Z = np.zeros((3,0)); S = np.zeros((0,0))
Rs, Zs, Ss, k = (R,), (Z,), (S,), 2
def run_model():
    try:
        compute_directional_residual(A, b, M, Rs, Zs, Ss, k)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_directional_residual(A, b, M, Rs, Zs, Ss, k)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A = np.array([[4.2,-.7,.2,0.0],[.3,2.8,-.4,.1],[0.0,.5,1.9,-.6],[.2,0.0,.35,1.3]])
K0 = np.diag([3.5,2.4,1.6,1.1])
Rp = np.array([[1.0,-.3,.2,0.0],[0.0,.5,-.2,.7]])
Zp = np.linalg.solve(K0, Rp.T)
Sp = np.eye(2) + Rp @ Zp
Mtilde = K0 + Rp.T @ Rp
b = np.array([1.0,-1.3,.6,.2])
x0 = np.array([.05,-.02,.03,0.0])
Rs = (np.zeros((0,4)), np.array([[.8,0.0,-.1,.2]]), np.array([[0.0,.6,.2,-.1],[.3,0.0,0.0,.7]]), np.array([[.4,-.2,.5,0.0]]))
Zs = tuple(np.linalg.solve(Mtilde, R.T) for R in Rs)
Ss = tuple(np.eye(R.shape[0]) + R @ Z for R, Z in zip(Rs, Zs))
k = 4
cycles = np.array([2,2])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in compute_directional_residual(A, b, K0, Rs, Zs, Ss, k, cycles, x0, Rp, Zp, Sp)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_compute_directional_residual(A, b, K0, Rs, Zs, Ss, k, cycles, x0, Rp, Zp, Sp)])",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0,.3],[-.1,1.5]])
x0 = np.array([.4,-.7])
b = A @ x0
M = np.eye(2)
R = np.zeros((0,2)); Z = np.zeros((2,0)); S = np.zeros((0,0))
Rs, Zs, Ss, k = (R,R), (Z,Z), (S,S), 2
cycles = np.array([1,1])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in compute_directional_residual(A, b, M, Rs, Zs, Ss, k, cycles, x0)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_compute_directional_residual(A, b, M, Rs, Zs, Ss, k, cycles, x0)])",
        },
        {
            "setup": """import numpy as np
A = np.eye(3); M = np.eye(3); b = np.ones(3)
R = np.zeros((0,3)); Z = np.zeros((3,0)); S = np.zeros((0,0))
Rs, Zs, Ss, k = (R,R,R), (Z,Z,Z), (S,S,S), 3
cycles = np.array([1,1])
def run_model():
    try:
        compute_directional_residual(A, b, M, Rs, Zs, Ss, k, cycles)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_directional_residual(A, b, M, Rs, Zs, Ss, k, cycles)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A = np.eye(2); K0 = np.eye(2); b = np.ones(2)
R = np.zeros((0,2)); Z = np.zeros((2,0)); S = np.zeros((0,0))
Rs, Zs, Ss, k = (R,), (Z,), (S,), 1
Rp = np.array([[1.0,0.0]]); Zp = np.array([[0.0],[1.0]]); Sp = np.eye(1)
def run_model():
    try:
        compute_directional_residual(A, b, K0, Rs, Zs, Ss, k, None, None, Rp, Zp, Sp)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_directional_residual(A, b, K0, Rs, Zs, Ss, k, None, None, Rp, Zp, Sp)
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
