"""
Step name: 08_solve_committor
Step description: Solve the Galerkin system for the basis coefficients and expand them into a committor value on every frame of the trajectory.

Step scientific background: The committor is the probability that the halted dynamics reach the product state before the reactant state, so it is pinned to zero throughout the reactant state and to one throughout the product state and is unknown only in between. Solving the projected boundary value problem supplies the missing values as one number per microstate.

Returns
-------
np.ndarray: float committor of shape (T,), zero on atomic frames and one on molecular frames.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def solve_committor(labels: np.ndarray, microstates: np.ndarray,
                    system: np.ndarray) -> np.ndarray:
    """Return the forward committor of every frame of the trajectory.

    The leading square block of ``system`` is the Galerkin matrix and its
    final column is the reference vector of the same projected problem, as
    produced by the previous step. The coefficient vector solves that system
    so that the expansion reproduces the projected boundary value problem;
    the reference column enters as an inhomogeneity, not as an extra unknown.

    Conventions fixed by this step. The committor is exactly ``0.0`` on every
    atomic frame and exactly ``1.0`` on every molecular frame, and an
    intermediate frame takes the coefficient of its own microstate, so the
    returned values are not clipped to the unit interval.

    Parameters
    ----------
    labels : np.ndarray
        Integer array of shape ``(T,)`` with values in ``{0, 1, 2}``.
    microstates : np.ndarray
        Integer array of shape ``(T,)`` holding a microstate index on
        intermediate frames and ``-1`` elsewhere.
    system : np.ndarray
        Float array of shape ``(n, n + 1)``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(T,)`` of committor values.

    Raises
    ------
    ValueError
        If ``labels`` and ``microstates`` are not integer arrays of equal
        length with the stated value ranges, if ``system`` is not a finite
        float array of shape ``(n, n + 1)`` with ``n`` at least 1, if an
        intermediate frame carries a microstate index outside ``0 .. n - 1``,
        or if the Galerkin matrix is singular.
    """
    return committor

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _solve_basis_coefficients(system):
    """Return the coefficient vector of the projected boundary value problem."""
    import numpy as np
    matrix = np.asarray(system, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != matrix.shape[0] + 1:
        raise ValueError("system must have shape (n, n + 1) with n at least 1")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("system must be finite")
    try:
        return np.linalg.solve(matrix[:, :-1], -matrix[:, -1])
    except np.linalg.LinAlgError as error:
        raise ValueError("the Galerkin matrix is singular") from error


import numpy as np
def _oracle_solve_committor(labels: np.ndarray, microstates: np.ndarray,
                            system: np.ndarray) -> np.ndarray:
    """Reference implementation (linear solve plus indicator expansion)."""
    import numpy as np

    tags = _validated_labels(labels)
    if not np.all(np.isin(tags, (0, 1, 2))):
        raise ValueError("labels must contain only 0, 1 and 2")
    cells = np.asarray(microstates)
    if cells.ndim != 1 or cells.shape[0] != tags.shape[0] or not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("microstates must be an integer array matching labels")
    coefficients = _solve_basis_coefficients(system)

    interior = tags == 2
    if np.any(cells[~interior] != -1):
        raise ValueError("only intermediate frames may carry a microstate index")
    if interior.any():
        inside = cells[interior]
        if inside.min() < 0 or inside.max() >= coefficients.size:
            raise ValueError("microstate indices must lie in 0 .. n - 1")

    committor = np.where(tags == 1, 1.0, 0.0)
    committor[interior] = coefficients[cells[interior]]
    return committor

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _sig(a):\n"
        "    f = np.asarray(a, dtype=float).ravel()\n"
        "    if f.size == 0:\n"
        "        return -1.0\n"
        "    w = np.cos(np.arange(f.size, dtype=float) + 1.0)\n"
        "    return float(np.dot(f, w) + np.abs(f).mean() + 1000.0 * f[0])\n"
        "traj = simulate_surface_trajectory(12000, 101, kt_ev=0.07)\n"
        "lab = label_reaction_endpoints(traj)\n"
        "comp = project_dynamical_components(build_local_descriptors(traj))\n"
        "mic = assign_intermediate_microstates(comp, lab, n_microstates=24)\n"
        "stp = compute_stopping_times(lab)\n"
        "sys20 = build_galerkin_system(lab, mic, stp, lag_frames=20, n_microstates=24)\n"
        "sys5 = build_galerkin_system(lab, mic, stp, lag_frames=5, n_microstates=24)\n"
        "traj_gold = _oracle_simulate_surface_trajectory(12000, 101, kt_ev=0.07)\n"
        "lab_gold = _oracle_label_reaction_endpoints(traj_gold)\n"
        "comp_gold = _oracle_project_dynamical_components(_oracle_build_local_descriptors(traj_gold))\n"
        "mic_gold = _oracle_assign_intermediate_microstates(comp_gold, lab_gold, n_microstates=24)\n"
        "stp_gold = _oracle_compute_stopping_times(lab_gold)\n"
        "sys20_gold = _oracle_build_galerkin_system(lab_gold, mic_gold, stp_gold, lag_frames=20, n_microstates=24)\n"
        "sys5_gold = _oracle_build_galerkin_system(lab_gold, mic_gold, stp_gold, lag_frames=5, n_microstates=24)\n"
        "tiny = np.array([0, 2, 2, 1, 2, 0, 2, 2, 1, 2, 2, 0], dtype=np.int64)\n"
        "tmic = np.where(tiny == 2, np.arange(12) % 3, -1).astype(np.int64)\n"
        "tsys = build_galerkin_system(tiny, tmic, compute_stopping_times(tiny), 2, 3)\n"
        "tsys_gold = _oracle_build_galerkin_system(tiny, tmic, _oracle_compute_stopping_times(tiny), 2, 3)\n"
    )
    status = (
        "import numpy as np\n"
        "tiny = np.array([0, 2, 2, 1, 2, 0, 2, 2, 1, 2, 2, 0], dtype=np.int64)\n"
        "tmic = np.where(tiny == 2, np.arange(12) % 3, -1).astype(np.int64)\n"
        "tsys = build_galerkin_system(tiny, tmic, compute_stopping_times(tiny), 2, 3)\n"
        "tsys_gold = _oracle_build_galerkin_system(tiny, tmic, _oracle_compute_stopping_times(tiny), 2, 3)\n"
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
            "setup": digest,
            "call": "_sig(solve_committor(lab, mic, sys20))",
            "gold_call": "_sig(_oracle_solve_committor(lab_gold, mic_gold, sys20_gold))",
        },
        {
            "setup": digest,
            "call": "float(np.min(solve_committor(lab, mic, sys20)[lab == 2]) + np.max(solve_committor(lab, mic, sys20)[lab == 2]))",
            "gold_call": "float(np.min(_oracle_solve_committor(lab_gold, mic_gold, sys20_gold)[lab_gold == 2]) + np.max(_oracle_solve_committor(lab_gold, mic_gold, sys20_gold)[lab_gold == 2]))",
        },
        {
            "setup": digest,
            "call": "_sig(solve_committor(lab, mic, sys5))",
            "gold_call": "_sig(_oracle_solve_committor(lab_gold, mic_gold, sys5_gold))",
        },
        {
            "setup": digest,
            "call": "_sig(solve_committor(tiny, tmic, tsys))",
            "gold_call": "_sig(_oracle_solve_committor(tiny, tmic, tsys_gold))",
        },
        {
            "setup": digest,
            "call": "float(np.sum(solve_committor(lab, mic, sys20)[lab < 2]) + 1e6 * solve_committor(lab, mic, sys20)[0])",
            "gold_call": "float(np.sum(_oracle_solve_committor(lab_gold, mic_gold, sys20_gold)[lab_gold < 2]) + 1e6 * _oracle_solve_committor(lab_gold, mic_gold, sys20_gold)[0])",
        },
        {
            "setup": status,
            "call": "_status(lambda: solve_committor(tiny, tmic, np.zeros((3, 4))))",
            "gold_call": "_status(lambda: _oracle_solve_committor(tiny, tmic, np.zeros((3, 4))))",
        },
        {
            "setup": status,
            "call": "_status(lambda: solve_committor(tiny, tmic, tsys[:, :3]))",
            "gold_call": "_status(lambda: _oracle_solve_committor(tiny, tmic, tsys_gold[:, :3]))",
        },
        {
            "setup": status,
            "call": "_status(lambda: solve_committor(tiny, np.zeros(12, dtype=np.int64), tsys))",
            "gold_call": "_status(lambda: _oracle_solve_committor(tiny, np.zeros(12, dtype=np.int64), tsys_gold))",
        },
    ]
