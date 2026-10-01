"""
Build the block-diagonal trial basis of the lifted system from temperature snapshots alone.

One proper-orthogonal-decomposition basis is extracted per state block, each

truncated by its own retained-energy criterion, and the two bases are placed

on the diagonal so that projection cannot mix the two blocks.

Returns
-------
tuple[np.ndarray, int, int] holding the trial basis of shape (2n, n1 + n2), the number of temperature modes n1, and the number of auxiliary modes n2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real

import math
import numpy as np


def block_pod_basis(
    state_snapshots: np.ndarray,
    energy_tolerance: float,
) -> tuple[np.ndarray, int, int]:
    """Assemble the block-diagonal proper-orthogonal-decomposition trial basis.

    The auxiliary snapshot matrix is obtained from state_snapshots by the same
    pointwise squaring that defines the auxiliary field, so no second full-order
    solve is performed. Each of the two snapshot matrices is factorised by a
    thin singular value decomposition, and the retained mode count is the
    smallest k for which

        1 - (sum of the first k squared singular values)
            / (sum of all squared singular values) < energy_tolerance,

    evaluated independently for the two blocks. Each retained left singular
    vector is rescaled by +1 or -1 so that its entry of largest absolute value
    is positive; if several entries tie in absolute value, the one with the
    smallest index decides. The returned basis has the temperature modes in its
    upper-left block and the auxiliary modes in its lower-right block, with
    zeros elsewhere.

    Parameters
    ----------
    state_snapshots : np.ndarray
        Finite real array of shape (n, m) with n >= 1 and m >= 1 holding the
        temperature at m sampled states.
    energy_tolerance : float
        Finite tolerance in (0, 1) on the neglected fraction of squared
        singular values.

    Returns
    -------
    trial_basis : np.ndarray
        Float array of shape (2 n, n1 + n2).
    n_temperature_modes : int
        Number of retained temperature modes n1.
    n_auxiliary_modes : int
        Number of retained auxiliary modes n2.

    Raises
    ------
    ValueError
        If state_snapshots has the wrong rank, is empty or holds a non-finite
        entry, or if energy_tolerance is not a finite scalar in (0, 1).
    """
    return (np.empty((0, 0), dtype=float), 0, 0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_block_pod_basis(
    state_snapshots: np.ndarray,
    energy_tolerance: float,
) -> tuple[np.ndarray, int, int]:
    """Reference implementation."""
    from numbers import Real

    import math
    import numpy as np

    def _is_number(value) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    snapshots = np.asarray(state_snapshots, dtype=float)
    if snapshots.ndim != 2 or snapshots.shape[0] < 1 or snapshots.shape[1] < 1:
        raise ValueError("state_snapshots must be a 2-D array with positive extents")
    if not np.all(np.isfinite(snapshots)):
        raise ValueError("state_snapshots entries must be finite")
    if not _is_number(energy_tolerance) or not math.isfinite(float(energy_tolerance)):
        raise ValueError("energy_tolerance must be a finite real scalar")
    tolerance = float(energy_tolerance)
    if not 0.0 < tolerance < 1.0:
        raise ValueError("energy_tolerance must lie strictly between 0 and 1")

    n_nodes = snapshots.shape[0]
    auxiliary = snapshots * snapshots

    modes = []
    counts = []
    for matrix in (snapshots, auxiliary):
        left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
        energy = singular ** 2
        total = float(np.sum(energy))
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("a snapshot block carries no energy")
        neglected = 1.0 - np.cumsum(energy) / total
        admissible = np.nonzero(neglected < tolerance)[0]
        retained = int(admissible[0]) + 1 if admissible.size else int(singular.size)
        block = np.array(left[:, :retained], dtype=float)
        for column in range(retained):
            pivot = int(np.argmax(np.abs(block[:, column])))
            if block[pivot, column] < 0.0:
                block[:, column] = -block[:, column]
        modes.append(block)
        counts.append(retained)

    first, second = counts
    basis = np.zeros((2 * n_nodes, first + second), dtype=float)
    basis[:n_nodes, :first] = modes[0]
    basis[n_nodes:, first:] = modes[1]
    return basis, first, second

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return a list of test case specifications."""
    reducer = (
        "import numpy as np\n"
        "def _sig(value, scale):\n"
        "    a = np.asarray(value, dtype=float)\n"
        "    b = np.concatenate([np.asarray([a.ndim, *a.shape], dtype=float), a.ravel()])\n"
        "    k = np.arange(1.0, b.size + 1.0)\n"
        "    return float((np.sum(np.abs(b)) + np.sum(b * np.cos(k))) / scale)\n"
        "def _sigt(parts, scale):\n"
        "    return float(sum((i + 1.0) * _sig(p, 1.0) for i, p in enumerate(parts)) / scale)\n"
        "def _snaps(n, m, seed):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    t = np.linspace(0.0, 1.0, m)\n"
        "    x = np.linspace(0.0, 1.0, n)[:, None]\n"
        "    base = (np.sin(np.pi * x) * (1.0 + 0.7 * np.cos(3.0 * t))\n"
        "            + 0.35 * np.sin(2.0 * np.pi * x) * np.sin(5.0 * t)\n"
        "            + 0.08 * np.sin(3.0 * np.pi * x) * np.cos(11.0 * t))\n"
        "    return base + 0.02 * rng.standard_normal((n, m)) + 0.4\n"
    )
    return [
        # Case 1: a smooth, well separated spectrum at a moderate tolerance.
        {
            "setup": reducer + "snaps = _snaps(24, 60, 1)\ntol = 1e-3\n",
            "call": "_sigt(block_pod_basis(snaps, tol), 1e1)",
            "gold_call": "_sigt(_oracle_block_pod_basis(snaps, tol), 1e1)",
        },
        # Case 2: a tighter tolerance retains more modes in both blocks.
        {
            "setup": reducer + "snaps = _snaps(24, 60, 1)\ntol = 1e-6\n",
            "call": "_sigt(block_pod_basis(snaps, tol), 1e1)",
            "gold_call": "_sigt(_oracle_block_pod_basis(snaps, tol), 1e1)",
        },
        # Case 3: a loose tolerance at the one-mode boundary.
        {
            "setup": reducer + "snaps = _snaps(16, 40, 7)\ntol = 0.5\n",
            "call": "_sigt(block_pod_basis(snaps, tol), 1e0)",
            "gold_call": "_sigt(_oracle_block_pod_basis(snaps, tol), 1e0)",
        },
        # --- Decisive: the two blocks are truncated independently, the basis is
        #     block diagonal with orthonormal columns, and the auxiliary block
        #     is built from the squared snapshots rather than from a copy of the
        #     temperature snapshots.
        {
            "setup": reducer + (
                "def structure():\n"
                "    snaps = _snaps(20, 50, 3)\n"
                "    basis, n1, n2 = block_pod_basis(snaps, 1e-4)\n"
                "    n = snaps.shape[0]\n"
                "    off = (float(np.max(np.abs(basis[:n, n1:])))\n"
                "           + float(np.max(np.abs(basis[n:, :n1]))))\n"
                "    gram = basis.T @ basis\n"
                "    orth = float(np.max(np.abs(gram - np.eye(n1 + n2))))\n"
                "    lower = basis[n:, n1:]\n"
                "    aux = snaps * snaps\n"
                "    captured = float(np.sum((lower @ (lower.T @ aux) - aux) ** 2) / np.sum(aux ** 2))\n"
                "    copied = basis[:n, :n1]\n"
                "    distinct = float(np.max(np.abs(np.abs(lower[:, 0]) - np.abs(copied[:, 0]))))\n"
                "    return (int(off < 1e-14) + 2 * int(orth < 1e-10)\n"
                "            + 4 * int(captured < 1e-4) + 8 * int(distinct > 1e-6))\n"
            ),
            "call": "structure()",
            "gold_call": "15",
        },
        # --- Decisive: truncation follows the squared singular values, so a
        #     spectrum whose singular values decay slowly but whose squares
        #     decay quickly keeps exactly one mode per block.
        {
            "setup": reducer + (
                "def energy_rule():\n"
                "    n, m = 12, 12\n"
                "    u, _ = np.linalg.qr(np.random.default_rng(9).standard_normal((n, n)))\n"
                "    v, _ = np.linalg.qr(np.random.default_rng(10).standard_normal((m, m)))\n"
                "    sv = np.array([1.0, 0.02] + [1e-6] * (min(n, m) - 2))\n"
                "    mat = u[:, :min(n, m)] @ np.diag(sv) @ v[:, :min(n, m)].T\n"
                "    _, n1, _ = block_pod_basis(mat, 1e-3)\n"
                "    _, n1b, _ = block_pod_basis(mat, 1e-5)\n"
                "    return int(n1) + 10 * int(n1b)\n"
            ),
            "call": "energy_rule()",
            "gold_call": "21",
        },
        # Case 6: invalid tolerance at the closed upper end.
        {
            "setup": reducer + (
                "snaps = _snaps(8, 12, 2)\n"
                "def run_model():\n"
                "    try:\n"
                "        block_pod_basis(snaps, 1.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_block_pod_basis(snaps, 1.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 7: invalid snapshot rank.
        {
            "setup": reducer + (
                "flat = np.linspace(0.0, 1.0, 10)\n"
                "def run_model():\n"
                "    try:\n"
                "        block_pod_basis(flat, 1e-4)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_block_pod_basis(flat, 1e-4)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
