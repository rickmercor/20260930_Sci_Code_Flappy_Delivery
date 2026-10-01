"""
Split the rate of change of each cell's ligand-receptor signalling score into the part driven by the changing ligand supply and the part driven by the changing receptor.

A cell's signalling score couples the ligand reaching it and the receptor it carries multiplicatively. Its instantaneous change can be attributed to ligand dynamics and receptor dynamics without treating either contribution as the whole signal.

Returns
-------
np.ndarray: (n, 2) array of [ligand-driven part, receptor-driven part] of dS/dt.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_cell_signalling_velocity_terms(
    sharing_weights: "np.ndarray",
    ligand_rates: "np.ndarray",
    receptor_subunit_rates: "np.ndarray",
) -> "np.ndarray":
    """Return the ligand-driven and receptor-driven parts of each cell's signalling velocity.

    ``ligand_rates[k]`` holds sender ``k``'s ligand state and its rate, while
    ``sharing_weights[k, i]`` maps material originating at sender ``k`` to
    receiver ``i``. ``receptor_subunit_rates[j, i]`` holds the corresponding
    state and rate for receptor subunit ``j`` at receiver ``i``. Construct the
    paper-defined cell signalling score and differentiate it into the
    contribution attributable to ligand dynamics and the contribution
    attributable to receptor dynamics. Apply the framework's multi-subunit
    receptor convention at the receiving cell.

    Parameters
    ----------
    sharing_weights : np.ndarray
        Finite nonnegative float array of shape ``(n, n)``, senders by receivers.
    ligand_rates : np.ndarray
        Finite float array of shape ``(n, 2)`` with nonnegative column 0.
    receptor_subunit_rates : np.ndarray
        Finite float array of shape ``(m, n, 2)``, ``m >= 1``, with
        nonnegative abundances in the last axis' entry 0.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n, 2)``: ligand-driven contribution, then
        receptor-driven contribution.

    Raises
    ------
    ValueError
        If any input is not a finite numeric array of the stated shape, if
        ``sharing_weights`` has a negative entry, or if a ligand or receptor
        subunit abundance is negative.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_cell_signalling_velocity_terms(
    sharing_weights: "np.ndarray",
    ligand_rates: "np.ndarray",
    receptor_subunit_rates: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation (shared ligand, summed receptor subunits, product rule)."""
    import numpy as np

    def _array(values, name):
        try:
            array = np.array(values, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a numeric array") from None
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        return array

    weights = _array(sharing_weights, "sharing_weights")
    ligand = _array(ligand_rates, "ligand_rates")
    receptor = _array(receptor_subunit_rates, "receptor_subunit_rates")
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1] or weights.shape[0] < 1:
        raise ValueError("sharing_weights must have shape (n, n) with n >= 1")
    n = weights.shape[0]
    if np.any(weights < 0.0):
        raise ValueError("sharing_weights must be nonnegative")
    if ligand.shape != (n, 2) or np.any(ligand[:, 0] < 0.0):
        raise ValueError("ligand_rates must have shape (n, 2) with nonnegative abundances")
    if receptor.ndim != 3 or receptor.shape[0] < 1 or receptor.shape[1:] != (n, 2):
        raise ValueError("receptor_subunit_rates must have shape (m, n, 2) with m >= 1")
    if np.any(receptor[:, :, 0] < 0.0):
        raise ValueError("receptor subunit abundances must be nonnegative")
    # Receiver i collects column i of the sender-by-receiver weights.
    available = weights.T @ ligand[:, 0]
    available_rate = weights.T @ ligand[:, 1]
    receptor_total = receptor[:, :, 0].sum(axis=0)
    receptor_rate = receptor[:, :, 1].sum(axis=0)
    return np.column_stack([receptor_total * available_rate, available * receptor_rate])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    head = (
        "import numpy as np\n"
        "def _sig(a, n):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (n, 2):\n"
        "        return -1.0\n"
        "    k = np.arange(a.size, dtype=float)\n"
        "    return float(np.sum(np.abs(a)) + np.sum(a.ravel() * np.cos(0.7 * k)))\n"
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
    tissue = (
        "W = np.array([[0.55, 0.30, 0.15, 0.00],\n"
        "              [0.20, 0.50, 0.20, 0.10],\n"
        "              [0.05, 0.25, 0.45, 0.25],\n"
        "              [0.00, 0.10, 0.30, 0.60]])\n"
        "lig = np.array([[4.2, -0.6], [3.1, -0.2], [1.0, 0.4], [0.3, 0.7]])\n"
        "rec = np.array([[[4.8, 0.05], [5.1, -0.10], [4.4, 0.02], [5.0, 0.08]],\n"
        "                [[0.2, 0.10], [0.9, 0.35], [1.6, 0.50], [2.2, 0.60]]])\n"
    )
    return [
        {
            "setup": head + tissue,
            "call": "_sig(compute_cell_signalling_velocity_terms(W, lig, rec), 4)",
            "gold_call": "_sig(_oracle_compute_cell_signalling_velocity_terms(W, lig, rec), 4)",
        },
        {
            "setup": head + tissue,
            "call": "_sig(compute_cell_signalling_velocity_terms(W, lig, rec[1:]), 4)",
            "gold_call": "_sig(_oracle_compute_cell_signalling_velocity_terms(W, lig, rec[1:]), 4)",
        },
        {
            "setup": head + tissue,
            "call": "_sig(compute_cell_signalling_velocity_terms(np.eye(4), lig, rec), 4)",
            "gold_call": "_sig(_oracle_compute_cell_signalling_velocity_terms(np.eye(4), lig, rec), 4)",
        },
        {
            "setup": head + tissue,
            "call": "float(np.sum(compute_cell_signalling_velocity_terms(W, lig, rec).sum(axis=1) * np.array([1.0, -2.0, 3.0, -4.0])))",
            "gold_call": "float(np.sum(_oracle_compute_cell_signalling_velocity_terms(W, lig, rec).sum(axis=1) * np.array([1.0, -2.0, 3.0, -4.0])))",
        },
        {
            "setup": head,
            "call": "_sig(compute_cell_signalling_velocity_terms(np.array([[1.0]]), np.array([[2.0, -0.5]]), np.array([[[3.0, 0.25]], [[1.0, 0.75]]])), 1)",
            "gold_call": "_sig(_oracle_compute_cell_signalling_velocity_terms(np.array([[1.0]]), np.array([[2.0, -0.5]]), np.array([[[3.0, 0.25]], [[1.0, 0.75]]])), 1)",
        },
        {
            "setup": head + tissue + status,
            "call": "_status(lambda: compute_cell_signalling_velocity_terms(W[:, :3], lig, rec))",
            "gold_call": "_status(lambda: _oracle_compute_cell_signalling_velocity_terms(W[:, :3], lig, rec))",
        },
        {
            "setup": head + tissue + status,
            "call": "_status(lambda: compute_cell_signalling_velocity_terms(W, lig * np.array([-1.0, 1.0]), rec))",
            "gold_call": "_status(lambda: _oracle_compute_cell_signalling_velocity_terms(W, lig * np.array([-1.0, 1.0]), rec))",
        },
        {
            "setup": head + tissue + status,
            "call": "_status(lambda: compute_cell_signalling_velocity_terms(W, lig, rec[:, :3, :]))",
            "gold_call": "_status(lambda: _oracle_compute_cell_signalling_velocity_terms(W, lig, rec[:, :3, :]))",
        },
    ]
