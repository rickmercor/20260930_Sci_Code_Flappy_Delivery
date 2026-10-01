"""
Compute, for every sending cell, the fraction of its secreted ligand that each cell of the tissue receives.

A finite-radius Gaussian transport operator represents how a diffusible ligand moves from producing cells to receiving cells. Its orientation, support and normalisation must preserve the physical meaning of a released amount.

Returns
-------
np.ndarray: (n, n) sender-by-receiver ligand transport fractions.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_ligand_sharing_weights(
    positions: "np.ndarray",
    radius: float,
    tail_mass: float,
) -> "np.ndarray":
    """Return the fractions of each sender's diffusible ligand received by every cell.

    Build the framework's finite-radius Gaussian transport operator from the
    supplied cell centroids and diffusion parameters. Entry ``[k, i]`` is the
    fraction of material originating at cell ``k`` that is assigned to cell
    ``i``; the finite support includes receivers exactly ``radius`` away.
    Apply the source method's conventions for kernel bandwidth,
    neighbourhood membership, self contribution and conservation of each
    sender's released amount.

    Parameters
    ----------
    positions : np.ndarray
        Float array of shape ``(n, 2)`` with ``n >= 1`` centroid coordinates.
    radius : float
        Neighbourhood radius, in the unit of ``positions``.
    tail_mass : float
        Fraction of a release allowed beyond ``radius``; fixes the bandwidth.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n, n)``; rows identify the material's cell of
        origin and columns identify the receiving cell.

    Raises
    ------
    ValueError
        If ``positions`` is not a finite numeric array of shape ``(n, 2)``
        with ``n >= 1``, if ``radius`` is not a finite number of at least
        the self-distance ``1e-9``, or if ``tail_mass`` is not a finite
        number strictly between 0 and 1 (booleans are rejected for both).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_ligand_sharing_weights(
    positions: "np.ndarray",
    radius: float,
    tail_mass: float,
) -> "np.ndarray":
    """Reference implementation (Gaussian kernel normalised over each sender)."""
    import numpy as np

    try:
        points = np.array(positions, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("positions must be a numeric array") from None
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
        raise ValueError("positions must have shape (n, 2) with n >= 1")
    if not np.all(np.isfinite(points)):
        raise ValueError("positions must be finite")
    sigma = _oracle_derive_diffusion_bandwidth(radius, tail_mass)
    if float(radius) < 1e-9:
        raise ValueError("radius must be at least the self-distance 1e-9")
    offsets = points[:, None, :] - points[None, :, :]
    distance = np.sqrt(np.sum(offsets ** 2, axis=2))
    np.fill_diagonal(distance, 1e-9)
    density = np.exp(-distance ** 2 / (2.0 * sigma ** 2)) / (2.0 * np.pi * sigma ** 2)
    density[distance > float(radius)] = 0.0
    # Row k holds the kernel from sender k to every receiver; dividing by the
    # row total shares the sender's release over its own neighbourhood.
    return density / density.sum(axis=1, keepdims=True)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    head = (
        "import numpy as np\n"
        "def _sig(a, n):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (n, n):\n"
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
    cluster = (
        "P = np.array([[0.0, 0.0], [2.1, 0.4], [0.8, 1.9], [2.6, 2.3], [9.5, 1.0], [6.4, 7.2]])\n"
    )
    return [
        {
            "setup": head + cluster,
            "call": "_sig(compute_ligand_sharing_weights(P, 12.0, 0.05), 6)",
            "gold_call": "_sig(_oracle_compute_ligand_sharing_weights(P, 12.0, 0.05), 6)",
        },
        {
            "setup": head + cluster,
            "call": "float(np.sum(compute_ligand_sharing_weights(P, 12.0, 0.05)[:, 4] * np.arange(1.0, 7.0)))",
            "gold_call": "float(np.sum(_oracle_compute_ligand_sharing_weights(P, 12.0, 0.05)[:, 4] * np.arange(1.0, 7.0)))",
        },
        {
            "setup": head,
            "call": "_sig(compute_ligand_sharing_weights(np.array([[0.0, 0.0], [3.0, 4.0], [20.0, 0.0]]), 5.0, 0.3), 3)",
            "gold_call": "_sig(_oracle_compute_ligand_sharing_weights(np.array([[0.0, 0.0], [3.0, 4.0], [20.0, 0.0]]), 5.0, 0.3), 3)",
        },
        {
            "setup": head,
            "call": "_sig(compute_ligand_sharing_weights(np.array([[4.0, -2.0]]), 1.0, 0.5), 1)",
            "gold_call": "_sig(_oracle_compute_ligand_sharing_weights(np.array([[4.0, -2.0]]), 1.0, 0.5), 1)",
        },
        {
            "setup": head + cluster,
            "call": "_sig(compute_ligand_sharing_weights(P, 100.0, 1e-9), 6)",
            "gold_call": "_sig(_oracle_compute_ligand_sharing_weights(P, 100.0, 1e-9), 6)",
        },
        {
            "setup": head + cluster + status,
            "call": "_status(lambda: compute_ligand_sharing_weights(P, -1.0, 0.05))",
            "gold_call": "_status(lambda: _oracle_compute_ligand_sharing_weights(P, -1.0, 0.05))",
        },
        {
            "setup": head + cluster + status,
            "call": "_status(lambda: compute_ligand_sharing_weights(P, 12.0, 0.0))",
            "gold_call": "_status(lambda: _oracle_compute_ligand_sharing_weights(P, 12.0, 0.0))",
        },
        {
            "setup": head + status,
            "call": "_status(lambda: compute_ligand_sharing_weights(np.array([[0.0, 1.0, 2.0]]), 5.0, 0.1))",
            "gold_call": "_status(lambda: _oracle_compute_ligand_sharing_weights(np.array([[0.0, 1.0, 2.0]]), 5.0, 0.1))",
        },
    ]
