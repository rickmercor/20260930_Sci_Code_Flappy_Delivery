"""
Step name: 02_build_local_descriptors
Step description: Convert every trajectory frame into an adsorbate-centred descriptor vector built from smoothly truncated radial basis functions over the tagged hydrogen's neighbours.

Step scientific background: Coarse-graining a catalytic simulation onto the reactants themselves requires a per-adsorbate representation of the local environment rather than a global coordinate. A radial expansion of the neighbour distances, damped to zero at an interaction cutoff, is smooth in the configuration and expressive enough to separate distinct adsorption environments.

Returns
-------
np.ndarray: float descriptor matrix of shape (T, 3 * n_radial).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def build_local_descriptors(trajectory: np.ndarray, n_radial: int = 7,
                            r_cut: float = 7.0, n_images: int = 2) -> np.ndarray:
    """Return the adsorbate-centred descriptor matrix of a trajectory.

    The rhodium row has period ``L = 8.10`` with three surface atoms per
    period at lateral offsets ``0``, ``L / 3`` and ``2 * L / 3``. The atom at
    offset ``0`` is the under-coordinated edge atom and sits ``0.30``
    angstrom above the plane of the other two; the tagged hydrogen sits
    ``1.00`` angstrom above that plane. Rh images are taken for cell indices
    ``-n_images`` through ``+n_images``, so the distance from the hydrogen at
    lateral coordinate ``s`` to an image at lateral position ``p`` and height
    ``z`` is ``sqrt((1.00 - z) ** 2 + (s - p) ** 2)``.

    Every neighbour distance ``d`` enters through the cosine cutoff
    ``f(d) = 0.5 * (1 + cos(pi * d / r_cut))`` for ``d < r_cut`` and ``0``
    otherwise, and through ``n_radial`` Gaussian bins centred at
    ``mu_k = (k + 0.5) * r_cut / n_radial`` with common width
    ``sigma = r_cut / (2 * n_radial)``. Bin ``k`` of the three channels is

        Rh channel : sum over Rh images of f(d) * exp(-(d - mu_k) ** 2
                     / (2 * sigma ** 2))
        H channel  : sum over r1 and r2 of f(r) * exp(-(r - mu_k) ** 2
                     / (2 * sigma ** 2))
        HH channel : f(r1) * f(r2) * exp(-(0.5 * (r1 + r2) - mu_k) ** 2
                     / (2 * sigma ** 2))

    The output columns are ordered Rh channel first, then H channel, then HH
    channel, each in increasing bin index.

    Parameters
    ----------
    trajectory : np.ndarray
        Float array of shape ``(T, 3)`` with rows ``(r1, r2, s)`` and ``T``
        at least 1.
    n_radial : int
        Number of radial bins per channel, at least 1.
    r_cut : float
        Interaction cutoff in angstrom, positive.
    n_images : int
        Number of periodic cell images taken on each side, at least 0.

    Returns
    -------
    np.ndarray
        Float array of shape ``(T, 3 * n_radial)``.

    Raises
    ------
    ValueError
        If ``trajectory`` is not a non-empty real array of shape ``(T, 3)``,
        if either hydrogen distance is not positive, if ``n_radial`` is not an
        integer of at least 1, if ``n_images`` is not an integer of at least
        0, or if ``r_cut`` is not a positive finite float.
    """
    return descriptors

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _descriptor_geometry() -> tuple:
    """Return the Rh row geometry used by the adsorbate-centred descriptors."""
    return 8.10, 0.30, 1.00   # lattice period, edge-atom rise, adsorbate height


def _cosine_cutoff(distance, r_cut: float):
    """Return the cosine cutoff, zero at and beyond ``r_cut``."""
    import numpy as np
    inside = distance < r_cut
    return np.where(inside, 0.5 * (1.0 + np.cos(np.pi * np.minimum(distance, r_cut) / r_cut)), 0.0)


def _rh_image_positions(n_images: int) -> tuple:
    """Return the lateral positions and heights of the Rh images."""
    import numpy as np
    period, rise, _ = _descriptor_geometry()
    spacing = period / 3.0
    cells = np.arange(-int(n_images), int(n_images) + 1, dtype=float)
    offsets = np.array([0.0, spacing, 2.0 * spacing])
    lateral = (cells[:, None] * period + offsets[None, :]).ravel()
    heights = np.tile(np.array([rise, 0.0, 0.0]), cells.size)
    return lateral, heights


import numpy as np
def _oracle_build_local_descriptors(trajectory: np.ndarray, n_radial: int = 7,
                                    r_cut: float = 7.0, n_images: int = 2) -> np.ndarray:
    """Reference implementation (three smoothly truncated radial channels)."""
    import numpy as np

    if not (_is_integer(n_radial) and int(n_radial) >= 1):
        raise ValueError("n_radial must be an integer of at least 1")
    if not (_is_integer(n_images) and int(n_images) >= 0):
        raise ValueError("n_images must be an integer of at least 0")
    if not _is_positive_float(r_cut):
        raise ValueError("r_cut must be a positive finite float")
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim != 2 or traj.shape[1] != 3 or traj.shape[0] < 1:
        raise ValueError("trajectory must be a non-empty array of shape (T, 3)")
    if not np.all(np.isfinite(traj)):
        raise ValueError("trajectory must be finite")
    if np.any(traj[:, :2] <= 0.0):
        raise ValueError("hydrogen-hydrogen distances must be positive")

    n_radial, r_cut = int(n_radial), float(r_cut)
    _, _, height = _descriptor_geometry()
    centres = (np.arange(n_radial, dtype=float) + 0.5) * r_cut / n_radial
    sigma = r_cut / (2.0 * n_radial)
    two_sigma_sq = 2.0 * sigma ** 2

    r1, r2, s = traj[:, 0], traj[:, 1], traj[:, 2]
    lateral, heights = _rh_image_positions(int(n_images))
    d_rh = np.sqrt((height - heights)[None, :] ** 2 + (s[:, None] - lateral[None, :]) ** 2)
    cut_rh = _cosine_cutoff(d_rh, r_cut)
    cut_1, cut_2 = _cosine_cutoff(r1, r_cut), _cosine_cutoff(r2, r_cut)
    mean_hh = 0.5 * (r1 + r2)

    descriptors = np.empty((traj.shape[0], 3 * n_radial), dtype=float)
    for k, mu in enumerate(centres):
        descriptors[:, k] = np.sum(cut_rh * np.exp(-((d_rh - mu) ** 2) / two_sigma_sq), axis=1)
        descriptors[:, n_radial + k] = (cut_1 * np.exp(-((r1 - mu) ** 2) / two_sigma_sq)
                                        + cut_2 * np.exp(-((r2 - mu) ** 2) / two_sigma_sq))
        descriptors[:, 2 * n_radial + k] = cut_1 * cut_2 * np.exp(-((mean_hh - mu) ** 2) / two_sigma_sq)
    return descriptors

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
        "traj = simulate_surface_trajectory(1200, 5)\n"
        "traj_gold = _oracle_simulate_surface_trajectory(1200, 5)\n"
        "grid = np.column_stack([np.linspace(0.6, 3.4, 40),\n"
        "                        np.linspace(3.4, 0.6, 40),\n"
        "                        np.linspace(0.0, 8.1, 40, endpoint=False)])\n"
    )
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "bad = np.array([[1.0, 2.0], [3.0, 4.0]])\n"
    )
    return [
        {
            "setup": digest,
            "call": "_sig(build_local_descriptors(traj))",
            "gold_call": "_sig(_oracle_build_local_descriptors(traj_gold))",
        },
        {
            "setup": digest,
            "call": "_sig(build_local_descriptors(grid, n_radial=4, r_cut=5.0, n_images=1))",
            "gold_call": "_sig(_oracle_build_local_descriptors(grid, n_radial=4, r_cut=5.0, n_images=1))",
        },
        {
            "setup": digest,
            "call": "_sig(build_local_descriptors(np.array([[0.80, 0.80, 0.0]])))",
            "gold_call": "_sig(_oracle_build_local_descriptors(np.array([[0.80, 0.80, 0.0]])))",
        },
        {
            "setup": digest,
            "call": "float(np.sum(build_local_descriptors(grid, n_radial=1, r_cut=2.0, n_images=0)))",
            "gold_call": "float(np.sum(_oracle_build_local_descriptors(grid, n_radial=1, r_cut=2.0, n_images=0)))",
        },
        {
            "setup": digest,
            "call": "float(build_local_descriptors(np.array([[0.80, 2.60, 4.05]]))[0, 14])",
            "gold_call": "float(_oracle_build_local_descriptors(np.array([[0.80, 2.60, 4.05]]))[0, 14])",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_local_descriptors(bad))",
            "gold_call": "_status(lambda: _oracle_build_local_descriptors(bad))",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_local_descriptors(np.array([[0.0, 1.0, 2.0]])))",
            "gold_call": "_status(lambda: _oracle_build_local_descriptors(np.array([[0.0, 1.0, 2.0]])))",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_local_descriptors(np.array([[1.0, 1.0, 0.0]]), n_radial=0))",
            "gold_call": "_status(lambda: _oracle_build_local_descriptors(np.array([[1.0, 1.0, 0.0]]), n_radial=0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_local_descriptors(np.array([[1.0, 1.0, 0.0]]), r_cut=-3.0))",
            "gold_call": "_status(lambda: _oracle_build_local_descriptors(np.array([[1.0, 1.0, 0.0]]), r_cut=-3.0))",
        },
    ]
