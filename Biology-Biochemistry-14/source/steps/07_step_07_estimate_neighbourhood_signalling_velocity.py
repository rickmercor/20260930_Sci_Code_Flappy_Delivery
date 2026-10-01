"""
Compose every earlier step to obtain the rate of change of diffusible ligand-receptor signalling for the neighbourhood of one cell.

Combining where a secreted ligand comes from, how fast its producers and the receptor-bearing cells are changing expression, and who the neighbours of a cell are tells whether signalling around that cell is strengthening or weakening.

Returns
-------
float: the neighbourhood-averaged signalling velocity of the chosen cell.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_neighbourhood_signalling_velocity(
    positions: "np.ndarray",
    ligand_counts: "np.ndarray",
    ligand_kinetics: tuple,
    receptor_subunit_counts: "np.ndarray",
    receptor_subunit_kinetics: "np.ndarray",
    cell_index: int,
    radius: float = 200.0,
    tail_mass: float = 1e-9,
) -> float:
    """Return the signalling velocity of one cell's neighbourhood for a diffusible ligand.

    ``ligand_counts[k]`` and ``receptor_subunit_counts[j, k]`` contain the two
    RNA species for the ligand and receptor subunits, with their fitted
    kinetics supplied in the corresponding rate arguments. Compose the six
    preceding public functions as the building blocks of the article's
    end-to-end calculation. Preserve their array orientations and scientific
    conventions, and return the resulting neighbourhood signalling velocity
    for ``cell_index`` without duplicating their internal algorithms here.

    Parameters
    ----------
    positions : np.ndarray
        Float array of shape ``(n, 2)`` of cell centroids.
    ligand_counts : np.ndarray
        Float array of shape ``(n, 2)``: unspliced, spliced.
    ligand_kinetics : tuple
        ``(transcription_rate, splicing_rate, degradation_rate)`` of the ligand.
    receptor_subunit_counts : np.ndarray
        Float array of shape ``(m, n, 2)``, ``m >= 1``: unspliced, spliced.
    receptor_subunit_kinetics : np.ndarray
        Float array of shape ``(m, 3)`` of subunit rates, in the ligand's order.
    cell_index : int
        Zero-based index of the cell whose neighbourhood is reported.
    radius : float
        Neighbourhood radius of the diffusion kernel.
    tail_mass : float
        Fraction of a release allowed beyond ``radius``.

    Returns
    -------
    float
        The neighbourhood signalling velocity of cell ``cell_index``.

    Raises
    ------
    ValueError
        If an array does not have the stated shape (the kinetics included),
        if a kinetic rate is a boolean, if ``cell_index`` is not an integer
        in ``[0, n)`` (booleans are rejected), or if any composed step
        rejects its input.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_estimate_neighbourhood_signalling_velocity(
    positions: "np.ndarray",
    ligand_counts: "np.ndarray",
    ligand_kinetics: tuple,
    receptor_subunit_counts: "np.ndarray",
    receptor_subunit_kinetics: "np.ndarray",
    cell_index: int,
    radius: float = 200.0,
    tail_mass: float = 1e-9,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    def _has_boolean(values):
        return any(isinstance(v, (bool, np.bool_)) for v in np.asarray(values, dtype=object).ravel())

    try:
        if _has_boolean(ligand_kinetics) or _has_boolean(receptor_subunit_kinetics):
            raise ValueError("kinetic rates must be numbers, not booleans")
        points = np.array(positions, dtype=float)
        ligand = np.array(ligand_counts, dtype=float)
        kinetics = np.array(ligand_kinetics, dtype=float)
        receptor = np.array(receptor_subunit_counts, dtype=float)
        subunit_kinetics = np.array(receptor_subunit_kinetics, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("inputs must be numeric arrays") from None
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("positions must have shape (n, 2)")
    n = points.shape[0]
    if ligand.shape != (n, 2) or kinetics.shape != (3,):
        raise ValueError("ligand_counts must be (n, 2) and ligand_kinetics must hold three rates")
    if receptor.ndim != 3 or receptor.shape[0] < 1 or receptor.shape[1:] != (n, 2):
        raise ValueError("receptor_subunit_counts must have shape (m, n, 2) with m >= 1")
    if subunit_kinetics.shape != (receptor.shape[0], 3):
        raise ValueError("receptor_subunit_kinetics must have shape (m, 3)")
    if not (_is_integer(cell_index) and 0 <= cell_index < n):
        raise ValueError("cell_index must be an integer in [0, n)")
    neighbourhoods = _oracle_build_delaunay_neighbourhoods(points)
    weights = _oracle_compute_ligand_sharing_weights(points, radius, tail_mass)
    ligand_rates = _oracle_compute_total_expression_rates(
        ligand[:, 0], ligand[:, 1], float(kinetics[0]), float(kinetics[1]), float(kinetics[2])
    )
    subunit_rates = np.stack([
        _oracle_compute_total_expression_rates(
            receptor[j, :, 0], receptor[j, :, 1],
            float(subunit_kinetics[j, 0]), float(subunit_kinetics[j, 1]), float(subunit_kinetics[j, 2]),
        )
        for j in range(receptor.shape[0])
    ])
    terms = _oracle_compute_cell_signalling_velocity_terms(weights, ligand_rates, subunit_rates)
    neighbourhood_terms = _oracle_average_over_neighbourhoods(terms, neighbourhoods)
    velocity = float(np.sum(neighbourhood_terms[int(cell_index)]))
    if not np.isfinite(velocity):
        raise ValueError("the neighbourhood signalling velocity is not finite")
    return velocity

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    tissue = (
        "import numpy as np\n"
        "from scipy.spatial import Delaunay\n"
        "P = np.array([[0.0, 0.0], [9.0, 1.5], [3.5, 8.0], [12.5, 9.5], [21.0, 2.0],\n"
        "              [26.5, 12.0], [17.0, 18.5], [6.0, 17.0], [30.5, -3.5]])\n"
        "LIG = np.array([[0.4, 5.2], [0.3, 6.1], [0.5, 4.8], [0.6, 3.9], [0.2, 1.1],\n"
        "                [0.1, 0.4], [0.3, 2.2], [0.5, 5.5], [0.1, 0.2]])\n"
        "KIN = (1.1, 0.8, 0.3)\n"
        "REC = np.array([[[1.1, 4.0], [1.2, 3.8], [0.9, 4.2], [1.3, 4.1], [1.0, 3.9],\n"
        "                 [1.2, 4.4], [1.1, 4.0], [1.0, 3.7], [0.9, 4.3]],\n"
        "                [[0.1, 0.2], [0.0, 0.3], [0.2, 0.1], [0.4, 0.8], [0.8, 1.5],\n"
        "                 [1.1, 2.1], [0.6, 1.0], [0.1, 0.2], [1.2, 1.9]]])\n"
        "RKIN = np.array([[1.0, 0.8, 0.25], [0.9, 0.6, 0.3]])\n"
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
            "setup": tissue,
            "call": "estimate_neighbourhood_signalling_velocity(P, LIG, KIN, REC, RKIN, 3, 15.0, 1e-3)",
            "gold_call": "_oracle_estimate_neighbourhood_signalling_velocity(P, LIG, KIN, REC, RKIN, 3, 15.0, 1e-3)",
        },
        {
            "setup": tissue,
            "call": "estimate_neighbourhood_signalling_velocity(P, LIG, KIN, REC, RKIN, 4)",
            "gold_call": "_oracle_estimate_neighbourhood_signalling_velocity(P, LIG, KIN, REC, RKIN, 4)",
        },
        {
            "setup": tissue,
            "call": "estimate_neighbourhood_signalling_velocity(P, LIG, KIN, REC, RKIN, 0, 8.0, 0.05)",
            "gold_call": "_oracle_estimate_neighbourhood_signalling_velocity(P, LIG, KIN, REC, RKIN, 0, 8.0, 0.05)",
        },
        {
            "setup": tissue,
            "call": "estimate_neighbourhood_signalling_velocity(P, LIG, KIN, REC[1:], RKIN[1:], 5, 25.0, 1e-4)",
            "gold_call": "_oracle_estimate_neighbourhood_signalling_velocity(P, LIG, KIN, REC[1:], RKIN[1:], 5, 25.0, 1e-4)",
        },
        {
            "setup": tissue + status,
            "call": "_status(lambda: estimate_neighbourhood_signalling_velocity(P, LIG, KIN, REC, RKIN, 9))",
            "gold_call": "_status(lambda: _oracle_estimate_neighbourhood_signalling_velocity(P, LIG, KIN, REC, RKIN, 9))",
        },
        {
            "setup": tissue + status,
            "call": "_status(lambda: estimate_neighbourhood_signalling_velocity(P, LIG, KIN, REC, RKIN, 2.0))",
            "gold_call": "_status(lambda: _oracle_estimate_neighbourhood_signalling_velocity(P, LIG, KIN, REC, RKIN, 2.0))",
        },
        {
            "setup": tissue + status,
            "call": "_status(lambda: estimate_neighbourhood_signalling_velocity(P, LIG, KIN, REC, RKIN[:1], 2))",
            "gold_call": "_status(lambda: _oracle_estimate_neighbourhood_signalling_velocity(P, LIG, KIN, REC, RKIN[:1], 2))",
        },
        {
            "setup": tissue + status,
            "call": "_status(lambda: estimate_neighbourhood_signalling_velocity(P, LIG, (True, 0.8, 0.3), REC, RKIN, 2))",
            "gold_call": "_status(lambda: _oracle_estimate_neighbourhood_signalling_velocity(P, LIG, (True, 0.8, 0.3), REC, RKIN, 2))",
        },
    ]
