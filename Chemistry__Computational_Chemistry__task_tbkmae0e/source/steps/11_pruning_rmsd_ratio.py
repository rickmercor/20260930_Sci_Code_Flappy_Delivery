"""
Implement pruning_rmsd_ratio, the end-to-end comparison of two ways of pruning the same input grid for least-squares tensor hypercontraction: it returns the root-mean-square deviation of the integrals reconstructed on a pivoted-Cholesky-pruned grid divided by that on a grid refitted and pruned by non-negative least squares against the overlap matrix.

The pipeline places the same s-type basis on every center, evaluates the exact one- and two-electron integrals it needs, and lays down the atom-centered octahedral input grid, whose points carry equal weights. One pruning refits every weight by non-negative least squares and keeps the points whose weight survives; the other leaves the weights alone and selects points by an incomplete pivoted Cholesky decomposition of the grid metric of the full input grid, at the given relative cutoff.



On each pruned grid the least-squares tensor hypercontraction of the exact repulsion integrals is built and its reconstruction is compared with the exact tensor. Because the least-squares central matrix absorbs any rescaling of the collocation columns, the comparison measures how well each procedure chose its points rather than how it weighted them; a ratio above one means the pivoted-Cholesky grid reproduces the integrals less accurately.

Returns
-------
float: the reconstruction error on the pivoted-Cholesky grid divided by the error on the reweighted grid, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pruning_rmsd_ratio(centers: "np.ndarray", exponents: "np.ndarray", radii: "np.ndarray",
                       cutoff: float, rcond: float) -> float:
    '''Ratio of LS-THC integral RMSDs: pivoted-Cholesky-pruned grid over NNLS-refitted grid.

    Every center carries one normalized s-type primitive Gaussian per exponent, and the
    orbital index is mu = n_exponents * c + k for exponent k on center c. The input grid
    is the atom-centered octahedral grid built from the same centers and radii, and all of
    its points carry equal weights. One grid comes from the non-negative least-squares
    reweighting, solved exactly, and carries its refitted weights; the other comes from the
    pivoted-Cholesky pruning at the given relative cutoff and keeps the equal input
    weights. On each grid the least-squares tensor hypercontraction of the exact repulsion
    integrals is built, inverting the grid metric through the pseudoinverse that discards
    singular values below rcond times the largest, and the error of that factorization is
    the root-mean-square deviation of the reconstructed integrals from the exact ones over
    all entries of the four-index tensor.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n_centers, 3) holding the center positions in bohr. Must contain
        at least one row.
    exponents : np.ndarray
        Array of shape (n_exponents,) holding the exponents shared by every center. Must
        contain at least one entry and every exponent must be strictly positive.
    radii : np.ndarray
        Array of shape (n_radii,) holding the shell radii of the input grid in bohr. Must
        contain at least one entry and every radius must be strictly positive.
    cutoff : float
        Relative pruning threshold of the pivoted-Cholesky selection, with
        0 < cutoff <= 1.
    rcond : float
        Non-negative relative singular-value cutoff for the metric pseudoinverse.

    Returns
    -------
    ratio : float
        Native Python float equal to the RMSD on the Cholesky grid divided by the RMSD on
        the NNLS grid.

    Raises
    ------
    ValueError
        If centers is not a two-dimensional array with three columns and at least one row,
        if exponents is not a one-dimensional array with at least one entry, if any
        exponent or radius is not strictly positive, if radii is not a one-dimensional array
        with at least one entry, if cutoff does not satisfy 0 < cutoff <= 1, or if rcond is
        negative or not finite.
    '''
    return ratio

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _expand_basis(centers, exponents):
    """Per-orbital centers and exponents, exponent index fastest."""
    centers = np.asarray(centers, dtype=float)
    exponents = np.asarray(exponents, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] < 1:
        raise ValueError("centers must have shape (n_centers, 3) with n_centers >= 1")
    if exponents.ndim != 1 or exponents.shape[0] < 1:
        raise ValueError("exponents must have shape (n_exponents,) with n_exponents >= 1")
    return (np.repeat(centers, exponents.shape[0], axis=0),
            np.tile(exponents, centers.shape[0]))


def _thc_reconstruction_rmsd(ao_centers, ao_exponents, points, weights, eri, rcond):
    """RMSD per entry of the LS-THC reconstruction of eri on a weighted grid."""
    collocation = _oracle_thc_collocation_matrix(ao_centers, ao_exponents, points, weights)
    metric = _oracle_thc_grid_metric(collocation)
    core = _oracle_thc_core_matrix(collocation, metric, eri, rcond)
    n_ao = collocation.shape[0]
    pair_density = (collocation[:, None, :] * collocation[None, :, :]).reshape(n_ao * n_ao, -1)
    reconstructed = pair_density @ core @ pair_density.T
    error = eri.reshape(n_ao * n_ao, n_ao * n_ao) - reconstructed
    return float(np.linalg.norm(error) / n_ao ** 2)


def _oracle_pruning_rmsd_ratio(centers: "np.ndarray", exponents: "np.ndarray",
                               radii: "np.ndarray", cutoff: float, rcond: float) -> float:
    cutoff = float(cutoff)
    if not (0.0 < cutoff <= 1.0):
        raise ValueError("cutoff must satisfy 0 < cutoff <= 1")
    ao_centers, ao_exponents = _expand_basis(centers, exponents)
    overlap = _oracle_ao_overlap_matrix(ao_centers, ao_exponents)
    eri = _oracle_ao_eri_tensor(ao_centers, ao_exponents)
    grid_points = _oracle_atom_centered_grid(centers, radii)

    fit_matrix = _oracle_overlap_fit_matrix(ao_centers, ao_exponents, grid_points)
    weights = _oracle_nnls_grid_weights(fit_matrix, overlap)
    nnls_points, nnls_weights = _oracle_prune_zero_weight_points(grid_points, weights)
    nnls_rmsd = _thc_reconstruction_rmsd(ao_centers, ao_exponents, nnls_points, nnls_weights,
                                         eri, rcond)

    unit_weights = np.ones(grid_points.shape[0], dtype=float)
    input_collocation = _oracle_thc_collocation_matrix(ao_centers, ao_exponents, grid_points,
                                                       unit_weights)
    pivots = _oracle_pivoted_cholesky_points(_oracle_thc_grid_metric(input_collocation), cutoff)
    cholesky_rmsd = _thc_reconstruction_rmsd(ao_centers, ao_exponents, grid_points[pivots],
                                             unit_weights[pivots], eri, rcond)
    return float(cholesky_rmsd / nnls_rmsd)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    invalid = """
def run_model():
    try:
        pruning_rmsd_ratio(centers.copy(), exponents.copy(), radii.copy(), cutoff, rcond)
        return 0.0
    except ValueError:
        return 1.0
def run_oracle():
    try:
        _oracle_pruning_rmsd_ratio(centers.copy(), exponents.copy(), radii.copy(), cutoff, rcond)
        return 0.0
    except ValueError:
        return 1.0
"""
    three_centers = """import numpy as np
centers = np.array([[0.0, 0.0, 0.0],
                    [1.3, 0.4, 0.2],
                    [0.3, 1.2, -0.5]], dtype=float)
exponents = np.array([0.25, 0.7, 1.9], dtype=float)
radii = np.array([0.5, 1.3], dtype=float)
rcond = 1e-12
"""
    return [
        # --- Typical: three low-symmetry centers, two shells, cutoff 1e-2 ---
        {
            "setup": three_centers + """
cutoff = 1e-2
""",
            "call": "pruning_rmsd_ratio(centers.copy(), exponents.copy(), radii.copy(), cutoff, rcond)",
            "gold_call": "_oracle_pruning_rmsd_ratio(centers.copy(), exponents.copy(), radii.copy(), cutoff, rcond)",
            "tol": 1e-6,
        },
        # --- Boundary: a single shell radius and a coarse cutoff that keeps few pivots ---
        {
            "setup": """import numpy as np
centers = np.array([[0.0, 0.0, 0.0],
                    [1.1, 0.7, 0.4],
                    [-0.5, 1.0, -0.3]], dtype=float)
exponents = np.array([0.3, 0.8, 2.0], dtype=float)
radii = np.array([0.9], dtype=float)
cutoff = 1e-1
rcond = 1e-12
""",
            "call": "pruning_rmsd_ratio(centers.copy(), exponents.copy(), radii.copy(), cutoff, rcond)",
            "gold_call": "_oracle_pruning_rmsd_ratio(centers.copy(), exponents.copy(), radii.copy(), cutoff, rcond)",
            "tol": 1e-6,
        },
        # --- Representative: the four-center, five-exponent, 100-point production system ---
        {
            "setup": """import numpy as np
centers = np.array([[0.0, 0.0, 0.0],
                    [1.9, 0.0, 0.2],
                    [-0.2, 2.0, 0.1],
                    [0.9, 1.1, 1.75]], dtype=float)
exponents = 0.18 * 2.5 ** np.arange(5)
radii = np.array([0.4, 0.9, 1.7, 3.0], dtype=float)
cutoff = 3.8e-4
rcond = 1e-12
""",
            "call": "pruning_rmsd_ratio(centers.copy(), exponents.copy(), radii.copy(), cutoff, rcond)",
            "gold_call": "_oracle_pruning_rmsd_ratio(centers.copy(), exponents.copy(), radii.copy(), cutoff, rcond)",
            "tol": 1e-6,
        },
        # --- Invalid: a cutoff above one would select no Cholesky pivot ---
        {
            "setup": three_centers + """
cutoff = 1.5
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: a non-positive exponent ---
        {
            "setup": three_centers + """
exponents = np.array([0.25, -0.7, 1.9], dtype=float)
cutoff = 1e-2
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
