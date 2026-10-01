"""
Fit the recovered network and evaluate its unique missing transition.

The outputs expose the fitted parameter vector, residuals, numerical rank and
the physical prediction associated with the absent bipartite edge.

Returns
-------
return parameters, residuals, rank, prediction
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def invert_network_and_predict(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        standard_uncertainties: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> tuple["np.ndarray", "np.ndarray", int, "np.ndarray"]:
    """Fit the gauged network and evaluate its unique absent edge.

    Parameters
    ----------
    line_ids, scans, reported_frequencies, b_lo, b_hi, delta, min_occurrences
        Inputs satisfying the contract of ``build_bipartite_level_network``
        in step 5.
    standard_uncertainties
        Finite strictly positive standard uncertainties aligned with the
        spectral lines and expressed in the same units as their frequencies.

    Returns
    -------
    parameters : np.ndarray
        Fitted vector containing upper energies, nongauge lower energies, and
        the refined block-B offset in that order.
    residuals : np.ndarray
        ``reported_frequencies - A @ parameters`` evaluated through the
        centred system to avoid cancellation.
    rank : int
        Rank of the weighted design matrix.
    prediction : np.ndarray
        Length-four float array containing, in order, the missing physical
        frequency, refined block-B offset, fitted missing-upper energy, and
        fitted missing-lower energy under the selected gauge.

    Raises
    ------
    ValueError
        If an earlier step rejects its inputs, the uncertainty array is not
        aligned, finite and strictly positive, the weighted design matrix is
        rank deficient, or a returned fitted quantity is not finite.

    Parameter ordering convention
    -----------------------------
    Use the upper and lower row ordering defined for build_bipartite_level_network:
    ascending lexicographic order of sorted string-valued incident line-ID tuples
    within each colour class. List upper energies in that order, followed by
    nongauge lower energies in their class order, followed by the scan-B offset.
    """
    return parameters, residuals, rank, prediction

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_invert_network_and_predict(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        standard_uncertainties: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> tuple["np.ndarray", "np.ndarray", int, "np.ndarray"]:
    """Reference implementation for invert_network_and_predict."""
    upper, lower, endpoints, missing, gauge = (
        _oracle_build_bipartite_level_network(
            line_ids, scans, reported_frequencies, b_lo, b_hi,
            delta, min_occurrences))
    reported = np.asarray(reported_frequencies, dtype=float)
    sigma = np.asarray(standard_uncertainties, dtype=float)
    if sigma.ndim != 1 or sigma.shape != reported.shape:
        raise ValueError("standard_uncertainties must be one-dimensional and aligned")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("standard_uncertainties must be finite and strictly positive")

    labels = np.asarray(scans).astype(str)
    n_upper, n_lower = upper.shape[0], lower.shape[0]
    n_parameters = n_upper + n_lower
    design = np.zeros((reported.size, n_parameters), dtype=float)
    lower_columns = {}
    next_column = n_upper
    for lower_index in range(n_lower):
        if lower_index != gauge:
            lower_columns[lower_index] = next_column
            next_column += 1
    offset_column = n_parameters - 1
    for row, (upper_index, lower_index) in enumerate(endpoints):
        design[row, int(upper_index)] = 1.0
        if int(lower_index) != gauge:
            design[row, lower_columns[int(lower_index)]] = -1.0
        design[row, offset_column] = float(labels[row] == "B")

    origin = float(reported[0])
    centred_reported = reported - origin
    weighted_design = design / sigma[:, None]
    weighted_rhs = centred_reported / sigma
    centred_parameters, _, rank, _ = np.linalg.lstsq(
        weighted_design, weighted_rhs, rcond=None)
    rank = int(rank)
    if rank != n_parameters:
        raise ValueError("the weighted design matrix is rank deficient")

    residuals = centred_reported - design @ centred_parameters
    parameters = centred_parameters.copy()
    parameters[:n_upper] += origin
    missing_upper, missing_lower = map(int, missing)
    upper_energy = float(parameters[missing_upper])
    if missing_lower == gauge:
        lower_energy = 0.0
    else:
        lower_before = sum(index != gauge
                           for index in range(missing_lower))
        lower_energy = float(parameters[n_upper + lower_before])
    refined_offset = float(parameters[-1])
    predicted = upper_energy - lower_energy
    prediction = np.asarray(
        [predicted, refined_offset, upper_energy, lower_energy], dtype=float)
    if not np.all(np.isfinite(prediction)):
        raise ValueError("the fitted prediction and endpoints must be finite")
    return (np.asarray(parameters, dtype=float),
            np.asarray(residuals, dtype=float), rank, prediction)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\nids = np.array([f'L{i:02d}' for i in range(1, 20)])\nscans = np.array(['B','A','A','B','A','B','A','B','B','B','A','A','B','A','B','A','B','B','A'])\nfreq = np.array([6291.6002896,6305.3101133,6309.9630221,6314.7284247,6321.1374199,6323.6823052,6333.0911534,6338.3752148,6339.5096148,6352.0897687,6361.4986205,6367.9123463,6374.7508896,6397.8742901,6404.2927505,6422.2346684,6435.9539540,6445.3675344,6451.7765297])\nsigma = 1e-6 * np.array([2.0,1.6,1.4,1.8,1.4,1.9,1.7,2.1,1.5,1.5,1.6,1.8,1.6,1.9,1.7,2.0,1.7,1.8,1.5])\n"
    return [
        {
            'setup': setup,
            'call': 'invert_network_and_predict(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)[0]',
            'gold_call': '_oracle_invert_network_and_predict(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)[0]',
            'tol': 1e-12,
        },
        {
            'setup': setup,
            'call': 'float(invert_network_and_predict(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)[0][-1])',
            'gold_call': 'float(_oracle_invert_network_and_predict(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)[0][-1])',
            'tol': 1e-08,
        },
        {
            'setup': setup,
            'call': 'invert_network_and_predict(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)[3]',
            'gold_call': '_oracle_invert_network_and_predict(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)[3]',
            'tol': 1e-12,
        },
        {
            'setup': setup,
            'call': 'float(invert_network_and_predict(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)[2])',
            'gold_call': 'float(_oracle_invert_network_and_predict(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)[2])',
            'tol': 0.0,
        },
        {
            'setup': setup,
            'call': 'invert_network_and_predict(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)[1]',
            'gold_call': '_oracle_invert_network_and_predict(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)[1]',
            'tol': 1e-09,
        },
        {
            'setup': setup + 'freq = freq + 0.25\n',
            'call': 'invert_network_and_predict(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)[3]',
            'gold_call': '_oracle_invert_network_and_predict(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)[3]',
            'tol': 1e-12,
        },
        {
            'setup': setup + 'sigma = sigma * np.linspace(0.7, 1.3, sigma.size)\n',
            'call': 'invert_network_and_predict(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)[3]',
            'gold_call': '_oracle_invert_network_and_predict(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)[3]',
            'tol': 1e-12,
        },
        {
            'setup': setup + 'sigma[5] = 0.0\n\ndef run_model():\n    try:\n        invert_network_and_predict(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_invert_network_and_predict(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
            'tol': 0.0,
        },
    ]
