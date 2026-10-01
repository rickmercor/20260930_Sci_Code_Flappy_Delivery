"""
Run the complete reconstruction and return the missing-line frequency.

The orchestrator validates agreement among the recurrence, local-closure,
global-topology, bipartition and weighted-inversion representations.

Returns
-------
return predicted_frequency
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reconstruct_missing_transition_frequency(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        standard_uncertainties: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2) -> float:
    """Execute all reconstruction stages and return the held-out frequency.

    Parameters
    ----------
    line_ids
        Unique transition-line identifiers.
    scans
        Scan labels A or B aligned with the lines.
    reported_frequencies
        Finite pairwise-distinct reported line centres.
    standard_uncertainties
        Finite strictly positive standard uncertainties aligned with the
        lines and expressed in the same units as the frequencies.
    b_lo, b_hi
        Closed instrument bracket for the topology-seed offset.
    delta
        Positive first-value clustering tolerance.
    min_occurrences
        Integer retained-cluster threshold of at least 2.

    Returns
    -------
    float
        Physical frequency of the unique unobserved transition.

    Raises
    ------
    ValueError
        If any earlier step rejects its inputs or independently returned
        intermediate representations are inconsistent.
    """
    return predicted_frequency  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_reconstruct_missing_transition_frequency(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        standard_uncertainties: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2) -> float:
    """Reference implementation for reconstruct_missing_transition_frequency."""
    seed, corrected, occurrences, counts = (
        _oracle_cluster_recurring_differences(
            line_ids, scans, reported_frequencies, b_lo, b_hi,
            delta, min_occurrences))
    motifs = _oracle_instantiate_k22_motifs(
        line_ids, scans, reported_frequencies, b_lo, b_hi,
        delta, min_occurrences)
    sharing_pairs = _oracle_derive_local_vertex_constraints(
        line_ids, scans, reported_frequencies, b_lo, b_hi,
        delta, min_occurrences)
    level_incidence = _oracle_stitch_energy_level_cliques(
        line_ids, scans, reported_frequencies, b_lo, b_hi,
        delta, min_occurrences)
    upper, lower, endpoints, missing, gauge = (
        _oracle_build_bipartite_level_network(
            line_ids, scans, reported_frequencies, b_lo, b_hi,
            delta, min_occurrences))
    parameters, residuals, rank, prediction = _oracle_invert_network_and_predict(
        line_ids, scans, reported_frequencies, standard_uncertainties,
        b_lo, b_hi, delta, min_occurrences)

    n_lines = np.asarray(line_ids).size
    if not np.isfinite(seed):
        raise ValueError("the topology seed is not finite")
    if corrected.shape != (n_lines,) or occurrences.shape[0] != int(np.sum(counts)):
        raise ValueError("the retained-difference representation is inconsistent")
    if motifs.ndim != 2 or motifs.shape[1] != 4 or sharing_pairs.shape[1] != 2:
        raise ValueError("the local-motif representation is inconsistent")
    if level_incidence.shape[0] != upper.shape[0] + lower.shape[0]:
        raise ValueError("the stitched levels and colour classes disagree")
    if endpoints.shape != (n_lines, 2) or missing.shape != (2,):
        raise ValueError("the bipartite endpoint representation is inconsistent")
    if not (0 <= int(gauge) < lower.shape[0]):
        raise ValueError("the gauge lower-level index is invalid")
    if parameters.size != upper.shape[0] + lower.shape[0] or rank != parameters.size:
        raise ValueError("the fitted parameter dimension or rank is inconsistent")
    if residuals.shape != (n_lines,):
        raise ValueError("the fitted residual vector is inconsistent")
    if prediction.shape != (4,):
        raise ValueError("the fitted prediction summary is inconsistent")
    if not np.isclose(parameters[-1], prediction[1], rtol=0.0, atol=1e-8):
        raise ValueError("the independently returned fitted offsets disagree")
    if not np.isclose(prediction[0], prediction[2] - prediction[3],
                      rtol=0.0, atol=1e-8):
        raise ValueError("the endpoint difference and prediction disagree")
    if not np.isfinite(prediction[0]):
        raise ValueError("the predicted frequency is not finite")
    return float(prediction[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\nids = np.array([f'L{i:02d}' for i in range(1, 20)])\nscans = np.array(['B','A','A','B','A','B','A','B','B','B','A','A','B','A','B','A','B','B','A'])\nfreq = np.array([6291.6002896,6305.3101133,6309.9630221,6314.7284247,6321.1374199,6323.6823052,6333.0911534,6338.3752148,6339.5096148,6352.0897687,6361.4986205,6367.9123463,6374.7508896,6397.8742901,6404.2927505,6422.2346684,6435.9539540,6445.3675344,6451.7765297])\nsigma = 1e-6 * np.array([2.0,1.6,1.4,1.8,1.4,1.9,1.7,2.1,1.5,1.5,1.6,1.8,1.6,1.9,1.7,2.0,1.7,1.8,1.5])\n"
    return [
        {
            'setup': setup,
            'call': 'float(round(reconstruct_missing_transition_frequency(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6), 7))',
            'gold_call': 'float(round(_oracle_reconstruct_missing_transition_frequency(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6), 7))',
            'tol': 0.0,
        },
        {
            'setup': setup + 'freq = freq + 0.125\n',
            'call': 'float(round(reconstruct_missing_transition_frequency(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6), 7))',
            'gold_call': 'float(round(_oracle_reconstruct_missing_transition_frequency(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6), 7))',
            'tol': 0.0,
        },
        {
            'setup': setup + 'sigma = 2.5 * sigma\n',
            'call': 'float(round(reconstruct_missing_transition_frequency(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6), 7))',
            'gold_call': 'float(round(_oracle_reconstruct_missing_transition_frequency(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6), 7))',
            'tol': 0.0,
        },
        {
            'setup': setup + 'order = np.array([18,0,7,3,11,5,16,1,14,8,6,2,17,10,4,12,9,15,13])\nids, scans, freq, sigma = ids[order], scans[order], freq[order], sigma[order]\n',
            'call': 'float(round(reconstruct_missing_transition_frequency(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6), 7))',
            'gold_call': 'float(round(_oracle_reconstruct_missing_transition_frequency(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6), 7))',
            'tol': 0.0,
        },
        {
            'setup': setup + "freq = freq + 2e-5 * (scans == 'B')\n",
            'call': 'float(round(reconstruct_missing_transition_frequency(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.00462, 0.00492, 8e-6), 7))',
            'gold_call': 'float(round(_oracle_reconstruct_missing_transition_frequency(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.00462, 0.00492, 8e-6), 7))',
            'tol': 0.0,
        },
        {
            'setup': setup + 'sigma[0] = 0.0\n\ndef run_model():\n    try:\n        reconstruct_missing_transition_frequency(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_reconstruct_missing_transition_frequency(ids.copy(), scans.copy(), freq.copy(), sigma.copy(), 0.0046, 0.0049, 8e-6)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
            'tol': 0.0,
        },
    ]
