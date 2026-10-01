"""
Run the complete source-calibrated colocalisation and collider sensitivity audit.

 

Chain every preceding stage in order. Subset collection identifiers, metadata, and credible-set flags by the retained signal indices before graph analysis. Build the full components from all selected edges and the restricted components from edges whose two endpoints have credible sets. Return the signed restricted-minus-full collider fraction in percentage points.

Large-scale colocalisation auditing couples preprocessing, Bayesian evidence, graph construction, identity diagnosis, and credible-set sensitivity.

Returns
-------
contrast : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_colocalisation_audit(
    log_bf: "np.ndarray",
    collection_id: "np.ndarray",
    tested_variant_mask: "np.ndarray",
    signal_metadata: "np.ndarray",
    has_credible_set: "np.ndarray",
    p1: float,
    p2: float,
    p12: float,
    overlap_min: float,
    h4_threshold: float,
    chunk_size: int,
) -> float:
    """
    Return the end-to-end credible-set collider-rate contrast.
 
    Parameters
    ----------
    log_bf : np.ndarray
        Signal-by-variant log Bayes factors.
    collection_id : np.ndarray
        Collection identifier per signal.
    tested_variant_mask : np.ndarray
        Collection-by-variant tested mask.
    signal_metadata : np.ndarray
        Integer rows [dataset ID, trait ID, signal ID].
    has_credible_set : np.ndarray
        Boolean credible-set flag per signal.
    p1, p2, p12 : float
        Positive prior coefficients.
    overlap_min, h4_threshold : float
        Inclusive pair eligibility thresholds.
    chunk_size : int
        Positive evidence row-block size.
 
    Returns
    -------
    float
        Restricted-minus-full collider fraction in percentage points.
 
    Raises
    ------
    ValueError
        If any stage input or resulting component table is invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _oracle_evaluate_colocalisation_audit(
    log_bf: "np.ndarray",
    collection_id: "np.ndarray",
    tested_variant_mask: "np.ndarray",
    signal_metadata: "np.ndarray",
    has_credible_set: "np.ndarray",
    p1: float,
    p2: float,
    p12: float,
    overlap_min: float,
    h4_threshold: float,
    chunk_size: int,
) -> float:
    metadata_raw = np.asarray(signal_metadata)
    credible_raw = np.asarray(has_credible_set)
    log_bf_raw = np.asarray(log_bf)
    n_signals = log_bf_raw.shape[0] if log_bf_raw.ndim == 2 else -1
    if metadata_raw.ndim != 2 or metadata_raw.shape != (n_signals, 3) or metadata_raw.dtype.kind not in "iu":
        raise ValueError("signal_metadata must be an aligned integer array with three columns")
    if credible_raw.ndim != 1 or credible_raw.shape[0] != n_signals or credible_raw.dtype.kind != "b":
        raise ValueError("has_credible_set must be an aligned boolean vector")
    retained = _oracle_screen_signal_indices(log_bf, collection_id, tested_variant_mask)
    if retained.size < 2:
        raise ValueError("at least two signals must survive screening")
    prepared = _oracle_prepare_signal_matrix(log_bf, collection_id, tested_variant_mask, retained)
    retained_collection_labels = np.asarray(collection_id)[retained]
    _, retained_collections = np.unique(retained_collection_labels, return_inverse=True)
    retained_collections = retained_collections.astype(int, copy=False)
    shared = _oracle_collection_shared_masks(prepared, retained_collections)
    coverage = _oracle_reciprocal_signal_coverage(prepared, retained_collections, shared)
    evidence = _oracle_batched_hypothesis_evidence(prepared, retained_collections, shared, p1, p2, p12, chunk_size)
    edges = _oracle_select_colocalisation_edges(evidence, coverage, overlap_min, h4_threshold)
    restricted_edges = _oracle_restrict_credible_set_edges(edges, credible_raw[retained])
    retained_metadata = metadata_raw[retained]
    full_components = _oracle_summarize_collider_components(edges, retained_metadata)
    restricted_components = _oracle_summarize_collider_components(restricted_edges, retained_metadata)
    return _oracle_collider_rate_contrast(full_components, restricted_components)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    import numpy as np
 
    def _audit_fixture():
        import math
 
        profiles = [
            {0: 18.0},
            {12: 18.0 + math.log(0.20), 8: 18.0 + math.log(0.25), 1: 18.0 + math.log(0.55)},
            {9: 18.0, 2: 18.0 + math.log(1.4)},
            {14: 18.0},
            {11: 4.9, 4: 4.9},
            {3: 18.0},
            {15: 18.0},
            {10: 18.0 + math.log(0.5), 5: 18.0 + math.log(0.5)},
            {13: 24.0, 6: 14.0},
            {4: 18.0},
            {7: 18.0 + math.log(0.5), 16: 18.0 + math.log(0.5)},
            {1: 18.0},
            {11: 18.0},
            {3: 18.0 + math.log(0.55), 12: 18.0 + math.log(0.25), 8: 18.0 + math.log(0.20)},
            {13: 14.0, 6: 24.0},
            {0: 18.0},
            {7: 18.0},
            {9: 18.0},
            {5: 18.0 + math.log(0.5), 14: 18.0 + math.log(0.5)},
            {11: 18.0},
            {10: 18.0},
            {4: 18.0},
            {16: 18.0},
            {15: 18.0},
        ]
        n, v = 24, 17
        collection = np.array(
            [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1,
             0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            dtype=int,
        )
        tested = np.ones((2, v), dtype=bool)
        tested[1, 2] = False
        log_bf = np.full((n, v), -12.0, dtype=float)
        log_bf[collection == 1, 2] = np.nan
        for row, columns in {1: [15], 9: [5], 15: [8], 18: [0]}.items():
            log_bf[row, columns] = np.nan
        for row, profile in enumerate(profiles):
            for variant, value in profile.items():
                log_bf[row, variant] = value
        dataset = np.arange(n, dtype=int) + 100
        trait = np.arange(n, dtype=int) + 200
        dataset[[5, 11]], trait[[5, 11]] = 10, 1000
        dataset[[3, 20]], trait[[3, 20]] = 20, 2000
        dataset[[16, 22]], trait[[16, 22]] = 30, 3000
        dataset[[9, 19]], trait[[9, 19]] = 40, 4000
        dataset[[0, 15]], trait[[0, 15]] = 50, 5000
        signal = np.arange(n, dtype=int) + 5000
        metadata = np.column_stack((dataset, trait, signal))
        credible = np.ones(n, dtype=bool)
        credible[[6, 16, 19, 21]] = False
        return log_bf, collection, tested, metadata, credible
 
    x0, c0, t0, m0, h0 = _audit_fixture()
 
    def _array_source(name, value, dtype):
        payload = repr(value.tolist()).replace("nan", "np.nan")
        return f"{name}=np.array({payload},dtype={dtype})"
 
    fixture_setup = "\n".join(
        [
            "import numpy as np",
            _array_source("x", x0, "float"),
            _array_source("c", c0, "int"),
            _array_source("t", t0, "bool"),
            _array_source("m", m0, "int"),
            _array_source("h", h0, "bool"),
        ]
    )
    common = "evaluate_colocalisation_audit(x,c,t,m,h,1e-4,1e-4,1e-5,q,.8,k)"
    gold = "_oracle_evaluate_colocalisation_audit(x,c,t,m,h,1e-4,1e-4,1e-5,q,.8,k)"
    return [
        {"setup": fixture_setup + "\nq=.5; k=4", "call": common, "gold_call": gold},
        {"setup": fixture_setup + "\nh[16]=True; q=.5; k=3", "call": common, "gold_call": gold},
        {"setup": fixture_setup + "\nq=.4; k=50", "call": common, "gold_call": gold},
        {
            "setup": fixture_setup + """
q=.5; k=4; m=m[:-1]
def candidate_code():
    try: evaluate_colocalisation_audit(x,c,t,m,h,1e-4,1e-4,1e-5,q,.8,k); return 0.0
    except ValueError: return 1.0
def oracle_code():
    try: _oracle_evaluate_colocalisation_audit(x,c,t,m,h,1e-4,1e-4,1e-5,q,.8,k); return 0.0
    except ValueError: return 1.0""",
            "call": "candidate_code()",
            "gold_call": "oracle_code()",
        },
    ]
