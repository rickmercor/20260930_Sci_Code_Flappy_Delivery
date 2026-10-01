"""
Compute the direct-binding saturation factor for every aligned state and reaction.

concentration_states_mM contains one row per metabolite state and one column per internal metabolite.

Each reaction has one substrate term and one product term. For an internal term, the aligned index gives the metabolite column and the aligned KM value is finite and strictly positive; the corresponding fixed-term entry must be NaN. For a fixed normalized term, the aligned index is -1, the corresponding KM is NaN, and the fixed-term value is already dimensionless. A fixed substrate term must be strictly positive. A fixed product term may be zero.

For each state and reaction form the normalized terms \(\bar s\) and \(\bar p\), then compute

\[
\kappa=\frac{\bar s}{1+\bar s+\bar p}.
\]

Preserve state and reaction order. Validate every aligned array and raise ValueError for an invalid index, inconsistent fixed/internal encoding, non-positive required quantity, non-finite quantity, or dimensional mismatch.

Returns
-------
2D NumPy float array of shape (n_states, n_reactions), with every value in (0, 1).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def compute_direct_binding_saturation_factors(concentration_states_mM: np.ndarray, substrate_indices: np.ndarray, substrate_km_mM: np.ndarray, substrate_fixed_terms: np.ndarray, product_indices: np.ndarray, product_km_mM: np.ndarray, product_fixed_terms: np.ndarray) -> np.ndarray:
    """Compute aligned direct-binding saturation factors."""
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_direct_binding_saturation_factors(
    concentration_states_mM,
    substrate_indices,
    substrate_km_mM,
    substrate_fixed_terms,
    product_indices,
    product_km_mM,
    product_fixed_terms,
):
    import numpy as np

    concentrations = np.asarray(concentration_states_mM, dtype=float)
    raw_sidx = np.asarray(substrate_indices)
    skm = np.asarray(substrate_km_mM, dtype=float)
    sfixed = np.asarray(substrate_fixed_terms, dtype=float)
    raw_pidx = np.asarray(product_indices)
    pkm = np.asarray(product_km_mM, dtype=float)
    pfixed = np.asarray(product_fixed_terms, dtype=float)

    if concentrations.ndim != 2 or min(concentrations.shape) < 1:
        raise ValueError(
            'concentration_states_mM must be a non-empty two-dimensional array'
        )

    if not np.all(np.isfinite(concentrations)) or np.any(concentrations <= 0.0):
        raise ValueError(
            'concentration_states_mM must contain finite strictly positive values'
        )

    if (
        raw_sidx.ndim != 1
        or raw_sidx.size < 1
        or not np.issubdtype(raw_sidx.dtype, np.integer)
    ):
        raise ValueError(
            'substrate_indices must be a non-empty one-dimensional integer array'
        )

    n_reactions = raw_sidx.size

    arrays = (
        skm,
        sfixed,
        raw_pidx,
        pkm,
        pfixed,
    )

    if any(
        arr.ndim != 1 or arr.size != n_reactions
        for arr in arrays
    ):
        raise ValueError(
            'all reaction-level arrays must be one-dimensional and aligned'
        )

    if not np.issubdtype(raw_pidx.dtype, np.integer):
        raise ValueError(
            'product_indices must be an integer array'
        )

    sidx = raw_sidx.astype(int, copy=False)
    pidx = raw_pidx.astype(int, copy=False)

    n_metabolites = concentrations.shape[1]

    result = np.empty(
        (concentrations.shape[0], n_reactions),
        dtype=float,
    )

    for reaction in range(n_reactions):
        if sidx[reaction] == -1:
            if (
                not np.isnan(skm[reaction])
                or not np.isfinite(sfixed[reaction])
                or sfixed[reaction] <= 0.0
            ):
                raise ValueError(
                    'a fixed substrate term requires index -1, NaN KM, and a finite positive fixed term'
                )

            sbar = np.full(
                concentrations.shape[0],
                sfixed[reaction],
                dtype=float,
            )

        else:
            if (
                sidx[reaction] < 0
                or sidx[reaction] >= n_metabolites
                or not np.isfinite(skm[reaction])
                or skm[reaction] <= 0.0
                or not np.isnan(sfixed[reaction])
            ):
                raise ValueError(
                    'an internal substrate requires a valid index, positive KM, and NaN fixed term'
                )

            sbar = (
                concentrations[:, sidx[reaction]]
                / skm[reaction]
            )

        if pidx[reaction] == -1:
            if (
                not np.isnan(pkm[reaction])
                or not np.isfinite(pfixed[reaction])
                or pfixed[reaction] < 0.0
            ):
                raise ValueError(
                    'a fixed product term requires index -1, NaN KM, and a finite non-negative fixed term'
                )

            pbar = np.full(
                concentrations.shape[0],
                pfixed[reaction],
                dtype=float,
            )

        else:
            if (
                pidx[reaction] < 0
                or pidx[reaction] >= n_metabolites
                or not np.isfinite(pkm[reaction])
                or pkm[reaction] <= 0.0
                or not np.isnan(pfixed[reaction])
            ):
                raise ValueError(
                    'an internal product requires a valid index, positive KM, and NaN fixed term'
                )

            pbar = (
                concentrations[:, pidx[reaction]]
                / pkm[reaction]
            )

        result[:, reaction] = (
            sbar
            / (1.0 + sbar + pbar)
        )

    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nconcentration_states_mM = np.array([[1.0, 2.0, 4.0], [3.0, 1.0, 2.0]], dtype=float)\nsubstrate_indices = np.array([-1, 0, 1, 2], dtype=int)\nsubstrate_km_mM = np.array([np.nan, 0.5, 1.0, 2.0], dtype=float)\nsubstrate_fixed_terms = np.array([3.0, np.nan, np.nan, np.nan], dtype=float)\nproduct_indices = np.array([0, 1, -1, 0], dtype=int)\nproduct_km_mM = np.array([0.7, 0.25, np.nan, 1.5], dtype=float)\nproduct_fixed_terms = np.array([np.nan, np.nan, 0.2, np.nan], dtype=float)\n', 'call': 'compute_direct_binding_saturation_factors(concentration_states_mM.copy(), substrate_indices.copy(), substrate_km_mM.copy(), substrate_fixed_terms.copy(), product_indices.copy(), product_km_mM.copy(), product_fixed_terms.copy())', 'gold_call': '_oracle_compute_direct_binding_saturation_factors(concentration_states_mM.copy(), substrate_indices.copy(), substrate_km_mM.copy(), substrate_fixed_terms.copy(), product_indices.copy(), product_km_mM.copy(), product_fixed_terms.copy())'}, {'setup': 'import numpy as np\nconcentration_states_mM = np.array([[2.0, 0.5], [2.0, 5.0], [4.0, 2.0]], dtype=float)\nsubstrate_indices = np.array([0, 0, -1], dtype=int)\nsubstrate_km_mM = np.array([1.0, 1.0, np.nan], dtype=float)\nsubstrate_fixed_terms = np.array([np.nan, np.nan, 1.75], dtype=float)\nproduct_indices = np.array([1, -1, -1], dtype=int)\nproduct_km_mM = np.array([1.0, np.nan, np.nan], dtype=float)\nproduct_fixed_terms = np.array([np.nan, 0.0, 0.4], dtype=float)\n', 'call': 'compute_direct_binding_saturation_factors(concentration_states_mM.copy(), substrate_indices.copy(), substrate_km_mM.copy(), substrate_fixed_terms.copy(), product_indices.copy(), product_km_mM.copy(), product_fixed_terms.copy())', 'gold_call': '_oracle_compute_direct_binding_saturation_factors(concentration_states_mM.copy(), substrate_indices.copy(), substrate_km_mM.copy(), substrate_fixed_terms.copy(), product_indices.copy(), product_km_mM.copy(), product_fixed_terms.copy())'}, {'setup': 'import numpy as np\nconcentration_states_mM = np.array([[1.2, 3.4, 5.6], [6.1, 2.3, 0.9]], dtype=float)\nsubstrate_indices = np.array([0, 1, -1, 2], dtype=int)\nsubstrate_km_mM = np.array([0.6, 1.1, np.nan, 2.2], dtype=float)\nsubstrate_fixed_terms = np.array([np.nan, np.nan, 2.2, np.nan], dtype=float)\nproduct_indices = np.array([1, -1, 0, 0], dtype=int)\nproduct_km_mM = np.array([0.7, np.nan, 1.8, 1.8], dtype=float)\nproduct_fixed_terms = np.array([np.nan, 0.3, np.nan, np.nan], dtype=float)\nmp = np.array([2, 0, 1], dtype=int)\nnew_index = np.empty(3, dtype=int); new_index[mp] = np.arange(3)\nconcentration_states_mM = concentration_states_mM[:, mp]\nsubstrate_indices = np.array([-1 if v == -1 else new_index[v] for v in substrate_indices], dtype=int)\nproduct_indices = np.array([-1 if v == -1 else new_index[v] for v in product_indices], dtype=int)\nrp = np.array([2, 0, 3, 1], dtype=int)\nsubstrate_indices = substrate_indices[rp]; substrate_km_mM = substrate_km_mM[rp]; substrate_fixed_terms = substrate_fixed_terms[rp]\nproduct_indices = product_indices[rp]; product_km_mM = product_km_mM[rp]; product_fixed_terms = product_fixed_terms[rp]\n', 'call': 'compute_direct_binding_saturation_factors(concentration_states_mM.copy(), substrate_indices.copy(), substrate_km_mM.copy(), substrate_fixed_terms.copy(), product_indices.copy(), product_km_mM.copy(), product_fixed_terms.copy())', 'gold_call': '_oracle_compute_direct_binding_saturation_factors(concentration_states_mM.copy(), substrate_indices.copy(), substrate_km_mM.copy(), substrate_fixed_terms.copy(), product_indices.copy(), product_km_mM.copy(), product_fixed_terms.copy())'}, {'setup': 'import numpy as np\nconcentration_states_mM = np.array([[2.0, 3.0]], dtype=float)\nsubstrate_indices = np.array([0], dtype=int)\nsubstrate_km_mM = np.array([1.0], dtype=float)\nsubstrate_fixed_terms = np.array([2.0], dtype=float)\nproduct_indices = np.array([-1], dtype=int)\nproduct_km_mM = np.array([np.nan], dtype=float)\nproduct_fixed_terms = np.array([0.2], dtype=float)\ndef candidate_wrapper():\n    try:\n        compute_direct_binding_saturation_factors(concentration_states_mM, substrate_indices, substrate_km_mM, substrate_fixed_terms, product_indices, product_km_mM, product_fixed_terms)\n    except ValueError:\n        return 1.0\n    return 0.0\ndef gold_wrapper():\n    try:\n        _oracle_compute_direct_binding_saturation_factors(concentration_states_mM, substrate_indices, substrate_km_mM, substrate_fixed_terms, product_indices, product_km_mM, product_fixed_terms)\n    except ValueError:\n        return 1.0\n    return 0.0\n', 'call': 'candidate_wrapper()', 'gold_call': 'gold_wrapper()'}]
