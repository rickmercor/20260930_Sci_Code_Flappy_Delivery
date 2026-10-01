"""
Construct carrier updates for multiallelic polarization.

Multiallelic polarization replaces an ancestral alternate by the former

reference allele while retaining every non-ancestral alternate. For each site

with observed set C and original alternate carrier sets A_j, the

former-reference carriers are C minus the union of all A_j. Surviving

alternates remain in input order and the former-reference replacement is last.

Returns
-------
np.ndarray of shape (n_sites * n_alternates, n_samples), binary replacement carriers as an int64 array
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def generate_polarization_updates(
    alternate_carriers: "np.ndarray",
    observed_haplotypes: "np.ndarray",
    ancestral_alternate_index: "np.ndarray",
) -> "np.ndarray":
    '''Generate replacement carrier rows for multiallelic polarization.

    Parameters
    ----------
    alternate_carriers : np.ndarray
        Binary array with one carrier row per alternate allele at each site.
    observed_haplotypes : np.ndarray
        Binary observed-haplotype mask for every site.
    ancestral_alternate_index : np.ndarray
        Index of the alternate allele inferred to be ancestral at each site.

    Raises
    ------
    ValueError
        If the arrays have incompatible shapes, fewer than two alternates, no
        samples, non-binary entries, overlapping alternate carrier sets,
        alternate carriers outside the observed set, or an invalid ancestral
        alternate index.

    Returns
    -------
    updates : np.ndarray
        Binary replacement carrier rows, ordered by site.
    '''
    return updates  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_generate_polarization_updates(
    alternate_carriers: "np.ndarray",
    observed_haplotypes: "np.ndarray",
    ancestral_alternate_index: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    alternate_carriers = np.asarray(alternate_carriers)
    observed_haplotypes = np.asarray(observed_haplotypes)
    ancestral_alternate_index = np.asarray(ancestral_alternate_index)

    if alternate_carriers.ndim != 3:
        raise ValueError("alternate_carriers must be three dimensional")
    n_sites, n_alternates, n_samples = alternate_carriers.shape
    if n_sites < 1 or n_alternates < 2 or n_samples < 1:
        raise ValueError("at least one site, two alternates, and one sample are required")
    if observed_haplotypes.shape != (n_sites, n_samples):
        raise ValueError("observed_haplotypes has an incompatible shape")
    if ancestral_alternate_index.shape != (n_sites,):
        raise ValueError("ancestral_alternate_index has an incompatible shape")
    if not np.all((alternate_carriers == 0) | (alternate_carriers == 1)):
        raise ValueError("alternate_carriers must be binary")
    if not np.all((observed_haplotypes == 0) | (observed_haplotypes == 1)):
        raise ValueError("observed_haplotypes must be binary")
    if not np.all(np.isfinite(ancestral_alternate_index)):
        raise ValueError("ancestral_alternate_index must be finite")
    if not np.all(ancestral_alternate_index == np.floor(ancestral_alternate_index)):
        raise ValueError("ancestral_alternate_index must contain integers")

    alternate_carriers = alternate_carriers.astype(np.int64)
    observed_haplotypes = observed_haplotypes.astype(np.int64)
    ancestral_alternate_index = ancestral_alternate_index.astype(np.int64)
    if np.any(ancestral_alternate_index < 0) or np.any(ancestral_alternate_index >= n_alternates):
        raise ValueError("ancestral_alternate_index is out of range")
    if np.any(alternate_carriers > observed_haplotypes[:, None, :]):
        raise ValueError("alternate carriers must be observed")
    if np.any(alternate_carriers.sum(axis=1) > 1):
        raise ValueError("alternate carrier sets must be disjoint within a site")

    replacement_rows = []
    for site in range(n_sites):
        ancestral = int(ancestral_alternate_index[site])
        for alternate in range(n_alternates):
            if alternate != ancestral:
                replacement_rows.append(alternate_carriers[site, alternate].copy())
        occupied = alternate_carriers[site].max(axis=0)
        former_reference = observed_haplotypes[site] * (1 - occupied)
        replacement_rows.append(former_reference)
    return np.asarray(replacement_rows, dtype=np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
alternate_carriers = np.array([
    [[0,0,0,0,0,0,1,1,0,0,1,1,0,0], [0,0,0,0,0,0,0,0,1,1,0,0,0,0]],
    [[0,0,1,1,1,1,0,0,0,0,0,0,0,0], [1,1,0,0,0,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,1]],
    [[0,0,0,0,0,0,1,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,1,1,1,1,1,1]],
    [[0,0,0,0,0,0,0,0,0,0,0,0,0,1], [0,0,1,1,1,1,1,1,1,1,1,1,0,0]],
], dtype=int)
observed_haplotypes = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
], dtype=int)
ancestral_alternate_index = np.array([0,1,0,1,0], dtype=int)
""",
            "call": "generate_polarization_updates(alternate_carriers, observed_haplotypes, ancestral_alternate_index).tolist()",
            "gold_call": "_oracle_generate_polarization_updates(alternate_carriers, observed_haplotypes, ancestral_alternate_index).tolist()",
        },
        {
            "setup": """import numpy as np
alternate_carriers = np.array([[[1,0,0], [0,1,0]]], dtype=int)
observed_haplotypes = np.ones((1,3), dtype=int)
ancestral_alternate_index = np.array([0], dtype=int)
""",
            "call": "generate_polarization_updates(alternate_carriers, observed_haplotypes, ancestral_alternate_index).tolist()",
            "gold_call": "_oracle_generate_polarization_updates(alternate_carriers, observed_haplotypes, ancestral_alternate_index).tolist()",
        },
        {
            "setup": """import numpy as np
alternate_carriers = np.array([[[1,0], [1,0]]], dtype=int)
observed_haplotypes = np.ones((1,2), dtype=int)
ancestral_alternate_index = np.array([0], dtype=int)
def run_model():
    try:
        generate_polarization_updates(alternate_carriers, observed_haplotypes, ancestral_alternate_index)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_generate_polarization_updates(alternate_carriers, observed_haplotypes, ancestral_alternate_index)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
