"""
Run the complete Mg-Nd ordering and class-comparison calculation and return the chance-corrected agreement retention.

Call the eight preceding public functions in order. Complete the synthetic symmetry reduction before reading energy shapes, reconstruct the paper integer matrix from its row fractions using the supplied significant-figure precision, and return kappa_synthetic divided by kappa_paper.

Return one finite Python float. Raise ValueError if the integrated result is not finite or if an earlier operation rejects its inputs.

Returns
-------
One finite Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def resolve_mg_nd_kappa_retention(
    generator_permutations: np.ndarray,
    occupation_counts: np.ndarray,
    axis_pairs: np.ndarray,
    profile_keys: np.ndarray,
    dft_profiles: np.ndarray,
    mlip_profiles: np.ndarray,
    class_labels: np.ndarray,
    paper_row_counts: np.ndarray,
    paper_row_normalized: np.ndarray,
    paper_significant_figures: np.ndarray,
) -> float:
    return float("nan")

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_resolve_mg_nd_kappa_retention(
    generator_permutations,
    occupation_counts,
    axis_pairs,
    profile_keys,
    dft_profiles,
    mlip_profiles,
    class_labels,
    paper_row_counts,
    paper_row_normalized,
    paper_significant_figures,
):
    group_action = _oracle_build_finite_group_action(generator_permutations)
    decoration_orbit_data = _oracle_enumerate_decoration_orbit_data(
        group_action,
        occupation_counts,
    )
    retained_axis_data = _oracle_reduce_undirected_axis_variants(
        decoration_orbit_data,
        group_action,
        axis_pairs,
    )
    aligned_profile_rows = _oracle_align_retained_profile_rows(
        retained_axis_data,
        profile_keys,
        dft_profiles,
        mlip_profiles,
    )
    classified_rows = _oracle_classify_transformation_landscapes(
        aligned_profile_rows,
        class_labels,
    )
    score_matrix = _oracle_score_classification_agreement(
        classified_rows,
        class_labels,
    )
    paper_confusion = _oracle_reconstruct_integer_confusion_matrix(
        paper_row_counts,
        paper_row_normalized,
        paper_significant_figures,
    )
    result = _oracle_compute_kappa_retention(
        score_matrix,
        paper_confusion,
    )
    if not np.isfinite(result):
        raise ValueError("final kappa retention must be finite")
    return float(result)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base_setup = """import numpy as np
generator_permutations=np.array([[1,2,3,4,5,0],[0,5,4,3,2,1]],dtype=int)
occupation_counts=np.array([2,3],dtype=int)
axis_pairs=np.array([[2,5],[0,3],[1,4]],dtype=int)
profile_keys=np.array([[52,1],[48,0],[42,2],[40,2],[36,2],[48,2],[56,1],[40,0],[42,0],[36,1],[52,0],[48,1],[42,1],[56,2],[36,0],[52,2],[40,1],[56,0]],dtype=int)
dft_profiles=np.array([[18,16,13,9,5,2,0],[0,2,5,9,13,16,18],[0,2,5,9,13,16,18],[2,12,21,25,22,15,8],[8,0,-5,-4,2,10,18],[18,16,13,9,5,2,0],[8,15,22,25,21,12,2],[8,15,22,25,21,12,2],[0,2,5,9,13,16,18],[18,10,2,-4,-5,0,8],[8,0,-5,-4,2,10,18],[18,16,13,9,5,2,0],[0,2,5,9,13,16,18],[2,12,21,25,22,15,8],[8,0,-5,-4,2,10,18],[12,7,1,-4,-5,-1,8],[8,15,22,25,21,12,2],[8,15,22,25,21,12,2]],dtype=float)
mlip_profiles=np.array([[17,15,12,8,5,2,0],[0,2,6,10,13,16,19],[0,2,6,10,13,16,19],[1,3,6,10,14,17,20],[7,1,-4,-3,2,11,17],[17,12,6,-1,-3,2,5],[9,14,21,24,20,11,3],[3,11,20,24,21,14,9],[0,2,6,10,13,16,19],[17,9,1,-3,-4,1,7],[7,1,-4,-3,2,11,17],[17,12,6,-1,-3,2,5],[0,2,6,10,13,16,19],[10,18,24,26,20,12,4],[7,1,-4,-3,2,11,17],[7,1,-4,-3,2,8,11],[3,11,20,24,21,14,9],[9,14,21,24,20,11,3]],dtype=float)
class_labels=np.array([1,2,3,4,5,6],dtype=int)
paper_row_counts=np.array([19,51,25,37,8,9],dtype=int)
paper_row_normalized=np.array([[.79,0,.11,.11,0,0],[0,.84,0,.14,.02,0],[.12,.04,.56,.24,.04,0],[.081,.11,.081,.7,0,.027],[.12,0,0,0,.88,0],[0,.11,0,0,.11,.78]],dtype=float)
paper_significant_figures=np.full((6,6),2,dtype=int)"""

    call = (
        "resolve_mg_nd_kappa_retention("
        "generator_permutations,occupation_counts,axis_pairs,profile_keys,"
        "dft_profiles,mlip_profiles,class_labels,paper_row_counts,"
        "paper_row_normalized,paper_significant_figures)"
    )
    gold_call = (
        "_oracle_resolve_mg_nd_kappa_retention("
        "generator_permutations,occupation_counts,axis_pairs,profile_keys,"
        "dft_profiles,mlip_profiles,class_labels,paper_row_counts,"
        "paper_row_normalized,paper_significant_figures)"
    )

    identity_setup = base_setup + """
paper_row_counts=np.array([2,3,4,5,6,7],dtype=int)
paper_row_normalized=np.eye(6,dtype=float)"""

    invariant_setup = base_setup + """
row_order=np.array([13,0,17,5,8,2,10,3,15,7,12,1,16,11,6,14,4,9],dtype=int)
profile_keys=profile_keys[row_order]
dft_profiles=dft_profiles[row_order]
mlip_profiles=mlip_profiles[row_order]
class_labels=np.array([6,4,2,5,3,1],dtype=int)"""

    invalid_setup = base_setup + """
paper_significant_figures=paper_significant_figures[:5]
def run_model():
 try:
  resolve_mg_nd_kappa_retention(generator_permutations,occupation_counts,axis_pairs,profile_keys,dft_profiles,mlip_profiles,class_labels,paper_row_counts,paper_row_normalized,paper_significant_figures)
  return 0
 except ValueError:return 1
 except Exception:return 2
def run_gold():
 try:
  _oracle_resolve_mg_nd_kappa_retention(generator_permutations,occupation_counts,axis_pairs,profile_keys,dft_profiles,mlip_profiles,class_labels,paper_row_counts,paper_row_normalized,paper_significant_figures)
  return 0
 except ValueError:return 1
 except Exception:return 2"""

    return [
        {
            "setup": base_setup,
            "call": call,
            "gold_call": gold_call,
        },
        {
            "setup": identity_setup,
            "call": call,
            "gold_call": gold_call,
        },
        {
            "setup": invariant_setup,
            "call": call,
            "gold_call": gold_call,
        },
        {
            "setup": invalid_setup,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
