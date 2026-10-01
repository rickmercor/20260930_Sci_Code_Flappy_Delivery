"""
Scores source-defined polymorph ordering and wrong-minimum penalties.

Relative-energy MAE, rank correlation, and the DFT cost of a predicted minimum answer
different scientific questions and therefore remain separate outputs.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rank_polymorphs(landscapes: object) -> object:
    """Score two model energy landscapes against DFT.

    Parameters
    ----------
    landscapes : object
        Finite array of shape (9, n_forms, 3), with n_forms >= 4 and columns
        ordered as DFT, fine-tuned, and foundation energies.

    Returns
    -------
    object
        Array of shape (9, 2, 5): MAE, Kendall tau-a, DFT energy at the
        predicted minimum, stable top-two overlap, and gap error. Top-two
        overlap is the shared count between stably sorted lowest-two sets
        divided by two, so it is 0, 0.5, or 1.

    Raises
    ------
    ValueError
        If landscapes is non-finite or has an incompatible shape.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_rank_polymorphs(landscapes: object) -> object:
    x = np.asarray(landscapes, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] != 9 or x.shape[2] != 3 or x.shape[1] < 4 or not np.isfinite(x).all():
        raise ValueError("landscapes must be finite with shape (9, n_forms, 3)")
    out = np.empty((9, 2, 5), dtype=np.float64)
    for s in range(9):
        ref = x[s, :, 0]
        true_order = np.argsort(ref, kind="stable")
        for m, col in enumerate((1, 2)):
            pred = x[s, :, col]
            mae = np.mean(np.abs(pred - ref))
            concordance = 0.0
            for i in range(ref.size - 1):
                for j in range(i + 1, ref.size):
                    concordance += np.sign((ref[i] - ref[j]) * (pred[i] - pred[j]))
            tau = concordance / (ref.size * (ref.size - 1) / 2.0)
            penalty = ref[int(np.argmin(pred))]
            top2 = len(set(true_order[:2]).intersection(np.argsort(pred, kind="stable")[:2])) / 2.0
            gap_error = abs(np.partition(pred, 1)[1] - np.partition(ref, 1)[1])
            out[s, m] = [mae, tau, penalty, top2, gap_error]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"xc=build_relative_landscapes(curate_active_pool(build_error_panel(58031,96)),8123,6);xg=_oracle_build_relative_landscapes(_oracle_curate_active_pool(_oracle_build_error_panel(58031,96)),8123,6)", "call":"rank_polymorphs(xc)", "gold_call":"_oracle_rank_polymorphs(xg)", "tol":1e-9},
        {"setup":"xc=build_relative_landscapes(curate_active_pool(build_error_panel(58079,84)),8177,4);xg=_oracle_build_relative_landscapes(_oracle_curate_active_pool(_oracle_build_error_panel(58079,84)),8177,4)", "call":"rank_polymorphs(xc)", "gold_call":"_oracle_rank_polymorphs(xg)", "tol":1e-9},
        {"setup":"xc=build_relative_landscapes(curate_active_pool(build_error_panel(58121,72)),8219,7);xg=_oracle_build_relative_landscapes(_oracle_curate_active_pool(_oracle_build_error_panel(58121,72)),8219,7)", "call":"rank_polymorphs(xc)", "gold_call":"_oracle_rank_polymorphs(xg)", "tol":1e-9},
        {"setup":"xc=build_relative_landscapes(curate_active_pool(build_error_panel(58163,108)),8263,5);xg=_oracle_build_relative_landscapes(_oracle_curate_active_pool(_oracle_build_error_panel(58163,108)),8263,5)", "call":"rank_polymorphs(xc)", "gold_call":"_oracle_rank_polymorphs(xg)", "tol":1e-9},
        {"setup":"xc=np.zeros((9,3,3));xg=xc.copy()\ndef _candidate_probe():\n    try: rank_polymorphs(xc); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef expected_probe():\n    try: _oracle_rank_polymorphs(xg); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call":"_candidate_probe()", "gold_call":"expected_probe()", "tol":1e-9},
    ]
