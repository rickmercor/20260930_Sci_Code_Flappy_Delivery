"""
Builds source-defined relative polymorph landscapes for DFT, fine-tuned, and foundation models.

The source compares polymorphs after independently zeroing each method to its own minimum.
This prevents a constant absolute-energy offset from being mistaken for a ranking error.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_relative_landscapes(summary: object, seed: int = 8123, n_forms: int = 6) -> object:
    """Return independently zeroed DFT, fine-tuned, and foundation landscapes.

    Parameters
    ----------
    summary : object
        Finite curation summary with shape (9, 7).
    seed : int
        Seed for one default_rng stream.
    n_forms : int
        Number of polymorph forms, at least four.

    Returns
    -------
    object
        Float64 array of shape (9, n_forms, 3), ordered as DFT,
        fine-tuned, and foundation.

    Raises
    ------
    ValueError
        If the summary, seed, or number of forms is invalid.

    Notes
    -----
    For zero-based system q and form k, the quadratic DFT term is
    0.33 * (k - (q % n_forms))**2. The modulus is the supplied n_forms.
    Draw exactly three arrays, in this order: base noise with shape (9, n_forms),
    fine-tuned noise with shape (9, n_forms), and foundation noise with shape
    (9, n_forms). Use the row-dependent scales by broadcasting. Apply the two
    foundation alterations before independently subtracting each method's minimum.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_relative_landscapes(summary: object, seed: int = 8123, n_forms: int = 6) -> object:
    s = np.asarray(summary, dtype=np.float64)
    if s.shape != (9, 7) or not np.isfinite(s).all() or not isinstance(seed, (int, np.integer)) or not isinstance(n_forms, (int, np.integer)) or n_forms < 4:
        raise ValueError("summary, seed, or n_forms is invalid")
    rng = np.random.default_rng(int(seed))
    k = np.arange(int(n_forms), dtype=np.float64)[None, :]
    q = np.arange(9, dtype=np.float64)[:, None]
    base = 0.33 * (k - (q % int(n_forms))) ** 2 + 0.21 * np.sin((q + 1) * (k + 2) / 3.0)
    base += 0.018 * s[:, 3, None] * (k + 1) + rng.normal(0, 0.018, size=base.shape)
    tuned = base + rng.normal(0, 0.065 + 0.025 * s[:, 3, None], size=base.shape)
    foundation = base + rng.normal(0, 0.24 + 0.11 * s[:, 4, None], size=base.shape)
    foundation[4] = foundation[4, ::-1]
    foundation[6] = foundation[6].mean() + 0.015 * (k[0] - k.mean())
    stack = np.stack([base, tuned, foundation], axis=-1)
    return stack - np.min(stack, axis=1, keepdims=True)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"sc=curate_active_pool(build_error_panel(58031,96));sg=_oracle_curate_active_pool(_oracle_build_error_panel(58031,96))", "call":"build_relative_landscapes(sc,8123,6)", "gold_call":"_oracle_build_relative_landscapes(sg,8123,6)", "tol":1e-9},
        {"setup":"sc=curate_active_pool(build_error_panel(58079,84));sg=_oracle_curate_active_pool(_oracle_build_error_panel(58079,84))", "call":"build_relative_landscapes(sc,8177,4)", "gold_call":"_oracle_build_relative_landscapes(sg,8177,4)", "tol":1e-9},
        {"setup":"sc=curate_active_pool(build_error_panel(58121,72));sg=_oracle_curate_active_pool(_oracle_build_error_panel(58121,72))", "call":"build_relative_landscapes(sc,8219,7)", "gold_call":"_oracle_build_relative_landscapes(sg,8219,7)", "tol":1e-9},
        {"setup":"sc=curate_active_pool(build_error_panel(58163,108));sg=_oracle_curate_active_pool(_oracle_build_error_panel(58163,108))", "call":"build_relative_landscapes(sc,8263,5)", "gold_call":"_oracle_build_relative_landscapes(sg,8263,5)", "tol":1e-9},
        {"setup":"sc=np.zeros((9,7));sg=sc.copy()\ndef _candidate_probe():\n    try: build_relative_landscapes(sc,1,3); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef expected_probe():\n    try: _oracle_build_relative_landscapes(sg,1,3); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call":"_candidate_probe()", "gold_call":"expected_probe()", "tol":1e-9},
    ]
