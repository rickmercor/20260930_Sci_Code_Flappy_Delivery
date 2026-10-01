"""
Computes the paper's orientational, pair-structure, NVE, and NPT stability diagnostics.

The source treats P2, RDF agreement, energy conservation, and flexible-cell volume stability as distinct post-training tests because a low validation MAE does not guarantee stable dynamics. The shared RNG stream is consumed per system in the order RMSD15, orientation, RDF noise, energy, and volume. With h=floor(n_frames/2), each energy or volume drift is abs(mean(x[h:]) - mean(x[:h])).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_dynamic_stability(summary: object, seed: int = 9917, n_frames: int = 240) -> object:
    """Return five structural and dynamical diagnostics per system.

    Parameters
    ----------
    summary : object
        Finite curation summary with shape (9, 7).
    seed : int
        Seed for the shared benchmark stream. For each system, consume
        RMSD15, orientation, RDF-noise, energy, and volume draws in that order.
    n_frames : int
        Number of frames, at least 60. With h=floor(n_frames/2), each energy
        or volume drift is abs(mean(x[h:]) - mean(x[:h])).

    Returns
    -------
    object
        Float64 array of shape (9, 5) containing mean P2, population standard
        deviation of P2, normalized RDF L1 error, energy drift, and volume drift.

    Raises
    ------
    ValueError
        If summary, seed, or n_frames is invalid.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_dynamic_stability(summary: object, seed: int = 9917, n_frames: int = 240) -> object:
    s = np.asarray(summary, dtype=np.float64)
    if s.shape != (9, 7) or not np.isfinite(s).all() or not isinstance(seed, (int, np.integer)) or not isinstance(n_frames, (int, np.integer)) or n_frames < 60:
        raise ValueError("summary, seed, or n_frames is invalid")
    rng = np.random.default_rng(int(seed))
    t = np.arange(int(n_frames), dtype=np.float64)
    out = np.empty((9, 5), dtype=np.float64)
    for k in range(9):
        rng.normal(0.135 + 0.012 * (k % 4) + 0.02 * s[k, 3], 0.052, size=int(n_frames))
        theta = 0.18 * np.sin((t + 1) * (k + 2) / 37.0) + rng.normal(0, 0.075 + 0.006 * k, size=int(n_frames))
        p2 = 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)
        r = np.linspace(0.0, 6.0, 96)
        g_ref = np.exp(-0.5 * ((r - (2.2 + 0.04 * k)) / 0.34) ** 2) + 0.48 * np.exp(-0.5 * ((r - 4.1) / 0.55) ** 2)
        g_model = g_ref * (1.0 + 0.025 * np.sin((k + 1) * r)) + rng.normal(0, 0.006, size=r.size)
        rdf = np.trapezoid(np.abs(g_model - g_ref), r) / np.trapezoid(np.abs(g_ref), r)
        energy = 0.003 * np.sin(2 * np.pi * t / 53.0 + k) + rng.normal(0, 0.00045, size=int(n_frames)) + (k - 4) * 2e-6 * t
        volume = 1.0 + 0.008 * np.sin(2 * np.pi * t / 71.0 + 0.3 * k) + rng.normal(0, 0.0012, size=int(n_frames)) + (k - 4) * 1.5e-6 * t
        h = int(n_frames) // 2
        out[k] = [p2.mean(), p2.std(ddof=0), rdf, abs(energy[h:].mean() - energy[:h].mean()), abs(volume[h:].mean() - volume[:h].mean())]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"sc=curate_active_pool(build_error_panel(58031,96));sg=_oracle_curate_active_pool(_oracle_build_error_panel(58031,96))", "call":"compute_dynamic_stability(sc,9917,240)", "gold_call":"_oracle_compute_dynamic_stability(sg,9917,240)", "tol":1e-9},
        {"setup":"sc=curate_active_pool(build_error_panel(58079,84));sg=_oracle_curate_active_pool(_oracle_build_error_panel(58079,84))", "call":"compute_dynamic_stability(sc,9967,180)", "gold_call":"_oracle_compute_dynamic_stability(sg,9967,180)", "tol":1e-9},
        {"setup":"sc=curate_active_pool(build_error_panel(58121,72));sg=_oracle_curate_active_pool(_oracle_build_error_panel(58121,72))", "call":"compute_dynamic_stability(sc,10009,60)", "gold_call":"_oracle_compute_dynamic_stability(sg,10009,60)", "tol":1e-9},
        {"setup":"sc=curate_active_pool(build_error_panel(58163,108));sg=_oracle_curate_active_pool(_oracle_build_error_panel(58163,108))", "call":"compute_dynamic_stability(sc,10061,300)", "gold_call":"_oracle_compute_dynamic_stability(sg,10061,300)", "tol":1e-9},
        {"setup":"sc=np.zeros((9,7));sg=sc.copy()\ndef _candidate_probe():\n    try: compute_dynamic_stability(sc,1,59); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef expected_probe():\n    try: _oracle_compute_dynamic_stability(sg,1,59); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call":"_candidate_probe()", "gold_call":"expected_probe()", "tol":1e-9},
    ]
