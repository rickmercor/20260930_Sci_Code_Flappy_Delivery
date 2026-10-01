"""
Measure how much of the AIMD pretrained-feature space is represented by a model-MD trajectory.

The supplied panels are pretrained feature embeddings from model-MD and AIMD. The default eigenvalue cutoff is 1.0. For this fixture, standardize with population standard deviation, replace a zero divisor by one, and include a value on the uppermost histogram edge in the last bin. The paper determines the coverage calculation.

Returns
-------
tuple[float, np.ndarray], the mean feature-coverage ratio and the per-retained-component coverage ratios
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def feature_coverage(mace: "np.ndarray", aimd: "np.ndarray", bins: int, eigen_cutoff: float = 1.0) -> tuple[float, "np.ndarray"]:
    """Return the source feature-coverage summary and its component ratios.

    Inputs are finite pretrained embedding matrices with at least two rows
    each and the same feature width. bins is an integer at least two;
    eigen_cutoff defaults to 1.0. For fixture preprocessing, use the
    population standard deviation on the joined panel, replacing a zero
    divisor by one. A value on the uppermost histogram edge belongs to the
    last bin. Source Eqs. 4-5 determine component retention and coverage.
    Raise ValueError on invalid inputs or if no component is retained.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_feature_coverage(
    mace: "np.ndarray", aimd: "np.ndarray", bins: int, eigen_cutoff: float = 1.0
) -> tuple[float, "np.ndarray"]:
    """Apply paper Eqs. 4–5 to jointly standardized, PCA-reduced embeddings.

    The input is *precomputed* fixed-length pretrained-feature embeddings.
    It does not simulate the embedding network or DIRECT cluster selection.
    Each coordinate uses shared bins over the combined projected range;
    the denominator counts AIMD-occupied bins.
    """
    a, b = np.asarray(mace, float), np.asarray(aimd, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("two feature matrices with equal width are required")
    if min(a.shape[0], b.shape[0]) < 2 or a.shape[1] < 1:
        raise ValueError("each panel needs at least two rows")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite feature")
    if not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError("bins must be an integer at least two")
    if not np.isfinite(eigen_cutoff) or eigen_cutoff <= 0:
        raise ValueError("invalid eigenvalue cutoff")
    joined = np.concatenate((a, b), axis=0)
    mu = joined.mean(axis=0)
    sd = joined.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    standardized = (joined - mu) / sd
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    eigen = singular**2 / (len(joined) - 1)
    keep = eigen > eigen_cutoff
    if not np.any(keep):
        raise ValueError("no retained pretrained-feature component")
    pa = (a - mu) / sd @ vt[keep].T
    pb = (b - mu) / sd @ vt[keep].T
    ratios = []
    for d in range(pa.shape[1]):
        lo = min(pa[:, d].min(), pb[:, d].min())
        hi = max(pa[:, d].max(), pb[:, d].max())
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        ha, edges = np.histogram(pa[:, d], bins=bins, range=(lo, hi))
        hb, _ = np.histogram(pb[:, d], bins=edges)
        ratios.append(np.count_nonzero((ha > 0) & (hb > 0)) / np.count_nonzero(hb))
    values = np.asarray(ratios, dtype=np.float64)
    return float(values.mean()), values

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\n"
    return [
        {"setup": setup + "rng=np.random.default_rng(12); a=rng.normal(size=(40,5)); b=a.copy(); ag=a.copy(); bg=b.copy()\n", "call": "feature_coverage(a,b,7)", "gold_call": "_oracle_feature_coverage(ag,bg,7)", "tol": 1e-9},
        {"setup": setup + "rng=np.random.default_rng(15); a=rng.normal(size=(18,3)); b=rng.normal(loc=.7,size=(18,3)); ag=a.copy(); bg=b.copy()\n", "call": "feature_coverage(a,b,3,.8)", "gold_call": "_oracle_feature_coverage(ag,bg,3,.8)", "tol": 1e-9},
        {"setup": setup + "rng=np.random.default_rng(77); a=rng.normal(size=(60,9)); b=.75*a+.25*rng.normal(size=(60,9)); ag=a.copy(); bg=b.copy()\n", "call": "feature_coverage(a,b,19,1.2)", "gold_call": "_oracle_feature_coverage(ag,bg,19,1.2)", "tol": 1e-9},
        {"setup": 'import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\na=np.zeros((8,3)); b=a.copy(); ag=a.copy(); bg=b.copy()', "call": '_raises(lambda: feature_coverage(a,b,4))', "gold_call": '_raises(lambda: _oracle_feature_coverage(ag,bg,4))', "tol": 0},
    ]
