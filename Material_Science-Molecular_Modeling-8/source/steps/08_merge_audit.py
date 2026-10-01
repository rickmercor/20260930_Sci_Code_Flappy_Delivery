"""
Build the two declared datasets from the integer constructors, run the full rate-preserving merge on each with eps, and report per dataset: the surviving particle count; the total post-merge weight; the Euclidean norm of the pre-versus-post difference of all conserved quantities (the 12 raw moments and the 2 unscaled rates); the weighted cube probe sum over survivors of weight times (u dot v)^3 for the declared probe u; the weighted position cube sum of weight times x^3; and the maximum and minimum surviving weights. Assemble by calling the earlier sub-problem functions.

The conserved block must come back at machine precision while the reported cube statistics quantify what merging genuinely changes; both sides of that ledger depend on every convention in the pipeline.

Returns
-------
return (2, 7) float64: audit rows per dataset
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def merge_audit(eps):
    """eps: retention threshold. Builds the two declared datasets, runs the
    rate-preserving merge on each, and returns a float64 array (2, 7) with
    columns: surviving count; total post-merge weight; the norm of the
    pre-versus-post difference of the 12 raw moments and 2 unscaled rates;
    the probe cube sum of weight times (u dot v)^3; the position cube sum of
    weight times x^3; and the maximum and minimum surviving weights.
    Assembled by calling the earlier sub-problem functions. Raises
    ValueError on an invalid threshold."""
    return np.zeros((2, 7))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 8 (final orchestrator): merge audit over the two declared datasets."""

import numpy as np

_N = 24
_NB = 10
_IV = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
       (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
_IX = [1, 2]


def _build(variant):
    a = 3 if variant == 1 else 5
    w = np.empty(_N)
    v = np.empty((_N, 3))
    x = np.empty(_N)
    for i in range(_N):
        w[i] = 0.5 + (((i + 1) * (i + 3) + a) % 7) / 10.0
        for c in range(3):
            v[i, c] = (((i + 1) * (i + 2) + (c + 2) * (i + 5) + a) % 47) / 23.5 - 1.0
        x[i] = (((i + 2) * (i + 4) * a + 1) % 41) / 41.0
    w2 = np.empty(_NB)
    v2 = np.empty((_NB, 3))
    for k in range(_NB):
        w2[k] = 0.8 + (((k + 1) * (k + 2) + a) % 5) / 10.0
        for c in range(3):
            v2[k, c] = (((k + 2) * (k + 3) + (c + 3) * (k + 1) + 2 * a) % 31) / 25.0 - 0.6
    return w, v, x, w2, v2


def _probe():
    return np.array([0.5, -0.25, 0.75])


def _sigma8(r, g):
    if r == 0:
        return 1.0 / (1.0 + g * g)
    return g / (1.0 + g)


def _conserved_vec(ww, vv, xx, w2, v2):
    m = np.empty(len(_IV) + len(_IX) + 2)
    for j, (mx, my, mz) in enumerate(_IV):
        m[j] = float(np.sum(ww * (vv[:, 0] ** mx) * (vv[:, 1] ** my) * (vv[:, 2] ** mz)))
    for k, mx_ in enumerate(_IX):
        m[len(_IV) + k] = float(np.sum(ww * xx ** mx_))
    for r in range(2):
        tot = 0.0
        for i in range(len(ww)):
            g = np.linalg.norm(vv[i] - v2, axis=1)
            tot += float(ww[i] * np.sum(w2 * g * _sigma8(r, g)))
        m[len(_IV) + len(_IX) + r] = tot
    return m


def _oracle_merge_audit(eps):
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("invalid threshold")
    u = _probe()
    rows = []
    for variant in (1, 2):
        w, v, x, w2, v2 = _build(variant)
        merged = _oracle_nnls_merge(w, v, x, w2, v2, eps)
        wp = merged[:, 0]
        vp = merged[:, 1:4]
        xp = merged[:, 4]
        res = float(np.linalg.norm(_conserved_vec(wp, vp, xp, w2, v2) - _conserved_vec(w, v, x, w2, v2)))
        uv = vp @ u
        rows.append([float(len(wp)), float(np.sum(wp)), res,
                     float(wp @ uv ** 3), float(wp @ xp ** 3),
                     float(np.max(wp)), float(np.min(wp))])
    return np.asarray(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'eps = 1e-8', "call": "merge_audit(eps)", "gold_call": "_oracle_merge_audit(eps)", "tol": 1e-08},
        {"setup": 'eps = 1e-6', "call": "merge_audit(eps)", "gold_call": "_oracle_merge_audit(eps)", "tol": 1e-08},
        {"setup": 'eps = 1e-7', "call": "merge_audit(eps)", "gold_call": "_oracle_merge_audit(eps)", "tol": 1e-08},
        {"setup": 'eps = 0.03', "call": "merge_audit(eps)", "gold_call": "_oracle_merge_audit(eps)", "tol": 1e-08},
        {"setup": 'eps = 0.12', "call": "merge_audit(eps)", "gold_call": "_oracle_merge_audit(eps)", "tol": 1e-08},
    ]
