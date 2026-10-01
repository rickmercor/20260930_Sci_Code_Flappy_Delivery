"""
Detect when model-MD structure deviates from the AIMD RDF reference.

The RDF panel is sampled at the supplied frame interval and only complete trailing windows count. The bin edges fix rectangular quadrature for any radial integral. The source stability test determines the deviation and failure rule; the window length and threshold are supplied controls.

Returns
-------
tuple[np.ndarray, float | None], the complete-window RDF discrepancies and the first stability-failure time in ps, or None if no complete window fails
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rdf_stability(aimd_rdf: "np.ndarray", model_rdf_by_frame: "np.ndarray", edges: "np.ndarray", frame_dt_ps: float, window_ps: float = 1.0, threshold: float = 0.2) -> tuple["np.ndarray", float | None]:
    """Return source Eq. 3 deviations and the first stability-failure time.

    The model RDF is sampled every frame_dt_ps; only complete trailing
    windows are evaluated. The default window is 1 ps and the supplied
    threshold defaults to 0.2. Use rectangular quadrature on the supplied
    radial bin edges. Return the first failing window end time in ps, or None
    when no complete window fails. Raise ValueError on mismatched arrays or
    invalid controls.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_rdf_stability(
    aimd_rdf: "np.ndarray",
    model_rdf_by_frame: "np.ndarray",
    edges: "np.ndarray",
    frame_dt_ps: float,
    window_ps: float = 1.0,
    threshold: float = 0.2,
) -> tuple["np.ndarray", float | None]:
    """Discretize the source Eq. 3 short-window RDF deviation test.

    The 1 ps window and 0.2 threshold are sourced. The supplied radial range,
    rectangular-bin quadrature, complete trailing windows, and first-crossing
    convention are disclosed instance choices.
    """
    ref = np.asarray(aimd_rdf, float)
    model = np.asarray(model_rdf_by_frame, float)
    r = np.asarray(edges, float)
    if model.ndim != 2 or ref.shape != (model.shape[1],) or r.shape != (ref.size + 1,):
        raise ValueError("RDF panels and radial edges have mismatched shape")
    if not np.isfinite(ref).all() or not np.isfinite(model).all() or not np.isfinite(r).all():
        raise ValueError("nonfinite RDF")
    if not np.all(np.diff(r) > 0) or frame_dt_ps <= 0 or window_ps <= 0 or threshold < 0:
        raise ValueError("invalid grid or stability controls")
    width = window_ps / frame_dt_ps
    nearest = round(width)
    if nearest < 1 or not np.isclose(width, nearest, atol=1e-10, rtol=0):
        raise ValueError("window must contain a whole number of frames")
    if model.shape[0] < nearest:
        raise ValueError("trajectory is shorter than the stability window")
    deviations = np.empty(model.shape[0] - nearest + 1, dtype=np.float64)
    for i in range(nearest - 1, model.shape[0]):
        short_mean = model[i - nearest + 1 : i + 1].mean(axis=0)
        deviations[i - nearest + 1] = np.dot(np.abs(ref - short_mean), np.diff(r))
    failures = np.flatnonzero(deviations > threshold)
    first_failure = None if failures.size == 0 else float((failures[0] + nearest) * frame_dt_ps)
    return deviations, first_failure

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\ndef _pack(result):\n    values, first = result\n    return np.r_[values, -1.0 if first is None else first]\n"
    return [
        {"setup": setup + "r=np.array([0.,1.,2.]); a=np.array([.2,.3]); m=np.array([[.2,.3],[.21,.28],[.7,.3]]); rg=r.copy(); ag=a.copy(); mg=m.copy()", "call": "_pack(rdf_stability(a,m,r,.5))", "gold_call": "_pack(_oracle_rdf_stability(ag,mg,rg,.5))", "tol": 1e-10},
        {"setup": setup + "r=np.array([0.,1.,2.]); a=np.array([.2,.3]); m=np.tile(a,(5,1)); rg=r.copy(); ag=a.copy(); mg=m.copy()", "call": "_pack(rdf_stability(a,m,r,.25))", "gold_call": "_pack(_oracle_rdf_stability(ag,mg,rg,.25))", "tol": 1e-10},
        {"setup": setup + "r=np.array([0.,.5,2.]); a=np.array([1.,1.]); m=np.array([[1.,1.],[1.,1.],[1.4,1.]]); rg=r.copy(); ag=a.copy(); mg=m.copy()", "call": "_pack(rdf_stability(a,m,r,.5))", "gold_call": "_pack(_oracle_rdf_stability(ag,mg,rg,.5))", "tol": 1e-10},
        {"setup": 'import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\na=np.zeros(2); m=np.zeros((5,2)); r=np.array([0.,1.,2.]); ag=a.copy(); mg=m.copy(); rg=r.copy()', "call": '_raises(lambda: rdf_stability(a,m,r,.3))', "gold_call": '_raises(lambda: _oracle_rdf_stability(ag,mg,rg,.3))', "tol": 0},
    ]
