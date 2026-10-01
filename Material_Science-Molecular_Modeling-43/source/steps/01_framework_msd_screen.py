"""
Screen a candidate high-temperature trajectory for abnormal migration of non-Li framework atoms.

The source workflow accepts sampling temperatures only after checking non-Li framework motion. Use unwrapped positions. The fixture's numerical cutoff is an explicitly supplied instance control, not a paper-reported threshold.

Returns
-------
tuple[np.ndarray, bool], the non-Li MSD trace and the pass/fail flag
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def framework_msd_screen(positions: "np.ndarray", non_li: "np.ndarray", max_msd: float, start: int = 0) -> tuple["np.ndarray", bool]:
    """Return the non-Li MSD trace and whether its maximum from ``start`` is at most ``max_msd``.

    Positions are unwrapped Cartesian coordinates, shape (time>=2, atoms, 3).
    ``non_li`` is an atom-length boolean mask with at least one selected atom.
    ``max_msd`` is a finite nonnegative fixture limit in square angstrom.
    ``start`` is the inclusive zero-based frame index; default 0.
    Raise ValueError on malformed geometry, mask, limit or start.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_framework_msd_screen(
    positions: "np.ndarray", non_li: "np.ndarray", max_msd: float, start: int = 0
) -> tuple["np.ndarray", bool]:
    """Check non-Li framework rigidity against a disclosed fixture limit.

    The paper specifies the non-Li MSD check, but not a numeric cutoff.
    ``max_msd`` is therefore an instance control, not a sourced threshold.
    Positions must be unwrapped; shape is (time, atoms, Cartesian 3).
    """
    x = np.asarray(positions, dtype=np.float64)
    mask = np.asarray(non_li)
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[2] != 3:
        raise ValueError("positions must have shape (time>=2, atoms, 3)")
    if mask.shape != (x.shape[1],) or mask.dtype != bool or not mask.any():
        raise ValueError("non_li must select at least one framework atom")
    if not np.isfinite(x).all() or not np.isfinite(max_msd) or max_msd < 0:
        raise ValueError("positions and cutoff must be finite")
    if not isinstance(start, (int, np.integer)) or not 0 <= start < x.shape[0]:
        raise ValueError("invalid start")
    displacement = x[:, mask] - x[0, mask]
    trace = np.mean(np.sum(displacement * displacement, axis=-1), axis=1)
    return trace, bool(np.max(trace[start:]) <= max_msd)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\ndef _pack(result):\n    trace, accepted = result\n    return np.r_[trace, float(accepted)]\n"
    return [
        {"setup": setup + "a=np.zeros((3,2,3)); a[:,0,0]=[0,1,2]; a[:,1,0]=[0,.1,.2]; b=a.copy(); m=np.array([False,True])\n", "call": "_pack(framework_msd_screen(a,m,.05))", "gold_call": "_pack(_oracle_framework_msd_screen(b,m.copy(),.05))", "tol": 1e-10},
        {"setup": setup + "a=np.zeros((3,2,3)); a[:,1,0]=[0,.1,.2]; b=a.copy(); m=np.array([False,True])\n", "call": "_pack(framework_msd_screen(a,m,.04))", "gold_call": "_pack(_oracle_framework_msd_screen(b,m.copy(),.04))", "tol": 1e-10},
        {"setup": setup + "a=np.zeros((3,3,3)); a[:,1,0]=[0,.1,.2]; a[:,2,0]=[0,.2,.4]; b=a.copy(); m=np.array([False,True,True])\n", "call": "_pack(framework_msd_screen(a,m,.11))", "gold_call": "_pack(_oracle_framework_msd_screen(b,m.copy(),.11))", "tol": 1e-10},
        {"setup": setup + "a=np.zeros((12,2,3)); a[1,1,0]=.3; a[10:,1,0]=.1; b=a.copy(); m=np.array([False,True])\n", "call": "_pack(framework_msd_screen(a,m,.04,start=10))", "gold_call": "_pack(_oracle_framework_msd_screen(b,m.copy(),.04,start=10))", "tol": 1e-10},
        {"setup": "import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\na=np.zeros((3,2,3)); m=np.array([False,False]); b=a.copy(); mg=m.copy()", "call": "_raises(lambda: framework_msd_screen(a,m,.02))", "gold_call": "_raises(lambda: _oracle_framework_msd_screen(b,mg,.02))", "tol": 0},
    ]
