"""
Find the short-range Li–Li force reversal used in the paper's stability diagnosis.

Force is minus the diatomic energy derivative. The fixture uses segment secants and linear interpolation between adjacent segment midpoints for the first repulsive-to-attractive sign change. A smooth curve with no such crossing receives the paper's 0.1-angstrom plotting marker, not a claimed physical turning point.

Returns
-------
float, the first Li-Li force-reversal position in angstrom, or 0.1 angstrom if no crossing occurs
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def diatomic_turning_point(distance: "np.ndarray", energy: "np.ndarray") -> float:
    """Return the first short-range repulsive-to-attractive Li-Li force reversal.

    Both vectors must be finite, equal length at least three, and distances
    strictly increasing. Segment force is minus the energy secant slope.
    Interpolate the first positive-to-nonpositive crossing between adjacent
    segment midpoints; return 0.1 angstrom as the paper's plotting marker
    if no crossing occurs. Raise ValueError on malformed curves.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_diatomic_turning_point(distance: "np.ndarray", energy: "np.ndarray") -> float:
    """Find the first repulsive-to-attractive Li–Li force sign change.

    Segment forces are minus secant slopes. Interpolate linearly between
    adjacent segment midpoints at the first positive-to-negative force
    crossing. A smooth curve without a crossing receives the paper's 0.1 Å
    plotting marker; this is a plotting convention, not a physical minimum.
    """
    r, e = np.asarray(distance, float), np.asarray(energy, float)
    if r.ndim != 1 or e.shape != r.shape or r.size < 3:
        raise ValueError("matched distance and energy vectors need >=3 points")
    if not np.isfinite(r).all() or not np.isfinite(e).all() or not np.all(np.diff(r) > 0):
        raise ValueError("distance must increase; all values must be finite")
    mid = (r[:-1] + r[1:]) / 2
    force = -np.diff(e) / np.diff(r)
    for i in range(len(force) - 1):
        if force[i] > 0 and force[i + 1] <= 0:
            weight = force[i] / (force[i] - force[i + 1])
            return float(mid[i] + weight * (mid[i + 1] - mid[i]))
    return 0.1

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nr=np.array([1.,2.,3.,4.]); e=np.array([4.,2.,1.,2.]); rg=r.copy(); eg=e.copy()", "call": "diatomic_turning_point(r,e)", "gold_call": "_oracle_diatomic_turning_point(rg,eg)", "tol": 1e-10},
        {"setup": "import numpy as np\nr=np.array([.1,.2,.3]); e=np.array([3.,2.,1.]); rg=r.copy(); eg=e.copy()", "call": "diatomic_turning_point(r,e)", "gold_call": "_oracle_diatomic_turning_point(rg,eg)", "tol": 1e-10},
        {"setup": "import numpy as np\nr=np.linspace(.2,1.5,22); e=3/r**2-3*np.exp(-((r-.91)/.12)**2); rg=r.copy(); eg=e.copy()", "call": "diatomic_turning_point(r,e)", "gold_call": "_oracle_diatomic_turning_point(rg,eg)", "tol": 1e-10},
        {"setup": 'import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\nr=np.array([.2,.1,.3]); e=np.ones(3); rg=r.copy(); eg=e.copy()', "call": '_raises(lambda: diatomic_turning_point(r,e))', "gold_call": '_raises(lambda: _oracle_diatomic_turning_point(rg,eg))', "tol": 0},
    ]
