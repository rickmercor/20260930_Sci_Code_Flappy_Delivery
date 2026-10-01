"""
Trace a directed neutrino path through concentric spherical shells.

Let $R$ be the outer radius, $r_d=R-d$ the detector radius, and

$c=\cos\theta_z$ with $c=-1$ for a diametrical upward-going path.

Measure $\ell$ backward from the detector toward the source, so



$$

r(\ell)^2=r_d^2+\ell^2+2r_dc\ell,\qquad

\ell_{\mathrm{src}}=-r_dc+\sqrt{R^2-r_d^2(1-c^2)}.

$$



For a shell radius $r_j$, retain each crossing

$\ell=-r_dc\pm\sqrt{r_j^2-r_d^2(1-c^2)}$ lying strictly between the detector and the surface source.

Sort the intersections from source to detector and label each positive-length

interval by its midpoint radius, assigning a boundary to the outer shell.

Split each interval into the requested number of equal path-length pieces.

A tangency contributes a single breakpoint, not a finite inner-shell interval;

a surface detector viewing outward has a zero-length path.

Discriminants within 32 machine epsilons of their squared-radius scale are

treated as zero to preserve tangencies under floating-point roundoff.

Returns
-------
A real array of shape (N, 3) containing ordered entry and exit distances in kilometers and zero-based shell indices.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def trace_shell_segments(
    radii_km: "np.ndarray",
    cos_zenith: float,
    detector_depth_km: float,
    subdivisions: int = 1,
) -> "np.ndarray":
    r"""Trace a directed neutrino path through concentric spherical shells.

    Parameters
    ----------
    radii_km : np.ndarray
        Nonempty finite positive strictly increasing shell outer radii, shape (S,).
    cos_zenith : float
        Finite direction cosine in [-1, 1]; -1 points through the center.
    detector_depth_km : float
        Finite depth d in [0, R), where R is the final shell radius.
    subdivisions : int, default 1
        Positive integer number of equal path-length pieces per shell interval.

    Returns
    -------
    segments : np.ndarray
        Real shape (N, 3) array in source-to-detector order. Columns contain
        backward distance at entry, backward distance at exit, and the
        zero-based integer shell index stored as a float. Entry exceeds exit.
        A zero-length path returns shape (0, 3).

    Raises
    ------
    ValueError
        If radii, direction, depth, or subdivisions violate their domains.
        Boolean values are not accepted for subdivisions.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _finite_scalar(value, name):
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array) or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _real_array(value, shape, name):
    array = np.asarray(value)
    if np.iscomplexobj(array) or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    array = np.asarray(array, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} has an invalid shape or nonfinite entries")
    return array


def _oracle_trace_shell_segments(
    radii_km: "np.ndarray",
    cos_zenith: float,
    detector_depth_km: float,
    subdivisions: int = 1,
) -> "np.ndarray":
    raw = np.asarray(radii_km)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("radii must be a nonempty vector")
    radii = _real_array(radii_km, raw.shape, "radii_km")
    direction = _finite_scalar(cos_zenith, "cos_zenith")
    depth = _finite_scalar(detector_depth_km, "detector_depth_km")
    if (
        np.any(radii <= 0.0)
        or np.any(np.diff(radii) <= 0.0)
        or not -1.0 <= direction <= 1.0
        or not 0.0 <= depth < radii[-1]
    ):
        raise ValueError("invalid spherical geometry")
    if (
        isinstance(subdivisions, (bool, np.bool_))
        or not isinstance(subdivisions, (int, np.integer))
        or subdivisions < 1
    ):
        raise ValueError("subdivisions must be a positive integer")
    detector_radius = radii[-1] - depth
    impact_sq = detector_radius**2 * (1.0 - direction**2)
    shift = -detector_radius * direction
    distance = shift + np.sqrt(max(radii[-1] ** 2 - impact_sq, 0.0))
    if distance == 0.0 or (depth == 0.0 and direction >= 0.0):
        return np.empty((0, 3))
    cuts = [0.0, float(distance)]
    for radius in radii[:-1]:
        discriminant = radius * radius - impact_sq
        roundoff = 32.0 * np.finfo(float).eps * max(radius * radius, impact_sq)
        if abs(discriminant) <= roundoff:
            discriminant = 0.0
        if discriminant >= 0.0:
            root = np.sqrt(discriminant)
            for crossing in (shift - root, shift + root):
                if 0.0 < crossing < distance:
                    cuts.append(float(crossing))
    cuts = sorted(set(cuts), reverse=True)
    rows = []
    for entry, exit_ in zip(cuts[:-1], cuts[1:]):
        midpoint = 0.5 * (entry + exit_)
        radius = np.sqrt(
            max(
                detector_radius**2
                + midpoint**2
                + 2.0 * detector_radius * direction * midpoint,
                0.0,
            )
        )
        shell = min(int(np.searchsorted(radii, radius, side="right")), len(radii) - 1)
        points = np.linspace(entry, exit_, subdivisions + 1)
        for start, stop in zip(points[:-1], points[1:]):
            if start > stop:
                rows.append((start, stop, shell))
    return np.asarray(rows, dtype=float).reshape(-1, 3)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative numerical and invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nr = np.array([3480.0, 6371.0])\n",
            "call": "trace_shell_segments(r.copy(), -1.0, 0.0)",
            "gold_call": "_oracle_trace_shell_segments(r.copy(), -1.0, 0.0)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([3480.0, 6371.0])\n",
            "call": "trace_shell_segments(r.copy(), 0.4, 0.0)",
            "gold_call": "_oracle_trace_shell_segments(r.copy(), 0.4, 0.0)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([1221.5, 3480.0, 5701.0, 6371.0])\n",
            "call": "trace_shell_segments(r.copy(), -0.82, 2.0, 3)",
            "gold_call": "_oracle_trace_shell_segments(r.copy(), -0.82, 2.0, 3)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([3.0, 5.0])\n",
            "call": "trace_shell_segments(r.copy(), -0.8, 0.0, 2)",
            "gold_call": "_oracle_trace_shell_segments(r.copy(), -0.8, 0.0, 2)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([2.0, 1.0])\ndef _raises_value_error(function):\n    try:\n        function()\n    except ValueError:\n        return 1\n    return 0\n",
            "call": "_raises_value_error(lambda: trace_shell_segments(r.copy(), -1.0, 0.0))",
            "gold_call": "_raises_value_error(lambda: _oracle_trace_shell_segments(r.copy(), -1.0, 0.0))",
            "tol": 0.0,
        },
    ]
