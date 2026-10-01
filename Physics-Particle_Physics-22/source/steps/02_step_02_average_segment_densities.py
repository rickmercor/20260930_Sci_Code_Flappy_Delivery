"""
Compute path-averaged electron densities inside the traced shell pieces.

A shell density is supplied as $\rho_j(r)=\sum_{k=0}^{3}c_{jk}(r/R)^k$,

with constant electron fraction $Y_{e,j}$ and outer Earth radius $R$.

For entry and exit distances $\ell_a>\ell_b$, let $L=\ell_a-\ell_b$ and

evaluate the composite midpoint rule at

$\ell_h=\ell_b+(h+1/2)L/Q$ for $h=0,\ldots,Q-1$:



$$

\overline{\rho Y_e}=\frac{Y_{e,j}}{Q}\sum_{h=0}^{Q-1}

\rho_j\!\left(\sqrt{r_d^2+\ell_h^2+2r_dc\ell_h}\right).

$$



The constant-layer propagation model uses this path average, not a radial

midpoint density or a volume average. Density coefficients are explicit task

inputs; they are not assumed to be a particular empirical Earth fit.

Returns
-------
A real array of shape (N, 2) containing segment lengths and path-averaged electron-density products.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def average_segment_densities(
    segments: "np.ndarray",
    radii_km: "np.ndarray",
    density_coefficients: "np.ndarray",
    electron_fractions: "np.ndarray",
    cos_zenith: float,
    detector_depth_km: float,
    quadrature_order: int = 16,
) -> "np.ndarray":
    r"""Compute path-averaged electron densities inside the traced shell pieces.

    Parameters
    ----------
    segments : np.ndarray
        Real shape (N, 3) entries from the shell trace: backward entry, exit,
        and integer-valued shell index. Require entry > exit >= 0.
    radii_km : np.ndarray
        Finite positive increasing outer radii, shape (S,).
    density_coefficients : np.ndarray
        Finite real shape (S, 4) ascending coefficients for powers of r/R,
        in grams per cubic centimeter.
    electron_fractions : np.ndarray
        Finite real shape (S,) fractions between zero and one inclusive.
    cos_zenith : float
        Finite direction cosine in [-1, 1].
    detector_depth_km : float
        Finite depth in [0, R), with R the final radius.
    quadrature_order : int, default 16
        Positive integer midpoint sample count per segment.

    Returns
    -------
    layers : np.ndarray
        Real shape (N, 2) array of path lengths in kilometers and mean density
        times electron fraction in grams per cubic centimeter.

    Raises
    ------
    ValueError
        If input shapes, finiteness, geometry, segment ordering, indices,
        fraction ranges, or quadrature count are invalid, or a sampled
        density is negative. Segment shell membership is an upstream
        precondition established by the trace function.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_average_segment_densities(
    segments: "np.ndarray",
    radii_km: "np.ndarray",
    density_coefficients: "np.ndarray",
    electron_fractions: "np.ndarray",
    cos_zenith: float,
    detector_depth_km: float,
    quadrature_order: int = 16,
) -> "np.ndarray":
    raw = np.asarray(radii_km)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("radii must be a nonempty vector")
    radii = _real_array(radii_km, raw.shape, "radii_km")
    coefficients = _real_array(
        density_coefficients, (len(radii), 4), "density_coefficients"
    )
    fractions = _real_array(electron_fractions, radii.shape, "electron_fractions")
    raw_segments = np.asarray(segments)
    if raw_segments.ndim != 2 or raw_segments.shape[1] != 3:
        raise ValueError("segments must have shape (N, 3)")
    paths = _real_array(segments, raw_segments.shape, "segments")
    direction = _finite_scalar(cos_zenith, "cos_zenith")
    depth = _finite_scalar(detector_depth_km, "detector_depth_km")
    if (
        np.any(radii <= 0)
        or np.any(np.diff(radii) <= 0)
        or not -1 <= direction <= 1
        or not 0 <= depth < radii[-1]
    ):
        raise ValueError("invalid geometry")
    if np.any((fractions < 0) | (fractions > 1)):
        raise ValueError("invalid electron fraction")
    if (
        isinstance(quadrature_order, (bool, np.bool_))
        or not isinstance(quadrature_order, (int, np.integer))
        or quadrature_order < 1
    ):
        raise ValueError("quadrature_order must be a positive integer")
    if np.any(paths[:, 0] <= paths[:, 1]) or np.any(paths[:, 1] < 0):
        raise ValueError("invalid segment endpoints")
    if (
        np.any(paths[:, 2] != np.floor(paths[:, 2]))
        or np.any(paths[:, 2] < 0)
        or np.any(paths[:, 2] >= len(radii))
    ):
        raise ValueError("invalid shell index")
    result = np.empty((len(paths), 2))
    detector_radius = radii[-1] - depth
    nodes = (np.arange(quadrature_order) + 0.5) / quadrature_order
    for index, (entry, exit_, shell_value) in enumerate(paths):
        shell = int(shell_value)
        length = entry - exit_
        distances = exit_ + nodes * length
        radial = (
            np.sqrt(
                np.maximum(
                    detector_radius**2
                    + distances**2
                    + 2.0 * detector_radius * direction * distances,
                    0.0,
                )
            )
            / radii[-1]
        )
        density = np.polynomial.polynomial.polyval(radial, coefficients[shell])
        if np.any(density < 0) or not np.all(np.isfinite(density)):
            raise ValueError("sampled density must be nonnegative and finite")
        result[index] = length, float(fractions[shell] * np.mean(density))
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative numerical and invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nr = np.array([3480.,6371.])\nc = np.array([[11.,0.,0.,0.],[4.5,0.,0.,0.]])\ny = np.array([.467,.495])\ns_model = trace_shell_segments(r.copy(),-1.,0.)\ns_gold = _oracle_trace_shell_segments(r.copy(),-1.,0.)\n",
            "call": "average_segment_densities(s_model.copy(), r.copy(), c.copy(), y.copy(), -1.0, 0.0)",
            "gold_call": "_oracle_average_segment_densities(s_gold.copy(), r.copy(), c.copy(), y.copy(), -1.0, 0.0)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([5.])\nc = np.array([[1.,2.,3.,4.]])\ny = np.array([0.5])\ns = np.empty((0,3))\n",
            "call": "average_segment_densities(s.copy(), r.copy(), c.copy(), y.copy(), 0.5, 0.0)",
            "gold_call": "_oracle_average_segment_densities(s.copy(), r.copy(), c.copy(), y.copy(), 0.5, 0.0)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([3.,5.])\nc = np.array([[10.,-1.,.2,0.],[4.,1.,.5,.1]])\ny = np.array([.4,.5])\ns_model = trace_shell_segments(r.copy(),-.9,.2,2)\ns_gold = _oracle_trace_shell_segments(r.copy(),-.9,.2,2)\n",
            "call": "average_segment_densities(s_model.copy(), r.copy(), c.copy(), y.copy(), -0.9, 0.2, 7)",
            "gold_call": "_oracle_average_segment_densities(s_gold.copy(), r.copy(), c.copy(), y.copy(), -0.9, 0.2, 7)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([5.])\nc = np.array([[-1.,0.,0.,0.]])\ny = np.array([.5])\ns = np.array([[1.,0.,0.]])\ndef _raises_value_error(function):\n    try:\n        function()\n    except ValueError:\n        return 1\n    return 0\n",
            "call": "_raises_value_error(lambda: average_segment_densities(s.copy(), r.copy(), c.copy(), y.copy(), 0.5, 0.0))",
            "gold_call": "_raises_value_error(lambda: _oracle_average_segment_densities(s.copy(), r.copy(), c.copy(), y.copy(), 0.5, 0.0))",
            "tol": 0.0,
        },
    ]
