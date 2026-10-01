"""
Propagate one or many angular rays through the paper's exact two-lens nonlinear thin-map.

Use the Appendix-D angle-dominated model without expanding or truncating in tau_x. Set x_1=L x'_0, y_1=L y'_0 and f=(L^{-1}+l^{-1})^{-1}. At either lens apply x' <- x' - [x+tau_x(x^2+y^2)/2]/f and y' <- y' - [y+tau_x x y]/f. The sequence is first kick, drift 2l, second identical kick, then drift L to the output (the initial drift L is already represented by x_1,y_1). The tau_x=0 map sends [x'_0,y'_0] to [0,x'_0,0,y'_0]. Preserve arbitrary leading batch dimensions.

Returns
-------
Return a NumPy float array with the input leading shape and final axis [x_3,x'_3,y_3,y'_3].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def transport_nonlinear_lens_pair(initial_slopes, length_m, half_gap_m, tau_per_m):
    """Return the exact final phase-space coordinates for angle-dominated rays.

    Args:
        initial_slopes: Finite float array with final dimension 2, [x'_0,y'_0].
        length_m: Positive drift length L in metres.
        half_gap_m: Positive half-cell gap l in metres.
        tau_per_m: Finite nonlinear taper tau_x in inverse metres.

    Returns:
        numpy.ndarray: Same leading shape as initial_slopes and final dimension 4,
        ordered [x_3,x'_3,y_3,y'_3] in [m,rad,m,rad].

    Raises:
        ValueError: If the ray array is empty, malformed, or nonfinite; a
            drift length is nonpositive; tau_x is nonfinite; or the mapped
            result is outside the finite supported numerical domain.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_transport_nonlinear_lens_pair(initial_slopes, length_m, half_gap_m, tau_per_m):
    import math
    import numpy as np
    rays = np.asarray(initial_slopes, dtype=float)
    length_m = float(length_m)
    half_gap_m = float(half_gap_m)
    tau_per_m = float(tau_per_m)
    if rays.ndim < 1 or rays.shape[-1] != 2 or rays.size == 0:
        raise ValueError("initial_slopes must have a nonempty final axis of length 2")
    if not np.all(np.isfinite(rays)) or not all(math.isfinite(x) for x in
            (length_m, half_gap_m, tau_per_m)):
        raise ValueError("inputs must be finite")
    if length_m <= 0 or half_gap_m <= 0:
        raise ValueError("drift lengths must be positive")
    flat = rays.reshape(-1, 2)
    xp0 = flat[:, 0]
    yp0 = flat[:, 1]
    focal = 1.0 / (1.0 / length_m + 1.0 / half_gap_m)
    x1 = length_m * xp0
    y1 = length_m * yp0
    xp1 = xp0 - (x1 + 0.5 * tau_per_m * (x1 * x1 + y1 * y1)) / focal
    yp1 = yp0 - (y1 + tau_per_m * x1 * y1) / focal
    x2 = x1 + 2.0 * half_gap_m * xp1
    y2 = y1 + 2.0 * half_gap_m * yp1
    xp2 = xp1 - (x2 + 0.5 * tau_per_m * (x2 * x2 + y2 * y2)) / focal
    yp2 = yp1 - (y2 + tau_per_m * x2 * y2) / focal
    x3 = x2 + length_m * xp2
    y3 = y2 + length_m * yp2
    out = np.column_stack((x3, xp2, y3, yp2))
    if not np.all(np.isfinite(out)):
        raise ValueError("mapped rays exceed the supported finite domain")
    return out.reshape(rays.shape[:-1] + (4,))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np", "call":"transport_nonlinear_lens_pair(np.array([[1e-4,-2e-4],[0.,3e-4]]),2.,4.,0.)", "gold_call":"_oracle_transport_nonlinear_lens_pair(np.array([[1e-4,-2e-4],[0.,3e-4]]),2.,4.,0.)"},
        {"setup":"import numpy as np", "call":"transport_nonlinear_lens_pair(np.array([2.5e-5,-7e-6]),5**.5,2*5**.5,-61.42)", "gold_call":"_oracle_transport_nonlinear_lens_pair(np.array([2.5e-5,-7e-6]),5**.5,2*5**.5,-61.42)"},
        {"setup":"import numpy as np", "call":"transport_nonlinear_lens_pair(np.array([[2e-4,1e-4],[-2e-4,1e-4],[3e-5,-4e-5]]),1.7,.9,35.)", "gold_call":"_oracle_transport_nonlinear_lens_pair(np.array([[2e-4,1e-4],[-2e-4,1e-4],[3e-5,-4e-5]]),1.7,.9,35.)"},
        {"setup":"import numpy as np", "call":"transport_nonlinear_lens_pair(np.array([[[1e-5,2e-5],[-3e-5,4e-5]]]),3.2,7.1,-140.)", "gold_call":"_oracle_transport_nonlinear_lens_pair(np.array([[[1e-5,2e-5],[-3e-5,4e-5]]]),3.2,7.1,-140.)"},
        {"setup":"import numpy as np", "call":"transport_nonlinear_lens_pair(np.array([[8e-7,-9e-7],[1.1e-6,1.3e-6]]),18.,36.,510.)", "gold_call":"_oracle_transport_nonlinear_lens_pair(np.array([[8e-7,-9e-7],[1.1e-6,1.3e-6]]),18.,36.,510.)"},
        {"setup":"import numpy as np", "call":"transport_nonlinear_lens_pair(np.array([[4e-5,0.],[0.,4e-5],[-4e-5,0.],[0.,-4e-5]]),2.4,5.3,-80.)", "gold_call":"_oracle_transport_nonlinear_lens_pair(np.array([[4e-5,0.],[0.,4e-5],[-4e-5,0.],[0.,-4e-5]]),2.4,5.3,-80.)"},
        {"setup":"import numpy as np", "call":"transport_nonlinear_lens_pair(np.array([[1.23e-4,-4.56e-5],[-7.8e-6,9.1e-5]]),.73,2.61,17.25)", "gold_call":"_oracle_transport_nonlinear_lens_pair(np.array([[1.23e-4,-4.56e-5],[-7.8e-6,9.1e-5]]),.73,2.61,17.25)"}
    ]
