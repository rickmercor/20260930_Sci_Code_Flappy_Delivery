"""
Derive a symmetric seven-point positive rule in correlated logit-reflectivity and log-finesse coordinates, then map every point back to physical optics parameters.

Use x=[logit(Ra),logit(Rb),ln(F)] and NumPy's lower-Cholesky column order. The centre carries half the total weight; the remaining points are equal-weight positive/negative column pairs. Derive the common pair displacement scale from exact covariance matching. Return the centre first, then each positive/negative pair.

Returns
-------
Return one real NumPy array of shape (7,4) in the documented sigma-point order.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def transformed_sigma_points(reflectivity_a, reflectivity_b, finesse,
                             transformed_covariance):
    """Return seven rows [Ra_h,Rb_h,F_h,weight_h] in the fixed order."""
    return np.empty((0, 4), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_transformed_sigma_points(reflectivity_a, reflectivity_b, finesse,
                                     transformed_covariance):
    """Seven positive-weight sigma points in [logit(Ra),logit(Rb),log(F)]."""
    import math
    import numpy as np
    ra = float(reflectivity_a)
    rb = float(reflectivity_b)
    finesse = float(finesse)
    covariance = np.asarray(transformed_covariance, dtype=float)
    if not (0.0 < ra < 1.0 and 0.0 < rb < 1.0 and finesse > 0.0):
        raise ValueError("invalid nominal optical parameters")
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ValueError("transformed_covariance must be a finite 3-by-3 matrix")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-14):
        raise ValueError("transformed_covariance must be symmetric")
    try:
        root = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("transformed_covariance must be positive definite") from exc
    mean = np.array([
        math.log(ra / (1.0 - ra)),
        math.log(rb / (1.0 - rb)),
        math.log(finesse),
    ], dtype=float)
    transformed = [mean]
    scale = math.sqrt(6.0)
    for k in range(3):
        transformed.extend((mean + scale * root[:, k], mean - scale * root[:, k]))

    def inverse_map(x):
        ra_i = 1.0 / (1.0 + math.exp(-float(x[0])))
        rb_i = 1.0 / (1.0 + math.exp(-float(x[1])))
        return [ra_i, rb_i, math.exp(float(x[2]))]

    weights = [0.5] + [1.0 / 12.0] * 6
    return np.asarray([inverse_map(x) + [weights[i]]
                       for i, x in enumerate(transformed)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nc=np.array([[.01,.007,-.00125],[.007,.04,.002],[-.00125,.002,.0025]])", "call":"transformed_sigma_points(.985,.9995,400.,c)", "gold_call":"_oracle_transformed_sigma_points(.985,.9995,400.,c)"},
        {"setup":"import numpy as np\nc=np.diag([.004,.009,.0016])", "call":"transformed_sigma_points(.97,.998,250.,c)", "gold_call":"_oracle_transformed_sigma_points(.97,.998,250.,c)"},
        {"setup":"import numpy as np\nc=np.array([[.02,-.006,.001],[-.006,.03,-.0015],[.001,-.0015,.004]])", "call":"transformed_sigma_points(.99,.999,600.,c)", "gold_call":"_oracle_transformed_sigma_points(.99,.999,600.,c)"},
        {"setup":"import numpy as np\nc=np.diag([.001,.002,.0004])", "call":"transformed_sigma_points(.9,.95,100.,c)", "gold_call":"_oracle_transformed_sigma_points(.9,.95,100.,c)"},
        {"setup":"import numpy as np\nc=np.array([[.006,.002,0.],[.002,.01,.001],[0.,.001,.003]])", "call":"transformed_sigma_points(.995,.9998,900.,c)", "gold_call":"_oracle_transformed_sigma_points(.995,.9998,900.,c)"},
        {"setup":"import numpy as np\nc=np.array([[.012,-.003,-.001],[-.003,.018,.002],[-.001,.002,.005]])", "call":"transformed_sigma_points(.8,.97,75.,c)", "gold_call":"_oracle_transformed_sigma_points(.8,.97,75.,c)"},
        {"setup":"import numpy as np\nc=np.array([[.003,.001,.0002],[.001,.007,-.0003],[.0002,-.0003,.001]])", "call":"transformed_sigma_points(.92,.995,320.,c)", "gold_call":"_oracle_transformed_sigma_points(.92,.995,320.,c)"}
    ]
