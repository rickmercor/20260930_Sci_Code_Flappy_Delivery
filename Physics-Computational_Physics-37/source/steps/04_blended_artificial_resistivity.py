"""
Evaluate the paper's low-order, residual, and effective nodal resistivities.

For each node i, define the local length



h_i = mesh_h * node_scale_i.



Compute the low-order and residual resistivities



r_low,i = c_low * h_i * electron_speed_i,

r_res,i = c_res * h_i^2 * rescaled_residual_i.



Apply the residual/low-order minimum before the physical floor:



r_effective,i = max(physical_resistivity, min(r_low,i, r_res,i)).



All min and max operations are elementwise. Return columns [r_low, r_res, r_effective] in node order.

Returns
-------
Return one n-by-3 real NumPy array in node order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def blended_artificial_resistivity(mesh_h, physical_resistivity,
                                   electron_speed, rescaled_residual,
                                   node_scale, c_low=0.25, c_res=1.0):
    """Construct low-order, residual, and effective resistivities.

    Parameters
    ----------
    mesh_h : float
        Positive mesh size.
    physical_resistivity : float
        Nonnegative physical floor.
    electron_speed : array_like, shape (n,)
        Ordered nonnegative nodal speeds.
    rescaled_residual : array_like, shape (n,)
        Ordered nonnegative residual indicators.
    node_scale : array_like, shape (n,)
        Positive local length scales.
    c_low : float
        Nonnegative low-order constant.
    c_res : float
        Nonnegative residual constant.
    Returns
    -------
    ndarray, shape (n,3)
        Columns [r_low,r_res,r_effective] in node order.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_blended_artificial_resistivity(mesh_h, physical_resistivity,
                                           electron_speed, rescaled_residual,
                                           node_scale, c_low=0.25, c_res=1.0):
    """Evaluate the low-order, residual, and blended resistivities of Eqs. (48),(52)."""
    import math
    import numpy as np
    h = float(mesh_h)
    r = float(physical_resistivity)
    c_low = float(c_low)
    c_res = float(c_res)
    speed = np.asarray(electron_speed, dtype=float)
    residual = np.asarray(rescaled_residual, dtype=float)
    scale = np.asarray(node_scale, dtype=float)
    if speed.ndim != 1 or speed.size == 0 or residual.shape != speed.shape or scale.shape != speed.shape:
        raise ValueError("node arrays must be nonempty and equal length")
    if (not all(math.isfinite(x) for x in (h, r, c_low, c_res)) or h <= 0 or r < 0
            or c_low < 0 or c_res < 0 or not np.all(np.isfinite(speed))
            or not np.all(np.isfinite(residual)) or not np.all(np.isfinite(scale))
            or np.any(speed < 0) or np.any(residual < 0) or np.any(scale <= 0)):
        raise ValueError("invalid resistivity inputs")
    local_h = h * scale
    low = c_low * local_h * speed
    high = c_res * local_h * local_h * residual
    effective = np.maximum(r, np.minimum(low, high))
    return np.column_stack((low, high, effective)).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\ns=np.array([1.,2.]); q=np.array([10.,20.]); z=np.array([1.,1.])', 'call': 'blended_artificial_resistivity(.1,.004,s,q,z)', 'gold_call': '_oracle_blended_artificial_resistivity(.1,.004,s,q,z)'}, {'setup': 'import numpy as np\ns=np.array([0.,.1]); q=np.array([0.,.01]); z=np.array([.5,2.])', 'call': 'blended_artificial_resistivity(.2,.03,s,q,z)', 'gold_call': '_oracle_blended_artificial_resistivity(.2,.03,s,q,z)'}, {'setup': 'import numpy as np\ns=np.array([4.,1.]); q=np.array([100.,100.]); z=np.array([1.,1.5])', 'call': 'blended_artificial_resistivity(.05,.001,s,q,z,.5,.25)', 'gold_call': '_oracle_blended_artificial_resistivity(.05,.001,s,q,z,.5,.25)'}, {'setup': 'import numpy as np\ns=np.array([.4,.8,1.2]); q=np.array([2.,5.,9.]); z=np.array([.8,1.,1.3])', 'call': 'blended_artificial_resistivity(.12,.006,s,q,z)', 'gold_call': '_oracle_blended_artificial_resistivity(.12,.006,s,q,z)'}, {'setup': 'import numpy as np\ns=np.linspace(.2,2.,5); q=np.linspace(1.,25.,5); z=np.linspace(.7,1.3,5)', 'call': 'blended_artificial_resistivity(.075,.008,s,q,z)', 'gold_call': '_oracle_blended_artificial_resistivity(.075,.008,s,q,z)'}, {'setup': 'import numpy as np\ns=np.array([10.,.01]); q=np.array([.01,100.]); z=np.array([2.,.4])', 'call': 'blended_artificial_resistivity(.03,.002,s,q,z)', 'gold_call': '_oracle_blended_artificial_resistivity(.03,.002,s,q,z)'}, {'setup': 'import numpy as np\ns=np.array([.9,1.1,1.4,1.8]); q=np.array([8.,12.,20.,30.]); z=np.array([.9,1.,1.1,1.2])', 'call': 'blended_artificial_resistivity(.06,.004,s,q,z,.3,1.2)', 'gold_call': '_oracle_blended_artificial_resistivity(.06,.004,s,q,z,.3,1.2)'}]
