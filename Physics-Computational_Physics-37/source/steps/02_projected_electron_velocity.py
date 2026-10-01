"""
Evaluate the projected electron velocity used by the induction subsystem.

For every node i, compute the projected electron velocity



v_e,i = v_i - (d_i / rho) J_i,



where rho is the positive scenario density, d_i is the nonnegative ion skin depth, v_i is the Cartesian ion velocity, and J_i is the projected current. Compute the Euclidean speed ||v_e,i||_2 and return columns [v_e,x, v_e,y, v_e,z, ||v_e||_2] in node order.

Returns
-------
Return one n-by-4 real NumPy array in node order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def projected_electron_velocity(density, velocity, projected_current, ion_skin_depth):
    """Form the nodal electron velocity and speed.

    Parameters
    ----------
    density : float
        Positive scenario density.
    velocity : array_like, shape (n,3)
        Ion velocity in Cartesian component order.
    projected_current : array_like, shape (n,3)
        Projected nodal current in matching order.
    ion_skin_depth : float
        Nonnegative Hall parameter.
    Returns
    -------
    ndarray, shape (n,4)
        Columns [ve_x,ve_y,ve_z,|ve|] in node order.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_projected_electron_velocity(density, velocity, projected_current, ion_skin_depth):
    """Evaluate the paper's projected electron velocity (47) at ordered nodes."""
    import math
    import numpy as np
    rho = float(density)
    di = float(ion_skin_depth)
    velocity = np.asarray(velocity, dtype=float)
    current = np.asarray(projected_current, dtype=float)
    if not math.isfinite(rho) or not math.isfinite(di) or rho <= 0 or di < 0:
        raise ValueError("rho must be positive and ion_skin_depth nonnegative")
    if velocity.ndim != 2 or velocity.shape[1] != 3 or current.shape != velocity.shape:
        raise ValueError("velocity and projected_current must have shape (n,3)")
    if velocity.shape[0] == 0 or not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(current)):
        raise ValueError("node data must be nonempty and finite")
    electron = velocity - (di / rho) * current
    return np.column_stack((electron, np.linalg.norm(electron, axis=1))).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nv=np.array([[1.,2.,3.],[0.,0.,0.]]); j=np.array([[.2,.4,.6],[1.,0.,0.]])', 'call': 'projected_electron_velocity(2.,v,j,.5)', 'gold_call': '_oracle_projected_electron_velocity(2.,v,j,.5)'}, {'setup': 'import numpy as np\nv=np.array([[1.,-1.,2.]]); j=np.array([[4.,3.,-2.]])', 'call': 'projected_electron_velocity(.5,v,j,0.)', 'gold_call': '_oracle_projected_electron_velocity(.5,v,j,0.)'}, {'setup': 'import numpy as np\nv=np.zeros((3,3)); j=np.eye(3)', 'call': 'projected_electron_velocity(.25,v,j,1.)', 'gold_call': '_oracle_projected_electron_velocity(.25,v,j,1.)'}, {'setup': 'import numpy as np\nv=np.array([[.2,-.1,0],[.3,.4,.5]]); j=np.array([[1.,2.,3.],[-1.,0.,2.]])', 'call': 'projected_electron_velocity(.8,v,j,.7)', 'gold_call': '_oracle_projected_electron_velocity(.8,v,j,.7)'}, {'setup': 'import numpy as np\nv=np.full((4,3),.2); j=np.arange(12.).reshape(4,3)/10', 'call': 'projected_electron_velocity(1.2,v,j,.3)', 'gold_call': '_oracle_projected_electron_velocity(1.2,v,j,.3)'}, {'setup': 'import numpy as np\nv=np.array([[0.,1.,0.],[1.,0.,1.]]); j=np.array([[0.,2.,0.],[2.,0.,-2.]])', 'call': 'projected_electron_velocity(.4,v,j,.8)', 'gold_call': '_oracle_projected_electron_velocity(.4,v,j,.8)'}, {'setup': 'import numpy as np\nv=np.linspace(-.2,.3,15).reshape(5,3); j=np.flip(v,axis=0)', 'call': 'projected_electron_velocity(.9,v,j,.55)', 'gold_call': '_oracle_projected_electron_velocity(.9,v,j,.55)'}]
