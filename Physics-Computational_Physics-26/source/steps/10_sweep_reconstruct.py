"""
Reconstruct every ghost value by visiting the dependency levels in increasing order, writing the values of one level before moving to the next. Within a level the ghost cells are independent of one another. Cells that are not ghosts are left untouched. A ghost cell is one tagged 2.0 or -2.0. The stencil is the general eight-term one: the ghost value is the sum over the seven stencil neighbours of that neighbour's weight times its current value, plus the eighth weight, so the caller supplies whichever closed form applies at each cell rather than the routine assuming one. Index the arrays [k, j, i] with k along z, j along y and i along x, so entry 0 of normals and mult belongs to the LAST index and entry 2 to the FIRST. With i2, j2 and k2 the indices one multiplier's worth of cells away in x, y and z, on the side the matching normal component points to, counting a vanishing component as the positive side, and clamped to the array bounds, the seven neighbours are taken in the order (k,j,i2), (k,j2,i), (k2,j,i), (k,j2,i2), (k2,j2,i), (k2,j,i2), (k2,j2,i2).

The reconstruction combines seven Cartesian neighbours with a boundary term, and the eighth weight absorbs whichever boundary datum applies, so one sweep serves both boundary-condition types.

Returns
-------
ndarray with the shape of field, float64: a copy of field with every ghost cell replaced by its reconstructed value.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sweep_reconstruct(field: "np.ndarray", tags: "np.ndarray", levels: "np.ndarray", weights: "np.ndarray", normals: "np.ndarray", mult: "np.ndarray") -> "np.ndarray":
    """Reconstruct every ghost value by visiting the dependency levels in increasing order, writing the values of one level before moving to the next. Within a level the ghost cells are independent of one another. Cells that are not ghosts are left untouched. A ghost cell is one tagged 2.0 or -2.0. The stencil is the general eight-term one: the ghost value is the sum over the seven stencil neighbours of that neighbour's weight times its current value, plus the eighth weight, so the caller supplies whichever closed form applies at each cell rather than the routine assuming one. Index the arrays [k, j, i] with k along z, j along y and i along x, so entry 0 of normals and mult belongs to the LAST index and entry 2 to the FIRST. With i2, j2 and k2 the indices one multiplier's worth of cells away in x, y and z, on the side the matching normal component points to, counting a vanishing component as the positive side, and clamped to the array bounds, the seven neighbours are taken in the order (k,j,i2), (k,j2,i), (k2,j,i), (k,j2,i2), (k2,j2,i), (k2,j,i2), (k2,j2,i2).

    Returns
    -------
    ndarray with the shape of field, float64: a copy of field with every ghost cell replaced by its reconstructed value.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sweep_reconstruct(field: "np.ndarray", tags: "np.ndarray", levels: "np.ndarray", weights: "np.ndarray", normals: "np.ndarray", mult: "np.ndarray") -> "np.ndarray":
    f = np.array(field, dtype=float, copy=True)
    t = np.asarray(tags, dtype=float)
    lv = np.asarray(levels, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = np.asarray(normals, dtype=float)
    r = np.asarray(mult, dtype=float)
    if w.shape[0] != 8:
        raise ValueError("weights must have shape (8,) + grid shape: seven neighbours and the boundary term")
    if not (f.shape == t.shape == lv.shape == w.shape[1:] == n.shape[1:] == r.shape[1:]):
        raise ValueError("every argument must agree on the grid shape")
    nz, ny, nx = f.shape
    ghost = (t == 2.0) | (t == -2.0)
    off = [(np.where(n[a] < 0.0, -1, 1) * r[a]).astype(int) for a in range(3)]
    nlev = int(lv[ghost].max()) + 1 if ghost.any() else 0
    for L in range(nlev):
        upd = {}
        for k, j, i in np.argwhere(ghost & (lv == float(L))):
            i2 = min(max(i + off[0][k, j, i], 0), nx - 1)
            j2 = min(max(j + off[1][k, j, i], 0), ny - 1)
            k2 = min(max(k + off[2][k, j, i], 0), nz - 1)
            p = [f[k, j, i2], f[k, j2, i], f[k2, j, i],
                 f[k, j2, i2], f[k2, j2, i], f[k2, j, i2], f[k2, j2, i2]]
            upd[(k, j, i)] = sum(w[s, k, j, i] * p[s] for s in range(7)) + w[7, k, j, i]
        for key, value in upd.items():
            f[key] = value
    return f

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nz=np.linspace(-0.9,0.9,31); y=np.linspace(-1.2,1.2,31); x=np.linspace(-1.5,1.5,31)\nZ,Y,X=np.meshgrid(z,y,x,indexing='ij')\npsi_c=signed_normal_distance(X,Y,Z,0.07,-0.11,0.05,0.61)\npsi_o=_oracle_signed_normal_distance(X,Y,Z,0.07,-0.11,0.05,0.61)\nn_c=boundary_normal(X,Y,Z,0.07,-0.11,0.05)\nn_o=_oracle_boundary_normal(X,Y,Z,0.07,-0.11,0.05)\nth_c=normal_cell_thickness(n_c,0.1,0.08,0.06)\nth_o=_oracle_normal_cell_thickness(n_o,0.1,0.08,0.06)\ntg_c=hybrid_ghost_tags(psi_c,th_c)\ntg_o=_oracle_hybrid_ghost_tags(psi_o,th_o)\nr_c=stencil_multipliers(psi_c,n_c,0.1,0.08,0.06)\nr_o=_oracle_stencil_multipliers(psi_o,n_o,0.1,0.08,0.06)\nlv_c=dependency_levels(tg_c,n_c,r_c)\nlv_o=_oracle_dependency_levels(tg_o,n_o,r_o)\nb_c=boundary_point(X,Y,Z,0.07,-0.11,0.05,0.61)\nb_o=_oracle_boundary_point(X,Y,Z,0.07,-0.11,0.05,0.61)\nv_c=trilinear_value_weights(X,Y,Z,b_c,n_c,r_c,0.1,0.08,0.06)\nv_o=_oracle_trilinear_value_weights(X,Y,Z,b_o,n_o,r_o,0.1,0.08,0.06)\nW_c=np.stack([v_c[0],v_c[1],v_c[2],-v_c[0]*v_c[1],-v_c[1]*v_c[2],-v_c[0]*v_c[2],v_c[0]*v_c[1]*v_c[2],(1-v_c[0])*(1-v_c[1])*(1-v_c[2])*np.sin(b_c[0])])\nF_c=np.where((tg_c==2.0)|(tg_c==-2.0),0.0,np.sin(X)*np.cos(Y)*np.sin(Z))",
            "call": 'sweep_reconstruct(F_c.copy(), tg_c.copy(), lv_c.copy(), W_c.copy(), n_c.copy(), r_c.copy())',
            "gold_call": '_oracle_sweep_reconstruct(np.where((tg_o==2.0)|(tg_o==-2.0),0.0,np.sin(X)*np.cos(Y)*np.sin(Z)).copy(), tg_o.copy(), lv_o.copy(), np.stack([v_o[0],v_o[1],v_o[2],-v_o[0]*v_o[1],-v_o[1]*v_o[2],-v_o[0]*v_o[2],v_o[0]*v_o[1]*v_o[2],(1-v_o[0])*(1-v_o[1])*(1-v_o[2])*np.sin(b_o[0])]).copy(), n_o.copy(), r_o.copy())',
        },
        {
            "setup": 'import numpy as np\nF=np.ones((3,3,3)); tg=np.zeros((3,3,3)); lv=np.full((3,3,3),-1.0)\nW=np.zeros((8,3,3,3)); n=np.stack([np.ones((3,3,3))]*3); r=np.ones((3,3,3,3))',
            "call": 'sweep_reconstruct(F.copy(), tg.copy(), lv.copy(), W.copy(), n.copy(), r.copy())',
            "gold_call": '_oracle_sweep_reconstruct(F.copy(), tg.copy(), lv.copy(), W.copy(), n.copy(), r.copy())',
        },
        {
            "setup": 'import numpy as np\nF=np.arange(27.0).reshape(3,3,3); tg=np.zeros((3,3,3)); tg[0,0,0]=-2.0\nlv=np.full((3,3,3),-1.0); lv[0,0,0]=0.0\nW=np.zeros((8,3,3,3)); W[0]=0.5; W[7]=1.25\nn=np.stack([np.ones((3,3,3))]*3); r=np.ones((3,3,3,3))',
            "call": 'sweep_reconstruct(F.copy(), tg.copy(), lv.copy(), W.copy(), n.copy(), r.copy())',
            "gold_call": '_oracle_sweep_reconstruct(F.copy(), tg.copy(), lv.copy(), W.copy(), n.copy(), r.copy())',
        },
        {
            "setup": 'import numpy as np\n# every stencil index clamps to the array bound here; wrapping instead changes the value.\nF=np.arange(8.0).reshape(2,2,2)\ntg=np.zeros((2,2,2)); tg[1,1,1]=-2.0\nlv=np.full((2,2,2),-1.0); lv[1,1,1]=0.0\nW=np.zeros((8,2,2,2)); W[0]=1.0; W[1]=2.0; W[2]=3.0; W[6]=4.0; W[7]=0.25\nn=np.stack([np.ones((2,2,2))]*3); r=np.ones((3,2,2,2))',
            "call": 'sweep_reconstruct(F.copy(), tg.copy(), lv.copy(), W.copy(), n.copy(), r.copy())',
            "gold_call": '_oracle_sweep_reconstruct(F.copy(), tg.copy(), lv.copy(), W.copy(), n.copy(), r.copy())',
        },
    ]
