"""
Assign every ghost cell to a dependency level, so that the reconstruction of a ghost cell in one level needs only cells that are already resolved. A ghost cell is one tagged 2.0 or -2.0; every other cell carries -1.0. A ghost cell whose active stencil neighbours contain no other ghost cell is at level 0; otherwise its level is one more than the largest level among the ghost cells in its active stencil. Index the arrays [k, j, i] with k along z, j along y and i along x, so entry 0 of normals and mult belongs to the LAST index and entry 2 to the FIRST. Write i2, j2 and k2 for the index one multiplier's worth of cells away in x, y and z, on the side the matching normal component points to, counting a vanishing component as the positive side, and clamped to the array bounds. The seven stencil neighbours are then (k,j,i2), (k,j2,i), (k2,j,i), (k,j2,i2), (k2,j2,i), (k2,j,i2) and (k2,j2,i2). Omit a neighbour whenever reaching it changes a coordinate whose normal component is zero: its reconstruction weight vanishes, so it creates no dependency. Memoise, because the same cell is reached from many others. Raise ValueError if the dependencies contain a cycle, which an active stencil dependency that clamps back onto its own cell produces.

A ghost cell's stencil can contain other ghost cells, which is normally resolved by iterating to convergence. Ordering the ghost cells instead removes the iteration entirely, and the ordering is what makes the sweep well posed.

Returns
-------
ndarray with the shape of tags, float64: the level index of each ghost cell as a float, counting from 0.0, and -1.0 at every cell that is not a ghost.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dependency_levels(tags: "np.ndarray", normals: "np.ndarray", mult: "np.ndarray") -> "np.ndarray":
    """Assign every ghost cell to a dependency level, so that the reconstruction of a ghost cell in one level needs only cells that are already resolved. A ghost cell is one tagged 2.0 or -2.0; every other cell carries -1.0. A ghost cell whose active stencil neighbours contain no other ghost cell is at level 0; otherwise its level is one more than the largest level among the ghost cells in its active stencil. Index the arrays [k, j, i] with k along z, j along y and i along x, so entry 0 of normals and mult belongs to the LAST index and entry 2 to the FIRST. Write i2, j2 and k2 for the index one multiplier's worth of cells away in x, y and z, on the side the matching normal component points to, counting a vanishing component as the positive side, and clamped to the array bounds. The seven stencil neighbours are then (k,j,i2), (k,j2,i), (k2,j,i), (k,j2,i2), (k2,j2,i), (k2,j,i2) and (k2,j2,i2). Omit a neighbour whenever reaching it changes a coordinate whose normal component is zero: its reconstruction weight vanishes, so it creates no dependency. Memoise, because the same cell is reached from many others. Raise ValueError if the dependencies contain a cycle, which an active stencil dependency that clamps back onto its own cell produces.

    Returns
    -------
    ndarray with the shape of tags, float64: the level index of each ghost cell as a float, counting from 0.0, and -1.0 at every cell that is not a ghost.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_dependency_levels(tags: "np.ndarray", normals: "np.ndarray", mult: "np.ndarray") -> "np.ndarray":
    t = np.asarray(tags, dtype=float)
    n = np.asarray(normals, dtype=float)
    r = np.asarray(mult, dtype=float)
    if n.shape[0] != 3 or r.shape[0] != 3:
        raise ValueError("normals and mult must have leading dimension 3")
    if n.shape[1:] != t.shape or r.shape[1:] != t.shape:
        raise ValueError("normals and mult must match the shape of tags")
    nz, ny, nx = t.shape
    ghost = (t == 2.0) | (t == -2.0)
    off = [(np.where(n[a] < 0.0, -1, 1) * r[a]).astype(int) for a in range(3)]
    lev = np.full(t.shape, -1.0)
    memo = {}

    def _neighbours(k, j, i):
        i2 = min(max(i + off[0][k, j, i], 0), nx - 1)
        j2 = min(max(j + off[1][k, j, i], 0), ny - 1)
        k2 = min(max(k + off[2][k, j, i], 0), nz - 1)
        points = [(k, j, i2), (k, j2, i), (k2, j, i),
                  (k, j2, i2), (k2, j2, i), (k2, j, i2), (k2, j2, i2)]
        directions = [(0,), (1,), (2,), (0, 1), (1, 2), (0, 2), (0, 1, 2)]
        return [q for q, axes in zip(points, directions)
                if all(n[a, k, j, i] != 0.0 for a in axes)]

    def _level_of(p, active):
        if p in memo:
            return memo[p]
        if p in active:
            raise ValueError("the ghost-cell dependencies contain a cycle")
        active = active | {p}
        deps = [q for q in _neighbours(*p) if ghost[q]]
        value = 0 if not deps else 1 + max(_level_of(q, active) for q in deps)
        memo[p] = value
        return value

    for p in (tuple(v) for v in np.argwhere(ghost)):
        lev[p] = float(_level_of(p, frozenset()))
    return lev

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nz=np.linspace(-0.9,0.9,31); y=np.linspace(-1.2,1.2,31); x=np.linspace(-1.5,1.5,31)\nZ,Y,X=np.meshgrid(z,y,x,indexing='ij')\npsi_c=signed_normal_distance(X,Y,Z,0.07,-0.11,0.05,0.61)\npsi_o=_oracle_signed_normal_distance(X,Y,Z,0.07,-0.11,0.05,0.61)\nn_c=boundary_normal(X,Y,Z,0.07,-0.11,0.05)\nn_o=_oracle_boundary_normal(X,Y,Z,0.07,-0.11,0.05)\nth_c=normal_cell_thickness(n_c,0.1,0.08,0.06)\nth_o=_oracle_normal_cell_thickness(n_o,0.1,0.08,0.06)\ntg_c=hybrid_ghost_tags(psi_c,th_c)\ntg_o=_oracle_hybrid_ghost_tags(psi_o,th_o)\nr_c=stencil_multipliers(psi_c,n_c,0.1,0.08,0.06)\nr_o=_oracle_stencil_multipliers(psi_o,n_o,0.1,0.08,0.06)",
            "call": 'dependency_levels(tg_c.copy(), n_c.copy(), r_c.copy())',
            "gold_call": '_oracle_dependency_levels(tg_o.copy(), n_o.copy(), r_o.copy())',
        },
        {
            "setup": 'import numpy as np\ntg=np.zeros((3,3,3))\nn=np.stack([np.ones((3,3,3))]*3)\nr=np.ones((3,3,3,3))',
            "call": 'dependency_levels(tg.copy(), n.copy(), r.copy())',
            "gold_call": '_oracle_dependency_levels(tg.copy(), n.copy(), r.copy())',
        },
        {
            "setup": 'import numpy as np\ntg=np.zeros((3,3,3)); tg[1,1,1]=2.0\nn=np.stack([np.ones((3,3,3))]*3)\nr=np.ones((3,3,3,3))',
            "call": 'dependency_levels(tg.copy(), n.copy(), r.copy())',
            "gold_call": '_oracle_dependency_levels(tg.copy(), n.copy(), r.copy())',
        },
        {
            "setup": 'import numpy as np\ntg=np.zeros((3,3,3)); tg[0,0,0]=2.0; tg[1,1,1]=2.0\nn=np.stack([np.ones((3,3,3))]*3)\nr=np.ones((3,3,3,3))',
            "call": 'dependency_levels(tg.copy(), n.copy(), r.copy())',
            "gold_call": '_oracle_dependency_levels(tg.copy(), n.copy(), r.copy())',
        },
        {
            "setup": 'import numpy as np\ntg=np.zeros((3,3,3)); tg[1,1,1]=-2.0\nn=np.stack([-np.ones((3,3,3)),np.ones((3,3,3)),-np.ones((3,3,3))])\nr=np.ones((3,3,3,3))',
            "call": 'dependency_levels(tg.copy(), n.copy(), r.copy())',
            "gold_call": '_oracle_dependency_levels(tg.copy(), n.copy(), r.copy())',
        },
        {
            "setup": 'import numpy as np\n# a ghost whose stencil clamps back onto itself is a cycle, not a level\ntg=np.zeros((2,2,2)); tg[0,0,0]=2.0\nn=np.stack([-np.ones((2,2,2))]*3)\nr=np.ones((3,2,2,2))\ndef probe(f):\n    try:\n        f()\n        return 0.0\n    except ValueError:\n        return 1.0\n',
            "call": 'probe(lambda: dependency_levels(tg.copy(), n.copy(), r.copy()))',
            "gold_call": 'probe(lambda: _oracle_dependency_levels(tg.copy(), n.copy(), r.copy()))',
        },
        {
            "setup": 'import numpy as np\ntg=np.zeros((5,5,5)); tg[2,2,2]=2.0; tg[2,3,2]=2.0\nn=np.zeros((3,5,5,5)); n[0]=1.0\nr=np.ones((3,5,5,5))',
            "call": 'dependency_levels(tg.copy(), n.copy(), r.copy())',
            "gold_call": '_oracle_dependency_levels(tg.copy(), n.copy(), r.copy())',
        },
        {
            "setup": 'import numpy as np\n# a vanishing normal component takes the POSITIVE side, which decides whether\n# these two ghosts depend on each other at all.\ntg=np.zeros((3,3,3)); tg[0,0,0]=2.0; tg[0,0,1]=2.0\nn=np.stack([np.zeros((3,3,3)),np.ones((3,3,3)),np.ones((3,3,3))])\nr=np.ones((3,3,3,3))',
            "call": 'dependency_levels(tg.copy(), n.copy(), r.copy())',
            "gold_call": '_oracle_dependency_levels(tg.copy(), n.copy(), r.copy())',
        },
    ]
