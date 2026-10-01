"""
Assign cubic primitives to nonpercolating void owners in one fixed forward pass. Input cubes are integer rows (i,j,k,r) in acceptance order, with nonnegative radius and inclusive boxes contained in a cubic grid of integer side n from 5 through 65. Sort by decreasing raw cell volume (2r+1)**3, breaking ties by input row. Labels are input row plus one. Start an empty ownership grid, unset cube owners and false major flags. At each cube's turn use its existing owner if assigned; otherwise inspect owned cells within its box expanded by one and clamped to the grid, choosing the cell nearest its seed by squared Euclidean index distance with ties in ascending (k,j,i). If there is no such cell, make the cube its own permanent major. Paint the current unexpanded box only into zero cells. Then scan all other cubes in fixed volume order; skip majors and assigned cubes, and immediately assign and paint any cube whose box and the current cube's box EACH expanded by one intersect on all three axes. Absorbed cubes also perform this sweep at their own turn using their original bounds. Return the integer ownership grid, with zero for unowned cells. Empty cubes must have shape (0,4). Invalid inputs raise ValueError.

A major owner cannot subsequently be absorbed, so this construction is not a connected-component closure of the proximity graph. Shared cells retain their first owner rather than being transferred to the group with the largest accumulated volume. The one-cell expansions affect the linking predicates only; they add no cells to a void's physical volume.

Returns
-------
integer array, the unique owner of every grid cell
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def merge_void_cubes(cubes, n):
    """Return the cell ownership grid after the ordered merge pass.

    Parameters
    ----------
    cubes : array_like
        Integer array of shape (m,4), rows are center coordinates and radius.
    n : int
        Grid side from 5 through 65, excluding booleans.

    Returns
    -------
    ndarray
        Integer (n,n,n) ownership array with positive acceptance-order labels.

    Raises
    ------
    ValueError
        If the side, cube shape, integer values or box bounds are invalid.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_merge_void_cubes(cubes, n):
    if isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or not 5<=n<=65:
        raise ValueError("invalid grid side")
    n=int(n)
    try:c=np.asarray(cubes)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError("invalid cubes") from exc
    if c.ndim!=2 or c.shape[1]!=4 or c.dtype.kind not in 'iu' or np.any(c<0) or np.any(c>=n):
        raise ValueError("integer cube rows required")
    c=c.astype(np.int64)
    lo=c[:,:3]-c[:,3,None];hi=c[:,:3]+c[:,3,None]
    if np.any(lo<0) or np.any(hi>=n):raise ValueError("cube leaves grid")
    order=sorted(range(len(c)),key=lambda a:(-int((2*c[a,3]+1)**3),a))
    grid=np.zeros((n,n,n),dtype=int);owner=np.zeros(len(c),dtype=int);major=np.zeros(len(c),dtype=bool)
    def paint(a,o):
        view=grid[tuple(slice(int(lo[a,k]),int(hi[a,k])+1) for k in range(3))]
        view[view==0]=o
    for a in order:
        o=int(owner[a])
        if o==0:
            low=np.maximum(lo[a]-1,0);high=np.minimum(hi[a]+1,n-1);best=None
            for k in range(int(low[2]),int(high[2])+1):
                for j in range(int(low[1]),int(high[1])+1):
                    for i in range(int(low[0]),int(high[0])+1):
                        if grid[i,j,k]>0:
                            key=(int(np.sum((c[a,:3]-[i,j,k])**2)),k,j,i)
                            if best is None or key<best:
                                best=key;o=int(grid[i,j,k])
            if o==0:o=a+1;major[a]=True
            owner[a]=o
        paint(a,o)
        for b in order:
            if b==a or major[b] or owner[b]!=0:continue
            if np.all(lo[a]-1<=hi[b]+1) and np.all(lo[b]-1<=hi[a]+1):
                owner[b]=o;paint(b,o)
    return grid

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    cases=[
        {'setup':'c=np.array([[5,10,10,2],[17,10,10,2],[11,10,10,2]],dtype=int)\n','call':'merge_void_cubes(c,23)','gold_call':'_oracle_merge_void_cubes(c,23)'},
        {'setup':'c=np.array([[5,5,5,1],[8,5,5,1],[11,5,5,1]],dtype=int)\n','call':'merge_void_cubes(c,17)','gold_call':'_oracle_merge_void_cubes(c,17)'},
        {'setup':'c=np.array([[2,2,2,0],[4,2,2,0]],dtype=int)\n','call':'merge_void_cubes(c,7)','gold_call':'_oracle_merge_void_cubes(c,7)'},
        {'setup':'c=np.empty((0,4),dtype=int)\n','call':'merge_void_cubes(c,5)','gold_call':'_oracle_merge_void_cubes(c,5)'},
        {'setup':'c=np.array([[5,5,5,2],[8,5,5,2]],dtype=int)\n','call':'merge_void_cubes(c,15)','gold_call':'_oracle_merge_void_cubes(c,15)'},
    ]
    for args in ('[[0,0,0,1]],5','[[2.,2.,2.,1.]],5','np.empty((0,4),dtype=int),4'):
        setup=''
        for name,function in (('run_model','merge_void_cubes'),('run_gold','_oracle_merge_void_cubes')):
            setup+=f'def {name}():\n    try:\n        {function}({args})\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    return cases
