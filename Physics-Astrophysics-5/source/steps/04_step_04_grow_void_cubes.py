"""
Grow isotropic cubes from an already ordered list of seed cells using six face-barrier arrays. The barrier shape is (3,2,n,n,n), with n from 5 through 65, axes x,y,z and direction indices minus then plus. Seeds are distinct integer coordinate rows of shape (m,3) inside the grid. Visit them in the supplied order, skipping a seed with any coordinate at most 1 or at least n-2 or inside an accepted cube. Start radius r=0. Before every trial reject the whole cube if any current bound is at most 1 or at least n-2. Otherwise inspect every barrier on its six current square faces, with transverse coordinates spanning only the current cube. Any barrier terminates growth at r; no barriers increases r by one along all axes together. Reject terminal radius zero, append each other cube as (i,j,k,r) and mark its inclusive box for subsequent seed exclusion. Marking does not prohibit cube overlap. Return an integer (q,4) array in acceptance order. Empty output has shape (0,4); invalid arrays raise ValueError.

The all-face stopping condition preserves cubic primitives instead of allowing anisotropic parallelepipeds. A face flag represents an axial neighbor test, so the newly added edges and corners are not part of the next-cell test at the same trial. This distinction affects the final primitive sizes even on a radially symmetric underdensity. The explicit boundary rejection is the benchmark's finite-grid guard.

Returns
-------
integer array, accepted cube centers and radii
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def grow_void_cubes(seeds, barriers):
    """Return accepted inclusive cubes in seed visitation order.

    Parameters
    ----------
    seeds : array_like
        Distinct integer coordinate rows with shape (m,3).
    barriers : array_like
        Integer or boolean zero/one array with shape (3,2,n,n,n).

    Returns
    -------
    ndarray
        Integer accepted-cube rows (i,j,k,r), shape (q,4).

    Raises
    ------
    ValueError
        If shapes, integer values, seed uniqueness or bounds are invalid.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_grow_void_cubes(seeds, barriers):
    try:
        b=np.asarray(barriers);s=np.asarray(seeds)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError("invalid arrays") from exc
    if b.ndim!=5 or b.shape[:2]!=(3,2) or len(set(b.shape[2:]))!=1 or not 5<=b.shape[2]<=65:
        raise ValueError("invalid barrier shape")
    if b.dtype.kind not in 'biu' or not np.isin(b,[0,1]).all():raise ValueError("binary barriers required")
    n=b.shape[2]
    if s.ndim!=2 or s.shape[1]!=3 or s.dtype.kind not in 'iu' or np.any(s<0) or np.any(s>=n):
        raise ValueError("integer seed coordinates required")
    rows=[tuple(int(t) for t in p) for p in s]
    if len(set(rows))!=len(rows):raise ValueError("duplicate seeds")
    marked=np.zeros((n,n,n),bool);cubes=[]
    for p in rows:
        if min(p)<=1 or max(p)>=n-2 or marked[p]:continue
        center=np.array(p);r=0;reject=False
        while True:
            lo=center-r;hi=center+r
            if np.any(lo<=1) or np.any(hi>=n-2):reject=True;break
            blocked=False
            for a in range(3):
                for si in (0,1):
                    ix=[slice(int(lo[k]),int(hi[k])+1) for k in range(3)]
                    ix[a]=int(lo[a] if si==0 else hi[a])
                    if np.any(b[(a,si)+tuple(ix)]):blocked=True
            if blocked:break
            r+=1
        if reject or r==0:continue
        cubes.append([*p,r])
        marked[tuple(slice(int(center[k]-r),int(center[k]+r)+1) for k in range(3))]=True
    return np.array(cubes,dtype=int).reshape(-1,4)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base='b=np.zeros((3,2,11,11,11),dtype=int);b[0,1,7,:,:]=1\ns=np.array([[5,5,5],[6,5,5],[1,1,1]],dtype=int)\n'
    cases=[{'setup':base,'call':'grow_void_cubes(s,b)','gold_call':'_oracle_grow_void_cubes(s,b)'},
           {'setup':base+'b[:]=1\n','call':'grow_void_cubes(s,b)','gold_call':'_oracle_grow_void_cubes(s,b)'},
           {'setup':base+'b[:]=0\n','call':'grow_void_cubes(s,b)','gold_call':'_oracle_grow_void_cubes(s,b)'},
           {'setup':base+'s=np.empty((0,3),dtype=int)\n','call':'grow_void_cubes(s,b)','gold_call':'_oracle_grow_void_cubes(s,b)'},
           {'setup':base+'b[:]=0;b[1,0,5,4,5]=1\n','call':'grow_void_cubes(s,b)','gold_call':'_oracle_grow_void_cubes(s,b)'}]
    for args in ('s.astype(float),b','s,b+2','np.array([[5,5,5],[5,5,5]]),b'):
        setup=base
        for name,function in (('run_model','grow_void_cubes'),('run_gold','_oracle_grow_void_cubes')):
            setup+=f'def {name}():\n    try:\n        {function}({args})\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    return cases
