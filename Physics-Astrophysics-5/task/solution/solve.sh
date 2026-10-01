#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def void_fields(n, centers, scales):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not 5 <= n <= 65:
        raise ValueError("n must be an integer from 5 through 65")
    arrays = []
    for value in (centers, scales):
        try:
            raw = np.asarray(value)
            if raw.dtype.kind not in 'iuf' or raw.ndim != 2 or raw.shape[1] != 3 or len(raw) == 0:
                raise ValueError("nonempty real (m,3) array required")
            arr = raw.astype(float)
        except (TypeError, OverflowError) as exc:
            raise ValueError("invalid array") from exc
        if not np.isfinite(arr).all():
            raise ValueError("finite arrays required")
        arrays.append(arr)
    c, a = arrays
    if c.shape != a.shape or np.any(c < 0) or np.any(c > n-1) or np.any(a < .25) or np.any(a > n):
        raise ValueError("centers or scales outside domain")
    xyz = np.indices((n,n,n), dtype=float)
    s = np.full((n,n,n), np.inf)
    for center, scale in zip(c,a):
        value = np.sum(((xyz-center[:,None,None,None])/scale[:,None,None,None])**2, axis=0)
        s = np.minimum(s, value)
    return 1-.92*np.exp(-s), 1.2-s+1e-6*(xyz[0]+n*xyz[1]+n*n*xyz[2])

import numpy as np


def void_candidates(density_ratio, divergence, density_threshold):
    arrays = []
    for value in (density_ratio, divergence):
        try:
            raw = np.asarray(value)
            if raw.dtype.kind not in 'iuf' or raw.ndim != 3 or len(set(raw.shape)) != 1 or not 5 <= raw.shape[0] <= 65:
                raise ValueError("real cubic field required")
            arr = raw.astype(float)
        except (TypeError, OverflowError) as exc:
            raise ValueError("invalid field") from exc
        if not np.isfinite(arr).all():
            raise ValueError("finite fields required")
        arrays.append(arr)
    d,v = arrays
    if d.shape != v.shape or np.any(d < 0):
        raise ValueError("invalid density or field shape")
    if isinstance(density_threshold,(bool,np.bool_)) or not np.isscalar(density_threshold) or np.iscomplexobj(density_threshold):
        raise ValueError("real scalar required")
    try:
        threshold = float(density_threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("real scalar required") from exc
    if not np.isfinite(threshold):
        raise ValueError("finite threshold required")
    cells = np.argwhere((d-1 < threshold) & (v > 0))
    result = sorted(cells.tolist(),key=lambda p:(-v[tuple(p)],p[2],p[1],p[0]))
    return np.array(result,dtype=int).reshape(-1,3)

import numpy as np


def face_barriers(density_ratio, divergence, gradient_threshold, density_threshold, divergence_threshold):
    arrays=[]
    for value in (density_ratio,divergence):
        try:
            raw=np.asarray(value)
            if raw.dtype.kind not in 'iuf' or raw.ndim!=3 or len(set(raw.shape))!=1 or not 5<=raw.shape[0]<=65:
                raise ValueError("real cubic array required")
            arr=raw.astype(float)
        except (TypeError,OverflowError) as exc:
            raise ValueError("invalid field") from exc
        if not np.isfinite(arr).all():raise ValueError("finite fields required")
        arrays.append(arr)
    d,v=arrays
    if d.shape!=v.shape or np.any(d<0):raise ValueError("invalid density or shape")
    thresholds=[]
    for value in (gradient_threshold,density_threshold,divergence_threshold):
        if isinstance(value,(bool,np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value):raise ValueError("real scalar required")
        try: value=float(value)
        except (TypeError,ValueError,OverflowError) as exc:raise ValueError("invalid scalar") from exc
        if not np.isfinite(value):raise ValueError("finite scalar required")
        thresholds.append(value)
    gt,dt,vt=thresholds
    if gt<0:raise ValueError("nonnegative gradient threshold required")
    n=d.shape[0];out=np.ones((3,2)+d.shape,dtype=int)
    for a in range(3):
        for si,sign in enumerate((-1,1)):
            cur=[slice(None)]*3;near=cur.copy();far=cur.copy()
            if sign>0:
                cur[a]=slice(0,n-2);near[a]=slice(1,n-1);far[a]=slice(2,n)
            else:
                cur[a]=slice(2,n);near[a]=slice(1,n-1);far[a]=slice(0,n-2)
            # Halve before subtraction to avoid overflow at finite density extremes.
            deriv=d[tuple(far)]/2-d[tuple(cur)]/2
            out[(a,si)+tuple(cur)]=((deriv>=gt)|(v[tuple(near)]<vt)|(d[tuple(near)]-1>=dt)).astype(int)
    return out

import numpy as np


def grow_void_cubes(seeds, barriers):
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

import numpy as np


def merge_void_cubes(cubes, n):
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

import numpy as np


def owned_void_volume(ownership, cell_side):
    try:o=np.asarray(ownership)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError("invalid grid") from exc
    if o.ndim!=3 or len(set(o.shape))!=1 or not 5<=o.shape[0]<=65 or o.dtype.kind not in 'iu':
        raise ValueError("integer cubic grid required")
    if np.any(o<0) or np.any(o>o.size):raise ValueError("invalid ownership labels")
    if isinstance(cell_side,(bool,np.bool_)) or not np.isscalar(cell_side) or np.iscomplexobj(cell_side):
        raise ValueError("real cell side required")
    try:h=float(cell_side)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError("invalid cell side") from exc
    if not np.isfinite(h) or h<0:raise ValueError("nonnegative finite cell side required")
    _,counts=np.unique(o[o>0],return_counts=True)
    maximum=int(counts.max(initial=0))
    from fractions import Fraction
    value=maximum*Fraction(h)**3
    try:answer=float(value)
    except OverflowError as exc:raise ValueError("volume is not representable") from exc
    if not np.isfinite(answer):raise ValueError("volume is not finite")
    return answer

def largest_void(n, centers, scales, cell_side):
    d,v=void_fields(n,centers,scales)
    seeds=void_candidates(d,v,-.6)
    barriers=face_barriers(d,v,.25,10.,0.)
    cubes=grow_void_cubes(seeds,barriers)
    ownership=merge_void_cubes(cubes,n)
    return owned_void_volume(ownership,cell_side)
SCICODE_GOLD_EOF
