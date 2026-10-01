"""
Construct six outward growth barriers on a nonperiodic cubic grid with unit index spacing. Return an integer array with shape (3,2,n,n,n), where axis a is x, y or z and direction index 0 means minus while 1 means plus. A current face cell p tests its neighbor t=p+s*e_a. It is blocked when the outward centered density derivative [density_ratio(t+s*e_a)-density_ratio(t-s*e_a)]/2 is at least gradient_threshold, divergence(t) is below divergence_threshold or density_ratio(t)-1 is at least density_threshold. Current coordinates at most 1 on a minus face or at least n-2 on a plus face are blocked. There is no gradient lookahead. Fields are finite real cubic arrays of equal shape with side 5 through 65 and nonnegative density ratio; thresholds are finite real scalars and gradient_threshold is nonnegative. Invalid inputs raise ValueError.

The derivative is signed along the outward normal rather than the magnitude of the three-dimensional gradient. Falling density in the outward direction does not create a positive barrier. Density contrast differs from density ratio only by a constant, so their centered derivatives agree. These flags concern prospective axial neighbors; cube growth decides which flags belong to its current square faces.

Returns
-------
integer array, six growth barriers at each current face cell
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def face_barriers(density_ratio, divergence, gradient_threshold, density_threshold, divergence_threshold):
    """Return integer blocking flags for six directions.

    Parameters
    ----------
    density_ratio : array_like
        Nonnegative finite real cubic field with side 5 through 65.
    divergence : array_like
        Finite real field of the same shape.
    gradient_threshold : float
        Nonnegative finite outward-derivative threshold.
    density_threshold : float
        Finite density-contrast threshold.
    divergence_threshold : float
        Finite growth divergence threshold.

    Returns
    -------
    ndarray
        Zero or one array with shape (3,2,n,n,n).

    Raises
    ------
    ValueError
        If shapes, values or thresholds violate the contract.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_face_barriers(density_ratio, divergence, gradient_threshold, density_threshold, divergence_threshold):
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base='x=np.indices((7,7,7),dtype=float);d=.1+.02*x[0]**2;v=np.ones_like(d)\n'
    cases=[
        {'setup':base,'call':'face_barriers(d,v,.1,10.,0.)','gold_call':'_oracle_face_barriers(d,v,.1,10.,0.)'},
        {'setup':base+'v[:]=0\n','call':'face_barriers(d,v,.1,10.,0.)','gold_call':'_oracle_face_barriers(d,v,.1,10.,0.)'},
        {'setup':base,'call':'face_barriers(d,v,0.,10.,0.)','gold_call':'_oracle_face_barriers(d,v,0.,10.,0.)'},
        {'setup':base,'call':'face_barriers(d,v,.1,-.6,2.)','gold_call':'_oracle_face_barriers(d,v,.1,-.6,2.)'},
    ]
    for args in ('d,v,-1.,10.,0.','d,v,.1,float("nan"),0.','d,v[:2],.1,10.,0.'):
        setup=base
        for name,function in (('run_model','face_barriers'),('run_gold','_oracle_face_barriers')):
            setup+=f'def {name}():\n    try:\n        {function}({args})\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    return cases
