"""
Evaluate the source-defined ordered multi-subunit assembly distance from rigid frames.

Compute the assembly distance defined in the matching treatment's simulation-evaluation section, with the candidate as its first configuration operand and the reference as its second. template has finite shape (A,3), A>=4, with affine rank three when singular values of the centered template above 1e-10 are counted. Both frame arrays have shape (M,7), M>=2; rows are fixed-order subunits and columns are [qw,qx,qy,qz,tx,ty,tz]. Quaternions are nonzero and are normalized before applying their active, right-handed rotations to column vectors; translations are in angstroms. Corresponding template rows identify the same atom. Use proper rotations. Return one nonnegative float in angstroms. Invalid shapes, nonfinite values, deficient template rank or a zero quaternion raise ValueError. Do not mutate inputs.

Returns
-------
assembly_distance : float — one finite nonnegative distance in angstroms.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def measure_assembly_distance(
    template: np.ndarray,
    candidate_frames: np.ndarray,
    reference_frames: np.ndarray,
) -> float:
    """Return the ordered candidate-to-reference assembly distance.
 
    Parameters
    ----------
    template
        Full-affine-rank landmark array of shape (A, 3).
    candidate_frames, reference_frames
        Rigid-frame arrays of shape (M, 7).
 
    Returns
    -------
    float
        Finite nonnegative distance in angstroms.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _frames_to_points(template,frames):
    import numpy as np
    a=np.asarray(template,dtype=float); f=np.asarray(frames,dtype=float)
    if a.ndim!=2 or a.shape[1]!=3 or a.shape[0]<4 or f.ndim!=2 or f.shape[1]!=7 or f.shape[0]<2:
        raise ValueError('template[N,3], N>=4, and frames[M,7], M>=2 required')
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(f)) or np.linalg.matrix_rank(a-a.mean(0),tol=1e-10)<3:
        raise ValueError('finite full affine rank template and frames required')
    norms=np.linalg.norm(f[:,:4],axis=1)
    if np.any(norms==0): raise ValueError('zero quaternion')
    pts=[]
    for row,norm in zip(f,norms):
        w,x,y,z=row[:4]/norm
        rot=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                      [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                      [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        pts.append(a@rot.T+row[4:])
    return np.asarray(pts)
 
def _kabsch_pair(q,u,v,w):
    import numpy as np
    qc=q-q.mean(0); vc=v-v.mean(0)
    left,_,right=np.linalg.svd(qc.T@vc)
    sign=np.linalg.det(left@right)
    r=left@np.diag([1.,1.,1. if sign>=0 else -1.])@right
    residual=(u-q.mean(0))@r+v.mean(0)-w
    return float(np.sqrt(np.mean(np.sum(residual**2,axis=1))))
 
def _oracle_measure_assembly_distance(
    template: np.ndarray,
    candidate_frames: np.ndarray,
    reference_frames: np.ndarray,
) -> float:
    import numpy as np
    import itertools
    x=_frames_to_points(template,candidate_frames)
    y=_frames_to_points(template,reference_frames)
    if x.shape!=y.shape: raise ValueError('matching candidate and reference shapes required')
    m=len(x); best=float('inf')
    for perm in itertools.permutations(range(m)):
        total=0.
        for i,j in itertools.combinations(range(m),2):
            total+=min(_kabsch_pair(x[i],x[j],y[perm[i]],y[perm[j]]),
                       _kabsch_pair(x[i],x[j],y[perm[j]],y[perm[i]]))
        best=min(best,total/(m*(m-1)/2))
    return float(best)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'a=np.array([[0.,0.,0.],[1.7,.2,-.4],[-.3,1.2,.5],[.1,-.8,2.1]])\n'
               'r=np.array([[1.,0,0,0,0,0,0],[1.,0,0,0,5,0,0],[1.,0,0,0,0,6,0]])\n'
               'c=r.copy()\n'
               'c[2,4]+=2.3\n',
      'call': 'measure_assembly_distance(a,c,r)',
      'gold_call': '_oracle_measure_assembly_distance(a,c,r)'},
     {'setup': 'import numpy as np\n'
               'a=np.array([[0.,0.,0.],[1.7,.2,-.4],[-.3,1.2,.5],[.1,-.8,2.1]])\n'
               'r=np.array([[1.,0,0,0,0,0,0],[1.,0,0,0,5,0,0],[1.,0,0,0,0,6,0]])\n'
               'c=r.copy()\n'
               'c[:,:4]*=-2.7;c[:,4:]+=np.array([3.,-7.,1.])\n',
      'call': 'measure_assembly_distance(a,c,r)',
      'gold_call': '_oracle_measure_assembly_distance(a,c,r)'},
     {'setup': 'import numpy as np\n'
               'a=np.array([[0.,0.,0.],[1.7,.2,-.4],[-.3,1.2,.5],[.1,-.8,2.1]])\n'
               'r=np.array([[1.,0,0,0,0,0,0],[1.,0,0,0,5,0,0],[1.,0,0,0,0,6,0]])\n'
               'c=r.copy()\n'
               'c[0,:4]=[.8,.2,-.1,.3];c[1,4:]+=[-.2,.8,.3];r=r[[2,0,1]]\n',
      'call': 'measure_assembly_distance(a,c,r)',
      'gold_call': '_oracle_measure_assembly_distance(a,c,r)'},
     {'setup': 'import numpy as np\n'
               'a=np.array([[0.,0.,0.],[1.7,.2,-.4],[-.3,1.2,.5],[.1,-.8,2.1]])\n'
               'r=np.array([[1.,0,0,0,0,0,0],[1.,0,0,0,5,0,0],[1.,0,0,0,0,6,0]])\n'
               'c=r.copy()\n'
               'c[1,:4]=0\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        measure_assembly_distance(a,c,r)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_measure_assembly_distance(a,c,r)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
