"""
A support-triggered local adaptation proposal before surface return.

Inspect the single particle whose identifier equals target, using the support values supplied in this call, and report which of the three admissible outcomes that particle's support selects, together with the trial position that outcome proposes. normals supplies the outward unit normal at each sample; mu supplies a three-component perturbation consumed by the trial construction. No particle is added, removed or moved here, and no trial is returned to the surface here.

Definitions used by this task. Let k be the row whose identifier equals target, S_k its supplied support, h_k its characteristic spacing, x_k its position and n_k its outward unit normal.

Merge outcome, selected when S_k > upper: the partner j is the other particle at the smallest Euclidean distance from x_k (equal distances select the smaller identifier). Return [-1, ids[j], midpoint of x_k and x_j].

Insertion outcome, selected when S_k < lower: let rho_k(x) be the weighted density defined in the preceding step, with central evaluation position x and all characteristic lengths and other particle positions fixed. Its central self contribution is constant; a neighbor exactly at the kernel cutoff has zero derivative contribution. Define the tangential gradient G = (I - n_k n_k^T) grad rho_k evaluated at x_k. The trial is x_k - h_k (1 + mu) * G / ||G||, with componentwise multiplication denoted by *. The output is [+1, -1, trial]. A tangential gradient norm ||G|| <= 1e-10 is a ValueError.

Unchanged outcome otherwise, including S_k equal to either threshold: return [0, -1, x_k].

Return a length-5 float array [event, neighbor_id, trial_x, trial_y, trial_z].

event is +1 when the outcome would add a particle, -1 when it would merge
the inspected particle with a partner, and 0 when the particle set is
unchanged. neighbor_id is the partner's existing identifier for the
merging outcome and -1 otherwise. For the unchanged outcome the three
trial coordinates equal the inspected particle's own position.

points and normals have shape (n,3) with n>=2; ids has shape (n,) and
holds unique nonnegative integers, and target is one of them; lengths has
shape (n,) and is strictly positive; supports has shape (n,) and is
nonnegative; mu has shape (3,) with every component in [-0.5, 0.5]. Every
normal has unit norm to absolute tolerance 1e-8. All data and thresholds
are finite with 0 <= lower < upper. Support exactly equal to either
threshold selects the unchanged outcome. Ties in any distance-based
selection are resolved in favour of the smaller existing identifier.
Any finite-range sum evaluated for the inspected particle takes its range
from that particle's own characteristic spacing, never from another
sample's spacing or from any combination of two samples' spacings.

Raises
------
ValueError
    If any shape, identifier, finiteness, positivity, unit-norm, mu-range
    or threshold condition above fails, if two distinct samples are
    separated by less than 1e-12, or if a proposed trial direction has
    Euclidean norm 1e-10 or less.

Returns
-------
return
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def event_proposal(points: 'np.ndarray | list | tuple', ids: 'np.ndarray | list | tuple', lengths: 'np.ndarray | list | tuple', normals: 'np.ndarray | list | tuple', supports: 'np.ndarray | list | tuple', target: int, lower: float, upper: float, mu: 'np.ndarray | list | tuple') -> 'np.ndarray':
    """Return a length-5 float array [event, neighbor_id, trial_x, trial_y, trial_z].

    event is +1 when the outcome would add a particle, -1 when it would merge
    the inspected particle with a partner, and 0 when the particle set is
    unchanged. neighbor_id is the partner's existing identifier for the
    merging outcome and -1 otherwise. For the unchanged outcome the three
    trial coordinates equal the inspected particle's own position.

    points and normals have shape (n,3) with n>=2; ids has shape (n,) and
    holds unique nonnegative integers, and target is one of them; lengths has
    shape (n,) and is strictly positive; supports has shape (n,) and is
    nonnegative; mu has shape (3,) with every component in [-0.5, 0.5]. Every
    normal has unit norm to absolute tolerance 1e-8. All data and thresholds
    are finite with 0 <= lower < upper. Support exactly equal to either
    threshold selects the unchanged outcome. Ties in any distance-based
    selection are resolved in favour of the smaller existing identifier.
    Any finite-range sum evaluated for the inspected particle takes its range
    from that particle's own characteristic spacing, never from another
    sample's spacing or from any combination of two samples' spacings.

    Raises
    ------
    ValueError
        If any shape, identifier, finiteness, positivity, unit-norm, mu-range
        or threshold condition above fails, if two distinct samples are
        separated by less than 1e-12, or if a proposed trial direction has
        Euclidean norm 1e-10 or less.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_event_proposal(points: 'np.ndarray | list | tuple', ids: 'np.ndarray | list | tuple', lengths: 'np.ndarray | list | tuple', normals: 'np.ndarray | list | tuple', supports: 'np.ndarray | list | tuple', target: int, lower: float, upper: float, mu: 'np.ndarray | list | tuple') -> 'np.ndarray':
    import numpy as np
    x=np.asarray(points,float);labels=np.asarray(ids);h=np.asarray(lengths,float)
    n=np.asarray(normals,float);s=np.asarray(supports,float);m=np.asarray(mu,float)
    if x.ndim!=2 or x.shape[1:]!=(3,) or len(x)<2 or n.shape!=x.shape:
        raise ValueError('invalid proposal point shapes')
    if labels.shape!=(len(x),) or h.shape!=(len(x),) or s.shape!=(len(x),) or m.shape!=(3,):
        raise ValueError('invalid proposal arrays')
    if not all(np.isfinite(z).all() for z in (x,labels,h,n,s,m)):
        raise ValueError('nonfinite proposal data')
    if np.any(labels<0) or np.any(labels!=np.floor(labels)) or len(np.unique(labels))!=len(labels) or target not in labels:
        raise ValueError('invalid identifiers')
    if np.any(h<=0) or np.any(s<0) or np.any(abs(m)>.5) or np.any(abs(np.linalg.norm(n,axis=1)-1)>1e-8):
        raise ValueError('invalid proposal fields')
    if not np.isfinite([lower,upper]).all() or lower<0 or upper<=lower:
        raise ValueError('invalid thresholds')
    dvec=x[:,None,:]-x[None,:,:];d=np.linalg.norm(dvec,axis=2)
    if np.any(d[np.triu_indices(len(x),1)]<1e-12):
        raise ValueError('coincident samples')
    k=int(np.flatnonzero(labels==target)[0])
    if s[k]>upper:
        distance=d[k].copy();distance[k]=np.inf
        j=int(np.lexsort((labels,distance))[0])
        return np.r_[-1.,float(labels[j]),(x[k]+x[j])/2]
    if s[k]<lower:
        r=2*h[k]
        unit=np.divide(dvec[k],d[k,:,None],out=np.zeros_like(x),where=d[k,:,None]>0)
        weights=(h/h[k])**2
        grad=np.sum((-3/(np.pi*r**3))*weights[:,None]*unit*(d[k]<r)[:,None],axis=0)
        grad-=n[k]*np.dot(n[k],grad)
        norm=np.linalg.norm(grad)
        if norm<=1e-10:
            raise ValueError('active birth requires nonzero tangent gradient')
        return np.r_[1.,-1.,x[k]-h[k]*(1+m)*grad/norm]
    return np.r_[0.,-1.,x[k]]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'x=np.array([[0.,0.,0.],[.2,0.,0.],[0.,.3,0.]]);ids=[4,2,7];h=[.3,.4,.3];n=np.tile([0.,0.,1.],(3,1));mu=[.13,-.21,.07]',
      'call': 'event_proposal(x,ids,h,n,[.4,1.,1.],4,.7,1.25,mu)',
      'gold_call': '_oracle_event_proposal(x,ids,h,n,[.4,1.,1.],4,.7,1.25,mu)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'x=np.array([[0.,0.,0.],[.2,0.,0.],[0.,.3,0.]]);ids=[4,2,7];h=[.3,.4,.3];n=np.tile([0.,0.,1.],(3,1));mu=[.13,-.21,.07]',
      'call': 'event_proposal(x,ids,h,n,[1.4,1.,1.],4,.7,1.25,mu)',
      'gold_call': '_oracle_event_proposal(x,ids,h,n,[1.4,1.,1.],4,.7,1.25,mu)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'x=np.array([[0.,0.,0.],[.2,0.,0.],[0.,.3,0.]]);ids=[4,2,7];h=[.3,.4,.3];n=np.tile([0.,0.,1.],(3,1));mu=[.13,-.21,.07]',
      'call': 'event_proposal(x,ids,h,n,[.7,1.,1.],4,.7,1.25,mu)',
      'gold_call': '_oracle_event_proposal(x,ids,h,n,[.7,1.,1.],4,.7,1.25,mu)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'x=np.array([[0.,0.,0.],[.5,0.,0.],[.18,0.,0.]]);ids=[3,5,9];h=[.3,.3,.3];n=np.tile([0.,0.,1.],(3,1));mu=[.13,-.21,.07]',
      'call': 'event_proposal(x,ids,h,n,[1.4,1.,1.],3,.7,1.25,mu)',
      'gold_call': '_oracle_event_proposal(x,ids,h,n,[1.4,1.,1.],3,.7,1.25,mu)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'x=np.array([[0.,0.,0.],[.3,0.,0.],[-.3,0.,0.]]);ids=[5,9,2];h=[.3,.3,.3];n=np.tile([0.,0.,1.],(3,1));mu=[.13,-.21,.07]',
      'call': 'event_proposal(x,ids,h,n,[1.4,1.,1.],5,.7,1.25,mu)',
      'gold_call': '_oracle_event_proposal(x,ids,h,n,[1.4,1.,1.],5,.7,1.25,mu)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'x=np.array([[0.,0.,0.],[.6,0.,0.],[0.,.4,0.]]);ids=[1,2,3];h=[.3,.3,.3];n=np.tile([0.,0.,1.],(3,1));mu=[.13,-.21,.07]',
      'call': 'event_proposal(x,ids,h,n,[.4,1.,1.],1,.7,1.25,mu)',
      'gold_call': '_oracle_event_proposal(x,ids,h,n,[.4,1.,1.],1,.7,1.25,mu)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'x=np.array([[0.,0.,0.],[.3,0.,0.],[-.3,0.,0.]]);ids=[5,9,2];h=[.3,.3,.3];n=np.tile([0.,0.,1.],(3,1));mu=[.13,-.21,.07]\n'
               'def _sym_public():\n'
               '    try: event_proposal(x,ids,h,n,[.4,1.,1.],5,.7,1.25,mu); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2\n'
               'def _sym_gold():\n'
               '    try: _oracle_event_proposal(x,ids,h,n,[.4,1.,1.],5,.7,1.25,mu); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2',
      'call': '_sym_public()',
      'gold_call': '_sym_gold()',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'x=np.array([[0.,0.,0.],[.2,0.,0.],[0.,.3,0.]]);ids=[4,2,7];h=[.3,.4,.3];n=np.tile([0.,0.,1.],(3,1));mu=[.13,-.21,.07]\n'
               'def _bad_public():\n'
               '    try: event_proposal(x,ids,h,n,[1.,1.,1.],99,.7,1.25,mu); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2\n'
               'def _bad_gold():\n'
               '    try: _oracle_event_proposal(x,ids,h,n,[1.,1.,1.],99,.7,1.25,mu); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2',
      'call': '_bad_public()',
      'gold_call': '_bad_gold()',
      'tol': 1e-09}]
