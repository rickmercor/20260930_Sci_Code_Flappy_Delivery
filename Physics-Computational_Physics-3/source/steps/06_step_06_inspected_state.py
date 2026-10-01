"""
One externally scheduled inspection with current geometric and support fields.

The inspected identifier is an input, not an inferred event scheduler. Apply
this task's support test to fields recomputed from the data supplied in this
call, return any newly created sample to the ellipsoid, and preserve the
ordering of the surviving rows. A later visit must recompute its own fields
from the state this call returns.

Return an (m,4) numeric array, each row [id, x, y, z], after one inspection.

Surviving rows keep their incoming order. points has shape (n,3) with n>=2
and every sample on the ellipsoid; ids holds unique nonnegative integers.
axes, h0, tau, radius, lower, upper, mu, kref and packing carry the
meanings and admissible ranges of the earlier primitives. target must be
present in ids and new_id must be a nonnegative integer not already in
use. An additive event appends one row carrying new_id; a merging event
removes the inspected particle and its selected partner and then appends
one row carrying new_id; an unchanged outcome preserves every row and does
not consume new_id. Recompute the current fields through the earlier
subproblem functions rather than reimplementing them.

Raises
------
ValueError
    If new_id is not an unused nonnegative integer, if any condition of
    the earlier primitives fails on the supplied data, or if the proposal
    reaches a branch those primitives exclude.

Returns
-------
return
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def inspected_state(points: 'np.ndarray | list | tuple', ids: 'np.ndarray | list | tuple', axes: 'np.ndarray | list | tuple', h0: float, tau: float, radius: float, lower: float, upper: float, mu: 'np.ndarray | list | tuple', target: int, new_id: int, kref: float=0.0, packing: float=0.933) -> 'np.ndarray':
    """Return an (m,4) numeric array, each row [id, x, y, z], after one inspection.

    Surviving rows keep their incoming order. points has shape (n,3) with n>=2
    and every sample on the ellipsoid; ids holds unique nonnegative integers.
    axes, h0, tau, radius, lower, upper, mu, kref and packing carry the
    meanings and admissible ranges of the earlier primitives. target must be
    present in ids and new_id must be a nonnegative integer not already in
    use. An additive event appends one row carrying new_id; a merging event
    removes the inspected particle and its selected partner and then appends
    one row carrying new_id; an unchanged outcome preserves every row and does
    not consume new_id. Recompute the current fields through the earlier
    subproblem functions rather than reimplementing them.

    Raises
    ------
    ValueError
        If new_id is not an unused nonnegative integer, if any condition of
        the earlier primitives fails on the supplied data, or if the proposal
        reaches a branch those primitives exclude.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_inspected_state(points: 'np.ndarray | list | tuple', ids: 'np.ndarray | list | tuple', axes: 'np.ndarray | list | tuple', h0: float, tau: float, radius: float, lower: float, upper: float, mu: 'np.ndarray | list | tuple', target: int, new_id: int, kref: float=0.0, packing: float=0.933) -> 'np.ndarray':
    import numpy as np
    x=np.asarray(points,float);labels=np.asarray(ids)
    if not np.isscalar(new_id) or not np.isfinite(new_id) or new_id<0 or new_id!=int(new_id) or new_id in labels:
        raise ValueError('invalid new identifier')
    geo=_oracle_surface_geometry(x,axes)
    lengths=_oracle_characteristic_lengths(x,geo[:,3],h0,tau,kref,radius)[:,1]
    support=_oracle_occupied_support(x,lengths,packing)[:,1]
    proposal=_oracle_event_proposal(x,labels,lengths,geo[:,:3],support,target,lower,upper,mu)
    kind=int(proposal[0])
    if kind==0:return np.column_stack((labels,x))
    point=_oracle_surface_return(proposal[2:],axes)
    if kind==-1:
        keep=(labels!=target)&(labels!=int(proposal[1]))
        x=x[keep];labels=labels[keep]
    return np.column_stack((np.r_[labels,new_id],np.vstack((x,point))))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'x=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]);ids=[0,1,2];a=[1.,1.,1.];mu=[.13,-.21,.07]',
      'call': 'inspected_state(x,ids,a,1.,0.,2.,.0,10.,mu,0,10)',
      'gold_call': '_oracle_inspected_state(x,ids,a,1.,0.,2.,.0,10.,mu,0,10)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'x=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]);ids=[0,1,2];a=[1.,1.,1.];mu=[.13,-.21,.07]',
      'call': 'inspected_state(x,ids,a,1.,0.,2.,.7,1.25,mu,0,10)',
      'gold_call': '_oracle_inspected_state(x,ids,a,1.,0.,2.,.7,1.25,mu,0,10)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'x=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]);ids=[0,1,2];a=[1.,1.,1.];mu=[.13,-.21,.07]',
      'call': 'inspected_state(x,ids,a,1.,0.,2.,.0,.1,mu,0,10)',
      'gold_call': '_oracle_inspected_state(x,ids,a,1.,0.,2.,.0,.1,mu,0,10)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'x=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]);ids=[0,1,2];a=[1.,1.,1.];mu=[.13,-.21,.07]\n'
               'def _bad_public():\n'
               '    try: inspected_state(x,ids,a,1.,0.,2.,0.,10.,mu,0,1); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2\n'
               'def _bad_gold():\n'
               '    try: _oracle_inspected_state(x,ids,a,1.,0.,2.,0.,10.,mu,0,1); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2',
      'call': '_bad_public()',
      'gold_call': '_bad_gold()',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'x=np.array([[1.,0.,0.],[-1.,0.,0.]])\n'
               'ids=[0,1]; a=[1.,.8,.6]; mu=[.13,-.21,.07]',
      'call': 'inspected_state(x.copy(),ids.copy(),a.copy(),1.,0.,3.,0.,.1,mu.copy(),0,2)',
      'gold_call': '_oracle_inspected_state(x.copy(),ids.copy(),a.copy(),1.,0.,3.,0.,.1,mu.copy(),0,2)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'z=.6*np.sqrt(1-.3**2-(.1/.8)**2); x=np.array([[.3,.1,z],[.3,.1,-z]])\n'
               'ids=[0,1]; a=[1.,.8,.6]; mu=[.13,-.21,.07]',
      'call': 'inspected_state(x.copy(),ids.copy(),a.copy(),1.,0.,3.,0.,.1,mu.copy(),0,2)',
      'gold_call': '_oracle_inspected_state(x.copy(),ids.copy(),a.copy(),1.,0.,3.,0.,.1,mu.copy(),0,2)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'eps=1e-12; z=.6*np.sqrt(1-.3**2-(.1/.8)**2); x=np.array([[.3,.1,z], '
               '[.3,.1,-z+2*eps]]); x[1,:2] *= np.sqrt((1-(x[1,2]/.6)**2)/(.3**2+(.1/.8)**2))\n'
               'ids=[0,1]; a=[1.,.8,.6]; mu=[.13,-.21,.07]',
      'call': 'inspected_state(x.copy(),ids.copy(),a.copy(),1.,0.,3.,0.,.1,mu.copy(),0,2)',
      'gold_call': '_oracle_inspected_state(x.copy(),ids.copy(),a.copy(),1.,0.,3.,0.,.1,mu.copy(),0,2)',
      'tol': 1e-09}]
