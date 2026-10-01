"""
Final orchestrator: finite local inspection response on a curved cloud.



Generate the complete ellipsoid instance, append the tangent-ring input, run

the three supplied inspections, and obtain the occupied-area-weighted second

moment. The final orchestrator must use the earlier subproblem functions.

Return the final finite scalar sum_i A_i*x_i^2/sum_i A_i.

Fixed axes are (1,.8,.6), h0=.42, kref=0, packing=.933 and mu=(.13,-.21,.07).
Generate 48 points: t=1-2(i+.5)/48, phi=i*pi*(3-sqrt(5)),
x=(sqrt(1-t*t)*cos(phi),.8*sqrt(1-t*t)*sin(phi),.6*t); delete x>.6,z>.1.
At surviving original ID34, let n be its outward normal,
e=normalize(n cross (0,0,1)), f=n cross e. Append closest points to
x34+ring_radius*(cos(2*pi*k/5)*e+sin(2*pi*k/5)*f), k=0,...,4, IDs100+k.
Inspect IDs [26,34,26], assigning new IDs [200,201,202] by visit if active.
Re-estimate fields between visits and for the final areas. tau>=0,
radius>0, ring_radius>0, 0<=lower<upper; all parameters finite. The final
particle count is not fixed.

Raises
------
ValueError
    If any parameter is not finite or violates the ranges above, if any
    branch excluded by an earlier primitive is reached, or if an
    identifier a later inspection needs has already been removed.

Returns
-------
return
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def adapted_moment(tau: float=0.5, radius: float=0.55, lower: float=0.7, upper: float=1.25, ring_radius: float=0.12) -> float:
    """Return the final finite scalar sum_i A_i*x_i^2/sum_i A_i.

    Fixed axes are (1,.8,.6), h0=.42, kref=0, packing=.933 and mu=(.13,-.21,.07).
    Generate 48 points: t=1-2(i+.5)/48, phi=i*pi*(3-sqrt(5)),
    x=(sqrt(1-t*t)*cos(phi),.8*sqrt(1-t*t)*sin(phi),.6*t); delete x>.6,z>.1.
    At surviving original ID34, let n be its outward normal,
    e=normalize(n cross (0,0,1)), f=n cross e. Append closest points to
    x34+ring_radius*(cos(2*pi*k/5)*e+sin(2*pi*k/5)*f), k=0,...,4, IDs100+k.
    Inspect IDs [26,34,26], assigning new IDs [200,201,202] by visit if active.
    Re-estimate fields between visits and for the final areas. tau>=0,
    radius>0, ring_radius>0, 0<=lower<upper; all parameters finite. The final
    particle count is not fixed.

    Raises
    ------
    ValueError
        If any parameter is not finite or violates the ranges above, if any
        branch excluded by an earlier primitive is reached, or if an
        identifier a later inspection needs has already been removed.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_adapted_moment(tau: float=0.5, radius: float=0.55, lower: float=0.7, upper: float=1.25, ring_radius: float=0.12) -> float:
    import numpy as np
    if not np.isfinite([tau,radius,lower,upper,ring_radius]).all() or tau<0 or radius<=0 or ring_radius<=0 or lower<0 or upper<=lower:
        raise ValueError('invalid instance parameters')
    axes=np.array([1.,.8,.6]);ids=np.arange(48)
    t=1-2*(ids+.5)/48;phi=ids*np.pi*(3-np.sqrt(5))
    x=np.column_stack((np.sqrt(1-t*t)*np.cos(phi),np.sqrt(1-t*t)*np.sin(phi),t))*axes
    keep=~((x[:,0]>.6)&(x[:,2]>.1));x=x[keep];ids=ids[keep]
    index=int(np.flatnonzero(ids==34)[0]);center=x[index]
    normal=_oracle_surface_geometry(x,axes)[index,:3]
    e=np.cross(normal,[0.,0.,1.]);e/=np.linalg.norm(e);f=np.cross(normal,e)
    ring=[]
    for k in range(5):
        trial=center+ring_radius*(np.cos(2*np.pi*k/5)*e+np.sin(2*np.pi*k/5)*f)
        ring.append(_oracle_surface_return(trial,axes))
    x=np.vstack((x,ring));ids=np.r_[ids,np.arange(100,105)]
    for target,new_id in [(26,200),(34,201),(26,202)]:
        # Redundant API consistency check; this adds no scientific novelty credit.
        geo=_oracle_surface_geometry(x,axes)
        lengths=_oracle_characteristic_lengths(x,geo[:,3],.42,tau,0.,radius)[:,1]
        supports=_oracle_occupied_support(x,lengths,.933)[:,1]
        proposal=_oracle_event_proposal(x,ids,lengths,geo[:,:3],supports,target,lower,upper,[.13,-.21,.07])
        expected_count=len(x)+int(proposal[0])
        packed=_oracle_inspected_state(x,ids,axes,.42,tau,radius,lower,upper,[.13,-.21,.07],target,new_id)
        if len(packed)!=expected_count:
            raise ValueError('proposal and state update disagree')
        ids=packed[:,0].astype(int);x=packed[:,1:]
    geo=_oracle_surface_geometry(x,axes)
    h=_oracle_characteristic_lengths(x,geo[:,3],.42,tau,0.,radius)[:,1]
    areas=_oracle_occupied_support(x,h,.933)[:,0]
    return float(np.dot(areas,x[:,0]**2)/np.sum(areas))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n',
      'call': 'adapted_moment()',
      'gold_call': '_oracle_adapted_moment()',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n',
      'call': 'adapted_moment(lower=0.,upper=10.)',
      'gold_call': '_oracle_adapted_moment(lower=0.,upper=10.)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n',
      'call': 'adapted_moment(radius=0.6)',
      'gold_call': '_oracle_adapted_moment(radius=0.6)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'def _bad_public():\n'
               '    try: adapted_moment(radius=-.1); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2\n'
               'def _bad_gold():\n'
               '    try: _oracle_adapted_moment(radius=-.1); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2',
      'call': '_bad_public()',
      'gold_call': '_bad_gold()',
      'tol': 1e-09}]
