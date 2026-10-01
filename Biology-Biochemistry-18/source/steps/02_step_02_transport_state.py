"""
Resolve the transported species and free-proton count for every transport record and condition.

The source transport convention selects a chemical species using its interior-pool abundance. Experimental identity takes precedence; pure proton transport has no organic species.

Returns
-------
A numerical ndarray of shape (P,T,4).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transport_state(reactants: "np.ndarray", transports: "np.ndarray",
                    species_h: "np.ndarray", species_z: "np.ndarray") -> "np.ndarray":
    """
    reactants is the (P,2,M,3+J) output contract of step 1. species_h and
    species_z have length J. transports has shape (T,6), integer columns
    (reaction index, reactant index, source compartment, destination
    compartment, extra free-proton count, species override). Compartments
    are 0 exterior and 1 interior; source and destination differ. Reactant
    index -1 denotes pure proton transport; override -1 denotes the source
    default. Reaction indices are distinct nonnegative integers. Counts
    are nonnegative. Return (P,T,4): transported bound-proton count,
    transported organic charge, extra free-proton count, selected species
    index. Pure proton rows use (0,0,k,-1). T may be zero. Resolve equal
    maximum abundances by the lowest species index. Invalid dimensions,
    noninteger records or out-of-range species/compartment indices raise
    ValueError. Other inputs satisfy the preceding step contracts.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_transport_state(reactants: "np.ndarray", transports: "np.ndarray",
                    species_h: "np.ndarray", species_z: "np.ndarray") -> "np.ndarray":
    import numpy as np
    a = np.asarray(reactants, dtype=float)
    t = np.asarray(transports)
    h, z = np.asarray(species_h, dtype=float), np.asarray(species_z, dtype=float)
    if (a.ndim != 4 or a.shape[1] != 2 or t.ndim != 2 or t.shape[1] != 6
            or a.shape[-1] != 3 + h.size or z.shape != h.shape
            or not np.isfinite(a).all() or not np.isfinite(t).all()
            or not np.equal(t, np.floor(t)).all()):
        raise ValueError('Aligned reactants and integer transport rows required')
    t = t.astype(int)
    if (np.any(t[:, 0] < 0) or len(set(t[:, 0])) != len(t)
            or np.any((t[:, 1] < -1) | (t[:, 1] >= a.shape[2]))
            or np.any((t[:, 2:4] < 0) | (t[:, 2:4] > 1))
            or np.any(t[:, 2] == t[:, 3]) or np.any(t[:, 4] < 0)
            or np.any((t[:, 5] < -1) | (t[:, 5] >= h.size))):
        raise ValueError('Invalid transport index, compartment or species')
    out = np.zeros((a.shape[0], len(t), 4))
    for p in range(a.shape[0]):
        for r, (_, compound, src, dst, k, override) in enumerate(t):
            if compound == -1:
                out[p, r] = [0., 0., k, -1.]
            else:
                index = override if override >= 0 else int(np.argmax(a[p, 1, compound, 3:]))
                out[p, r] = [h[index], z[index], k, index]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'a=np.array([[[[0.,.9,-.1,.1,.9]],[[0.,.2,-.8,.8,.2]]]])\n'
               'h=np.array([0.,1.]); z=h-1.\n'
               't=np.array([[0,0,0,1,1,-1],[1,0,1,0,0,1],[2,-1,0,1,1,-1]])',
      'call': 'transport_state(a,t,h,z)',
      'gold_call': '_oracle_transport_state(a,t,h,z)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'a=np.array([[[[0.,.9,-.1,.1,.9]],[[0.,.2,-.8,.8,.2]]]])\n'
               'h=np.array([0.,1.]); z=h-1.\n'
               'a[:,:,:,3:]=.5\n'
               't=np.array([[0,0,1,0,0,-1]])',
      'call': 'transport_state(a,t,h,z)',
      'gold_call': '_oracle_transport_state(a,t,h,z)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'a=np.array([[[[0.,.9,-.1,.1,.9]],[[0.,.2,-.8,.8,.2]]]])\n'
               'h=np.array([0.,1.]); z=h-1.\n'
               't=np.empty((0,6),dtype=int)',
      'call': 'transport_state(a,t,h,z)',
      'gold_call': '_oracle_transport_state(a,t,h,z)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'a=np.array([[[[0.,.9,-.1,.1,.9]],[[0.,.2,-.8,.8,.2]]]])\n'
               'h=np.array([0.,1.]); z=h-1.\n'
               't=np.array([[0,0,0,1,0,2]])\n'
               '\n'
               'def _raises_value_error(fn, arguments):\n'
               '    try:\n'
               '        fn(*arguments)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_raises_value_error(transport_state, (a,t,h,z,))',
      'gold_call': '_raises_value_error(_oracle_transport_state, (a,t,h,z,))',
      'tol': 0.0}]
