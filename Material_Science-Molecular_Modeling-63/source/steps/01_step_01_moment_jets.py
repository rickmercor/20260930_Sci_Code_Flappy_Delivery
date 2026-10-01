"""
Compute periodic moment tensors of ranks zero, one and two and their first and second Cartesian derivatives.

Moment tensors encode the local atomic environment through weighted periodic neighbor sums. Include every image inside the cutoff, including nonzero self images. Self-image vectors are fixed under displacement of their base atom and therefore contribute zero coordinate derivatives.

Returns
-------
return m, g, h
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def moment_jets(positions, box, species, pair_weights, cutoff):
    """Compute periodic moment tensors and their Cartesian derivatives through order two.
    
    Parameters
    ----------
    positions : (N,3) real array, N >= 1
        Cartesian coordinates; wrap each coordinate into [0, box[a]).
    box : (3,) positive real array
        Orthorhombic cell lengths; cell is fixed during differentiation.
    species : (N,) integer array
        Type indices in [0,S).
    pair_weights : (S,S) real array
        Directed type weights c[central_type, neighbor_type].
    cutoff : positive float
        Include every image d=r_j+box*n-r_i with 0<|d|<cutoff,
        n in Z^3. Include nonzero self images; exclude (i=j,n=0).
    
    Define w(d)=c*(1-|d|/cutoff)^4 inside the cutoff and zero outside.
    M0_i=sum w; M1_i=sum w*d; M2_i=sum w*outer(d,d).
    The coordinate order is x=(r_0x,r_0y,r_0z,r_1x,...).
    Image labels stay fixed under differentiation. Nonzero self-image
    vectors do not change when their atom moves. At |d|=cutoff all returned
    contributions vanish. No minimum-image replacement of the image sum.
    
    Returns
    -------
    m : (N,13) float array
        Concatenate M0, M1, and row-major M2.
    g : (N,13,3*N) float array
        First Cartesian derivatives of m.
    h : (N,13,3*N,3*N) float array
        Second Cartesian derivatives of m, with no factorial normalization.
    
    Raises
    ------
    ValueError
        If an array has the wrong shape, entries are nonfinite, box or cutoff
        is nonpositive, species is not integer-valued or out of range, or
        distinct atoms have periodic separation below 1e-10.

    Runtime
    -------
    Import required libraries inside the submitted function and inside any
    helper that uses them. Do not assume that np, itertools or erfc exists
    in the function's global namespace. Imports in test setup are separate.
    """
    return (m, g, h)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _mj_validate(positions, box, species, pair_weights, cutoff):
    import numpy as np
    r = np.asarray(positions, dtype=float)
    L = np.asarray(box, dtype=float)
    z0 = np.asarray(species)
    c = np.asarray(pair_weights, dtype=float)
    if r.ndim != 2 or r.shape[1:] != (3,) or len(r) < 1 or (L.shape != (3,)):
        raise ValueError('geometry shape')
    if c.ndim != 2 or c.shape[0] != c.shape[1] or len(c) < 1 or (z0.shape != (len(r),)):
        raise ValueError('type shape')
    if not all((np.all(np.isfinite(a)) for a in (r, L, z0, c))) or not np.isfinite(cutoff):
        raise ValueError('nonfinite input')
    if np.any(L <= 0) or cutoff <= 0 or np.any(z0 != np.floor(z0)) or np.any(z0 < 0) or np.any(z0 >= len(c)):
        raise ValueError('input domain')
    z = z0.astype(int)
    r = np.mod(r, L)
    for i in range(len(r)):
        for j in range(i):
            d = r[i] - r[j]
            d -= L * np.rint(d / L)
            if np.linalg.norm(d) < 1e-10:
                raise ValueError('coincident sites')
    return (r, L, z, c)

def _oracle_moment_jets(positions, box, species, pair_weights, cutoff):
    """Reference implementation."""
    import numpy as np
    import itertools
    r, L, z, c = _mj_validate(positions, box, species, pair_weights, cutoff)
    n, D = (len(r), 3 * len(r))
    m = np.zeros((n, 13))
    g = np.zeros((n, 13, D))
    h = np.zeros((n, 13, D, D))
    bounds = np.ceil(cutoff / L).astype(int)
    eye = np.eye(3)
    for i in range(n):
        for j in range(n):
            B = np.zeros((3, D))
            B[:, 3 * j:3 * j + 3] += eye
            B[:, 3 * i:3 * i + 3] -= eye
            for image in itertools.product(*(range(-v, v + 1) for v in bounds)):
                if i == j and image == (0, 0, 0):
                    continue
                d = r[j] + L * np.array(image) - r[i]
                distance = np.linalg.norm(d)
                if distance >= cutoff:
                    continue
                e = d / distance
                t = 1 - distance / cutoff
                weight = c[z[i], z[j]]
                w = weight * t ** 4
                wp = -4 * weight * t ** 3 / cutoff
                wpp = 12 * weight * t ** 2 / cutoff ** 2
                dw = wp * e
                hw = wpp * np.outer(e, e) + wp / distance * (eye - np.outer(e, e))
                vals = np.r_[1.0, d, np.outer(d, d).ravel()]
                dv = np.zeros((13, 3))
                dv[1:4] = eye
                hv = np.zeros((13, 3, 3))
                for a in range(3):
                    for b in range(3):
                        k = 4 + 3 * a + b
                        dv[k] = eye[a] * d[b] + eye[b] * d[a]
                        hv[k] = np.outer(eye[a], eye[b]) + np.outer(eye[b], eye[a])
                local_g = vals[:, None] * dw + w * dv
                local_h = vals[:, None, None] * hw + w * hv + np.einsum('ka,b->kab', dv, dw) + np.einsum('a,kb->kab', dw, dv)
                m[i] += w * vals
                g[i] += local_g @ B
                h[i] += np.einsum('ad,kab,be->kde', B, local_h, B)
    return (m, g, h)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Normal, boundary, edge and declared-invalid fixtures."""
    return [
        {
            'setup': (
                'import numpy as np\n'
                'r=np.array([[.1,.2,.3],[1.4,1.7,.9],[2.5,.4,2.]])\n'
                'L=np.array([3.,3.6,4.2])\n'
                'z=np.array([0,1,0])\n'
                'c=np.array([[.8,1.2],[.6,1.]])\n'
                'r_g = r.copy(); L_g = L.copy(); z_g = z.copy(); c_g = c.copy()\n'
            ),
            'call': 'moment_jets(r,L,z,c,4.1)',
            'gold_call': '_oracle_moment_jets(r_g,L_g,z_g,c_g,4.1)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'r=np.array([[.2,.3,.4]])\n'
                'r_g = r.copy()\n'
            ),
            'call': 'moment_jets(r,[2.,2.5,3.],[0],[[1.]],3.1)',
            'gold_call': '_oracle_moment_jets(r_g,[2.,2.5,3.],[0],[[1.]],3.1)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'r=np.array([[0.,0.,0.],[1.5,0.,0.]])\n'
                'r_g = r.copy()\n'
            ),
            'call': 'moment_jets(r,[5.,5.,5.],[0,0],[[1.]],1.5)',
            'gold_call': '_oracle_moment_jets(r_g,[5.,5.,5.],[0,0],[[1.]],1.5)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'r=np.array([[4.1,-2.8,.3],[1.4,1.7,.9]])\n'
                'r_g = r.copy()\n'
            ),
            'call': 'moment_jets(r,[3.,3.6,4.2],[0,1],[[.8,-1.2],[.6,1.]],2.2)',
            'gold_call': '_oracle_moment_jets(r_g,[3.,3.6,4.2],[0,1],[[.8,-1.2],[.6,1.]],2.2)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'r=np.array([[0.,0.,0.],[3.,0.,0.]])\n'
                'def _raise_code(f):\n'
                '    try:\n'
                '        f()\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        raise\n'
                '    raise AssertionError("Expected ValueError")\n'
                'r_g = r.copy()\n'
            ),
            'call': '_raise_code(lambda: moment_jets(r,[3.,4.,5.],[0,0],[[1.]],2.))',
            'gold_call': '_raise_code(lambda: _oracle_moment_jets(r_g,[3.,4.,5.],[0,0],[[1.]],2.))',
        },
    ]
