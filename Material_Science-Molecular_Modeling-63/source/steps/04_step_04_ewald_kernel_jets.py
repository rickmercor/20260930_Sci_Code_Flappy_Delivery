"""
Construct the finite periodic Ewald kernel and its first and second Cartesian derivatives.

The Coulomb kernel contains real-space, reciprocal-space, self and uniform-background terms. The specified image and reciprocal-index bounds define the finite numerical model. Differentiate the position-dependent kernel terms at fixed cell; retain the coordinate-independent terms in the kernel value.

Returns
-------
return k, kg, kh
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ewald_kernel_jets(positions, box, alpha, real_extent, reciprocal_extent):
    """Build a point-charge Ewald matrix and its first two Cartesian derivatives.
    
    Parameters
    ----------
    positions : (N,3) real array, N>=1
        Wrap each coordinate into [0,box[a]); hold cell and image labels fixed.
    box : (3,) positive real array
    alpha : positive float
        Ewald inverse length. Coulomb prefactor is exactly 1.
    real_extent : nonnegative integer
        Include every integer image n in [-real_extent,real_extent]^3.
    reciprocal_extent : (3,) nonnegative integer array
        Include every m with -M[a]<=m[a]<=M[a], except m=(0,0,0).
    
    Let V=prod(box), k=2*pi*m/box and d_ij,n=r_i-r_j+box*n.
    K_ij=sum_n' erfc(alpha*|d_ij,n|)/|d_ij,n|
     + (4*pi/V)*sum_k exp(-k^2/(4*alpha^2))*cos(k dot (r_i-r_j))/k^2
     - (2*alpha/sqrt(pi))*delta_ij - pi/(alpha^2*V).
    The prime removes only i=j,n=0. Both +/- reciprocal modes occur.
    The last term is the uniform neutralizing-background convention and
    occurs in every entry. The self term is diagonal. Neither has coordinate
    derivatives. No real-space distance cutoff is applied. These finite sums
    are the definition of the task; no adaptive convergence changes.
    Coordinate index a=3*i+component. Charge values are not arguments.
    
    Returns
    -------
    k : (N,N) float array
    kg : (N,N,3*N) float array
    kh : (N,N,3*N,3*N) float array
        Value, first derivative and unnormalized second derivative of K.
    
    Raises
    ------
    ValueError
        If shapes do not match, inputs are nonfinite, box or alpha is
        nonpositive, extents are not nonnegative integers, or distinct atoms
        have periodic separation below 1e-10.

    Runtime
    -------
    Import required libraries inside the submitted function and inside any
    helper that uses them. Do not assume that np, itertools or erfc exists
    in the function's global namespace. Imports in test setup are separate.
    """
    return (k, kg, kh)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_ewald_kernel_jets(positions, box, alpha, real_extent, reciprocal_extent):
    """Reference implementation."""
    import numpy as np
    import itertools
    from scipy.special import erfc
    r = np.asarray(positions, float)
    L = np.asarray(box, float)
    M0 = np.asarray(reciprocal_extent)
    if r.ndim != 2 or r.shape[1:] != (3,) or len(r) < 1 or (L.shape != (3,)) or (M0.shape != (3,)):
        raise ValueError('shape')
    if not all((np.all(np.isfinite(a)) for a in (r, L, M0))) or not np.isfinite(alpha) or (not np.isfinite(real_extent)):
        raise ValueError('nonfinite')
    if np.any(L <= 0) or alpha <= 0 or real_extent < 0 or (real_extent != int(real_extent)) or np.any(M0 < 0) or np.any(M0 != np.floor(M0)):
        raise ValueError('domain')
    r = np.mod(r, L)
    M = M0.astype(int)
    R = int(real_extent)
    n = len(r)
    D = 3 * n
    V = L.prod()
    eye = np.eye(3)
    for i in range(n):
        for j in range(i):
            d = r[i] - r[j]
            d -= L * np.rint(d / L)
            if np.linalg.norm(d) < 1e-10:
                raise ValueError('coincident sites')
    modes = np.array([v for v in itertools.product(*(range(-v, v + 1) for v in M)) if v != (0, 0, 0)], float).reshape(-1, 3)
    kv = 2 * np.pi * modes / L
    k2 = np.einsum('ka,ka->k', kv, kv)
    weights = 4 * np.pi / V * np.exp(-k2 / (4 * alpha ** 2)) / k2
    K = np.zeros((n, n))
    G = np.zeros((n, n, D))
    H = np.zeros((n, n, D, D))
    images = list(itertools.product(range(-R, R + 1), repeat=3))
    for i in range(n):
        for j in range(n):
            B = np.zeros((3, D))
            B[:, 3 * i:3 * i + 3] += eye
            B[:, 3 * j:3 * j + 3] -= eye
            delta = r[i] - r[j]
            phase = kv @ delta
            value = np.dot(weights, np.cos(phase)) - np.pi / (alpha ** 2 * V)
            grad = -(weights * np.sin(phase)) @ kv
            hes = -np.einsum('k,ka,kb->ab', weights * np.cos(phase), kv, kv)
            if i == j:
                value -= 2 * alpha / np.sqrt(np.pi)
            for image in images:
                if i == j and image == (0, 0, 0):
                    continue
                d = delta + L * np.array(image)
                rho = np.linalg.norm(d)
                e = d / rho
                f = erfc(alpha * rho)
                fp = -2 * alpha / np.sqrt(np.pi) * np.exp(-(alpha * rho) ** 2)
                fpp = -2 * alpha ** 2 * rho * fp
                v = f / rho
                vp = fp / rho - f / rho ** 2
                vpp = fpp / rho - 2 * fp / rho ** 2 + 2 * f / rho ** 3
                value += v
                grad += vp * e
                hes += vpp * np.outer(e, e) + vp / rho * (eye - np.outer(e, e))
            K[i, j] = value
            G[i, j] = grad @ B
            H[i, j] = B.T @ hes @ B
    return (K, G, H)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Normal, boundary, edge and declared-invalid fixtures."""
    return [
        {
            'setup': (
                'import numpy as np\n'
                'r=np.array([[.2,.4,.7],[1.3,1.7,2.1],[2.7,.9,1.4]])\n'
                'r_g = r.copy()\n'
            ),
            'call': 'ewald_kernel_jets(r,[3.4,4.1,4.7],.8,1,[3,2,3])',
            'gold_call': '_oracle_ewald_kernel_jets(r_g,[3.4,4.1,4.7],.8,1,[3,2,3])',
        },
        {
            'setup': (
                'import numpy as np\n'
                'r=np.array([[.3,.4,.7]])\n'
                'r_g = r.copy()\n'
            ),
            'call': 'ewald_kernel_jets(r,[3.,4.,5.],.7,0,[0,0,0])',
            'gold_call': '_oracle_ewald_kernel_jets(r_g,[3.,4.,5.],.7,0,[0,0,0])',
        },
        {
            'setup': (
                'import numpy as np\n'
                'r=np.array([[.2,.4,.7],[1.3,1.7,2.1]])\n'
                'r_g = r.copy()\n'
            ),
            'call': 'ewald_kernel_jets(r,[3.4,4.1,4.7],.8,0,[0,0,2])',
            'gold_call': '_oracle_ewald_kernel_jets(r_g,[3.4,4.1,4.7],.8,0,[0,0,2])',
        },
        {
            'setup': (
                'import numpy as np\n'
                'r=np.array([[.2,.4,.7]])\n'
                'r_g = r.copy()\n'
            ),
            'call': 'ewald_kernel_jets(r,[3.,4.,5.],.7,2,[2,1,2])',
            'gold_call': '_oracle_ewald_kernel_jets(r_g,[3.,4.,5.],.7,2,[2,1,2])',
        },
        {
            'setup': (
                'import numpy as np\n'
                'r=np.array([[.2,.4,.7],[.2,.4,.7]])\n'
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
            'call': '_raise_code(lambda: ewald_kernel_jets(r,[3.4,4.1,4.7],.8,1,[3,2,3]))',
            'gold_call': '_raise_code(lambda: _oracle_ewald_kernel_jets(r_g,[3.4,4.1,4.7],.8,1,[3,2,3]))',
        },
    ]
