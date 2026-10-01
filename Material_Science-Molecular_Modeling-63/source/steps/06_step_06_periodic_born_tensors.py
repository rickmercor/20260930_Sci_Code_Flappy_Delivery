"""
Compute scaled Born tensors from the phase-compensated derivative of the complex periodic dipole.

The periodic dipole derivative contains the direct displacement contribution and the response of every environment-dependent charge. Apply the compensating phase of the displaced atom before taking the real part. Tensor indices are atom, polarization and displacement; the last two indices are not generally interchangeable.

Returns
-------
return z0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def periodic_born_tensors(positions, box, charges, charge_first):
    """Compute scaled Born tensors using a phase-compensated periodic dipole.
    
    Parameters
    ----------
    positions : (N,3) real array, N>=1
    box : (3,) positive real array
    charges : (N,) real array
    charge_first : (N,3*N) real array
        J[j,3*l+b]=dq_j/dr_l,b, including the redistribution derivative.
    
    Define P0_a=L_a/(2*pi*i)*sum_j q_j*exp(2*pi*i*r_j,a/L_a).
    Define Z0[l,a,b]=Re(exp(-2*pi*i*r_l,a/L_a)*dP0_a/dr_l,b).
    The axes of Z0 are atom, polarization component, displacement component.
    The phase factor depends on the displaced atom l. There is no volume
    factor in P0, no epsilon_infinity factor in Z0, and no acoustic correction
    at this step. Coordinates can lie outside the primary cell.
    
    Returns
    -------
    z0 : (N,3,3) float array
        Scaled Born tensors. Include the derivative of the phase as well as
        the derivative of every charge. Do not approximate exp by 1+i*phase.
    
    Raises
    ------
    ValueError
        If shapes do not match, inputs are nonfinite, or box is nonpositive.

    Runtime
    -------
    Import required libraries inside the submitted function and inside any
    helper that uses them. Do not assume that np, itertools or erfc exists
    in the function's global namespace. Imports in test setup are separate.
    """
    return z0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_periodic_born_tensors(positions, box, charges, charge_first):
    """Reference implementation."""
    import numpy as np
    r = np.asarray(positions, float)
    L = np.asarray(box, float)
    q = np.asarray(charges, float)
    J = np.asarray(charge_first, float)
    if r.ndim != 2 or r.shape[1:] != (3,) or len(r) < 1 or (L.shape != (3,)) or (q.shape != (len(r),)) or (J.shape != (len(r), 3 * len(r))):
        raise ValueError('shape')
    if not all((np.all(np.isfinite(a)) for a in (r, L, q, J))) or np.any(L <= 0):
        raise ValueError('domain')
    n = len(r)
    Z = np.zeros((n, 3, 3))
    phase = 2 * np.pi * r / L
    for l in range(n):
        for a in range(3):
            Z[l, a] = L[a] / (2 * np.pi) * np.sin(phase[:, a] - phase[l, a]) @ J[:, 3 * l:3 * l + 3]
            Z[l, a, a] += q[l]
    return Z

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Normal, boundary, edge and declared-invalid fixtures."""
    return [
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(314)\n'
                'r=np.array([[.2,.7,1.1],[1.9,.6,2.2],[2.8,2.4,.3]])\n'
                'q=np.array([.8,-.5,-.3]);j=rng.normal(size=(3,9));j-=j.mean(axis=0)\n'
                'L=np.array([3.4,3.8,4.3])\n'
                'r_g = r.copy(); L_g = L.copy(); q_g = q.copy(); j_g = j.copy()\n'
            ),
            'call': 'periodic_born_tensors(r,L,q,j)',
            'gold_call': '_oracle_periodic_born_tensors(r_g,L_g,q_g,j_g)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(314)\n'
                'r=np.array([[.2,.7,1.1],[1.9,.6,2.2],[2.8,2.4,.3]])\n'
                'q=np.array([.8,-.5,-.3]);j=rng.normal(size=(3,9));j-=j.mean(axis=0)\n'
                'L=np.array([3.4,3.8,4.3])\n'
                'j[:]=0\n'
                'r_g = r.copy(); L_g = L.copy(); q_g = q.copy(); j_g = j.copy()\n'
            ),
            'call': 'periodic_born_tensors(r,L,q,j)',
            'gold_call': '_oracle_periodic_born_tensors(r_g,L_g,q_g,j_g)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(314)\n'
                'r=np.array([[.2,.7,1.1],[1.9,.6,2.2],[2.8,2.4,.3]])\n'
                'q=np.array([.8,-.5,-.3]);j=rng.normal(size=(3,9));j-=j.mean(axis=0)\n'
                'L=np.array([3.4,3.8,4.3])\n'
                'q[:]=0;r+=np.array([[3.4,0.,0.],[0.,-3.8,0.],[0.,0.,8.6]])\n'
                'r_g = r.copy(); L_g = L.copy(); q_g = q.copy(); j_g = j.copy()\n'
            ),
            'call': 'periodic_born_tensors(r,L,q,j)',
            'gold_call': '_oracle_periodic_born_tensors(r_g,L_g,q_g,j_g)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(314)\n'
                'r=np.array([[.2,.7,1.1],[1.9,.6,2.2],[2.8,2.4,.3]])\n'
                'q=np.array([.8,-.5,-.3]);j=rng.normal(size=(3,9));j-=j.mean(axis=0)\n'
                'L=np.array([3.4,3.8,4.3])\n'
                'L[0]=0\n'
                'def _raise_code(f):\n'
                '    try:\n'
                '        f()\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        raise\n'
                '    raise AssertionError("Expected ValueError")\n'
                'r_g = r.copy(); L_g = L.copy(); q_g = q.copy(); j_g = j.copy()\n'
            ),
            'call': '_raise_code(lambda: periodic_born_tensors(r,L,q,j))',
            'gold_call': '_raise_code(lambda: _oracle_periodic_born_tensors(r_g,L_g,q_g,j_g))',
        },
    ]
