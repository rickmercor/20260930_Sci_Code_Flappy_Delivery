"""
Compute electrostatic energy, forces and the Cartesian Hessian from the charge and Coulomb-kernel derivatives.

The prescribed charges depend explicitly on geometry. Differentiating the quadratic electrostatic energy therefore requires charge gradients, charge Hessians, kernel derivatives and mixed charge–kernel terms. The Ewald self energy contributes through charge derivatives even though its kernel coefficient is coordinate independent.

Returns
-------
return energy, force, hessian
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def electrostatic_jet(charges, charge_first, charge_second, kernel, kernel_first, kernel_second):
    """Differentiate E(x)=0.5*q(x)^T*K(x)*q(x) through second order.
    
    Parameters
    ----------
    charges : (N,) real array
    charge_first : (N,D) real array
    charge_second : (N,D,D) real array
    kernel : (N,N) real array
    kernel_first : (N,N,D) real array
    kernel_second : (N,N,D,D) real array
        N,D>=1. Kernel arrays are symmetric in their first two indices.
        All second derivatives are unnormalized. Include all derivatives of
        q as well as K. Charge conservation does not make charge derivatives
        vanish. No Hellmann-Feynman stationarity assumption is permitted:
        these charges are a prescribed geometry map, not an energy minimizer.
    
    Returns
    -------
    energy : float
    force : (D,) float array
        Minus the energy gradient.
    hessian : (D,D) float array
        Full energy Hessian, including the term containing charge_second.
    
    Raises
    ------
    ValueError
        If shapes do not match or any entry is nonfinite. Symmetry is a
        precondition and is not separately validated.

    Runtime
    -------
    Import required libraries inside the submitted function and inside any
    helper that uses them. Do not assume that np, itertools or erfc exists
    in the function's global namespace. Imports in test setup are separate.
    """
    return (energy, force, hessian)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_electrostatic_jet(charges, charge_first, charge_second, kernel, kernel_first, kernel_second):
    """Reference implementation."""
    import numpy as np
    q = np.asarray(charges, float)
    J = np.asarray(charge_first, float)
    Q = np.asarray(charge_second, float)
    K = np.asarray(kernel, float)
    G = np.asarray(kernel_first, float)
    H = np.asarray(kernel_second, float)
    if q.ndim != 1 or len(q) < 1 or J.ndim != 2 or (J.shape[0] != len(q)) or (J.shape[1] < 1):
        raise ValueError('charge shape')
    n, D = J.shape
    if Q.shape != (n, D, D) or K.shape != (n, n) or G.shape != (n, n, D) or (H.shape != (n, n, D, D)):
        raise ValueError('derivative shape')
    if not all((np.all(np.isfinite(a)) for a in (q, J, Q, K, G, H))):
        raise ValueError('nonfinite')
    Kq = K @ q
    energy = 0.5 * q @ Kq
    grad = J.T @ Kq + 0.5 * np.einsum('i,ija,j->a', q, G, q)
    mixed = np.einsum('ia,ijb,j->ab', J, G, q)
    hess = np.einsum('iab,i->ab', Q, Kq) + J.T @ K @ J + mixed + mixed.T + 0.5 * np.einsum('i,ijab,j->ab', q, H, q)
    return (float(energy), -grad, hess)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Normal, boundary, edge and declared-invalid fixtures."""
    return [
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(785)\n'
                'n=3;d=4\n'
                'q=rng.normal(size=n);j=rng.normal(size=(n,d));q2=rng.normal(size=(n,d,d));q2=(q2+q2.swapaxes(-1,-2))/2\n'
                'k=rng.normal(size=(n,n));k=(k+k.T)/2\n'
                'g=rng.normal(size=(n,n,d));g=(g+g.swapaxes(0,1))/2\n'
                'h=rng.normal(size=(n,n,d,d));h=(h+h.swapaxes(0,1))/2;h=(h+h.swapaxes(-1,-2))/2\n'
                'q_g = q.copy(); j_g = j.copy(); q2_g = q2.copy(); k_g = k.copy(); g_g = g.copy(); h_g = h.copy()\n'
            ),
            'call': 'electrostatic_jet(q,j,q2,k,g,h)',
            'gold_call': '_oracle_electrostatic_jet(q_g,j_g,q2_g,k_g,g_g,h_g)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(785)\n'
                'n=3;d=4\n'
                'q=rng.normal(size=n);j=rng.normal(size=(n,d));q2=rng.normal(size=(n,d,d));q2=(q2+q2.swapaxes(-1,-2))/2\n'
                'k=rng.normal(size=(n,n));k=(k+k.T)/2\n'
                'g=rng.normal(size=(n,n,d));g=(g+g.swapaxes(0,1))/2\n'
                'h=rng.normal(size=(n,n,d,d));h=(h+h.swapaxes(0,1))/2;h=(h+h.swapaxes(-1,-2))/2\n'
                'j[:]=0;q2[:]=0\n'
                'q_g = q.copy(); j_g = j.copy(); q2_g = q2.copy(); k_g = k.copy(); g_g = g.copy(); h_g = h.copy()\n'
            ),
            'call': 'electrostatic_jet(q,j,q2,k,g,h)',
            'gold_call': '_oracle_electrostatic_jet(q_g,j_g,q2_g,k_g,g_g,h_g)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(785)\n'
                'n=3;d=4\n'
                'q=rng.normal(size=n);j=rng.normal(size=(n,d));q2=rng.normal(size=(n,d,d));q2=(q2+q2.swapaxes(-1,-2))/2\n'
                'k=rng.normal(size=(n,n));k=(k+k.T)/2\n'
                'g=rng.normal(size=(n,n,d));g=(g+g.swapaxes(0,1))/2\n'
                'h=rng.normal(size=(n,n,d,d));h=(h+h.swapaxes(0,1))/2;h=(h+h.swapaxes(-1,-2))/2\n'
                'g[:]=0;h[:]=0\n'
                'q_g = q.copy(); j_g = j.copy(); q2_g = q2.copy(); k_g = k.copy(); g_g = g.copy(); h_g = h.copy()\n'
            ),
            'call': 'electrostatic_jet(q,j,q2,k,g,h)',
            'gold_call': '_oracle_electrostatic_jet(q_g,j_g,q2_g,k_g,g_g,h_g)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(785)\n'
                'n=3;d=4\n'
                'q=rng.normal(size=n);j=rng.normal(size=(n,d));q2=rng.normal(size=(n,d,d));q2=(q2+q2.swapaxes(-1,-2))/2\n'
                'k=rng.normal(size=(n,n));k=(k+k.T)/2\n'
                'g=rng.normal(size=(n,n,d));g=(g+g.swapaxes(0,1))/2\n'
                'h=rng.normal(size=(n,n,d,d));h=(h+h.swapaxes(0,1))/2;h=(h+h.swapaxes(-1,-2))/2\n'
                'q[:]=0\n'
                'q_g = q.copy(); j_g = j.copy(); q2_g = q2.copy(); k_g = k.copy(); g_g = g.copy(); h_g = h.copy()\n'
            ),
            'call': 'electrostatic_jet(q,j,q2,k,g,h)',
            'gold_call': '_oracle_electrostatic_jet(q_g,j_g,q2_g,k_g,g_g,h_g)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(785)\n'
                'n=3;d=4\n'
                'q=rng.normal(size=n);j=rng.normal(size=(n,d));q2=rng.normal(size=(n,d,d));q2=(q2+q2.swapaxes(-1,-2))/2\n'
                'k=rng.normal(size=(n,n));k=(k+k.T)/2\n'
                'g=rng.normal(size=(n,n,d));g=(g+g.swapaxes(0,1))/2\n'
                'h=rng.normal(size=(n,n,d,d));h=(h+h.swapaxes(0,1))/2;h=(h+h.swapaxes(-1,-2))/2\n'
                'k[0,0]=np.nan\n'
                'def _raise_code(f):\n'
                '    try:\n'
                '        f()\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        raise\n'
                '    raise AssertionError("Expected ValueError")\n'
                'q_g = q.copy(); j_g = j.copy(); q2_g = q2.copy(); k_g = k.copy(); g_g = g.copy(); h_g = h.copy()\n'
            ),
            'call': '_raise_code(lambda: electrostatic_jet(q,j,q2,k,g,h))',
            'gold_call': '_raise_code(lambda: _oracle_electrostatic_jet(q_g,j_g,q2_g,k_g,g_g,h_g))',
        },
    ]
