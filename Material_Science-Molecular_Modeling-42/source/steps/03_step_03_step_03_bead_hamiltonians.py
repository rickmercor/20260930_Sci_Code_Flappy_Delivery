"""
Build bead Hamiltonians and their first two displacement derivatives.

A nonlinear exchange law contributes directly to the force constants. Each bond derivative acts on two neighboring sites with opposite signs, preserving the translational sum rule.

Returns
-------
Return (h, dh, d2h), with shapes (p,d,d), (p,n,d,d), and (n,n,d,d).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bead_hamiltonian_jet(q: "np.ndarray", j0: float, g: float, alpha: float, bonds: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Return each bead Hamiltonian and its first two coordinate derivatives.
    
    Parameters
    ----------
    q : finite real array (p, n)
        Reduced displacements, p=1,...,8; n=4,6,8. Site and bead indices
        start at zero. Sites are periodic. These are displacements, so
        do not apply a minimum-image convention to site differences.
    j0 : positive float
    g, alpha : nonnegative floats
        Exchange parameters, with g in inverse length and alpha in
        inverse length squared. All exchanges in this input are positive.
    bonds : finite real array (n, d, d)
        bonds[k]=S_k dot S_(k+1 mod n) in the common Sz=0 basis.
    
    Returns
    -------
    h : float array (p, d, d)
    dh : float array (p, n, d, d)
    d2h : float array (n, n, d, d)
        Set v[k,i]=delta_(i,(k+1)%n)-delta_(i,k), x[t,k]=sum_i v[k,i]q[t,i],
        and J[t,k]=j0*(1-g*x[t,k]+alpha*x[t,k]**2).
        h[t]=sum_k J[t,k]*bonds[k].
        dh[t,i]=sum_k j0*(-g+2*alpha*x[t,k])*v[k,i]*bonds[k].
        d2h[i,j]=2*j0*alpha*sum_k v[k,i]*v[k,j]*bonds[k].
        The second derivative is bead independent. Any derivative
        involving coordinates on different beads is zero at this stage.
        Both coordinate indices of d2h are ordinary derivatives, with
        no factorial scaling. All inputs are valid and are not mutated.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_bead_hamiltonian_jet(q: "np.ndarray", j0: float, g: float, alpha: float, bonds: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Return each bead Hamiltonian and its first two coordinate derivatives.

    Parameters
    ----------
    q : finite real array (p, n)
        Reduced displacements, p=1,...,8; n=4,6,8. Site and bead indices
        start at zero. Sites are periodic. These are displacements, so
        do not apply a minimum-image convention to site differences.
    j0 : positive float
    g, alpha : nonnegative floats
        Exchange parameters, with g in inverse length and alpha in
        inverse length squared. All exchanges in this input are positive.
    bonds : finite real array (n, d, d)
        bonds[k]=S_k dot S_(k+1 mod n) in the common Sz=0 basis.

    Returns
    -------
    h : float array (p, d, d)
    dh : float array (p, n, d, d)
    d2h : float array (n, n, d, d)
        Set v[k,i]=delta_(i,(k+1)%n)-delta_(i,k), x[t,k]=sum_i v[k,i]q[t,i],
        and J[t,k]=j0*(1-g*x[t,k]+alpha*x[t,k]**2).
        h[t]=sum_k J[t,k]*bonds[k].
        dh[t,i]=sum_k j0*(-g+2*alpha*x[t,k])*v[k,i]*bonds[k].
        d2h[i,j]=2*j0*alpha*sum_k v[k,i]*v[k,j]*bonds[k].
        The second derivative is bead independent. Any derivative
        involving coordinates on different beads is zero at this stage.
        Both coordinate indices of d2h are ordinary derivatives, with
        no factorial scaling. All inputs are valid and are not mutated.
    """
    q = np.asarray(q, dtype=float)
    n = q.shape[1]
    incidence = np.roll(np.eye(n), 1, axis=1) - np.eye(n)
    x = q @ incidence.T
    exchange = j0 * (1.0 - g*x + alpha*x*x)
    h = np.einsum('tk,kab->tab', exchange, bonds)
    dh = np.einsum('tk,ki,kab->tiab', j0*(-g+2*alpha*x), incidence, bonds)
    d2h = 2*j0*alpha*np.einsum('ki,kj,kab->ijab', incidence, incidence, bonds)
    return h, dh, d2h

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'q=np.array([[0.36, -0.24, 0.12, -0.33, 0.21, -0.12], [-0.09, 0.3, -0.18, 0.27, -0.36, '
               '0.06], [0.24, -0.06, -0.3, 0.15, 0.09, -0.12], [-0.21, 0.12, 0.33, -0.09, -0.27, '
               '0.12]],dtype=float)\n'
               'bonds,s2=_oracle_spin_operators(q.shape[1])\n'
               'projectors=_oracle_spin_projectors(s2,q.shape[1])\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in bead_hamiltonian_jet(q, 1.0, .8, .3, bonds)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_bead_hamiltonian_jet(q, 1.0, .8, .3, bonds)])',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               'q=np.array([[0.36, -0.24, 0.12, -0.33, 0.21, -0.12], [-0.09, 0.3, -0.18, 0.27, -0.36, '
               '0.06], [0.24, -0.06, -0.3, 0.15, 0.09, -0.12], [-0.21, 0.12, 0.33, -0.09, -0.27, '
               '0.12]],dtype=float)\n'
               'bonds,s2=_oracle_spin_operators(q.shape[1])\n'
               'projectors=_oracle_spin_projectors(s2,q.shape[1])\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in bead_hamiltonian_jet(q, 1.0, .8, 0.0, bonds)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_bead_hamiltonian_jet(q, 1.0, .8, 0.0, bonds)])',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               'q=np.array([[0.36, -0.24, 0.12, -0.33, 0.21, -0.12], [-0.09, 0.3, -0.18, 0.27, -0.36, '
               '0.06], [0.24, -0.06, -0.3, 0.15, 0.09, -0.12], [-0.21, 0.12, 0.33, -0.09, -0.27, '
               '0.12]],dtype=float)\n'
               'q=q[:,:4]\n'
               'bonds,s2=_oracle_spin_operators(q.shape[1])\n'
               'projectors=_oracle_spin_projectors(s2,q.shape[1])\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in bead_hamiltonian_jet(q, .9, 0.0, .2, bonds)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_bead_hamiltonian_jet(q, .9, 0.0, .2, bonds)])',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               'q=np.array([[0.36, -0.24, 0.12, -0.33, 0.21, -0.12], [-0.09, 0.3, -0.18, 0.27, -0.36, '
               '0.06], [0.24, -0.06, -0.3, 0.15, 0.09, -0.12], [-0.21, 0.12, 0.33, -0.09, -0.27, '
               '0.12]],dtype=float)\n'
               'bonds,s2=_oracle_spin_operators(q.shape[1])\n'
               'projectors=_oracle_spin_projectors(s2,q.shape[1])\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in bead_hamiltonian_jet(q, 1.0, 0.0, 0.0, bonds)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_bead_hamiltonian_jet(q, 1.0, 0.0, 0.0, bonds)])',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               'q=np.array([[0.36, -0.24, 0.12, -0.33, 0.21, -0.12], [-0.09, 0.3, -0.18, 0.27, -0.36, '
               '0.06], [0.24, -0.06, -0.3, 0.15, 0.09, -0.12], [-0.21, 0.12, 0.33, -0.09, -0.27, '
               '0.12]],dtype=float)\n'
               'q=q+np.array([.2,-.3,.4,-.1])[:,None]\n'
               'bonds,s2=_oracle_spin_operators(q.shape[1])\n'
               'projectors=_oracle_spin_projectors(s2,q.shape[1])\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in bead_hamiltonian_jet(q, 1.0, .8, .3, bonds)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_bead_hamiltonian_jet(q, 1.0, .8, .3, bonds)])',
      'tol': 1e-11}]
