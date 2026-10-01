"""
Resolve the least-curved internal ring-polymer displacement.

The bead springs penalize imaginary-time variations. Constraining each bead to zero net site displacement removes rigid translations and retains internal distortions. An Einstein term shifts every retained curvature by the same stiffness.

Returns
-------
Return (k_star, eigenvalues), a scalar and an ascending array of length p*(n-1).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def internal_curvature(hessian: "np.ndarray", p: int, n: int, beta: float, mass: float) -> "tuple[float, np.ndarray]":
    """Find the Einstein stiffness needed for nonnegative internal curvature.
    
    Parameters
    ----------
    hessian : finite real symmetric array (p*n, p*n)
        Spin free-energy Hessian in bead-major order; symmetry holds to
        absolute error 1e-9. p=1,...,8 and n=4,6,8.
    p, n : int
    beta, mass : positive floats
        Equal reduced mass at every site; hbar=1.
    
    Returns
    -------
    k_star : float
    eigenvalues : float array (p*(n-1),)
        Add the bead spring Hessian gamma*kron(L_p,I_n), where
        gamma=mass*(p/beta)**2 and
        L_p=sum_(t=0..p-1)(e_t-e_((t+1)%p))(e_t-e_((t+1)%p)).T.
        Thus L_1=0; for p=2 both cyclic edges are counted, giving
        L_2=[[2,-2],[-2,2]]. This is the Hessian of
        mass/(2*(beta/p)**2)*sum_(t,i)(q[t,i]-q[(t+1)%p,i])**2.
        Restrict to displacements whose site sum is zero on every bead.
        One valid orthonormal basis B of shape (n,n-1) has, in column j,
        entries 1/sqrt((j+1)*(j+2)) in rows 0,...,j, entry
        -(j+1)/sqrt((j+1)*(j+2)) in row j+1, and zero otherwise.
        With C=kron(I_p,B), return the ascending eigenvalues of
        C.T @ (hessian+gamma*kron(L_p,I_n)) @ C and
        k_star=max(0,-eigenvalues[0]). Average hessian with its transpose
        before diagonalizing to remove roundoff asymmetry only.
        Adding Einstein energy k/2*sum q**2 shifts these eigenvalues by k.
        The result concerns local curvature at fixed q, not an equilibrium
        transition or a phonon frequency at a stationary structure.
        Inputs are not mutated.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_internal_curvature(hessian: "np.ndarray", p: int, n: int, beta: float, mass: float) -> "tuple[float, np.ndarray]":
    """Find the Einstein stiffness needed for nonnegative internal curvature.

    Parameters
    ----------
    hessian : finite real symmetric array (p*n, p*n)
        Spin free-energy Hessian in bead-major order; symmetry holds to
        absolute error 1e-9. p=1,...,8 and n=4,6,8.
    p, n : int
    beta, mass : positive floats
        Equal reduced mass at every site; hbar=1.

    Returns
    -------
    k_star : float
    eigenvalues : float array (p*(n-1),)
        Add the bead spring Hessian gamma*kron(L_p,I_n), where
        gamma=mass*(p/beta)**2 and
        L_p=sum_(t=0..p-1)(e_t-e_((t+1)%p))(e_t-e_((t+1)%p)).T.
        Thus L_1=0; for p=2 both cyclic edges are counted, giving
        L_2=[[2,-2],[-2,2]]. This is the Hessian of
        mass/(2*(beta/p)**2)*sum_(t,i)(q[t,i]-q[(t+1)%p,i])**2.
        Restrict to displacements whose site sum is zero on every bead.
        One valid orthonormal basis B of shape (n,n-1) has, in column j,
        entries 1/sqrt((j+1)*(j+2)) in rows 0,...,j, entry
        -(j+1)/sqrt((j+1)*(j+2)) in row j+1, and zero otherwise.
        With C=kron(I_p,B), return the ascending eigenvalues of
        C.T @ (hessian+gamma*kron(L_p,I_n)) @ C and
        k_star=max(0,-eigenvalues[0]). Average hessian with its transpose
        before diagonalizing to remove roundoff asymmetry only.
        Adding Einstein energy k/2*sum q**2 shifts these eigenvalues by k.
        The result concerns local curvature at fixed q, not an equilibrium
        transition or a phonon frequency at a stationary structure.
        Inputs are not mutated.
    """
    laplacian = np.zeros((p,p))
    for t in range(p):
        edge = np.zeros(p)
        edge[t] += 1
        edge[(t+1)%p] -= 1
        laplacian += np.outer(edge,edge)
    basis = np.zeros((n,n-1))
    for j in range(n-1):
        norm = np.sqrt((j+1)*(j+2))
        basis[:j+1,j] = 1/norm
        basis[j+1,j] = -(j+1)/norm
    c = np.kron(np.eye(p),basis)
    total = .5*(hessian+hessian.T)+mass*(p/beta)**2*np.kron(laplacian,np.eye(n))
    values = np.linalg.eigvalsh(c.T@total@c)
    return float(max(0.,-values[0])),values

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nk=-2*np.eye(24)\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in internal_curvature(k, 4, 6, 3.7, .2)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_internal_curvature(k, 4, 6, 3.7, .2)])',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nk=np.zeros((6,6))\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in internal_curvature(k, 1, 6, 3.7, .2)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_internal_curvature(k, 1, 6, 3.7, .2)])',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nk=np.diag(np.linspace(-3.,2.,8))\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in internal_curvature(k, 2, 4, 2.1, .3)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_internal_curvature(k, 2, 4, 2.1, .3)])',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nk=np.eye(12)\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in internal_curvature(k, 3, 4, 2.1, .3)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_internal_curvature(k, 3, 4, 2.1, .3)])',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'q=np.array([[0.36, -0.24, 0.12, -0.33, 0.21, -0.12], [-0.09, 0.3, -0.18, 0.27, -0.36, '
               '0.06], [0.24, -0.06, -0.3, 0.15, 0.09, -0.12], [-0.21, 0.12, 0.33, -0.09, -0.27, '
               '0.12]],dtype=float)\n'
               'bonds,s2=_oracle_spin_operators(q.shape[1])\n'
               'projectors=_oracle_spin_projectors(s2,q.shape[1])\n'
               '_,_,k,_=_oracle_rped_force_constants(q,3.7,1.,.8,.3,bonds,projectors)\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in internal_curvature(k, 4, 6, 3.7, .2)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_internal_curvature(k, 4, 6, 3.7, .2)])',
      'tol': 1e-08}]
