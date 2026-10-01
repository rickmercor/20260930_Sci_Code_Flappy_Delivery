"""
Assemble the whole-chain spin free energy, forces and Hessian.

Magnetic multiplicities determine the thermal weights of the spin sectors. Those weights change with geometry, so the full Hessian contains a sector-force covariance in addition to the averaged sector Hessians.

Returns
-------
Return (free_energy, forces, hessian, probabilities), with shapes scalar, (p,n), (p*n,p*n), and (n/2+1,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rped_force_constants(q: "np.ndarray", beta: float, j0: float, g: float, alpha: float, bonds: "np.ndarray", projectors: "np.ndarray") -> "tuple[float, np.ndarray, np.ndarray, np.ndarray]":
    """Return exact finite-bead spin free energy, forces and analytic Hessian.
    
    Parameters
    ----------
    q : finite real array (p, n), p=1,...,8; n=4,6,8
    beta, j0 : positive floats
    g, alpha : nonnegative floats
        Units have hbar=kB=1. J=j0*(1-g*Delta q+alpha*Delta q**2)>0.
        Every scaled sector trace satisfies sector_free_energy_jet's
        domain, and each projected b*h spectral width is at most 500.
    bonds : float array (n, d, d)
    projectors : float array (n/2+1, d, d)
        Consistent zero-magnetization operators and total-spin projectors
        from spin_operators and spin_projectors, ordered S=0,...,n/2.
    
    Returns
    -------
    free_energy : float
    forces : float array (p, n)
    hessian : float array (p*n, p*n)
    probabilities : float array (n/2+1,)
        Set b=beta/p. For each spin S obtain log Q_S, force f_S and
        Hessian K_S from sector_free_energy_jet. The complete partition
        factor Z=sum_S(2*S+1)*Q_S defines F_p=-log(Z)/b and weights
        w_S=(2*S+1)*Q_S/Z. Combine logarithms by log-sum-exp.
        The full force is f=sum_S w_S*f_S. In bead-major coordinates,
        K=sum_S w_S*K_S-b*(sum_S w_S*outer(f_S,f_S)-outer(f,f)).
        Return F_p,f,K,w. The covariance term comes from differentiating
        the thermal sector weights and must be retained.
        Compute the analytic whole-chain Hessian, including same-bead
        d2h and cross-bead derivative insertions. No finite differences,
        subsystem corrections or trace factorization are part of this
        calculation. NumPy only; inputs are not mutated.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_rped_force_constants(q: "np.ndarray", beta: float, j0: float, g: float, alpha: float, bonds: "np.ndarray", projectors: "np.ndarray") -> "tuple[float, np.ndarray, np.ndarray, np.ndarray]":
    """Return exact finite-bead spin free energy, forces and analytic Hessian.

    Parameters
    ----------
    q : finite real array (p, n), p=1,...,8; n=4,6,8
    beta, j0 : positive floats
    g, alpha : nonnegative floats
        Units have hbar=kB=1. J=j0*(1-g*Delta q+alpha*Delta q**2)>0.
        Every scaled sector trace satisfies sector_free_energy_jet's
        domain, and each projected b*h spectral width is at most 500.
    bonds : float array (n, d, d)
    projectors : float array (n/2+1, d, d)
        Consistent zero-magnetization operators and total-spin projectors
        from spin_operators and spin_projectors, ordered S=0,...,n/2.

    Returns
    -------
    free_energy : float
    forces : float array (p, n)
    hessian : float array (p*n, p*n)
    probabilities : float array (n/2+1,)
        Set b=beta/p. For each spin S obtain log Q_S, force f_S and
        Hessian K_S from sector_free_energy_jet. The complete partition
        factor Z=sum_S(2*S+1)*Q_S defines F_p=-log(Z)/b and weights
        w_S=(2*S+1)*Q_S/Z. Combine logarithms by log-sum-exp.
        The full force is f=sum_S w_S*f_S. In bead-major coordinates,
        K=sum_S w_S*K_S-b*(sum_S w_S*outer(f_S,f_S)-outer(f,f)).
        Return F_p,f,K,w. The covariance term comes from differentiating
        the thermal sector weights and must be retained.
        Compute the analytic whole-chain Hessian, including same-bead
        d2h and cross-bead derivative insertions. No finite differences,
        subsystem corrections or trace factorization are part of this
        calculation. NumPy only; inputs are not mutated.
    """
    q = np.asarray(q,dtype=float)
    b = beta/q.shape[0]
    h,dh,d2h = _oracle_bead_hamiltonian_jet(q,j0,g,alpha,bonds)
    logs,fs,ks = [],[],[]
    for spin,projector in enumerate(projectors):
        jets = [_oracle_sector_propagator_jet(h[t],dh[t],d2h,projector,b) for t in range(len(q))]
        e,de,d2e = [np.stack([a[k] for a in jets]) for k in range(3)]
        scales = np.array([a[3] for a in jets])
        logq,force,k = _oracle_sector_free_energy_jet(e,de,d2e,scales,b)
        logs.append(logq+np.log(2*spin+1))
        fs.append(force)
        ks.append(k)
    logs = np.array(logs)
    raw = np.exp(logs-logs.max())
    weights = raw/raw.sum()
    logz = float(logs.max()+np.log(raw.sum()))
    fs = np.stack(fs)
    flat = fs.reshape(len(fs),-1)
    force = np.einsum('s,sti->ti',weights,fs)
    mean = force.reshape(-1)
    covariance = np.einsum('s,sa,sb->ab',weights,flat,flat)-np.outer(mean,mean)
    hessian = np.einsum('s,sab->ab',weights,np.stack(ks))-b*covariance
    return float(-logz/b),force,hessian,weights

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
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in rped_force_constants(q, 3.7, 1.0, .8, .3, bonds, projectors)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_rped_force_constants(q, 3.7, 1.0, .8, .3, bonds, projectors)])',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               'q=np.array([[0.36, -0.24, 0.12, -0.33, 0.21, -0.12], [-0.09, 0.3, -0.18, 0.27, -0.36, '
               '0.06], [0.24, -0.06, -0.3, 0.15, 0.09, -0.12], [-0.21, 0.12, 0.33, -0.09, -0.27, '
               '0.12]],dtype=float)\n'
               'q=q[:,:4]\n'
               'bonds,s2=_oracle_spin_operators(q.shape[1])\n'
               'projectors=_oracle_spin_projectors(s2,q.shape[1])\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in rped_force_constants(q, 2.1, .9, .6, .2, bonds, projectors)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_rped_force_constants(q, 2.1, .9, .6, .2, bonds, projectors)])',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               'q=np.array([[0.36, -0.24, 0.12, -0.33, 0.21, -0.12], [-0.09, 0.3, -0.18, 0.27, -0.36, '
               '0.06], [0.24, -0.06, -0.3, 0.15, 0.09, -0.12], [-0.21, 0.12, 0.33, -0.09, -0.27, '
               '0.12]],dtype=float)\n'
               'q=q[:1]\n'
               'bonds,s2=_oracle_spin_operators(q.shape[1])\n'
               'projectors=_oracle_spin_projectors(s2,q.shape[1])\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in rped_force_constants(q, 3.7, 1.0, .8, .3, bonds, projectors)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_rped_force_constants(q, 3.7, 1.0, .8, .3, bonds, projectors)])',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               'q=np.array([[0.36, -0.24, 0.12, -0.33, 0.21, -0.12], [-0.09, 0.3, -0.18, 0.27, -0.36, '
               '0.06], [0.24, -0.06, -0.3, 0.15, 0.09, -0.12], [-0.21, 0.12, 0.33, -0.09, -0.27, '
               '0.12]],dtype=float)\n'
               'q=q[:2]\n'
               'bonds,s2=_oracle_spin_operators(q.shape[1])\n'
               'projectors=_oracle_spin_projectors(s2,q.shape[1])\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in rped_force_constants(q, 3.7, 1.0, .8, .3, bonds, projectors)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_rped_force_constants(q, 3.7, 1.0, .8, .3, bonds, projectors)])',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               'q=np.array([[0.36, -0.24, 0.12, -0.33, 0.21, -0.12], [-0.09, 0.3, -0.18, 0.27, -0.36, '
               '0.06], [0.24, -0.06, -0.3, 0.15, 0.09, -0.12], [-0.21, 0.12, 0.33, -0.09, -0.27, '
               '0.12]],dtype=float)\n'
               'bonds,s2=_oracle_spin_operators(q.shape[1])\n'
               'projectors=_oracle_spin_projectors(s2,q.shape[1])\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in rped_force_constants(q, 3.7, 1.0, .8, 0.0, bonds, projectors)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_rped_force_constants(q, 3.7, 1.0, .8, 0.0, bonds, projectors)])',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               'q=np.array([[0.36, -0.24, 0.12, -0.33, 0.21, -0.12], [-0.09, 0.3, -0.18, 0.27, -0.36, '
               '0.06], [0.24, -0.06, -0.3, 0.15, 0.09, -0.12], [-0.21, 0.12, 0.33, -0.09, -0.27, '
               '0.12]],dtype=float)\n'
               'bonds,s2=_oracle_spin_operators(q.shape[1])\n'
               'projectors=_oracle_spin_projectors(s2,q.shape[1])\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in rped_force_constants(q, 3.7, 1.0, 0.0, 0.0, bonds, projectors)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_rped_force_constants(q, 3.7, 1.0, 0.0, 0.0, bonds, projectors)])',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               'q=np.array([[0.36, -0.24, 0.12, -0.33, 0.21, -0.12], [-0.09, 0.3, -0.18, 0.27, -0.36, '
               '0.06], [0.24, -0.06, -0.3, 0.15, 0.09, -0.12], [-0.21, 0.12, 0.33, -0.09, -0.27, '
               '0.12]],dtype=float)\n'
               'q=np.zeros_like(q)\n'
               'bonds,s2=_oracle_spin_operators(q.shape[1])\n'
               'projectors=_oracle_spin_projectors(s2,q.shape[1])\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in rped_force_constants(q, 3.7, 1.0, 0.0, .3, bonds, projectors)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_rped_force_constants(q, 3.7, 1.0, 0.0, .3, bonds, projectors)])',
      'tol': 2e-08}]
