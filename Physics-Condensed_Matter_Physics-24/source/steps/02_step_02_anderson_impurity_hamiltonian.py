"""
Assemble the two-orbital Anderson impurity Hamiltonian in the many-body basis. The scalar interactions, bath arrays and system sizes determine a real symmetric matrix whose entries are many-body transition amplitudes and occupation energies in the Fock basis. Invalid input raises ValueError: the two size arguments must be positive integers, both bath arrays must have shape (n_orb, n_bath_per_orb) and the four scalar energies must be finite.

A crystal field splits the impurity, placing orbital zero at delta / 2 - mu and orbital one at -delta / 2 - mu. Each bath level carries its supplied energy and each orbital hybridizes only with its own bath levels, so the local Coulomb interaction is the sole channel through which the orbitals communicate. That interaction is the rotationally invariant Kanamori form: density terms use intra-orbital repulsion U, opposite-spin inter-orbital repulsion U' = U - 2J and parallel-spin inter-orbital repulsion U' - J, while directed spin-flip and pair-hopping sums over distinct orbitals complete the rotational invariance and are already Hermitian without an additional conjugate. Modes follow mode(orb, spin, site) = orb * 2 * (1 + n_bath_per_orb) + spin * (1 + n_bath_per_orb) + site, where site zero is the impurity, site k+1 is bath level k, spin zero is up and spin one is down.

Returns
-------
numpy.ndarray, the real symmetric Anderson impurity Hamiltonian
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def anderson_impurity_hamiltonian(U, J, delta, mu, eps_bath, V_bath, n_orb, n_bath_per_orb):
    """Build the Anderson impurity Hamiltonian in the many-body basis.

    Parameters
    ----------
    U : float
        Intra-orbital Coulomb repulsion.
    J : float
        Hund coupling in the Kanamori interaction.
    delta : float
        Crystal-field splitting between the two impurity orbitals.
    mu : float
        Chemical potential shifting both impurity orbital energies.
    eps_bath : array_like
        Bath energies with shape ``(n_orb, n_bath_per_orb)``.
    V_bath : array_like
        Orbital-preserving impurity-bath hybridizations with shape
        ``(n_orb, n_bath_per_orb)``.
    n_orb : int
        Number of impurity orbitals.
    n_bath_per_orb : int
        Number of bath levels attached to each orbital.

    Returns
    -------
    H : numpy.ndarray
        Real symmetric many-body Hamiltonian with shape
        ``(2**n_modes, 2**n_modes)``, where
        ``n_modes = n_orb * 2 * (1 + n_bath_per_orb)``.
    
    Raises
    ------
    ValueError
        If ``n_orb`` or ``n_bath_per_orb`` is not a positive integer, if the
        bath arrays do not both have shape ``(n_orb, n_bath_per_orb)``, or if
        ``U``, ``J``, ``delta`` or ``mu`` is not a finite scalar.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_anderson_impurity_hamiltonian(U, J, delta, mu, eps_bath, V_bath, n_orb, n_bath_per_orb):
    if isinstance(n_orb, (bool, np.bool_)) or not isinstance(n_orb, (int, np.integer)) or n_orb < 1:
        raise ValueError("n_orb must be a positive integer")
    if isinstance(n_bath_per_orb, (bool, np.bool_)) or not isinstance(n_bath_per_orb, (int, np.integer)) or n_bath_per_orb < 1:
        raise ValueError("n_bath_per_orb must be a positive integer")
    eps_bath = np.asarray(eps_bath)
    V_bath = np.asarray(V_bath)
    expected_shape = (n_orb, n_bath_per_orb)
    if eps_bath.shape != expected_shape or V_bath.shape != expected_shape:
        raise ValueError("bath arrays have incompatible shapes")
    try:
        finite_scalars = all(np.asarray(value).ndim == 0 and bool(np.isfinite(value)) for value in (U, J, delta, mu))
    except (TypeError, ValueError):
        finite_scalars = False
    if not finite_scalars:
        raise ValueError("U, J, delta, and mu must be finite scalars")
    n_spin = 2
    n_sites = 1 + n_bath_per_orb
    n_modes = n_orb * n_spin * n_sites
    c_dag, c = _oracle_fock_space_operators(n_modes)
    dim = 2 ** n_modes
    H = np.zeros((dim, dim))

    def mode_idx(orb, spin, site):
        return orb * n_spin * n_sites + spin * n_sites + site

    def n_op(orb, spin, site):
        idx = mode_idx(orb, spin, site)
        return c_dag[idx] @ c[idx]

    for orb in range(n_orb):
        eps_orb = (-1)**orb * delta / 2.0 - mu
        for spin in range(n_spin):
            H += eps_orb * n_op(orb, spin, 0)
    for orb in range(n_orb):
        for k in range(n_bath_per_orb):
            for spin in range(n_spin):
                H += eps_bath[orb, k] * n_op(orb, spin, k + 1)
    for orb in range(n_orb):
        for k in range(n_bath_per_orb):
            for spin in range(n_spin):
                imp = mode_idx(orb, spin, 0)
                bath = mode_idx(orb, spin, k + 1)
                H += V_bath[orb, k] * (c_dag[imp] @ c[bath] + c_dag[bath] @ c[imp])
    Up = U - 2.0 * J
    for orb in range(n_orb):
        H += U * n_op(orb, 0, 0) @ n_op(orb, 1, 0)
    for a in range(n_orb):
        for b in range(a + 1, n_orb):
            for s1 in range(n_spin):
                for s2 in range(n_spin):
                    if s1 == s2:
                        H += (Up - J) * n_op(a, s1, 0) @ n_op(b, s2, 0)
                    else:
                        H += Up * n_op(a, s1, 0) @ n_op(b, s2, 0)
    for a in range(n_orb):
        for b in range(n_orb):
            if a != b:
                ia_up = mode_idx(a, 0, 0)
                ia_dn = mode_idx(a, 1, 0)
                ib_up = mode_idx(b, 0, 0)
                ib_dn = mode_idx(b, 1, 0)
                H += -J * (c_dag[ia_up] @ c[ia_dn] @ c_dag[ib_dn] @ c[ib_up])
                H += J * (c_dag[ia_up] @ c_dag[ia_dn] @ c[ib_dn] @ c[ib_up])
    return H

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'eps = np.zeros((2, 1), dtype=float)\n'
               'V = np.array([[0.60], [0.35]], dtype=float)\n'
               'idx = lambda o, s, p: o * 4 + s * 2 + p\n'
               'bits = lambda *ms: sum(1 << i for i in ms)\n'
               'probe = [\n'
               '    (bits(idx(0, 0, 0)), bits(idx(0, 0, 0))),\n'
               '    (bits(idx(1, 0, 0)), bits(idx(1, 0, 0))),\n'
               '    (bits(idx(0, 0, 0), idx(0, 1, 0)), bits(idx(0, 0, 0), idx(0, 1, 0))),\n'
               '    (bits(idx(0, 0, 0), idx(1, 1, 0)), bits(idx(0, 0, 0), idx(1, 1, 0))),\n'
               '    (bits(idx(0, 0, 0), idx(1, 0, 0)), bits(idx(0, 0, 0), idx(1, 0, 0))),\n'
               '    (bits(idx(0, 0, 0)), bits(idx(0, 0, 1))),\n'
               '    (bits(idx(1, 0, 0)), bits(idx(1, 0, 1))),\n'
               '    (bits(idx(0, 0, 0), idx(1, 1, 0)), bits(idx(0, 1, 0), idx(1, 0, 0))),\n'
               '    (bits(idx(0, 0, 0), idx(0, 1, 0)), bits(idx(1, 0, 0), idx(1, 1, 0))),\n'
               ']\n',
      'call': '[round(float(anderson_impurity_hamiltonian(2.5, 0.5, 1.2, 2.5, eps, V, 2, 1)[r, c]), 10) for r, c '
              'in probe]',
      'gold_call': '[round(float(_oracle_anderson_impurity_hamiltonian(2.5, 0.5, 1.2, 2.5, eps, V, 2, 1)[r, c]), '
                   '10) for r, c in probe]'},
     {'setup': 'import numpy as np\n'
               'eps = np.zeros((2, 1), dtype=float)\n'
               'V = np.array([[0.60], [0.35]], dtype=float)\n'
               'idx = lambda o, s, p: o * 4 + s * 2 + p\n'
               'bits = lambda *ms: sum(1 << i for i in ms)\n'
               'probe = [\n'
               '    (bits(idx(0, 0, 0)), bits(idx(0, 0, 0))),\n'
               '    (bits(idx(1, 0, 0)), bits(idx(1, 0, 0))),\n'
               '    (bits(idx(0, 0, 0), idx(0, 1, 0)), bits(idx(0, 0, 0), idx(0, 1, 0))),\n'
               '    (bits(idx(0, 0, 0), idx(1, 1, 0)), bits(idx(0, 0, 0), idx(1, 1, 0))),\n'
               '    (bits(idx(0, 0, 0), idx(1, 0, 0)), bits(idx(0, 0, 0), idx(1, 0, 0))),\n'
               '    (bits(idx(0, 0, 0)), bits(idx(0, 0, 1))),\n'
               '    (bits(idx(1, 0, 0)), bits(idx(1, 0, 1))),\n'
               '    (bits(idx(0, 0, 0), idx(1, 1, 0)), bits(idx(0, 1, 0), idx(1, 0, 0))),\n'
               '    (bits(idx(0, 0, 0), idx(0, 1, 0)), bits(idx(1, 0, 0), idx(1, 1, 0))),\n'
               ']\n',
      'call': '[round(float(anderson_impurity_hamiltonian(0.0, 0.0, 0.0, 0.0, eps, V, 2, 1)[r, c]), 10) for r, c '
              'in probe]',
      'gold_call': '[round(float(_oracle_anderson_impurity_hamiltonian(0.0, 0.0, 0.0, 0.0, eps, V, 2, 1)[r, c]), '
                   '10) for r, c in probe]'},
     {'setup': 'import numpy as np\n'
               'eps = np.zeros((2, 1), dtype=float)\n'
               'V = np.array([[0.60], [0.35]], dtype=float)\n'
               'idx = lambda o, s, p: o * 4 + s * 2 + p\n'
               'bits = lambda *ms: sum(1 << i for i in ms)\n'
               'probe = [\n'
               '    (bits(idx(0, 0, 0)), bits(idx(0, 0, 0))),\n'
               '    (bits(idx(1, 0, 0)), bits(idx(1, 0, 0))),\n'
               '    (bits(idx(0, 0, 0), idx(0, 1, 0)), bits(idx(0, 0, 0), idx(0, 1, 0))),\n'
               '    (bits(idx(0, 0, 0), idx(1, 1, 0)), bits(idx(0, 0, 0), idx(1, 1, 0))),\n'
               '    (bits(idx(0, 0, 0), idx(1, 0, 0)), bits(idx(0, 0, 0), idx(1, 0, 0))),\n'
               '    (bits(idx(0, 0, 0)), bits(idx(0, 0, 1))),\n'
               '    (bits(idx(1, 0, 0)), bits(idx(1, 0, 1))),\n'
               '    (bits(idx(0, 0, 0), idx(1, 1, 0)), bits(idx(0, 1, 0), idx(1, 0, 0))),\n'
               '    (bits(idx(0, 0, 0), idx(0, 1, 0)), bits(idx(1, 0, 0), idx(1, 1, 0))),\n'
               ']\n',
      'call': '[round(float(anderson_impurity_hamiltonian(0.0, 0.4, 0.0, 0.0, eps, V, 2, 1)[r, c]), 10) for r, c '
              'in probe]',
      'gold_call': '[round(float(_oracle_anderson_impurity_hamiltonian(0.0, 0.4, 0.0, 0.0, eps, V, 2, 1)[r, c]), '
                   '10) for r, c in probe]'},
     {'setup': 'import numpy as np\n'
               'eps = np.zeros((2, 1), dtype=float)\n'
               'V = np.array([[0.60], [0.35]], dtype=float)\n',
      'call': '(lambda H: float(np.max(np.abs(H - H.T))))(anderson_impurity_hamiltonian(2.5, 0.5, 1.2, 2.5, eps, '
              'V, 2, 1))',
      'gold_call': '(lambda H: float(np.max(np.abs(H - H.T))))(_oracle_anderson_impurity_hamiltonian(2.5, 0.5, '
                   '1.2, 2.5, eps, V, 2, 1))'},
     {'setup': '\n'
               '\n'
               'import numpy as np\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(anderson_impurity_hamiltonian, 1.0, 0.1, np.nan, 0.0, np.zeros((2, 1)), '
              'np.zeros((2, 1)), 2, 1)',
      'gold_call': '_exception_code(_oracle_anderson_impurity_hamiltonian, 1.0, 0.1, np.nan, 0.0, np.zeros((2, '
                   '1)), np.zeros((2, 1)), 2, 1)'}]
