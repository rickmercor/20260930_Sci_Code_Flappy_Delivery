"""
Assemble the 1+1D T1 Kogut-Susskind qubit Hamiltonian.

Truncation T1 of the same paper is the qubit theory whose local link states are the singlet and the fundamental, in that order. The global space is the Kronecker product with site 0 the most significant factor, so a computational-basis index is sum_j q_j * 2**(n_links-1-j) with each local label q_j in {0,1}. Site indices are identified periodically. The operator is the T1 Hamiltonian of the paper's 1+1D truncation section: the one-body electric and mass pieces written there together with the three-body hopping sandwich of that truncation. Local operators follow the same projector / Pauli-X / Pauli-Z dictionary restricted to those two irreps. coupling is the Yang-Mills g and mass is the staggered mass m. Use the Z^N form of Eqs. (26)-(27), retaining the identity piece of Z^N = I - 2P^N. A matrix shifted by a multiple of the identity is a different code return. n_links is either 4 or 6; n_colors is an integer from 3 through 64. Section II places staggered fermions on a bipartite lattice, so a periodic chain has even length. The color range matches the T3 comparison, with N_c at least 3. This task evaluates the even lengths 4 and 6. Finite negative mass is allowed. Paper provenance: Section IV.A.1 defines T1; Eq. (19) fixes the local operator dictionary, Eq. (24) fixes the retained irreps, Eq. (26) and Eq. (27) give the one-body terms, and Eq. (28) with Table III gives the hopping channel. Include every import the implementation needs inside the function body. n_links and n_colors must have Python int or NumPy integer type; integer-valued floats such as 3.0 are outside the count domain. coupling and mass accept Python or NumPy integer/floating scalars. Boolean and temporal values are excluded from all scalar inputs. Calculations and numeric returns use float64.

Returns
-------
hamiltonian : np.ndarray, shape (2**n_links, 2**n_links), float Dense real symmetric T1 Hamiltonian in the computational basis above.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_t1_hamiltonian(n_links: int, n_colors: int, coupling: float,
                            mass: float) -> "np.ndarray":
    """Return the dense T1 qubit Hamiltonian on a periodic chain.

    Parameters
    ----------
    n_links : int
        Number of links on the periodic chain. Either 4 or 6 (an even periodic chain).
    n_colors : int
        Number of colors N_c. Integer from 3 through 64.
    coupling : float
        Finite Yang-Mills coupling g.
    mass : float
        Finite staggered fermion mass m.

    Returns
    -------
    hamiltonian : np.ndarray, shape (2**n_links, 2**n_links), float
        Dense real symmetric T1 Hamiltonian in the computational basis above.

    Raises
    ------
    ValueError
        If either count is not an integer in its stated range, coupling or mass
        is not a finite real scalar, or the parameters produce non-finite entries.
    """
    return hamiltonian

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_assemble_t1_hamiltonian(n_links: int, n_colors: int, coupling: float,
                                    mass: float) -> "np.ndarray":
    import numpy as np

    if (isinstance(n_links, (bool, np.bool_, np.timedelta64, np.datetime64))
            or not isinstance(n_links, (int, np.integer))):
        raise ValueError('n_links must be an integer')
    if (isinstance(n_colors, (bool, np.bool_, np.timedelta64, np.datetime64))
            or not isinstance(n_colors, (int, np.integer))):
        raise ValueError('n_colors must be an integer')
    L = int(n_links)
    nc_i = int(n_colors)
    if L not in (4, 6):
        raise ValueError('n_links must be 4 or 6')
    if not 3 <= nc_i <= 64:
        raise ValueError('n_colors must be between 3 and 64')

    def _finite_real(value, name):
        if (isinstance(value, (bool, np.bool_, np.timedelta64, np.datetime64))
                or not isinstance(value, (int, float, np.integer, np.floating))):
            raise ValueError(f'{name} must be a finite real scalar')
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{name} must be a finite real scalar') from exc
        if not np.isfinite(result):
            raise ValueError(f'{name} must be a finite real scalar')
        return result

    nc = float(nc_i)
    g = _finite_real(coupling, 'coupling')
    m = _finite_real(mass, 'mass')
    I = np.eye(2, dtype=float)
    Z = np.diag([1.0, -1.0])
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    P0 = np.diag([1.0, 0.0])

    def _embed(op, site):
        parts = [I] * L
        parts[site % L] = op
        out = parts[0]
        for p in parts[1:]:
            out = np.kron(out, p)
        return out

    def _embed3(left, mid, right, j):
        ops = [I] * L
        ops[(j - 1) % L] = left
        ops[j % L] = mid
        ops[(j + 1) % L] = right
        out = ops[0]
        for p in ops[1:]:
            out = np.kron(out, p)
        return out

    H = np.zeros((2 ** L, 2 ** L), dtype=float)
    elec_pre = -(g * ((nc * nc - 1.0) / (8.0 * nc))) * g
    a = 0.5 * np.sqrt(nc)
    if not np.all(np.isfinite([elec_pre, a])):
        raise ValueError('parameters must produce finite Hamiltonian coefficients')
    for j in range(L):
        H += (elec_pre - m) * _embed(Z, j)
        H += a * _embed3(P0, X, P0, j)
    if not np.all(np.isfinite(H)):
        raise ValueError('parameters must produce finite Hamiltonian entries')
    return H

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: assemble_t1_hamiltonian(7, 3, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t1_hamiltonian(7, 3, 1.0, 1.0))'},
 {'setup': '# case: normal\n',
  'call': 'assemble_t1_hamiltonian(4, 64, 0.2, 0.7)',
  'gold_call': '_oracle_assemble_t1_hamiltonian(4, 64, 0.2, 0.7)'},
 {'setup': '# case: normal\n',
  'call': 'assemble_t1_hamiltonian(4, 4, 0.0, 1.3)',
  'gold_call': '_oracle_assemble_t1_hamiltonian(4, 4, 0.0, 1.3)'},
 {'setup': '# case: normal\n',
  'call': 'assemble_t1_hamiltonian(4, 3, 1.2, 0.0)',
  'gold_call': '_oracle_assemble_t1_hamiltonian(4, 3, 1.2, 0.0)'},
 {'setup': '# case: normal\n',
  'call': 'assemble_t1_hamiltonian(4, 3, 1.0, 1.0)',
  'gold_call': '_oracle_assemble_t1_hamiltonian(4, 3, 1.0, 1.0)'},
 {'setup': '# case: edge\n',
  'call': 'assemble_t1_hamiltonian(4, 3, 1.0, -0.6)',
  'gold_call': '_oracle_assemble_t1_hamiltonian(4, 3, 1.0, -0.6)'},
 {'setup': '# case: normal\n',
  'call': 'assemble_t1_hamiltonian(4, 5, 0.5, 0.25)',
  'gold_call': '_oracle_assemble_t1_hamiltonian(4, 5, 0.5, 0.25)'},
 {'setup': '# case: normal\n',
  'call': 'assemble_t1_hamiltonian(6, 3, 1.0, 1.0)',
  'gold_call': '_oracle_assemble_t1_hamiltonian(6, 3, 1.0, 1.0)'},
 {'setup': '# case: boundary\n',
  'call': 'assemble_t1_hamiltonian(4, 3, -0.8, 1.2)',
  'gold_call': '_oracle_assemble_t1_hamiltonian(4, 3, -0.8, 1.2)'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: assemble_t1_hamiltonian(4, 65, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t1_hamiltonian(4, 65, 1.0, 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: edge\n',
  'call': "capture_value_error(lambda: assemble_t1_hamiltonian(4, 3, float('inf'), 1.0))",
  'gold_call': "capture_value_error(lambda: _oracle_assemble_t1_hamiltonian(4, 3, float('inf'), "
               '1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: edge\n',
  'call': "capture_value_error(lambda: assemble_t1_hamiltonian(4, 3, 1.0, float('nan')))",
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t1_hamiltonian(4, 3, 1.0, '
               "float('nan')))"},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: edge\n',
  'call': 'capture_value_error(lambda: assemble_t1_hamiltonian(4.0, 3, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t1_hamiltonian(4.0, 3, 1.0, 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: assemble_t1_hamiltonian(2, 3, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t1_hamiltonian(2, 3, 1.0, 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: assemble_t1_hamiltonian(4, 1, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t1_hamiltonian(4, 1, 1.0, 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: edge\n',
  'call': 'capture_value_error(lambda: assemble_t1_hamiltonian(4, 3.0, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t1_hamiltonian(4, 3.0, 1.0, 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: assemble_t1_hamiltonian(3, 3, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t1_hamiltonian(3, 3, 1.0, 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: assemble_t1_hamiltonian(4, 2, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t1_hamiltonian(4, 2, 1.0, 1.0))'},
 {'setup': '# case: normal; declared return representation\n'
           'import numpy as np\n'
           'def _has_return_type(value):\n'
           '    return int(isinstance(value, np.ndarray) and value.dtype.kind == "f" and value.dtype.itemsize == 8)\n',
  'call': '_has_return_type(assemble_t1_hamiltonian(4, 3, 0.8, 1.2))',
  'gold_call': '_has_return_type(_oracle_assemble_t1_hamiltonian(4, 3, 0.8, 1.2))'}]
