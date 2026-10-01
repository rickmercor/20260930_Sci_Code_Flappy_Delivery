"""
Assemble the 1+1D T3 Kogut-Susskind qutrit Hamiltonian.

The source paper's truncation T3 of SU(n_colors) lattice QCD with staggered fermions is a periodic chain of n_links qutrits. The computational basis on each link is labeled by the three retained irreps, ordered singlet, fundamental, anti-fundamental. The global Hilbert space is the Kronecker product of those local spaces with site 0 the most significant factor, so a computational-basis index is sum_j q_j * 3**(n_links-1-j) with each local label q_j in {0,1,2}. Site indices are identified periodically. The operator returned here is the full T3 Hamiltonian of the paper's 1+1D truncation section: the one-body electric and mass pieces written there, plus the three-body hopping interaction of that same truncation, summed over all sites. Local operators on a qutrit follow the paper's projector / Pauli-X / Pauli-Z dictionary (its equation that defines P^u, X^{uv} and Z^u). Hopping amplitudes are the ones that truncation keeps from the hopping table, each converted into a three-site sandwich. coupling is the Yang-Mills g and mass is the staggered mass m of the Kogut-Susskind Hamiltonian. The returned matrix retains the additive constant in the displayed equations. A matrix shifted by a multiple of the identity has the same gaps but is a different code return. n_links is either 4 or 6. n_colors is an integer from 3 through 64. coupling and mass are finite real scalars. Section II places staggered fermions on a bipartite lattice, so a periodic chain has even length. The color range matches the T1 comparison, with N_c at least 3. This task evaluates the even lengths 4 and 6. Finite negative mass is allowed. Paper provenance: Section IV.A.3 defines T3; Eq. (19) fixes the local operator dictionary, Eq. (40) fixes the retained irreps, Eq. (41) gives the electric and mass groups, and Eq. (42) with Table III gives the displayed hopping contributions and their strengths. Include every import the implementation needs inside the function body. n_links and n_colors must have Python int or NumPy integer type; integer-valued floats such as 3.0 are outside the count domain. coupling and mass accept Python or NumPy integer/floating scalars. Boolean and temporal values are excluded from all scalar inputs. Calculations and numeric returns use float64.

Returns
-------
hamiltonian : np.ndarray, shape (3**n_links, 3**n_links), float Dense real symmetric T3 Hamiltonian in the computational basis above.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_t3_hamiltonian(n_links: int, n_colors: int, coupling: float,
                            mass: float) -> "np.ndarray":
    """Return the dense T3 qutrit Hamiltonian on a periodic chain.

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
    hamiltonian : np.ndarray, shape (3**n_links, 3**n_links), float
        Dense real symmetric T3 Hamiltonian in the computational basis above.

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

def _oracle_assemble_t3_hamiltonian(n_links: int, n_colors: int, coupling: float,
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
    # 0 = singlet, 1 = N, 2 = Nbar
    I3 = np.eye(3, dtype=float)
    P1 = np.diag([1.0, 0.0, 0.0])
    PN = np.diag([0.0, 1.0, 0.0])
    PNb = np.diag([0.0, 0.0, 1.0])
    ZN = I3 - 2.0 * PN
    ZNb = I3 - 2.0 * PNb
    X1N = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    X1Nb = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def _embed(op, site):
        parts = [I3] * L
        parts[site % L] = op
        out = parts[0]
        for part in parts[1:]:
            out = np.kron(out, part)
        return out

    def _embed3(left, mid, right, j):
        ops = [I3] * L
        ops[(j - 1) % L] = left
        ops[j % L] = mid
        ops[(j + 1) % L] = right
        out = ops[0]
        for part in ops[1:]:
            out = np.kron(out, part)
        return out

    dim = 3 ** L
    hamiltonian = np.zeros((dim, dim), dtype=float)
    elec_pre = -(g * ((nc * nc - 1.0) / (8.0 * nc))) * g
    root = np.sqrt
    leading = 0.5 * root(nc)
    joining = 0.5 / root(nc)
    adjacent = 0.5 * (root(nc - 1.0) - root(nc))
    between = 0.5 * ((nc - 1.0) / root(nc) - root(nc))
    if not np.all(np.isfinite([elec_pre, leading, joining, adjacent, between])):
        raise ValueError('parameters must produce finite Hamiltonian coefficients')
    # Combine local terms before embedding to avoid cancelling large mass terms.
    local_diagonal = elec_pre * (ZN + ZNb) - m * (ZN - ZNb)
    for j in range(L):
        hamiltonian += _embed(local_diagonal, j)
        hamiltonian += leading * _embed3(P1 + PN, X1N, P1 + PN, j)
        hamiltonian += joining * _embed3(PN, X1Nb, PN, j)
        hamiltonian += adjacent * _embed3(P1, X1N, PN, j)
        hamiltonian += adjacent * _embed3(PN, X1N, P1, j)
        hamiltonian += between * _embed3(PN, X1N, PN, j)
    if not np.all(np.isfinite(hamiltonian)):
        raise ValueError('parameters must produce finite Hamiltonian entries')
    return hamiltonian

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
  'call': 'capture_value_error(lambda: assemble_t3_hamiltonian(7, 3, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t3_hamiltonian(7, 3, 1.0, 1.0))'},
 {'setup': '# case: normal\n',
  'call': 'assemble_t3_hamiltonian(4, 64, 0.2, 0.7)',
  'gold_call': '_oracle_assemble_t3_hamiltonian(4, 64, 0.2, 0.7)'},
 {'setup': '# case: normal\n',
  'call': 'assemble_t3_hamiltonian(4, 4, 0.0, 0.0)',
  'gold_call': '_oracle_assemble_t3_hamiltonian(4, 4, 0.0, 0.0)'},
 {'setup': '# case: boundary\n',
  'call': 'assemble_t3_hamiltonian(4, 3, 1.0, 1.0)',
  'gold_call': '_oracle_assemble_t3_hamiltonian(4, 3, 1.0, 1.0)'},
 {'setup': '# case: boundary\n',
  'call': 'assemble_t3_hamiltonian(4, 3, -0.8, 1.2)',
  'gold_call': '_oracle_assemble_t3_hamiltonian(4, 3, -0.8, 1.2)'},
 {'setup': '# case: normal\n',
  'call': 'assemble_t3_hamiltonian(4, 5, 0.5, 0.25)',
  'gold_call': '_oracle_assemble_t3_hamiltonian(4, 5, 0.5, 0.25)'},
 {'setup': '# case: edge\n',
  'call': 'assemble_t3_hamiltonian(4, 3, 1.0, -0.4)',
  'gold_call': '_oracle_assemble_t3_hamiltonian(4, 3, 1.0, -0.4)'},
 {'setup': '# case: normal\n',
  'call': 'assemble_t3_hamiltonian(6, 3, 1.0, 1.0)',
  'gold_call': '_oracle_assemble_t3_hamiltonian(6, 3, 1.0, 1.0)'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: assemble_t3_hamiltonian(4, 65, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t3_hamiltonian(4, 65, 1.0, 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: edge\n',
  'call': "capture_value_error(lambda: assemble_t3_hamiltonian(4, 3, float('inf'), 1.0))",
  'gold_call': "capture_value_error(lambda: _oracle_assemble_t3_hamiltonian(4, 3, float('inf'), "
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
  'call': "capture_value_error(lambda: assemble_t3_hamiltonian(4, 3, 1.0, float('nan')))",
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t3_hamiltonian(4, 3, 1.0, '
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
  'call': 'capture_value_error(lambda: assemble_t3_hamiltonian(4.0, 3, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t3_hamiltonian(4.0, 3, 1.0, 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: assemble_t3_hamiltonian(2, 3, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t3_hamiltonian(2, 3, 1.0, 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: assemble_t3_hamiltonian(4, 1, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t3_hamiltonian(4, 1, 1.0, 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: edge\n',
  'call': 'capture_value_error(lambda: assemble_t3_hamiltonian(4, 3.0, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t3_hamiltonian(4, 3.0, 1.0, 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: assemble_t3_hamiltonian(3, 3, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t3_hamiltonian(3, 3, 1.0, 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: assemble_t3_hamiltonian(4, 2, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_assemble_t3_hamiltonian(4, 2, 1.0, 1.0))'},
 {'setup': '# case: normal; declared return representation\n'
           'import numpy as np\n'
           'def _has_return_type(value):\n'
           '    return int(isinstance(value, np.ndarray) and value.dtype.kind == "f" and value.dtype.itemsize == 8)\n',
  'call': '_has_return_type(assemble_t3_hamiltonian(4, 3, 0.8, 1.2))',
  'gold_call': '_has_return_type(_oracle_assemble_t3_hamiltonian(4, 3, 0.8, 1.2))'}]
