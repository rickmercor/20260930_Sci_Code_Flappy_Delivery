"""
Truncation gap index of the SU(Nc) link truncations.

Truncations T1 and T3 keep different sets of link irreps. Compose the preceding functions so their returned matrices, indices, gaps and meson energy determine the result. Compute each truncation's gap on the block of states reachable from the free vacuum. Return T3's gap minus T1's, divided by the energy of one isolated meson. Use a magnitude threshold of 1e-12 when deciding which off-diagonal entries connect two basis states. Return a native Python float. n_links is either 4 or 6. n_colors is an integer from 3 through 64. coupling and mass are finite real scalars, and the isolated-meson energy must not be zero. Paper provenance: this comparison uses the vacuum-started Krylov construction of Section III.A, Eq. (4), the T1 energy and Hamiltonian in Section IV.A.1, Eq. (25), Eq. (26), Eq. (27), and Eq. (28), and the T3 Hamiltonian in Section IV.A.3, Eq. (41) and Eq. (42). The blocks are those Krylov spaces; the signed ratio of their gaps to the isolated-meson energy is the diagnostic. Return the unrounded ratio; the problem statement rounds only the tagged final answer. A negative isolated-meson energy retains its sign in the denominator. Include every import the implementation needs inside the function body. n_links and n_colors must have Python int or NumPy integer type; integer-valued floats such as 3.0 are outside the count domain. coupling and mass accept Python or NumPy integer/floating scalars. Boolean and temporal values are excluded from all scalar inputs. Calculations and numeric returns use float64.

Returns
-------
index : float (T3 vacuum-block gap minus T1 vacuum-block gap) divided by the isolated-meson energy. Native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_truncation_gap_index(n_links: int, n_colors: int, coupling: float,
                                 mass: float) -> float:
    """Vacuum-block gap of T3 minus that of T1, per isolated meson.

    Parameters
    ----------
    n_links : int
        Number of links on the periodic chain. Either 4 or 6 (an even periodic chain).
    n_colors : int
        Number of colors. Integer from 3 through 64.
    coupling : float
        Gauge coupling. Finite.
    mass : float
        Fermion mass. Finite. Together with coupling it must give a non-zero
        isolated-meson energy.

    Returns
    -------
    index : float
        (T3 vacuum-block gap minus T1 vacuum-block gap) divided by the isolated-meson
        energy. Native Python float.

    Raises
    ------
    ValueError
        If either count is not an integer in its stated range, coupling or mass is
        not a finite real scalar, the isolated-meson energy is zero, or a finite
        Hamiltonian, spectral gap or final index cannot be computed.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_truncation_gap_index(n_links: int, n_colors: int, coupling: float,
                                         mass: float) -> float:
    import numpy as np

    if (isinstance(n_links, (bool, np.bool_, np.timedelta64, np.datetime64))
            or not isinstance(n_links, (int, np.integer))):
        raise ValueError('n_links must be an integer')
    if (isinstance(n_colors, (bool, np.bool_, np.timedelta64, np.datetime64))
            or not isinstance(n_colors, (int, np.integer))):
        raise ValueError('n_colors must be an integer')
    if int(n_links) not in (4, 6):
        raise ValueError('n_links must be 4 or 6')
    if not 3 <= int(n_colors) <= 64:
        raise ValueError('n_colors must be between 3 and 64')
    for value, name in ((coupling, 'coupling'), (mass, 'mass')):
        if (isinstance(value, (bool, np.bool_, np.timedelta64, np.datetime64))
                or not isinstance(value, (int, float, np.integer, np.floating))):
            raise ValueError(f'{name} must be a finite real scalar')
        try:
            finite = np.isfinite(float(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{name} must be a finite real scalar') from exc
        if not finite:
            raise ValueError(f'{name} must be a finite real scalar')

    pair = _oracle_isolated_meson_energy(n_colors, coupling, mass)
    if pair == 0.0:
        raise ValueError('isolated meson energy must not be zero')
    h3 = _oracle_assemble_t3_hamiltonian(n_links, n_colors, coupling, mass)
    h1 = _oracle_assemble_t1_hamiltonian(n_links, n_colors, coupling, mass)
    block3 = _oracle_vacuum_connected_indices(h3, 1e-12)
    block1 = _oracle_vacuum_connected_indices(h1, 1e-12)
    gap3 = _oracle_subspace_gap(h3, block3)
    # Compute the T1 gap from its principal vacuum block.
    gap1 = _oracle_lowest_spectral_gap(h1[np.ix_(block1, block1)])
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        result = (gap3 - gap1) / pair
    if not np.isfinite(result):
        raise ValueError('parameters must produce a finite truncation gap index')
    return float(result)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '# case: normal\n',
  'call': 'compute_truncation_gap_index(6, 4, 0.6, 0.9)',
  'gold_call': '_oracle_compute_truncation_gap_index(6, 4, 0.6, 0.9)'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: compute_truncation_gap_index(7, 3, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_compute_truncation_gap_index(7, 3, 1.0, 1.0))'},
 {'setup': '# case: normal\n',
  'call': 'compute_truncation_gap_index(4, 3, 0.0, 0.7)',
  'gold_call': '_oracle_compute_truncation_gap_index(4, 3, 0.0, 0.7)'},
 {'setup': '# case: normal\n',
  'call': 'compute_truncation_gap_index(4, 64, 0.2, 0.7)',
  'gold_call': '_oracle_compute_truncation_gap_index(4, 64, 0.2, 0.7)'},
 {'setup': '# case: normal\n',
  'call': 'compute_truncation_gap_index(6, 3, 1.0, 1.0)',
  'gold_call': '_oracle_compute_truncation_gap_index(6, 3, 1.0, 1.0)'},
 {'setup': '# case: normal\n',
  'call': 'compute_truncation_gap_index(4, 3, 0.8, 1.2)',
  'gold_call': '_oracle_compute_truncation_gap_index(4, 3, 0.8, 1.2)'},
 {'setup': '# case: normal\n',
  'call': 'compute_truncation_gap_index(4, 5, 0.5, 0.25)',
  'gold_call': '_oracle_compute_truncation_gap_index(4, 5, 0.5, 0.25)'},
 {'setup': '# case: normal\n',
  'call': 'compute_truncation_gap_index(4, 3, 1.0, 1.0)',
  'gold_call': '_oracle_compute_truncation_gap_index(4, 3, 1.0, 1.0)'},
 {'setup': '# case: edge\n',
  'call': 'compute_truncation_gap_index(4, 3, 1.0, -1.0)',
  'gold_call': '_oracle_compute_truncation_gap_index(4, 3, 1.0, -1.0)'},
 {'setup': '# case: edge\n',
  'call': 'compute_truncation_gap_index(4, 3, 1.0, -0.8)',
  'gold_call': '_oracle_compute_truncation_gap_index(4, 3, 1.0, -0.8)'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: compute_truncation_gap_index(4, 65, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_compute_truncation_gap_index(4, 65, 1.0, '
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
  'call': "capture_value_error(lambda: compute_truncation_gap_index(4, 3, 1.0, float('nan')))",
  'gold_call': 'capture_value_error(lambda: _oracle_compute_truncation_gap_index(4, 3, 1.0, '
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
  'call': 'capture_value_error(lambda: compute_truncation_gap_index(4, 3, 1.0, -1.0/3.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_compute_truncation_gap_index(4, 3, 1.0, '
               '-1.0/3.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: compute_truncation_gap_index(2, 3, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_compute_truncation_gap_index(2, 3, 1.0, 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: compute_truncation_gap_index(4, 1, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_compute_truncation_gap_index(4, 1, 1.0, 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: edge\n',
  'call': 'capture_value_error(lambda: compute_truncation_gap_index(4.0, 3, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_compute_truncation_gap_index(4.0, 3, 1.0, '
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
  'call': 'capture_value_error(lambda: compute_truncation_gap_index(4, 3.0, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_compute_truncation_gap_index(4, 3.0, 1.0, '
               '1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: compute_truncation_gap_index(3, 3, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_compute_truncation_gap_index(3, 3, 1.0, 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: compute_truncation_gap_index(4, 2, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_compute_truncation_gap_index(4, 2, 1.0, 1.0))'},
 {'setup': '# case: normal; declared return representation\n'
           'import numpy as np\n'
           'def _has_return_type(value):\n'
           '    return int(type(value) is float)\n',
  'call': '_has_return_type(compute_truncation_gap_index(4, 3, 0.8, 1.2))',
  'gold_call': '_has_return_type(_oracle_compute_truncation_gap_index(4, 3, 0.8, 1.2))'},
 {'setup': '# case: normal; retain precision in the returned index\n'
           'import numpy as np\n'
           'def _accurate_index(value):\n'
           '    return int(np.isfinite(value) and\n'
           '               abs(value - (-0.24066933384569345)) <= 1e-12)\n',
  'call': '_accurate_index(compute_truncation_gap_index(4, 3, 0.8, 1.2))',
  'gold_call': '_accurate_index(_oracle_compute_truncation_gap_index(4, 3, 0.8, 1.2))'}]
