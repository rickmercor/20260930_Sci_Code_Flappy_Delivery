"""
Isolated meson energy of the truncated Kogut-Susskind theory.

An isolated quark-antiquark pair occupying a single link in the fundamental of SU(n_colors) carries the sum of its staggered mass energy and its electric energy. The mass contribution is twice the staggered mass. The electric contribution is half the Yang-Mills coupling squared times the quadratic Casimir of the fundamental. Return that sum as a native Python float. n_colors is an integer at least 2 that is representable as a finite float64. Paper provenance: Section IV.A.1, Eq. (25), gives the isolated-pair mass plus fundamental-Casimir electric energy used here. Include every import the implementation needs inside the function body. n_colors must have Python int or NumPy integer type; integer-valued floats such as 3.0 are outside the count domain. coupling and mass accept Python or NumPy integer/floating scalars. Boolean and temporal values are excluded from all scalar inputs. Calculations and numeric returns use float64.

Returns
-------
energy : float Isolated-pair energy 2 m + (g^2 / 2) C_2(N).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def isolated_meson_energy(n_colors: int, coupling: float, mass: float) -> float:
    """Energy of an isolated fundamental meson.

    Parameters
    ----------
    n_colors : int
        Number of colors N_c. Integer at least 2, representable as finite float64.
    coupling : float
        Finite Yang-Mills coupling g.
    mass : float
        Finite staggered fermion mass m.

    Returns
    -------
    energy : float
        Isolated-pair energy 2 m + (g^2 / 2) C_2(N).

    Raises
    ------
    ValueError
        If n_colors is not an integer at least 2, if coupling or mass is not a
        finite real scalar, or if those inputs produce a non-finite energy.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_isolated_meson_energy(n_colors: int, coupling: float, mass: float) -> float:
    import numpy as np

    if (isinstance(n_colors, (bool, np.bool_, np.timedelta64, np.datetime64))
            or not isinstance(n_colors, (int, np.integer))):
        raise ValueError('n_colors must be an integer')
    nc_i = int(n_colors)
    if nc_i < 2:
        raise ValueError('n_colors must be at least 2')

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

    try:
        nc = float(nc_i)
    except OverflowError as exc:
        raise ValueError('n_colors is too large to evaluate') from exc
    if not np.isfinite(nc):
        raise ValueError('n_colors is too large to evaluate')
    g = _finite_real(coupling, 'coupling')
    m = _finite_real(mass, 'mass')
    with np.errstate(over='ignore', invalid='ignore'):
        c2 = 0.5 * (nc - 1.0 / nc)
        energy = 2.0 * m + (g * (0.5 * c2)) * g
        if not np.isfinite(energy):
            # Rescale only overflowing sums so subnormal masses are preserved.
            energy = 4.0 * (0.5 * m + (g * (0.125 * c2)) * g)
    if not np.isfinite(energy):
        raise ValueError('inputs must produce a finite isolated-meson energy')
    return float(energy)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '# case: normal\n',
  'call': 'isolated_meson_energy(64, 0.2, 0.7)',
  'gold_call': '_oracle_isolated_meson_energy(64, 0.2, 0.7)'},
 {'setup': '# case: normal\n',
  'call': 'isolated_meson_energy(4, 0.0, 1.3)',
  'gold_call': '_oracle_isolated_meson_energy(4, 0.0, 1.3)'},
 {'setup': '# case: normal\n',
  'call': 'isolated_meson_energy(128, -0.4, 0.0)',
  'gold_call': '_oracle_isolated_meson_energy(128, -0.4, 0.0)'},
 {'setup': '# case: edge\n',
  'call': 'isolated_meson_energy(3, 1.0, -0.75)',
  'gold_call': '_oracle_isolated_meson_energy(3, 1.0, -0.75)'},
 {'setup': '# case: normal\n',
  'call': 'isolated_meson_energy(3, 1.0, 1.0)',
  'gold_call': '_oracle_isolated_meson_energy(3, 1.0, 1.0)'},
 {'setup': '# case: boundary\n',
  'call': 'isolated_meson_energy(2, 0.8, 1.2)',
  'gold_call': '_oracle_isolated_meson_energy(2, 0.8, 1.2)'},
 {'setup': '# case: normal\n',
  'call': 'isolated_meson_energy(5, 0.5, 0.25)',
  'gold_call': '_oracle_isolated_meson_energy(5, 0.5, 0.25)'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: edge\n',
  'call': 'capture_value_error(lambda: isolated_meson_energy(2.5, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_isolated_meson_energy(2.5, 1.0, 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: edge\n',
  'call': 'capture_value_error(lambda: isolated_meson_energy(3, float("inf"), 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_isolated_meson_energy(3, float("inf"), 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: edge\n',
  'call': 'capture_value_error(lambda: isolated_meson_energy(3, 1.0, float("nan")))',
  'gold_call': 'capture_value_error(lambda: _oracle_isolated_meson_energy(3, 1.0, float("nan")))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: edge\n',
  'call': 'capture_value_error(lambda: isolated_meson_energy(3, 1.0 + 0.0j, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_isolated_meson_energy(3, 1.0 + 0.0j, 1.0))'},
 {'setup': 'def capture_value_error(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n'
           '# case: boundary\n',
  'call': 'capture_value_error(lambda: isolated_meson_energy(1, 1.0, 1.0))',
  'gold_call': 'capture_value_error(lambda: _oracle_isolated_meson_energy(1, 1.0, 1.0))'},
 {'setup': '# case: normal; declared return representation\n'
           'import numpy as np\n'
           'def _has_return_type(value):\n'
           '    return int(type(value) is float)\n',
  'call': '_has_return_type(isolated_meson_energy(3, 0.8, 1.2))',
  'gold_call': '_has_return_type(_oracle_isolated_meson_energy(3, 0.8, 1.2))'}]
