"""
Generate the deterministic scalar stripe disorder and execute all seven preceding functions, calling each directly and using its returned result. Compute the Chern marker on the uniform zero-field momentum grid and each magnetic density by continuous momentum quadrature. Return the boundary-minus-interior RMS discrepancy.

This task uses the paper’s Haldane ribbon and comparison of local topological markers, with a more demanding treatment of physical magnetic phases, occupation changes and momentum integration. Density and marker use different explicitly fixed momentum discretizations. The final value describes one finite ribbon and one scalar-disorder realization, not a disorder average or a universal scaling exponent.

Returns
-------
One finite Python float: R_boundary-R_interior.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_ribbon_comparison(nx: int=14, ny: int=129, seed: int=54, flux: float=0.003, mu: float=0.08, edge: int=2) -> float:
    """Run all seven earlier steps for the ribbon marker comparison.

    Parameters
    ----------
    nx, ny : int
        Cell count>=6 and marker momentum count>=5, booleans excluded.
    seed : int
        Unsigned 32-bit seed, booleans excluded. For each x, update
        s=(1664525*s+1013904223) mod 2**32 then delta[x]=.4*(s/2**32-.5).
    flux, mu, edge : float, float, int
        Positive flux, finite mu, and edge width accepted by earlier steps.

    Returns
    -------
    float
        R_edge-R_bulk. Compute the zero-flux marker on k_j=2*pi*j/ny.
        For each signed flux, obtain cuts with scan=128, start partition
        quadrature at order 8, then refine with tol=1e-10,max_order=256.
        Density integrates continuous ky; marker retains the stated ny grid.
        All seven earlier step functions must be called directly and their outputs
        used. Keep mu fixed at all fluxes; do not impose half filling.

    Raises
    ------
    ValueError
        Invalid parameters, nonconvergent integration or earlier input errors.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_run_ribbon_comparison(nx: int = 14, ny: int = 129, seed: int = 54,
                                   flux: float = .003, mu: float = .08,
                                   edge: int = 2) -> float:
    """Run all seven earlier steps for the ribbon marker comparison.

    Parameters
    ----------
    nx, ny : int
        Cell count>=6 and marker momentum count>=5, booleans excluded.
    seed : int
        Unsigned 32-bit seed, booleans excluded. For each x, update
        s=(1664525*s+1013904223) mod 2**32 then delta[x]=.4*(s/2**32-.5).
    flux, mu, edge : float, float, int
        Positive flux, finite mu, and edge width accepted by earlier steps.

    Returns
    -------
    float
        R_edge-R_bulk. Compute the zero-flux marker on k_j=2*pi*j/ny.
        For each signed flux, obtain cuts with scan=128, start partition
        quadrature at order 8, then refine with tol=1e-10,max_order=256.
        Density integrates continuous ky; marker retains the stated ny grid.
        All seven earlier oracles must be called directly and their outputs
        used. Keep mu fixed at all fluxes; do not impose half filling.

    Raises
    ------
    ValueError
        Invalid parameters, nonconvergent integration or earlier input errors.
    """
    for v,lo in ((nx,6),(ny,5)):
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or v<lo:
            raise ValueError('invalid grid size')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or not 0<=seed<2**32:
        raise ValueError('invalid seed')
    if not np.isscalar(flux) or not np.isrealobj(flux) or not np.isfinite(flux) or flux<=0:
        raise ValueError('invalid flux')
    s=int(seed); delta=np.empty(nx)
    for x in range(nx):
        s=(1664525*s+1013904223)%2**32
        delta[x]=.4*(s/2**32-.5)
    h=_oracle_magnetic_ribbon(2*np.pi*np.arange(ny)/ny,delta,0.)
    p=_oracle_fixed_mu_projectors(h,mu)
    c=_oracle_embedded_chern(p)
    densities=[]
    for f in (flux,-flux):
        cuts=_oracle_fermi_partition(delta,f,mu,128)
        first=_oracle_partition_density(delta,f,mu,cuts,8)
        densities.append(_oracle_refine_density(delta,f,mu,cuts,first,8,1e-10,256))
    diagnostics=_oracle_boundary_discrepancy(c,densities[0],densities[1],flux,edge)
    return float(diagnostics[3])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nfrom copy import deepcopy\n\n',
      'call': 'run_ribbon_comparison(deepcopy(7), deepcopy(17), deepcopy(4), deepcopy(0.003), '
              'deepcopy(0.08), deepcopy(1))',
      'gold_call': '_oracle_run_ribbon_comparison(deepcopy(7), deepcopy(17), deepcopy(4), '
                   'deepcopy(0.003), deepcopy(0.08), deepcopy(1))'},
     {'setup': 'import numpy as np\nfrom copy import deepcopy\n\n',
      'call': 'run_ribbon_comparison(deepcopy(9), deepcopy(24), deepcopy(51), deepcopy(0.003), '
              'deepcopy(-0.02), deepcopy(2))',
      'gold_call': '_oracle_run_ribbon_comparison(deepcopy(9), deepcopy(24), deepcopy(51), '
                   'deepcopy(0.003), deepcopy(-0.02), deepcopy(2))'},
     {'setup': 'import numpy as np\nfrom copy import deepcopy\n\n',
      'call': 'run_ribbon_comparison(deepcopy(10), deepcopy(35), deepcopy(4294967295), '
              'deepcopy(0.005), deepcopy(0.08), deepcopy(1))',
      'gold_call': '_oracle_run_ribbon_comparison(deepcopy(10), deepcopy(35), deepcopy(4294967295), '
                   'deepcopy(0.005), deepcopy(0.08), deepcopy(1))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: run_ribbon_comparison(deepcopy(True)))',
      'gold_call': 'raises_value_error(lambda: _oracle_run_ribbon_comparison(deepcopy(True)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: run_ribbon_comparison(deepcopy(5)))',
      'gold_call': 'raises_value_error(lambda: _oracle_run_ribbon_comparison(deepcopy(5)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: run_ribbon_comparison(deepcopy(6), deepcopy(4)))',
      'gold_call': 'raises_value_error(lambda: _oracle_run_ribbon_comparison(deepcopy(6), '
                   'deepcopy(4)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: run_ribbon_comparison(deepcopy(6), deepcopy(17), '
              'deepcopy(-1)))',
      'gold_call': 'raises_value_error(lambda: _oracle_run_ribbon_comparison(deepcopy(6), '
                   'deepcopy(17), deepcopy(-1)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: run_ribbon_comparison(deepcopy(6), deepcopy(17), '
              'deepcopy(3), deepcopy(0.0)))',
      'gold_call': 'raises_value_error(lambda: _oracle_run_ribbon_comparison(deepcopy(6), '
                   'deepcopy(17), deepcopy(3), deepcopy(0.0)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: run_ribbon_comparison(deepcopy(6), deepcopy(17), '
              'deepcopy(3), deepcopy(0.002), deepcopy(0.08), deepcopy(3)))',
      'gold_call': 'raises_value_error(lambda: _oracle_run_ribbon_comparison(deepcopy(6), '
                   'deepcopy(17), deepcopy(3), deepcopy(0.002), deepcopy(0.08), deepcopy(3)))'}]
