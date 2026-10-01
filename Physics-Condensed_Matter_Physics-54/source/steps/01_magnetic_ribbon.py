"""
Construct the complex Hamiltonian blocks from the physical geometry, directed hopping list, Bloch convention and straight-bond Peierls phase in the interface. Include the Hermitian conjugate of every directed bond once. Treat the scalar stripe disorder equally on both orbitals and truncate only at the open x edges.

The source, Local topological markers for Chern insulators in ribbon geometry, Section III, treats a Haldane ribbon with scalar disorder constant along the periodic direction and a magnetic field introduced by Peierls phases. Equation 11 fixes the line-integral convention. The explicit primitive vectors, chirality, gauge, seed and bond enumeration here are task conventions; the phase must use the physical bond, including transverse offsets.

Returns
-------
Complex NumPy array of shape (len(k), 2*nx, 2*nx).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def magnetic_ribbon(k: np.ndarray, delta: np.ndarray, flux: float=0.0, center: float=0.0) -> np.ndarray:
    """Construct a magnetic Haldane ribbon with two orbitals per cell.

    Parameters
    ----------
    k : numpy.ndarray
        Nonempty finite real vector of dimensionless y momenta.
    delta : numpy.ndarray
        Finite real vector, length nx>=4, scalar on-site stripe disorder.
    flux : float
        Finite real magnetic flux per primitive cell, in flux-quantum units.
    center : float
        Finite real gauge origin x0 in physical length units.

    Returns
    -------
    numpy.ndarray
        Complex stack (len(k),2*nx,2*nx), basis (x,A),(x,B).
        Geometry: a1=(sqrt(3)/2,1/2), a2=(0,1), rA=(0,0),
        rB=(1/sqrt(3),0), open x and infinite periodic y. On-sites are
        delta[x]+0.2 on A and delta[x]-0.2 on B. NN directed bonds
        A(x,n)->B(x+dx,n+m): (dx,m)=(0,0),(-1,0),(-1,1), t=1.
        NNN directed same-sublattice bonds have (dx,m)=(1,0),(0,-1),
        (-1,1), amplitude i/3 on A and -i/3 on B. Each directed bond
        contributes to H[target,source] and its h.c.; truncate only at x edges.
        Its Bloch multiplier is exp(-i*k*m). Additionally multiply by
        exp[-2*pi*i*flux/Acell*((Xs+Xt)/2-center)*(m+Yt-Ys)],
        Acell=sqrt(3)/2, X=x*Acell+(alpha==B)/sqrt(3), Y=x/2.
        Use physical bond displacement, not just the integer m, in Peierls.

    Raises
    ------
    ValueError
        If k, delta, flux or center violates the stated domain.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_magnetic_ribbon(k: np.ndarray, delta: np.ndarray, flux: float = 0.0,
                            center: float = 0.0) -> np.ndarray:
    """Construct a magnetic Haldane ribbon with two orbitals per cell.

    Parameters
    ----------
    k : numpy.ndarray
        Nonempty finite real vector of dimensionless y momenta.
    delta : numpy.ndarray
        Finite real vector, length nx>=4, scalar on-site stripe disorder.
    flux : float
        Finite real magnetic flux per primitive cell, in flux-quantum units.
    center : float
        Finite real gauge origin x0 in physical length units.

    Returns
    -------
    numpy.ndarray
        Complex stack (len(k),2*nx,2*nx), basis (x,A),(x,B).
        Geometry: a1=(sqrt(3)/2,1/2), a2=(0,1), rA=(0,0),
        rB=(1/sqrt(3),0), open x and infinite periodic y. On-sites are
        delta[x]+0.2 on A and delta[x]-0.2 on B. NN directed bonds
        A(x,n)->B(x+dx,n+m): (dx,m)=(0,0),(-1,0),(-1,1), t=1.
        NNN directed same-sublattice bonds have (dx,m)=(1,0),(0,-1),
        (-1,1), amplitude i/3 on A and -i/3 on B. Each directed bond
        contributes to H[target,source] and its h.c.; truncate only at x edges.
        Its Bloch multiplier is exp(-i*k*m). Additionally multiply by
        exp[-2*pi*i*flux/Acell*((Xs+Xt)/2-center)*(m+Yt-Ys)],
        Acell=sqrt(3)/2, X=x*Acell+(alpha==B)/sqrt(3), Y=x/2.
        Use physical bond displacement, not just the integer m, in Peierls.

    Raises
    ------
    ValueError
        If k, delta, flux or center violates the stated domain.
    """
    k,delta=np.asarray(k),np.asarray(delta)
    if k.ndim!=1 or k.size<1 or not np.isrealobj(k) or not np.all(np.isfinite(k)):
        raise ValueError('invalid momentum vector')
    if delta.ndim!=1 or delta.size<4 or not np.isrealobj(delta) or not np.all(np.isfinite(delta)):
        raise ValueError('invalid disorder vector')
    for a in (flux,center):
        if not np.isscalar(a) or not np.isrealobj(a) or not np.isfinite(a):
            raise ValueError('invalid flux or center')
    nx=len(delta); area=np.sqrt(3)/2
    X=np.repeat(np.arange(nx)*area,2)+np.tile([0,1/np.sqrt(3)],nx)
    Y=np.repeat(np.arange(nx)/2,2)
    h=np.zeros((len(k),2*nx,2*nx),complex)
    h[:,np.arange(2*nx),np.arange(2*nx)]=np.repeat(delta,2)+np.tile([.2,-.2],nx)
    bonds=[]
    for x in range(nx):
        for dx,m in [(0,0),(-1,0),(-1,1)]:
            if 0<=x+dx<nx: bonds.append((2*x,2*(x+dx)+1,m,1.))
        for alpha in [0,1]:
            for dx,m in [(1,0),(0,-1),(-1,1)]:
                if 0<=x+dx<nx:
                    bonds.append((2*x+alpha,2*(x+dx)+alpha,m,(1 if alpha==0 else -1)*1j/3))
    for s,t,m,amp in bonds:
        phase=-2*np.pi*flux/area*((X[s]+X[t])/2-center)*(m+Y[t]-Y[s])
        v=amp*np.exp(-1j*k*m+1j*phase)
        h[:,t,s]+=v
        h[:,s,t]+=v.conj()
    return h

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'import numpy as np\n'
               '\n'
               'def pack(a):\n'
               '    a = np.asarray(a)\n'
               '    return np.stack((a.real, a.imag), axis=-1)\n',
      'call': 'pack(magnetic_ribbon(deepcopy(np.array([0.0, 0.7, 2.9])), deepcopy(np.array([0.1, -0.2, '
              '0.03, 0.05]))))',
      'gold_call': 'pack(_oracle_magnetic_ribbon(deepcopy(np.array([0.0, 0.7, 2.9])), '
                   'deepcopy(np.array([0.1, -0.2, 0.03, 0.05]))))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'import numpy as np\n'
               '\n'
               'def pack(a):\n'
               '    a = np.asarray(a)\n'
               '    return np.stack((a.real, a.imag), axis=-1)\n',
      'call': 'pack(magnetic_ribbon(deepcopy(np.linspace(0, 6, 5)), deepcopy(np.linspace(-0.1, 0.2, '
              '6)), deepcopy(0.013), deepcopy(0.4)))',
      'gold_call': 'pack(_oracle_magnetic_ribbon(deepcopy(np.linspace(0, 6, 5)), '
                   'deepcopy(np.linspace(-0.1, 0.2, 6)), deepcopy(0.013), deepcopy(0.4)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'import numpy as np\n'
               '\n'
               'def pack(a):\n'
               '    a = np.asarray(a)\n'
               '    return np.stack((a.real, a.imag), axis=-1)\n',
      'call': 'pack(magnetic_ribbon(deepcopy(np.array([-0.4, 6.1])), deepcopy(np.zeros(5)), '
              'deepcopy(-0.02), deepcopy(-0.3)))',
      'gold_call': 'pack(_oracle_magnetic_ribbon(deepcopy(np.array([-0.4, 6.1])), '
                   'deepcopy(np.zeros(5)), deepcopy(-0.02), deepcopy(-0.3)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: magnetic_ribbon(deepcopy(np.array([])), '
              'deepcopy(np.zeros(4))))',
      'gold_call': 'raises_value_error(lambda: _oracle_magnetic_ribbon(deepcopy(np.array([])), '
                   'deepcopy(np.zeros(4))))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: magnetic_ribbon(deepcopy(np.array([0])), '
              'deepcopy(np.zeros(3))))',
      'gold_call': 'raises_value_error(lambda: _oracle_magnetic_ribbon(deepcopy(np.array([0])), '
                   'deepcopy(np.zeros(3))))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: magnetic_ribbon(deepcopy(np.array([0])), '
              'deepcopy(np.zeros(4)), deepcopy(np.nan)))',
      'gold_call': 'raises_value_error(lambda: _oracle_magnetic_ribbon(deepcopy(np.array([0])), '
                   'deepcopy(np.zeros(4)), deepcopy(np.nan)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: magnetic_ribbon(deepcopy(np.array([1j])), '
              'deepcopy(np.zeros(4))))',
      'gold_call': 'raises_value_error(lambda: _oracle_magnetic_ribbon(deepcopy(np.array([1j])), '
                   'deepcopy(np.zeros(4))))'}]
