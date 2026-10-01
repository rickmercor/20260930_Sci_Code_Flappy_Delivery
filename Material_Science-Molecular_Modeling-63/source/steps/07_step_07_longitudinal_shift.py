"""
Apply acoustic corrections, construct the longitudinal dynamical matrix and calculate the shift of the highest optical frequency.

Apply the prescribed acoustic corrections to the Cartesian Hessian and scaled Born tensors before mass weighting. Isotropic electronic screening cancels from the nonanalytic correction expressed through scaled Born tensors. Restrict both dynamical matrices to the complement of the mass-weighted translations and compare their largest optical eigenvalues.

Returns
-------
return shift
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def longitudinal_shift(analytic_hessian, born_scaled, masses, box, direction):
    """Return the longitudinal shift of the highest optical frequency in reduced units.
    
    Parameters
    ----------
    analytic_hessian : (3*N,3*N) real array, N>=2
        Analytic Cartesian force constants before acoustic correction. The
        short-range term is already included; no macroscopic nonanalytic
        correction is contained here.
    born_scaled : (N,3,3) real array
        Atom, polarization, displacement indices; Z*=sqrt(epsilon_inf)*Z0.
    masses : (N,) positive real array
    box : (3,) positive real array
    direction : (3,) nonzero real array
        Wavevector direction; normalization is Euclidean.
    
    Electronic screening is isotropic. First set Z=Z0-mean_atoms(Z0), and
    Hc=P*(H+H.T)/2*P with P=I-T*T.T/N and T=tile(I3,(N,1)). These are the
    specified Euclidean minimum-norm acoustic corrections in Cartesian
    coordinates. Then D_ab=Hc_ab/sqrt(m_atom(a)*m_atom(b)).
    For unit direction n, set v[l,b]=sum_a n[a]*Z[l,a,b]/sqrt(m_l).
    NAC=(4*pi/prod(box))*outer(v.ravel(),v.ravel()). The epsilon_inf factor
    cancels; do not divide NAC by epsilon_inf again. Project D and D+NAC
    onto the complement of the three MASS-WEIGHTED translations, whose
    columns are sqrt(m_l)*e_a. For each projected matrix use its largest
    eigenvalue lambda0 or lambdaL. No tracking of a labeled eigenvector.
    
    Returns
    -------
    shift : float
        sqrt(lambdaL)-sqrt(lambda0), with no 2*pi or SI conversion.
    
    Raises
    ------
    ValueError
        If shapes do not match, any input is nonfinite, masses or box are
        nonpositive, direction has zero norm, or either largest optical
        eigenvalue is <=0.

    Runtime
    -------
    Import required libraries inside the submitted function and inside any
    helper that uses them. Do not assume that np, itertools or erfc exists
    in the function's global namespace. Imports in test setup are separate.
    """
    return shift

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_longitudinal_shift(analytic_hessian, born_scaled, masses, box, direction):
    """Reference implementation."""
    import numpy as np
    H = np.asarray(analytic_hessian, float)
    Z = np.asarray(born_scaled, float)
    m = np.asarray(masses, float)
    L = np.asarray(box, float)
    vdir = np.asarray(direction, float)
    if m.ndim != 1 or len(m) < 2 or H.shape != (3 * len(m), 3 * len(m)) or (Z.shape != (len(m), 3, 3)) or (L.shape != (3,)) or (vdir.shape != (3,)):
        raise ValueError('shape')
    if not all((np.all(np.isfinite(a)) for a in (H, Z, m, L, vdir))) or np.any(m <= 0) or np.any(L <= 0) or (np.linalg.norm(vdir) == 0):
        raise ValueError('domain')
    n = len(m)
    T = np.tile(np.eye(3), (n, 1))
    P = np.eye(3 * n) - T @ T.T / n
    Hc = P @ ((H + H.T) / 2) @ P
    Z = Z - Z.mean(axis=0)
    scale = np.repeat(np.sqrt(m), 3)
    D = Hc / np.outer(scale, scale)
    normal = vdir / np.linalg.norm(vdir)
    vector = np.einsum('a,lab->lb', normal, Z) / np.sqrt(m)[:, None]
    nac = 4 * np.pi / L.prod() * np.outer(vector.ravel(), vector.ravel())
    Q, _ = np.linalg.qr(scale[:, None] * T, mode='complete')
    optical = Q[:, 3:]
    lo0 = np.linalg.eigvalsh(optical.T @ D @ optical)[-1]
    lo1 = np.linalg.eigvalsh(optical.T @ (D + nac) @ optical)[-1]
    if lo0 <= 0 or lo1 <= 0:
        raise ValueError('nonpositive top optical curvature')
    return float(np.sqrt(lo1) - np.sqrt(lo0))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Normal, boundary, edge and declared-invalid fixtures."""
    return [
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(812)\n'
                'x=rng.normal(size=(9,9));h=x.T@x+np.eye(9)\n'
                'z=rng.normal(size=(3,3,3));m=np.array([1.,2.,3.]);L=np.array([4.,5.,6.]);direction=np.array([2.,-1.,3.])\n'
                'h_g = h.copy(); z_g = z.copy(); m_g = m.copy(); L_g = L.copy(); direction_g = direction.copy()\n'
            ),
            'call': 'longitudinal_shift(h,z,m,L,direction)',
            'gold_call': '_oracle_longitudinal_shift(h_g,z_g,m_g,L_g,direction_g)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(812)\n'
                'x=rng.normal(size=(9,9));h=x.T@x+np.eye(9)\n'
                'z=rng.normal(size=(3,3,3));m=np.array([1.,2.,3.]);L=np.array([4.,5.,6.]);direction=np.array([2.,-1.,3.])\n'
                'z[:]=np.eye(3)\n'
                'h_g = h.copy(); z_g = z.copy(); m_g = m.copy(); L_g = L.copy(); direction_g = direction.copy()\n'
            ),
            'call': 'longitudinal_shift(h,z,m,L,direction)',
            'gold_call': '_oracle_longitudinal_shift(h_g,z_g,m_g,L_g,direction_g)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'h=3*np.eye(6);z=np.array([np.eye(3),-np.eye(3)]);m=np.ones(2);L=np.array([3.,3.,3.]);direction=np.array([0.,0.,4.])\n'
                'h_g = h.copy(); z_g = z.copy(); m_g = m.copy(); L_g = L.copy(); direction_g = direction.copy()\n'
            ),
            'call': 'longitudinal_shift(h,z,m,L,direction)',
            'gold_call': '_oracle_longitudinal_shift(h_g,z_g,m_g,L_g,direction_g)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(812)\n'
                'x=rng.normal(size=(9,9));h=x.T@x+np.eye(9)\n'
                'z=rng.normal(size=(3,3,3));m=np.array([1.,2.,3.]);L=np.array([4.,5.,6.]);direction=np.array([2.,-1.,3.])\n'
                'h=-np.eye(9)\n'
                'def _raise_code(f):\n'
                '    try:\n'
                '        f()\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        raise\n'
                '    raise AssertionError("Expected ValueError")\n'
                'h_g = h.copy(); z_g = z.copy(); m_g = m.copy(); L_g = L.copy(); direction_g = direction.copy()\n'
            ),
            'call': '_raise_code(lambda: longitudinal_shift(h,z,m,L,direction))',
            'gold_call': '_raise_code(lambda: _oracle_longitudinal_shift(h_g,z_g,m_g,L_g,direction_g))',
        },
    ]
