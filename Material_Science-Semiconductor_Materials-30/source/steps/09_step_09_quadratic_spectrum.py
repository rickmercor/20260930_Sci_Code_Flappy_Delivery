"""
Compute the full discrete-frequency spectrum of a quadratic Gaussian readout

Both intra-cycle correlations and inter-cycle memory contribute. Quadratic readout couples products of Gaussian lag correlations.

Returns
-------
return real_part, imaginary_part
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def quadratic_spectrum(M, C, Qxx, Qxz, Qzz, L, H, omega):
    """All matrices are real. M(d,d) has spectral radius<=.98; C(g,d),
    Qxx(d,d), Qxz(d,g), Qzz(g,g), L(o,g), H(o,g,g), with d,g,o>=1.
    The joint innovation covariance [[Qxx,Qxz],[Qxz.T,Qzz]] is symmetric PSD.
    Each H[a] is symmetric; it may be dense and indefinite. omega is any finite
    real number in radians/cycle. Consider the stationary zero-mean Gaussian model
     x_(n+1)=M*x_n+epsilon_x,n, z_n=C*x_n+epsilon_z,n.
    Innovation pairs have the stated covariance, are independent between cycles,
    and independent of x_n. Define R0=Cov(z_n) and the centered detector
     y_a,n=L[a]@z_n+0.5*(z_n.T@H[a]@z_n-tr(H[a]@R0)).
    Return real_part(o,o), imag_part(o,o) of the infinite two-sided spectrum
     S(omega)=sum over integer k of exp(-1j*omega*k)*Cov(y_(n+k),y_n).
    Cov(y_(n+k),y_n) means E[y_(n+k) y_n.T]. No 1/(2*pi) or sampling-period
    factor is applied. Keep the complex cross-spectrum, including its sign.
    Correlated process/measurement innovations, M=0, L=0, H=0, singular noise,
    omega=0 and omega=pi are supported. There is no finite lag cut-off in the
    specified answer. The earlier stationary_covariance function is available.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return real_part, imaginary_part

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_quadratic_spectrum(M, C, Qxx, Qxz, Qzz, L, H, omega):
    import numpy as np
    M, C, Qxx, Qxz, Qzz, L, H = [np.asarray(v, float) for v in (M, C, Qxx, Qxz, Qzz, L, H)]
    P = _oracle_stationary_covariance(M, Qxx)
    R0 = C@P@C.T+Qzz
    K = M@P@C.T+Qxz
    phase = np.exp(-1j*omega)
    d, g, o = M.shape[0], C.shape[0], L.shape[0]
    hh = H.reshape(o, g*g)
    zero = L@R0@L.T+.5*hh@np.kron(R0, R0)@hh.T
    linear = L@C@np.linalg.solve(np.eye(d)-phase*M, phase*K)@L.T
    lifted = np.kron(C, C)@np.linalg.solve(np.eye(d*d)-phase*np.kron(M, M),
                                          phase*np.kron(K, K))
    positive = linear+.5*hh@lifted@hh.T
    spectrum = zero+positive+positive.conj().T
    return spectrum.real, spectrum.imag

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'dense_quadratic',
      'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(6107)\n'
               'M=np.array([[.85,.65],[0.,-.35]])\n'
               'C=rng.normal(size=(3,2))\n'
               'T=rng.normal(size=(5,5))*.2\n'
               'Q=T@T.T\n'
               'Qxx=Q[:2,:2];Qxz=Q[:2,2:];Qzz=Q[2:,2:]\n'
               'L=rng.normal(size=(2,3))\n'
               'H=rng.normal(size=(2,3,3));H=.5*(H+H.transpose(0,2,1))\n'
               'omega=.35\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(quadratic_spectrum(M,C,Qxx,Qxz,Qzz,L,H,omega))',
      'gold_call': '_check_result(_oracle_quadratic_spectrum(M,C,Qxx,Qxz,Qzz,L,H,omega))',
      'tol': 2e-07},
     {'name': 'negative_frequency',
      'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(6107)\n'
               'M=np.array([[.85,.65],[0.,-.35]])\n'
               'C=rng.normal(size=(3,2))\n'
               'T=rng.normal(size=(5,5))*.2\n'
               'Q=T@T.T\n'
               'Qxx=Q[:2,:2];Qxz=Q[:2,2:];Qzz=Q[2:,2:]\n'
               'L=rng.normal(size=(2,3))\n'
               'H=rng.normal(size=(2,3,3));H=.5*(H+H.transpose(0,2,1))\n'
               'omega=.35\n'
               'omega=-.35\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(quadratic_spectrum(M,C,Qxx,Qxz,Qzz,L,H,omega))',
      'gold_call': '_check_result(_oracle_quadratic_spectrum(M,C,Qxx,Qxz,Qzz,L,H,omega))',
      'tol': 2e-07},
     {'name': 'zero_frequency',
      'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(6107)\n'
               'M=np.array([[.85,.65],[0.,-.35]])\n'
               'C=rng.normal(size=(3,2))\n'
               'T=rng.normal(size=(5,5))*.2\n'
               'Q=T@T.T\n'
               'Qxx=Q[:2,:2];Qxz=Q[:2,2:];Qzz=Q[2:,2:]\n'
               'L=rng.normal(size=(2,3))\n'
               'H=rng.normal(size=(2,3,3));H=.5*(H+H.transpose(0,2,1))\n'
               'omega=.35\n'
               'omega=0.\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(quadratic_spectrum(M,C,Qxx,Qxz,Qzz,L,H,omega))',
      'gold_call': '_check_result(_oracle_quadratic_spectrum(M,C,Qxx,Qxz,Qzz,L,H,omega))',
      'tol': 2e-07},
     {'name': 'nyquist',
      'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(6107)\n'
               'M=np.array([[.85,.65],[0.,-.35]])\n'
               'C=rng.normal(size=(3,2))\n'
               'T=rng.normal(size=(5,5))*.2\n'
               'Q=T@T.T\n'
               'Qxx=Q[:2,:2];Qxz=Q[:2,2:];Qzz=Q[2:,2:]\n'
               'L=rng.normal(size=(2,3))\n'
               'H=rng.normal(size=(2,3,3));H=.5*(H+H.transpose(0,2,1))\n'
               'omega=.35\n'
               'omega=np.pi\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(quadratic_spectrum(M,C,Qxx,Qxz,Qzz,L,H,omega))',
      'gold_call': '_check_result(_oracle_quadratic_spectrum(M,C,Qxx,Qxz,Qzz,L,H,omega))',
      'tol': 2e-07},
     {'name': 'pure_quadratic',
      'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(6107)\n'
               'M=np.array([[.85,.65],[0.,-.35]])\n'
               'C=rng.normal(size=(3,2))\n'
               'T=rng.normal(size=(5,5))*.2\n'
               'Q=T@T.T\n'
               'Qxx=Q[:2,:2];Qxz=Q[:2,2:];Qzz=Q[2:,2:]\n'
               'L=rng.normal(size=(2,3))\n'
               'H=rng.normal(size=(2,3,3));H=.5*(H+H.transpose(0,2,1))\n'
               'omega=.35\n'
               'L[:]=0.\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(quadratic_spectrum(M,C,Qxx,Qxz,Qzz,L,H,omega))',
      'gold_call': '_check_result(_oracle_quadratic_spectrum(M,C,Qxx,Qxz,Qzz,L,H,omega))',
      'tol': 2e-07},
     {'name': 'zero_transition_correlated_innovations',
      'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(6107)\n'
               'M=np.array([[.85,.65],[0.,-.35]])\n'
               'C=rng.normal(size=(3,2))\n'
               'T=rng.normal(size=(5,5))*.2\n'
               'Q=T@T.T\n'
               'Qxx=Q[:2,:2];Qxz=Q[:2,2:];Qzz=Q[2:,2:]\n'
               'L=rng.normal(size=(2,3))\n'
               'H=rng.normal(size=(2,3,3));H=.5*(H+H.transpose(0,2,1))\n'
               'omega=.35\n'
               'M[:]=0.\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(quadratic_spectrum(M,C,Qxx,Qxz,Qzz,L,H,omega))',
      'gold_call': '_check_result(_oracle_quadratic_spectrum(M,C,Qxx,Qxz,Qzz,L,H,omega))',
      'tol': 2e-07},
     {'name': 'linear_only',
      'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(6107)\n'
               'M=np.array([[.85,.65],[0.,-.35]])\n'
               'C=rng.normal(size=(3,2))\n'
               'T=rng.normal(size=(5,5))*.2\n'
               'Q=T@T.T\n'
               'Qxx=Q[:2,:2];Qxz=Q[:2,2:];Qzz=Q[2:,2:]\n'
               'L=rng.normal(size=(2,3))\n'
               'H=rng.normal(size=(2,3,3));H=.5*(H+H.transpose(0,2,1))\n'
               'omega=.35\n'
               'H[:]=0.\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(quadratic_spectrum(M,C,Qxx,Qxz,Qzz,L,H,omega))',
      'gold_call': '_check_result(_oracle_quadratic_spectrum(M,C,Qxx,Qxz,Qzz,L,H,omega))',
      'tol': 2e-07},
     {'name': 'zero_noise',
      'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(6107)\n'
               'M=np.array([[.85,.65],[0.,-.35]])\n'
               'C=rng.normal(size=(3,2))\n'
               'T=rng.normal(size=(5,5))*.2\n'
               'Q=T@T.T\n'
               'Qxx=Q[:2,:2];Qxz=Q[:2,2:];Qzz=Q[2:,2:]\n'
               'L=rng.normal(size=(2,3))\n'
               'H=rng.normal(size=(2,3,3));H=.5*(H+H.transpose(0,2,1))\n'
               'omega=.35\n'
               'Qxx[:]=0.\n'
               'Qxz[:]=0.\n'
               'Qzz[:]=0.\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(quadratic_spectrum(M,C,Qxx,Qxz,Qzz,L,H,omega))',
      'gold_call': '_check_result(_oracle_quadratic_spectrum(M,C,Qxx,Qxz,Qzz,L,H,omega))',
      'tol': 2e-07}]
