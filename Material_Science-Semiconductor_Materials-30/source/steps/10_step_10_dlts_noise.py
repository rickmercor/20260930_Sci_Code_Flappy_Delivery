"""
Calculate the signed cross-spectral coherence between two defect families.

The sign records the phase orientation between the selected family signals.

Returns
-------
return signed_coherence
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dlts_noise(cfg, protocol, gates, beta, population, omega, numerator, denominator):
    """cfg has T>0 (K); sigma, depth, degeneracy of shape(m,2), with sigma>0 (cm^2),
    depth>=0 (eV), degeneracy>0; weights shape(m,), strictly positive with sum 1;
    alpha>=0 and eta>=0, dimensionless. p has d=2*m entries, ordered
    (p_01,p_02,p_11,p_12,...); p_i0=1-p_i1-p_i2. p_i1,p_i2>=0 and p_i1+p_i2<=1.

    Use periodic_state, hybrid_cycle, detector_coefficients and quadratic_spectrum
    with their stated conventions. protocol/gates obey the stable event-controlled
    domain; beta>=0 scalar, population>0, omega finite radians/cycle.
    numerator and denominator are valid zero-based family indices, possibly equal.
    Let S be the stationary spectrum of the specified centered quadratic detector.
    Return float Im(S[numerator,denominator])/
    sqrt(Re(S[numerator,numerator])*Re(S[denominator,denominator])).
    Raises
    ------
    ValueError: if a prerequisite fails or either selected auto-spectrum is nonpositive,
    including zero-signal choices such as identical gates or beta=0.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return signed_coherence

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_dlts_noise(cfg, protocol, gates, beta, population, omega, numerator, denominator):
    import numpy as np
    if beta == 0 or gates[0] == gates[1]:
        raise ValueError('selected auto-spectra must be positive')
    p = _oracle_periodic_state(cfg, protocol, gates)
    _, M, C, Qxx, Qxz, Qzz, mean_gates, _ = _oracle_hybrid_cycle(p, cfg, protocol, gates)
    L, H = _oracle_detector_coefficients(mean_gates, cfg['weights'], beta, population)
    real, imag = _oracle_quadratic_spectrum(M, C, Qxx, Qxz, Qzz, L, H, omega)
    first, second = real[numerator, numerator], real[denominator, denominator]
    if first <= 0 or second <= 0:
        raise ValueError('selected auto-spectra must be positive')
    return float(imag[numerator, denominator]/np.sqrt(first*second))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'main',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n',
      'call': 'dlts_noise(cfg,protocol,gates,2.,500.,.35,2,0)',
      'gold_call': '_oracle_dlts_noise(cfg,protocol,gates,2.,500.,.35,2,0)',
      'tol': 2e-07},
     {'name': 'reverse_families',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n',
      'call': 'dlts_noise(cfg,protocol,gates,2.,500.,.35,0,2)',
      'gold_call': '_oracle_dlts_noise(cfg,protocol,gates,2.,500.,.35,0,2)',
      'tol': 2e-07},
     {'name': 'same_family',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n',
      'call': 'dlts_noise(cfg,protocol,gates,2.,500.,.35,1,1)',
      'gold_call': '_oracle_dlts_noise(cfg,protocol,gates,2.,500.,.35,1,1)',
      'tol': 2e-07},
     {'name': 'negative_frequency',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n',
      'call': 'dlts_noise(cfg,protocol,gates,2.,500.,-.35,2,0)',
      'gold_call': '_oracle_dlts_noise(cfg,protocol,gates,2.,500.,-.35,2,0)',
      'tol': 2e-07},
     {'name': 'two_species',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n'
               '\n'
               'for key in ("sigma","depth","degeneracy"):\n'
               '    cfg[key]=np.asarray(cfg[key])[:2].copy()\n'
               'cfg["weights"]=[.58,.42]\n'
               'p=np.array([.10,.20,.15,.24])\n',
      'call': 'dlts_noise(cfg,protocol,gates,2.,500.,.35,1,0)',
      'gold_call': '_oracle_dlts_noise(cfg,protocol,gates,2.,500.,.35,1,0)',
      'tol': 2e-07},
     {'name': 'stronger_quadratic',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n',
      'call': 'dlts_noise(cfg,protocol,gates,2.,100.,.35,2,0)',
      'gold_call': '_oracle_dlts_noise(cfg,protocol,gates,2.,100.,.35,2,0)',
      'tol': 2e-07},
     {'name': 'nonpositive_spectrum',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n'
               'gates=np.array([.5,.5])\n'
               '\n'
               'def _capture_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               "    raise AssertionError('expected ValueError')\n",
      'call': '_capture_value_error(lambda: dlts_noise(cfg,protocol,gates,2.,500.,.35,2,0))',
      'gold_call': '_capture_value_error(lambda: '
                   '_oracle_dlts_noise(cfg,protocol,gates,2.,500.,.35,2,0))',
      'tol': 2e-07}]
