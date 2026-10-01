"""
Calculate transition-resolved capture and emission coefficients.

Capture and emission belong to consecutive transitions within the same three-state population.

Returns
-------
return capture, emission
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def emission_coefficients(T, sigma, depth, degeneracy):
    """Return c,e of shape(m,2), in cm^3/s and s^-1 respectively.
    T>0 is in kelvin. sigma>0 is in cm^2, depth>=0 in eV, degeneracy>0 dimensionless;
    the three arrays have shape(m,2), m>=1. Use v=1e7*sqrt(T/300) cm/s,
    Nc=2.8e19*(T/300)^1.5 cm^-3 and kB=8.617333262145e-5 eV/K.
    c=sigma*v; e=c*Nc*degeneracy*exp(-depth/(kB*T)).
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return capture, emission

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_emission_coefficients(T, sigma, depth, degeneracy):
    import numpy as np
    sigma = np.asarray(sigma, float)
    depth = np.asarray(depth, float)
    degeneracy = np.asarray(degeneracy, float)
    velocity = 1.e7*np.sqrt(T/300.)
    Nc = 2.8e19*(T/300.)**1.5
    c = sigma*velocity
    e = c*Nc*degeneracy*np.exp(-depth/(8.617333262145e-5*T))
    return c, e

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'three_species',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(emission_coefficients(cfg["T"],cfg["sigma"],cfg["depth"],cfg["degeneracy"]))',
      'gold_call': '_check_result(_oracle_emission_coefficients(cfg["T"],cfg["sigma"],cfg["depth"],cfg["degeneracy"]))',
      'tol': 1e-12},
     {'name': 'low_temperature',
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
               '    cfg[key]=np.asarray(cfg[key])[:1].copy()\n'
               'cfg["weights"]=[1.]\n'
               'p=np.array([.2,.3])\n'
               'cfg["T"]=110.\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(emission_coefficients(cfg["T"],cfg["sigma"],cfg["depth"],cfg["degeneracy"]))',
      'gold_call': '_check_result(_oracle_emission_coefficients(cfg["T"],cfg["sigma"],cfg["depth"],cfg["degeneracy"]))',
      'tol': 1e-12},
     {'name': 'zero_depth',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n'
               'cfg["depth"][0][0]=0.\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(emission_coefficients(cfg["T"],cfg["sigma"],cfg["depth"],cfg["degeneracy"]))',
      'gold_call': '_check_result(_oracle_emission_coefficients(cfg["T"],cfg["sigma"],cfg["depth"],cfg["degeneracy"]))',
      'tol': 1e-12},
     {'name': 'array_inputs',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n'
               'for key in ("sigma","depth","degeneracy"):\n'
               '    cfg[key]=np.asarray(cfg[key])\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(emission_coefficients(cfg["T"],cfg["sigma"],cfg["depth"],cfg["degeneracy"]))',
      'gold_call': '_check_result(_oracle_emission_coefficients(cfg["T"],cfg["sigma"],cfg["depth"],cfg["degeneracy"]))',
      'tol': 1e-12}]
