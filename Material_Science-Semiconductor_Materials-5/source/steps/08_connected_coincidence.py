"""
Compute the noninvasive connected coincidence from the joint stationary family.

This constructed third cumulant applies the paper’s arbitrary-order photon spectrum to dependence shared by three detection channels.

Returns
-------
result : float     Native dimensionless scalar: the lambda->0+ limit of the joint     third cumulant of the three probe occupations divided by their     stationary means. Required absolute accuracy is 2e-5. Compose     joint_operators, bath_response through phonon_rate_series,     phonon_rate_series, phonon_superoperator, total_liouvillian,     stationary_density, and subset_spectra.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def connected_coincidence(parameters: dict) -> float:
    """Compute the noninvasive connected third photon coincidence (orchestrator).

    Parameters
    ----------
    parameters : dict
        Keys delta, omega, detunings, gamma, and widths give the dot
        detuning omega0'-omega_L of its polaron-shifted transition above the
        laser, Rabi frequency, three probe detunings omega_m-omega_L, dot
        population decay, and three probe population decays in ps^-1.
        Decay rates are positive.
        directions is a real length-three sequence of nonzero dimensionless
        exchange ratios, epsilon_m=lambda*directions[m], lambda in ps^-1.
        alpha>=0 is in ps^2, 0.8<=cutoff<=3 is in ps^-1, and temperature
        is in K with 1<=temperature<=10. Use k_B/hbar=0.1309203391 ps^-1 K^-1.
        delta uses the compensated shifted-transition convention. The
        uncoupled joint Hamiltonian has spectral span <=4*cutoff. The
        stationary family is unique near zero, with positive leading
        single-probe occupations.

    Returns
    -------
    result : float
        Native dimensionless scalar: the lambda->0+ limit of the joint
        third cumulant of the three probe occupations divided by their
        stationary means. Required absolute accuracy is 2e-5. Compose
        joint_operators, bath_response through phonon_rate_series,
        phonon_rate_series, phonon_superoperator, total_liouvillian,
        stationary_density, and subset_spectra.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_connected_coincidence(parameters: dict) -> float:
    p=parameters
    operators=_oracle_joint_operators(p['delta'],p['omega'],np.asarray(p['detunings']),np.asarray(p['directions']))
    z=_oracle_phonon_rate_series(*operators[:3],p['alpha'],p['cutoff'],p['temperature'],6)
    k=_oracle_phonon_superoperator(operators[2],z)
    liouvillian=_oracle_total_liouvillian(operators,k,p['gamma'],np.asarray(p['widths']),p['alpha'],p['cutoff'])
    rho=_oracle_stationary_density(liouvillian)
    spectra=_oracle_subset_spectra(rho,operators[4:],np.asarray(p['widths']),np.asarray(p['directions']))
    g12=spectra[2]/spectra[0]/spectra[1]
    g13=spectra[4]/spectra[0]/spectra[3]
    g23=spectra[5]/spectra[1]/spectra[3]
    g123=spectra[6]/spectra[0]/spectra[1]/spectra[3]
    return float(g123-g12-g13-g23+2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independently initialized scientific cases."""
    return [{'setup': 'import numpy as np\n'
               '## Noninvasive three-probe configuration 1\n'
               'import copy\n'
               "parameters = {'delta': 0.013, 'omega': 0.22, 'detunings': [-0.45, -0.64, -0.84], "
               "'directions': [1.0, 1.2, 0.9], 'temperature': 4.0, 'cutoff': 2.2, 'alpha': 0.015, 'gamma': "
               "0.012, 'widths': [0.025, 0.033, 0.029]}\n",
      'call': 'connected_coincidence(copy.deepcopy(parameters))',
      'gold_call': '_oracle_connected_coincidence(copy.deepcopy(parameters))',
      'tol': 2e-05},
     {'setup': 'import numpy as np\n'
               '## Noninvasive three-probe configuration 2\n'
               'import copy\n'
               "parameters = {'delta': 0.013, 'omega': 0.22, 'detunings': [-0.45, -0.64, -0.84], "
               "'directions': [1.0, 1.2, 0.9], 'temperature': 4.0, 'cutoff': 2.2, 'alpha': 0.035, 'gamma': "
               "0.012, 'widths': [0.025, 0.033, 0.029]}\n",
      'call': 'connected_coincidence(copy.deepcopy(parameters))',
      'gold_call': '_oracle_connected_coincidence(copy.deepcopy(parameters))',
      'tol': 2e-05},
     {'setup': 'import numpy as np\n'
               '## Noninvasive three-probe configuration 3\n'
               'import copy\n'
               "parameters = {'delta': 0.013, 'omega': 0.22, 'detunings': [-0.45, -0.64, -0.84], "
               "'directions': [1.0, 1.2, 0.9], 'temperature': 1.0, 'cutoff': 2.2, 'alpha': 0.027, 'gamma': "
               "0.012, 'widths': [0.025, 0.033, 0.029]}\n",
      'call': 'connected_coincidence(copy.deepcopy(parameters))',
      'gold_call': '_oracle_connected_coincidence(copy.deepcopy(parameters))',
      'tol': 2e-05},
     {'setup': 'import numpy as np\n'
               '## Noninvasive three-probe configuration 4\n'
               'import copy\n'
               "parameters = {'delta': -0.021, 'omega': 0.22, 'detunings': [-0.45, -0.64, -0.84], "
               "'directions': [1.0, 1.2, 0.9], 'temperature': 4.0, 'cutoff': 2.2, 'alpha': 0.027, 'gamma': "
               "0.012, 'widths': [0.025, 0.033, 0.029]}\n",
      'call': 'connected_coincidence(copy.deepcopy(parameters))',
      'gold_call': '_oracle_connected_coincidence(copy.deepcopy(parameters))',
      'tol': 2e-05},
     {'setup': 'import numpy as np\n'
               '## Noninvasive three-probe configuration 5\n'
               'import copy\n'
               "parameters = {'delta': 0.013, 'omega': 0.18, 'detunings': [-0.45, -0.64, -0.84], "
               "'directions': [1.0, 1.2, 0.9], 'temperature': 4.0, 'cutoff': 2.2, 'alpha': 0.027, 'gamma': "
               "0.012, 'widths': [0.025, 0.033, 0.029]}\n",
      'call': 'connected_coincidence(copy.deepcopy(parameters))',
      'gold_call': '_oracle_connected_coincidence(copy.deepcopy(parameters))',
      'tol': 2e-05},
     {'setup': 'import numpy as np\n'
               '## Noninvasive three-probe configuration 6\n'
               'import copy\n'
               "parameters = {'delta': 0.013, 'omega': 0.22, 'detunings': [-0.35, -0.53, -0.72], "
               "'directions': [1.0, 1.2, 0.9], 'temperature': 4.0, 'cutoff': 2.2, 'alpha': 0.027, 'gamma': "
               "0.012, 'widths': [0.025, 0.033, 0.029]}\n",
      'call': 'connected_coincidence(copy.deepcopy(parameters))',
      'gold_call': '_oracle_connected_coincidence(copy.deepcopy(parameters))',
      'tol': 2e-05},
     {'setup': 'import numpy as np\n'
               '## Noninvasive three-probe configuration 7\n'
               'import copy\n'
               "parameters = {'delta': 0.013, 'omega': 0.22, 'detunings': [-0.45, -0.64, -0.84], "
               "'directions': [1.0, 1.2, 0.9], 'temperature': 4.0, 'cutoff': 2.2, 'alpha': 0.027, 'gamma': "
               "0.012, 'widths': [0.04, 0.025, 0.03]}\n",
      'call': 'connected_coincidence(copy.deepcopy(parameters))',
      'gold_call': '_oracle_connected_coincidence(copy.deepcopy(parameters))',
      'tol': 2e-05},
     {'setup': 'import numpy as np\n'
               '## Noninvasive three-probe configuration 8\n'
               'import copy\n'
               "parameters = {'delta': 0.013, 'omega': 0.22, 'detunings': [-0.45, -0.64, -0.84], "
               "'directions': [1.0, 1.2, 0.9], 'temperature': 4.0, 'cutoff': 1.8, 'alpha': 0.027, 'gamma': "
               "0.012, 'widths': [0.025, 0.033, 0.029]}\n",
      'call': 'connected_coincidence(copy.deepcopy(parameters))',
      'gold_call': '_oracle_connected_coincidence(copy.deepcopy(parameters))',
      'tol': 2e-05},
     {'setup': 'import numpy as np\n'
               '## Noninvasive three-probe configuration 9\n'
               'import copy\n'
               "parameters = {'delta': 0.013, 'omega': 0.22, 'detunings': [-0.45, -0.64, -0.84], "
               "'directions': [-1.3, 1.1, 0.8], 'temperature': 4.0, 'cutoff': 2.2, 'alpha': 0.027, 'gamma': "
               "0.012, 'widths': [0.025, 0.033, 0.029]}\n",
      'call': 'connected_coincidence(copy.deepcopy(parameters))',
      'gold_call': '_oracle_connected_coincidence(copy.deepcopy(parameters))',
      'tol': 2e-05},
     {'setup': 'import numpy as np\n'
               '## Noninvasive three-probe configuration 10\n'
               'import copy\n'
               "parameters = {'delta': 0.013, 'omega': 0.22, 'detunings': [-0.28, -0.28, -0.28], "
               "'directions': [1.0, 1.2, 0.9], 'temperature': 4.0, 'cutoff': 2.2, 'alpha': 0.027, 'gamma': "
               "0.012, 'widths': [0.03, 0.03, 0.03]}\n",
      'call': 'connected_coincidence(copy.deepcopy(parameters))',
      'gold_call': '_oracle_connected_coincidence(copy.deepcopy(parameters))',
      'tol': 2e-05}]
