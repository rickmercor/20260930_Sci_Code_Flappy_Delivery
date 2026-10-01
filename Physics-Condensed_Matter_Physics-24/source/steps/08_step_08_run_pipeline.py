"""
Orbital entanglement of a two-orbital Kanamori impurity at finite temperature.

This step chains every preceding piece into the graded number. The mode

operators are built first, since they fix the Fock-space dimension every later

object lives in and the assembled Hamiltonian is checked against it. That

Hamiltonian carries the given interaction, crystal-field splitting and bath, is

diagonalized exactly in the eight-mode space, and becomes a thermal density

matrix at the stated inverse temperature. From that state the calculation reads

the weights of the four retained configurations, in which the two electrons sit

either one in each orbital or both on one orbital, together with the two

coherences that connect those configurations.



Reducing each orbital to a two-level system restricts the available operations

to those that respect local particle number, and that restriction dephases

coherences between different local charge sectors. The pair-transfer coherence

connects a configuration of local charge two and zero to one of charge zero and

two, so the dephasing map sends it to zero, while the spin-exchange coherence

connects two configurations of charge one and one and survives. Applying that

map to the measured pair-transfer amplitude, then taking the larger of the two

resulting branches, is what reduces the general expression to the single

surviving branch. The returned quantity is the entanglement measure, which

vanishes for a product state and reaches one for a maximally entangled pair of

the retained configurations. Invalid input raises ValueError: every argument must be finite and beta must be positive.

Returns
-------
float, the orbital entanglement measure of the two correlated orbitals as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_pipeline(U, J, delta, V_1, V_2, beta):
    """Compute the orbital entanglement measure of the two correlated orbitals.

    Parameters
    ----------
    U : float
        Intra-orbital Coulomb repulsion.
    J : float
        Hund's coupling. The inter-orbital repulsion is U - 2 J.
    delta : float
        Crystal-field splitting between the two impurity orbitals.
    V_1 : float
        Hybridization amplitude of the first orbital with its bath level.
    V_2 : float
        Hybridization amplitude of the second orbital with its bath level.
    beta : float
        Inverse temperature of the thermal state.

    Returns
    -------
    float
        Orbital entanglement measure of the two correlated orbitals.
    
    Raises
    ------
    ValueError
        If any argument is not finite, or if ``beta`` is not positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_pipeline(U, J, delta, V_1, V_2, beta):
    for name, value in (("U", U), ("J", J), ("delta", delta),
                        ("V_1", V_1), ("V_2", V_2), ("beta", beta)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    n_orb = 2
    n_bath_per_orb = 1
    n_modes = n_orb * 2 * (1 + n_bath_per_orb)
    mu = (3.0 * U - 5.0 * J) / 2.0
    eps_bath = np.zeros((n_orb, n_bath_per_orb))
    V_bath = np.array([[float(V_1)], [float(V_2)]])

    # The mode operators fix the Fock-space dimension every later object lives
    # in, so they are built first and their shape is what the Hamiltonian is
    # checked against.
    c_dag, _c = _oracle_fock_space_operators(n_modes)
    dim = c_dag.shape[1]

    H = _oracle_anderson_impurity_hamiltonian(U, J, delta, mu, eps_bath, V_bath,
                                            n_orb, n_bath_per_orb)
    if H.shape != (dim, dim):
        raise ValueError("Hamiltonian does not match the Fock-space dimension")
    rho = _oracle_thermal_density_matrix(H, beta)

    weights = _oracle_retained_configuration_weights(rho, n_orb, n_bath_per_orb)
    z_magnitude = _oracle_spin_exchange_correlator(rho, n_orb, n_bath_per_orb)

    # Reducing each orbital to a two-level system restricts the available
    # operations to those that respect local particle number, and that
    # restriction dephases coherences between different local charge sectors.
    # The pair-transfer coherence links local charge (2, 0) to (0, 2), so the
    # dephasing map sends it to zero; applying that map explicitly is what turns
    # the general two-branch concurrence into the single branch evaluated below.
    raw_pair = _oracle_pair_transfer_coherence(rho, n_orb, n_bath_per_orb)
    if not np.isfinite(raw_pair) or raw_pair < 0.0:
        raise ValueError("pair-transfer coherence must be a finite magnitude")
    # The coherence is a magnitude drawn from a normalized thermal state, so it
    # cannot exceed the geometric mean of the two populations it connects. That
    # bound is checked before the dephasing map discards the value, which is the
    # only way an error in this step can be caught: multiplying by zero first
    # would let any wrong number, including a NaN, pass through unnoticed.
    u_plus, u_minus = float(weights[0]), float(weights[3])
    if raw_pair > np.sqrt(u_plus * u_minus) + 1e-12:
        raise ValueError("pair-transfer coherence exceeds its Cauchy-Schwarz bound")
    dephased_pair = 0.0 * raw_pair
    w_1, w_2 = float(weights[1]), float(weights[2])
    pair_branch = 2.0 * max(0.0, dephased_pair - np.sqrt(w_1 * w_2))

    spin_branch = float(_oracle_orbital_concurrence(weights, z_magnitude))
    return float(max(spin_branch, pair_branch))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '',
      'call': 'round(run_pipeline(2.5, 0.5, 1.2, 0.60, 0.35, 8.0), 9)',
      'gold_call': 'round(_oracle_run_pipeline(2.5, 0.5, 1.2, 0.60, 0.35, 8.0), 9)'},
     {'setup': '',
      'call': 'round(run_pipeline(2.5, 0.5, 0.0, 0.60, 0.35, 8.0), 9)',
      'gold_call': 'round(_oracle_run_pipeline(2.5, 0.5, 0.0, 0.60, 0.35, 8.0), 9)'},
     {'setup': '',
      'call': 'round(run_pipeline(2.5, 0.5, 1.2, 0.60, 0.35, 0.05), 9)',
      'gold_call': 'round(_oracle_run_pipeline(2.5, 0.5, 1.2, 0.60, 0.35, 0.05), 9)'},
     {'setup': '',
      'call': 'round(run_pipeline(1.5, 0.3, 0.8, 0.50, 0.30, 12.0), 9)',
      'gold_call': 'round(_oracle_run_pipeline(1.5, 0.3, 0.8, 0.50, 0.30, 12.0), 9)'},
     {'setup': '\n'
               '\n'
               'import numpy as np\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(run_pipeline, 2.5, 0.5, 1.2, 0.6, 0.35, -1.0)',
      'gold_call': '_exception_code(_oracle_run_pipeline, 2.5, 0.5, 1.2, 0.6, 0.35, -1.0)'},
     {'setup': '\n'
               '\n'
               'import numpy as np\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': "_exception_code(run_pipeline, float('nan'), 0.5, 1.2, 0.6, 0.35, 8.0)",
      'gold_call': "_exception_code(_oracle_run_pipeline, float('nan'), 0.5, 1.2, 0.6, 0.35, 8.0)"}]
