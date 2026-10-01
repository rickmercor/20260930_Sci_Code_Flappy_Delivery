"""
Compute each pooled reactant state from the already transformed species energies.

Protonation species occupy a common reactant pool. Its transformed formation energy, mean proton count and mean charge must be consistent with the species probabilities in that compartment.

Returns
-------
A finite ndarray of shape (P,2,M,3+J).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reactant_state(species_energies: "np.ndarray", species_h: "np.ndarray",
                   species_z: "np.ndarray", rt: float) -> "np.ndarray":
    """
    species_energies has shape (P,2,M,J), in kJ/mol; axes are condition,
    compartment (exterior then interior), reactant and species. species_h and
    species_z are length-J proton counts and charges shared by all reactants.
    rt is the positive thermal energy in kJ/mol. Return shape (P,2,M,3+J),
    with trailing entries pooled formation energy, mean bound-proton count,
    mean charge, then the J fractions in supplied order. P,M,J are positive.
    Finite aligned data are required; invalid shapes, nonfinite data or
    nonpositive rt raise ValueError. Tied energies have equal probabilities.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_reactant_state(species_energies: "np.ndarray", species_h: "np.ndarray",
                   species_z: "np.ndarray", rt: float) -> "np.ndarray":
    import numpy as np
    from scipy.special import logsumexp
    e = np.asarray(species_energies, dtype=float)
    h, z = np.asarray(species_h, dtype=float), np.asarray(species_z, dtype=float)
    if (e.ndim != 4 or e.shape[1] != 2 or min(e.shape) == 0
            or h.shape != (e.shape[-1],) or z.shape != h.shape
            or not np.isfinite(e).all() or not np.isfinite(h).all()
            or not np.isfinite(z).all() or not np.isfinite(rt) or rt <= 0):
        raise ValueError('Finite aligned species data and positive RT required')
    log_partition = logsumexp(-e / rt, axis=-1)
    fractions = np.exp(-e / rt - log_partition[..., None])
    result = np.concatenate(((-rt * log_partition)[..., None],
                             (fractions @ h)[..., None],
                             (fractions @ z)[..., None], fractions), axis=-1)
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'e=np.array([[[[0.,2.],[-1.,-3.]],[[1.,-1.],[0.,4.]]]])\n'
               'h=np.array([0.,1.]); z=h-1.',
      'call': 'reactant_state(e,h,z,2.5)',
      'gold_call': '_oracle_reactant_state(e,h,z,2.5)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\ne=np.zeros((1,2,1,2)); h=np.array([0.,1.]); z=h-1.',
      'call': 'reactant_state(e,h,z,1.)',
      'gold_call': '_oracle_reactant_state(e,h,z,1.)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'e=np.array([[[[1000.,1001.,999.]],[[0.,-30.,30.]]],[[[-1000.,-999.,-1001.]],[[4.,4.,4.]]]])\n'
               'h=np.array([0.,1.,2.]); z=h-2.',
      'call': 'reactant_state(e,h,z,1.)',
      'gold_call': '_oracle_reactant_state(e,h,z,1.)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'e=np.zeros((1,2,1,2)); h=np.array([0.,1.]); z=h-1.\n'
               '\n'
               'def _raises_value_error(fn, arguments):\n'
               '    try:\n'
               '        fn(*arguments)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_raises_value_error(reactant_state, (e,h,z,0.,))',
      'gold_call': '_raises_value_error(_oracle_reactant_state, (e,h,z,0.,))',
      'tol': 0.0}]
