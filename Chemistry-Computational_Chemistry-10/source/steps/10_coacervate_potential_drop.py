"""
Orchestrator: potential drop on the self-consistent binodal.

This step assembles the whole calculation. From the two chain lengths, the two linear charge densities (species 1 positively charged, species 2 negatively charged) and the electrostatic prefactor it performs the electroneutrality reduction, locates the critical point, builds the near-critical seed, advances the coexistence fixed-point map by the requested number of iterations, and evaluates the interphase electrostatic potential difference on the resulting pair of coexisting fractions.

Every earlier step is part of that chain. The reduction, the critical point, the near-critical seed, the seed operators and the iteration all feed the result directly. The scalar combinations are evaluated once more on the pair the iteration reaches and must agree with the closed-form recursion, and the free-energy density is checked against the two coexistence potentials through the Legendre relation that connects them, so a fault in any one of them stops the calculation rather than passing silently into the answer.

The iteration count is the order of the self-consistent solution. One means the pair obtained directly from the seed arguments, two means one further application of the map, and so on. The order must be at least one, since the raw seed is not itself a pair of physical fractions.

If the interaction parameter implied by the charge densities does not exceed its critical value the mixture does not separate at all and there are no coexisting phases to report.

Return the potential difference in units of thermal energy per elementary charge, dense phase minus dilute phase.

Returns
-------
float, the interphase potential difference in units of thermal energy per elementary charge, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coacervate_potential_drop(r1: float, r2: float, sigma1: float, sigma2: float,
                             alpha: float = 3.655, order: int = 2) -> float:
    """End-to-end interphase potential difference for a coacervating pair.

    Parameters
    ----------
    r1, r2 : float
        Chain lengths of species 1 and 2 in units of the reference volume, both > 0.
    sigma1, sigma2 : float
        Linear charge densities of species 1 and 2, both > 0.
    alpha : float
        Dimensionless electrostatic prefactor of the model.
    order : int
        Order of the self-consistent solution, that is the number of iterations of the
        coexistence fixed-point map counted from the near-critical seed. Must be >= 1.

    Returns
    -------
    dpsi : float
        Potential difference between the coexisting phases, dense minus dilute, in
        units of thermal energy per elementary charge.

    Raises
    ------
    ValueError
        If any argument is out of range, if order is not an integer >= 1, if the
        mixture does not phase separate, if the iteration leaves the physical range,
        or if the earlier steps it chains disagree with one another.
    """
    return dpsi  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_coacervate_potential_drop(r1: float, r2: float, sigma1: float, sigma2: float,
                             alpha: float = 3.655, order: int = 2) -> float:
    """Reference implementation."""
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
        raise ValueError("order must be an integer")
    n = int(order)
    if n < 1:
        raise ValueError("order must be at least 1")
    # chain the earlier sub-problems rather than reimplementing them
    u, S, chi, r_eff, gamma = _oracle_reduce_polyelectrolyte_pair(
        r1, r2, sigma1, sigma2, alpha)
    phi_c, chi_c = _oracle_critical_point(r_eff)
    if chi <= chi_c:
        raise ValueError("no phase separation: chi does not exceed its critical value")
    D = _oracle_landau_seed(phi_c, chi, r_eff)[3]
    state = _oracle_seed_operators(phi_c, chi, gamma, r_eff, D)
    for _ in range(n - 1):
        state = _oracle_advance_fixed_point(state[0], state[1], chi, gamma, r_eff)
    b, Hv = float(state[1]), float(state[2])
    if not np.isfinite(Hv) or Hv <= 0.0:
        raise ValueError("iterate has left the physical branch")
    phi_I = float(np.exp(b) * Hv)
    phi_II = float(np.exp(-b) * Hv)
    if not (0.0 < phi_II < phi_I < 1.0):
        raise ValueError("iterate is not a physical pair of volume fractions")
    # the exact scalar combinations evaluated on the pair the iteration has reached;
    # one further application of the closed-form recursion must reproduce them, which
    # is what ties the two ways of writing the same map together
    ops = _oracle_fixed_point_operators(phi_I, phi_II, chi, gamma, r_eff)
    nxt = _oracle_advance_fixed_point(state[0], state[1], chi, gamma, r_eff)
    if not np.allclose(np.asarray(ops)[2:], np.asarray(nxt), rtol=1e-9, atol=1e-12):
        raise ValueError("the two forms of the fixed-point map disagree")
    # the free-energy density and the two coexistence potentials are a Legendre pair,
    # so on each coexisting fraction the pressure must equal phi*(mu - gamma) - f
    for phi in (phi_I, phi_II):
        f_phi = _oracle_free_energy_density(phi, r_eff, chi)
        mu_phi, pi_phi = _oracle_coexistence_potentials(phi, r_eff, chi)
        if not np.isclose(pi_phi, phi * (mu_phi - gamma) - f_phi,
                          rtol=1e-9, atol=1e-12):
            raise ValueError("free energy and coexistence potentials are inconsistent")
    return _oracle_interphase_potential(phi_I, phi_II, r1, r2, sigma1, sigma2, alpha)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # reference system at the target order
            "setup": 'import numpy as np\nr1, r2, s1, s2 = 4.0, 2.0, 0.80, 0.85\n',
            "call": 'coacervate_potential_drop(r1, r2, s1, s2, 3.655, 2)',
            "gold_call": '_oracle_coacervate_potential_drop(r1, r2, s1, s2, 3.655, 2)',
        },
        {
            # reference system at first order
            "setup": 'import numpy as np\nr1, r2, s1, s2 = 4.0, 2.0, 0.80, 0.85\n',
            "call": 'coacervate_potential_drop(r1, r2, s1, s2, 3.655, 1)',
            "gold_call": '_oracle_coacervate_potential_drop(r1, r2, s1, s2, 3.655, 1)',
        },
        {
            # reference system at third order
            "setup": 'import numpy as np\nr1, r2, s1, s2 = 4.0, 2.0, 0.80, 0.85\n',
            "call": 'coacervate_potential_drop(r1, r2, s1, s2, 3.655, 3)',
            "gold_call": '_oracle_coacervate_potential_drop(r1, r2, s1, s2, 3.655, 3)',
        },
        {
            # reference system far into the iteration
            "setup": 'import numpy as np\nr1, r2, s1, s2 = 4.0, 2.0, 0.80, 0.85\n',
            "call": 'coacervate_potential_drop(r1, r2, s1, s2, 3.655, 8)',
            "gold_call": '_oracle_coacervate_potential_drop(r1, r2, s1, s2, 3.655, 8)',
        },
        {
            # symmetric pair gives exactly zero
            "setup": 'import numpy as np\nr1, r2, s1, s2 = 2.0, 2.0, 1.00, 1.00\n',
            "call": 'coacervate_potential_drop(r1, r2, s1, s2, 3.655, 2)',
            "gold_call": '_oracle_coacervate_potential_drop(r1, r2, s1, s2, 3.655, 2)',
        },
        {
            # a different asymmetric pair
            "setup": 'import numpy as np\nr1, r2, s1, s2 = 4.5, 1.5, 0.80, 0.95\n',
            "call": 'coacervate_potential_drop(r1, r2, s1, s2, 3.655, 2)',
            "gold_call": '_oracle_coacervate_potential_drop(r1, r2, s1, s2, 3.655, 2)',
        },
        {
            # swapping the two species reverses the sign
            "setup": 'import numpy as np\nr1, r2, s1, s2 = 2.0, 4.0, 0.85, 0.80\n',
            "call": 'coacervate_potential_drop(r1, r2, s1, s2, 3.655, 2)',
            "gold_call": '_oracle_coacervate_potential_drop(r1, r2, s1, s2, 3.655, 2)',
        },
        {
            # rejects order zero
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        coacervate_potential_drop(4.0, 2.0, 0.8, 0.85, 3.655, 0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_coacervate_potential_drop(4.0, 2.0, 0.8, 0.85, 3.655, 0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects negative order
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        coacervate_potential_drop(4.0, 2.0, 0.8, 0.85, 3.655, -2)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_coacervate_potential_drop(4.0, 2.0, 0.8, 0.85, 3.655, -2)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects non-integer order
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        coacervate_potential_drop(4.0, 2.0, 0.8, 0.85, 3.655, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_coacervate_potential_drop(4.0, 2.0, 0.8, 0.85, 3.655, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects charge densities too low for phase separation
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        coacervate_potential_drop(4.0, 2.0, 0.20, 0.25, 3.655, 2)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_coacervate_potential_drop(4.0, 2.0, 0.20, 0.25, 3.655, 2)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects zero charge density
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        coacervate_potential_drop(4.0, 2.0, 0.8, 0.0, 3.655, 2)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_coacervate_potential_drop(4.0, 2.0, 0.8, 0.0, 3.655, 2)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects negative chain length
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        coacervate_potential_drop(-4.0, 2.0, 0.8, 0.85, 3.655, 2)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_coacervate_potential_drop(-4.0, 2.0, 0.8, 0.85, 3.655, 2)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
