"""
Final orchestrator. The chain is: excitonic units from the reduced mass (step 1); the

averaged bare interaction of the chosen lateral distribution (step 2); the local form

factor N_f and the partially nonlocal form factor (step 3); the dielectric function and

the screened interaction (steps 4 and 5); the kernel integral and the variational energy

(steps 6 and 7); and the bounded maximisation over lambda (step 8). The orchestrator calls

every earlier step directly. It obtains the excitonic units, evaluates the bare

interaction, the nonlocal form factor, the dielectric function and the screened

interaction at a reference wave vector and verifies the defining relations

eps = 1 + alpha_1D q^2 chi V_bare and V eps = V_bare, runs the optimisation twice for the

same wire, once with the local form factor and once with the partially nonlocal one,

cross-checks each returned radius against the units through r_B lambda_0 = 3 a_exc / 2,

re-evaluates the kernel integral and the variational energy at each returned lambda_0

against the optimiser energy, and returns the nonlocality correction to the exciton

binding energy

Final orchestrator. The chain is: excitonic units from the reduced mass (step 1); the
averaged bare interaction of the chosen lateral distribution (step 2); the local form
factor N_f and the partially nonlocal form factor (step 3); the dielectric function and
the screened interaction (steps 4 and 5); the kernel integral and the variational energy
(steps 6 and 7); and the bounded maximisation over lambda (step 8). The orchestrator calls
every earlier step directly. It obtains the excitonic units, evaluates the bare
interaction, the nonlocal form factor, the dielectric function and the screened
interaction at a reference wave vector and verifies the defining relations
eps = 1 + alpha_1D q^2 chi V_bare and V eps = V_bare, runs the optimisation twice for the
same wire, once with the local form factor and once with the partially nonlocal one,
cross-checks each returned radius against the units through r_B lambda_0 = 3 a_exc / 2,
re-evaluates the kernel integral and the variational energy at each returned lambda_0
against the optimiser energy, and returns the nonlocality correction to the exciton
binding energy

  Delta E = E_B(nonlocal) - E_B(local)   in meV.

For the homogeneous distribution the nonlocal factor is below 1 at finite |q| R, which
weakens the screening and makes Delta E positive. The ribbon distribution has no nonlocal
variant and is rejected.

Returns
-------
float: the nonlocality correction Delta E = E_B(nonlocal) - E_B(local) in meV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nonlocality_correction(R: float, alpha: float, mu: float, distribution: str = 'homogeneous', z0: "float | None" = None) -> float:
    '''Nonlocality correction Delta E = E_B(nonlocal) - E_B(local) of the exciton binding energy in meV.

    Calls every earlier step directly. Obtains the excitonic units from exciton_units;
    evaluates bare_potential_fourier, nonlocal_form_factor, dielectric_function and
    screened_potential_fourier at the reference wave vector q = 1 per Angstrom and checks
    eps = 1 + alpha q^2 chi V_bare and V eps = V_bare; runs optimise_exciton twice, once
    with the local and once with the partially nonlocal form factor; cross-checks each
    returned radius against the units (r_B lambda_0 = 3 a_exc / 2); and re-evaluates
    variational_binding_energy and variational_kernel_integral at each returned lambda_0,
    requiring both to reproduce the optimiser energy through
    E_B = R_exc (-lambda_0^2 + lambda_0 (4/pi) I). Any failed check raises ValueError.

    Parameters
    ----------
    R : float
        Wire radius in Angstrom, > 0.
    alpha : float
        Static 1D polarizability in Angstrom^2, >= 0.
    mu : float
        Interband reduced mass in free-electron masses, > 0.
    distribution : str
        One of 'surface', 'homogeneous', 'centre' ('ribbon' is rejected).
    z0 : float or None
        Cutoff length of the 'centre' distribution in Angstrom; None selects 2 R / sqrt(5).

    Returns
    -------
    delta_e : float
        E_B(nonlocal) - E_B(local) in meV.
        Raises ValueError for the ribbon distribution or invalid wire parameters.
    '''
    return delta_e

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy import special as sp
from scipy import integrate, optimize


def _oracle_nonlocality_correction(R: float, alpha: float, mu: float, distribution: str = 'homogeneous', z0: "float | None" = None) -> float:
    if distribution == "ribbon":
        raise ValueError("the ribbon distribution has no nonlocal variant")
    units = _oracle_exciton_units(mu)
    q_ref = 1.0
    vb = float(_oracle_bare_potential_fourier(q_ref, R, distribution, z0))
    chi = float(_oracle_nonlocal_form_factor(q_ref, R, distribution, True, z0))
    eps = float(_oracle_dielectric_function(q_ref, R, alpha, distribution, True, z0))
    vs = float(_oracle_screened_potential_fourier(q_ref, R, alpha, distribution, True, z0))
    if abs(eps - (1.0 + alpha * q_ref**2 * chi * vb)) > 1e-9 * max(1.0, abs(eps)):
        raise ValueError("dielectric function is inconsistent with the form factor and the bare interaction")
    if abs(vs * eps - vb) > 1e-9 * max(1.0, abs(vb)):
        raise ValueError("screened interaction is inconsistent with the bare interaction")
    energies = []
    for use_nl in (False, True):
        out = _oracle_optimise_exciton(R, alpha, mu, distribution, use_nl, z0)
        lam0, e_b = float(out[0]), float(out[1])
        if abs(out[0] * out[2] - units[2]) > 1e-6:
            raise ValueError("optimiser radius is inconsistent with the excitonic units")
        e_re = float(_oracle_variational_binding_energy(lam0, R, alpha, mu, distribution, use_nl, z0))
        kern = float(_oracle_variational_kernel_integral(lam0, R, alpha, mu, distribution, use_nl, z0))
        e_fun = units[0] * (-lam0 * lam0 + lam0 * 4.0 / np.pi * kern)
        if abs(e_re - e_b) > 1e-8 or abs(e_fun - e_b) > 1e-8:
            raise ValueError("energy functional is inconsistent with the optimiser energy")
        energies.append(e_b)
    return 1000.0 * (energies[1] - energies[0])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np
from scipy import special as sp
from scipy import integrate, optimize


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
""",
            "call": "round(nonlocality_correction(2.28, 12.06, 0.39, 'homogeneous'), 2)",
            "gold_call": "round(_oracle_nonlocality_correction(2.28, 12.06, 0.39, 'homogeneous'), 2)",
        },
        {
            "setup": """import numpy as np
""",
            "call": "round(nonlocality_correction(2.28, 12.06, 0.39, 'centre'), 2)",
            "gold_call": "round(_oracle_nonlocality_correction(2.28, 12.06, 0.39, 'centre'), 2)",
        },
        {
            "setup": """import numpy as np
""",
            "call": "round(nonlocality_correction(2.13, 17.43, 0.30, 'surface'), 2)",
            "gold_call": "round(_oracle_nonlocality_correction(2.13, 17.43, 0.30, 'surface'), 2)",
        },
        {
            "setup": """import numpy as np
""",
            "call": "round(nonlocality_correction(2.28, 0.0, 0.39, 'homogeneous'), 6)",
            "gold_call": "round(_oracle_nonlocality_correction(2.28, 0.0, 0.39, 'homogeneous'), 6)",
        },
        {
            "setup": """import numpy as np
""",
            "call": "round(nonlocality_correction(4.02, 51.63, 0.11, 'centre', z0=4.02), 2)",
            "gold_call": "round(_oracle_nonlocality_correction(4.02, 51.63, 0.11, 'centre', z0=4.02), 2)",
        },
        {
            "setup": """import numpy as np


def run_model():
    try:
        nonlocality_correction(2.28, 12.06, 0.39, 'ribbon')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2


def run_gold():
    try:
        _oracle_nonlocality_correction(2.28, 12.06, 0.39, 'ribbon')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            "setup": """import numpy as np


def run_model():
    try:
        nonlocality_correction(0.0, 12.06, 0.39, 'homogeneous')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2


def run_gold():
    try:
        _oracle_nonlocality_correction(0.0, 12.06, 0.39, 'homogeneous')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
