"""
Salt concentrations, below a given upper limit, at which the binding mode holding the most cations changes as salt is added.

Salt concentrations at which the predominant binding mode of a multivalent cation changes.

For a z-valent cation (z >= 2) the equilibrium state of the swollen, doped complex distributes the bound
cations over the binding modes k = 1..z (k = 1 mixed, k = z fully bridged). A cation bound in mode k
occupies k polyanion repeat units, so mode k holds xi_k / k cations per repeat unit, where xi_k is its
equilibrium site fraction. At a given salt concentration the predominant mode is the mode holding the
most cations. As the salt concentration of the bath rises from zero the predominant mode changes at a
sequence of salt concentrations, and at each of them the outgoing and the incoming mode hold equal
numbers of cations. Depending on the binding free energies some modes are never predominant.

Return every salt concentration C with 0 < C <= c_max at which the predominant mode changes, in
increasing order, each with a relative accuracy of about 1e-7. No search interval is supplied beyond
c_max, and the changes can lie anywhere from well below one millimolar to several molar. If the
predominant mode does not change up to c_max the result is an empty array.

Returns
-------
np.ndarray of shape (n,): increasing salt concentrations in mol/L at which the predominant binding mode changes (empty if none up to c_max)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dominant_mode_transitions(c_max: float, valence: int, ions: "np.ndarray", delta_g: "np.ndarray", chi: float,
                              polymer_conc: float, omega_p: float, n_p: float, entanglement: float,
                              crosslink: float, temperature: float, dielectric: float) -> "np.ndarray":
    '''Salt concentrations below c_max at which the binding mode holding the most cations changes.

    Parameters
    ----------
    c_max : float
        Largest salt concentration considered, mol/L.
    valence : int
        Cation charge number z (at least 2).
    ions : np.ndarray
        Shape (2, 2): [[cation radius, cation hydration number], [anion radius, anion hydration number]].
    delta_g : np.ndarray
        Shape (z,): standard free energies of modes k = 1..z in units of k_B T.
    chi : float
        Polymer-water Flory-Huggins parameter.
    polymer_conc : float
        Repeat-unit concentration C_P0 before phase separation, mol/L.
    omega_p : float
        Size of a repeat unit relative to water.
    n_p : float
        Number of repeat units per chain.
    entanglement : float
        Entanglement group v_P N_e alpha_tube (b/a_0)^2.
    crosslink : float
        Crosslink group v_P N_+-.
    temperature : float
        Absolute temperature in K.
    dielectric : float
        Relative permittivity used in the Bjerrum length.

    Returns
    -------
    result : np.ndarray
        Shape (n,), n >= 0: the salt concentrations in mol/L at which the predominant mode changes, in
        increasing order.

    Raises
    ------
    ValueError
        If valence is not an integer >= 2, c_max is not positive, delta_g does not have shape (valence,),
        or any input is invalid for the equilibrium doping of the complex.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_dominant_mode_transitions(c_max: float, valence: int, ions: "np.ndarray", delta_g: "np.ndarray", chi: float,
                                      polymer_conc: float, omega_p: float, n_p: float, entanglement: float,
                                      crosslink: float, temperature: float, dielectric: float) -> "np.ndarray":
    import numpy as np
    from scipy.optimize import brentq
    if isinstance(valence, bool) or not float(valence).is_integer() or int(valence) < 2:
        raise ValueError("valence must be an integer >= 2")
    if not c_max > 0.0:
        raise ValueError("c_max must be positive")
    z = int(valence)
    dg = np.array(delta_g, dtype=float)
    if dg.shape != (z,):
        raise ValueError("delta_g must have shape (valence,)")
    # From the mass-action laws, ln(xi_k / k) = k (1 - dG_k) + k ln R + (terms common to every mode), with
    # R = (1 - xi_PS) / (xi_PS Phi_P): the correlation term z (l_B / (pi l)) I and the free-ion activities are
    # shared by all modes. The predominant mode is the upper envelope of these lines in y = ln R, and y falls
    # monotonically as salt is added, so the changes are fixed values of ln R located on the equilibrium branch.
    kk = np.arange(1, z + 1)
    offset = kk * (1.0 - dg)
    changes = []
    current = z
    while current > 1:
        y_over = np.array([(offset[j - 1] - offset[current - 1]) / (current - j) for j in range(1, current)])
        y_next = float(y_over.max())
        nxt = int(np.nonzero(y_over == y_next)[0].min()) + 1
        changes.append(y_next)
        current = nxt

    def _ln_r(u):
        state = _oracle_equilibrium_doping(float(np.exp(u)), z, ions, dg, chi, polymer_conc, omega_p, n_p,
                                           entanglement, crosslink, temperature, dielectric)
        s = float(np.sum(state[:z]))
        return float(np.log1p(-s) - np.log(s) - np.log(state[z]))

    u_max = float(np.log(c_max))
    y_at_max = _ln_r(u_max)
    inside = [y for y in changes if y >= y_at_max]
    found = []
    u_floor = None
    for y_change in inside:
        if u_floor is None:
            # lowest change: walk down from c_max in decades until ln R lies above the change
            u_high = u_max
            u_low = u_max
            while True:
                u_low -= np.log(10.0)
                if u_low < np.log(1e-30):
                    raise ValueError("no change of the predominant mode above 1e-30 mol/L")
                if _ln_r(u_low) - y_change > 0.0:
                    break
                u_high = u_low
        else:
            # later changes lie between the previous change and c_max
            u_low, u_high = u_floor, u_max
        root = brentq(lambda u: _ln_r(u) - y_change, u_low, u_high, xtol=1e-14, rtol=1e-15, maxiter=500)
        u_floor = root
        found.append(float(np.exp(root)))
    return np.array(found, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal: divalent calcium, a single change from bridged to mixed binding
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.81, 2.0]])
dg = np.array([-4.2, -0.1])
""",
            "call": "dominant_mode_transitions(1.0, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "gold_call": "_oracle_dominant_mode_transitions(1.0, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "tol": 1.4e-8,
        },
        # boundary: strongly hydrated divalent cation, the single change lies above 1 M
        {
            "setup": """import numpy as np
ions = np.array([[0.72, 10.0], [1.81, 2.0]])
dg = np.array([-2.6, -0.7])
""",
            "call": "dominant_mode_transitions(3.0, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "gold_call": "_oracle_dominant_mode_transitions(3.0, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "tol": 1e-7,
        },
        # normal: trivalent cation whose intermediate mode is predominant over a wide salt range
        {
            "setup": """import numpy as np
ions = np.array([[1.03, 9.0], [1.81, 2.0]])
dg = np.array([-3.0, -1.5, -0.5])
""",
            "call": "dominant_mode_transitions(3.0, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)",
            "gold_call": "_oracle_dominant_mode_transitions(3.0, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)",
            "tol": 7e-8,
        },
        # edge: trivalent cation close to a triple point, intermediate mode predominant over about 0.1 % in salt
        {
            "setup": """import numpy as np
ions = np.array([[1.03, 9.0], [1.81, 2.0]])
dg = np.array([-3.0, -1.1255, -0.5])
""",
            "call": "dominant_mode_transitions(3.0, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)",
            "gold_call": "_oracle_dominant_mode_transitions(3.0, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)",
            "tol": 1e-7,
        },
        # edge: trivalent cation whose intermediate mode is never predominant
        {
            "setup": """import numpy as np
ions = np.array([[1.03, 9.0], [1.81, 2.0]])
dg = np.array([-3.0, -0.8, -0.5])
""",
            "call": "dominant_mode_transitions(3.0, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)",
            "gold_call": "_oracle_dominant_mode_transitions(3.0, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)",
            "tol": 1e-7,
        },
        # normal: tetravalent cation in which one of the four modes is never predominant
        {
            "setup": """import numpy as np
ions = np.array([[0.94, 12.0], [1.81, 2.0]])
dg = np.array([-5.0, -3.0, -2.0, -1.5])
""",
            "call": "dominant_mode_transitions(3.0, 4, ions.copy(), dg.copy(), 0.4, 0.05, 6.5, 1500.0, 1.0, 1.0, 298.15, 78.0)",
            "gold_call": "_oracle_dominant_mode_transitions(3.0, 4, ions.copy(), dg.copy(), 0.4, 0.05, 6.5, 1500.0, 1.0, 1.0, 298.15, 78.0)",
            "tol": 8e-8,
        },
        # edge: tetravalent cation passing through all four modes, the second change close to the first
        {
            "setup": """import numpy as np
ions = np.array([[0.94, 12.0], [1.81, 2.0]])
dg = np.array([-6.0, -3.3, -2.1003, -1.5])
""",
            "call": "dominant_mode_transitions(3.0, 4, ions.copy(), dg.copy(), 0.4, 0.05, 6.5, 1500.0, 1.0, 1.0, 298.15, 78.0)",
            "gold_call": "_oracle_dominant_mode_transitions(3.0, 4, ions.copy(), dg.copy(), 0.4, 0.05, 6.5, 1500.0, 1.0, 1.0, 298.15, 78.0)",
            "tol": 6e-8,
        },
        # boundary: trivalent cation below c_max only leaves the fully bridged mode
        {
            "setup": """import numpy as np
ions = np.array([[1.03, 9.0], [1.81, 2.0]])
dg = np.array([-3.0, -1.5, -0.5])
""",
            "call": "dominant_mode_transitions(1.0, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)",
            "gold_call": "_oracle_dominant_mode_transitions(1.0, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)",
            "tol": 7e-8,
        },
        # boundary: strongly bound bromide salt, the change falls below one millimolar
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.96, 1.5]])
dg = np.array([-13.0, -2.5])
""",
            "call": "dominant_mode_transitions(2.0, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "gold_call": "_oracle_dominant_mode_transitions(2.0, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "tol": 4e-11,
        },
        # edge: weakly binding short-chain complex, no change up to 3 M (empty result)
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.81, 2.0]])
dg = np.array([1.0, -0.5])
""",
            "call": "dominant_mode_transitions(3.0, 2, ions.copy(), dg.copy(), 0.1, 0.05, 3.5, 120.0, 1.0, 0.1, 298.15, 79.0)",
            "gold_call": "_oracle_dominant_mode_transitions(3.0, 2, ions.copy(), dg.copy(), 0.1, 0.05, 3.5, 120.0, 1.0, 0.1, 298.15, 79.0)",
            "tol": 1e-9,
        },
        # invalid: delta_g does not have one entry per mode
        {
            "setup": """import numpy as np
ions = np.array([[1.03, 9.0], [1.81, 2.0]])
dg = np.array([-3.0, -1.5])
def run_model():
    try:
        dominant_mode_transitions(3.0, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_dominant_mode_transitions(3.0, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
