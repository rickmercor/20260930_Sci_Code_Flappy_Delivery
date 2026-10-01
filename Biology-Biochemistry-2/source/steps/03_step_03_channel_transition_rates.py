"""
Step 3 - per-channel transition rates of the two-class kinetic strategy.

Each channel of the couplon gates between a closed state and an open state, and the contact-free class additionally visits two inactivated states. The opening and closing rates of every channel carry the treatment energy splitting: the contact energy change of step 02 is scaled by the distribution coefficient nu on the forward barrier and by its complement on the backward one, and for the voltage-coupled class the electrical pulse energy is split likewise with the coefficient nu_V. Inactivation rates are fixed by the measured half-times: open to I1 at ln(2)/3.5 ms^-1, I1 back to closed at ln(2)/20 ms^-1 and deeper to I2 at ln(2)/50 ms^-1 (which means there is two competing exits with the faster firing first), and I2 back to I1 at ln(2)/50 ms^-1. Only the contact-free class is inactivated, and only while open. This step returns the vector of rates out of a given local state!

Returns
-------
rates : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def channel_transition_rates(state: int, is_v: bool, delta_energy: float,
                             eps_v: float, nu: float = 1.0, nu_v: float = 0.8) -> np.ndarray:
    """Rates of every transition leaving one channel, indexed by destination state.

    Parameters
    ----------
    state : int
        Current state of the channel, encoded 0 = closed, 1 = open, 2 = first inactivated
        state, 3 = second inactivated state.
    is_v : bool
        True if the channel belongs to the class in contact with the voltage sensors.
    delta_energy : float
        Net contact energy, in kT, associated with this channel's closed-to-open transition
        given the current states of its neighbours.
    eps_v : float
        Electrical energy of the applied pulse, in kT, delivered to the sensor-coupled class.
    nu : float, optional
        Distribution coefficient for the contact energy.
    nu_v : float, optional
        Distribution coefficient for the electrical energy.

    Returns
    -------
    numpy.ndarray
        Array of shape (4,) of native floats, entry d holding the rate in ms^-1 of the
        transition from `state` to state d, and 0.0 wherever no such transition exists,
        including entry `state` itself.

    Raises
    ------
    ValueError
        If `state` is not one of 0, 1, 2, 3; if a sensor-coupled channel is given an
        inactivated state; or if `delta_energy`, `eps_v`, `nu` or `nu_v` is not finite.
    """
    rates = np.empty(4, dtype=float)
    return rates  # placeholder to complete!

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_channel_transition_rates(state, is_v, delta_energy, eps_v, nu=1.0, nu_v=0.8):
    """Reference implementation for channel_transition_rates."""
    import numpy as np

    # Table 1 constants of the matched treatment (ms^-1 and kT).
    K_PLUS, K_MINUS = 0.001, 2.0
    I1_HT, I2_HT, R1_HT, R2_HT = 3.5, 50.0, 20.0, 50.0
    N_STATES = 4
    if state not in (0, 1, 2, 3):
        raise ValueError("state must be one of 0, 1, 2, 3")
    is_v = bool(is_v)
    for name, value in (("delta_energy", delta_energy), ("eps_v", eps_v),
                        ("nu", nu), ("nu_v", nu_v)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if is_v and state in (2, 3):
        raise ValueError("sensor-coupled channels have no inactivated states")

    # Electrical energy acts on the sensor-coupled class only.
    e_v = float(eps_v) if is_v else 0.0
    rates = np.zeros(N_STATES, dtype=float)

    if state == 0:
        # Opening: both energies enter the forward barrier scaled by their coefficients.
        rates[1] = (K_PLUS * np.exp(-nu * delta_energy) * np.exp(-nu_v * e_v))
    elif state == 1:
        # Closing carries the complementary fractions with the sign of the reverse
        # direction, so that the ratio of the two rates reproduces the treatment's
        # equilibrium constant independently of either coefficient.
        rates[0] = (K_MINUS * np.exp((1.0 - nu) * delta_energy)
                    * np.exp((1.0 - nu_v) * e_v))
        if not is_v:
            rates[2] = np.log(2.0) / I1_HT
    elif state == 2:
        # Two exits compete from the first inactivated state; both are armed on entry and
        # the earlier one fires, which is a race of two exponentials.
        rates[0] = np.log(2.0) / R1_HT
        rates[3] = np.log(2.0) / I2_HT
    else:
        rates[2] = np.log(2.0) / R2_HT

    return rates

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\n"
    invalid = setup + (
        "def run_model(**kw):\n"
        "    try:\n"
        "        channel_transition_rates(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_channel_transition_rates(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: closed sensor-coupled channel mid-family, one open and two closed contacts.
        {"setup": setup,
         "call": "channel_transition_rates(0, True, -5.7 - 2 * -0.7, -8.0)",
         "gold_call": "_oracle_channel_transition_rates(0, True, -5.7 - 2 * -0.7, -8.0)"},
        # Normal: the discriminating case - an open sensor-coupled channel, whose forward and
        # backward rates together fix the equilibrium constant at this pulse energy.
        {"setup": setup,
         "call": "channel_transition_rates(1, True, 0.0, -8.0)",
         "gold_call": "_oracle_channel_transition_rates(1, True, 0.0, -8.0)"},
        # Normal: open contact-free channel, which alone may leave towards inactivation.
        {"setup": setup,
         "call": "channel_transition_rates(1, False, -5.7, -8.0)",
         "gold_call": "_oracle_channel_transition_rates(1, False, -5.7, -8.0)"},
        # Boundary: resting pulse energy, where the electrical factor is exactly one.
        {"setup": setup,
         "call": "channel_transition_rates(0, True, 2.1, 0.0)",
         "gold_call": "_oracle_channel_transition_rates(0, True, 2.1, 0.0)"},
        # Boundary: saturating pulse energy at the end of the supplied family.
        {"setup": setup,
         "call": "channel_transition_rates(1, True, -17.1, -14.0)",
         "gold_call": "_oracle_channel_transition_rates(1, True, -17.1, -14.0)"},
        # Edge: first inactivated state, the only state with two competing exits.
        {"setup": setup,
         "call": "channel_transition_rates(2, False, -5.7, -8.0)",
         "gold_call": "_oracle_channel_transition_rates(2, False, -5.7, -8.0)"},
        # Edge: second inactivated state, whose single exit returns towards the first.
        {"setup": setup,
         "call": "channel_transition_rates(3, False, 0.0, -14.0)",
         "gold_call": "_oracle_channel_transition_rates(3, False, 0.0, -14.0)"},
        # Edge: a contact-free channel carries no electrical factor at any pulse energy.
        {"setup": setup,
         "call": "channel_transition_rates(0, False, -5.7, -14.0)",
         "gold_call": "_oracle_channel_transition_rates(0, False, -5.7, -14.0)"},
        # Invalid: state outside the encoding.
        {"setup": invalid,
         "call": "run_model(state=4, is_v=False, delta_energy=0.0, eps_v=-8.0)",
         "gold_call": "run_gold(state=4, is_v=False, delta_energy=0.0, eps_v=-8.0)"},
        # Invalid: an inactivated state for a sensor-coupled channel.
        {"setup": invalid,
         "call": "run_model(state=2, is_v=True, delta_energy=0.0, eps_v=-8.0)",
         "gold_call": "run_gold(state=2, is_v=True, delta_energy=0.0, eps_v=-8.0)"},
        # Invalid: non-finite contact energy.
        {"setup": invalid,
         "call": "run_model(state=0, is_v=True, delta_energy=float('nan'), eps_v=-8.0)",
         "gold_call": "run_gold(state=0, is_v=True, delta_energy=float('nan'), eps_v=-8.0)"},
    ]
