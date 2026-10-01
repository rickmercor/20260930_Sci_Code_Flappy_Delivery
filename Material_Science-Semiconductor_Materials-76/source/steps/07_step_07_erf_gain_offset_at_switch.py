"""
Step 07 - Gain maximum minus effective Rabi frequency at the order change (orchestrator).

Distance between the parametric-gain maximum and the effective Rabi frequency at the pump where the predicted harmonic order first changes (orchestrator).

Self-starting harmonic frequency combs in fast-gain lasers have been interpreted as a resonance between an intrinsic oscillation frequency of the gain medium, the effective Rabi frequency (ERF, step 02), and a cavity mode, with the ERF expected to sit close to the maximum of the parametric gain of the continuous wave. This step tests that picture quantitatively at one well-defined operating point of a ring laser.

The operating point is the pump mu_star at which the comb order predicted by linear stability first changes above the multimode threshold (step 06, cavity sidebands 1 ... n_max, pump window [mu_min, mu_max]). At mu_star:

- the ERF is evaluated for the intensity X of the fundamental (k = 0) continuous wave at that pump (steps 01 and 02) and converted to an ordinary frequency in GHz by dividing the scaled value by 2 pi tau_d;

- the maximum of the parametric gain of the same wave, over sideband offsets f_max/4000 <= f <= f_max, is located as in step 04.

The returned number is the frequency of the gain maximum minus the ERF, in GHz; it is positive when the gain maximum lies above the ERF.

Returns
-------
float, frequency of the parametric-gain maximum minus the ERF at mu_star, in GHz
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def erf_gain_offset_at_switch(L_mm: float, n_group: float, tau_d_ps: float, Gamma: float, alpha: float, sigma: float, b: float, n_max: int, mu_min: float, mu_max: float, f_max_ghz: float) -> float:
    '''Gain-maximum frequency minus effective Rabi frequency at the first order change.

    Parameters
    ----------
    L_mm : float
        Ring length in millimetres, > 0.
    n_group : float
        Group index, > 0.
    tau_d_ps : float
        Polarisation dephasing time in picoseconds, > 0.
    Gamma, alpha, sigma, b : float
        Scaled gain bandwidth (> 0), linewidth enhancement factor, field loss
        rate (> 0) and carrier recovery rate (> 0).
    n_max : int
        Highest cavity sideband index considered, >= 2.
    mu_min, mu_max : float
        Pump window of step 06.
    f_max_ghz : float
        Upper end of the sideband offsets searched for the gain maximum, GHz.

    Returns
    -------
    offset : float
        f(gain maximum) - ERF at mu_star, in GHz.

    Raises
    ------
    ValueError
        For any condition under which steps 01, 02, 04 or 06 raise for these
        arguments.
    '''
    return offset

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_erf_gain_offset_at_switch(L_mm: float, n_group: float, tau_d_ps: float, Gamma: float, alpha: float, sigma: float, b: float, n_max: int, mu_min: float, mu_max: float, f_max_ghz: float) -> float:
    switch = _oracle_harmonic_order_switch(L_mm, n_group, tau_d_ps, Gamma, alpha, sigma, b, n_max, mu_min, mu_max)
    mu_star = float(switch[2])
    x = float(_oracle_cw_emission_state(0.0, Gamma, alpha, sigma, mu_star)[2])
    erf_scaled = float(_oracle_effective_rabi_frequency(x, Gamma, alpha, b)[0])
    erf_ghz = erf_scaled / (2.0 * np.pi * float(tau_d_ps) * 1e-12) * 1e-9
    peak = _oracle_parametric_gain_maximum(Gamma, alpha, sigma, b, mu_star, tau_d_ps, f_max_ghz)
    return float(peak[0] - erf_ghz)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: 4.7 mm terahertz ring, alpha below one ---
        {
            "setup": "import numpy as np\n",
            "call": "erf_gain_offset_at_switch(4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 16, 2.0, 12.0, 400.0)",
            "gold_call": "_oracle_erf_gain_offset_at_switch(4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 16, 2.0, 12.0, 400.0)",
            "tol": 1e-5,
        },
        # --- Boundary: long ring, the order changes shortly after the threshold ---
        {
            "setup": "import numpy as np\n",
            "call": "erf_gain_offset_at_switch(6.0, 3.5, 0.1, 0.06, 0.93, 1.6e-3, 0.012, 20, 2.0, 12.0, 400.0)",
            "gold_call": "_oracle_erf_gain_offset_at_switch(6.0, 3.5, 0.1, 0.06, 0.93, 1.6e-3, 0.012, 20, 2.0, 12.0, 400.0)",
            "tol": 1e-5,
        },
        # --- Edge: alpha close to one, different group index and loss ---
        {
            "setup": "import numpy as np\n",
            "call": "erf_gain_offset_at_switch(3.2, 3.3, 0.1, 0.06, 0.98, 2.0e-3, 0.016, 14, 2.0, 12.0, 400.0)",
            "gold_call": "_oracle_erf_gain_offset_at_switch(3.2, 3.3, 0.1, 0.06, 0.98, 2.0e-3, 0.016, 14, 2.0, 12.0, 400.0)",
            "tol": 1e-5,
        },
        # --- Invalid: the pump window closes before any sideband grows ---
        {
            "setup": "import numpy as np\n"
                     "def run_model():\n"
                     "    try:\n"
                     "        erf_gain_offset_at_switch(4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 16, 2.0, 7.0, 400.0)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "def run_gold():\n"
                     "    try:\n"
                     "        _oracle_erf_gain_offset_at_switch(4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 16, 2.0, 7.0, 400.0)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {   # a 3.9 mm ring at alpha = 0.92 with the larger carrier rate
            "setup": 'import numpy as np',
            "call": "erf_gain_offset_at_switch(3.9, 3.6, 0.1, 0.06, 0.92, 1.6e-3, 0.028, 16, 2.0, 12.0, 400.0)",
            "gold_call": "_oracle_erf_gain_offset_at_switch(3.9, 3.6, 0.1, 0.06, 0.92, 1.6e-3, 0.028, 16, 2.0, 12.0, 400.0)",
            "tol": 1e-5,
        },
        {   # a 4.0 mm ring at alpha = 0.88 with the faster carrier rate
            "setup": 'import numpy as np',
            "call": "erf_gain_offset_at_switch(4.0, 3.5, 0.1, 0.06, 0.88, 1.6e-3, 0.018, 16, 2.0, 12.0, 400.0)",
            "gold_call": "_oracle_erf_gain_offset_at_switch(4.0, 3.5, 0.1, 0.06, 0.88, 1.6e-3, 0.018, 16, 2.0, 12.0, 400.0)",
            "tol": 1e-5,
        },
        {   # a 6.2 mm ring at alpha = 0.95 and the larger loss
            "setup": 'import numpy as np',
            "call": "erf_gain_offset_at_switch(6.2, 3.6, 0.1, 0.06, 0.95, 2.5e-3, 0.014, 16, 2.0, 12.0, 400.0)",
            "gold_call": "_oracle_erf_gain_offset_at_switch(6.2, 3.6, 0.1, 0.06, 0.95, 2.5e-3, 0.014, 16, 2.0, 12.0, 400.0)",
            "tol": 1e-5,
        },
        # --- Normal: ring whose threshold sideband stops growing before the order changes ---
        {
            "setup": "import numpy as np\n",
            "call": "erf_gain_offset_at_switch(4.2, 3.6, 0.1, 0.06, 0.96, 1.6e-3, 0.012, 16, 2.0, 12.0, 400.0)",
            "gold_call": "_oracle_erf_gain_offset_at_switch(4.2, 3.6, 0.1, 0.06, 0.96, 1.6e-3, 0.012, 16, 2.0, 12.0, 400.0)",
            "tol": 1e-6,
        },
        # --- Edge: ring whose predicted order falls at the first change ---
        {
            "setup": "import numpy as np\n",
            "call": "erf_gain_offset_at_switch(3.0, 3.6, 0.1, 0.06, 0.94, 1.6e-3, 0.030, 16, 2.0, 12.0, 400.0)",
            "gold_call": "_oracle_erf_gain_offset_at_switch(3.0, 3.6, 0.1, 0.06, 0.94, 1.6e-3, 0.030, 16, 2.0, 12.0, 400.0)",
            "tol": 1e-6,
        },
    ]
