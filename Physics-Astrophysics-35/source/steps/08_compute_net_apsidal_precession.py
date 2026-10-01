"""
Orchestrator: compute the net orbit-averaged apsidal precession rate, in arcseconds per century, that a BAB Wisdom-Holman simulation in democratic heliocentric coordinates with the GR potential shows for the star-companion orbit.

The simulated orbit precesses for two reasons. The first is physical: the GR potential rotates it (Step 7). The second is artificial: the BAB map with step h does not solve the two-body problem exactly (Step 6). To leading order in h and in 1/c^2 the two rates add.

The timestep is given in minutes and must be converted to years (1 yr = 365.25 days). The result is reported in arcseconds per century, positive for prograde precession.

This step is the final orchestrator. Called with no arguments, it reproduces the benchmark: m0 = 1.0 Msun, m1 = 0.07 Msun, a = 0.06 au, e = 0.5 and h = 14 minutes.

Returns
-------
float, the net orbit-averaged apsidal precession rate (GR plus artificial) in arcseconds per century, positive for prograde.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_net_apsidal_precession(m0: float = 1.0, m1: float = 0.07, a: float = 0.06,
                                   e: float = 0.5, timestep_minutes: float = 14.0,
                                   n_samples: int = 512) -> float:
    """Net apsidal precession rate (GR plus artificial) of the simulated orbit, in arcsec / century.

    Parameters
    ----------
    m0 : float
        Mass of the star (Msun), a finite number > 0.
    m1 : float
        Mass of the companion (Msun), a finite number > 0.
    a : float
        Semimajor axis (au), a finite number > 0.
    e : float
        Eccentricity, a finite number with 0 < e < 1.
    timestep_minutes : float
        Map step size in minutes, a finite number > 0.
    n_samples : int
        Number of sample times used for the orbit average, an integer >= 16.

    Returns
    -------
    rate : float
        Net orbit-averaged apsidal precession rate in arcsec / century
        (positive = prograde).

    Raises
    ------
    ValueError
        If timestep_minutes is not a finite number > 0, or if any underlying
        step raises ValueError on its own inputs.
    """
    return rate  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_net_apsidal_precession(m0: float = 1.0, m1: float = 0.07, a: float = 0.06,
                                           e: float = 0.5, timestep_minutes: float = 14.0,
                                           n_samples: int = 512) -> float:
    import numpy as np

    if not (isinstance(timestep_minutes, (int, float, np.floating)) and np.isfinite(timestep_minutes)
            and float(timestep_minutes) > 0.0):
        raise ValueError("timestep_minutes must be a finite number > 0")
    h = float(timestep_minutes) / (60.0 * 24.0 * 365.25)            # years
    artificial = _oracle_orbit_averaged_artificial_precession(m0, m1, a, e, h, n_samples)
    gr = _oracle_gr_precession_rate(m0, m1, a, e)
    rad_per_yr_to_arcsec_per_century = 180.0 / np.pi * 3600.0 * 100.0
    return float((gr + artificial) * rad_per_yr_to_arcsec_per_century)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark, all default arguments ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_net_apsidal_precession()",
            "gold_call": "_oracle_compute_net_apsidal_precession()",
        },
        # --- Valid: a coarser step (20 minutes), where the artificial term exceeds GR and the net precession is retrograde ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_net_apsidal_precession(timestep_minutes=20.0)",
            "gold_call": "_oracle_compute_net_apsidal_precession(timestep_minutes=20.0)",
        },
        # --- Valid: a Jupiter-mass hot Jupiter (a = 0.05 au, e = 0.3) with a 1-hour step ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_net_apsidal_precession(m1=9.547919e-4, a=0.05, e=0.3, timestep_minutes=60.0)",
            "gold_call": "_oracle_compute_net_apsidal_precession(m1=9.547919e-4, a=0.05, e=0.3, timestep_minutes=60.0)",
        },
        # --- Edge: a very fine step (1 minute), where the result is essentially the GR rate ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_net_apsidal_precession(timestep_minutes=1.0, n_samples=256)",
            "gold_call": "_oracle_compute_net_apsidal_precession(timestep_minutes=1.0, n_samples=256)",
        },
        # --- Invalid: non-positive timestep ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_net_apsidal_precession(timestep_minutes=0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_net_apsidal_precession(timestep_minutes=0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: unbound orbit ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_net_apsidal_precession(e=1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_net_apsidal_precession(e=1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
