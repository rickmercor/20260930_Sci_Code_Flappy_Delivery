"""
Implement coronal_heating_summary, the end-to-end pipeline (final step,

orchestrator).

For each region in the configuration this step chains the previous steps in

sequence: trace the field line from its footpoint (step 02), evaluate the

plasma background along it (step 03), form the Alfven speed and its

logarithmic gradient (step 04), measure the transverse deformation rate

(step 05), form the single rate per unit length that steps 06 and 07 both

take as input (they work in 1/cm, while step 05 returns 1/R_sun), transport

the outward amplitude (step 06), and evaluate the local heating rate

(step 07). The far

end of each line gives one end-of-line heating rate; the summary scalar is

the base-10 logarithm of the largest of them.



Inputs

------

config: dict with keys

    "regions": list of region_params dicts (step 01)

    "bg_params": background dict (step 03)

    "zp0_kms": float, footpoint amplitude in km/s

    "ell_max": float, arclength per line in R_sun

    "n_points": int, stations per line

    "h_fd": float, differencing half-step in R_sun

    "r_sun_cm": float, R_sun in cm



Returns

-------

summary: float, log10 of the maximum end-of-line heating rate over the

    regions (heating rates in erg cm^-3 s^-1); each line starts on its

    region axis at (0, 0, z_foot)

Returns
-------
summary : float     log10 of the largest end-of-line heating rate in erg cm^-3 s^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def coronal_heating_summary(config: dict) -> float:
    '''End-to-end heating summary over all configured regions.

    Parameters
    ----------
    config : dict
        Keys "regions" (non-empty list of region parameter dicts),
        "bg_params" (background parameter dict), "zp0_kms" (float > 0),
        "ell_max" (float > 0, R_sun), "n_points" (int >= 3), "h_fd"
        (float > 0, R_sun), "r_sun_cm" (float > 0).

    Returns
    -------
    summary : float
        log10 of the largest end-of-line heating rate in erg cm^-3 s^-1.

    Raises
    ------
    ValueError
        If config is missing any of the seven required keys, if
        config["regions"] is not a non-empty list, if zp0_kms or r_sun_cm is
        not a positive finite number, or if n_points is less than 3. Values
        rejected by an earlier step propagate that step's ValueError.
    '''
    return summary  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_coronal_heating_summary(config: dict) -> float:
    for key in (
        "regions", "bg_params", "zp0_kms", "ell_max",
        "n_points", "h_fd", "r_sun_cm"
    ):
        if key not in config:
            raise ValueError(f"config missing key {key!r}")

    regions = config["regions"]
    if not isinstance(regions, list) or len(regions) == 0:
        raise ValueError("config['regions'] must be a non-empty list")

    zp0 = float(config["zp0_kms"])
    if not np.isfinite(zp0) or zp0 <= 0.0:
        raise ValueError("zp0_kms must be a positive finite number")

    n_points = int(config["n_points"])
    if n_points < 3:
        raise ValueError("n_points must be >= 3")

    rsun = float(config["r_sun_cm"])
    if not np.isfinite(rsun) or rsun <= 0.0:
        raise ValueError("r_sun_cm must be a positive finite number")

    ell_max = float(config["ell_max"])
    h_fd = float(config["h_fd"])
    bg = config["bg_params"]

    q_end = []
    for rp in regions:
        # Validate the region through Step 01 before accessing z_foot.
        _oracle_evaluate_field(np.empty((0, 3), dtype=float), rp)

        seed = np.array([0.0, 0.0, float(rp["z_foot"])])
        pos = _oracle_trace_field_line(rp, seed, ell_max, n_points)
        s_cm = np.linspace(0.0, ell_max, n_points) * rsun

        back = _oracle_plasma_background(pos, bg)
        rho, u_cms = back[:, 0], back[:, 1]

        bmag = np.linalg.norm(
            _oracle_evaluate_field(pos, rp),
            axis=1
        )
        alf = _oracle_alfven_speed_gradient(
            bmag,
            rho,
            s_cm[1] - s_cm[0]
        )
        va, kva = alf[:, 0], alf[:, 1]

        smag = _oracle_deformation_rate(rp, pos, h_fd) / rsun
        eta = 0.5 * np.abs(kva) + smag

        zp = _oracle_wave_amplitude(
            zp0 * 1.0e5,
            va,
            u_cms,
            eta,
            s_cm
        )
        q = _oracle_heating_rate(rho, zp, u_cms, va, eta)
        q_end.append(float(q[-1]))

    return float(np.log10(max(q_end)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the full three-region configuration of the problem ---
        {
            "setup": """import numpy as np
shared = {"B0": 3.0, "hB": 0.8, "aB": 1.9, "z_foot": 1.0}
config = {
    "regions": [
        dict(shared, p0=0.06, up=8.0, wp=1.2, q0=0.03, uq=7.0, wq=1.5),
        dict(shared, p0=0.12, up=3.0, wp=1.0, q0=0.07, uq=3.5, wq=1.0),
        dict(shared, p0=0.0, up=0.0, wp=1.0, q0=0.0, uq=0.0, wq=1.0),
    ],
    "bg_params": {"rho0": 2e-16, "r0": 1.0, "alpha_rho": 4.0,
                  "u0": 10.0, "uinf": 650.0, "lu": 3.0},
    "zp0_kms": 30.0, "ell_max": 8.0, "n_points": 2001,
    "h_fd": 2e-5, "r_sun_cm": 6.957e10,
}
""",
            "call": "coronal_heating_summary(config)",
            "gold_call": "_oracle_coronal_heating_summary(config)",
        },
        # --- Normal: a different region set (generalization) ---
        {
            "setup": """import numpy as np
shared = {"B0": 2.0, "hB": 1.0, "aB": 2.1, "z_foot": 1.0}
config = {
    "regions": [
        dict(shared, p0=0.04, up=5.0, wp=2.0, q0=0.02, uq=6.0, wq=1.5),
        dict(shared, p0=0.0, up=0.0, wp=1.0, q0=0.0, uq=0.0, wq=1.0),
    ],
    "bg_params": {"rho0": 1e-16, "r0": 1.0, "alpha_rho": 3.8,
                  "u0": 15.0, "uinf": 550.0, "lu": 2.5},
    "zp0_kms": 25.0, "ell_max": 6.0, "n_points": 1201,
    "h_fd": 5e-5, "r_sun_cm": 6.957e10,
}
""",
            "call": "coronal_heating_summary(config)",
            "gold_call": "_oracle_coronal_heating_summary(config)",
        },
        # --- Boundary: single purely axial region (parallel-only limit) ---
        {
            "setup": """import numpy as np
config = {
    "regions": [{"B0": 3.0, "hB": 0.8, "aB": 1.9, "z_foot": 1.0,
                 "p0": 0.0, "up": 0.0, "wp": 1.0, "q0": 0.0, "uq": 0.0, "wq": 1.0}],
    "bg_params": {"rho0": 2e-16, "r0": 1.0, "alpha_rho": 4.0,
                  "u0": 10.0, "uinf": 650.0, "lu": 3.0},
    "zp0_kms": 30.0, "ell_max": 8.0, "n_points": 1001,
    "h_fd": 2e-5, "r_sun_cm": 6.957e10,
}
""",
            "call": "coronal_heating_summary(config)",
            "gold_call": "_oracle_coronal_heating_summary(config)",
        },
        # --- Invalid: empty region list ---
        {
            "setup": """import numpy as np
config = {
    "regions": [],
    "bg_params": {"rho0": 2e-16, "r0": 1.0, "alpha_rho": 4.0,
                  "u0": 10.0, "uinf": 650.0, "lu": 3.0},
    "zp0_kms": 30.0, "ell_max": 8.0, "n_points": 1001,
    "h_fd": 2e-5, "r_sun_cm": 6.957e10,
}
def run_model():
    try:
        coronal_heating_summary(config)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_coronal_heating_summary(config)
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
