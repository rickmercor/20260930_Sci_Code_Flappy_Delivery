"""
Builds a deterministic nine-system error panel from the paper-order validation metrics.

This support step keeps the nine molecular-crystal systems separate so later source-method
checks can expose compounds that would be hidden by an aggregate validation error.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_error_panel(seed: int = 58031, n_configs: int = 96) -> object:
    """Return the deterministic error panel.

    Parameters
    ----------
    seed : int
        Controls both the default_rng stream and the modular outlier selector.
    n_configs : int
        Number of configurations, at least 24.

    Returns
    -------
    object
        A float64 array with shape (n_configs, 9, 4), ordered as energy,
        force, joint normalized error, and leverage.

    Raises
    ------
    ValueError
        If seed is not an integer or n_configs is not an integer at least 24.

    Notes
    -----
    Create rng=default_rng(seed) and draw z once with independent standard-normal
    entries using rng.normal(0.0, 1.0, size=(n_configs, 9, 4)). Use the supplied
    seed in the modular outlier selector.

    Use these fixed source-table values in printed compound order, in
    kJ mol^-1 atom^-1 and kJ mol^-1 A^-1 respectively:
    e0 = [0.128, 0.069, 0.161, 0.159, 0.184, 0.116, 0.146, 0.158, 0.151]
    f0 = [0.762, 0.848, 0.414, 0.501, 1.043, 0.562, 0.695, 0.627, 0.377]
    The order is Benzoic acid, Benzamide, Coumarin, Durene, Isonicotinamide,
    Nicotinic acid, Niacinamide, Pyrazinamide, Resorcinol. Convert these
    reference scales to eV units with 1 eV = 96.48533212 kJ/mol before
    constructing the synthetic per-atom-energy and force-magnitude panel.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_error_panel(seed: int = 58031, n_configs: int = 96) -> object:
    ENERGY_MAE = np.array([0.128, 0.069, 0.161, 0.159, 0.184, 0.116, 0.146, 0.158, 0.151], dtype=np.float64)
    FORCE_MAE = np.array([0.762, 0.848, 0.414, 0.501, 1.043, 0.562, 0.695, 0.627, 0.377], dtype=np.float64)
    if not isinstance(seed, (int, np.integer)) or not isinstance(n_configs, (int, np.integer)) or n_configs < 24:
        raise ValueError("seed must be an integer and n_configs must be an integer >= 24")
    rng = np.random.default_rng(int(seed))
    z = rng.normal(size=(int(n_configs), 9, 4))
    i = np.arange(int(n_configs), dtype=np.float64)[:, None]
    j = np.arange(9, dtype=np.float64)[None, :]
    energy_scale = ENERGY_MAE / 96.48533212
    force_scale = FORCE_MAE / 96.48533212
    e = energy_scale[None, :] * (1.0 + 0.16 * z[:, :, 0]) + 0.0003 * np.sin((i + 1.0) * (j + 2.0) / 13.0)
    f = force_scale[None, :] * (1.0 + 0.13 * z[:, :, 1]) + 0.0012 * np.cos((i + 2.0) * (j + 1.0) / 17.0)
    e = np.abs(e)
    f = np.abs(f)
    selector = ((np.arange(int(n_configs))[:, None] * 7 + np.arange(9)[None, :] * 11 + int(seed)) % 53) == 0
    e = e + selector * (0.225 + 0.015 * (j % 3))
    f = f + selector * (10.2 + 0.3 * (j % 2))
    joint = np.sqrt((e / energy_scale[None, :]) ** 2 + (f / force_scale[None, :]) ** 2)
    leverage = 0.55 * z[:, :, 2] + 0.25 * z[:, :, 3] + 0.2 * np.sin((i + 1) * (j + 1) / 7.0)
    return np.stack([e, f, joint, leverage], axis=-1).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"", "call":"build_error_panel(58031,96)", "gold_call":"_oracle_build_error_panel(58031,96)", "tol":1e-9},
        {"setup":"", "call":"build_error_panel(58079,84)", "gold_call":"_oracle_build_error_panel(58079,84)", "tol":1e-9},
        {"setup":"", "call":"build_error_panel(58121,24)", "gold_call":"_oracle_build_error_panel(58121,24)", "tol":1e-9},
        {"setup":"", "call":"build_error_panel(58163,108)", "gold_call":"_oracle_build_error_panel(58163,108)", "tol":1e-9},
        {"setup":"def _candidate_probe():\n    try: build_error_panel(1,23); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef expected_probe():\n    try: _oracle_build_error_panel(1,23); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call":"_candidate_probe()", "gold_call":"expected_probe()", "tol":1e-9},
    ]
