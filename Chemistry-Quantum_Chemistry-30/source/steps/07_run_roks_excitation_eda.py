"""
Orchestrate the complete ROKS excitation EDA on the printed fixture.

Validate the orbitals, form the ground-state density, construct the
frozen intermediate, form the final-state density, and return the
total OVOCV spectator promotion of the relaxation. Every earlier
return is consumed here.

Do not run a new ROKS or DFT job. Do not replace the tagged scalar by
a published table energy, by a table charge in me⁻, or by a primary
GS → ES promotion. The tagged value is this fixture's total spectator
OVOCV promotion.

Returns
-------
return Q_relax
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_roks_excitation_eda(
    S: np.ndarray,
    C_GS: np.ndarray,
    C_d_ES: np.ndarray,
    h_ES: np.ndarray,
    l_ES: np.ndarray,
) -> float:
    """
    Execute the complete ROKS excitation EDA pipeline.

    Parameters
    ----------
    S : np.ndarray
        AO overlap, shape (n, n).
    C_GS : np.ndarray
        Closed-shell ground-state occupied MO coefficients, shape (n, n_occ).
    C_d_ES : np.ndarray
        Final-state doubly occupied MO coefficients, shape (n, n_d).
    h_ES : np.ndarray
        First final-state SOMO coefficients, shape (n,).
    l_ES : np.ndarray
        Second final-state SOMO coefficients, shape (n,).

    Returns
    -------
    float
        Total spectator OVOCV promotion of the supplied instance.

    Raises
    ------
    ValueError
        If any input array is invalid or a pipeline stage returns a
        non-finite promotion.
    """
    return Q_relax

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_roks_excitation_eda(S: np.ndarray, C_GS: np.ndarray, C_d_ES: np.ndarray, h_ES: np.ndarray, l_ES: np.ndarray) -> float:
    n_occ = _oracle_validate_roks_orbitals(S, C_GS, C_d_ES, h_ES, l_ES)
    n_occ = int(np.asarray(n_occ).reshape(-1)[0])
    P_GS = np.asarray(_oracle_ground_state_density(C_GS), dtype=float)
    P_PRJ = np.asarray(
        _oracle_frozen_projection_density(P_GS, h_ES, l_ES, S), dtype=float
    )
    P_INT = np.asarray(
        _oracle_purify_intermediate_state(P_PRJ, h_ES, l_ES, S, n_occ),
        dtype=float,
    )
    P_ES = np.asarray(
        _oracle_excited_state_density(C_d_ES, h_ES, l_ES), dtype=float
    )
    if P_GS.shape != P_PRJ.shape or P_INT.shape != P_GS.shape:
        raise ValueError("densities must share one AO dimension")
    if P_ES.shape != P_GS.shape:
        raise ValueError("final-state density must match P_GS")
    dQ = np.asarray(
        _oracle_ovocv_relaxation_promotions(P_INT, P_ES, S), dtype=float
    ).reshape(-1)
    if dQ.size == 0 or not np.all(np.isfinite(dQ)):
        raise ValueError("OVOCV promotions must be finite")
    if np.any(dQ < -1e-12):
        raise ValueError("OVOCV promotions must be nonnegative")
    return float(np.sum(dQ))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np
S = np.array([
    [1.00, 0.14, 0.05, 0.02, 0.01, 0.00],
    [0.14, 1.00, 0.12, 0.04, 0.02, 0.01],
    [0.05, 0.12, 1.00, 0.10, 0.04, 0.02],
    [0.02, 0.04, 0.10, 1.00, 0.13, 0.05],
    [0.01, 0.02, 0.04, 0.13, 1.00, 0.11],
    [0.00, 0.01, 0.02, 0.05, 0.11, 1.00],
], dtype=float)
C_GS = np.array([
    [ 1.0080406599907743e+00, -6.9107734218542841e-02, -1.8590005312309209e-02],
    [-6.9107734218542841e-02,  1.0129364516936252e+00, -5.7436889827125644e-02],
    [-1.8590005312309157e-02, -5.7436889827125685e-02,  1.0098591410196720e+00],
    [-6.4935187113773817e-03, -1.4397439364686361e-02, -4.7260175428624082e-02],
    [-3.0063239218411234e-03, -6.3781347257252600e-03, -1.4208074758959877e-02],
    [ 1.2504076860857564e-03, -3.2331890567884867e-03, -6.8473280638734531e-03],
], dtype=float)
C_d_ES = np.array([
    [-4.6576117886276669e-01,  1.0639291920844979e-01],
    [ 9.0863740302918772e-01, -1.6814437758770151e-01],
    [ 1.5572227232294725e-01,  9.4660762156982658e-01],
    [ 1.3340540574226461e-02, -1.8170610498559844e-01],
    [-1.7641371855794502e-01,  2.2510340558592395e-01],
    [ 1.9650940743279659e-02, -2.3842184982255774e-01],
], dtype=float)
h_ES = np.array([
    8.0062071599233220e-01, 3.0231974389437583e-01, -5.6840749389178262e-02,
    4.3523291657322111e-01, -3.2531705238407906e-02, -8.7198934719083226e-03,
], dtype=float)
l_ES = np.array([
    -2.6573150693216712e-03, -1.4102687216989006e-02, 1.5210872916881013e-01,
    -4.3751913834973843e-02, 2.3206081914053003e-01, 9.3467599363415466e-01,
], dtype=float)
""",
            "call": "run_roks_excitation_eda(S, C_GS, C_d_ES, h_ES, l_ES)",
            "gold_call": "_oracle_run_roks_excitation_eda(S, C_GS, C_d_ES, h_ES, l_ES)",
        },
        {
            "setup": """import numpy as np
S = np.eye(5)
C_GS = np.eye(5)[:, :3]
C_d_ES = np.array([
    [1.0, 0.0],
    [0.0, np.cos(0.25)],
    [0.0, 0.0],
    [0.0, np.sin(0.25)],
    [0.0, 0.0],
], dtype=float)
h_ES = np.eye(5)[:, 2]
l_ES = np.eye(5)[:, 4]
""",
            "call": "run_roks_excitation_eda(S, C_GS, C_d_ES, h_ES, l_ES)",
            "gold_call": "_oracle_run_roks_excitation_eda(S, C_GS, C_d_ES, h_ES, l_ES)",
        },
        {
            "setup": """import numpy as np
S = np.eye(4)
C_GS = np.eye(4)[:, :2]
C_d_ES = np.eye(4)[:, :1].astype(float)
C_d_ES[0, 0] = np.nan
h_ES = np.eye(4)[:, 1]
l_ES = np.eye(4)[:, 2]

def run_model():
    try:
        run_roks_excitation_eda(S, C_GS, C_d_ES, h_ES, l_ES)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_run_roks_excitation_eda(S, C_GS, C_d_ES, h_ES, l_ES)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
S = np.eye(3)
C_GS = np.eye(3)[:, :2]
C_d_ES = np.eye(3)[:, :1]
h_ES = np.eye(3)[:, 0]
l_ES = np.eye(3)[:, 2]

def run_model():
    try:
        run_roks_excitation_eda(S, C_GS, C_d_ES, h_ES, l_ES)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_run_roks_excitation_eda(S, C_GS, C_d_ES, h_ES, l_ES)
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
