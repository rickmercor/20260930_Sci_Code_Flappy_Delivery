#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


# Oracle implementation for public function: arm_emission_ratios
def arm_emission_ratios(g2: float, nbar: float) -> "np.ndarray":
    if g2 < 0.0:
        raise ValueError("g2 must be non-negative")
    if nbar <= 0.0:
        raise ValueError("nbar must be positive")
    p1 = nbar * (1.0 - g2 * nbar)
    p2 = 0.5 * g2 * nbar ** 2
    p0 = 1.0 - p1 - p2
    if p1 <= 0.0 or p0 < 0.0:
        raise ValueError("(g2, nbar) do not define a valid two-photon-truncated distribution")
    return np.array([p2 / p1, p0 / p1])

import numpy as np


# Oracle implementation for public function: effective_indistinguishability
def effective_indistinguishability(eta: float, m_sn: float) -> float:
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must lie in [0, 1]")
    if not (0.0 <= m_sn <= 1.0):
        raise ValueError("m_sn must lie in [0, 1]")
    return float((1.0 + 3.0 * m_sn) / (2.0 * (1.0 + m_sn)) * eta)

import numpy as np


# Oracle implementation for public function: cnot_event_table
def cnot_event_table(eta: float, m_sn: float) -> "np.ndarray":
    e_eff = effective_indistinguishability(eta, m_sn)
    tr11 = (2.0 - eta) / 9.0
    ov11 = (1.0 + eta) / 18.0
    tr_pair = 2.0 / 9.0
    tr3 = 2.0 * (3.0 - e_eff) / 9.0
    ov3 = (1.0 + e_eff) / 9.0
    return np.array([
        [tr11, ov11],
        [tr_pair, 0.0],
        [tr_pair, 0.0],
        [tr3, ov3],
        [tr3, ov3],
    ])

import numpy as np


# Oracle implementation for public function: hom_event_rates
def hom_event_rates(eta: float, m_sn: float) -> "np.ndarray":
    e_eff = effective_indistinguishability(eta, m_sn)
    return np.array([0.5 * (1.0 - eta), 0.5, 1.5 - e_eff])

import numpy as np


# Oracle implementation for public function: hom_visibility
def hom_visibility(eta: float, m_sn: float, g2_m: float, nbar_m: float,
                           g2_s: float, nbar_s: float) -> float:
    w_m, r_m = arm_emission_ratios(g2_m, nbar_m)
    w_s, r_s = arm_emission_ratios(g2_s, nbar_s)
    event_weights = np.array([1.0, w_m * r_s + w_s * r_m, w_m + w_s])
    rate = float(event_weights @ hom_event_rates(eta, m_sn))
    rate_delayed = float(event_weights @ hom_event_rates(0.0, m_sn))
    return 1.0 - rate / rate_delayed

import numpy as np


# Oracle implementation for public function: decode_intrinsic_eta
def decode_intrinsic_eta(v: float, m_sn: float, g2_m: float, nbar_m: float,
                                 g2_s: float, nbar_s: float) -> float:
    if not (0.0 < v <= 1.0):
        raise ValueError("visibility must lie in (0, 1]")
    v_unit = hom_visibility(1.0, m_sn, g2_m, nbar_m, g2_s, nbar_s)
    eta = v / v_unit
    if eta > 1.0:
        raise ValueError("decoded intrinsic indistinguishability exceeds 1")
    return float(eta)

import numpy as np


# Oracle implementation for public function: cnot_gate_fidelity
def cnot_gate_fidelity(eta: float, m_sn: float, g2_m: float, nbar_m: float,
                               g2_s: float, nbar_s: float) -> float:
    w_m, r_m = arm_emission_ratios(g2_m, nbar_m)
    w_s, r_s = arm_emission_ratios(g2_s, nbar_s)
    table = cnot_event_table(eta, m_sn)
    event_weights = np.array([1.0, w_m * r_s, w_s * r_m, w_m, w_s])
    return float((event_weights @ table[:, 1]) / (event_weights @ table[:, 0]))

import numpy as np


# Oracle implementation for public function: run_cnot_fidelity_pipeline
def run_cnot_fidelity_pipeline(v: float, m_sn: float, g2_m: float, nbar_m: float,
                                       g2_s: float, nbar_s: float) -> float:
    eta = decode_intrinsic_eta(v, m_sn, g2_m, nbar_m, g2_s, nbar_s)
    v_check = hom_visibility(eta, m_sn, g2_m, nbar_m, g2_s, nbar_s)
    if abs(v_check - v) > 1e-9:
        raise ValueError("visibility round trip failed")
    return cnot_gate_fidelity(eta, m_sn, g2_m, nbar_m, g2_s, nbar_s)
SCICODE_GOLD_EOF
