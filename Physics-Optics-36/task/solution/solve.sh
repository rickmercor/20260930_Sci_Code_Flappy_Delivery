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

def recover_scatter_sites(records: "np.ndarray") -> "np.ndarray":
    records = np.asarray(records, dtype=float)
    exits = np.column_stack((records[:, :2], np.zeros(len(records))))
    return exits - records[:, 5, None] * records[:, 2:5]

import numpy as np

def compute_mode_transfer(sites: "np.ndarray", waist: float, rayleigh: float, focus: float) -> "np.ndarray":
    sites = np.asarray(sites, dtype=float)
    width2 = waist**2 * (1 + ((sites[:, 2] - focus) / rayleigh)**2)
    radius2 = np.sum(sites[:, :2]**2, axis=1)
    return (waist / rayleigh)**2 * waist**2 / width2 * np.exp(-2 * radius2 / width2)

import numpy as np

def compute_detected_weights(records: "np.ndarray", sites: "np.ndarray", transfer: "np.ndarray", absorption: float, scattering: float, angle_limit: float) -> "np.ndarray":
    records = np.asarray(records, dtype=float)
    c = -records[:, 4]
    keep = c > np.cos(np.deg2rad(angle_limit))
    result = np.zeros(len(records), dtype=float)
    exponent = (-absorption * records[keep, 6]
                + (absorption + scattering) * (records[keep, 5] - sites[keep, 2]))
    result[keep] = transfer[keep] * c[keep] * np.exp(exponent)
    return result

import numpy as np

def compute_hybrid_opl(records: "np.ndarray", sites: "np.ndarray", refractive_index: float) -> "np.ndarray":
    records = np.asarray(records, dtype=float)
    return refractive_index * (records[:, 6] - records[:, 5] + sites[:, 2])

import numpy as np

def compute_gated_responses(opl: "np.ndarray", detected: "np.ndarray", resolution: float, bins: "np.ndarray") -> "np.ndarray":
    opl = np.asarray(opl, dtype=float)
    centres = resolution * np.asarray(bins)
    accepted = np.abs(opl[:, None] / 2 - centres[None, :]) < resolution / 2
    return np.asarray(detected, dtype=float)[:, None] * accepted

import numpy as np

def compute_noise_coefficient(responses: "np.ndarray", probabilities: "np.ndarray") -> float:
    responses = np.asarray(responses, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    means = probabilities @ responses
    second = probabilities @ (responses**2)
    variance = np.maximum(second - means**2, 0.0)
    return float(np.sqrt(responses.shape[1] * np.sum(variance)) / np.sum(means))

import numpy as np

def compute_oct_noise(records: "np.ndarray", probabilities: "np.ndarray", waist: float, rayleigh: float, focus: float, absorption: float, scattering: float, refractive_index: float, angle_limit: float, resolution: float, bins: "np.ndarray") -> float:
    sites = recover_scatter_sites(records)
    transfer = compute_mode_transfer(sites, waist, rayleigh, focus)
    detected = compute_detected_weights(records, sites, transfer, absorption, scattering, angle_limit)
    opl = compute_hybrid_opl(records, sites, refractive_index)
    responses = compute_gated_responses(opl, detected, resolution, bins)
    return compute_noise_coefficient(responses, probabilities)
SCICODE_GOLD_EOF
