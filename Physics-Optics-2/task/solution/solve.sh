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

def prepare_transport(flux: "np.ndarray", volumes: "np.ndarray", pixel: int) -> "np.ndarray":
    density = np.sum(flux, axis=1)
    return np.column_stack((density * volumes, flux[:, pixel] / density))

import numpy as np

def compose_proposal(local: "np.ndarray", global_prob: "np.ndarray", large_step: float) -> "np.ndarray":
    return (1.0 - large_step) * local + large_step * global_prob[None, :]

import numpy as np

def compute_acceptance(masses: "np.ndarray", proposal: "np.ndarray") -> "np.ndarray":
    forward = masses[:, None] * proposal
    reverse = forward.T
    ratio = np.divide(reverse, forward, out=np.ones_like(forward), where=forward > 0)
    return np.minimum(1.0, ratio)

import numpy as np

def compute_rejection_moments(proposal: "np.ndarray", acceptance: "np.ndarray") -> "np.ndarray":
    rho = np.sum(proposal * acceptance, axis=1)
    r2 = np.sum(proposal * (1.0 - acceptance) ** 2, axis=1)
    return np.column_stack((rho, r2))

import numpy as np

def compute_stopped_weights(moments: "np.ndarray") -> "np.ndarray":
    rho = moments[:, 0]
    r2 = moments[:, 1]
    return 1.0 + (1.0 - rho) / (1.0 - r2)

import numpy as np

def compute_tour_law(proposal: "np.ndarray", acceptance: "np.ndarray", moments: "np.ndarray") -> "np.ndarray":
    transition = proposal * acceptance / moments[:, 0, None]
    n = len(transition)
    system = transition.T - np.eye(n)
    system[-1, :] = 1.0
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    stationary = np.linalg.solve(system, rhs)
    return stationary

import numpy as np

def reconstruct_radiances(quantities: "np.ndarray", mean_weights: "np.ndarray", stationary: "np.ndarray") -> "np.ndarray":
    masses = quantities[:, 0]
    observable = quantities[:, 1]
    weighted_law = stationary * mean_weights
    exact = np.dot(masses, observable)
    vanilla = masses.sum() * np.dot(weighted_law, observable) / weighted_law.sum()
    return np.array([exact, vanilla])

import numpy as np

def compute_radiance_distortion(flux: "np.ndarray", volumes: "np.ndarray", local: "np.ndarray", global_prob: "np.ndarray", large_step: float, pixel: int) -> float:
    quantities = prepare_transport(flux, volumes, pixel)
    proposal = compose_proposal(local, global_prob, large_step)
    acceptance = compute_acceptance(quantities[:, 0], proposal)
    moments = compute_rejection_moments(proposal, acceptance)
    mean_weights = compute_stopped_weights(moments)
    stationary = compute_tour_law(proposal, acceptance, moments)
    radiances = reconstruct_radiances(quantities, mean_weights, stationary)
    return float(100.0 * (radiances[1] / radiances[0] - 1.0))
SCICODE_GOLD_EOF
