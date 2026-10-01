"""
Return the next time step chosen by the energy-controlled adaptive rule. The step shrinks where the free energy is falling fast and relaxes towards $\\tau_{\\max}$ once the flow is quiet, and it is then held back by two hard limits: the floor $\\tau_{\\min}$ and the **step-ratio clamp** $\\gamma_{\\max}\\tau_n$, with $\\gamma_{\\max}=4.86454$ the maximal ratio under which the modified discrete energy of this scheme is proved to dissipate. The clamp is the only place in the pipeline where that constant enters, and it is what keeps an adaptive run inside the proved-stable regime; the value must be taken from the source rather than assumed, since the corresponding bound for Cahn-Hilliard is the much tighter $1.534$. Use the constant **as the source prints it**, $4.86454$, rather than re-solving its defining equation $\\gamma^{3/2}=1+2\\gamma$ to higher precision; that is the convention every field of this task uses.

Adaptive time stepping for gradient flows keys off the instantaneous dissipation rate: microstructural evolution alternates between fast transitions, which need small steps for accuracy, and long quasi-steady relaxations, in which the step may grow by orders of magnitude. A standard controller sets the candidate step to $\\tau_{\\max}/\\sqrt{1+\\beta\\dot E^2}$, which is $\\tau_{\\max}$ when the energy is stationary and shrinks like $1/(\\sqrt{\\beta}|\\dot E|)$ when it is not. On a variable grid the *ratio* of adjacent steps is itself constrained: variable-step BDF2 energy estimates close only when $\\gamma_n$ is bounded, so a growth clamp is part of the method, not a convenience.




**## Formulas**



With $\\dot E=(E^{n}-E^{n-1})/\\tau_{n}$ and $\\gamma_{\\max}=4.86454$,




$$\\tau_{\\mathrm{ad}}=\\frac{\\tau_{\\max}}{\\sqrt{1+\\beta\\,\\dot E^{\\,2}}}, \\qquad \\tau_{n+1}=\\max\\Big\\{\\tau_{\\min},\\ \\min\\big\\{\\tau_{\\max},\\ \\tau_{\\mathrm{ad}},\\ \\gamma_{\\max}\\tau_{n}\\big\\}\\Big\\}.$$




The floor is applied **after** the three-way minimum, so it wins over the ratio clamp when the two conflict. With $\\beta=0$ the candidate is exactly $\\tau_{\\max}$, so growth is limited by the clamp alone; with $\\tau_{\\min}=\\tau_{\\max}$ the step is constant and the ratio is identically one.

Returns
-------
A Python `float` in $[\\tau_{\\min},\\ \\max(\\tau_{\\min},\\tau_{\\max})]$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def adaptive_time_step(tau_n: float, E_n: float, E_nm1: float,
                       tau_min: float, tau_max: float,
                       beta: float) -> float:
    """tau_n: float > 0, the step just taken.
    E_n, E_nm1: floats, the free energies after and before that step.
    tau_min, tau_max: floats with 0 < tau_min <= tau_max.
    beta: float >= 0, the controller gain.
    Return the next time step tau_{n+1} as a float."""
    # Implement per the formulas above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_adaptive_time_step(tau_n: float, E_n: float, E_nm1: float,
                               tau_min: float, tau_max: float,
                               beta: float) -> float:
    gamma_max = 4.86454          # the source's maximal admissible step ratio
    tau_n = float(tau_n)
    tau_min = float(tau_min)
    tau_max = float(tau_max)
    beta = float(beta)
    if not (tau_n > 0.0):
        raise ValueError("tau_n must be positive")
    if not (0.0 < tau_min <= tau_max):
        raise ValueError("require 0 < tau_min <= tau_max")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    edot = (float(E_n) - float(E_nm1)) / tau_n
    tau_ad = tau_max / np.sqrt(1.0 + beta * edot * edot)
    return float(max(tau_min, min(tau_max, tau_ad, gamma_max * tau_n)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: a quiet flow, where the adaptive candidate is close to tau_max and
        # none of the three limits binds.
        {"setup": "",
         "call": "adaptive_time_step(0.64, 283.6248, 283.6255, 1e-4, 1.0, 4.0)",
         "gold_call": "_oracle_adaptive_time_step(0.64, 283.6248, 283.6255, 1e-4, "
                      "1.0, 4.0)"},
        # normal: a fast transient, where the energy rate is large and the adaptive
        # candidate alone selects the step.
        {"setup": "",
         "call": "adaptive_time_step(0.02, 288.14, 290.61, 1e-4, 1.0, 4.0)",
         "gold_call": "_oracle_adaptive_time_step(0.02, 288.14, 290.61, 1e-4, 1.0, "
                      "4.0)"},
        # boundary: the STEP-RATIO CLAMP binding. The energy is essentially flat, so
        # the candidate is tau_max = 1, but the previous step was 1e-4, and the
        # answer must be exactly 4.86454e-4 rather than 1. This is the second step
        # of the benchmark run.
        {"setup": "",
         "call": "adaptive_time_step(1e-4, 294.30983833, 294.33292334, 1e-4, 1.0, 4.0)",
         "gold_call": "_oracle_adaptive_time_step(1e-4, 294.30983833, 294.33292334, "
                      "1e-4, 1.0, 4.0)"},
        # boundary: the FLOOR binding, and beating the ratio clamp. A violent energy
        # drop drives the candidate below tau_min, and the previous step is tiny, so
        # both the candidate and the clamp lie under the floor; the floor wins.
        {"setup": "",
         "call": "adaptive_time_step(1e-6, 100.0, 1.0e5, 1e-4, 1.0, 4.0)",
         "gold_call": "_oracle_adaptive_time_step(1e-6, 100.0, 1.0e5, 1e-4, 1.0, 4.0)"},
        # boundary: the CEILING binding. The energy is exactly stationary and the
        # previous step is already large, so the answer is exactly tau_max.
        {"setup": "",
         "call": "adaptive_time_step(0.9, 276.6007, 276.6007, 1e-4, 1.0, 4.0)",
         "gold_call": "_oracle_adaptive_time_step(0.9, 276.6007, 276.6007, 1e-4, 1.0, "
                      "4.0)"},
        # edge: beta = 0 removes the energy feedback entirely, so the candidate is
        # tau_max and the clamp is the only limiter - a pure geometric ramp at the
        # maximal admissible ratio.
        {"setup": "",
         "call": "adaptive_time_step(1e-6, 500.0, 900.0, 1e-6, 2.0, 0.0)",
         "gold_call": "_oracle_adaptive_time_step(1e-6, 500.0, 900.0, 1e-6, 2.0, 0.0)"},
        # edge: tau_min equal to tau_max, the constant-step case, where the answer
        # must be that common value whatever the energies do.
        {"setup": "",
         "call": "adaptive_time_step(0.05, 1.0, 1.0e6, 0.05, 0.05, 1.0)",
         "gold_call": "_oracle_adaptive_time_step(0.05, 1.0, 1.0e6, 0.05, 0.05, 1.0)"},
        # edge: an energy that INCREASES. The rate enters squared, so the controller
        # is symmetric and shrinks the step just as it would for a decrease.
        {"setup": "",
         "call": "adaptive_time_step(0.3, 290.0, 283.0, 1e-4, 1.0, 4.0)",
         "gold_call": "_oracle_adaptive_time_step(0.3, 290.0, 283.0, 1e-4, 1.0, 4.0)"},
    ]
