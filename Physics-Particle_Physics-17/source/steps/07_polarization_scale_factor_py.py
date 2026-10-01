"""
Rescale an unpolarized Upsilon cross section to an arbitrary polar anisotropy $\lambda_\theta$ in the helicity frame ($\lambda_\phi=\lambda_{\theta\phi}=0$), using the two extreme-scenario factors of the paper's Table 1: $k_+\approx A_0/A_{+1}$ (fully transverse, $\lambda_\theta=+1$) and $k_-\approx A_0/A_{-1}$ (fully longitudinal, $\lambda_\theta=-1$), where $A_\lambda$ is the dimuon acceptance of the bin when the parent is produced with decay distribution



$$W(\cos\theta)\propto1+\lambda_\theta\cos^2\theta.$$

The published factors are rounded to two decimals, so they are not exactly consistent with each other: taken literally they do not give a correction of exactly 1 for unpolarized production. The factor returned here is therefore defined by three requirements:



- $k(0)=1$ exactly (the published values are the unpolarized ones);

- $k(+1)/k(-1)=k_+/k_-$ exactly (the ratio is taken as exact; rounding shifts each factor by at most about 0.5%);

- the $\lambda_\theta$ dependence follows from how the normalized decay distribution, and hence the accepted fraction, depends on $\lambda_\theta$ (not a linear interpolation of the factors in $\lambda_\theta$).

Returns
-------
A native Python float gives $k(\lambda_\theta)=\sigma(\lambda_\theta)/\sigma_{\mathrm{unpol}}$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def polarization_scale_factor(lambda_theta: float, k_plus: float, k_minus: float) -> float:
    r'''Cross-section scaling factor $\sigma(\lambda_\theta)/\sigma_{\mathrm{unpol}}$.

    Parameters
    ----------
    lambda_theta : float
        Polar anisotropy $\lambda_\theta$ in the helicity frame,
        $-1\leq\lambda_\theta\leq1$.
    k_plus : float
        Rounded Table 1 factor $k_+>0$ for $\lambda_\theta=+1$.
    k_minus : float
        Rounded Table 1 factor $k_->0$ for $\lambda_\theta=-1$.

    Returns
    -------
    k : float
        Native Python float $\sigma(\lambda_\theta)/\sigma_{\mathrm{unpol}}$,
        with $k(0)=1$ and $k(+1)/k(-1)=k_+/k_-$ exactly.

    Raises
    ------
    ValueError
        If any input is not finite, if $\lambda_\theta$ lies outside
        $[-1,1]$, if $k_+\leq0$ or $k_-\leq0$, or if $k_+/k_->2$ (no physical
        acceptance can produce such a ratio).
    '''
    return k  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_polarization_scale_factor(lambda_theta: float, k_plus: float, k_minus: float) -> float:
    if not all(np.isfinite(v) for v in (lambda_theta, k_plus, k_minus)):
        raise ValueError("inputs must be finite")
    if lambda_theta < -1.0 or lambda_theta > 1.0:
        raise ValueError("lambda_theta must lie in [-1, 1]")
    if k_plus <= 0 or k_minus <= 0:
        raise ValueError("k_plus and k_minus must be > 0")
    # Normalized decay distribution W = 3(1 + lam c^2) / [2(3 + lam)], c = cos(theta).
    # Accepted fraction: A_lam / A_0 = 3 (1 + lam r) / (3 + lam), with r = <c^2> over the
    # accepted unpolarized events.  Hence k(lam) = A_0 / A_lam = (3 + lam) / [3 (1 + lam r)],
    # which gives k(0) = 1 identically and k(+1)/k(-1) = 2 (1 - r) / (1 + r).
    q = k_plus / k_minus
    if q > 2.0:
        raise ValueError("k_plus / k_minus must not exceed 2")
    r = (2.0 - q) / (2.0 + q)          # acceptance-weighted <cos^2 theta>, in [0, 1)
    return float((3.0 + lambda_theta) / (3.0 * (1.0 + lambda_theta * r)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: longitudinal-leaning lambda ---
        {
            "setup": "",
            "call": "polarization_scale_factor(-0.5, 1.20, 0.75)",
            "gold_call": "_oracle_polarization_scale_factor(-0.5, 1.20, 0.75)",
        },
        # --- Normal: positive lambda ---
        {
            "setup": "",
            "call": "polarization_scale_factor(0.4, 1.32, 0.67)",
            "gold_call": "_oracle_polarization_scale_factor(0.4, 1.32, 0.67)",
        },
        # --- Boundary: lambda = +1 returns k_plus exactly ---
        {
            "setup": "",
            "call": "polarization_scale_factor(1.0, 1.20, 0.75)",
            "gold_call": "_oracle_polarization_scale_factor(1.0, 1.20, 0.75)",
        },
        # --- Boundary: lambda = -1 returns k_minus exactly ---
        {
            "setup": "",
            "call": "polarization_scale_factor(-1.0, 1.20, 0.75)",
            "gold_call": "_oracle_polarization_scale_factor(-1.0, 1.20, 0.75)",
        },
        # --- Edge: lambda = 0 with self-consistent factors 4/3, 2/3 returns 1 ---
        {
            "setup": "",
            "call": "polarization_scale_factor(0.0, 4.0/3.0, 2.0/3.0)",
            "gold_call": "_oracle_polarization_scale_factor(0.0, 4.0/3.0, 2.0/3.0)",
        },
        # --- Normal: mildly transverse ---
        {
            "setup": "",
            "call": "polarization_scale_factor(0.3, 1.06, 0.90)",
            "gold_call": "_oracle_polarization_scale_factor(0.3, 1.06, 0.90)",
        },
        # --- Edge: lambda just above -1 (nearly longitudinal) ---
        {
            "setup": "",
            "call": "polarization_scale_factor(-0.999999, 1.20, 0.75)",
            "gold_call": "_oracle_polarization_scale_factor(-0.999999, 1.20, 0.75)",
        },
        # --- Invalid: lambda outside [-1, 1] ---
        {
            "setup": """def run_model():
    try:
        polarization_scale_factor(1.5, 1.20, 0.75)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_polarization_scale_factor(1.5, 1.20, 0.75)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: unphysical ratio k_plus / k_minus > 2 ---
        {
            "setup": """def run_model():
    try:
        polarization_scale_factor(0.2, 1.5, 0.7)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_polarization_scale_factor(0.2, 1.5, 0.7)
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
