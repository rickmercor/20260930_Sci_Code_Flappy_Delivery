#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def mixture_probabilities(n_snps: int, causal_counts: tuple) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(n_snps, bool) or not isinstance(n_snps, (int, np.integer)):
        raise ValueError("n_snps must be an integer")
    n_snps = int(n_snps)
    if n_snps <= 0:
        raise ValueError("n_snps must be > 0")
    if not isinstance(causal_counts, (tuple, list, np.ndarray)):
        raise ValueError("causal_counts must be a sequence")
    counts = list(causal_counts)
    if len(counts) < 1:
        raise ValueError("causal_counts must hold at least one component")
    for val in counts:
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise ValueError("causal_counts must contain integers only")
        if int(val) < 0:
            raise ValueError("causal_counts must be non-negative")
    total = sum(int(val) for val in counts)
    if total >= n_snps:
        raise ValueError("the causal variants must be fewer than the SNPs fitted")

    # Non-null mixing probabilities are the component shares of the panel; the
    # null probability absorbs the rest, so the vector sums to one by
    # construction.
    non_null = np.array([int(val) / n_snps for val in counts], dtype=float)
    probabilities = np.empty(non_null.size + 1, dtype=float)
    probabilities[0] = 1.0 - float(total) / n_snps
    probabilities[1:] = non_null
    return probabilities

def mixture_scale_factors(n: float, h2_snp: float, gamma: tuple) -> np.ndarray:
    import numpy as np

    if isinstance(n, bool) or not isinstance(n, (int, float, np.integer, np.floating)):
        raise ValueError("n must be a real number")
    n = float(n)
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("n must be a finite number > 0")
    if isinstance(h2_snp, bool) or not isinstance(h2_snp, (int, float, np.integer, np.floating)):
        raise ValueError("h2_snp must be a real number")
    h2_snp = float(h2_snp)
    if not np.isfinite(h2_snp) or h2_snp <= 0.0 or h2_snp >= 1.0:
        raise ValueError("h2_snp must lie strictly between 0 and 1")
    if not isinstance(gamma, (tuple, list, np.ndarray)):
        raise ValueError("gamma must be a sequence")
    scales = np.asarray(gamma, dtype=float).ravel()
    if scales.size < 2:
        raise ValueError("gamma must hold a null component and at least one non-null one")
    if not np.all(np.isfinite(scales)):
        raise ValueError("gamma must be finite")
    if scales[0] != 0.0:
        raise ValueError("the first gamma entry is the null component and must be zero")
    if np.any(scales[1:] <= 0.0):
        raise ValueError("every non-null gamma entry must be > 0")

    # Standardised genotypes with a unit-variance phenotype put the genetic
    # variance at the heritability and the residual variance at its complement.
    sigma_g2 = h2_snp
    sigma_e2 = 1.0 - h2_snp

    # The shrinkage ratio is the residual variance over the prior effect
    # variance of the component, so a large-effect component has a small ratio.
    lam = sigma_e2 / (scales[1:] * sigma_g2)
    c_precision = n + lam

    factors = np.empty((lam.size, 2), dtype=float)
    factors[:, 0] = lam
    factors[:, 1] = c_precision
    return factors

def pip_curve_constants(mixture_probs: np.ndarray, scale_factors: np.ndarray,
                                n: float) -> np.ndarray:
    import numpy as np

    probs = np.asarray(mixture_probs, dtype=float).ravel()
    factors = np.asarray(scale_factors, dtype=float)
    if isinstance(n, bool) or not isinstance(n, (int, float, np.integer, np.floating)):
        raise ValueError("n must be a real number")
    n = float(n)
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("n must be a finite number > 0")
    if factors.ndim != 2 or factors.shape[1] != 2 or factors.shape[0] < 1:
        raise ValueError("scale_factors must have shape (n_components, 2)")
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise ValueError("scale_factors must be finite and > 0")
    if probs.size != factors.shape[0] + 1:
        raise ValueError("mixture_probs must hold one null and one per non-null component")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("mixture_probs must be finite and non-negative")
    if probs[0] <= 0.0:
        raise ValueError("the null mixing probability must be > 0")
    if abs(float(np.sum(probs)) - 1.0) > 1e-9:
        raise ValueError("mixture_probs must sum to 1")

    lam = factors[:, 0]
    c_precision = factors[:, 1]

    # Prior odds against the null, times the shrinkage factor the Gaussian
    # marginal likelihood contributes; the rate is the coefficient of the
    # chi-square statistic in the log posterior odds and carries no variance.
    amplitude = (probs[1:] / probs[0]) * np.sqrt(lam / c_precision)
    rate = n / (2.0 * c_precision)

    constants = np.empty((lam.size, 2), dtype=float)
    constants[:, 0] = amplitude
    constants[:, 1] = rate
    return constants

def pip_threshold_statistic(curve_constants: np.ndarray, alpha: float) -> float:
    import numpy as np

    constants = np.asarray(curve_constants, dtype=float)
    if constants.ndim != 2 or constants.shape[1] != 2 or constants.shape[0] < 1:
        raise ValueError("curve_constants must have shape (n_components, 2)")
    if not np.all(np.isfinite(constants)) or np.any(constants <= 0.0):
        raise ValueError("curve_constants must be finite and > 0")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.integer, np.floating)):
        raise ValueError("alpha must be a real number")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")

    log_amplitude = np.log(constants[:, 0])
    rate = constants[:, 1]

    # Work with the log posterior odds: the threshold PIP = S / (1 + S) is met
    # exactly when log S reaches log(alpha / (1 - alpha)), and the odds
    # themselves would overflow long before the statistic runs out of range.
    log_target = float(np.log(alpha / (1.0 - alpha)))

    def _log_odds_gap(z):
        terms = log_amplitude + rate * z
        shift = float(np.max(terms))
        return shift + float(np.log(np.sum(np.exp(terms - shift)))) - log_target

    # A variant with no evidence at all already carries the prior odds, so a
    # threshold below that is met immediately.
    if _log_odds_gap(0.0) >= 0.0:
        return 0.0

    # Double the upper end until the crossing is bracketed. The rates are
    # positive, so this terminates unless the constants are degenerate.
    lo = 0.0
    hi = 1.0
    while _log_odds_gap(hi) < 0.0:
        hi *= 2.0
        if hi > 1.0e9:
            raise ValueError("the PIP threshold is not reachable for these constants")

    # The gap is strictly increasing, so bisection converges to the crossing;
    # 200 halvings take the bracket well below double precision.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _log_odds_gap(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))

def component_detection_power(scale_factors: np.ndarray,
                                      z_threshold: float) -> np.ndarray:
    import numpy as np
    from math import erfc, sqrt

    factors = np.asarray(scale_factors, dtype=float)
    if factors.ndim != 2 or factors.shape[1] != 2 or factors.shape[0] < 1:
        raise ValueError("scale_factors must have shape (n_components, 2)")
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise ValueError("scale_factors must be finite and > 0")
    lam = factors[:, 0]
    c_precision = factors[:, 1]
    if np.any(c_precision <= lam):
        raise ValueError("the posterior precision must exceed the shrinkage ratio")
    if isinstance(z_threshold, bool) or not isinstance(
            z_threshold, (int, float, np.integer, np.floating)):
        raise ValueError("z_threshold must be a real number")
    z_threshold = float(z_threshold)
    if not np.isfinite(z_threshold) or z_threshold < 0.0:
        raise ValueError("z_threshold must be a finite number >= 0")

    # Marginally over the prior, the statistic of a causal variant is the
    # central chi-square scaled by C_k / lambda_k, so the threshold moves to
    # z * lambda_k / C_k on the standard scale.
    scaled = z_threshold * lam / c_precision

    # Two-sided standard normal tail, written with the complementary error
    # function so that the far tail keeps its relative accuracy.
    power = np.array([erfc(sqrt(0.5 * value)) for value in scaled], dtype=float)
    return power

def component_tail_second_moment(scale_factors: np.ndarray, z_threshold: float,
                                         detection_power: np.ndarray) -> np.ndarray:
    import numpy as np

    factors = np.asarray(scale_factors, dtype=float)
    power = np.asarray(detection_power, dtype=float).ravel()
    if factors.ndim != 2 or factors.shape[1] != 2 or factors.shape[0] < 1:
        raise ValueError("scale_factors must have shape (n_components, 2)")
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise ValueError("scale_factors must be finite and > 0")
    lam = factors[:, 0]
    c_precision = factors[:, 1]
    if np.any(c_precision <= lam):
        raise ValueError("the posterior precision must exceed the shrinkage ratio")
    if power.size != factors.shape[0]:
        raise ValueError("detection_power must hold one entry per component")
    if not np.all(np.isfinite(power)) or np.any(power < 0.0) or np.any(power > 1.0):
        raise ValueError("detection_power must lie in [0, 1]")
    if isinstance(z_threshold, bool) or not isinstance(
            z_threshold, (int, float, np.integer, np.floating)):
        raise ValueError("z_threshold must be a real number")
    z_threshold = float(z_threshold)
    if not np.isfinite(z_threshold) or z_threshold < 0.0:
        raise ValueError("z_threshold must be a finite number >= 0")

    # The detection region is |W| > c on the standard scale, with the same cut
    # point the detection probability uses.
    cut = np.sqrt(z_threshold * lam / c_precision)
    density = np.exp(-0.5 * cut ** 2) / np.sqrt(2.0 * np.pi)

    # Integrating w**2 over both tails leaves the boundary term plus the tail
    # probability itself, so the moment reduces to a two-term expression.
    return np.asarray(2.0 * cut * density + power, dtype=float)

def component_detected_variance(scale_factors: np.ndarray, detection_power: np.ndarray,
                                        tail_second_moment: np.ndarray, h2_snp: float,
                                        n: float) -> np.ndarray:
    import numpy as np

    factors = np.asarray(scale_factors, dtype=float)
    power = np.asarray(detection_power, dtype=float).ravel()
    moment = np.asarray(tail_second_moment, dtype=float).ravel()
    if factors.ndim != 2 or factors.shape[1] != 2 or factors.shape[0] < 1:
        raise ValueError("scale_factors must have shape (n_components, 2)")
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise ValueError("scale_factors must be finite and > 0")
    if power.size != factors.shape[0] or moment.size != factors.shape[0]:
        raise ValueError("detection_power and tail_second_moment need one entry per component")
    if not np.all(np.isfinite(power)) or np.any(power < 0.0) or np.any(power > 1.0):
        raise ValueError("detection_power must lie in [0, 1]")
    if not np.all(np.isfinite(moment)) or np.any(moment < 0.0):
        raise ValueError("tail_second_moment must be finite and non-negative")
    if isinstance(h2_snp, bool) or not isinstance(h2_snp, (int, float, np.integer, np.floating)):
        raise ValueError("h2_snp must be a real number")
    h2_snp = float(h2_snp)
    if not np.isfinite(h2_snp) or h2_snp <= 0.0 or h2_snp >= 1.0:
        raise ValueError("h2_snp must lie strictly between 0 and 1")
    if isinstance(n, bool) or not isinstance(n, (int, float, np.integer, np.floating)):
        raise ValueError("n must be a real number")
    n = float(n)
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("n must be a finite number > 0")

    lam = factors[:, 0]
    c_precision = factors[:, 1]
    # The precision is the sample size plus the shrinkage ratio, so the three
    # arguments have to describe the same study.
    if np.any(np.abs(c_precision - lam - n) > 1e-6 * np.maximum(1.0, c_precision)):
        raise ValueError("scale_factors and n describe different sample sizes")

    sigma_e2 = 1.0 - h2_snp

    # Flat piece: the conditional variance of the effect, which survives even
    # where the deviate itself carries no information.
    flat = (sigma_e2 / c_precision) * power
    # Regression piece: the conditional mean squared, whose coefficient grows
    # with the sample size and shrinks with the component's shrinkage ratio.
    regression = (n * sigma_e2 / (lam * c_precision)) * moment
    return np.asarray(flat + regression, dtype=float)

def expected_heritability_explained(mixture_probs: np.ndarray,
                                            detected_variance: np.ndarray,
                                            h2_snp: float, n_snps: int) -> float:
    import numpy as np

    probs = np.asarray(mixture_probs, dtype=float).ravel()
    detected = np.asarray(detected_variance, dtype=float).ravel()
    if detected.size < 1:
        raise ValueError("detected_variance must hold at least one component")
    if probs.size != detected.size + 1:
        raise ValueError("mixture_probs must hold one null and one per non-null component")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("mixture_probs must be finite and non-negative")
    if abs(float(np.sum(probs)) - 1.0) > 1e-9:
        raise ValueError("mixture_probs must sum to 1")
    if not np.all(np.isfinite(detected)) or np.any(detected < 0.0):
        raise ValueError("detected_variance must be finite and non-negative")
    if isinstance(h2_snp, bool) or not isinstance(h2_snp, (int, float, np.integer, np.floating)):
        raise ValueError("h2_snp must be a real number")
    h2_snp = float(h2_snp)
    if not np.isfinite(h2_snp) or h2_snp <= 0.0 or h2_snp >= 1.0:
        raise ValueError("h2_snp must lie strictly between 0 and 1")
    if isinstance(n_snps, bool) or not isinstance(n_snps, (int, np.integer)):
        raise ValueError("n_snps must be an integer")
    n_snps = int(n_snps)
    if n_snps <= 0:
        raise ValueError("n_snps must be > 0")

    # Every SNP in the panel contributes, weighted by the prior probability of
    # the component it would come from; the null component contributes nothing.
    detected_genetic_variance = float(n_snps) * float(np.sum(probs[1:] * detected))
    return float(detected_genetic_variance / h2_snp)

def required_case_count(n_equivalent: float, prevalence: float,
                                case_fraction: float) -> float:
    import numpy as np
    from math import erf, exp, sqrt, pi

    for name, val in (("n_equivalent", n_equivalent), ("prevalence", prevalence),
                      ("case_fraction", case_fraction)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(val)):
            raise ValueError(f"{name} must be finite")
    n_equivalent = float(n_equivalent)
    prevalence = float(prevalence)
    case_fraction = float(case_fraction)
    if n_equivalent <= 0.0:
        raise ValueError("n_equivalent must be > 0")
    if prevalence <= 0.0 or prevalence >= 1.0:
        raise ValueError("prevalence must lie strictly between 0 and 1")
    if case_fraction <= 0.0 or case_fraction >= 1.0:
        raise ValueError("case_fraction must lie strictly between 0 and 1")

    # Standard normal quantile at 1 - K, obtained by bisecting the error
    # function so that no external special-function package is needed.
    def _standard_normal_quantile(p):
        lo, hi = -40.0, 40.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    truncation = _standard_normal_quantile(1.0 - prevalence)
    density = exp(-0.5 * truncation * truncation) / sqrt(2.0 * pi)

    # Mean liability of cases: the taller this is, the more a single case
    # tells us, so a rarer disease needs fewer cases per unit of information.
    mean_liability_cases = density / prevalence

    # Invert the equivalent-sample-size relation for the total sample, then
    # take the case fraction of it.
    total = (n_equivalent * (1.0 - prevalence) ** 2
             / (mean_liability_cases ** 2 * case_fraction * (1.0 - case_fraction)))
    return float(case_fraction * total)

def run_finemapping_power_pipeline(n_snps: int, causal_counts: tuple, gamma: tuple,
                                           h2_snp: float, pip_threshold: float,
                                           target_h2_fraction: float, prevalence: float,
                                           case_fraction: float) -> float:
    import numpy as np

    # -- Validate the orchestrator inputs.
    if isinstance(n_snps, bool) or not isinstance(n_snps, (int, np.integer)):
        raise ValueError("n_snps must be an integer")
    if not isinstance(causal_counts, (tuple, list)) or len(causal_counts) < 1:
        raise ValueError("causal_counts must be a non-empty sequence")
    if not isinstance(gamma, (tuple, list)) or len(gamma) != len(causal_counts) + 1:
        raise ValueError("gamma must hold a null component plus one per causal count")
    unit_interval = {"h2_snp": h2_snp, "pip_threshold": pip_threshold,
                     "target_h2_fraction": target_h2_fraction,
                     "prevalence": prevalence, "case_fraction": case_fraction}
    for name, val in unit_interval.items():
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(val)) or float(val) <= 0.0 or float(val) >= 1.0:
            raise ValueError(f"{name} must lie strictly between 0 and 1")

    # -- Sub-problem 01: the prior of the mixture model, fixed across sizes.
    # The oracle chains the  twins of the earlier steps, never their
    # public names, so that the reference answer is independent of the
    # submission and a wrong candidate pipeline fails the differential test.
    probs = mixture_probabilities(int(n_snps), tuple(causal_counts))

    def _explained_share(sample_size):
        """Sub-problems 02-08 evaluated at one trial sample size."""
        factors = mixture_scale_factors(
            sample_size, float(h2_snp), tuple(gamma))                           # step 02
        constants = pip_curve_constants(probs, factors, sample_size)    # step 03
        z_cut = pip_threshold_statistic(
            constants, float(pip_threshold))                                    # step 04
        power = component_detection_power(factors, z_cut)               # step 05
        moment = component_tail_second_moment(factors, z_cut, power)    # step 06
        detected = component_detected_variance(
            factors, power, moment, float(h2_snp), sample_size)                 # step 07
        return expected_heritability_explained(
            probs, detected, float(h2_snp), int(n_snps))                        # step 08

    # -- The explained share rises monotonically with the sample size, so
    # doubling brackets the target and bisection locates it.
    lo = 1.0e-6
    hi = 1.0
    while _explained_share(hi) < float(target_h2_fraction):
        lo = hi
        hi *= 2.0
        if hi > 1.0e15:
            raise ValueError("the target share of heritability is out of reach")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _explained_share(mid) < float(target_h2_fraction):
            lo = mid
        else:
            hi = mid
    n_equivalent = 0.5 * (lo + hi)

    # -- Sub-problem 09: the liability-scale price of ascertainment.
    return required_case_count(n_equivalent, float(prevalence), float(case_fraction))
SCICODE_GOLD_EOF
