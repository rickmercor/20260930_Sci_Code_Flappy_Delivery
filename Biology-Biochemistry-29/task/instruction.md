# Biology-Biochemistry-29

## Problem

The CRITERIA paper by Villejo, de los Reyes, and Hernandez develops an equilibrium-parametrization framework for biochemical reaction networks, including EnvZ–OmpR. Consult its EnvZ–OmpR model, retaining the source's species labels and kinetic-rate numbering, to solve a new inverse-calibration and experimental-design problem whose inputs are the kinetic constraints and equilibrium observations below and whose output is the fraction \(F^*\) at maximal absolute logarithmic sensitivity.

Recover the equilibrium parametrization from the paper and use the fixed rates \(k_1=1.7\), \(k_2=0.83\), \(k_4=0.47\), \(k_5=1.11\), \(k_7=0.62\), \(k_9=1.37\), \(k_{10}=0.58\), \(k_{11}=1.43\), \(k_{13}=0.66\), and \(k_{14}=1.21\), with unknown rates constrained by \(0.7\leq k_3\leq1.8\), \(1.0\leq k_6\leq5.0\), \(0.4\leq k_8\leq1.2\), and \(0.5\leq k_{12}\leq1.6\). Let \(Q\) denote the concentration of the complex \(\mathrm{XpY}\), define the experimental total coordinates by \(T_1=X+\mathrm{XD}+\mathrm{XDYp}+\mathrm{XT}+\mathrm{XTYp}+\mathrm{Xp}+Q\) and \(T_2=-X-\mathrm{XD}-\mathrm{XT}-\mathrm{Xp}+Y+\mathrm{Yp}\), and define \(F(T_1,T_2)=(\mathrm{Xp}+Q+\mathrm{XTYp})/T_1\) and \(G(T_1,T_2)=\mathrm{d}F/\mathrm{d}\log(k_6)\), where the derivative includes equilibrium changes while holding both totals and every other kinetic rate fixed. Treat the four observations \(\mathrm{Yp}=0.96334092858302567\), \(F(1.0,0.8)=0.5405066201174169\), \(F(1.8,0.3)=0.53608216437829215\), and \(G(1.4,1.1)=-0.070312225957344945\) as exact decimal inputs, with \(\mathrm{Yp}\) independent of the totals.

Determine the unique rate vector in the stated box consistent with all four observations and strictly positive concentrations for all nine species at each experimental condition, with a mathematically justified global uniqueness certificate and a maximum absolute numerical observation residual no greater than \(10^{-8}\); numerical fitting or multistart convergence alone does not establish exact uniqueness. Holding the calibrated rates fixed, set \(T_1=1.35\), let \(T_2^*\) globally maximize \(\lvert G(1.35,T_2)\rvert\) over \(T_2\in[0.2,2.0]\) on the strictly positive equilibrium, and report \(F^*=F(1.35,T_2^*)\) rounded to eight digits after the decimal point. Cite the source equations or section used.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 8 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_compute_equilibrium_coefficients.py

Goal
----
Evaluate the equilibrium parametrization from a complete vector of fourteen kinetic rate constants. Return the five coefficients multiplying Q, the rational coefficient beta, and the constant concentration Yp. Validate that the input and all returned coefficients are finite and strictly positive.

```python
import numpy as np

def compute_equilibrium_coefficients(
    rate_constants: np.ndarray,
) -> np.ndarray:
    """Return [c_X, c_XD, c_XDYp, c_XT, c_XTYp, beta, Yp].

    The input has shape (14,) in k1,...,k14 order.
    Evaluate the complete equilibrium parametrization without
    setting small positive coefficients to zero.

    Returns:
        np.ndarray: Seven finite, strictly positive coefficients.

    Raises:
        ValueError: If the rate vector has an invalid shape,
            contains nonfinite or nonpositive values, or produces
            nonfinite or nonpositive coefficients.
    """
    return np.empty(7, dtype=float)
```

### Step 2

02_solve_positive_equilibrium.py

Goal
----
Solve the two conservation laws for the complete nine-species equilibrium. Use the physically admissible root of the reduced equation and reconstruct every species. Validate the original conservation laws and reject nonpositive or nonfinite states.

```python
import numpy as np

def solve_positive_equilibrium(
    coeffs: np.ndarray,
    T1: float,
    T2: float,
) -> np.ndarray:
    """Return [Q, Y, X, XD, XDYp, XT, XTYp, Xp, Yp].

    coeffs has the seven-entry layout from Step 1.
    T1 and T2 are finite strictly positive scalars. Solve the
    conservation laws and retain only a strictly positive
    equilibrium satisfying the original equations. Do not impose
    a positivity threshold larger than zero.

    Returns:
        np.ndarray: The nine finite, strictly positive
            equilibrium concentrations.

    Raises:
        ValueError: If the coefficients or totals are invalid,
            or if no strictly positive admissible equilibrium exists.
    """
    return np.empty(9, dtype=float)
```

### Step 3

03_compute_fraction_sensitivity.py

Goal
----
Compute the steady-state fraction F and signed total logarithmic sensitivity G=dF/dlog(k6) at fixed conserved totals and other kinetic rates. Include the induced response of both free concentrations.

```python
import numpy as np

def compute_fraction_sensitivity(
    coeffs: np.ndarray,
    T1: float,
    T2: float,
) -> np.ndarray:
    """Return [F, G], where G=dF/dlog(k6).

    Compute F=(Xp+Q+XTYp)/T1 and the signed total equilibrium
    logarithmic sensitivity. Hold T1, T2, and every kinetic rate
    other than k6 fixed. Include the induced equilibrium response
    of both Q and Y. coeffs has the Step 1 layout.

    Returns:
        np.ndarray: A finite two-entry array [F, G].

    Raises:
        ValueError: If the inputs are invalid or no strictly
            positive admissible equilibrium exists.
    """
    return np.empty(2, dtype=float)
```

### Step 4

04_construct_inverse_constraints.py

Goal
----
Eliminate the experiment-specific equilibrium concentrations from the two calibration fraction observations. Return a rational relation between the shared normalized inverse variables s and v.

```python
import numpy as np

def construct_inverse_constraints(
    observations: np.ndarray,
) -> np.ndarray:
    """Return [p0,p1,q0,q1], defining s(v)=(p0+p1*v)/(q0+q1*v).

    observations=[Yp,F_A,F_B,G_C], where A uses totals
    (1.0,0.8), B uses (1.8,0.3), and C uses (1.4,1.1).
    Eliminate the shared inverse variable using the two fraction
    observations. Require finite observations, 0<Yp<1.8,
    0<F_A,F_B<1, and G_C<0.

    Returns:
        np.ndarray: Four finite rational-relation coefficients.

    Raises:
        ValueError: If the observation vector is malformed,
            nonfinite, or violates the stated admissibility
            conditions.
    """
    return np.empty(4, dtype=float)
```

### Step 5

05_reconstruct_kinetic_parameters.py

Goal
----
Recover the four unknown rates from normalized inverse invariants, measured Yp, and the fixed rate constants. Preserve the shared-denominator structure and reject singular or nonphysical reconstructions.

```python
import numpy as np

def reconstruct_kinetic_parameters(
    normalized: np.ndarray,
    Yp: float,
    fixed_rates: np.ndarray,
) -> np.ndarray:
    """Return [k3,k6,k8,k12] from normalized=[s,v,r].

    Yp is positive and fixed_rates is a length-14 vector.
    Positions 2,5,7,11 (zero-based) contain arbitrary positive
    placeholders; all other entries are fixed. Recover the unknown
    rates using the structural coefficient identities. This
    function does not enforce the calibration bounds, which are
    supplied to Step 6.

    Returns:
        np.ndarray: Four finite, strictly positive kinetic rates
            in k3,k6,k8,k12 order.

    Raises:
        ValueError: If inputs are invalid, an inverse denominator
            is singular, or the reconstructed rates are nonfinite
            or nonpositive.
    """
    return np.empty(4, dtype=float)
```

### Step 6

06_calibrate_kinetic_parameters.py

Goal
----
Solve the coupled inverse problem using the constant Yp, two fraction observations, and one logarithmic-sensitivity observation. Enumerate algebraic inverse candidates, reconstruct rates, enforce the parameter box, and validate candidates using the original forward model.

```python
import numpy as np

def calibrate_kinetic_parameters(
    fixed_rates: np.ndarray,
    observations: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """Return [k3,k6,k8,k12] from the coupled calibration problem.

    fixed_rates is a positive length-14 vector with placeholders
    at indices 2,5,7,11. observations=[Yp,F_A,F_B,G_C].
    bounds is a finite positive (4,2) array of lower/upper bounds.

    Use the fixed calibration conditions specified in Step 4.
    Enumerate inverse candidates, reconstruct the original rates,
    enforce the bounds, and validate the original forward model.
    Require exactly one distinct admissible solution in the box,
    with maximum absolute original-observation residual <=1e-8.
    Numerical refinement is permitted, but the solution must not
    be hard-coded.

    Returns:
        np.ndarray: Four finite rates [k3,k6,k8,k12] within
            the supplied parameter bounds.

    Raises:
        ValueError: If the inputs or parameter bounds are invalid,
            no admissible calibration satisfies the residual
            tolerance, or more than one distinct admissible
            inverse solution survives in the parameter box.
    """
    return np.empty(4, dtype=float)
```

### Step 7

07_optimize_sensitivity.py

Goal
----
For a calibrated equilibrium model, hold all kinetic rates and T1 fixed and maximize |dF/dlog(k6)| over a supplied interval of T2. Consider the admissible interior stationary point and the interval endpoints.

```python
import numpy as np

def optimize_sensitivity(
    coeffs: np.ndarray,
    T1: float,
    T2_lower: float,
    T2_upper: float,
) -> np.ndarray:
    """Return [T2_star,F_star,maximum_absolute_sensitivity].

    coeffs has the Step 1 layout. Hold the kinetic rates and T1
    fixed while maximizing |dF/dlog(k6)| over the supplied closed
    interval of T2. The objective is absolute sensitivity, not
    F or signed G. Consider every admissible interior stationary
    point and both interval endpoints to establish globality.
    Accept an endpoint optimum.

    Returns:
        np.ndarray: Finite [T2_star,F_star,max_abs_G], with
            T2_star inside the supplied interval.

    Raises:
        ValueError: If the coefficients, fixed total, or interval
            are invalid; if the bounds are not finite and strictly
            positive with lower<upper; or if no admissible
            equilibrium exists in the interval.
    """
    return np.empty(3, dtype=float)
```

### Step 8

08_compute_optimized_fraction.py

Goal
----
Compose the earlier steps to calibrate the four kinetic rates, construct the calibrated equilibrium model, maximize absolute logarithmic sensitivity, and return the fraction at the resulting admissible equilibrium.

```python
import numpy as np

def compute_optimized_fraction(
    fixed_rates: np.ndarray,
    observations: np.ndarray,
    bounds: np.ndarray,
    T1: float = 1.35,
    T2_lower: float = 0.2,
    T2_upper: float = 2.0,
) -> float:
    """Return the fraction at the globally sensitivity-maximizing equilibrium.

    Inputs are the fixed-rate vector, four calibration observations,
    parameter bounds, and optional final-condition T1 and T2 interval.
    The defaults are T1=1.35, T2_lower=0.2, T2_upper=2.0.

    Compose the previous public functions to calibrate the four
    unknown kinetic rates, construct the calibrated equilibrium
    model, and maximize absolute logarithmic sensitivity. Return
    the fraction evaluated at the resulting admissible optimum.

    Returns:
        float: The finite optimized steady-state fraction.

    Raises:
        ValueError: If any input is invalid, calibration has no
            unique admissible solution, or the optimization interval
            contains no admissible equilibrium. Propagate the
            relevant ValueError from the preceding steps.
    """
    return 0.0
```
