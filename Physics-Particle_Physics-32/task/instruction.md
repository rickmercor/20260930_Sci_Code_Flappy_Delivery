# Physics-Particle_Physics-32

## Background

Two-loop scattering amplitudes are commonly represented as special functions multiplied by rational coefficient functions. Direct reconstruction of a large multivariate numerator over a common denominator can require an impractical number of finite-field samples. A partial-fraction representation can distribute denominator structure among several lower-degree terms and substantially reduce the reconstruction cost.

The supplied source develops an algebraic procedure for learning such a representation from finite-field evaluations on a generic bivariate phase-space slice. Its essential scientific contribution is the way candidate denominator structures are tested, combined, and converted into a compact canonical ansatz.

For this task, the solver must recover the relevant membership criterion, combination rule, validity check, and canonical coefficient convention from the supplied source. These source-specific decisions are intentionally not reproduced in the problem statement or background.

## Problem

Using the supplied source paper, apply its finite-field bivariate partial-fraction reconstruction procedure to the following deterministic instance over the prime field F_101. Recover the procedure and its scientific justification from the source; only the task-specific canonicalization conventions are stated below.

Represent every residue by an integer in {0,...,100}. Variables are ordered as u,v, and denominator factors are ordered as D1,...,D4. A row [c,a,b] represents c+a*u+b*v:

D1 = [1,1,0]
D2 = [2,0,1]
D3 = [4,1,1]
D4 = [5,2,100]

Thus,

D1 = u+1,
D2 = v+2,
D3 = u+v+4,
D4 = 2u-v+5

in F_101[u,v].

The common denominator is D1*D2*D3*D4, with every factor power equal to one. The following table contains samples of the rational coefficient r(u,v):

          v=0  v=1  v=2  v=3
u=0        97   35    19     7
u=1        21   12    81    77
u=2        25   98    89    89
u=3        58   91    51    99

No listed sample lies on the common denominator.

Process these candidate factor pairs in the stated order:

(D1,D2), (D2,D3), (D1,D4), (D3,D4).

Use the source procedure to obtain:

1. the four pair-membership flags;
2. the final ordered generator exponent vectors;
3. the canonical quotient-coefficient rows;
4. the reconstructed rational values at the held-out points (7,11), (13,17), and (19,23).

For reproducibility, encode each final denominator generator by its four-entry exponent vector in the fixed order D1,D2,D3,D4. Sort the final generator vectors first by total exponent and then lexicographically.

Fit the quotient numerators in the ordered basis (1,u,v). Construct the coefficient-matching system using generator-major columns and quotient-basis-minor columns. Perform modular reduced row-echelon elimination by scanning columns from left to right and selecting the first available pivot row. Set every free variable to zero. These conventions define the canonical result used by the checksum.

Flatten the canonical quotient-coefficient matrix in row-major order as a0,a1,... and the final ordered generator-exponent matrix in row-major order as e0,e1,.... Denote the three held-out values by h0,h1,h2.

Calculate

chi = [
    sum_t (t+1)*a_t
    + sum_t (t+17)*e_t
    + sum_{t=0}^{2} (t+31)*h_t
] mod 101.

Compute chi. In the reasoning, give a concise scientific derivation covering denominator clearing, numerator interpolation, pair membership, the simplified merges and their membership rechecks, the canonical coefficient solve, the held-out evaluations, and the checksum. Include the numerator coefficient tensor C with C[i,j] multiplying u^i v^j (or the equivalent polynomial), the four pair-membership flags, the final ordered generator exponent vectors, the coefficient-system rank and number of free variables, the canonical quotient-coefficient rows, and the three held-out values. Explain why membership must be rechecked after a simplified intersection.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

Clear denominator samples

Goal
----
Clear the known common denominator from rational finite-field samples.

```python
import numpy as np


def clear_denominator_samples(
    samples: np.ndarray,
    factors: np.ndarray,
    powers: np.ndarray,
    prime: int,
) -> np.ndarray:
    """Clear a known denominator from modular rational samples.

    Parameters
    ----------
    samples : np.ndarray
        Integer array of shape (n, 3) containing u, v, and r(u,v).
    factors : np.ndarray
        Integer array of shape (f, 3), with rows [c, a, b].
    powers : np.ndarray
        Nonnegative integer vector of shape (f,).
    prime : int
        Odd prime modulus.

    Returns
    -------
    np.ndarray
        Integer array of shape (n, 3) containing u, v, and N(u,v).

    Raises
    ------
    ValueError
        If the inputs are malformed, the modulus is invalid, or a sample
        lies on the supplied denominator.
    """
    return None
```

### Step 2

Interpolate bivariate numerator

Goal
----
Recover the bivariate numerator coefficient tensor by modular interpolation.

```python
import numpy as np

def interpolate_bivariate(
    numerator_samples: np.ndarray,
    max_u: int,
    max_v: int,
    prime: int,
) -> np.ndarray:
    """Interpolate a tensor-product bivariate polynomial over a prime field.

    Parameters
    ----------
    numerator_samples : np.ndarray
        Integer array of shape (n, 3) containing u, v, and N(u,v).
    max_u : int
        Maximum exponent of u.
    max_v : int
        Maximum exponent of v.
    prime : int
        Prime modulus.

    Returns
    -------
    np.ndarray
        Coefficient matrix of shape (max_u+1, max_v+1), where entry
        [i, j] multiplies u**i * v**j.

    Raises
    ------
    ValueError
        If the inputs, degrees, prime, or sample count are invalid, or
        if the modular interpolation system is singular.
    """
    return None
```

### Step 3

Pair ideal membership

Goal
----
Identify the candidate denominator pairs that can be separated in the partial-fraction decomposition.

```python
import numpy as np

def pair_ideal_membership(
    numerator_coefficients: np.ndarray,
    factors: np.ndarray,
    pairs: np.ndarray,
    prime: int,
) -> np.ndarray:
    """Test membership in each pair-generated affine ideal.

    Parameters
    ----------
    numerator_coefficients : np.ndarray
        Integer matrix C where C[i,j] multiplies u**i * v**j.
    factors : np.ndarray
        Integer array of shape (f,3), with rows [c,a,b]
        representing c + a*u + b*v.
    pairs : np.ndarray
        Integer array of shape (m,2) containing factor-index pairs.
    prime : int
        Odd prime modulus.

    Returns
    -------
    np.ndarray
        Integer vector of length m. Entry k is 1 when the numerator
        belongs to the ideal generated by the selected factor pair,
        and 0 otherwise.

    Raises
    ------
    ValueError
        If an input is malformed, the modulus is invalid, or a pair
        contains equal or out-of-range factor indices.
    """
    return None
```

### Step 4

04_encode_pair_ideals

Goal
----
Convert accepted pairs into denominator-factor exponent tensors.

```python
import numpy as np

def encode_accepted_pair_ideals(
    pair_flags: np.ndarray,
    pairs: np.ndarray,
    factor_count: int,
) -> np.ndarray:
    """Encode accepted factor pairs as exponent tensors.

    Parameters
    ----------
    pair_flags : np.ndarray
        Binary integer vector of length m.
    pairs : np.ndarray
        Integer array of shape (m,2) containing factor-index pairs.
    factor_count : int
        Total number of denominator factors.

    Returns
    -------
    np.ndarray
        Integer exponent array of shape (a,2,factor_count), where a is
        the number of accepted pairs.

    Raises
    ------
    ValueError
        If inputs are malformed, flags are nonbinary, pair indices are
        invalid, or no pair is accepted.
    """
    return None
```

### Step 5

05_simplified_intersection

Goal
----
Combine accepted constraints through LCM generators and membership rechecks.

```python
import numpy as np

def simplified_intersection_generators(
    numerator_coefficients: np.ndarray,
    factors: np.ndarray,
    accepted_ideals: np.ndarray,
    prime: int,
) -> np.ndarray:
    """Construct simplified intersection generators.

    Parameters
    ----------
    numerator_coefficients : np.ndarray
        Integer matrix C where C[i,j] multiplies u**i * v**j.
    factors : np.ndarray
        Integer array of shape (f,3), with rows [c,a,b]
        representing c + a*u + b*v.
    accepted_ideals : np.ndarray
        Nonnegative integer exponent array of shape (a,2,f).
    prime : int
        Odd prime modulus.

    Returns
    -------
    np.ndarray
        Ordered minimal generator-exponent matrix of shape (g,f).

    Raises
    ------
    ValueError
        If polynomial, factor, exponent, shape, or modulus data are
        invalid.
    """
    return None
```

### Step 6

06_fit_pfd_numerators

Goal
----
Fit a deterministic set of lower-degree partial-fraction numerators.

```python
import numpy as np

def fit_canonical_pfd_numerators(
    numerator_coefficients: np.ndarray,
    factors: np.ndarray,
    generators: np.ndarray,
    quotient_degree: int,
    prime: int,
) -> np.ndarray:
    """Fit canonical quotient polynomials for the PFD generators.

    Parameters
    ----------
    numerator_coefficients : np.ndarray
        Integer matrix C where C[i,j] multiplies u**i * v**j.
    factors : np.ndarray
        Integer array of shape (f,3), with rows [c,a,b].
    generators : np.ndarray
        Nonnegative exponent matrix of shape (g,f).
    quotient_degree : int
        Maximum total degree of each quotient polynomial.
    prime : int
        Odd prime modulus.

    Returns
    -------
    np.ndarray
        Generator-major quotient coefficients. Columns follow increasing
        total degree and, within each degree, descending u exponent.

    Raises
    ------
    ValueError
        If the inputs are malformed, the degree or prime is invalid, or
        the modular coefficient system is inconsistent.
    """
    return None
```

### Step 7

07_run_pfd_pipeline

Goal
----
Run the complete reconstruction workflow and return its checksum and diagnostics.

```python
import numpy as np

def run_pfd_pipeline(
    samples: np.ndarray,
    factors: np.ndarray,
    powers: np.ndarray,
    pairs: np.ndarray,
    prime: int,
    heldout_points: np.ndarray,
) -> np.ndarray:
    """Run the complete finite-field PFD reconstruction pipeline.

    Parameters
    ----------
    samples : np.ndarray
        Integer array of shape (n,3) containing u, v, and r(u,v).
    factors : np.ndarray
        Integer array of shape (f,3), with rows [c,a,b].
    powers : np.ndarray
        Nonnegative denominator-factor powers of shape (f,).
    pairs : np.ndarray
        Integer array of shape (m,2) containing candidate factor pairs.
    prime : int
        Odd prime modulus.
    heldout_points : np.ndarray
        Integer array of shape (h,2) containing evaluation points.

    Returns
    -------
    np.ndarray
        Vector containing the checksum, pair flags, generator count,
        and held-out rational values.

    Raises
    ------
    ValueError
        If any preceding step contract fails or a held-out point lies
        on the denominator.
    """
    return None
```
