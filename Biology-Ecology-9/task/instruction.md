# Biology-Ecology-9

## Background

Comparative demography now draws on large databases of matrix population models assembled by different authors, with stage or age classes of different widths and with different projection intervals, and a fair comparison requires expressing every model at a common resolution. The long-established way of merging adjacent age classes reproduces the growth rate and the stable age structure of the original model but not the reproductive values of the merged classes, and therefore not the elasticities of the growth rate to the vital rates, on which comparisons of life histories, of conservation priorities and of selection pressures rest. The requirement stated above asks for a reduced model that has no such defect. Methods that merge age classes apply to age-classified models, so a stage-classified model has first to be re-expressed by age, and when the reduced classes do not contain a whole number of years of age, their boundaries fall inside years of age.

## Problem

A published stage-classified model of an animal population has a projection interval of one year and four stages, first-year juveniles, second-year subadults, third-year young adults and a plus-group of adults aged three years and older, with projection matrix

$$\mathbf{S} = \begin{pmatrix} 0 & 0.3 & 1.0 & 2.0 \\ 0.6 & 0 & 0 & 0 \\ 0 & 0.7 & 0 & 0 \\ 0 & 0 & 0.85 & 0.55 \end{pmatrix},$$

so that adults remain adults with probability $0.55$ per year and reproduce at the same rate at every age. The population's age-classified model is the shortest Leslie model with one-year age classes that gives adults of every age the adult fertility and survival probability, lets no individual survive beyond its last age class, and has an asymptotic growth rate within a relative error of $0.1$ per cent of that of $\mathbf{S}$. The population breeds continuously through the year, and within a year of age an individual only grows older: the survival probability and the fertility of that year of age take effect as it completes the year. For comparison with species whose published models have three age classes, the population is to be re-expressed as a $3 \times 3$ Leslie matrix $\mathbf{B}$ whose three classes are of equal width and together span the ages of the age-classified model, so that the projection interval of $\mathbf{B}$ is one third of that span. $\mathbf{B}$ must be consistent with the age-classified model over that interval: it has the same growth rate over the interval, its stable age distribution is the population's stable age distribution summed over the ages that each of its classes covers, and the elasticity of its growth rate to each of its entries equals the total elasticity of the population's growth rate over the same interval to the transitions that the entry aggregates. Compute the fertility of the oldest age class of $\mathbf{B}$.

Report that fertility to at least six significant figures. The answer is graded within $2 \times 10^{-4}$.

Report also, inside the reasoning and each to at least five significant figures: the growth rate of $\mathbf{S}$; the number of age classes of the age-classified model and its annual growth rate; the growth rate of $\mathbf{B}$ per projection interval; the three fertilities and the two survival probabilities of $\mathbf{B}$; the reproductive values of $\mathbf{B}$ scaled so that its youngest class has reproductive value one; and the net reproductive rates of $\mathbf{B}$ and of the age-classified model.

State as well, in a line each: how you obtained the age-classified model; how you obtained the stable age distribution of $\mathbf{B}$; how the elasticity requirement fixes the entries of $\mathbf{B}$, and what it implies for the reproductive values of its classes; and whether the reduction preserves the net reproductive rate, with the reason.

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

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_perron_triplet

Goal
----
Given a finite, real, nonnegative, irreducible square matrix, return its Perron root, the positive right eigenvector belonging to it normalised to sum to one, and the positive left eigenvector belonging to it normalised so that its inner product with the right eigenvector is one. Select the Perron root as the eigenvalue of largest real part, never by largest modulus, so that the result is correct for imprimitive matrices.

Also report the number of eigenvalues whose modulus lies within a relative distance of 1e-8 of the Perron root. This count is one for a primitive matrix and equals the index of imprimitivity otherwise.

Test irreducibility directly from the pattern of nonzero entries, as reachability of every index from every other along directed paths, and reject a reducible matrix.

```python
def perron_triplet(matrix: np.ndarray) -> dict:
    """Perron root, stable distribution and reproductive values of an irreducible matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Finite real nonnegative irreducible square array, shape (N, N).

    Returns
    -------
    dict
        Under the keys growth_rate, stable_distribution, reproductive_values and
        peripheral_count.

    Raises
    ------
    ValueError
        When the matrix is not a finite real square array, has a negative entry, or is
        reducible.
    """
    return
```

### Step 2

02_age_expand_stage_model

Goal
----
Given a stage-classified matrix of size s, with s at least two, that is zero everywhere except its first row, its subdiagonal and its southeast corner, and a relative tolerance, build the age-classified expansion of every length n greater than s in increasing order and return the first whose growth rate differs from that of the stage matrix by less than the tolerance times the stage matrix's growth rate. In the expansion of length n the fertility of age class i is the first-row entry of stage i for i up to s and the first-row entry of stage s beyond it; the survival probability from age class j to j + 1 is the subdiagonal entry leaving stage j for j below s and the stasis probability of stage s from j = s onwards. Obtain both growth rates from the Perron triplet. Raise ValueError if no length up to max_classes meets the tolerance.

```python
def age_expand_stage_model(
    stage_matrix: np.ndarray,
    tolerance: float = 1e-3,
    max_classes: int = 500,
) -> dict:
    """Shortest age-classified expansion of a stage model with a plus-group.

    Parameters
    ----------
    stage_matrix : np.ndarray
        Stage-classified matrix with Leslie pattern plus a positive southeast entry, shape (s, s).
        Its subdiagonal survival probabilities lie in (0, 1], its last stage is fertile, and its
        southeast stasis probability lies strictly between zero and one.
    tolerance : float
        Relative tolerance on the growth rate, above zero.
    max_classes : int
        Largest expansion length examined, greater than s.

    Returns
    -------
    dict
        Under the keys fertility, survival, age_classes, stage_growth_rate, age_growth_rate,
        relative_error and previous_error. previous_error is NaN when the chosen length is
        the shortest examined, s + 1, because no shorter expansion was evaluated.

    Raises
    ------
    ValueError
        When the stage matrix does not have the stated pattern and ranges, when the tolerance is
        not finite and above zero, when max_classes is not an integer greater than s, or when no
        expansion up to max_classes meets the tolerance. A count given as a bool, or as a float
        even when its value is integral, is not accepted as an integer, and a bool or a
        non-numeric value is not accepted as a real number.
    """
    return
```

### Step 3

03_disaggregate_leslie

Goal
----
Given the fertilities $F_1$ to $F_n$ and the survival probabilities $P_1$ to $P_{n-1}$ of an irreducible Leslie model and the number of sub-classes $m$ per original class, return the Leslie projection matrix for the demographic timing specified above on $n m$ equal-width sub-classes ordered from youngest to oldest. Each original age class contains $m$ consecutive sub-classes, and one projection interval is the time required to traverse one sub-class. With $m$ equal to one the result is the original Leslie matrix.

```python
def disaggregate_leslie(
    fertility: np.ndarray,
    survival: np.ndarray,
    subdivisions: int,
) -> np.ndarray:
    """Resolve each age class of a Leslie model into equal sub-classes.

    Parameters
    ----------
    fertility : np.ndarray
        Fertilities F_1 to F_n, shape (n,), nonnegative with F_n positive.
    survival : np.ndarray
        Survival probabilities P_1 to P_(n-1), shape (n - 1,), each in (0, 1].
    subdivisions : int
        Number m of sub-classes per original class, at least one.

    Returns
    -------
    np.ndarray
        The resolved Leslie matrix, shape (n m, n m).

    Raises
    ------
    ValueError
        When the vital rates have the wrong shapes, are not finite, violate their ranges, or
        when subdivisions is not an integer of at least one.
        A count given as a bool, or as a float even when its value is integral, is
        not accepted as an integer.
    """
    return
```

### Step 4

04_interstage_flow_aggregate

Goal
----
Given a nonnegative irreducible projection matrix of size N, a number of groups m that divides N, and a whole number of intervals s, return the reduced matrix B = G M^s Q built from the stable distribution of the given matrix, together with its effectiveness and the grouped stable distribution G w. Group I consists of the classes (I - 1) N/m + 1 to I N/m. Obtain the stable distribution from the Perron triplet of the given matrix.

With m equal to N the reduced matrix is M^s itself and the effectiveness is one.

```python
def interstage_flow_aggregate(
    matrix: np.ndarray,
    groups: int,
    steps: int,
) -> dict:
    """Collapse a projection matrix by matching summed interstage flows.

    Parameters
    ----------
    matrix : np.ndarray
        Nonnegative irreducible projection matrix, shape (N, N).
    groups : int
        Number of reduced classes, dividing N.
    steps : int
        Number of original intervals spanned by one reduced interval, at least one.

    Returns
    -------
    dict
        Under the keys reduced_matrix, effectiveness and grouped_distribution.

    Raises
    ------
    ValueError
        When the matrix fails the checks of the Perron triplet, when groups is not an integer
        dividing N, or when steps is not an integer of at least one.
        A count given as a bool, or as a float even when its value is integral, is
        not accepted as an integer.
    """
    return
```

### Step 5

05_elasticity_matrix

Goal
----
Given a nonnegative irreducible projection matrix, return its elasticity matrix, the matrix whose entry in row i and column j is v_i m_ij w_j divided by the product of the Perron root and the inner product of v and w, where m_ij is the entry of the matrix and v and w are its reproductive values and stable distribution. Obtain the three objects from the Perron triplet, so that the result is correct for imprimitive matrices. Entries of the matrix that are zero have zero elasticity.

```python
def elasticity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Elasticities of the Perron root to every entry of a projection matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Nonnegative irreducible projection matrix, shape (N, N), with a positive Perron root.

    Returns
    -------
    np.ndarray
        The elasticity matrix, shape (N, N).

    Raises
    ------
    ValueError
        When the matrix fails the checks of the Perron triplet or its Perron root is zero.
    """
    return
```

### Step 6

06_elasticity_consistent_aggregate

Goal
----
Given a nonnegative irreducible projection matrix of size N, a number of groups m that divides N, and a whole number of intervals s, construct the reduced matrix that is consistent with the given matrix in growth rate, grouped stable distribution, reproductive values and elasticities.

Return the reduced matrix, the reduced reproductive values scaled so that their inner product with the grouped stable distribution is one when the stable distribution sums to one, and the effectiveness defined above.

```python
def elasticity_consistent_aggregate(
    matrix: np.ndarray,
    groups: int,
    steps: int,
) -> dict:
    """Collapse a projection matrix consistently with its reproductive values and elasticities.

    Parameters
    ----------
    matrix : np.ndarray
        Nonnegative irreducible projection matrix, shape (N, N).
    groups : int
        Number of reduced classes, dividing N.
    steps : int
        Number of original intervals spanned by one reduced interval, at least one.

    Returns
    -------
    dict
        Under the keys reduced_matrix, reduced_reproductive_values and effectiveness.

    Raises
    ------
    ValueError
        When the matrix fails the checks of the Perron triplet, when groups is not an integer
        dividing N, or when steps is not an integer of at least one.
        A count given as a bool, or as a float even when its value is integral, is
        not accepted as an integer.
    """
    return
```

### Step 7

07_consistency_residuals

Goal
----
Given a nonnegative irreducible projection matrix of size N, a number of groups m that divides N, a whole number of intervals s, and a candidate reduced matrix of size m, return four nonnegative residuals. The growth-rate residual is the absolute difference between the Perron root of the candidate and lambda^s, divided by lambda^s. The stable-structure residual is the largest absolute difference between the candidate's stable distribution and the grouped stable distribution of M, both summing to one. The reproductive-value residual is the largest absolute difference between the candidate's reproductive values and the stable-distribution-weighted group averages of the reproductive values of M, divided by the largest of those averages, with both vectors scaled to unit inner product with the grouped stable distribution. The elasticity residual is the largest absolute difference between the elasticity matrix of the candidate and the elasticity matrix of M^s summed over pairs of groups. Obtain every eigen-quantity from the Perron triplet and the elasticities of the candidate from the elasticity matrix. Form the elasticities of M^s from the Perron triplet of M itself, as v_i (M^s)_ij w_j divided by lambda^s and by the inner product of v and w: M^s shares its Perron vectors with M, but it is reducible whenever the index of imprimitivity of M shares a factor with s, as it does for a semelparous model resolved into sub-classes, and its own Perron triplet is then not defined.

```python
def consistency_residuals(
    matrix: np.ndarray,
    groups: int,
    steps: int,
    reduced_matrix: np.ndarray,
) -> dict:
    """Discrepancies of a reduced model from the four consistency requirements.

    Parameters
    ----------
    matrix : np.ndarray
        Nonnegative irreducible projection matrix, shape (N, N).
    groups : int
        Number of reduced classes, dividing N.
    steps : int
        Number of original intervals spanned by one reduced interval, at least one.
    reduced_matrix : np.ndarray
        Candidate reduced model, nonnegative irreducible, shape (m, m).

    Returns
    -------
    dict
        Under the keys growth_rate_residual, stable_structure_residual,
        reproductive_value_residual and elasticity_residual.

    Raises
    ------
    ValueError
        When either matrix fails the checks of the Perron triplet, when groups is not an integer
        dividing N, when steps is not an integer of at least one, or when the candidate does not
        have shape (m, m).
        A count given as a bool, or as a float even when its value is integral, is
        not accepted as an integer.
    """
    return
```

### Step 8

08_demographic_parameters

Goal
----
Given a Leslie matrix and its projection interval, return its Perron root, its net reproductive rate, its generation time in the units of the interval, and its Demetrius entropy, as defined above. Take the reproductive values and stable distribution from the Perron triplet. Treat as zero any entry outside the first row and the subdiagonal whose magnitude does not exceed 1e-10 times the largest entry, and reject the matrix if any such entry is larger.

```python
def demographic_parameters(leslie_matrix: np.ndarray, interval: float) -> dict:
    """Growth rate, net reproductive rate, generation time and Demetrius entropy of a Leslie matrix.

    Parameters
    ----------
    leslie_matrix : np.ndarray
        Nonnegative irreducible Leslie matrix, shape (L, L).
    interval : float
        Projection interval, finite and above zero.

    Returns
    -------
    dict
        Under the keys growth_rate, net_reproductive_rate, generation_time and entropy.

    Raises
    ------
    ValueError
        When the matrix fails the checks of the Perron triplet or does not have the Leslie
        pattern, or when the interval is not finite and above zero.
        A bool or a non-numeric value is not accepted as a real number.
    """
    return
```

### Step 9

09_reduce_stage_model

Goal
----
Given the stage model, its projection interval $dt$ and the requested number $m$ of reduced age classes, return its reduction to $m$ equal-width age classes spanning the same ages as the age-classified model defined in step 2. Let $n$ be the number of classes in that age-classified model. The reduced projection interval is the width of one reduced age class, and the demographic timing is that specified in step 3. Use the diagnostic age grid fixed by the returned resolution convention below for quantities associated with the matrix being reduced.

Construct the consistent reduced matrix with step 6 and, for comparison, the flow-matching collapse of the same matrix with step 4. Certify the result before reporting it: the consistent matrix must pass all four requirements of step 7, its survival probabilities must equal those of the flow-matching collapse, and the Perron root of the matrix being reduced, raised to the number of sub-classes per class, must equal the growth rate of the age-classified model. Raise RuntimeError when the largest of these discrepancies exceeds the certificate tolerance, since the reduced model is then not the consistent one.

Return the fertility of the oldest reduced class as the headline result together with the elasticity of the reduced growth rate to it from step 5, and alongside them the expansion, the whole reduced model, its reproductive values scaled so that the first equals one, and the demographic parameters of the consistent reduction, of the flow-matching collapse and of the age-classified model from step 8, the generation times in the units of dt.

```python
def reduce_stage_model(
    stage_matrix: np.ndarray,
    reduced_classes: int,
    interval: float = 1.0,
    tolerance: float = 1e-3,
    certificate_tol: float = 1e-9,
) -> dict:
    """Reduce a stage model with a plus-group to fewer, wider age classes consistently.

    Parameters
    ----------
    stage_matrix : np.ndarray
        Stage-classified matrix with Leslie pattern plus a positive southeast entry, shape (s, s).
        Its subdiagonal survival probabilities lie in (0, 1], its last stage is fertile, and its
        southeast stasis probability lies strictly between zero and one.
    reduced_classes : int
        Number of reduced age classes, from one to the length of the age expansion.
    interval : float
        Projection interval of the stage model, above zero.
    tolerance : float
        Relative tolerance on the growth rate fixing the length of the age expansion.
    certificate_tol : float
        Largest discrepancy accepted in the consistency certificate.

    Returns
    -------
    dict
        Under the keys oldest_fertility, oldest_fertility_elasticity, age_classes,
        stage_growth_rate, expansion_error, reduced_fertility, reduced_survival,
        reduced_growth_rate, reduced_interval, reduced_reproductive_values, growth_rate,
        resolution, peripheral_count, standard_fertility, balanced_effectiveness,
        standard_effectiveness, net_reproductive_rate, generation_time, entropy,
        original_net_reproductive_rate, original_generation_time, original_entropy,
        standard_net_reproductive_rate, standard_generation_time, standard_entropy,
        standard_reproductive_value_residual, standard_elasticity_residual and
        certificate_residual.
        resolution is one when reduced_classes divides age_classes, and reduced_classes
        otherwise; it fixes the number of equal-width sub-classes per original age class
        in the diagnostic grid.

    Raises
    ------
    ValueError
        When the stage matrix or tolerance fails the checks of step 2, when reduced_classes is
        not an integer from one to the length of the expansion, or when interval or
        certificate_tol is not finite and above zero. A count given as a bool, or as a float
        even when its value is integral, is not accepted as an integer, and a bool or a
        non-numeric value is not accepted as a real number.
    RuntimeError
        When the consistency certificate exceeds certificate_tol.
    """
    return
```
