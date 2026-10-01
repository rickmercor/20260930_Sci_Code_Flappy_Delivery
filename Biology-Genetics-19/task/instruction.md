# Biology-Genetics-19

## Background

Polygenic scores aggregate information across many genetic variants into a quantitative measure of genetic predisposition to a complex trait. Although a polygenic score or a phenotype prediction based on that score can be useful for risk stratification, a single point prediction does not describe the uncertainty associated with that prediction. Estimation error in variant effects, finite training-sample size, differences among individuals and other sources of prediction error can cause uncertainty to vary across individuals.

Prediction intervals provide a way to represent this uncertainty directly on the predicted phenotype scale. Instead of attempting to quantify uncertainty around an unobserved true polygenic score, phenotype prediction intervals target an observable outcome and can therefore be assessed through empirical prediction coverage.

The source study introduces PredInterval, a nonparametric framework for constructing calibrated phenotype prediction intervals from polygenic-score predictions. Its individual-level implementation adapts the cross-validation-plus framework to K-fold cross-validation. Training individuals receive out-of-sample predictions from models fitted without their assigned folds, allowing prediction errors to be estimated without requiring a parametric error distribution.

For a new individual, each cross-validation model produces a separate phenotype prediction. The prediction-interval procedure retains information about which training residual arose from which held-out model rather than reducing the fold-specific test predictions to a single point estimate before interval construction. The empirical candidate values generated from the training residuals are then converted into prediction-interval bounds using a finite-sample order-statistic procedure.

Prediction intervals can also support risk identification. A point-prediction rule uses only the central prediction and therefore ignores uncertainty, whereas an interval-based rule can identify an individual whose point prediction lies below a risk threshold when the prediction interval nevertheless extends into the high-risk phenotype range.

For quantitative traits, the source study defines the high-risk region using an upper-tail phenotype criterion derived from training phenotypes. This benchmark requires the exact quantitative-trait risk definition to be recovered from the source study and then applied to the supplied training cohort. The test-set observed phenotypes are used only for retrospective evaluation of identification success and do not contribute to prediction-interval construction or threshold derivation.

## Problem

A polygenic score summarizes an individual's inherited genetic predisposition to a trait, but a point prediction alone does not convey the uncertainty associated with that prediction. This uncertainty can differ across individuals. Consequently, screening based only on a point-prediction threshold can miss individuals whose point prediction is below a high-risk threshold even though the uncertainty around their predicted phenotype extends into the high-risk range.

A five-fold cross-validated polygenic-score analysis was performed on a training cohort of 39 individuals. For each training individual, the table below gives the assigned fold, the observed phenotype, and the phenotype value predicted by a model fitted with that individual's fold left out.

The five fitted models were also applied to five new test individuals. Therefore, each test individual has five fold-specific predicted phenotype values, one from each cross-validated model. The test individuals' observed phenotypes are supplied only for retrospective evaluation of high-risk identification and must not be used when constructing their prediction intervals or deriving the high-risk threshold.

Using these inputs, construct a 95% phenotype prediction interval for each of the five test individuals following the prediction-interval framework described in the source study. Determine from the source study how the cross-validated training residuals and fold-specific test predictions are combined, including the finite-sample order-statistic rule used for the interval bounds.

Then determine from the source study how high risk is defined for a quantitative trait. Apply that source-study definition to the 39 observed training phenotypes to derive the high-risk phenotype threshold used for this benchmark. If an empirical phenotype quantile is required, use the conventional linearly interpolated empirical quantile, equivalent to the default type-7 quantile in R and the default linear quantile in NumPy.

Apply the source study's interval-based high-risk identification principle using the derived threshold.

For comparison with interval-based screening, define each test individual's point prediction as the arithmetic mean of its five fold-specific predicted phenotype values. A test individual is point-estimate high risk only if this mean is strictly greater than the derived high-risk phenotype threshold.

Report within the reasoning:

1. The 95% prediction interval for each test individual, with both limits rounded to three decimal places.
2. The high-risk phenotype threshold derived from the training phenotypes, rounded to three decimal places.
3. Which test individuals are classified as high risk using the interval-based criterion.
4. The number classified as high risk using the point-prediction criterion and why this differs from the interval-based result.
5. The interval-based high-risk identification success rate, defined as the proportion of truly high-risk test individuals, based on their observed phenotypes, that are identified by the interval-based criterion.
6. The total number of test individuals classified as high risk using the interval-based criterion.

Give as the final answer the number in item 6: the total number of test individuals classified as high risk by the interval-based criterion.

Training data (N = 39, K = 5)

fold    id        observed_y              heldout_pgs
1       train3   -1.07193126153838        0.0474152
1       train7   -0.714576252144282       0.197661
1       train9   -1.3335395922125        -0.15331
1       train13   1.04188316015733       -0.105084
1       train14   1.95227061789933       -0.0280626
1       train17  -0.531085140872945       0.475418
1       train18   0.508689051380338       0.362633
1       train22  -0.41805176941992        0.262017

2       train12  -1.23088602963666        0.470604
2       train25   1.54550456352704        0.322906
2       train26   0.499306008357879       0.277403
2       train38  -0.288393487576561       0.0515611
2       train41  -0.0973871088801502     -0.237816
2       train51   0.425917836209112       0.234821
2       train52   1.57909035208406       -0.148765
2       train55  -1.33744915441249       -0.289245

3       train6   -1.23589946119516       -0.257503
3       train8    0.614779154838674      -0.461834
3       train10   0.0809927821210182     -0.0782572
3       train15   0.617668431123423      -0.0975087
3       train16  -0.640971435911181       0.0812068
3       train20  -0.0183908429358861     -0.155305
3       train24   0.0349868860205995     -0.0617764
3       train27   1.04085365224679       -0.0391055

4       train2   -1.38803382719725        0.151504
4       train11   0.908368542064629       0.102889
4       train19  -0.53235171566502       -0.376862
4       train36   0.863771647531947      -0.106427
4       train63  -0.757773318739676      -0.0528193
4       train64   0.814719924225094      -0.203823
4       train69  -0.410894407379081      -0.103075
4       train84  -1.0119735041883        -0.160579

5       train1   -1.2709917733104         0.0358643
5       train4   -0.0751825298521078      0.281602
5       train5    1.90560370465264        0.0360186
5       train21   0.315770240037499      -0.0353832
5       train23  -0.3353630810161        -0.186936
5       train28  -1.56017303065233        0.102539
5       train30   0.306370709711454       0.298602

Test individuals

ID      observed_y    fold1       fold2        fold3        fold4        fold5
test1    2.410        0.191831     0.0974564    0.353347     0.150763     0.316860
test2   -0.338       -0.384081    -0.664761    -0.710119    -0.504852    -0.254116
test3    2.864       -0.272650    -0.351310    -0.103433    -0.291327    -0.426867
test4    2.127        0.278045     0.380019     0.264186     0.245849     0.329073
test5    1.283       -0.114600    -0.0803053    0.0383443   -0.190363    -0.231246

Confidence level:

0.95

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show enough intermediate calculations to justify the prediction intervals, the source-derived high-risk threshold, and both high-risk screening calculations without turning the response into a general pipeline summary.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_compute_cv_residuals

Goal
----
Compute the magnitude of the out-of-sample phenotype prediction error for each training individual using the observed phenotype and the prediction produced by the model trained with that individual's fold held out.

```python
def compute_cv_residuals(
    observed: np.ndarray,
    heldout_predictions: np.ndarray
) -> np.ndarray:
    """
    Compute absolute held-out residuals for the training individuals.

    Parameters
    ----------
    observed : np.ndarray
        One-dimensional array of observed training phenotypes.
    heldout_predictions : np.ndarray
        One-dimensional array of out-of-sample predictions from the model
        that held out each individual's fold. Must have the same length as
        observed.

    Returns
    -------
    residuals : np.ndarray
        One-dimensional float array containing the absolute prediction
        residual for each training individual.
    """
    return
```

### Step 2

02_align_test_predictions_by_fold

Goal
----
Assign to each training individual the test-individual prediction produced by the model that held out the same cross-validation fold.

```python
def align_test_predictions_by_fold(
    fold_ids: np.ndarray,
    test_fold_predictions: np.ndarray
) -> np.ndarray:
    """
    Match each training individual to the test prediction from its fold model.

    Parameters
    ----------
    fold_ids : np.ndarray
        One-dimensional array of one-based fold identifiers for the training
        individuals.
    test_fold_predictions : np.ndarray
        One-dimensional array containing one test-individual prediction for
        each fold, ordered by one-based fold number.

    Returns
    -------
    matched_test_predictions : np.ndarray
        One-dimensional float array containing one fold-matched test
        prediction per training individual.
    """
    return
```

### Step 3

03_construct_lower_candidates

Goal
----
Construct one candidate lower-bound value for each training individual by subtracting that individual's absolute cross-validated residual from the corresponding fold-matched test prediction.

```python
def construct_lower_candidates(
    matched_test_predictions: np.ndarray,
    residuals: np.ndarray
) -> np.ndarray:
    """
    Construct candidate values for the lower prediction-interval bound.

    Parameters
    ----------
    matched_test_predictions : np.ndarray
        One-dimensional array of fold-matched test predictions.
    residuals : np.ndarray
        One-dimensional array of non-negative absolute cross-validated
        residuals with the same length.

    Returns
    -------
    lower_candidates : np.ndarray
        One-dimensional float array formed by subtracting each residual
        from its matched test prediction.
    """
    return
```

### Step 4

04_construct_upper_candidates

Goal
----
Construct one candidate upper-bound value for each training individual by adding that individual's absolute cross-validated residual to the corresponding fold-matched test prediction.

```python
def construct_upper_candidates(
    matched_test_predictions: np.ndarray,
    residuals: np.ndarray
) -> np.ndarray:
    """
    Construct candidate values for the upper prediction-interval bound.

    Parameters
    ----------
    matched_test_predictions : np.ndarray
        One-dimensional array of fold-matched test predictions.
    residuals : np.ndarray
        One-dimensional array of non-negative absolute cross-validated
        residuals with the same length.

    Returns
    -------
    upper_candidates : np.ndarray
        One-dimensional float array formed by adding each residual
        to its matched test prediction.
    """
    return
```

### Step 5

05_compute_interval_ranks

Goal
----
Compute the one-based finite-sample order-statistic ranks used for the lower and upper prediction-interval bounds from the training-sample size and requested confidence level.

```python
def compute_interval_ranks(
    n_training: int,
    confidence_level: float
) -> np.ndarray:
    """
    Compute one-based lower and upper finite-sample order-statistic ranks.

    Parameters
    ----------
    n_training : int
        Positive number of training individuals.
    confidence_level : float
        Requested prediction-interval confidence level strictly between
        0 and 1.

    Returns
    -------
    ranks : np.ndarray
        Integer array of shape (2,) containing
        [lower_rank, upper_rank].
    """
    return
```

### Step 6

06_select_order_statistic

Goal
----
Sort a numerical candidate set in ascending order and return the value at a specified one-based rank without percentile interpolation.

```python
def select_order_statistic(
    values: np.ndarray,
    rank: int
) -> float:
    """
    Select a one-based ascending order statistic without interpolation.

    Parameters
    ----------
    values : np.ndarray
        One-dimensional finite numeric candidate array.
    rank : int
        One-based integer rank between 1 and len(values).

    Returns
    -------
    selected_value : float
        Candidate value at the requested ascending rank.
    """
    return
```

### Step 7

07_compute_prediction_interval

Goal
----
Compute the lower and upper phenotype prediction-interval bounds for one test individual from cross-validated training observations, held-out training predictions, fold assignments, and fold-specific test predictions.

```python
def compute_prediction_interval(
    observed: np.ndarray,
    heldout_predictions: np.ndarray,
    fold_ids: np.ndarray,
    test_fold_predictions: np.ndarray,
    confidence_level: float
) -> np.ndarray:
    """
    Compute one phenotype prediction interval.

    Parameters
    ----------
    observed : np.ndarray
        One-dimensional array of observed training phenotypes.
    heldout_predictions : np.ndarray
        One-dimensional array of cross-validated held-out predictions.
    fold_ids : np.ndarray
        One-dimensional array of one-based fold identifiers for the
        training individuals.
    test_fold_predictions : np.ndarray
        One-dimensional array containing one prediction for the new
        individual from each fold-specific model.
    confidence_level : float
        Requested prediction-interval confidence level.

    Returns
    -------
    interval : np.ndarray
        Float array of shape (2,) containing
        [lower_bound, upper_bound].
    """
    return
```

### Step 8

08_compute_prediction_intervals_for_tests

Goal
----
Compute phenotype prediction intervals for multiple test individuals using common cross-validated training data and each test individual's fold-specific predictions.

```python
def compute_prediction_intervals_for_tests(
    observed: np.ndarray,
    heldout_predictions: np.ndarray,
    fold_ids: np.ndarray,
    test_fold_predictions_matrix: np.ndarray,
    confidence_level: float
) -> np.ndarray:
    """
    Compute prediction intervals for multiple test individuals.

    Parameters
    ----------
    observed : np.ndarray
        Observed training phenotypes.
    heldout_predictions : np.ndarray
        Cross-validated held-out training predictions.
    fold_ids : np.ndarray
        One-based fold identifier for each training individual.
    test_fold_predictions_matrix : np.ndarray
        Two-dimensional array with shape (n_test, n_folds),
        containing one fold-specific prediction per test
        individual and fold.
    confidence_level : float
        Requested prediction-interval confidence level.

    Returns
    -------
    intervals : np.ndarray
        Float array of shape (n_test, 2), with columns
        [lower_bound, upper_bound].
    """
    return
```

### Step 9

09_compute_screening_metrics

Goal
----
Derive the source-study quantitative-trait high-risk threshold from the training phenotypes and compare interval-based screening, point-prediction screening, and held-out observed test phenotypes at that threshold.

```python
def compute_screening_metrics(
    intervals: np.ndarray,
    test_fold_predictions_matrix: np.ndarray,
    observed_train: np.ndarray,
    observed_test: np.ndarray
) -> np.ndarray:
    """
    Compute source-derived high-risk screening metrics.

    Parameters
    ----------
    intervals : np.ndarray
        Prediction intervals with shape (n_test, 2).
    test_fold_predictions_matrix : np.ndarray
        Fold-specific predictions with shape
        (n_test, n_folds).
    observed_train : np.ndarray
        Observed training phenotypes used to derive the
        quantitative-trait high-risk threshold.
    observed_test : np.ndarray
        Held-out observed phenotypes for the test individuals.

    Returns
    -------
    metrics : np.ndarray
        Float array containing:
        [high_risk_threshold,
         interval_high_risk_count,
         point_high_risk_count,
         true_high_risk_count,
         true_high_risk_identified,
         identification_success_rate].
    """
    return
```

### Step 10

10_compute_target_interval_high_risk_count

Goal
----
Compute the final requested number of test individuals classified as high risk by constructing their prediction intervals, deriving the source-study quantitative-trait high-risk threshold from training phenotypes, and applying interval-based screening.

```python
def compute_target_interval_high_risk_count(
    observed: np.ndarray,
    heldout_predictions: np.ndarray,
    fold_ids: np.ndarray,
    test_fold_predictions_matrix: np.ndarray,
    observed_test: np.ndarray,
    confidence_level: float
) -> int:
    """
    Compute the interval-based high-risk count.

    Parameters
    ----------
    observed : np.ndarray
        Observed training phenotypes.
    heldout_predictions : np.ndarray
        Cross-validated held-out training predictions.
    fold_ids : np.ndarray
        One-based training-fold identifiers.
    test_fold_predictions_matrix : np.ndarray
        Fold-specific predictions for all test individuals,
        with shape (n_test, n_folds).
    observed_test : np.ndarray
        Held-out observed phenotypes for test individuals.
    confidence_level : float
        Requested prediction-interval confidence level.

    Returns
    -------
    interval_high_risk_count : int
        Number of test individuals whose prediction interval
        extends above the source-derived quantitative-trait
        high-risk threshold.
    """
    return
```
