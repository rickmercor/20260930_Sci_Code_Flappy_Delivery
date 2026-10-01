# Biology-Ecology-20

## Background

Explainable machine learning has become the standard route by which an ecological model is turned into an ecological claim. A model predicts coral cover from disturbance histories, an attribution method divides each prediction among the covariates, and the covariate carrying the largest attribution is reported as the driver of the change. The step that is rarely examined is the one from the attribution to the claim, and it rests on an assumption that has not held wherever it has been tested: that the attribution is a property of the data rather than of the model that was fitted to it.

Architectures differ in what they can represent. An additive model spreads a response across single covariates because it has no other way to express it; a tree ensemble concentrates it wherever its splits fall; a fully local smoother divides it according to which training rows happen to lie nearby. Where the data determine the response strongly, these differences are invisible, because every architecture is driven to the same answer. Where the data are thin, or where several covariates carry the same information, the architectures are free, and they use that freedom differently. Predictive skill does not reveal the difference, since the freedom is exercised in directions that leave the fitted values almost unchanged.

That makes explanation disagreement a form of uncertainty in its own right, distinct from uncertainty in the prediction, and it is not addressed by the usual practice of fitting one model well. It has to be measured on the explanations themselves, per observation rather than on average, because the disagreement is local: a set of architectures can rank the covariates identically across a whole dataset and still contradict one another about one reef in one year, which is exactly the scale at which a management decision is made. A measure of that kind has to do three things at once. It must compare vectors whose magnitudes vary enormously across observations for reasons that have nothing to do with whether the architectures agree, since an observation whose prediction is far from the mean model output must carry a large attribution vector while an observation close to it may carry a large one or a small one. It must remain interpretable when no architecture can be declared correct, which is the case whenever the ground truth is unavailable. And it must produce a number whose size can be read directly, so that an observation can be flagged for a domain expert without a threshold being invented for each dataset.

## Problem

Two machine learning models fitted to the same reef monitoring record can predict next year's coral cover equally well and still disagree completely about which disturbance caused the decline, and a manager who reads the attribution of a single model has no way to tell whether the story it tells is a property of the reef or a property of the architecture. Your task is to quantify that disagreement on a synthetic record whose mechanism is known, and to report a single number: the strength of the case against the most contested observation in the evaluation set, measured by the explanation discrepancy measure and read across every choice of reference model.

Build the record, fit four architectures to it, explain every evaluation observation under each of them, and reduce the four explanations at an observation to a discrepancy that is comparable across observations. Use the following deterministic configuration throughout. No step of it contains a random draw, and none may be replaced by one.

The record covers 20 sites over 22 years. With the site index $s$ and the year index $t$ both counted from zero, the three disturbance intensities are laid down by fixed integer maps,

\(c_{s,t} = \max[0, ((7 s + 11 t + 3 s t) \operatorname{mod} 23) - 17] / 6,\)

\(b_{s,t} = \max[0, ((5 s + 13 t + s^2) \operatorname{mod} 19) - 14] / 5,\)

\(o_{s,t} = \max[0, ((3 s + 17 t + t^2) \operatorname{mod} 29) - 22] / 7,\)

for cyclones, bleaching and a residual class respectively. Mean hard coral cover starts at $C_{s,0} = 0.25 + 0.01 (s \operatorname{mod} 11)$ and advances by logistic regrowth followed by proportional mortality,

\(C_{s,t} = C_{s,t-1} + g C_{s,t-1} (1 - C_{s,t-1} / Kcap) - C_{s,t-1} \min(1, p_{s,t}),\)

\(p_{s,t} = w_c (c_{s,t} + r_1 c_{s,t-1} + r_2 c_{s,t-2}) + w_b b_{s,t} + w_o o_{s,t},\)

with the result clipped to the interval between the cover floor and the carrying capacity, and lagged intensities taken as zero before the record starts.

* growth rate $g = 0.8$, carrying capacity $Kcap = 0.80$, cover floor $0.02$
* mortality weights $w_c = 0.55$, $r_1 = 0.45$, $r_2 = 0.20$, $w_b = 0.30$, $w_o = 0.18$
* the response is the cover at a site-year; the six covariates are, in this order, cyclone intensity at the current year and at lags one and two, bleaching intensity at the current year, residual intensity at the current year, and a time covariate $(t - 2)/(T - 3)$ with $T = 22$
* a site-year is admissible only if both cyclone lags exist, so each site contributes 20 rows; rows are laid out site-major, all admissible years of site 0 in ascending order, then site 1, and so on
* rows whose index is divisible by 6 form the evaluation set and every other row is a training row
* every eighth training row, starting at the first, forms the background set over which an absent covariate is averaged
* the additive model carries one smooth per covariate and no interaction: piecewise-linear hat functions on 5 equally spaced knots spanning the unit interval, extended flat beyond the end knots, an intercept, a second-difference roughness penalty of weight 2.0 on each covariate's coefficients, and a ridge term of $10^{-8}$ on every coefficient so that the penalised normal equations are non-singular
* both tree ensembles are built from least-squares regression trees grown by exhaustive binary splitting on the midpoints between consecutive distinct covariate values among the rows at a node, maximising the reduction in the within-node sum of squares, with at least 6 rows in any node; a later candidate replaces the best candidate only when its reduction is larger by more than $10^{-12}$ times the parent sum of squares, so candidates tied within that tolerance keep the first candidate found when covariates are scanned in index order and thresholds in ascending order
* a row whose covariate value equals a split threshold goes to the left branch, both while a tree is grown and whenever one is evaluated; a node is a leaf when its depth budget is exhausted, when no admissible split leaves at least 6 rows on both sides, when every response at the node is the same value, or when no admissible split reduces the within-node sum of squares by more than $10^{-12}$ times its value at the node, and its prediction is the mean response of the rows it holds
* the averaged ensemble holds 8 trees of depth at most 6, where tree $k$ is grown on every training row whose position in the training set is not congruent to $k$ modulo 8, and predicts their plain mean
* the boosted expansion starts at the mean training response and runs 30 rounds, each fitting a tree of depth at most 3 to the current residuals and adding $0.1$ times it
* the kernel smoother predicts the weighted mean training response with Gaussian weights $\exp[-|x - x_i|^2 / (2 h^2)]$ at bandwidth $h = 0.10$, the norm being Euclidean on the six covariates
* the four architectures are ordered as additive model, averaged ensemble, boosted expansion, kernel smoother, and all four are fitted on the same training rows and given the same covariates
* an explanation is the vector of exact Shapley values of the game whose players are the six covariates and whose payoff at a covariate subset is the model output averaged over the background rows with the covariates in the subset overwritten by those of the observation being explained; with six covariates this is to be computed exactly rather than sampled
* the discrepancy between explanations is taken on the five disturbance covariates only, the time covariate being a nuisance direction along which the architectures may differ without ecological consequence; the restriction is applied to the attribution vectors themselves, so every norm the measure takes is a norm of a restricted vector, and all such norms are Euclidean
* arithmetic throughout in IEEE float64

For every evaluation observation and every choice of one of the four architectures as the reference model, compute the explanation discrepancy measure of that observation under that reference: a dimensionless numerical diagnostic of how far the reference model's attribution vector stands from those of the other three at that observation, standardised by a scale computed across the whole evaluation set so that the values of one reference model are comparable from observation to observation. The measure is not symmetric in the two models it compares, so an observation that stands apart under one choice of reference need not stand apart under another. The consensus-discrepancy diagnostic for an observation is the smallest of its four measures, which is the part of its discrepancy that survives every choice of reference. Report the largest value of that smallest measure over the evaluation observations, to five significant figures. Compute this diagnostic even if the framework's equivalent-predictive-performance prerequisite is not met; if it is not met, state that the result cannot be interpreted as evidence of explanation disagreement independent of predictive-skill differences. The answer is graded within $0.001$.

Your reasoning must also report the following, in this order: (1) the site index of the flagged observation, as an exact integer on the conventions above; (2) the year index of that observation, likewise; (3) the four measures at that observation, one per reference model, each to at least five significant figures; (4) the same smallest-measure statistic at the runner-up observation, to at least five significant figures, so that the separation between the first and second can be judged; (5) the coefficient of determination of each of the four architectures on the evaluation rows, each to at least four significant figures; (6) the normalising scale used for each of the four reference models, each to at least four significant figures; and (7) the largest absolute residual in the identity that the attributions of any model at any observation must satisfy against that model's prediction and its mean prediction over the background set.

Your reasoning must also state whether the four architectures rank the covariates in the same order when their attributions are aggregated over the whole evaluation set, and how that compares with what they do locally at the flagged observation, giving the ranking each architecture produces.

Your reasoning must also justify the conventions the configuration prescribes but does not explain: why the discrepancy between two explanations must be referred to a scale drawn from the explanations themselves rather than compared raw, and which scale that is; why the measure is not symmetric in the two models it compares, and what the reference model therefore means; why the standardising scale is computed across observations rather than within one, and what a measure of three at an observation therefore says about that observation; why the attributions are enumerated exactly rather than sampled; whether the restriction to the disturbance covariates applies to the normalising scale as well as to the distance; and why an ensemble grown on deterministic folds is used in place of a bootstrap resample, and a kernel smoother in place of a feed-forward network.

Your reasoning must also state the condition the framework places on the set of models before their explanations may be compared at all, and say whether the four architectures here satisfy it.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:

* The tags are required. Do not omit them or leave them empty.
* The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
* Put only that one number between the tags. No units, no words, no extra lines.
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

01_01_simulate_reef_cover

Goal
----
01_simulate_reef_cover

Population dynamics on a coral reef are the balance of two opposed processes. Between disturbances the cover of hard coral regrows towards the carrying capacity of the substrate, and the regrowth is fastest at intermediate cover, since a nearly bare reef has few colonies to propagate and a nearly saturated one has no space left to colonise. Against that, three kinds of disturbance remove cover: cyclones, which break and scour colonies, thermal stress, which bleaches them, and a residual class covering predation and disease. The first of these acts over more than one year, because a reef stripped by a cyclone in one year carries rubble and weakened colonies into the next, so the mortality in a given year responds to the cyclone intensity of that year and of the two before it.

The state variable is the mean hard coral cover at a site, a proportion between a floor and the carrying capacity. Writing $C_{s,t}$ for the cover at site $s$ in year $t$, $g$ for the intrinsic growth rate, $Kcap$ for the carrying capacity and $c_{s,t}$, $b_{s,t}$, $o_{s,t}$ for the cyclone, bleaching and other intensities, each in the unit interval, the update is logistic regrowth followed by proportional mortality,

$$C_{s,t} = C_{s,t-1} + g C_{s,t-1} (1 - C_{s,t-1} / Kcap) - C_{s,t-1} \min(1, p_{s,t}),$$

with the disturbance pressure

$$p_{s,t} = w_c (c_{s,t} + r_1 c_{s,t-1} + r_2 c_{s,t-2}) + w_b b_{s,t} + w_o o_{s,t},$$

the result clipped below at the cover floor and above at the carrying capacity. The pressure is capped at unity so that mortality cannot exceed the standing cover. Lagged intensities are taken as zero before the record starts. This is a cyclone-dominated scenario: the cyclone weight exceeds the other two, and the cyclone term alone carries lags.

The disturbance fields themselves are laid down by fixed integer maps, so that the whole record is reproducible in exact arithmetic with no random draw anywhere. With $s$ and $t$ counted from zero,

$$c_{s,t} = \max[0, ((7 s + 11 t + 3 s t) \operatorname{mod} 23) - 17] / 6,$$

$$b_{s,t} = \max[0, ((5 s + 13 t + s^2) \operatorname{mod} 19) - 14] / 5,$$

$$o_{s,t} = \max[0, ((3 s + 17 t + t^2) \operatorname{mod} 29) - 22] / 7.$$

Each map leaves the intensity at zero in most site-years and raises it to a value in the unit interval in the remainder, which reproduces the sparse, spiky character of real disturbance records rather than a smoothly varying field. The initial cover varies across sites as $0.25 + 0.01 (s \operatorname{mod} 11)$, so that sites do not begin in a common state.

```python
def simulate_reef_cover(
    n_sites: int,
    n_years: int,
    growth_rate: float,
    carrying_capacity: float,
    disturbance_weights: dict,
    cover_floor: float,
) -> dict:
    """Advance mean hard coral cover at every site under logistic regrowth and lagged disturbance mortality.
 
    Parameters
    ----------
    n_sites : int
        Number of monitoring sites.
    n_years : int
        Number of years in the record.
    growth_rate : float
        Intrinsic regrowth rate.
    carrying_capacity : float
        Upper bound on cover.
    disturbance_weights : dict
        Disturbance weights under the keys cyclone (w_c), cyclone_lag1 (r_1), cyclone_lag2 (r_2), bleaching (w_b) and other (w_o). The lag entries r_1 and r_2 scale the lagged cyclone intensities inside the cyclone term, so the pressure is w_c (c_t + r_1 c_{t-1} + r_2 c_{t-2}) + w_b b_t + w_o o_t.
    cover_floor : float
        Lower bound on cover.
 
    Returns
    -------
    dict
        Under the keys cover, cyclone, bleaching, other, mean_cover, min_cover and final_mean_cover.
 
    Raises
    ------
    ValueError
        When n_sites is not an integer of one or more, when n_years is not an integer of three or more, when growth_rate is not finite and above zero, when carrying_capacity is not finite, above zero and at most one, when one of the five weight keys is absent, when a weight fails to be finite and not below zero, or when cover_floor is not finite, not below zero and below the carrying capacity.
    """
    return
```

### Step 2

02_02_assemble_lagged_design

Goal
----
02_assemble_lagged_design

A disturbance record and a cover record become a supervised learning problem only after three decisions are made: which covariates carry the disturbance history, which site-years are admissible as rows, and how the rows are divided between fitting and evaluation.

The covariate set follows the mortality structure of the system. Cyclone intensity acts over the current year and the two before it, so it contributes three covariates; thermal stress and the residual class act in the current year only, so they contribute one each. A sixth covariate carries time itself, rescaled to the unit interval so that every covariate occupies the same range and no covariate dominates a distance or a penalty through its units alone. With $t$ running over the admissible years and $T$ the number of years in the record, the time covariate is $(t - 2)/(T - 3)$, which is zero in the first admissible year and one in the last. The covariate order is fixed throughout as cyclone at lag zero, cyclone at lag one, cyclone at lag two, bleaching, other, time.

A site-year is admissible only if both cyclone lags exist, so the first two years of the record are dropped and each site contributes $T - 2$ rows. Rows are laid out site-major: all admissible years of the first site in ascending order, then all admissible years of the second, and so on, which makes the row index a deterministic function of the site and the year and lets a later step recover the identity of any flagged row.

The split is deterministic by construction rather than drawn at random, because a graded quantity that depends on a random partition is not reproducible by anyone who does not share the draw. Rows whose index is divisible by the evaluation stride form the evaluation set; every other row is a training row. Taking every $k$th row rather than a contiguous block leaves gaps in both space and time, which is what a random split is normally used to achieve: the training set then covers the full range of disturbance conditions rather than one era or one region of the domain.

One further subset is needed downstream. An attribution method that measures what a covariate contributes to a prediction has to say what it would mean for that covariate to be absent, and the usual answer is to average the model over the empirical distribution of the absent covariates. That average is taken over a background set, here every $m$th training row, small enough that an exact enumeration over covariate subsets stays cheap and fixed so that the attribution is a deterministic function of the data.

```python
def assemble_lagged_design(
    cover: np.ndarray,
    cyclone: np.ndarray,
    bleaching: np.ndarray,
    other: np.ndarray,
    eval_stride: int,
    background_stride: int,
) -> dict:
    """Build the lagged covariate matrix and split it deterministically into training, evaluation and background sets.
 
    Parameters
    ----------
    cover : np.ndarray
        Mean hard coral cover, shape (n_sites, n_years).
    cyclone : np.ndarray
        Cyclone intensity on the same grid.
    bleaching : np.ndarray
        Bleaching intensity on the same grid.
    other : np.ndarray
        Residual disturbance intensity on the same grid.
    eval_stride : int
        Row stride that selects the evaluation set.
    background_stride : int
        Row stride that selects the background set from the training rows.
 
    Returns
    -------
    dict
        Under the keys train_features, train_response, eval_features, eval_response, background, eval_site, eval_year, n_train, n_eval and n_background. When n_years is 3, the time covariate of the single admissible year is 0.
 
    Raises
    ------
    ValueError
        When cover is not a two-dimensional array with at least one site and at least three years, when the three intensity arrays do not share the shape of cover, when any entry fails to be finite, when eval_stride is not an integer of two or more, when background_stride is not an integer of one or more, or when either stride leaves one of the three sets empty.
    """
    return
```

### Step 3

03_03_fit_additive_spline_model

Goal
----
03_fit_additive_spline_model

The first of the four architectures is the statistical one: an additive model in which the response is an intercept plus one smooth function of each covariate, with no interaction terms. Ecologists reach for this class because it admits non-linear responses to environmental drivers while remaining readable, each fitted smooth being a curve that can be plotted and argued about.

The smooths are built from piecewise-linear hat functions. On a grid of $K$ equally spaced knots spanning the unit interval, the hat centred at knot $a$ rises linearly from zero at the neighbouring knot below to one at knot $a$ and falls linearly to zero at the neighbouring knot above; the two end hats are extended flat beyond the ends of the grid so that the basis remains a partition of unity outside the knot range. Every covariate carries its own copy of this basis, and the design matrix is the intercept column followed by the six blocks of $K$ columns.

An unpenalised fit on such a basis is both rough and rank deficient, since each block sums to the intercept column. Both problems are met at once by penalised least squares. Writing $B$ for the design matrix, $y$ for the response, $D_j$ for the second-difference operator acting on the coefficients of covariate $j$, lam for the penalty weight and eps for a small ridge term, the coefficients solve

$$(B^T B + lam \sum_j D_j^T D_j + eps I) beta = B^T y.$$

The second-difference penalty charges curvature in the knot direction, which is the discrete analogue of the integrated squared second derivative used by spline smoothers, so raising lam drives each smooth towards a straight line rather than towards zero. The ridge term is not a modelling choice but a numerical one: it removes the exact rank deficiency between the intercept and the six blocks. Because the null space it fixes lies in the kernel of $B$, it shifts the coefficients without shifting the fitted values, so predictions are unaffected by its value at any small setting.

Once fitted, the model is evaluated at an arbitrary matrix of query rows. That matters downstream, because an attribution method has to interrogate the fitted function at points that are mixtures of an observation and a background row rather than at the training data.

```python
def fit_additive_spline_model(
    train_features: np.ndarray,
    train_response: np.ndarray,
    query_features: np.ndarray,
    n_knots: int,
    penalty_weight: float,
    ridge_nugget: float,
) -> dict:
    """Fit a penalised additive model on piecewise-linear hat bases and evaluate it at the query rows.
 
    Parameters
    ----------
    train_features : np.ndarray
        Training covariates, shape (n_train, 6).
    train_response : np.ndarray
        Training response, shape (n_train,).
    query_features : np.ndarray
        Rows at which the fitted model is evaluated, shape (n_query, 6).
    n_knots : int
        Knots per covariate.
    penalty_weight : float
        Weight on the second-difference penalty.
    ridge_nugget : float
        Weight on the ridge term.
 
    Returns
    -------
    dict
        Under the keys coefficients, predictions, train_rss and effective_curvature. train_rss is the residual sum of squares at the training rows, and effective_curvature is beta^T (sum_j D_j^T D_j) beta, the squared second differences of the fitted coefficients summed over every covariate block, without the penalty weight.
 
    Raises
    ------
    ValueError
        When train_features is not a two-dimensional array of six columns with at least one row, when train_response does not match it in length, when query_features does not carry six columns, when any entry fails to be finite, when n_knots is not an integer of three or more, when penalty_weight is not finite and not below zero, or when ridge_nugget is not finite and above zero.
    """
    return
```

### Step 4

04_04_grow_regression_tree

Goal
----
04_grow_regression_tree

Both tree architectures rest on the same object, a regression tree grown by recursive binary splitting. At a node holding a set of rows, every admissible split is scored, the best is taken, and the two children are grown in turn until a depth limit or a size limit stops the recursion. A node that is not split becomes a leaf whose prediction is the mean response of the rows it holds.

The score is the reduction in the within-node sum of squares. Writing $y$ for the responses at the node and $y_L$, $y_R$ for the two parts a candidate split produces, the parent sum of squares is $\sum (y_i - \overline{y})^2$ and the reduction is that quantity less the same sum computed separately in each part. Maximising the reduction is equivalent to minimising the pooled within-part variance, and it is the criterion that makes a regression tree a piecewise-constant least-squares fit.

Candidate thresholds are the midpoints between consecutive distinct values of a covariate among the rows at the node, which is the coarsest set of thresholds that realises every distinct partition of those rows by that covariate. A candidate is admissible only if both parts retain at least the minimum node size. The rule that decides among equally good candidates has to be stated, because covariates measured on a coarse grid produce exact ties routinely and a tree that breaks them arbitrarily is not reproducible: the first candidate found wins, scanning covariates in index order and thresholds in ascending order. Reductions are compared to within rounding: a later candidate displaces the best so far only if its reduction is larger by more than 1e-12 times the node's sum of squares, so an exact tie that floating point separates in the last digits still goes to the first candidate. Recursion stops when the depth budget is exhausted, when the node holds fewer than twice the minimum node size, when every response at the node is the same value, or when no admissible split reduces the sum of squares by more than 1e-12 times its value at the node. The third of those is not implied by the fourth. Summing a constant vector and subtracting its mean leaves a residue at the last bit, so a node whose responses are identical reports an apparent reduction of order 1e-31 and would otherwise be split on nothing.

The grown tree is returned in flattened array form rather than as a nested structure, one entry per node in the order a depth-first walk creates them, a node before its left subtree and the whole left subtree before the right. Every node carries the mean response of the rows it holds, internal nodes included, where it records what the node would predict if the walk stopped there. A leaf carries a covariate index of minus one, a threshold of not-a-number, and the mean response of its rows; an internal node carries its covariate index, its threshold, and the positions of its two children, with rows satisfying covariate at most threshold going left. That representation is what the ensembles consume, and it makes a tree comparable entry by entry between two implementations.

```python
def grow_regression_tree(
    features: np.ndarray,
    response: np.ndarray,
    max_depth: int,
    min_node: int,
) -> dict:
    """Grow a least-squares regression tree by exhaustive binary splitting and return it in flattened form.

    Parameters
    ----------
    features : np.ndarray
        Covariates at the rows the tree is grown on, shape (n_rows, n_covariates).
    response : np.ndarray
        Response at those rows, shape (n_rows,).
    max_depth : int
        Maximum number of splits on any root-to-leaf path.
    min_node : int
        Minimum number of rows a node may hold.

    Returns
    -------
    dict
        Under the keys node_feature, node_threshold, node_value, node_left, node_right, n_nodes, n_leaves and train_rss. A leaf's node_left and node_right entries are minus one, and train_rss is the residual sum of squares of the tree's predictions at the rows it was grown on.

    Raises
    ------
    ValueError
        When features is not a two-dimensional array with at least one row and one covariate, when response does not match it in length, when any entry fails to be finite, when max_depth is not an integer of one or more, or when min_node is not an integer of one or more. A node whose responses are all equal is returned as a leaf rather than split.
    """
    return
```

### Step 5

05_05_predict_tree_ensembles

Goal
----
05_predict_tree_ensembles

Two of the four architectures are ensembles of regression trees, and they differ in how the ensemble is assembled rather than in the trees themselves.

The first averages trees grown on overlapping subsets of the training rows. Variance reduction by averaging needs the members to differ, and the usual way to make them differ is to resample the training rows at random. A random resample cannot be used where the result must be reproducible by someone who does not share the draw, so the subsets are taken deterministically instead: with $n$ trees, tree $k$ is grown on every training row whose position in the training set is not congruent to $k$ modulo $n$. Each tree then sees a fixed fraction $(n-1)/n$ of the rows, the omitted fractions are disjoint across trees, and the ensemble prediction is the plain mean over the members. The members are grown deep, since averaging is what controls their variance.

The second builds an additive expansion in a forward stagewise fashion. The expansion starts at the mean response of the training rows and, at each round, grows a shallow tree on the current residuals and adds a shrunken multiple of it. With $F_0$ the training mean, $h_r$ the tree grown at round $r$ and nu the learning rate,

$$F_r(x) = F_{r-1}(x) + nu h_r(x),$$

in which $h_r$ is grown on the current residual $y - F_{r-1}$.

Shrinkage below one is what makes the procedure work as regularisation: each round corrects only part of the residual, so the expansion approaches the data slowly and many shallow trees combine into a smooth response rather than one deep tree memorising the rows. The members are grown shallow, since depth here controls the order of interaction the expansion can represent rather than the variance.

Both ensembles are evaluated at an arbitrary matrix of query rows, since the attribution step interrogates them away from the training data. Neither contains a random draw at any point, so both are reproducible from the training data and the stated settings alone.

```python
def predict_tree_ensembles(
    train_features: np.ndarray,
    train_response: np.ndarray,
    query_features: np.ndarray,
    n_trees: int,
    forest_depth: int,
    n_rounds: int,
    boost_depth: int,
    learning_rate: float,
    min_node: int,
) -> dict:
    """Fit the deterministically bagged ensemble and the stagewise boosted expansion and evaluate both at the query rows.

    Parameters
    ----------
    train_features : np.ndarray
        Training covariates, shape (n_train, n_covariates).
    train_response : np.ndarray
        Training response, shape (n_train,).
    query_features : np.ndarray
        Rows at which both ensembles are evaluated, shape (n_query, n_covariates).
    n_trees : int
        Members of the averaged ensemble.
    forest_depth : int
        Depth budget of an averaged member.
    n_rounds : int
        Rounds of the stagewise expansion.
    boost_depth : int
        Depth budget of a stagewise member.
    learning_rate : float
        Shrinkage applied to each stagewise member.
    min_node : int
        Minimum rows a node may hold.

    Returns
    -------
    dict
        Under the keys forest_predictions, boosted_predictions, forest_leaf_total, boosted_leaf_total, forest_train_rss and boosted_train_rss.

    Raises
    ------
    ValueError
        When train_features is not a two-dimensional array with at least one row and one covariate, when train_response does not match it in length, when query_features does not share the covariate count, when any entry fails to be finite, when n_trees is not an integer of two or more, when forest_depth, n_rounds, boost_depth or min_node is not an integer of one or more, or when learning_rate is not finite, above zero and at most one.
    """
    return
```

### Step 6

06_06_evaluate_model_consensus

Goal
----
06_evaluate_model_consensus

A framework that compares explanations across architectures needs a set of architectures that differ in how they represent the response while agreeing on what they are given: the same covariates, the same training rows, and comparable predictive skill. Without the first of those the comparison is trivial, and without the last it is confounded, because a model that fits worse can be dismissed on accuracy grounds before its explanation is examined at all.

Four architectures are used here and they span the usual classes. The additive model is smooth and has no interactions, so its response to one covariate is a curve that does not depend on the others. The averaged tree ensemble is piecewise constant and captures interactions, but only those its splits happen to isolate. The stagewise boosted expansion is also piecewise constant and, at a depth of three, represents interactions of up to third order, though it reaches them by many small corrections rather than by a few deep splits. The kernel smoother is smooth, fully local, and represents interactions of every order through the joint distance in covariate space, with no parametric form at all.

The fourth of these takes the place of the feed-forward network that would otherwise complete the set. A network fitted by stochastic gradient descent from a random initialisation is not reproducible without sharing the initialisation and the batch order, and a graded quantity built on it would be unreachable rather than merely difficult. The kernel smoother preserves what the network contributes to the comparison, a smooth non-parametric response with unrestricted interactions and no additive structure, while remaining a deterministic function of the training data. Its prediction at $x$ is the weight-averaged training response

$$m(x) = \sum_i w_i(x) y_i / \sum_i w_i(x),$$

in which the weight of training row $i$ is $w_i(x) = \exp[-\|x - x_i\|^2 / (2 h^2)]$, computed after subtracting the squared distance to the nearest training row, which cancels between the numerator and the denominator and keeps the weights of a distant query row from underflowing together,

with $h$ the bandwidth and the norm the Euclidean one on the covariate vector. Because the weights vary continuously with $x$, the smoother has no ordering step and therefore no tie-breaking convention, which a nearest-neighbour rule would need: covariates on a coarse grid put many training rows at exactly equal distance, and the prediction would then depend on how equal distances are sorted.

The four predictions are returned stacked in a fixed order, additive model first, then the averaged ensemble, the boosted expansion and the kernel smoother. That order is carried unchanged through the attribution and discrepancy steps, so that a reference model can be named by its position.

```python
def evaluate_model_consensus(
    train_features: np.ndarray,
    train_response: np.ndarray,
    query_features: np.ndarray,
    model_config: dict,
) -> dict:
    """Fit the four architectures on common training rows and evaluate every one of them at the query rows.
 
    Parameters
    ----------
    train_features : np.ndarray
        Training covariates, shape (n_train, 6).
    train_response : np.ndarray
        Training response, shape (n_train,).
    query_features : np.ndarray
        Rows at which every architecture is evaluated, shape (n_query, 6).
    model_config : dict
        Settings under the keys n_knots, penalty_weight, ridge_nugget, n_trees, forest_depth, n_rounds, boost_depth, learning_rate, min_node and bandwidth.
 
    Returns
    -------
    dict
        Under the keys query_predictions, train_predictions, train_r_squared and n_models, with the model axis ordered as additive model, averaged ensemble, boosted expansion, kernel smoother. train_r_squared is one minus the residual sum of squares over the total sum of squares about the mean training response, and is zero for every model when the training response has no spread.
 
    Raises
    ------
    ValueError
        When train_features is not a two-dimensional array of six columns with at least one row, when train_response does not match it in length, when query_features does not carry six columns, when any entry fails to be finite, when one of the ten configuration keys is absent, when bandwidth is not finite and above zero, or when bandwidth is so small that twice its square underflows to zero.
    """
    return
```

### Step 7

07_07_exact_shapley_attributions

Goal
----
07_exact_shapley_attributions

An explanation of a single prediction, in the additive attribution family, is a vector that divides the gap between the prediction and a baseline among the covariates. The division that is uniquely determined by the standard fairness axioms is the Shapley value of the cooperative game whose players are the covariates and whose payoff is the model output when only a stated subset of them is known.

The game has to be defined before the value can be computed, and the definition turns on what it means for a covariate to be unknown. Here an unknown covariate is drawn from the empirical distribution of a fixed background set independently of the covariates that are known, so the payoff of a subset $S$ at the observation $x$ is the average model output over the background rows with the entries in $S$ overwritten by those of $x$,

$$v_x(S) = (1/|B|) \sum_{b \in B} f(x_S; b_{-S}),$$

with $B$ the background set. The attribution to covariate $j$ is then the weighted average of what $j$ adds to every subset that excludes it,

$$phi_j(x) = \sum_{S \subseteq F - j} [|S|! (K - |S| - 1)! / K!] [v_x(S + j) - v_x(S)],$$

where $K$ is the number of covariates and $F$ the full set. With six covariates the sum has 64 distinct subsets, so the attribution can be computed exactly by enumeration rather than estimated by sampling. Exactness matters twice over: a sampled attribution carries Monte Carlo error that would be indistinguishable from genuine disagreement between architectures, and it would make the result depend on a random draw.

The construction admits one strong check. Summing the attributions telescopes every path through the subset lattice, so for any model and any observation

$$\sum_j phi_j(x) = v_x(F) - v_x(\emptyset) = f(x) - (1/|B|) \sum_{b \in B} f(b),$$

which is the local accuracy property: the attributions account exactly for the gap between the prediction at the observation and the mean prediction over the background. A residual larger than rounding error in this identity means the enumeration, the weights or the payoff is wrong, and the step reports the largest such residual over all models and observations so that the check cannot be skipped.

Every model in the set is interrogated on the same lattice, so the payoffs for all four architectures at all subsets and all observations are gathered into one query matrix and evaluated in a single pass. This matters in practice, since the architectures are refitted on every call and a naive loop over observations would refit them thousands of times.

```python
def exact_shapley_attributions(
    train_features: np.ndarray,
    train_response: np.ndarray,
    query_rows: np.ndarray,
    background: np.ndarray,
    model_config: dict,
) -> dict:
    """Compute exact Shapley attributions for every architecture at every query row by enumerating the covariate subsets.
 
    Parameters
    ----------
    train_features : np.ndarray
        Training covariates, shape (n_train, 6).
    train_response : np.ndarray
        Training response, shape (n_train,).
    query_rows : np.ndarray
        Observations to be explained, shape (n_query, 6).
    background : np.ndarray
        Background rows over which absent covariates are averaged, shape (n_background, 6).
    model_config : dict
        Settings under the keys n_knots, penalty_weight, ridge_nugget, n_trees, forest_depth, n_rounds, boost_depth, learning_rate, min_node and bandwidth.
 
    Returns
    -------
    dict
        Under the keys attributions, baselines, query_predictions and local_accuracy_residual.
 
    Raises
    ------
    ValueError
        When train_features is not a two-dimensional array of six columns with at least one row, when train_response does not match it in length, when query_rows or background is not a two-dimensional array of six columns with at least one row, when any entry fails to be finite, or when one of the ten configuration keys is absent.
    """
    return
```

### Step 8

08_08_explanation_discrepancy

Goal
----
08_explanation_discrepancy

Four architectures that agree on the response of a reef need not agree on why. Each supplies, at every observation, a vector of attributions over the covariates, and the question this step answers is how far those vectors stand apart. The answer has to be a single number per observation per reference model, comparable across observations, and large exactly where the architectures tell conflicting stories about the same reef in the same year.

Three choices turn a set of attribution vectors into such a number, and each is a decision rather than a detail.

The first is what to compare. The attribution vectors carry one entry per covariate, but not every covariate is of interest: the disturbance covariates are what a manager acts on, while a covariate carrying time is a nuisance direction along which the architectures are free to differ without any ecological consequence. Restricting the vectors to a stated subset of their entries before any norm is taken gives a projected measure that answers the question actually asked.

The second is how to normalise a distance between two vectors. A raw distance is not comparable across observations, because the size of an attribution vector varies for reasons unrelated to agreement: an observation whose prediction is far from the mean model output must carry a large vector, since the attributions sum to that gap, while one at the mean may carry a large vector or a small one, since large attributions can cancel. The distance must therefore be referred to a scale drawn from the explanations themselves. Which scale is the substance of the construction, and it is not the only available one, so the choice has consequences that an implementation must get right rather than guess.

The third is how to combine the pairwise comparisons and how to make the result readable. A comparison of one model against the rest leaves one number per model per observation, and the measure is asymmetric in the pair, so the role of reference model is a choice the analyst makes rather than a symmetry to be averaged away. The per-observation numbers are then referred to a scale computed across the whole evaluation set, which turns the measure into a signal-to-noise ratio in discrepancy space: values well above unity mark observations whose explanations disagree far more than the run of the data, and those are the observations that carry an ecological question a model cannot settle.

This step computes all of that from a stack of attribution vectors and a stated subset of covariates, returning the pairwise relative discrepancies, the per-reference summary, the normalising spread, and the final standardised measure. Which of the available scales and which centring convention the construction uses is the content of the step, and the values it returns depend on getting them right.

```python
def explanation_discrepancy(
    attributions: np.ndarray,
    feature_subset: tuple,
) -> dict:
    """Turn a stack of attribution vectors into the standardised explanation discrepancy measure for every reference model.
 
    Parameters
    ----------
    attributions : np.ndarray
        Attribution vectors, shape (n_models, n_observations, n_covariates).
    feature_subset : tuple
        Positions of the covariates the discrepancy is computed on.
 
    Returns
    -------
    dict
        Under the keys relative_discrepancy, mean_discrepancy, reference_spread, discrepancy_measure, largest_measure and smallest_projected_norm. The first axis of relative_discrepancy, mean_discrepancy, reference_spread and discrepancy_measure is the reference model, and the second axis of relative_discrepancy is the model compared with it.
 
    Raises
    ------
    ValueError
        When attributions is not a three-dimensional array with at least two models, at least two observations and at least one covariate, when an entry fails to be finite, when feature_subset is empty, holds a repeated or non-ascending position, or holds a position outside the covariate range, when the projected attribution vector of some model at some observation has zero length, which leaves the relative discrepancy undefined, or when some reference model's mean discrepancy is the same at every observation, which leaves its normalising spread at zero and the standardised measure undefined.
    """
    return
```

### Step 9

09_09_flag_consensus_outlier

Goal
----
09_flag_consensus_outlier

This final step runs the whole workflow from the ecological parameters to the single number the study reports, and it exists because the parts only mean something in sequence.

Step 01 advances mean hard coral cover at every site under logistic regrowth and lagged disturbance mortality, laying down the three disturbance fields from fixed integer maps so that the record carries a known mechanism and no random draw. Step 02 turns that record into a supervised problem, building the six covariates, dropping the two years for which the cyclone lags do not exist, and splitting the rows deterministically into training rows, evaluation rows and the background set the attributions average over. Steps 03, 04 and 05 supply three of the four architectures, the penalised additive model, the regression tree that both ensembles are built from, and the two ensembles themselves; step 06 adds the kernel smoother and puts the four on a common footing, fitted on the same rows and evaluated at the same query points. Step 07 explains every evaluation observation under every architecture, computing exact Shapley attributions by enumerating the covariate subsets against the background set, and checks them against the local accuracy identity. Step 08 turns the four attribution vectors at an observation into the standardised explanation discrepancy measure, once for each choice of reference model.

What remains is the reading of that measure, and it is the point of the exercise. A large measure under one reference model says that the observation stands apart when that model is taken as the reference. The statement is not symmetric, because the normalisation divides by the reference model's own explanation length, so an observation at which one model explains little shows a large discrepancy under that model and a smaller one under the others. Taking the smallest of the measures keeps only the part of the discrepancy that survives every choice of reference, which is the strictest form of the rule that an observation flagged under more than one reference is a consistent outlier rather than an artefact of one comparison. It does not apportion the disagreement: a single architecture that contradicts the rest at one observation raises the measure under every reference. The observation the framework directs to a domain expert first is the one whose smallest measure is largest.

That number, the largest over evaluation observations of the smallest measure over reference models, is what this step returns, along with the identity of the observation it belongs to, the four measures at that observation, the runner-up so that the separation can be judged, the out-of-sample skill of the four architectures, which the framework requires to be comparable before any explanation is compared, and the local accuracy residual of the attribution stage. It also returns the global ranking each architecture gives the covariates, ordered by the mean of the absolute attribution over the evaluation set with ties going to the lower covariate index, because the contrast between that ranking and what the architectures do at a single observation is the finding the whole construction exists to make.

```python
def flag_consensus_outlier(
    n_sites: int,
    n_years: int,
    ecology: dict,
    sampling: dict,
    model_config: dict,
    feature_subset: tuple,
) -> dict:
    """Run the whole workflow and report the strength and the identity of the most contested evaluation observation.
 
    Parameters
    ----------
    n_sites : int
        Number of monitoring sites.
    n_years : int
        Number of years in the record.
    ecology : dict
        Simulator settings under the keys growth_rate, carrying_capacity, cover_floor, cyclone, cyclone_lag1, cyclone_lag2, bleaching and other. The last five are the disturbance weights of step 01, with cyclone_lag1 and cyclone_lag2 the relative lag factors r_1 and r_2 inside the cyclone term.
    sampling : dict
        Row selection under the keys eval_stride and background_stride.
    model_config : dict
        Architecture settings under the keys n_knots, penalty_weight, ridge_nugget, n_trees, forest_depth, n_rounds, boost_depth, learning_rate, min_node and bandwidth.
    feature_subset : tuple
        Positions of the covariates the discrepancy is computed on.
 
    Returns
    -------
    dict
        Under the keys consensus_discrepancy, runner_up_discrepancy, local_accuracy_residual, flagged_row, flagged_site, flagged_year, flagged_reference_measures, eval_r_squared, reference_spread and global_ranking. flagged_row is the position of the flagged observation among the evaluation rows, counted from zero, not its index in the full design, and eval_r_squared is zero for every model when the evaluation response has no spread.
 
    Raises
    ------
    ValueError
        When one of the eight ecology keys or one of the two sampling keys is absent, or when any argument fails the validation of the step it is passed to.
    """
    return
```
