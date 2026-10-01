# Orders of magnitude of spurious cell-cell communication evidence in a confounded donor cohort

## Background

Cells in a tissue do not act alone. A ligand made in one cell type binds a cognate receptor on another and sets a transcriptional programme running in the receiver, and the map of which cell types talk to which, through which molecules, is the organising picture of immunology, development and tumour biology alike. Single-cell transcriptomics made that map look computable. Given an annotated catalogue of ligand receptor pairs and an expression matrix with cell type labels, a screen can ask which pairs are co-expressed above chance across the cell types of a sample, and tools built on that idea are now standard equipment. What such a screen actually measures, however, is coordination, not causation. Ligand expression in the sender, receptor expression in the receiver and the activity of the receiver's downstream pathway are all traits of the same donor, and anything that varies between donors, age, ancestry, infection history, cell composition, the batch a library was prepared in, or an unobserved cell state programme, pushes all three in the same direction at once. A screen with no way to tell such coordination from signalling will report communication wherever the coordination is strong, and no amount of extra donors will correct it: the displacement of the estimated ligand effect is a function of how strongly the latent factor loads, not of how large the cohort is, so the significance of a spurious pair improves as the study grows.

The way out has been standard in genetic epidemiology for twenty years. Germline genotype is fixed at conception, so a variant that raises expression of a particular gene is, in the absence of pleiotropy, independent of the environmental and cell state factors that confound everything else. Using such variants as instruments turns an association into a causal contrast, and single-cell expression quantitative trait locus cohorts, in which hundreds of donors have both genotypes and cell type resolved expression, now supply cis-eQTLs for both the ligand in the sender and the receptor in the receiver. Bringing instrumental variable reasoning to communication inference is not a matter of running an existing procedure, though, because the biology has a shape that the standard procedures do not. Multivariable Mendelian randomization handles several exposures but assumes their effects add; a ligand's effect is not additive in the receptor, since a signal can only be as strong as the receiver's capacity to receive it, so the quantity of interest is a ligand effect that slides with receptor abundance rather than a single number. Modelling that slide requires a product term between two endogenous exposures, and it is not obvious in advance that instrumenting each exposure separately still identifies the coefficients of a model containing their product.

There is a second difficulty of a different kind. A screen evaluates thousands of ligand receptor pathway combinations, so whatever is reported per combination has to double as a selection device. A frequentist tail probability is awkward in that role: it saturates once the evidence is strong, it is not a statement about the hypothesis, and thresholding it across a large family reopens the multiplicity problem. A Bayesian formulation in which each combination carries a latent indicator for the presence of communication, with a prior that concentrates the effect at zero when the indicator is off and leaves it free when the indicator is on, returns instead a posterior probability that communication is present. That number is interpretable on its own, comparable across combinations measured on different scales, and it regularises: unless the data supply real evidence, the effect is shrunk to nothing and the probability stays low. Fitting such a model needs the whole parameter vector, exposure coefficients included, to be sampled rather than plugged in, so that the uncertainty in the instrument stage propagates into the communication verdict instead of being silently discarded.

Putting the two ideas together makes a comparison possible that neither analysis can make about itself. Run the associational screen and the instrumented Bayesian analysis on the same donors, with the same covariates and the same ligand receptor pathway triplet, and the only thing that differs is what each is willing to assume about the unobserved. The distance between their verdicts, read on a common scale, is a direct measurement of what the associational paradigm costs, and its behaviour as the confounder loading, the cohort size and the true effect are varied tells a practitioner when a cheap co-expression screen is a reasonable first pass and when it is an engine for false discovery.

## Problem

A cell-cell communication screen asks whether ligand expression in a sender cell type drives downstream pathway activity in a receiver cell type, and the screens in routine use answer it from co-expression, which cannot separate signalling from a latent donor factor that loads on ligand, receptor and pathway alike. I have a donor level cohort of my own in which I know there is no communication whatsoever and in which exactly such a factor is present, and I want to know how much evidence for communication an associational analysis manufactures from it relative to an analysis that instruments both expression traits with their own cis-eQTLs and lets the ligand effect be modulated by receptor abundance in the receiver.

Cohort. There are $600$ donors, and with `numpy.random.default_rng(20260826)` I drew, in this order, a $600 \times 4$ matrix of ligand cis-eQTL dosages, a $600 \times 4$ matrix of receptor cis-eQTL dosages, a $600 \times 3$ matrix of donor covariates, a $600$ vector for the latent donor factor and then three $600$ vectors of residuals for ligand, receptor and pathway, every entry an independent standard normal. Ligand expression is the sum of its own four dosages at coefficient $0.45$ each, the three covariates at $0.25$ each, the latent factor at $0.7$ and its own residual, and receptor expression is built the same way from its own four dosages. Pathway activity is $0.5$ times receptor expression plus the three covariates at $0.25$ each plus the latent factor at $0.7$ plus its own residual, carrying no ligand term and no ligand by receptor term at all, and every column of the cohort is centred afterwards.

The two analyses. For the associational arm, regress pathway activity on an intercept, the observed ligand, the observed receptor, their elementwise product and the covariates, take the joint Wald $F$ test of the two ligand carrying coefficients on two numerator degrees of freedom, and let its communication score be one minus the upper tail probability. For the causal arm, fit one Bayesian model whose exposure and outcome equations are estimated jointly in a single sampler, in which the two component communication effect vector, the ligand main effect together with the ligand by receptor interaction effect, is given a spike and slab prior whose slab is a Zellner $g$ prior and whose spike is that same prior with its covariance multiplied by $\nu_1=10^{-4}$, and let its communication score be the posterior inclusion probability. Take $g=\min(n,100)$ for every Zellner $g$ prior, $\mathrm{IG}(3,2)$ for every residual variance, $\mathrm{Beta}(3,1)$ for the inclusion probability, a ridge of $10^{-6}$ in every matrix inverse, and $20000$ sampler sweeps with a $2000$ sweep burn in and a thinning factor of $5$ driven by `numpy.random.default_rng(2026)`. Writing the communication odds of a score $s$ as $s/(1-s)$, your final answer must be a single number: the base ten logarithm of the associational communication odds divided by the causal communication odds. Report alongside it the joint Wald statistic and its upper tail probability, the two communication scores and the base ten log odds each arm gives, and, briefly, the reasoning the causal arm's design and its prior and the comparison itself rest on.

Output Format Requirements: Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure. Rules: - The tags are required. Do not omit them or leave them empty. - The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose. - Put only that one number between the tags. No units, no words, no extra lines. Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_simulate_communication_dataset

Goal
----
Draw the donor level cohort in which ligand expression, receptor expression and receiver pathway activity are all loaded by one latent confounder, and return it as a single centred matrix that every later step reads.

```python
import numpy as np
def simulate_communication_dataset(n_donors: int, n_instruments: int, n_covariates: int,
                                   instrument_strength: float, covariate_effect: float,
                                   receptor_effect: float, confounder_loading: float,
                                   ligand_effect: float, interaction_effect: float,
                                   seed: int) -> np.ndarray:
    '''Draw one donor level cohort from the structural communication model.

    Every instrument, covariate, confounder and residual entry is an
    independent standard normal. Randomness comes from
    numpy.random.default_rng(seed), which draws, in this order, the ligand
    instrument matrix of shape (n_donors, n_instruments), the receptor
    instrument matrix of the same shape, the covariate matrix of shape
    (n_donors, n_covariates), the confounder vector of length n_donors, and
    then the ligand, receptor and pathway residual vectors, each of length
    n_donors.

    The ligand instruments act on ligand expression only and the receptor
    instruments on receptor expression only, every instrument carrying the
    same coefficient instrument_strength. The covariates act on all three
    observed quantities with the common coefficient covariate_effect. The
    confounder acts on all three with the common coefficient
    confounder_loading. Pathway activity is built from the uncentred ligand
    and receptor values, receiving ligand_effect times the ligand,
    receptor_effect times the receptor and interaction_effect times their
    elementwise product. Every column of the returned matrix is then centred
    to mean zero.

    Parameters
    ----------
    n_donors : int
        Number of donors, at least 2 * n_instruments + n_covariates + 5.
    n_instruments : int
        Number of cis-eQTL instruments for each of the two exposures,
        n_instruments >= 1.
    n_covariates : int
        Number of observed donor covariates, n_covariates >= 1.
    instrument_strength : float
        Common coefficient of every instrument on its own exposure.
    covariate_effect : float
        Common coefficient of every covariate on each observed quantity.
    receptor_effect : float
        Coefficient of receptor expression on pathway activity.
    confounder_loading : float
        Common coefficient of the unmeasured donor factor on each observed
        quantity.
    ligand_effect : float
        Coefficient of ligand expression on pathway activity.
    interaction_effect : float
        Coefficient of the ligand by receptor product on pathway activity.
    seed : int
        Seed of the random generator.

    Returns
    -------
    data : np.ndarray
        Array of shape (n_donors, 2 * n_instruments + n_covariates + 3) whose
        columns are, in order, the n_instruments ligand instruments, the
        n_instruments receptor instruments, the n_covariates covariates,
        ligand expression, receptor expression and pathway activity. Every
        column has mean zero.

    Raises
    ------
    ValueError
        If n_instruments or n_covariates is below one, if n_donors is below
        2 * n_instruments + n_covariates + 5, if seed is not an integer, or if
        any coefficient argument is not a finite real number.
    '''
    return data  # placeholder
```

### Step 2

02_compute_naive_communication_score

Goal
----
Score the cohort the way a co-expression screen does, by regressing pathway activity on the observed ligand, the observed receptor, their product and the covariates, and turning the joint test of the two ligand terms into a communication score on the logit scale.

```python
import numpy as np
def compute_naive_communication_score(data: np.ndarray, n_instruments: int,
                                      n_covariates: int) -> np.ndarray:
    '''Score a cohort with the associational regression that ignores the instruments.

    Pathway activity is regressed by ordinary least squares on a design whose
    columns are, in order, an intercept, ligand expression, receptor
    expression, the elementwise product of ligand and receptor expression, and
    the covariates. The two ligand carrying coefficients are tested jointly by
    the Wald F ratio on two numerator degrees of freedom and on the residual
    degrees of freedom of the fit, using the usual homoscedastic coefficient
    covariance estimate. The score is reported on the logit scale as the base
    ten logarithm of the ratio of the lower tail of that F distribution to its
    upper tail at the observed value.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_donors, 2 * n_instruments + n_covariates + 3) laid
        out as returned by the cohort simulation step.
    n_instruments : int
        Number of cis-eQTL instruments for each exposure, n_instruments >= 1.
    n_covariates : int
        Number of observed donor covariates, n_covariates >= 1.

    Returns
    -------
    summary : np.ndarray
        Array of four floats holding, in order, the observed F statistic, the
        base ten logarithm of the associational communication odds, the fitted
        ligand coefficient and the fitted ligand by receptor coefficient.

    Raises
    ------
    ValueError
        If data is not a two dimensional finite array with the column count
        implied by n_instruments and n_covariates, if either count is below
        one, if the cohort leaves no residual degrees of freedom, or if the
        design or the contrast covariance is numerically singular.
    '''
    return summary  # placeholder
```

### Step 3

03_initialise_gibbs_state

Goal
----
Build the starting parameter vector of the sampler from three least squares fits, packing the two exposure blocks, the outcome nuisance block, the communication effect vector and the selection parameters into one flat array.

```python
import numpy as np
def initialise_gibbs_state(data: np.ndarray, n_instruments: int, n_covariates: int,
                           a_rho: float, b_rho: float) -> np.ndarray:
    '''Build the starting parameter vector of the sampler by least squares.

    Ligand expression is regressed on the ligand instruments and the
    covariates, receptor expression on the receptor instruments and the
    covariates, and pathway activity on an intercept, the two fitted exposure
    values, their elementwise product and the covariates. Residual variances
    are the residual sums of squares divided by the residual degrees of
    freedom of their own fit, floored at 1e-8. The inclusion indicator starts
    at one and the inclusion probability at a_rho / (a_rho + b_rho).

    The returned vector has length 2 * n_instruments + 3 * n_covariates + 9
    and holds, in order, the ligand instrument coefficients, the ligand
    covariate coefficients, the ligand residual variance, the receptor
    instrument coefficients, the receptor covariate coefficients, the receptor
    residual variance, the outcome intercept, the outcome covariate
    coefficients, the receptor main effect, the outcome residual variance, the
    ligand main effect, the ligand by receptor interaction effect, the
    inclusion indicator and the inclusion probability.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_donors, 2 * n_instruments + n_covariates + 3) laid
        out as returned by the cohort simulation step.
    n_instruments : int
        Number of cis-eQTL instruments for each exposure, n_instruments >= 1.
    n_covariates : int
        Number of observed donor covariates, n_covariates >= 1.
    a_rho : float
        First shape of the beta prior on the inclusion probability, a_rho > 0.
    b_rho : float
        Second shape of the beta prior on the inclusion probability, b_rho > 0.

    Returns
    -------
    state : np.ndarray
        Array of 2 * n_instruments + 3 * n_covariates + 9 floats holding the
        starting value of every parameter in the order given above.

    Raises
    ------
    ValueError
        If data is not a two dimensional finite array with the column count
        implied by n_instruments and n_covariates, if either count is below
        one, if either beta shape is not a positive finite number, or if the
        cohort leaves no residual degrees of freedom for one of the three
        fits.
    '''
    return state  # placeholder
```

### Step 4

04_update_exposure_equations

Goal
----
Draw both exposure equations of one sampler sweep from the flat parameter state, refreshing the instrument coefficients, the covariate coefficients and the residual variance of the ligand equation and then those of the receptor equation.

```python
import numpy as np
def update_exposure_equations(data: np.ndarray, n_instruments: int, n_covariates: int,
                              state: np.ndarray, g_prior: float, a_sigma: float,
                              b_sigma: float, ridge: float,
                              rng: np.random.Generator) -> np.ndarray:
    '''Draw the ligand exposure equation and then the receptor exposure equation.

    The two exposure equations are visited in the order the state packs them,
    the ligand equation first and the receptor equation second, and the
    receptor equation conditions on the ligand equation as it has just been
    drawn rather than as it stood on entry. Within an equation the instrument
    coefficients are drawn first, then the covariate coefficients conditioning
    on those instrument coefficients, then the residual variance conditioning
    on both. No entry of the outcome block of the state is modified. Every
    matrix inverted along the way carries ridge added to its diagonal
    beforehand.

    Randomness comes from rng, which for each equation in turn draws one
    vector of n_instruments standard normals, then one vector of n_covariates
    standard normals, then one standard gamma variate. Each coefficient block
    is its conditional mean plus the lower Cholesky factor of its conditional
    covariance times its standard normal vector, and each residual variance is
    the scale of its inverse gamma conditional divided by its standard gamma
    variate.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_donors, 2 * n_instruments + n_covariates + 3) laid
        out as returned by the cohort simulation step.
    n_instruments : int
        Number of cis-eQTL instruments for each exposure, n_instruments >= 1.
    n_covariates : int
        Number of observed donor covariates, n_covariates >= 1.
    state : np.ndarray
        Array of 2 * n_instruments + 3 * n_covariates + 9 floats laid out as
        returned by the state initialisation step.
    g_prior : float
        Zellner g value shared by every coefficient prior, strictly positive.
    a_sigma : float
        Shape of the inverse gamma prior on a residual variance, a_sigma > 0.
    b_sigma : float
        Scale of the inverse gamma prior on a residual variance, b_sigma > 0.
    ridge : float
        Non negative diagonal regularisation added before any inversion.
    rng : np.random.Generator
        Generator supplying the draws, in the order described above.

    Returns
    -------
    exposures : np.ndarray
        Array of 2 * (n_instruments + n_covariates + 1) floats holding the
        drawn ligand instrument coefficients, ligand covariate coefficients
        and ligand residual variance, followed by the same three quantities of
        the receptor equation, in that order.

    Raises
    ------
    ValueError
        If data is not a two dimensional finite array with the column count
        implied by n_instruments and n_covariates, if either count is below
        one, if state is not a finite one dimensional array of the implied
        length, if any residual variance held in state is not strictly
        positive, if g_prior, a_sigma or b_sigma is not a positive finite
        number, if ridge is negative, or if a matrix that has to be inverted
        is not positive definite.
    '''
    return exposures  # placeholder
```

### Step 5

05_update_outcome_equation

Goal
----
Draw the four parameters of the outcome equation that carry no selection decision, the intercept, the outcome covariate coefficients, the receptor main effect and the residual variance of pathway activity, in that order and from the flat parameter state.

```python
import numpy as np
def update_outcome_equation(data: np.ndarray, n_instruments: int, n_covariates: int,
                            state: np.ndarray, g_prior: float, nu_spike: float,
                            a_sigma: float, b_sigma: float, ridge: float,
                            rng: np.random.Generator) -> np.ndarray:
    '''Draw the intercept, the covariate coefficients, the receptor main effect and the outcome variance.

    The four conditionals are visited in the order in which they are returned,
    and each one conditions on the values the earlier ones have just taken
    rather than on the values they held on entry. The communication effect
    vector and the inclusion indicator are read from state and are not
    modified, and neither is any entry of the two exposure blocks. The spike
    and slab scale factor is one when the inclusion indicator held in state is
    at least one half and nu_spike otherwise. Every matrix inverted along the
    way carries ridge added to its diagonal beforehand.

    Randomness comes from rng, which draws one vector of n_covariates + 2
    standard normals and then one standard gamma variate. The first entry of
    the normal vector drives the intercept, the next n_covariates entries
    drive the covariate block through the lower Cholesky factor of its
    conditional covariance, and the last entry drives the receptor main
    effect. The residual variance is the scale of its inverse gamma
    conditional divided by the standard gamma variate.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_donors, 2 * n_instruments + n_covariates + 3) laid
        out as returned by the cohort simulation step.
    n_instruments : int
        Number of cis-eQTL instruments for each exposure, n_instruments >= 1.
    n_covariates : int
        Number of observed donor covariates, n_covariates >= 1.
    state : np.ndarray
        Array of 2 * n_instruments + 3 * n_covariates + 9 floats laid out as
        returned by the state initialisation step.
    g_prior : float
        Zellner g value shared by the outcome priors, strictly positive.
    nu_spike : float
        Scale factor of the spike component, 0 < nu_spike <= 1.
    a_sigma : float
        Shape of the inverse gamma prior on the variance, a_sigma > 0.
    b_sigma : float
        Scale of the inverse gamma prior on the variance, b_sigma > 0.
    ridge : float
        Non negative diagonal regularisation added before any inversion.
    rng : np.random.Generator
        Generator supplying the draws, in the order described above.

    Returns
    -------
    outcome_block : np.ndarray
        Array of n_covariates + 3 floats holding the drawn intercept, the
        drawn outcome covariate coefficients, the drawn receptor main effect
        and the drawn residual variance of pathway activity, in that order.

    Raises
    ------
    ValueError
        If data is not a two dimensional finite array with the column count
        implied by n_instruments and n_covariates, if either count is below
        one, if state is not a finite one dimensional array of the implied
        length, if the outcome residual variance held in state is not strictly
        positive, if the inclusion indicator held in state is neither zero nor
        one, if g_prior, a_sigma or b_sigma is not a positive finite number,
        if nu_spike is outside the half open interval from zero to one, if
        ridge is negative, or if a matrix that has to be inverted is not
        positive definite.
    '''
    return outcome_block  # placeholder
```

### Step 6

06_update_communication_state

Goal
----
Draw the two-element communication effect vector from its spike and slab conditional, then draw the inclusion indicator that decides which component the vector belongs to, then refresh the inclusion probability from its own conditional.

```python
import numpy as np
def update_communication_state(data: np.ndarray, n_instruments: int, n_covariates: int,
                               state: np.ndarray, g_prior: float, nu_spike: float,
                               a_rho: float, b_rho: float, ridge: float,
                               rng: np.random.Generator) -> np.ndarray:
    '''Draw the communication effect vector, the inclusion indicator and the inclusion probability.

    The three conditionals are visited in the order in which they are
    returned. The effect vector is drawn against the inclusion indicator held
    in state, and the indicator is then drawn against the effect vector that
    has just been drawn rather than against the one held on entry. No entry of
    the two exposure blocks and no entry of the outcome nuisance block is
    modified. The spike and slab scale factor is one when the indicator being
    conditioned on is at least one half and nu_spike otherwise. Every matrix
    inverted along the way carries ridge added to its diagonal beforehand.

    Randomness comes from rng, which draws one vector of two standard normals,
    then one uniform variate, then one beta variate. The effect vector is its
    conditional mean plus the lower Cholesky factor of its conditional
    covariance times the standard normal vector. The indicator is one when the
    uniform variate is strictly below the conditional inclusion probability
    and zero otherwise. The beta variate has its first shape raised by the
    drawn indicator and its second by the complement of that indicator.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_donors, 2 * n_instruments + n_covariates + 3) laid
        out as returned by the cohort simulation step.
    n_instruments : int
        Number of cis-eQTL instruments for each exposure, n_instruments >= 1.
    n_covariates : int
        Number of observed donor covariates, n_covariates >= 1.
    state : np.ndarray
        Array of 2 * n_instruments + 3 * n_covariates + 9 floats laid out as
        returned by the state initialisation step.
    g_prior : float
        Zellner g value of the communication prior, strictly positive.
    nu_spike : float
        Scale factor of the spike component, 0 < nu_spike <= 1.
    a_rho : float
        First shape of the beta prior on the inclusion probability, a_rho > 0.
    b_rho : float
        Second shape of the beta prior on the inclusion probability, b_rho > 0.
    ridge : float
        Non negative diagonal regularisation added before any inversion.
    rng : np.random.Generator
        Generator supplying the draws, in the order described above.

    Returns
    -------
    communication_state : np.ndarray
        Array of five floats holding the drawn ligand main effect, the drawn
        ligand by receptor interaction effect, the drawn inclusion indicator,
        the conditional inclusion probability that indicator was drawn from,
        and the refreshed inclusion probability, in that order.

    Raises
    ------
    ValueError
        If data is not a two dimensional finite array with the column count
        implied by n_instruments and n_covariates, if either count is below
        one, if state is not a finite one dimensional array of the implied
        length, if the outcome residual variance held in state is not strictly
        positive, if the inclusion indicator held in state is neither zero nor
        one, if the inclusion probability held in state is not strictly inside
        the unit interval, if g_prior, a_rho or b_rho is not a positive finite
        number, if nu_spike is outside the half open interval from zero to
        one, if ridge is negative, or if the conditional covariance of the
        effect vector is not positive definite.
    '''
    return communication_state  # placeholder
```

### Step 7

07_summarise_communication_evidence

Goal
----
Put the associational verdict and the posterior inclusion probability on one common odds scale, report the gap between them in orders of magnitude, and express both causal communication coefficients in standard deviation units of pathway activity.

```python
import numpy as np
def summarise_communication_evidence(naive_summary: np.ndarray,
                                     posterior_inclusion_probability: float,
                                     posterior_ligand_effect: float,
                                     posterior_interaction_effect: float,
                                     ligand_sd: float, receptor_sd: float,
                                     pathway_sd: float) -> np.ndarray:
    '''Compare the associational and causal verdicts on a common odds scale.

    The associational log odds is taken ready made from the second entry of
    naive_summary and is not recomputed here. The causal communication score
    is posterior_inclusion_probability, and both scores are read as odds and
    reported in base ten logarithms. The gap is the associational communication
    odds divided by the causal communication odds, in those same logarithms.
    Both posterior effects are reported in standard deviation units of pathway
    activity.

    Parameters
    ----------
    naive_summary : np.ndarray
        Array of four floats as returned by the associational scoring step,
        whose second entry is the base ten logarithm of the associational
        communication odds.
    posterior_inclusion_probability : float
        Posterior probability that the communication effect vector belongs to
        the slab, strictly between zero and one.
    posterior_ligand_effect : float
        Posterior mean of the causal ligand main effect.
    posterior_interaction_effect : float
        Posterior mean of the causal ligand by receptor interaction effect.
    ligand_sd : float
        Standard deviation of ligand expression across donors, strictly
        positive.
    receptor_sd : float
        Standard deviation of receptor expression across donors, strictly
        positive.
    pathway_sd : float
        Standard deviation of pathway activity across donors, strictly
        positive.

    Returns
    -------
    evidence : np.ndarray
        Array of five floats holding, in order, the gap in base ten logarithms
        between the associational and causal communication odds, the
        associational log odds, the causal log odds, the standardised ligand
        main effect and the standardised interaction effect.

    Raises
    ------
    ValueError
        If naive_summary is not a finite array of four floats, if
        posterior_inclusion_probability is not strictly inside the unit
        interval, if either posterior effect is not finite, or if any of the
        three standard deviations is not a positive finite number.
    '''
    return evidence  # placeholder
```

### Step 8

08_run_causal_communication_pipeline

Goal
----
Chain the sub-problem functions 01-07 end-to-end on the confounded donor cohort and return the gap in base ten logarithms between the associational and the causal communication odds. Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (simulate_communication_dataset, compute_naive_communication_score, initialise_gibbs_state, update_exposure_equations, update_outcome_equation, update_communication_state, summarise_communication_evidence) rather than reimplementing them.

```python
# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def run_causal_communication_pipeline(n_donors: int = 600, n_instruments: int = 4,
                                      n_covariates: int = 3,
                                      instrument_strength: float = 0.45,
                                      covariate_effect: float = 0.25,
                                      receptor_effect: float = 0.5,
                                      confounder_loading: float = 0.7,
                                      ligand_effect: float = 0.0,
                                      interaction_effect: float = 0.0,
                                      data_seed: int = 20260826,
                                      n_iterations: int = 20000, burn_in: int = 2000,
                                      thin: int = 5, nu_spike: float = 1e-4,
                                      a_sigma: float = 3.0, b_sigma: float = 2.0,
                                      a_rho: float = 3.0, b_rho: float = 1.0,
                                      ridge: float = 1e-6,
                                      sampler_seed: int = 2026) -> float:
    '''Run the whole associational versus causal comparison on one donor cohort.

    The Zellner g value shared by every prior is the smaller of n_donors and
    one hundred. The sampler is driven by numpy.random.default_rng of
    sampler_seed and takes n_iterations sweeps, each visiting the two exposure
    equations, then the four unselected outcome parameters, then the
    communication effect vector and the selection parameters, in that order,
    with every drawn block written back into the parameter state before the
    next step of the same sweep reads it. Sweeps numbered burn_in and above
    are retained whenever their index minus burn_in is a multiple of thin. The
    posterior inclusion probability is the mean inclusion indicator over the
    retained sweeps, and the two posterior effects are the mean ligand main
    effect and the mean interaction effect over the same sweeps. The three
    standard deviations used to standardise those effects are the population
    standard deviations of ligand expression, receptor expression and pathway
    activity in the cohort.

    Parameters
    ----------
    n_donors : int
        Number of donors, at least 2 * n_instruments + n_covariates + 5.
    n_instruments : int
        Number of cis-eQTL instruments for each exposure, n_instruments >= 1.
    n_covariates : int
        Number of observed donor covariates, n_covariates >= 1.
    instrument_strength : float
        Common coefficient of every instrument on its own exposure.
    covariate_effect : float
        Common coefficient of every covariate on each observed quantity.
    receptor_effect : float
        Coefficient of receptor expression on pathway activity.
    confounder_loading : float
        Common coefficient of the unmeasured donor factor on each observed
        quantity.
    ligand_effect : float
        Coefficient of ligand expression on pathway activity.
    interaction_effect : float
        Coefficient of the ligand by receptor product on pathway activity.
    data_seed : int
        Seed of the generator that draws the cohort.
    n_iterations : int
        Number of sampler sweeps, n_iterations > burn_in.
    burn_in : int
        Number of leading sweeps discarded, burn_in >= 0.
    thin : int
        Thinning factor applied after the burn in, thin >= 1.
    nu_spike : float
        Scale factor of the spike component, 0 < nu_spike <= 1.
    a_sigma : float
        Shape of the inverse gamma priors on the residual variances.
    b_sigma : float
        Scale of the inverse gamma priors on the residual variances.
    a_rho : float
        First shape of the beta prior on the inclusion probability.
    b_rho : float
        Second shape of the beta prior on the inclusion probability.
    ridge : float
        Non negative diagonal regularisation used in every matrix inverse.
    sampler_seed : int
        Seed of the generator that drives the sampler.

    Returns
    -------
    evidence_gap : float
        Base ten logarithm of the associational communication odds divided by
        the causal communication odds, as a native Python float.

    Raises
    ------
    ValueError
        If any argument fails the validation of the step it is passed to, if
        n_iterations does not exceed burn_in, if thin is below one, or if the
        retained sweeps leave the posterior inclusion probability at exactly
        zero or exactly one, where the odds are undefined.
    '''
    return evidence_gap  # placeholder
```
