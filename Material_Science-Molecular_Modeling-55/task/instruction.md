# Material_Science-Molecular_Modeling-55

## Background

Force predictions from a machine-learned interatomic potential have errors that vary with the local atomic environment. A baseline uncertainty estimate may rank easy and difficult environments imperfectly and may be miscalibrated in magnitude. Post-hoc conformal methods can apply one global scale or separate scales for groups of environments. A smooth descriptor-dependent calibration can adapt continuously, while physically weighted fitting emphasizes larger force discrepancies. Generalization is particularly important when atomic environments or the electronic-structure setting shift between calibration and evaluation. The benchmark uses synthetic force vectors so that numerical implementations can be checked independently of a large atomistic data set.

## Problem

The attached paper studies post-hoc uncertainty calibration for atomic force predictions from machine-learned interatomic potentials, where a single global multiplier can be inadequate for unseen local environments. Apply its atomwise score, regular finite-sample conformal calibration, class-conditioned calibration, and smooth environment-dependent alternative to the deterministic force panel below. Compare the methods across the specified descriptor, uncertainty, and error shifts, and report the scalar audit value J. The panel generator, Gaussian-mixture implementation, random-feature basis, optimizer settings, transfer grid, and reduction are benchmark-instance conventions; use the paper for the scientific calibration method rather than treating those conventions as claims about its production code.

For the default configuration, explain why global positive rescaling preserves rank ordering and why a local calibrator may respond differently under covariate shift. Report q_global, score_checksum, class_checksum, flexible_objective, flexible_checksum, tensor_checksum, improvement, median_risk, risk_spread, worst_risk, leading_singular, transfer_risk, and J. Use float64, stable sorting and C-order traversal, and round only J to eight decimal places. The public orchestrator must execute Steps 1–6 and match the integration checks.

Benchmark instance specifications:
For the default run use seed=260427, n_cal=96, n_test=72, and d=8. Accept n_cal>=40, n_test>=24 and d>=6 as integers. Draw x_cal then x_test from default_rng(seed), each with standard-normal entries and shapes (n_cal,d), (n_test,d). Before constructing test forces, add 0.18*tanh(x_test[:,0])[:,None]*cos(arange(d)[None,:]+0.5) to x_test. Set W[j,k]=sin(0.371*(3*j+k+1)) for j=0..d-1 and k=0..2. For either descriptor panel x, draw an independent standard-normal z with shape (number of rows,3), first for calibration and then for test. Define predicted force f=x@W/sqrt(d)+0.12*[sin(x0*x1), cos(x2-x3), tanh(x4+x5)]. Define sigma=0.075+0.019*||x[:,:3]||_2+0.013*|sin(x3+0.4*x5)|. Let local=0.72+0.28*exp(0.35*x0-0.18*x2)+0.11*|x_{6 mod d}|; in particular, for d=6 this last coordinate is x0. Set m=sigma*local*(0.62+0.28*|z0|+0.10*|z1|), direction=z/||z||_2, and reference force y=f+m*direction. All operations are row-wise and float64. Step 2 applies the paper-derived atomwise force score and forms score_checksum by dotting the scores with 1+(index modulo 13)/17.

Step 3 applies the paper's finite-sample regular conformal rule at its operating alpha=0.5 and reports mean pinball loss, using the paper's loss orientation. Step 4 uses K=7 classes by default, with 2<=K<=min(20,n_cal//4), and standardizes calibration descriptors by mu0=mean(x,axis=0) and sd0=std(x,axis=0,ddof=0)+1e-12. Write z=(x-mu0)/sd0. Project onto p_j=cos(0.73*j+0.2)/sqrt(d); stable-sort this scalar projection and initialize component k's mean from the row at clipped zero-based index floor((k+0.5)*n_cal/K). Initialize all diagonal variances to var(z,axis=0,ddof=0)+0.15 and priors to 1/K. Run exactly 17 EM updates: log responsibilities use log(prior+1e-15)-0.5*sum_j[log(2*pi*variance)+(z-mean)^2/variance]; subtract each row maximum before exponentiating and normalizing. In the M-step use nk=sum_i resp_ik+1e-9, prior=nk/sum_k nk, mean=(resp.T@z)/nk and variance=sum_i resp_ik*(z_i-mean_k)^2/nk+0.025. AFTER the final M-step, recompute component log probabilities from the returned parameters and assign each calibration row by argmax, with first-index tie breaking. Apply the paper's class-conditional calibration rule within each component, falling back to the global value only when a component has fewer than three samples. Set class_checksum=dot(q_class,1..K)+dot(priors,(0..K-1)^2).

Step 5 builds H=[1,z,tanh(zW+b)] with width 13 by default, W[j,k]=[sin(0.173*(j*width+k+1))+0.37*cos(0.113*(j*width+k+3))]/sqrt(d), and b[k]=0.31*cos(0.47*k). The positive multiplicative calibrator is q_beta(x)=exp(clip(H(x)@beta,-2.5,2.5)). The benchmark's physical weight is w_i=0.25+error_i/median(error); fit the paper's physically weighted DIRECT force-space alignment target, not a log-score surrogate. Initialization only: set y_i=log(max(score_i,1e-12)) and D=diag(w_i), then solve (H.T D H+ridge I) beta=H.T D y with ridge=0.065; ridge applies to every coefficient, including the intercept. Run exactly iterations IRLS updates (default 31): r=y-H beta, v_i=w_i/sqrt(r_i^2+2.5e-5), then solve (H.T diag(v) H+ridge I) beta_next=H.T diag(v) y. The smoothing is inside the square root only. The DIRECT objective is the sample mean L(beta)=(1/n_cal) sum_i w_i*abs(sigma_i*q_beta(x_i)-error_i), with no ridge term. Its Adam subgradient is (1/n_cal) H.T [w_i*sign(sigma_i*q_beta(x_i)-error_i)*sigma_i*q_beta(x_i)*c_i], where c_i=1 for -2.5<(H beta)_i<2.5 and 0 otherwise, including at the clipping bounds. Start Adam moments at zero and make 10*iterations subgradient updates (default 310), with learning rate 0.001, moment coefficients 0.9 and 0.999, epsilon 1e-8, zero subgradient at an exact residual zero, and zero derivative outside the clipping interval. Track and return the iterate with the lowest direct objective, including the initializer. Return beta, descriptor mean and scale, direct objective, and flex_checksum=dot(beta,sin(index+0.7)). The optimizer and warm start are disclosed benchmark choices; the direct force-space target comes from the paper.

Step 6 evaluates shifts=(-0.45,-0.15,0.2,0.55), uncertainty_scales=(0.78,1,1.24), and correction_scales=(0.88,1.07,1.31). For each shift t, let x=x_test+t*cos(0.61*arange(d)+0.4); classify x with the FINAL mixture using z=(x-class_mu)/class_sd and the same diagonal-Gaussian log probabilities, then take q_cb=q_class[argmax(logp)]. Let q_flex=exp(clip(H(x)@beta,-2.5,2.5)) using the Step 5 feature map. For uncertainty scale u use sigma=sigma_test*u*(1+0.045*t^2). Model a descriptor-dependent target-functional force correction: c(x,t,e)=0.08*e*(1+0.25*t*tanh(x0))*[sin(x0+x2),cos(x1-x3),tanh(x4+x5)] and error=||(pred_test-ref_test)-c(x,t,e)||_2. The three predicted radii are q_global*sigma, q_cb*sigma and q_flex*sigma. Iterate shift, uncertainty scale, then correction scale in that order (36 cells). For each method compute alignment=mean(|radius-error|)/mean(error), coverage=mean(error<=radius), and rho as Pearson correlation of stable argsort ranks 0..n_test-1 of radius and error. To assess the paper's high-error selection use both nonoverlapping window sizes 6 and 9: in each window of at least two rows a hit occurs when the first-index argmax predicted radius equals the first-index argmax true error; window_accuracy is the mean over all retained windows of both sizes. Set k=min(max(2,ceil(n_test/6)),n_test-1); top_k_precision is the overlap fraction of the stable descending-radius and descending-error top-k index sets. Define risk=0.35*alignment+0.20*|coverage-0.5|+0.15*(1-rho)/2+0.15*(1-window_accuracy)+0.15*(1-top_k_precision). Store five channels per method (risk,coverage,rho,window_accuracy,top_k_precision), in global/class/flexible order. Form tensor_checksum by C-flattening the 36-by-15 tensor and dotting with 1+(index modulo 29)/31.

Step 7 uses the flexible-risk channel. Compute improvement as mean(global risk-flexible risk), median_risk, population risk_spread, worst_risk, the leading singular value of the centered three-method risk matrix, and transfer_risk as the mean flexible risk on the largest descriptor shift. Set J=exp(-(1.7*improvement+0.8*median_risk+0.6*risk_spread+0.25*worst_risk+0.15*transfer_risk+0.04*leading_singular))/(1+flexible_objective+pinball), then round only J to eight decimal places for the final answer.

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

01_generate_uncertainty_panel

Goal
----
Generate the deterministic descriptor, force, and baseline-uncertainty panel from the delivered specification. Require integer n_cal, n_test, and d with n_cal at least 40, n_test at least 24, and d at least 6. Preserve every default_rng draw in the stated order. Return x_cal, predicted and reference calibration forces, sigma_cal, x_test, predicted and reference test forces, and sigma_test.

```python
import numpy as np

def generate_uncertainty_panel(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8) -> tuple:
    """Return deterministic calibration and shifted-test force panels.

    Raises:
        ValueError: If a requested panel size or descriptor dimension is inadmissible.
    """
    return None, None, None, None, None, None, None, None
```

### Step 2

02_compute_site_scores

Goal
----
Call public Step 1 and compute the source-defined site-specific force calibration score for every calibration environment. Use the Euclidean norm of the force-vector discrepancy and the positive baseline uncertainty exactly as defined in the paper. Return the scores, force-error magnitudes, and the specified index-weighted score checksum.

```python
import numpy as np

def compute_site_scores(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8) -> tuple:
    """Return atomwise force scores, force errors, and their checksum.

    Raises:
        ValueError: If an upstream panel argument is inadmissible.
    """
    return None, None, None
```

### Step 3

03_fit_global_conformal

Goal
----
Call public Step 2. Require the miscoverage level alpha strictly between zero and one, so the target coverage is 1-alpha. Apply the paper's finite-sample conformal order statistic, clipping its rank to the available scores, to obtain the global multiplicative calibrator. Compute the mean pinball loss at quantile level 1-alpha, the level that this multiplier estimates, and return q_global, pinball, and the unchanged score checksum.

```python
import numpy as np

def fit_global_conformal(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8, alpha: float = 0.5) -> tuple:
    """Return the finite-sample global calibrator, pinball loss, and score checksum.

    Raises:
        ValueError: If alpha or an upstream panel argument is inadmissible.
    """
    return None, None, None
```

### Step 4

04_fit_class_conformal

Goal
----
Call public Steps 1 and 2, standardize the calibration descriptors, and fit the specified diagonal Gaussian mixture for exactly 17 EM updates. Require 2 <= n_classes <= min(20,n_cal//4) and the miscoverage level alpha strictly between zero and one. Recompute responsibilities from the final mixture parameters before assigning calibration labels. Apply the paper's class-conditional quantile rule within each component and use the global quantile only for a component with fewer than three members. Return class quantiles, mixture parameters, standardization arrays, and class_checksum.

```python
import numpy as np

def fit_class_conformal(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8, n_classes: int = 7, alpha: float = 0.5) -> tuple:
    """Return class calibrators, mixture parameters, standardization, and checksum.

    Raises:
        ValueError: If alpha, class count, or an upstream panel argument is inadmissible.
    """
    return None, None, None, None, None, None, None
```

### Step 5

05_fit_flexible_quantile

Goal
----
Call public Steps 1 and 2. Require width at least 5, iterations at least 5, and positive ridge. Build the disclosed smooth random-feature basis. Use the weighted log-score regression and fixed IRLS updates only to initialize beta; then perform the stated deterministic subgradient optimization on the physically weighted DIRECT force-space absolute error. Return the best beta, descriptor standardization, its attained direct force-space objective, and flexible_checksum.

```python
import numpy as np

def fit_flexible_quantile(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8, width: int = 13, iterations: int = 31, ridge: float = 0.065) -> tuple:
    """Return smooth quantile parameters, standardization, objective, and checksum.

    Raises:
        ValueError: If model settings or an upstream panel argument is inadmissible.
    """
    return None, None, None, None, None
```

### Step 6

06_evaluate_transfer_tensor

Goal
----
Call public Steps 1, 3, 4, and 5. Require each supplied tensor axis to contain at least two values. Evaluate every descriptor shift, uncertainty scale, and functional-correction scale in C order. At each cell apply the global scalar, the maximum-responsibility class scale, and the smooth descriptor scale; for each method store risk, coverage, stable-rank Spearman correlation, windowed high-error selection accuracy, and top-k precision. Return the complete fifteen-channel tensor and tensor_checksum.

```python
import numpy as np

def evaluate_transfer_tensor(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8, shifts: tuple = (-0.45,-0.15,0.2,0.55), uncertainty_scales: tuple = (0.78,1.0,1.24), correction_scales: tuple = (0.88,1.07,1.31)) -> tuple:
    """Return the complete transfer-risk tensor and its checksum.

    Raises:
        ValueError: If a tensor axis or upstream argument is inadmissible.
    """
    return None, None
```

### Step 7

07_compute_uncertainty_calibration_audit

Goal
----
This is the final orchestrator. Require an integer seed. Call public Steps 1 through 6 with the stated default path, including the full 4 by 3 by 3 tensor. From the three risk channels compute improvement, median_risk, population risk_spread, worst_risk, leading_singular, transfer_risk, and J exactly as specified. Return J rounded to eight decimals together with q_global, four checksums, the flexible direct-loss objective, and all six risk diagnostics. The protected implementation must make the identical chain using protected Steps 1 through 6 only.

```python
import numpy as np

def compute_uncertainty_calibration_audit(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8) -> tuple:
    """Return J and twelve named audit diagnostics from the full default path.

    Raises:
        ValueError: If seed or an upstream argument is inadmissible.
    """
    return None, None, None, None, None, None, None, None, None, None, None, None, None
```
