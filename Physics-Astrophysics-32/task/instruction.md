# Physics-Astrophysics-32

## Background

# Scientific background

The late time gravitational radiation from a binary black hole merger is described by damped quasinormal modes of the remnant Kerr black hole. Numerical relativity strain is recorded in spin-weighted spherical multipoles, while perturbative calculations use frequency dependent angular modes, so remnant inference must account for mode content and numerical uncertainty.

The anchor study supplies the trained standard kernel hyperparameters used in the numerical error covariance, the Gaussian likelihood used to score correlated resolution residuals, and the analytic amplitude inference framework used for remnant inference. It also discusses how incomplete mode content can distort ringdown fits. Those source dependent elements enter the dominant numerical calculation.

The displayed waveform, resolution differences, joint hierarchy for the highest resolution error and resolution difference, regime-specific variance ratios and correlations, radial transfers, angular truncation, response conversions, three coupled calibration regimes and their prior probabilities, real vector orders, proper Cartesian amplitude priors, equal conditional parent model odds, score normalization, mass prior, trial grid, quadrature, and two moment predictive closure are choices supplied for this instance. The regime correlations are explicit synthetic calibration assumptions: the observed residual informs the signed highest resolution numerical error, while the parent and child residuals share a correlated component. The resulting Gaussian update from parent to child is a supplied extension, not a construction attributed to the anchor. The mixed 220 by 320 response and the weighted six cell average over regimes and models are likewise synthetic extensions and are not presented as calculations from the anchor paper. The supplied spherical 64 record can also contain residual multipolar content outside this reduced child model, so its conditional score is not a claim that the child model alone provides a statistically complete fit.

## Problem

## Setup

Numerical relativity strain is supplied in spherical 22 and 64 multipoles, while the ringdown model is uncertain between a single Kerr parent and a two parent mixture with a nonlinear child response; infer the dimensionless remnant spin by marginalizing that model uncertainty with the registered anchor's correlated numerical error treatment, the cross mode residual hierarchy specified below, and the spherical 64 response.

## Inputs

Times use fixed reference mass units, each complex resolution difference is highest minus next highest resolution, and all displayed waveform, resolution, and transfer values are synthetic task inputs. For this synthetic instance, conditional on $(\chi_f,m)$ and its observed scale $A_\beta$, the displayed resolution difference and the additive numerical error in the corresponding highest resolution waveform follow the joint Gaussian hierarchy specified below; treat $A_\beta$ as an observed conditioning scale and do not assign it a density or integrate over it.

```text
t,Re_h22,Im_h22,Re_h64,Im_h64,Re_Dh22,Im_Dh22,Re_Dh64,Im_Dh64
0.00,0.607954649691,0.295051751110,0.000628378919,-0.000337532774,-0.037867242596,-0.001909207733,0.000044617819,-0.000067295330
3.00,0.147374778600,-0.432772573681,0.000144374885,0.000297055983,-0.013031289479,0.008698702522,-0.000040102718,-0.000121982286
6.50,-0.303886427800,0.075363262462,0.000046548106,0.000234305339,0.029983620352,0.019511427616,-0.000092589566,-0.000105810338
10.50,0.206496858961,0.222099527098,0.000017845774,0.000196568821,0.048606458833,0.001108395423,0.000046898531,-0.000025239149
15.00,-0.060781083081,-0.123253180953,0.000032917024,0.000059030411,0.059110971375,-0.059988566549,0.000061689148,-0.000106702268
20.00,0.087855912970,0.131424789334,-0.000048979985,-0.000053579051,0.031337457207,0.013158966817,-0.000027079682,0.000062232067
26.00,-0.044012797140,-0.023897529367,0.000015531811,0.000054570561,-0.034883268283,0.023852694832,-0.000009366161,-0.000023349673
```

The state table gives intrinsic dimensionless frequencies, spherical projections, and radial transfers, with every real and imaginary column interpolated linearly and separately in spin.

|chi|w220|w320|w640|mu22|mu32|rho4|rho5|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|0.70|0.532600243597-0.080792873136i|0.759174723145-0.084189645815i|1.541269014176-0.086754879659i|0.997417280443-0.009984915659i|-0.100258037153+0.008919401626i|-0.0146100910845+0.2685306133210i|-0.0000431583755+0.0057182161510i|
|0.72|0.541793731483-0.079990805121i|0.767600177103-0.083373330711i|1.558104443344-0.085893956678i|0.997176808792-0.010342555366i|-0.104222397641+0.008993049256i|-0.0135624224605+0.2594809831015i|-0.0000440202685+0.0058059579770i|
|0.74|0.551630332971-0.079092730022i|0.776508654282-0.082467481710i|1.575854074157-0.084937290752i|0.996913703333-0.010696053255i|-0.108307676796+0.009045655703i|-0.0125189913410+0.2504395563260i|-0.0000496081975+0.0058990191325i|
|0.76|0.562200718998-0.078081685942i|0.785958791816-0.081457397033i|1.594626097338-0.083869410692i|0.996624850763-0.011042407814i|-0.112526148250+0.009074892257i|-0.0114774776610+0.2413981510995i|-0.0000538512135+0.0059813295865i|
|0.78|0.573616428376-0.076936363177i|0.796021310470-0.080324671683i|1.614550655886-0.082671004345i|0.996306431311-0.011377667052i|-0.116892351938+0.009077706061i|-0.0104350677610+0.2323464427360i|-0.0000564632375+0.0060520126375i|
|0.80|0.586016974889-0.075629552389i|0.806782753193-0.079045861888i|1.635786572623-0.081317547659i|0.995953682283-0.011696506766i|-0.121423923726+0.009050514766i|-0.0093882979190+0.2232712080030i|-0.0000633699605+0.0061239641660i|
|0.82|0.599580345727-0.074125837403i|0.818350853396-0.077590486692i|1.658530992051-0.079777259961i|0.995560546571-0.011991671098i|-0.126142547285+0.008988807405i|-0.0083328258905+0.2141552033065i|-0.0000680304500+0.0061813114210i|
|0.84|0.614539083929-0.072378038263i|0.830862496923-0.075917929225i|1.683033655782-0.078007941274i|0.995119134957-0.012253083617i|-0.131075225463+0.008886290872i|-0.0072630847515+0.2049754391135i|-0.0000761402500+0.0062223026810i|
|0.86|0.631206038694-0.070321478391i|0.844496006954-0.073972411405i|1.709618885024-0.075951859914i|0.994618869600-0.012466362882i|-0.136256982615+0.008735425951i|-0.0061717338255+0.1957004072510i|-0.0000810477215+0.0062578720305i|
## Physical model

Evaluate the competing projected parent contents in the spherical 22 record and their driven spherical 64 child, recovering the paper's covariance and amplitude inference machinery and the trained pair $(\lambda_{\rm GP},\mu_{\rm GP})$ from the registered anchor; the following entries, including the nonlinear response scale, are supplied choices of this synthetic instance rather than conclusions to derive from the paper.

| Component | Configuration supplied for this task |
|---|---|
| Frequency and mass convention | $\widehat\omega=M_f\omega$ and $m=M_f/M_{\rm ref}$, with time domain carriers using $\omega=\widehat\omega/m$ and angular responses using $\widehat\omega$ so they are mass independent at fixed spin |
| Angular sectors | Spin weight $s=-2$, parent $(m,l)=(2,2)$ and $(2,3)$, and driven $(m,l)=(4,4)$ and $(4,5)$ |
| Angular discretization | Expand in spin-weighted spherical harmonics through $L=12$, with the single padding level $L=13$ included before squaring the cosine coupling matrix |
| Branch and phase | Select the right eigenvector nearest the target spherical eigenvalue in complex modulus, impose $b^\dagger b=1$ and $b_l>0$ real, and reject a tie between the two nearest branches within $10^{-10}$ |
| Angular contraction | Use $u_L=-\sqrt{(L+2)(L-1)}b_L$, the standard integer Wigner $3j$ Gaunt contraction with the driven eigenvector conjugated, and the spherical 64 projection $p_l=b_6$ |
| Nonlinear carriers | $\Omega_{22}=2\widehat\omega_{220}$ and $\Omega_{23}=\widehat\omega_{220}+\widehat\omega_{320}$, using the principal complex square root |
| Strain conversion for target $l$ | Multiply the self parent angular contraction by $-i\widehat\omega_{220}\sqrt{(l+2)!/(l-2)!}/48$ and the unequal parent contraction by $-i\sqrt{\widehat\omega_{220}\widehat\omega_{320}}\sqrt{(l+2)!/(l-2)!}/(2\sqrt{24\cdot120})$ |
| Radial transfer | Sum the $l=4,5$ driven contributions, using the displayed $\rho_l$ for the 220 by 220 response and the supplied $\kappa_l=-0.20\rho_l$ for the synthetic 220 by 320 extension |

For complex parent amplitudes $C$ and $E$, the child supplied for this task is

$$
h_{64}^{(2)}(t)=R_{64}^{22}C^2e^{-2i\omega_{220}t}+2R_{64}^{23}CEe^{-i(\omega_{220}+\omega_{320})t}.
$$

| Inference component | Configuration supplied for this task |
|---|---|
| Records and ordering | Use all seven samples, with every complex vector stacking all real rows before all imaginary rows and the parent coordinates ordered as $(\Re C,\Im C,\Re E,\Im E)$ |
| Parent models | Model $M_0$ uses only the interpolated $\mu_{22}$ projection, the 220 carrier, and the self coupled child; model $M_1$ uses the interpolated $\mu_{22}$ and $\mu_{32}$ projections, the 220 and 320 carriers, and both self and mixed child responses |
| Coupled calibration regimes | Marginalize the three paired values $(s_r,s_a,q,\rho,\gamma)=(0.75,0.55,0.40,0.82,0.10),(1.00,0.75,0.50,0.85,0.08),(1.25,1.20,0.48,0.82,0.30)$ with prior probabilities $p_r=(0.75,0.15,0.10)$ in the same order, where $s_r$ multiplies both nonlinear responses, every active Cartesian amplitude coordinate has the proper prior $N(0,s_a^2)$, and each regime requires its own covariance and conditional likelihood; these are three coupled regimes, not independent parameter grids |
| Inactive coordinates | Inactive $E$ coordinates in $M_0$ are absent, not integrated with a zero template |
| Numerical error scale | For the 22 block use the mass-scaled carrier $\omega_{220}=\widehat\omega_{220}/m$ and for the 64 block use $\omega_{640}=\widehat\omega_{640}/m$, set the observed conditioning scale $A_\beta=\max_t|\Delta_\beta(t)|$, and apply the anchor's standard correlated kernel with its retrieved trained pair, envelope cap $1.1A_\beta$, smooth minimum parameter $10^{-3}$, and the supplied $0.02A_\beta^2$ diagonal term in place of the paper's late time jitter construction |
| Cross mode residual covariance | In each regime let $K_{22}=L_{22}L_{22}^T$ and $K_{64}=L_{64}L_{64}^T$ be lower Cholesky factorizations. For both the real and imaginary pairs use that regime's $C=\gamma L_{22}L_{64}^T$, and form $D=\operatorname{diag}\!\left(\begin{bmatrix}K_{22}&C\\C^T&K_{64}\end{bmatrix},\begin{bmatrix}K_{22}&C\\C^T&K_{64}\end{bmatrix}\right)$ in $(\Re22,\Re64,\Im22,\Im64)$ order; cross blocks between real and imaginary components are exact zero |
| Joint resolution hierarchy | For the complete 28 component vectors $e$ and $d$ in that order, use the regime's $q$ and $\rho$ in $\operatorname{Cov}(e)=qD$, $\operatorname{Cov}(e,d)=\rho\sqrt q\,D$, and $\operatorname{Cov}(d)=D$ |
| Conditional waveform error | Condition the complete system before selecting mode blocks: $E[e\mid d]=\rho\sqrt q\,d$ and $K_{e\mid d}=q(1-\rho^2)D$ |
| Parent score | Apply analytic Gaussian amplitude marginalization to the spherical-22 data after subtracting $E[e_{22}\mid d_{22}]$ and using $\operatorname{Cov}(e_{22}\mid d_{22})$; for active design $X$ of dimension $d$, use $F=X^TK^{-1}X+s_a^{-2}I_d$, $V=F^{-1}$, $\bar a=VX^TK^{-1}y$, and $\log Z_p=-\tfrac12[y^TK^{-1}y-\bar a^TF\bar a+\log|K|+\log|F|+d\log(s_a^2)]$, omitting only the common $14\log(2\pi)$ constant |
| Child prediction | Partition $K_{e\mid d}$ into parent and child blocks, where $y_p$ is the observed spherical 22 vector and $m_p,m_c$ are the parent and child blocks of $E[e\mid d]$. With $G=K_{cp}K_{pp}^{-1}$, $B=GX$, $b=m_c+G(y_p-m_p)$, and row $i$ of $Q=\operatorname{Cov}(f,a)$ equal to $2\bar a^TA_iV$, use mean $E[f]+b-B\bar a$ and covariance $\operatorname{Cov}(f)+K_{cc}-GK_{pc}+BVB^T-QB^T-BQ^T$; this conditions the child numerical error on the observed parent waveform as well as on $d$, and for residual $r$, score the child as $-\tfrac12[r^T\Sigma^{-1}r+\log|\Sigma|]$, omitting only the common $14\log(2\pi)$ constant |
| Resolution score | Use the complete correlated residual score $\ell_\Delta=-\tfrac12[d^TD^{-1}d+\log|D|]$, with no additional score for the observed $A_\beta$ |
| Total state score | Within each model, factor the state likelihood as $p(d_{22},d_{64})p(h_{22},h_{64}\mid d_{22},d_{64},M_j)$ and add the conditional parent, conditional child, and resolution log scores before mass marginalization |
| Model and calibration uncertainty | Within regime $r$ give $M_0$ and $M_1$ equal conditional probability; marginalize the seven mass nodes separately in all six cells defined by regime and model, then combine their evidence at each spin with a six term log-sum-exp using cell weight $p_r/2$ before normalizing across spins; do not reuse one regime's covariance and do not average spin posteriors that were normalized separately |
| Spin prior | $\chi_k=0.738+0.010k$ for $k=0,\ldots,11$ with equal discrete weights |
| Mass prior | Uniform $m\in[0.96,1.04]$, integrated with the standard seven-point Gauss-Legendre nodes and weights mapped from $[-1,1]$ and normalized to unit total weight within each spin |

## Task

Return **the posterior mean dimensionless remnant spin** after conditioning the correlated waveform error system on the complete supplied residual record, conditioning the child error on the observed parent waveform, marginalizing the parent amplitudes and remnant mass in all six cells defined by the coupled calibration regimes and parent models, averaging their evidences at the stated prior odds, and normalizing over the twelve trial spins.

In `<reasoning>`:

- cite the anchor and report the retrieved $(\lambda_{\rm GP},\mu_{\rm GP})$
- give a concise justification grounded in the source for the data choice, correlated covariance, and analytic amplitude treatment, then state how the supplied cross mode hierarchy conditions the complete waveform error system and how the observed parent waveform updates the child prediction
- state how the proper amplitude prior makes evidence from models with different dimensions comparable and how the six regime-specific evidences are weighted before spin normalization
- let $L_k$ be the log score at trial spin $\chi_k$ after averaging over regimes and models and marginalizing mass, and let $L_\star=\max_k L_k$
- report only $L_\star$ and its trial spin, $S_0=\sum_k\exp(L_k-L_\star)$, and $S_1=\sum_k\chi_k\exp(L_k-L_\star)$, giving $L_\star$, $S_0$, and $S_1$ to at least seven decimal places
- compute the requested mean as $S_1/S_0$

## Numerical conventions

Require direct Cholesky success before every factored solve involving a joint, residual, conditional, parent, amplitude posterior precision, or child predictive covariance, while the additive quadratic predictive covariance need only be finite, symmetric, and positive semidefinite.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that
determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that
determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.
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

load ringdown data

Goal
----
Recorded spherical 22 and 64 ringdown multipoles and differences.

```python
import numpy as np
_BASE_DATA = np.array([
    [0.00, 0.607954649691, 0.295051751110, 0.000628378919, -0.000337532774, -0.037867242596, -0.001909207733, 0.000044617819, -0.000067295330],
    [3.00, 0.147374778600, -0.432772573681, 0.000144374885, 0.000297055983, -0.013031289479, 0.008698702522, -0.000040102718, -0.000121982286],
    [6.50, -0.303886427800, 0.075363262462, 0.000046548106, 0.000234305339, 0.029983620352, 0.019511427616, -0.000092589566, -0.000105810338],
    [10.50, 0.206496858961, 0.222099527098, 0.000017845774, 0.000196568821, 0.048606458833, 0.001108395423, 0.000046898531, -0.000025239149],
    [15.00, -0.060781083081, -0.123253180953, 0.000032917024, 0.000059030411, 0.059110971375, -0.059988566549, 0.000061689148, -0.000106702268],
    [20.00, 0.087855912970, 0.131424789334, -0.000048979985, -0.000053579051, 0.031337457207, 0.013158966817, -0.000027079682, 0.000062232067],
    [26.00, -0.044012797140, -0.023897529367, 0.000015531811, 0.000054570561, -0.034883268283, 0.023852694832, -0.000009366161, -0.000023349673],
], dtype=float)
def load_ringdown_data(amplitude_scale: float = 1.0, time_scale: float = 1.0) -> np.ndarray:
    """Return the supplied spherical-multipole record at a common scale.

    Parameters
    ----------
    amplitude_scale : float, default=1.0
        Positive multiplier applied to waveform and resolution-difference
        columns.
    time_scale : float, default=1.0
        Positive multiplier applied to every supplied time sample.

    Returns
    -------
    numpy.ndarray, shape (7, 9)
        Time; real and imaginary 22 and 64 multipoles; then real and
        imaginary two-resolution differences for those multipoles. Components
        are compared at numerical tolerance 1e-9.

    Raises
    ------
    ValueError
        If either scale is nonfinite or not strictly positive.
    """
    return None
import numpy as np
```

### Step 2

interpolate spin tables

Goal
----
Componentwise interpolation of the supplied Kerr-mode state table and pairing with remnant mass.

```python
import numpy as np
_RAW_TABLE = np.array([
    [0.70, 0.532600243597, -0.080792873136, 0.759174723145, -0.084189645815, 1.541269014176, -0.086754879659, 0.997417280443, -0.009984915659, -0.100258037153, 0.008919401626, -0.0146100910845, 0.2685306133210, -0.0000431583755, 0.0057182161510],
    [0.72, 0.541793731483, -0.079990805121, 0.767600177103, -0.083373330711, 1.558104443344, -0.085893956678, 0.997176808792, -0.010342555366, -0.104222397641, 0.008993049256, -0.0135624224605, 0.2594809831015, -0.0000440202685, 0.0058059579770],
    [0.74, 0.551630332971, -0.079092730022, 0.776508654282, -0.082467481710, 1.575854074157, -0.084937290752, 0.996913703333, -0.010696053255, -0.108307676796, 0.009045655703, -0.0125189913410, 0.2504395563260, -0.0000496081975, 0.0058990191325],
    [0.76, 0.562200718998, -0.078081685942, 0.785958791816, -0.081457397033, 1.594626097338, -0.083869410692, 0.996624850763, -0.011042407814, -0.112526148250, 0.009074892257, -0.0114774776610, 0.2413981510995, -0.0000538512135, 0.0059813295865],
    [0.78, 0.573616428376, -0.076936363177, 0.796021310470, -0.080324671683, 1.614550655886, -0.082671004345, 0.996306431311, -0.011377667052, -0.116892351938, 0.009077706061, -0.0104350677610, 0.2323464427360, -0.0000564632375, 0.0060520126375],
    [0.80, 0.586016974889, -0.075629552389, 0.806782753193, -0.079045861888, 1.635786572623, -0.081317547659, 0.995953682283, -0.011696506766, -0.121423923726, 0.009050514766, -0.0093882979190, 0.2232712080030, -0.0000633699605, 0.0061239641660],
    [0.82, 0.599580345727, -0.074125837403, 0.818350853396, -0.077590486692, 1.658530992051, -0.079777259961, 0.995560546571, -0.011991671098, -0.126142547285, 0.008988807405, -0.0083328258905, 0.2141552033065, -0.0000680304500, 0.0061813114210],
    [0.84, 0.614539083929, -0.072378038263, 0.830862496923, -0.075917929225, 1.683033655782, -0.078007941274, 0.995119134957, -0.012253083617, -0.131075225463, 0.008886290872, -0.0072630847515, 0.2049754391135, -0.0000761402500, 0.0062223026810],
    [0.86, 0.631206038694, -0.070321478391, 0.844496006954, -0.073972411405, 1.709618885024, -0.075951859914, 0.994618869600, -0.012466362882, -0.136256982615, 0.008735425951, -0.0061717338255, 0.1957004072510, -0.0000810477215, 0.0062578720305],
], dtype=float)
def interpolate_spin_tables(spins: np.ndarray, remnant_mass_ratios: np.ndarray | None = None) -> np.ndarray:
    """Interpolate state-table columns and pair them with remnant-mass ratios.

    Parameters
    ----------
    spins : array_like, shape (n,)
        Trial dimensionless spins in the closed interval [0.70, 0.86].
    remnant_mass_ratios : array_like, shape (n,), optional
        Finite positive values of M_f/M_ref paired row by row with the spins.
        When omitted, use one for every row.

    Returns
    -------
    numpy.ndarray, shape (n, 16)
        Spin; remnant-mass ratio; intrinsic dimensionless 220, 320, and 640
        frequencies; two parent projections; two
        task-defined synthetic radial response coefficients used with the
        step-03 mixed-parent scaling convention.
        Components are compared at numerical tolerance 1e-9.

    Raises
    ------
    ValueError
        If spins is empty, nonfinite, not one-dimensional, or outside the
        closed interpolation interval, or if remnant_mass_ratios has the wrong
        shape or contains a nonfinite or nonpositive value.
    """
    return None
import numpy as np
```

### Step 3

synthesize quadratic response

Goal
----
Angular construction of two spherical quadratic responses.

```python
import numpy as np
def synthesize_quadratic_response(table: np.ndarray) -> np.ndarray:
    """Construct the self-coupled and mixed-parent responses in spherical 64.

    Parameters
    ----------
    table : numpy.ndarray, shape (n, 16)
        Interpolated table in the exact column order declared by step 02.

    Returns
    -------
    numpy.ndarray, shape (n, 4)
        Real and imaginary parts of the intrinsic spherical-64 220-by-220
        response followed by the 220-by-320 response, compared at tolerance
        1e-9. No cells are unused.

    Raises
    ------
    ValueError
        If the table is empty, nonfinite, has a shape other than `(n, 16)`,
        has spin outside `[0.70, 0.86]`, has nonpositive mass, or has a 220
        or 320 frequency whose real part is not positive or imaginary part
        is not negative. Also raised if an angular eigensystem is nonfinite,
        ambiguous at the selected branch, or cannot be normalized with the
        required phase convention.

    Notes
    -----
    Use Python's standard `math.factorial`, not the removed `numpy.math`
    namespace. Cast parity exponents to built-in `int` before applying `(-1)`
    or evaluate parity directly, so negative NumPy-integer exponents cannot
    trigger an environment-dependent exception.
    """
    return None
import math
import numpy as np
```

### Step 4

build linearized templates

Goal
----
Build the mixed-mode parent design used by the analytic amplitude posterior.

```python
import numpy as np
def build_linearized_templates(
    data: np.ndarray,
    table: np.ndarray,
    quadratic_response: np.ndarray,
    include_mixed_parent: bool = True,
) -> np.ndarray:
    """Build the real parent design and pack the child-response metadata.

    Parameters
    ----------
    data : numpy.ndarray, shape (7, 9)
        Finite ringdown record with strictly increasing sample times.
    table : numpy.ndarray, shape (n, 16)
        Interpolated state table in the exact step-02 column order.
    quadratic_response : numpy.ndarray, shape (n, 4)
        Real and imaginary parts of the intrinsic spherical-64 responses
        `(R64^22, R64^23)` from step 03.
    include_mixed_parent : bool, default=True
        Exact boolean. False leaves the 320 design columns at exact zero;
        true activates them.

    Returns
    -------
    numpy.ndarray, shape (n, 18, 4)
        Rows `0:14` contain the real parent design. Rows `14:18` contain,
        respectively, `(spin,mass,Re omega220,Im omega220)`,
        `(Re omega320,Im omega320,Re omega640,Im omega640)`, both complex
        responses, and both complex spherical projections. Components are
        compared at tolerance `1e-9`.

    Raises
    ------
    ValueError
        If the flag is not an exact boolean; if an array has an incompatible
        shape or nonfinite entry; if times are not strictly increasing; if a
        mass is not positive; or if a physical frequency lacks positive real
        and negative imaginary parts.
    """
    return None
import numpy as np
```

### Step 5

build correlated covariances

Goal
----
Numerical-error covariances for the two spherical multipoles.

```python
import numpy as np
def build_correlated_covariances(
    data: np.ndarray,
    block_frequencies: np.ndarray,
    error_model: str = "anchor_gp",
    error_amplitude_scale: float = 1.0,
    error_period_scale: float = 1.0,
    highest_error_variance_ratio: float = 0.5,
    error_residual_correlation: float = 0.85,
    cross_mode_correlation: float = 0.08,
) -> np.ndarray:
    """Construct the joint waveform-error and residual covariance.

    Parameters
    ----------
    data : numpy.ndarray, shape (7, 9)
        Finite record containing times and both complex resolution differences.
    block_frequencies : numpy.ndarray, shape (n, 4)
        `(Re omega220, Im omega220, Re omega640, Im omega640)` for each state.
    error_model : str, default="anchor_gp"
        One of `"anchor_gp"`, `"generic_exponential"`, or `"white_noise"`.
        For `"white_noise"`, every unscaled mode-covariance diagonal equals
        `0.20*A`; this quantity is the covariance entry and is not squared.
    error_amplitude_scale : float, default=1.0
        Finite positive multiplier on standard deviation; its square multiplies
        the complete covariance.
    error_period_scale : float, default=1.0
        Finite positive multiplier on correlated-model periods. It is checked
        but inactive for white noise.
    highest_error_variance_ratio : float, default=0.5
        Finite positive ratio `q` between the marginal highest-resolution
        error covariance and the resolution-residual covariance, common to the
        two modes.
    error_residual_correlation : float, default=0.85
        Finite common correlation coefficient `rho` between the highest-
        resolution waveform error and observed residual, strictly between
        -1 and 1.
    cross_mode_correlation : float, default=0.08
        Finite residual correlation coefficient `gamma` between the spherical
        22 and 64 modes, strictly between -1 and 1. Its oriented cross block is
        `gamma * chol(K22) @ chol(K64).T` for both real and imaginary parts.

    Returns
    -------
    numpy.ndarray, shape (n, 56, 56)
        Joint covariance of `z=(e,d)`, with each 28-vector in
        `(Re22,Re64,Im22,Im64)` block order. Every cell is compared at
        tolerance `1e-9`.

    Raises
    ------
    ValueError
        If an array has an incompatible shape or nonfinite entry; times are not
        strictly increasing; a carrier has an invalid sign; a scale is not
        finite and positive; either correlation is outside the open interval
        `(-1, 1)`; a resolution-difference block has zero amplitude; the model
        name is unsupported; or a covariance produced within the stated input
        domain is not finite, symmetric, and positive definite.
    """
    return None
import numpy as np
```

### Step 6

evaluate conditional posteriors

Goal
----
Analytic Cartesian-amplitude posterior for the spherical-22 parent.

```python
import numpy as np
def evaluate_conditional_posteriors(
    data: np.ndarray,
    templates: np.ndarray,
    covariances: np.ndarray,
    amplitude_prior_scale: float = 0.75,
) -> np.ndarray:
    """Evaluate the analytic mixed-parent amplitude posterior at every state.

    Parameters
    ----------
    data : numpy.ndarray, shape (7, 9)
        Finite ringdown record.
    templates : numpy.ndarray, shape (n, 18, 4)
        Step-04 parent designs and packed state metadata.
    covariances : numpy.ndarray, shape (n, 56, 56)
        Finite joint numerical-error covariances in step-05 `z=(e,d)` order.
    amplitude_prior_scale : float, default=0.75
        Finite positive standard deviation `s_a` of each active Cartesian
        amplitude coordinate. The same scale is used in both parent models.

    Returns
    -------
    numpy.ndarray, shape (n, 25)
        `(spin,mass,mean[4],covariance[16],log_Z_parent,S220,S320)` with the
        covariance flattened in C row-major order. Components are compared at
        tolerance `1e-9`.

    Raises
    ------
    ValueError
        If an input has an incompatible shape or nonfinite entry; state rows
        disagree; a supplied joint covariance is asymmetric, singular, or not
        positive definite; `amplitude_prior_scale` is nonpositive or nonfinite;
        a derived covariance cannot be factored; the
        active design is rank deficient; or an active posterior precision or
        two-dimensional marginal covariance is not positive definite.
    """
    return None
import numpy as np
```

### Step 7

score heldout h64

Goal
----
Posterior prediction of the driven spherical-64 child.

```python
import numpy as np
def score_heldout_h64(
    data: np.ndarray,
    templates: np.ndarray,
    covariances: np.ndarray,
    posteriors: np.ndarray,
    response_scale: float = 1.0,
) -> np.ndarray:
    """Score the nonlinear child predicted by the analytic parent posterior.

    Parameters
    ----------
    data : numpy.ndarray, shape (7, 9)
        Finite ringdown record containing the highest-resolution waveforms and
        their observed resolution differences.
    templates : numpy.ndarray, shape (n, 18, 4)
        Parent designs and nonlinear response metadata from step 04.
    covariances : numpy.ndarray, shape (n, 56, 56)
        Joint numerical covariances in step-05 `z=(e,d)` order.
    posteriors : numpy.ndarray, shape (n, 25)
        Analytic parent posterior rows from step 06.
    response_scale : float, default=1.0
        Finite nonnegative multiplier on both nonlinear response coefficients.
        Zero applies the zero-response control.

    Returns
    -------
    numpy.ndarray, shape (n, 8)
        `(spin,mass,parent_log_score,child_log_score,total_log_score,
        child_mahalanobis,Re predicted_child[0],Im predicted_child[0])`.
        `total_log_score` is the conditional parent score plus the conditional
        child score plus the two-mode resolution score. The final two entries
        are the total conditional child predictive mean at the first sample.
        `child_mahalanobis` is exactly `r.T @ solve(Sigma, r)`, not its
        square root.
        Components are compared at tolerance `1e-9`.

    Raises
    ------
    ValueError
        If an input has an incompatible shape or nonfinite entry; state rows
        disagree; response_scale is negative or nonfinite; inactive parent
        packing is nonzero; either amplitude-significance column lies outside
        `[0, 1]`; a supplied joint covariance is asymmetric, singular, or not
        positive definite; or a derived covariance cannot be factored.
    """
    return None
import numpy as np
```

### Step 8

normalize spin weights

Goal
----
Marginalize remnant mass and normalize the spin posterior.

```python
import numpy as np
def normalize_spin_weights(
    predictive_scores: np.ndarray,
    mass_log_weights: np.ndarray,
) -> np.ndarray:
    """Marginalize the mass nodes and return normalized spin weights.

    Parameters
    ----------
    predictive_scores : numpy.ndarray, shape (n, 8)
        Step-07 rows ordered with spin outer and mass inner.
    mass_log_weights : numpy.ndarray, shape (n,)
        Finite log quadrature weights paired with the rows. They are
        normalized separately inside every spin.

    Returns
    -------
    numpy.ndarray, shape (s, 7)
        `(spin,log_marginal,weight,E[m],E[parent_score],E[child_score],
        E[child_mahalanobis])`, compared at tolerance `1e-9`.

    Raises
    ------
    ValueError
        If an input has an incompatible shape or nonfinite entry; masses are
        not positive; a spin has duplicate masses; or spin groups have unequal
        counts.
    """
    return None
import numpy as np
```

### Step 9

infer remnant spin

Goal
----
Final binary-black-hole remnant-spin inference.

```python
import numpy as np
def infer_remnant_spin(
    response_scale: float = 1.0,
    error_model: str = "anchor_gp",
    amplitude_scale: float = 1.0,
    time_scale: float = 1.0,
    amplitude_prior_scale: float = 1.0,
) -> float:
    """Return the posterior mean remnant spin for the requested configuration.

    Parameters
    ----------
    response_scale : float, default=1.0
        Finite nonnegative multiplier on all three coupled-regime nonlinear
        response scales. The task answer uses one; zero applies the zero-
        response option in every regime.
    error_model : str, default="anchor_gp"
        One of `"anchor_gp"`, `"generic_exponential"`, or `"white_noise"`.
        The task answer uses `"anchor_gp"`.
    amplitude_scale : float, default=1.0
        Finite positive multiplier on all waveform and resolution-difference
        columns.
    time_scale : float, default=1.0
        Finite positive multiplier on all recorded times.
    amplitude_prior_scale : float, default=1.0
        Finite positive multiplier on all three coupled-regime Cartesian
        amplitude-prior standard deviations.

    Returns
    -------
    float
        Posterior mean dimensionless remnant spin after analytic amplitude,
        mass, coupled calibration-regime, and equal conditional parent-model
        marginalization of state scores that include residual-conditioned
        waveform likelihoods and the residual likelihood, compared at
        tolerance `1e-9`.

    Raises
    ------
    ValueError
        If response_scale is negative or nonfinite, a data scale or the
        amplitude-prior multiplier is not finite and positive, or error_model
        is unsupported.
        Errors raised by an earlier step retain their documented meaning.
    """
    return None
import numpy as np
```
