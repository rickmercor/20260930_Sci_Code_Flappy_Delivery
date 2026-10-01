# Biology-Ecology-5

## Problem

Microbial cross-feeding can alter both a mutant's chance of fixation and the time taken by the lineages that eventually fix. Compute the mixed environmental sensitivity \(\left.\partial_\varepsilon\partial_\zeta\ln\mathbb E_{f_0}[T\mid f(T)=1]\right|_{(0,0)}\), where \(T\) is the first hitting time of either frequency boundary in the rapidly relaxing generalized consumer–resource reduction with a fixed combined focal abundance.

Use saturating uptake \(q_{i\alpha}(R_\alpha)=U_{i\alpha}R_\alpha/(h_{i\alpha}+R_\alpha)\), per-capita growth \((1-\ell)\sum_\alpha q_{i\alpha}-m_i\), and resource balance \(\dot R=K-R-(I-\ell B)q(R)^TN\), with unit resource energies, conversion efficiency and dilution, \(\ell=0.35\), and these exact dimensionless inputs, where the first three rows are cavity residents and the last two rows are the parent and mutant:
\[
U=\begin{pmatrix}1&.2&.1&.05\\.1&.9&.25&.1\\.15&.1&.8&.3\\.4&.3&.2&.1\\.425&.28&.215&.09\end{pmatrix},\quad
h=\begin{pmatrix}.4&.7&.3&.8\\.6&.2&.9&.5\\.3&.8&.4&.6\\.5&.4&.7&.3\\.52&.38&.72&.28\end{pmatrix},\quad
B=\begin{pmatrix}0&.1&.2&.3\\.5&0&.3&.2\\.3&.6&0&.5\\.2&.3&.5&0\end{pmatrix}.
\]
Here \(B_{\alpha\beta}\) converts consumed resource \(\beta\) to leaked resource \(\alpha\); the cavity state is \(N^*=(1.2,.9,1.1)^T\), \(R^*=(1,.8,1.2,.9)^T\), and its fixed mortality vector and baseline supply are defined by this state's stationary equations using only the first three uptake rows. The focal parameters are \(m_p=.4\), \(m_m=.4029148343\), \(N_0=.1\), \(D=2\times10^{-8}\), and \(f_0=.07\), with independent lineage abundance-noise variance rate \(2DN\).

Set \(K(\varepsilon,\zeta)=K_0+\varepsilon(.3,-.2,.15,-.1)^T+\zeta(-.1,.25,.2,-.15)^T\) and follow the local stationary cavity branch with unchanged resident composition, holding all remaining inputs fixed. Apply static environmental response and the generalized frequency-dependent mutation reduction, evaluating growth gradients and focal impacts at each cavity equilibrium and retaining the parent-induced invasion correction. Include the mixed dependence of the stationary state, growth gradients, resource susceptibility and focal impacts, and evaluate the conditional first-passage moment within the resulting diffusion with absorbing endpoints zero and one. Return one finite decimal for the mixed logarithmic derivative; in the reasoning identify the conditioning event and give the baseline invasion fitness, feedback coefficient, fixation probability, conditional mean time, and the two first derivatives and mixed derivative of that conditional mean as the only supporting numerical quantities.

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

01_uptake_tensors.py

Goal
----
Calculate resource derivatives of saturating uptake through third order.

```python
def uptake_tensors(U: 'np.ndarray', half: 'np.ndarray', R: 'np.ndarray') -> 'np.ndarray':
    """Calculate resource derivatives of saturating uptake through third order.

    Parameters
    ----------
    U : np.ndarray
        Nonnegative maximum uptake rates (P,M), P and M positive.
    half : np.ndarray
        Strictly positive half-saturation constants (P,M).
    R : np.ndarray
        Strictly positive resource abundances (M,).

    Returns
    -------
    result : np.ndarray
        Shape (4,P,M). Entry [k,i,a] is the kth ordinary derivative in
        resource a, k=0,1,2,3, of U[i,a]*R[a]/(half[i,a]+R[a]).
        Derivatives are not divided by factorials; other-resource derivatives vanish.

    Raises
    ------
    ValueError
        If inputs are nonreal, nonfinite, have incompatible or empty shapes,
        any U is negative, any half or R is nonpositive, or a result is nonfinite.
    """
    return result
```

### Step 2

02_cavity_state_jet.py

Goal
----
Calculate the two first responses and mixed stationary response of the cavity community.

```python
def cavity_state_jet(tensors: 'np.ndarray', B: 'np.ndarray', leakage: float, N: 'np.ndarray', R: 'np.ndarray', v: 'np.ndarray', w: 'np.ndarray') -> 'np.ndarray':
    """Calculate the two first responses and mixed stationary response of the cavity community.

    Parameters
    ----------
    tensors : np.ndarray
        Finite (4,S,M) uptake derivatives from uptake_tensors for residents,
        1 <= S <= M. The derivative-order axis is not factorial-scaled.
    B : np.ndarray
        Nonnegative (M,M) conversion matrix, columns sum to one within 1e-12.
    leakage : float
        Common leakage fraction in [0,1).
    N : np.ndarray
        Positive resident abundances (S,).
    R : np.ndarray
        Positive resource abundances (M,).
    v : np.ndarray
        First signed supply direction (M,).
    w : np.ndarray
        Second signed supply direction (M,).

    Returns
    -------
    result : np.ndarray
        Shape (4,S+M), rows value, epsilon derivative, zeta derivative,
        mixed epsilon-zeta derivative of concatenated (N,R) at zero.
        Resident growth is (1-leakage)*sum_a q[i,a]-mortality[i];
        resource loss is R+(I-leakage*B)@q.T@N. Baseline mortality
        and supply are defined by the given state and are held fixed except
        for supply K0+epsilon*v+zeta*w. Preserve the resident set.
        No mixed supply forcing is present. The coupled Jacobian is nonsingular.

    Raises
    ------
    ValueError
        If inputs are nonreal or nonfinite, stated shapes or positivity fail,
        leakage is outside [0,1), B is negative or not column-stochastic
        to 1e-12, the coupled response is singular, or a result is nonfinite.
    """
    return result
```

### Step 3

03_susceptibility_jet.py

Goal
----
Calculate the resource susceptibility and its full mixed environmental response.

```python
def susceptibility_jet(tensors: 'np.ndarray', B: 'np.ndarray', leakage: float, state: 'np.ndarray') -> 'np.ndarray':
    """Calculate the resource susceptibility and its full mixed environmental response.

    Parameters
    ----------
    tensors : np.ndarray
        Finite (4,S,M) resident uptake derivatives, 1 <= S <= M.
    B : np.ndarray
        Nonnegative (M,M) column-stochastic conversion, tolerance 1e-12.
    leakage : float
        Common leakage fraction in [0,1).
    state : np.ndarray
        Finite (4,S+M) stationary state jet from cavity_state_jet; rows
        value, epsilon, zeta, epsilon-zeta, with no factorial scaling.

    Returns
    -------
    result : np.ndarray
        Shape (4,M,M), jet of the resource-resource block of the inverse
        stationary Jacobian. This Jacobian differentiates resident per-capita
        growth and resource loss with respect to (N,R), for the same microbial
        model as cavity_state_jet. All blocks are evaluated along state.
        Include growth-gradient, resident-impact and resource-loss changes.
        Only the full coupled matrix must be nonsingular; no separate resource
        block invertibility is required.

    Raises
    ------
    ValueError
        If finite real inputs have incompatible shapes, S is outside [1,M],
        leakage is outside [0,1), B is negative or not column-stochastic
        to 1e-12, the coupled response is singular, or a result is nonfinite.
    """
    return result
```

### Step 4

04_selection_mixed_jet.py

Goal
----
Calculate the invasion and feedback coefficients with their mixed environmental derivatives.

```python
def selection_mixed_jet(focal: 'np.ndarray', B: 'np.ndarray', leakage: float, resources: 'np.ndarray', response: 'np.ndarray', N0: float, mp: float, mm: float) -> 'np.ndarray':
    """Calculate the invasion and feedback coefficients with their mixed environmental derivatives.

    Parameters
    ----------
    focal : np.ndarray
        Finite (4,2,M) uptake tensors, parent first then mutant.
    B : np.ndarray
        Nonnegative column-stochastic conversion (M,M), tolerance 1e-12.
    leakage : float
        Common leakage in [0,1).
    resources : np.ndarray
        Resource jet (4,M), rows value, epsilon, zeta, epsilon-zeta.
    response : np.ndarray
        Resident-adjusted resource susceptibility jet (4,M,M).
    N0 : float
        Positive fixed combined focal abundance.
    mp : float
        Nonnegative fixed parent mortality.
    mm : float
        Nonnegative fixed mutant mortality.

    Returns
    -------
    result : np.ndarray
        Shape (4,2), columns invasion fitness s and ecological feedback eta,
        rows value, epsilon, zeta, epsilon-zeta, without factorial scaling.
        These are the generalized consumer-resource cavity coefficients giving
        reduced selection s-eta*f. Use focal growth (1-leakage)*sum q and
        net depletion (I-leakage*B)@q, retain the parent-induced invasion
        correction, and differentiate focal growth gradients as well as
        resource impacts and susceptibility. The coefficients may have either sign.

    Raises
    ------
    ValueError
        If inputs are nonreal, nonfinite or have incompatible or empty shapes,
        leakage is outside [0,1), B is negative or not column-stochastic
        to 1e-12, N0 <= 0, either mortality is negative, or results are nonfinite.
    """
    return result
```

### Step 5

05_log_scale_integrals.py

Goal
----
Evaluate logarithms of diffusion scale integrals and their mixed parameter derivatives.

```python
def log_scale_integrals(selection: 'np.ndarray', N0: float, D: float, intervals: 'np.ndarray', order: int = 64) -> 'np.ndarray':
    """Evaluate logarithms of diffusion scale integrals and their mixed parameter derivatives.

    Parameters
    ----------
    selection : np.ndarray
        Finite (4,2) jet of (s,eta), rows value, epsilon, zeta, epsilon-zeta.
    N0 : float
        Positive fixed combined abundance.
    D : float
        Positive fixed noise coefficient, lineage variance rate 2*D*N.
    intervals : np.ndarray
        Finite (K,2), K >= 1, with 0 <= lower < upper <= 1 in each row.
    order : int
        Gauss-Legendre nodes per interval, integer in [16,256], not boolean.

    Returns
    -------
    result : np.ndarray
        Shape (4,K): value and epsilon, zeta, epsilon-zeta derivatives of
        the natural log of each scale integral for the absorbing diffusion
        df=f*(1-f)*(s-eta*f)dt+sqrt(2*D*f*(1-f)/N0)dW.
        Use scale density normalized to one at frequency zero and the stated
        quadrature order. All derivatives are ordinary, not factorial-scaled.
        Large finite exponents must be handled by a shift before exponentiation.

    Raises
    ------
    ValueError
        If selection or intervals are nonreal, nonfinite or have invalid shapes,
        interval bounds fail 0 <= lower < upper <= 1, N0 or D is nonpositive
        or nonfinite, order is not an integer in [16,256] or is boolean,
        or effective size, arithmetic or final results are nonfinite.
    """
    return result
```

### Step 6

06_hitting_moment_jet.py

Goal
----
Calculate fixation probability and the fixation-weighted first passage-time moment.

```python
def hitting_moment_jet(selection: 'np.ndarray', N0: float, D: float, f0: float, order: int = 64) -> 'np.ndarray':
    """Calculate fixation probability and the fixation-weighted first passage-time moment.

    Parameters
    ----------
    selection : np.ndarray
        Finite (4,2) jet of (s,eta), rows value, epsilon, zeta, epsilon-zeta.
    N0 : float
        Positive fixed combined focal abundance.
    D : float
        Positive fixed demographic-noise coefficient, lineage variance 2*D*N.
    f0 : float
        Fixed initial frequency strictly between zero and one.
    order : int
        Gauss-Legendre order in [16,256], integer and not boolean.

    Returns
    -------
    result : np.ndarray
        Shape (4,2), columns p=P(f(T)=1) and u=E[T*1_{f(T)=1}],
        with T the first hitting time of 0 or 1 for the same diffusion as
        log_scale_integrals. Rows are value, epsilon, zeta, epsilon-zeta.
        Evaluate the killed-generator Green integral using order-point
        Gauss-Legendre quadrature on each outer interval (0,f0),(f0,1),
        and order-point quadrature for every inner scale integral.
        Integrate complementary scale intervals directly rather than
        subtracting nearly equal cumulative integrals. Differentiate the
        quadrature-defined moments at fixed f0,N0,D; no factorial scaling.
        Use the earlier scale-integral step. The moment u is not conditional
        time itself and is not the unconditional absorption-time moment.

    Raises
    ------
    ValueError
        If selection has invalid shape or nonreal/nonfinite entries, N0 or D
        is nonpositive or nonfinite, f0 is not finite and strictly interior,
        order is not an integer in [16,256] or is boolean, or intermediate
        effective size, arithmetic or output is nonfinite.
    """
    return result
```

### Step 7

07_conditional_time_statistics.py

Goal
----
Recover conditional fixation-time derivatives and mixed logarithmic sensitivity.

```python
def conditional_time_statistics(moments: 'np.ndarray') -> 'np.ndarray':
    """Recover conditional fixation-time derivatives and mixed logarithmic sensitivity.

    Parameters
    ----------
    moments : np.ndarray
        Finite (4,2) jet, columns fixation probability p and fixation-weighted
        time moment u. Rows are value, epsilon, zeta, epsilon-zeta.
        Baseline 0 < p <= 1 and u > 0; derivatives can have either sign.

    Returns
    -------
    result : np.ndarray
        Shape (5,), ordered conditional mean time, its epsilon derivative,
        its zeta derivative, its mixed derivative, and the mixed derivative
        of its natural logarithm. Conditioning is on hitting frequency one
        before zero. Derivatives are not factorial-scaled.

    Raises
    ------
    ValueError
        If moments is not finite real shape (4,2), baseline probability is
        outside (0,1], baseline weighted moment is nonpositive, or a computed
        result is nonfinite.
    """
    return result
```

### Step 8

08_mixed_fixation_time.py

Goal
----
Orchestrator: calculate the mixed supply sensitivity of conditional mean fixation time.

```python
def mixed_fixation_time(config: dict, order: int = 64) -> float:
    """Orchestrator: calculate the mixed supply sensitivity of conditional mean fixation time.

    Parameters
    ----------
    config : dict
        Required keys: U, half, B, leakage, N, R, v, w, N0, mp, mm, D, f0.
        U and half have shape (S+2,M), with S residents followed by parent
        and mutant; 1 <= S <= M. U is nonnegative, half positive; N is
        positive (S,), R positive (M,); v,w are signed (M,) supply directions.
        B is nonnegative (M,M), columns sum to one to tolerance 1e-12;
        leakage lies in [0,1). N0,D are positive, mp,mm nonnegative, 0<f0<1.
        All data are finite and real. Resident mortality and baseline supply
        are implied by the cavity state; only supply changes by epsilon*v
        plus zeta*w. Fixed resident set, combined focal abundance and noise.
        Use the specified Monod model and the generalized cavity reduction,
        including parent correction. Extra keys are ignored. The stationary
        Jacobian must be nonsingular.
    order : int
        Inner and outer Gauss-Legendre order in [16,256], not boolean.

    Returns
    -------
    result : float
        Native Python float: mixed epsilon-zeta derivative of the natural
        log conditional mean time to fixation. Compose all earlier steps,
        including the scale-integral step reached through hitting_moment_jet.
        Use the stated quadrature convention. A zero supply direction or
        identical focal uptake, saturation and mortality gives zero sensitivity.

    Raises
    ------
    ValueError
        If config is not a dictionary or a required key is absent, any stated
        shape, real/finiteness, sign, leakage, conversion, frequency or order
        condition fails, the stationary Jacobian is singular, or intermediate
        arithmetic/results are nonfinite or a required probability or moment
        is nonpositive.
    """
    return result
```
