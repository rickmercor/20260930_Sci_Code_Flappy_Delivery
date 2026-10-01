# Physics-Quantum_Information_Computing-44

## Background

Alice chooses each symbol uniformly. BPSK uses amplitudes `(-1)**x * amplitude * exp(i*phase)` and Bob's homodyne outcome is `y=0` for `q>0`, otherwise `y=1`, with vacuum quadrature variance `1/2`. QPSK uses `amplitude * exp(i*(pi/4+phase+x*pi/2))`; Bob's heterodyne outcome is the counterclockwise quadrant `y=0,1,2,3`, starting with positive real and imaginary parts. For received coherent amplitude `beta`, homodyne `q` has mean `sqrt(2)*Re(beta)` and variance `1/2`, and the real and imaginary parts of the heterodyne amplitude are independent Gaussians with means `Re(beta), Im(beta)` and variance `1/2`. A channel of transmittance `eta` gives Bob amplitude `sqrt(eta)` times the launch amplitude and Eve amplitude `sqrt(1-eta)` times it. Eve retains the lost optical mode in an i.i.d. passive attack. Quadrant labels refer to the symbol alphabet, so the reconciliation leakage is the conditional Shannon entropy `H(Y|X)` in bits per signal.

For the resulting uniform classical-quantum state `rho_YE`, let `h_a` be the supremum of `-Dtilde_a(rho_YE || I_Y tensor sigma)` over unit-trace positive `sigma` invariant under the cyclic symbol rotation, let `d_a` use `sigma=rho_E`, and let `h_inf` be the infinite-order limit of `h_a`. Here `Dtilde_a(rho||sigma) = log2 Tr[(sigma**((1-a)/(2*a)) rho sigma**((1-a)/(2*a)))**a] / (a-1)`, and the infinite-order divergence is `log2 ||sigma**(-1/2) rho sigma**(-1/2)||_infinity`. Matrix functions act spectrally on their support. Coherent states obey `<z|w> = exp(-(abs(z)**2+abs(w)**2)/2 + conj(z)*w)`. The cyclic basis convention is `<psi_s|gamma_k> = sqrt(w_s)*exp(2*pi*i*s*k/N)`, with `s,k=0,...,N-1`, and the coefficient for `k=0` is positive; zero-weight basis vectors contribute zero. All logarithms used for entropies are base two.

For each scenario define `R_a = h_a + (1+2*log2(epsilon_prime))/n - g(epsilon)/(n*(a-1)) - H(Y|X)`, where `g(epsilon)=-log2(1-sqrt(1-epsilon**2))`. The term containing `1/(a-1)` vanishes at infinite order. The marginal-state comparison replaces `h_a` by `d_a` at the selected `t` and active scenario. The infinite-order comparison is the minimum scenario `R_inf` for the selected modulation setting. All auxiliary-state optimizations are scenario-specific.

The two other comparators use the same selected modulation setting and the same scenario family. With `H=H(Y|E)`, conditional entropy variance `V=Tr[rho_YE*(log2(rho_YE)-log2(I_Y tensor rho_E))**2]-D(rho_YE||I_Y tensor rho_E)**2`, and Petz conditional entropy `H_a_down=-log2 Tr[rho_YE**a*(I_Y tensor rho_E)**(1-a)]/(a-1)`, define `B_a=H-(a-1)*ln(2)*V/2-(a-1)**2*K(a)`, where `K(a)=2**((a-1)*(-H_a_down+H))*ln(2**(-H_2_down+H)+exp(2))**3/(6*(2-a)**3*ln(2))`. The robust continuity rate is the supremum over one common `1<a<2` of the minimum scenario rate with `B_a` replacing `h_a`. The robust AEP rate is the minimum scenario `[H+(1+2*log2(epsilon_prime))/n-4*log2(2+sqrt(N))*sqrt(log2(2/epsilon**2))/sqrt(n)-H(Y|X)]`. These are the paper's model-specific estimates with asymptotic Shannon leakage. Numerical work uses full-precision intermediates; support weights below `1e-15` may be discarded and the retained state normalized. Reported scalar-rate error up to `5e-7` bits per signal is accepted.

## Problem

A reverse-reconciled optical QKD link must use one modulation setting and one Rényi proof order across its entire channel uncertainty set. For the constructed pure-loss instance below, determine the largest signed finite-block rate estimate obtained from the symmetry-invariant optimized sandwiched conditional Rényi entropy, subject to the launch-energy budget. The ordered menu `(N, amplitude)` is `[(2,0.9),(2,1.1),(4,0.9),(4,1.1),(4,1.3),(4,1.5),(4,1.7)]`, the ordered channel scenarios `(transmittance, phase offset in radians)` are `[(0.88,0.02),(0.895,0.72),(0.92,0.43)]`, the block contains `n=400` signals, `epsilon=epsilon_prime=1e-8`, and the budget is `1.75` mean photons per transmitted signal. The same reciprocal order `t=1/a` is chosen for all scenarios from `[0,0.98]`, with `t=0` denoting the infinite-order limit, while each scenario has its own invariant auxiliary quantum state. Maximize the minimum scenario rate over `t` and then over menu entries satisfying `amplitude**2 <= 1.75`; retain signed estimates throughout. Report the rate in bits per transmitted signal to six decimal places and give the eight-value certificate `(menu index, active scenario index, t, optimized entropy at that scenario, marginal-state rate there at the chosen t, robust continuity-expansion rate, robust AEP rate, robust infinite-order rate)`. Use zero-based indices, the lowest index for rates tied within `1e-9` at menu selection or within `1e-7` at the active scenario, and the smallest optimizing `t` if the maximum is attained on an interval. Briefly justify the source-dependent entropy construction and the physical meaning of these model-specific estimates; the phase-offset uncertainty set, common proof order, finite menu, and photon budget are constructed extensions of the source's pure-loss analysis. The certificate tolerances are `2e-5` for `t`, `5e-5` bits for the optimized entropy, and `5e-5` bits per signal for the four comparison rates; equivalent rounded values within these tolerances receive credit.

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

psk_channel

Goal
----
Obtain the symbol transition matrix for the stated optical receiver.

```python
def psk_channel(N: int, amplitude: float, eta: float, phase: float) -> np.ndarray:
    """Obtain the symbol transition matrix for the stated optical receiver.

    N : int
        Alphabet size, 2 (BPSK) or 4 (QPSK).
    amplitude : float
        Nonnegative coherent launch amplitude; its square is mean photons per signal.
    eta : float
        Channel power transmittance in [0,1].
    phase : float
        Finite constellation phase offset in radians.
    Returns
    -------
    ndarray, shape (N,N), float
        Dimensionless P[y,x]=Pr(Y=y|X=x). BPSK: y=0 means q>0.
        QPSK: counterclockwise quadrants starting with Re>0, Im>0;
        launch angles are pi/4+phase+2*pi*x/N. Use the global receiver variances.
    Raises
    ------
    ValueError
        If N, amplitude, eta or phase is outside its domain.
    """
    return
```

### Step 2

cyclic_weights

Goal
----
Determine the exact coherent-state mixture spectrum in cyclic order.

```python
def cyclic_weights(N: int, mean_photons: float) -> np.ndarray:
    """Determine the exact coherent-state mixture spectrum in cyclic order.

    N : int
        Alphabet size, 2 or 4.
    mean_photons : float
        Mean photon number of Eve's coherent state, in [0,16].
    Returns
    -------
    ndarray, shape (N,), float
        The normalized nonnegative spectrum w_s of the uniform lost-state mixture,
        ordered by the cyclic index s=0,...,N-1 in the specified basis.
        At zero photons, w_0=1 and all remaining weights vanish.
    Raises
    ------
    ValueError
        If N or mean_photons is outside its domain.
    """
    return
```

### Step 3

reverse_states

Goal
----
Construct Eve's states conditional on the reverse-reconciled symbol.

```python
def reverse_states(channel: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Construct Eve's states conditional on the reverse-reconciled symbol.

    channel : ndarray, shape (N,N), float
        Nonnegative doubly stochastic P[y,x], N=2 or 4.
    weights : ndarray, shape (N,), float
        Nonnegative normalized cyclic weights in the stated basis convention.
    Returns
    -------
    ndarray, shape (N,N,N), complex
        result[y,s,t]=<psi_s|rho_(E|Y=y)|psi_t>; every outcome state has trace one.
        The first axis is the classical outcome y; the other two axes are cyclic
        basis row and column. All entries are dimensionless.
    Raises
    ------
    ValueError
        If dimensions, stochastic normalization or weights are invalid.
    """
    return
```

### Step 4

entropy_statistics

Goal
----
Evaluate entropy statistics for the cyclic classical-quantum ensemble.

```python
def entropy_statistics(state: np.ndarray) -> np.ndarray:
    """Evaluate entropy statistics for the cyclic classical-quantum ensemble.

    state : ndarray, shape (N,N), complex
        Normalized Hermitian positive semidefinite rho_(E|Y=0), N=2 or 4,
        in the cyclic basis; its unitary orbit has equal outcome probabilities.
    Returns
    -------
    ndarray, shape (3,), float
        [H(Y|E), V(Y|E), H_2_down(Y|E)], in [bits, bits**2, bits].
        The marginal is diag(diag(state)). Spectral support weights below 1e-15
        may be removed and the retained state normalized.
    Raises
    ------
    ValueError
        If state has invalid dimensions, entries, normalization or positivity.
    """
    return
```

### Step 5

sandwiched_optimum

Goal
----
Optimize the finite-order quantum reference state for a cyclic ensemble.

```python
def sandwiched_optimum(state: np.ndarray, reciprocal_order: float) -> np.ndarray:
    """Optimize the finite-order quantum reference state for a cyclic ensemble.

    state : ndarray, shape (N,N), complex
        Normalized Hermitian positive semidefinite conditional state in cyclic basis,
        N=2 or 4; its cyclic orbit defines the classical-quantum ensemble.
    reciprocal_order : float
        t=1/a in (0,1), dimensionless.
    Returns
    -------
    ndarray, shape (N+1,), float
        [h_a, sigma_0,...,sigma_(N-1)]: invariant optimized sandwiched conditional
        entropy in bits followed by the optimizing auxiliary state's cyclic
        diagonal probabilities. Retain the input's cyclic order. Set entries
        on marginal support weights below 1e-15 to zero. Entropy tolerance 5e-7
        bits; auxiliary probabilities 2e-6 absolute. Inputs with a unique active
        optimum are used. The input array remains unchanged.
    Raises
    ------
    ValueError
        If state is invalid or t is outside (0,1).
    """
    return
```

### Step 6

min_entropy_optimum

Goal
----
Evaluate the optimized infinite-order endpoint and its auxiliary state.

```python
def min_entropy_optimum(state: np.ndarray) -> np.ndarray:
    """Evaluate the optimized infinite-order endpoint and its auxiliary state.

    state : ndarray, shape (N,N), complex
        Normalized Hermitian positive semidefinite conditional state in cyclic basis,
        N=2 or 4, defining a uniform cyclic classical-quantum ensemble.
    Returns
    -------
    ndarray, shape (N+1,), float
        [h_inf, sigma_0,...,sigma_(N-1)], entropy in bits and unit-trace auxiliary
        diagonal probabilities at the infinite-order optimum, in cyclic order.
        Entries on support weights below 1e-15 are zero. Entropy tolerance 5e-7
        bits and auxiliary probabilities 2e-6 absolute. Tests use unique optima.
    Raises
    ------
    ValueError
        If state has invalid dimensions, entries, normalization or positivity.
    """
    return
```

### Step 7

robust_rate

Goal
----
Find the common-order worst-scenario finite-block rate and its comparators.

```python
def robust_rate(states: np.ndarray, channels: np.ndarray, statistics: np.ndarray, n: int, epsilon: float, epsilon_prime: float) -> np.ndarray:
    """Find the common-order worst-scenario finite-block rate and its comparators.

    states : ndarray, shape (S,N,N), complex
        The S conditional states rho_(E|Y=0), in scenario order, N=2 or 4.
    channels : ndarray, shape (S,N,N), float
        Corresponding optical transition matrices P[y,x].
    statistics : ndarray, shape (S,3), float
        Each row [H(Y|E), V(Y|E), H_2_down(Y|E)] from entropy_statistics.
    n : int
        Positive total block size in transmitted signals.
    epsilon, epsilon_prime : float
        Smoothing and extraction parameters, each in (0,1).
    Returns
    -------
    ndarray, shape (8,), float
        [R_star, t_star, active_scenario, h_star, R_down, R_B, R_AEP, R_inf].
        All rates are signed bits per signal; h_star is bits; t_star in [0,0.98].
        R_star maximizes the minimum scenario direct rate using one common t.
        At t=0 use min_entropy_optimum. Otherwise use sandwiched_optimum.
        Choose the first scenario within 1e-7 of the minimum; choose the least t
        on a maximizing interval. R_B optimizes one common 1<a<2 independently;
        R_AEP and R_inf are scenario minima at the same modulation setting.
        The global background defines all rate and comparator equations.
        Rate tolerance 5e-7, t tolerance 2e-5, comparison tolerances 5e-5.
    Raises
    ------
    ValueError
        If arrays have incompatible shapes, a state is invalid, or block/security
        parameters are outside their domains.
    """
    return
```

### Step 8

select_protocol

Goal
----
Select the energy-feasible protocol from its robust finite-block certificates.

```python
def select_protocol(menu: np.ndarray, certificates: np.ndarray, photon_budget: float) -> np.ndarray:
    """Select the energy-feasible protocol from its robust finite-block certificates.

    menu : ndarray, shape (C,2), float
        Rows [N,amplitude], N in {2,4}, amplitude >= 0, in candidate order.
    certificates : ndarray, shape (C,8), float
        Corresponding robust_rate outputs, in candidate order.
    photon_budget : float
        Nonnegative maximum amplitude**2 in mean photons per transmitted signal.
    Returns
    -------
    ndarray, shape (9,), float
        [R_star, menu_index, active_scenario, t_star, h_star, R_down, R_B, R_AEP,
        R_inf] for the feasible candidate with largest signed R_star.
        Index tie-break: first candidate within 1e-9 of the best feasible rate.
        Budget equality is feasible (1e-12 absolute arithmetic tolerance).
        Units are rates in bits/signal, h_star in bits, t dimensionless.
    Raises
    ------
    ValueError
        If arrays or budget are invalid, or no candidate is energy feasible.
    """
    return
```

### Step 9

qkd_benchmark

Goal
----
Compose the preceding scientific functions into the QKD design benchmark.

```python
def qkd_benchmark(menu: np.ndarray, scenarios: np.ndarray, n: int, epsilon: float, epsilon_prime: float, photon_budget: float) -> np.ndarray:
    """Compose the preceding scientific functions into the QKD design benchmark.

    menu : ndarray, shape (C,2), float
        Ordered rows [N,amplitude], N=2 or 4 and amplitude >= 0.
    scenarios : ndarray, shape (S,2), float
        Ordered rows [eta,phase], eta in [0,1] and finite phase in radians.
        Require (1-eta)*amplitude**2 <= 16 for every evaluated pair.
    n : int
        Positive block size in transmitted signals.
    epsilon, epsilon_prime : float
        Smoothing and extraction parameters in (0,1).
    photon_budget : float
        Nonnegative maximum mean launch photon number per signal.
    Returns
    -------
    ndarray, shape (9,), float
        [R_star, menu_index, active_scenario, t_star, h_star, R_down, R_B, R_AEP,
        R_inf]. These are exactly the selected certificate from select_protocol.
        Rates are signed bits/signal, h_star is bits, t dimensionless, and the
        two indices are zero-based integers stored as floats. Compose all eight
        preceding public functions, with transitive calls to the two quantum
        optimizers through robust_rate. Preserve all caller arrays.
    Raises
    ------
    ValueError
        If any input is outside the preceding functions' domains, arrays are empty
        or malformed, or no menu entry is energy feasible.
    """
    return
```
