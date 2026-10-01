# Chemistry-Computational_Chemistry-56

## Background

Photosynthetic antenna complexes move electronic excitation between closely spaced pigments with high efficiency. Two-dimensional electronic spectroscopy and pump-probe experiments on these complexes, and on synthetic pigment dimers, show oscillations that persist for hundreds of femtoseconds, which started a long discussion about whether coherent superpositions of pigment excitations take part in energy transfer or are merely a by-product of the excitation. Part of the difficulty is that coherence is a property of a state, while efficiency, yield and transfer time are properties of a process and of what is measured. A statement about the functional role of coherence therefore has to compare what a given dynamics does with and without coherence in its input, for a given readout.

The excitonic dynamics of a pigment pair depends on the electronic coupling between the pigments, on the energy difference between their excitations, and on how strongly each pigment couples to the vibrations of its surroundings. When the protein vibrations relax on a time scale similar to that of the electronic motion, perturbative and Markovian master equations become unreliable, and numerically exact reduced-dynamics methods for harmonic environments are used instead. Irreversible processes such as fluorescence, internal conversion and charge separation at a reaction center can be added as Markovian channels that remove the excitation from the pigment pair.

Real samples are inhomogeneous: slightly different local environments shift the excitation energies of the pigments from one complex to the next, and on the ultrafast time scale these shifts are static. Spectroscopic signals are averages over this distribution, so oscillations of individual complexes dephase in the ensemble even when each complex keeps its own phase. In addition, signals recorded while the pump and probe pulses still overlap in time are distorted by coherent artifacts, so dynamics are usually interpreted only after the overlap has ended. Whether coherence can affect an ensemble measurement in the usable delay range therefore depends jointly on the coupling to the environment, on the static disorder and on the choice of observable.

## Problem

Long-lived oscillations in ultrafast spectra of photosynthetic pigment-protein complexes have kept alive the question of whether electronic coherence created by the excitation pulse matters for energy transfer. A recent operational framework answers it for a fixed dynamics and a fixed measured quantity instead of relying on state-based coherence measures: every initial preparation is compared with the same preparation stripped of its site-basis coherences, and the largest change this can cause in the measured signal is a state-independent bound on what coherence can do. The framework has been applied to donor-acceptor dimers and homogeneous chains, and its authors name ensemble-averaged bounds for systems with static disorder as an important next step. This task takes that step for an inhomogeneous ensemble of dimers probed after the pump-probe pulse overlap. The input is the dimer, bath, loss and disorder model, and the output is the bath reorganization energy at which the bound on the coherence effect in the acceptor signal falls to 0.10.

Every dimer has a donor site |D> and an acceptor site |A> with electronic Hamiltonian eps_D|D><D| + eps_A|A><A| + J(|D><A| + |A><D|), J = +85 cm^-1, together with a common ground state |g> and a sink |s>. The excitation recombines from either site to |g> at 0.1 ps^-1 (jump operators sqrt(0.1 ps^-1)|g><D| and sqrt(0.1 ps^-1)|g><A|) and is trapped from the acceptor into |s> at 1 ps^-1 (jump operator sqrt(1 ps^-1)|s><A|). Each site couples through its projector |j><j| to its own harmonic bath with the Drude-Lorentz spectral density S(omega) = 2 E_R gamma_c omega / (omega^2 + gamma_c^2), with the same reorganization energy E_R on both sites, gamma_c = 60 cm^-1 and temperature 295 K, and the baths start in thermal equilibrium, uncorrelated with the system. Treat the reduced dynamics with the hierarchical equations of motion: represent each bath correlation function by its Drude term and its first Matsubara term with no correction for the omitted terms, keep every auxiliary density operator whose indices sum to at most 6, and let the Lindblad terms act on every auxiliary density operator. Convert wavenumbers to angular frequencies with omega = 2 pi c nu, c = 2.99792458 x 10^10 cm s^-1, and use k_B = 0.6950348 cm^-1 K^-1.

In the ensemble, eps_D and eps_A are independent normal random variables, each with standard deviation 70 cm^-1, and the mean of eps_D - eps_A is 140 cm^-1; J, the baths and the rates are the same for every molecule. Every molecule is prepared in the same state rho_0 of the singly excited pair, and the signal is the ensemble-averaged acceptor population P_A(t). At each delay t, let C(t) be the largest absolute change of P_A(t) that any rho_0 can produce relative to its site-dephased counterpart (rho_0 with its off-diagonal elements in the {|D>, |A>} basis set to zero), and let Q(E_R) be the maximum of C(t) over the delays 150 fs <= t <= 600 fs. Q decreases with E_R between 20 and 35 cm^-1; find the reorganization energy E_R* in this range at which Q = 0.10.

Give E_R* in cm^-1 to two decimal places as the final answer. In the reasoning, report the standard deviation of the energy gap eps_D - eps_A, the decay rates of the Drude and first Matsubara terms of each bath correlation function, and the number of auxiliary density operators kept per molecule. Then report at E_R* the delay t* (fs) at which C(t) = Q, the relative phase phi in (-pi, pi] for which the preparation (|D> + e^(i phi)|A>)/sqrt(2) gives the largest increase of P_A(t*) over its site-dephased counterpart, P_A(t*) for ensembles excited on the donor alone and on the acceptor alone, and the maximum of C(t) at delays below 150 fs with the delay at which it occurs. Report Q at E_R = 20 cm^-1 and Q at E_R* for a sample without static disorder. Finally, with the definitions of the earlier paper that introduced this coherence-impact functional, evaluate at E_R* the unpaired benchmark of the same readout at t*, its instantaneous rate at the early maximum, and that paper's variation bound on the change of C(t) between the early maximum and t*.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> focused: report every quantity requested above with its value and the relation used to obtain it, without a long derivation.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_drude_lorentz_exponents

Goal
----
Expand the correlation function of an overdamped (Drude-Lorentz) harmonic bath into decaying exponentials.

```python
def drude_lorentz_exponents(reorg_cm: float, cutoff_cm: float, temperature_K: float,
                            n_matsubara: int) -> "np.ndarray":
    '''Exponential expansion of the Drude-Lorentz bath correlation function, truncated after n_matsubara terms.

    Parameters
    ----------
    reorg_cm : float
        Reorganization energy E_R in cm^-1, non-negative. The spectral density is
        S(omega) = 2 E_R gamma_c omega / (omega^2 + gamma_c^2), with E_R and gamma_c converted to rad/ps.
    cutoff_cm : float
        Cutoff gamma_c in cm^-1, positive.
    temperature_K : float
        Temperature in kelvin, positive.
    n_matsubara : int
        Number of Matsubara terms kept after the Drude term, n_matsubara >= 0.

    Returns
    -------
    exponents : np.ndarray
        Shape (n_matsubara + 1, 3). Row k holds [Re c_k, Im c_k, nu_k] such that the bath correlation function
        C(t) = (1/pi) int_0^inf S(omega) [coth(beta omega / 2) cos(omega t) - i sin(omega t)] d omega, t >= 0,
        equals sum_k c_k exp(-nu_k t) when all Matsubara terms are kept. Row 0 is the Drude term (nu_0 = gamma_c) and
        row k >= 1 is the k-th Matsubara term. Amplitudes c_k are in ps^-2 and rates nu_k in ps^-1, using
        omega[rad/ps] = 2 pi c nu[cm^-1] with c = 2.99792458e-2 cm/ps and beta = 1 / (k_B T) with
        k_B = 0.6950348 cm^-1/K, so that hbar = 1.

    Raises
    ------
    ValueError
        If cutoff_cm or temperature_K is not positive, reorg_cm is negative, n_matsubara is negative, or a kept
        Matsubara frequency equals gamma_c to within a relative 1e-9.
    '''
    return exponents
```

### Step 2

02_heom_readout_operator

Goal
----
Propagate a site-population readout of a donor-acceptor dimer backwards through its exact reduced dynamics, giving the time-dependent operator whose expectation value in any initial state is the population at a later time.

```python
def heom_readout_operator(dimer: "np.ndarray", bath: "np.ndarray", n_matsubara: int, depth: int,
                          readout_site: int, times_ps: "np.ndarray") -> "np.ndarray":
    '''Heisenberg-picture site-population readout of a dimer with Drude-Lorentz site baths, from the hierarchy.

    Parameters
    ----------
    dimer : np.ndarray
        Shape (4,), [J, gap, gamma_rec, kappa]. The electronic Hamiltonian of the singly excited pair in the basis
        (|D>, |A>) is (gap/2)(|D><D| - |A><A|) + J(|D><A| + |A><D|), with J and gap = eps_D - eps_A in cm^-1.
        Excitations recombine from each site to a ground state |g> with rate gamma_rec (jump operators
        sqrt(gamma_rec)|g><D| and sqrt(gamma_rec)|g><A|) and are trapped from the acceptor into a sink |s> with
        rate kappa (jump operator sqrt(kappa)|s><A|); both rates in ps^-1 and non-negative.
    bath : np.ndarray
        Shape (3,), [E_R, gamma_c, T]: reorganization energy (cm^-1), cutoff (cm^-1) and temperature (K) of the two
        identical, independent Drude-Lorentz baths. Site j couples to its bath through the projector |j><j|, with the
        spectral density and exponential amplitudes and rates of drude_lorentz_exponents.
    n_matsubara : int
        Number of Matsubara terms kept after the Drude term in each bath correlation function, >= 0. The omitted
        terms are dropped without any correction.
    depth : int
        Hierarchy truncation, >= 0: auxiliary density operators whose non-negative integer indices (one per site and
        exponent) sum to more than depth are set to zero. depth = 0 keeps the system density operator only, so the
        baths have no effect. The Lindblad terms act on every auxiliary density operator.
    readout_site : int
        0 for the donor population |D><D|, 1 for the acceptor population |A><A|.
    times_ps : np.ndarray
        Shape (n_t,), non-negative delays in ps. The baths are in thermal equilibrium and uncorrelated with the
        system at t = 0.

    Returns
    -------
    readout : np.ndarray
        Shape (n_t, 8), one row per delay in the input order: [M_DD, M_AA, Re M_DA, Im M_DA, dM_DD/dt, dM_AA/dt,
        Re dM_DA/dt, Im dM_DA/dt]. M(t) is the Hermitian operator on the singly excited pair with
        Tr[M(t) rho_0] = <r| rho(t) |r> for every initial state rho_0 of the pair, where rho(t) is the system density
        operator of the truncated hierarchy and r the readout site, M_DA = <D|M(t)|A>, and the derivatives are the
        exact time derivatives of the truncated hierarchy solution (ps^-1). Energies convert to rad/ps with
        omega = 2 pi c nu, c = 2.99792458e-2 cm/ps (hbar = 1). Values are accurate to a relative 1e-9.

    Raises
    ------
    ValueError
        If depth or n_matsubara is negative, readout_site is not 0 or 1, a rate is negative, or a delay is negative.
    '''
    return readout
```

### Step 3

03_ensemble_readout_operator

Goal
----
Average the Heisenberg-picture site-population readout over an ensemble of donor-acceptor dimers with static site-energy disorder.

```python
def ensemble_readout_operator(dimer: "np.ndarray", site_sigma_cm: float, bath: "np.ndarray", n_matsubara: int,
                              depth: int, n_nodes: int, readout_site: int, times_ps: "np.ndarray") -> "np.ndarray":
    '''Disorder-averaged Heisenberg-picture site-population readout of an inhomogeneous dimer ensemble.

    Parameters
    ----------
    dimer : np.ndarray
        Shape (4,), [J, mean_gap, gamma_rec, kappa] with J and the mean gap mean(eps_D) - mean(eps_A) in cm^-1 and
        the loss rates in ps^-1, as in heom_readout_operator.
    site_sigma_cm : float
        Standard deviation (cm^-1, >= 0) of each site energy. eps_D and eps_A are independent normal variables with
        this common standard deviation; J, the baths and the rates are the same for every molecule.
    bath : np.ndarray
        Shape (3,), [E_R, gamma_c, T] as in heom_readout_operator.
    n_matsubara : int
        Matsubara terms per bath, >= 0, as in heom_readout_operator.
    depth : int
        Hierarchy truncation, >= 0, as in heom_readout_operator.
    n_nodes : int
        Number of nodes, >= 1, of the Gauss-Hermite rule for the normal distribution (probabilists' Hermite nodes
        x_i with weights normalized to sum to 1) used for the average over the gap distribution; the molecule at node
        i has the gap mean_gap + s x_i, where s is the standard deviation of the gap.
    readout_site : int
        0 for the donor population, 1 for the acceptor population.
    times_ps : np.ndarray
        Shape (n_t,), non-negative delays in ps.

    Returns
    -------
    readout : np.ndarray
        Shape (n_t, 8), the weighted sum over the nodes of the rows returned by heom_readout_operator,
        [M_DD, M_AA, Re M_DA, Im M_DA, dM_DD/dt, dM_AA/dt, Re dM_DA/dt, Im dM_DA/dt], so that Tr[M(t) rho_0] is the
        ensemble-averaged population of the readout site at delay t when every molecule starts in rho_0.

    Raises
    ------
    ValueError
        If site_sigma_cm is negative or n_nodes is smaller than 1, and in the cases listed for heom_readout_operator.
    '''
    return readout
```

### Step 4

04_coherence_diagnostics

Goal
----
Turn a Heisenberg-picture population readout and its rate of change into four coherence diagnostics for the preparation of a donor-acceptor pair.

```python
def coherence_diagnostics(readout: "np.ndarray") -> "np.ndarray":
    '''Coherence impact, unpaired benchmark, optimal relative phase and instantaneous rate for readout operators.

    Parameters
    ----------
    readout : np.ndarray
        Shape (n, 8), rows [M_DD, M_AA, Re M_DA, Im M_DA, dM_DD/dt, dM_AA/dt, Re dM_DA/dt, Im dM_DA/dt] of a Hermitian
        2 x 2 operator M on the basis (|D>, |A>), with M_DA = <D|M|A>, and of its time derivative, as returned by
        heom_readout_operator or ensemble_readout_operator.

    Returns
    -------
    diagnostics : np.ndarray
        Shape (n, 4), rows [C, Pi, phi, Gamma]. With rho ranging over all density operators of the pair and
        G(rho) the same operator with its off-diagonal elements set to zero:
        C = sup_rho |Tr[M (rho - G(rho))]|;
        Pi = sup_rho |Tr[M rho]| - sup_sigma |Tr[M sigma]|, where sigma ranges over the diagonal density operators;
        phi in (-pi, pi] is the relative phase for which the pure state (|D> + exp(i phi)|A>)/sqrt(2) maximizes
        Tr[M (rho - G(rho))], with phi = 0 when M_DA = 0;
        Gamma = sup_rho |Tr[(dM/dt) (rho - G(rho))]|, in the units of the derivative columns.

    Raises
    ------
    ValueError
        If readout is not a two-dimensional array with 8 columns.
    '''
    return diagnostics
```

### Step 5

05_post_overlap_supremum

Goal
----
Find the largest coherence impact on the ensemble acceptor population inside a window of pump-probe delays, and the delay at which it occurs.

```python
def post_overlap_supremum(dimer: "np.ndarray", site_sigma_cm: float, bath: "np.ndarray", n_matsubara: int,
                          depth: int, n_nodes: int, window_ps: "np.ndarray") -> "np.ndarray":
    '''Maximum over a delay window of the coherence impact on the ensemble-averaged acceptor population.

    Parameters
    ----------
    dimer : np.ndarray
        Shape (4,), [J, mean_gap, gamma_rec, kappa], as in ensemble_readout_operator.
    site_sigma_cm : float
        Standard deviation of each site energy (cm^-1), as in ensemble_readout_operator.
    bath : np.ndarray
        Shape (3,), [E_R, gamma_c, T], as in heom_readout_operator.
    n_matsubara : int
        Matsubara terms per bath, as in heom_readout_operator.
    depth : int
        Hierarchy truncation, as in heom_readout_operator.
    n_nodes : int
        Gauss-Hermite nodes of the disorder average, as in ensemble_readout_operator.
    window_ps : np.ndarray
        Shape (2,), [t_a, t_b] with 0 <= t_a < t_b, the closed delay window in ps.

    Returns
    -------
    supremum : np.ndarray
        Shape (2,), [Q, t_star]. C(t) is the coherence impact C of coherence_diagnostics for the ensemble acceptor
        readout (readout_site = 1) at delay t, Q is the maximum of C(t) over t_a <= t <= t_b, and t_star (ps) is the
        delay at which it is reached, located to 1e-7 ps on the continuous time axis (the earliest such delay if the
        maximum is reached more than once).

    Raises
    ------
    ValueError
        If the window does not satisfy 0 <= t_a < t_b, and in the cases listed for ensemble_readout_operator.
    '''
    return supremum
```

### Step 6

06_critical_reorganization_energy

Goal
----
Find the bath reorganization energy at which the largest coherence impact on the ensemble acceptor population inside a delay window falls to a prescribed level.

```python
def critical_reorganization_energy(dimer: "np.ndarray", site_sigma_cm: float, cutoff_cm: float,
                                   temperature_K: float, n_matsubara: int, depth: int, n_nodes: int,
                                   window_ps: "np.ndarray", target: float, reorg_bracket: "np.ndarray") -> float:
    '''Reorganization energy at which the maximum coherence impact over the delay window equals a target.

    Parameters
    ----------
    dimer : np.ndarray
        Shape (4,), [J, mean_gap, gamma_rec, kappa], as in ensemble_readout_operator.
    site_sigma_cm : float
        Standard deviation of each site energy (cm^-1), as in ensemble_readout_operator.
    cutoff_cm : float
        Drude-Lorentz cutoff gamma_c (cm^-1) of both baths.
    temperature_K : float
        Bath temperature (K).
    n_matsubara : int
        Matsubara terms per bath, as in heom_readout_operator.
    depth : int
        Hierarchy truncation, as in heom_readout_operator.
    n_nodes : int
        Gauss-Hermite nodes of the disorder average, as in ensemble_readout_operator.
    window_ps : np.ndarray
        Shape (2,), the delay window [t_a, t_b] in ps, as in post_overlap_supremum.
    target : float
        Threshold value of the maximum coherence impact Q, 0 < target < 1.
    reorg_bracket : np.ndarray
        Shape (2,), [E_lo, E_hi] with 0 <= E_lo < E_hi, reorganization energies in cm^-1 at which Q - target has
        opposite signs.

    Returns
    -------
    reorg_star : float
        The reorganization energy E_R* (cm^-1) inside the bracket at which Q(E_R) from post_overlap_supremum, with the
        bath [E_R, cutoff_cm, temperature_K], equals target, located to 1e-6 cm^-1.

    Raises
    ------
    ValueError
        If the bracket is not ordered and non-negative, target is outside (0, 1), or Q - target has the same sign at
        both ends of the bracket.
    '''
    return reorg_star
```

### Step 7

07_coherence_relevance_report

Goal
----
Report where site coherence stops being able to change the measured acceptor population of an inhomogeneous donor-acceptor ensemble by a set amount after the pulse overlap, together with the diagnostics at that point (orchestrator).

```python
def coherence_relevance_report(dimer: "np.ndarray", site_sigma_cm: float, cutoff_cm: float, temperature_K: float,
                               n_matsubara: int, depth: int, n_nodes: int, window_ps: "np.ndarray",
                               target: float, reorg_bracket: "np.ndarray", reorg_reference: float) -> "np.ndarray":
    '''Critical reorganization energy of the ensemble coherence impact and the diagnostics at that point.

    Parameters
    ----------
    dimer : np.ndarray
        Shape (4,), [J, mean_gap, gamma_rec, kappa], as in ensemble_readout_operator.
    site_sigma_cm : float
        Standard deviation of each site energy (cm^-1), as in ensemble_readout_operator.
    cutoff_cm : float
        Drude-Lorentz cutoff gamma_c (cm^-1) of both baths.
    temperature_K : float
        Bath temperature (K).
    n_matsubara : int
        Matsubara terms per bath, as in heom_readout_operator.
    depth : int
        Hierarchy truncation, as in heom_readout_operator.
    n_nodes : int
        Gauss-Hermite nodes of the disorder average, as in ensemble_readout_operator.
    window_ps : np.ndarray
        Shape (2,), delay window [t_a, t_b] in ps with 0 < t_a < t_b, as in post_overlap_supremum.
    target : float
        Threshold of the maximum coherence impact, as in critical_reorganization_energy.
    reorg_bracket : np.ndarray
        Shape (2,), bracket of reorganization energies (cm^-1), as in critical_reorganization_energy.
    reorg_reference : float
        Reference reorganization energy (cm^-1, >= 0) at which the window maximum is also reported.

    Returns
    -------
    report : np.ndarray
        Shape (12,), [E_R*, t_star, phi, P_A_donor, P_A_acceptor, Q_reference, Q_homogeneous, C_early, t_early, Pi,
        Gamma, V]: E_R* (cm^-1) from critical_reorganization_energy; t_star (ps) the delay of the window maximum at
        E_R* from post_overlap_supremum; phi and Pi from coherence_diagnostics applied to the ensemble acceptor
        readout at E_R* and t_star; P_A_donor = M_DD and P_A_acceptor = M_AA of that readout, the ensemble-averaged
        acceptor populations at t_star when every molecule starts on the donor or on the acceptor; Q_reference the
        window maximum of the ensemble at reorg_reference; Q_homogeneous the window maximum at E_R* with
        site_sigma_cm = 0 and a single node; C_early and t_early (ps) the maximum of the same coherence impact at E_R*
        over the delays 0 <= t <= t_a before the window, and its delay, from post_overlap_supremum; Gamma (ps^-1) from
        coherence_diagnostics applied to the ensemble acceptor readout at E_R* and t_early; V the integral of the same
        rate Gamma(t) from t_early to t_star, evaluated with the composite Simpson rule on a uniform grid with an even
        number of intervals no wider than 2e-4 ps.

    Raises
    ------
    ValueError
        In the cases listed for critical_reorganization_energy and post_overlap_supremum, or if reorg_reference is
        negative or t_a is not positive.
    '''
    return report
```
