# Physics-Particle_Physics-28

## Background

The rate of a superallowed positron decay is, at tree level, proportional to $p_eE_e(E_0-E_e)^2$ integrated over the positron energy. Two classes of electromagnetic corrections modify it: the Coulomb interaction of the positron with the daughter nucleus, resummed to all orders in $\alpha Z$ by the Fermi function and repulsive for positrons (so that the spectrum is exponentially suppressed at threshold), and the exchange and emission of soft photons, which produce the energy-dependent outer correction $\delta'_R(E_e)$ multiplying the Fermi-corrected spectrum. Because the half-life is an integral over the spectrum, what matters phenomenologically is the phase-space average of $\delta'_R$ with the Fermi-corrected weight. In an effective field theory the nucleus is a static charge, the photon modes are separated into potential (Coulomb) and ultrasoft, the couplings and functions are renormalized in $\overline{\rm MS}$, and the residual dependence on the renormalization scale, which cancels against the running of the vector coupling, is used to estimate missing higher orders. The corrections enhanced by $Z$ at second order change the rate at the per-mille level and therefore matter for $V_{ud}$ at its current precision.

## Problem

Superallowed $0^+\to0^+$ nuclear $\beta$ decays give the most precise value of the CKM element $V_{ud}$, at the $3\times10^{-4}$ level, because the Fermi matrix element is fixed by isospin and the first corrections are electromagnetic. The extraction rests on radiative corrections computed to first order in $\alpha$ decades ago (the Fermi function to all orders in $\alpha Z$ and Sirlin's energy-dependent "outer" correction), and its precision is now limited by the next order, in particular by the terms enhanced by the daughter charge, of order $\alpha^2Z$. A recent paper computes exactly these terms in a heavy-particle effective field theory in which ultrasoft photons couple to the nucleus as a whole: it evaluates the two-loop virtual and one-loop real-virtual diagrams with one Coulomb exchange and one ultrasoft photon, factors the Fermi function out of the rate, and obtains the $\mathcal O(\alpha^2Z)$ outer correction as a closed-form function $\hat g^{(2)}(\beta,\bar\mu)$ of the positron velocity $\beta$ and the $\overline{\rm MS}$ renormalization scale $\bar\mu$, together with its threshold and relativistic limits. In the same framework the one-loop outer correction is the Sirlin function written with an explicit $\overline{\rm MS}$ logarithm $L_\mu=\ln(\bar\mu^2/m_e^2)$ and a scheme-dependent constant, and the Fermi function is the $\overline{\rm MS}$-renormalized all-orders expression $\bar F(\beta,\bar\mu)$ of the effective theory. Your task is to evaluate the quantity that enters the corrected half-life of one transition: the phase-space-averaged outer correction $\bar\delta'_R(\bar\mu)=\alpha\,\bar g^{(1)}+\alpha^2Z\,\bar g^{(2)}$, where the bar denotes the average over the positron spectrum with the weight $p_eE_e\bar E^2\bar F(\beta,\bar\mu)$, $\bar E=E_0-E_e$ being the neutrino energy, using the paper's one-loop function, its two-loop function and its $\overline{\rm MS}$ Fermi function. The paper's two-loop function, the constant and logarithm of its one-loop function and the normalization of its Fermi function are not restated here — see the paper for their exact form. Use the following configuration:

- Transition $^{26}{\rm Al}\to{}^{26}{\rm Mg}$ (positron emission): daughter charge $Z=12$, electron-capture $Q$ value $Q_{EC}=4.2327$ MeV, endpoint $E_0=Q_{EC}-m_e$
- Positron mass $m_e=0.51099895$ MeV and fine-structure constant $\alpha=1/137.035999084$, held fixed (no running; the scale dependence is carried by the explicit logarithms of the correction functions)
- Renormalization scale: the paper's central low-energy choice, $\mu_{\rm ext}=2E_0$ in the $\overline{\rm MS}_\chi$ scheme, i.e. $\bar\mu=2E_0\,e^{-1}$ in $\overline{\rm MS}$, used in the Fermi function and in both correction functions
- Point-like nucleus in the phase-space factor (no finite-size correction to the spectrum), nuclear-structure corrections $\delta_C$, $\delta_{NS}$ and the matching coefficient of the vector coupling excluded
- Spectral averages evaluated by a quadrature converged to better than $10^{-9}$ relative

Let $\bar\delta'_R$ denote the averaged outer correction so obtained. Compute $\bar\delta'_R$ (dimensionless).

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, -0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> concise but complete: show the endpoint and scale, how the spectral average is set up and the value of its normalization integral, the check of your two-loop function against the paper's threshold and relativistic limits, the averaged one-loop and two-loop functions, the evidence that the quadrature is converged, and the final combination, since these are exactly what determines the final number.
Do not re-paste the configuration; reference it only as needed for the computation above.

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

step_01_lepton_kinematics

Goal
----
Convert a positron total energy in a superallowed beta decay into its velocity, its momentum and the energy left to the neutrino, and validate that the energy lies inside the allowed spectrum.

```python
def lepton_kinematics(E_e: float, m_e: float, E_0: float) -> tuple:
    '''Velocity, momentum and neutrino energy for a positron of energy E_e.

    Parameters
    ----------
    E_e : float
        Positron total energy in MeV, with m_e < E_e < E_0.
    m_e : float
        Positron mass in MeV, > 0.
    E_0 : float
        Endpoint energy E_0 = Q_EC - m_e in MeV, > m_e.

    Returns
    -------
    result : tuple of (float, float, float)
        (beta, p_e, Ebar): the velocity beta = p_e / E_e, the momentum
        p_e = sqrt(E_e^2 - m_e^2) in MeV and the neutrino energy Ebar = E_0 -
        E_e in MeV, all as native Python floats.

    Raises
    ------
    ValueError
        If m_e <= 0, E_0 <= m_e, or E_e is not strictly inside (m_e, E_0).
    '''
    return beta, p_e, Ebar  # placeholder
```

### Step 2

step_02_fermi_function_msbar

Goal
----
Evaluate the all-orders-in-alpha*Z Fermi function of a positron emitter in the MS-bar scheme of the heavy-particle effective theory, at a given lepton velocity and renormalization scale.

```python
def fermi_function_msbar(beta: float, Z: int, mubar: float, m_e: float, alpha: float) -> float:
    '''MS-bar Fermi function of a positron emitter.

    Parameters
    ----------
    beta : float
        Positron velocity, 0 < beta < 1.
    Z : int
        Charge of the daughter (final-state) nucleus, >= 1.
    mubar : float
        MS-bar renormalization scale in MeV, > 0.
    m_e : float
        Positron mass in MeV, > 0.
    alpha : float
        Fine-structure constant, with 0 < alpha * Z < 1.

    Returns
    -------
    Fbar : float
        The MS-bar Fermi function for positron emission, as a native Python
        float.

    Raises
    ------
    ValueError
        If beta is outside (0, 1), Z < 1, mubar <= 0, m_e <= 0, or
        alpha * Z is not in (0, 1).
    '''
    return Fbar  # placeholder
```

### Step 3

step_03_sirlin_g1

Goal
----
Evaluate the one-loop order-alpha outer radiative correction g^(1) to the positron spectrum of a superallowed decay (the Sirlin function in the heavy-particle effective theory, MS-bar scheme) at a given velocity, neutrino energy and renormalization scale.

```python
def sirlin_g1(beta: float, Ebar: float, mubar: float, m_e: float) -> float:
    '''Order-alpha outer correction g^(1)(beta, Ebar, mubar).

    Parameters
    ----------
    beta : float
        Lepton velocity, 0 < beta < 1.
    Ebar : float
        Neutrino energy E_0 - E_e in MeV, > 0.
    mubar : float
        MS-bar renormalization scale in MeV, > 0.
    m_e : float
        Lepton mass in MeV, > 0.

    Returns
    -------
    g1 : float
        The dimensionless function g^(1) (the correction factor is
        1 + alpha g^(1)), as a native Python float.

    Raises
    ------
    ValueError
        If beta is outside (0, 1), or Ebar, mubar or m_e is not strictly
        positive.
    '''
    return g1  # placeholder
```

### Step 4

step_04_g2_alpha2z

Goal
----
Evaluate the paper's two-loop order-alpha^2 Z outer radiative correction ghat^(2)(beta, mubar) to the positron spectrum of a superallowed decay, with the Fermi function factored out, at a given velocity and renormalization scale.

```python
def g2_alpha2z(beta: float, mubar: float, m_e: float) -> float:
    '''Order-alpha^2 Z outer correction ghat^(2)(beta, mubar) for positron emission.

    Parameters
    ----------
    beta : float
        Positron velocity, 0 < beta < 1.
    mubar : float
        MS-bar renormalization scale in MeV, > 0.
    m_e : float
        Positron mass in MeV, > 0.

    Returns
    -------
    g2 : float
        The dimensionless function ghat^(2) for a positron emitter (the
        correction factor is 1 + alpha^2 Z ghat^(2)), as a native Python
        float.

    Raises
    ------
    ValueError
        If beta is outside (0, 1), or mubar or m_e is not strictly positive.
    '''
    return g2  # placeholder
```

### Step 5

step_05_spectrum_weight

Goal
----
Evaluate the Coulomb-corrected positron momentum spectrum of a superallowed decay, i.e. the phase-space weight p_e^2 Ebar^2 Fbar(beta, mubar) used to average the outer radiative corrections over the spectrum.

```python
def spectrum_weight(p_e: float, Z: int, E_0: float, mubar: float, m_e: float, alpha: float) -> float:
    '''Phase-space weight per unit momentum, p_e^2 Ebar^2 Fbar(beta, mubar).

    Parameters
    ----------
    p_e : float
        Positron momentum in MeV, with 0 < p_e < sqrt(E_0^2 - m_e^2).
    Z : int
        Charge of the daughter nucleus, >= 1.
    E_0 : float
        Endpoint energy in MeV, > m_e.
    mubar : float
        MS-bar renormalization scale in MeV, > 0.
    m_e : float
        Positron mass in MeV, > 0.
    alpha : float
        Fine-structure constant, with 0 < alpha * Z < 1.

    Returns
    -------
    w : float
        p_e^2 (E_0 - E_e)^2 Fbar(beta, mubar) in MeV^4 (uses
        lepton_kinematics and fermi_function_msbar), as a native Python
        float.

    Raises
    ------
    ValueError
        If p_e is not strictly inside (0, sqrt(E_0^2 - m_e^2)), or propagated
        from lepton_kinematics or fermi_function_msbar for invalid inputs.
    '''
    return w  # placeholder
```

### Step 6

step_06_phase_space_average

Goal
----
Compute the phase-space normalization and the spectrum-averaged one-loop and two-loop outer correction functions of a superallowed positron emitter by Gauss-Legendre quadrature over the positron momentum.

```python
def phase_space_average(Z: int, E_0: float, mubar: float, m_e: float, alpha: float, n_nodes: int) -> "np.ndarray":
    '''Normalization and spectrum-averaged outer corrections by Gauss-Legendre quadrature.

    Parameters
    ----------
    Z : int
        Charge of the daughter nucleus, >= 1.
    E_0 : float
        Endpoint energy in MeV, > m_e.
    mubar : float
        MS-bar renormalization scale in MeV, > 0.
    m_e : float
        Positron mass in MeV, > 0.
    alpha : float
        Fine-structure constant, with 0 < alpha * Z < 1.
    n_nodes : int
        Number of Gauss-Legendre nodes on [0, p_max], >= 1.

    Returns
    -------
    result : np.ndarray of shape (3,)
        [N, gbar1, gbar2]: the normalization integral of the weight in MeV^5,
        the phase-space average of g^(1)(beta, Ebar, mubar) and the
        phase-space average of ghat^(2)(beta, mubar).

    Raises
    ------
    ValueError
        If n_nodes < 1 or E_0 <= m_e, or propagated from the earlier steps for
        invalid Z, mubar, m_e or alpha.
    '''
    return result  # placeholder
```

### Step 7

step_07_averaged_outer_correction

Goal
----
[ORCHESTRATOR] Compute the phase-space-averaged outer radiative correction deltabar'_R of a superallowed positron emitter through order alpha^2 Z, at the paper's central renormalization scale, from the transition's electron-capture Q value and daughter charge.

```python
def averaged_outer_correction(Q_EC: float, Z: int, m_e: float, alpha: float, n_nodes: int) -> float:
    '''Phase-space-averaged outer correction deltabar'_R at mubar = 2 E_0 / e.

    Parameters
    ----------
    Q_EC : float
        Electron-capture Q value of the transition in MeV, > 2 m_e.
    Z : int
        Charge of the daughter nucleus, >= 1.
    m_e : float
        Positron mass in MeV, > 0.
    alpha : float
        Fine-structure constant, with 0 < alpha * Z < 1.
    n_nodes : int
        Number of Gauss-Legendre nodes for the spectral averages, >= 1.

    Returns
    -------
    delta_R : float
        deltabar'_R = alpha gbar^(1) + alpha^2 Z gbar^(2), dimensionless, as a
        native Python float.

    Raises
    ------
    ValueError
        If Q_EC <= 2 m_e, or propagated from the earlier steps for invalid Z,
        m_e, alpha or n_nodes.
    '''
    return delta_R  # placeholder
```
