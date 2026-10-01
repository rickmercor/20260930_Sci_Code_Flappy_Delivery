# Physics-Particle_Physics-35

## Background

A form factor measures how an extended object responds to a probe, and for hadrons it is the object that carries all the strong-interaction physics a decay rate or a scattering cross section depends on. It is a function of one kinematic variable, conventionally the squared centre-of-mass energy, and its most useful property is that it is analytic: knowing it in one region constrains it everywhere else. That analyticity is not perfect, because whenever the energy is high enough for a new physical channel to open the function develops a branch cut starting at that threshold. A two-pion form factor has right-hand cuts starting at the pion and kaon pair-production thresholds. Analytic continuation onto unphysical sheets introduces a left-hand cut extending along the negative axis from the equal-mass pseudothreshold at the origin; this is not an additional cut on the physical sheet.

Branch cuts turn a function into a multivalued one, and the sheets that multivaluedness produces are where the interesting physics lives. The first sheet, reached by approaching the real axis from above, is the physical one and is where measurements happen. Resonances are not on it. A resonance is a pole, and a pole on the physical sheet would mean a genuinely stable state, so an unstable particle must sit on one of the other sheets, reached by continuing through a cut. For a pole at squared energy $s_p$, choose the resonance root with positive real part and negative imaginary part: its mass is $M = \operatorname{Re}\sqrt{s_p}$ and its width is $\Gamma = -2\operatorname{Im}\sqrt{s_p}$, both in energy units. This is why extracting resonance parameters from data is a question about analytic continuation rather than about curve fitting.

The standard way to build a parametrisation is to map the cut plane conformally onto the unit disc and expand in the resulting variable, since a polynomial in a bounded variable converges quickly and needs few parameters. Different maps trade convergence for reach: which singularities a map resolves decides which parts of the Riemann surface the expansion can describe, and a map is a construction rather than a consequence of analyticity. The choices inside it are decisions the author of the parametrisation makes and justifies, and two parametrisations built on different maps are different functions of the same coefficients.

One feature of any such construction matters for anyone implementing it. Square roots carry branch choices, and a formula written on paper is ambiguous until the branch of every root in it is fixed. The same algebra with different branch conventions produces different sheets, and the difference is not a sign error but a different physical statement, so an implementation is checked by reproducing a known point on a named sheet rather than by inspecting the formula.

## Problem

A hadronic form factor is analytic in the squared centre-of-mass energy apart from branch cuts, and it is expanded by mapping the cut plane onto a bounded variable in which the expansion is a polynomial. Use the published composed variable for this two-channel pion form factor with its left-hand cut. Retain that construction's orientation and reference-sheet choices, without an additional disc rotation or reparameterization.

Work with a two-channel pion system. The lower threshold sits at four times the squared pion mass with the pion mass taken to be 0.13957 GeV, the upper threshold at four times the squared kaon mass with the kaon mass taken to be 0.493677 GeV, the left-hand cut opens at the origin and the expansion is normalized so that the bounded variable vanishes at minus 0.60, everything in GeV squared once the masses are squared.

The form factor is a pole factor times a polynomial. The pole factor is the reciprocal of a product of monic linear factors, one for each resonance pole and one for its complex conjugate partner in the bounded variable, so it diverges at each and carries no further constant. The three poles are quoted as complex energies in GeV, each with the sheet it occupies, and a complex energy is squared before it is mapped: 0.452 minus 0.271i on the sheet reached across the elastic cut, 0.998 minus 0.029i on that same sheet and 0.981 minus 0.043i on the sheet directly adjacent to the physical sheet above the upper threshold. The polynomial coefficients, ascending from the constant term, are 0.462, -0.236, 0.559, -0.706, 0.228, -0.510, -0.147, 1.088, -0.526, 1.578, -0.268 and 0.533.

Evaluate at the squared-energy point -3.50 plus 0.20i on the sheet directly adjacent to the physical sheet above the upper threshold, and at -0.75 minus 0.16i on the remaining nonphysical sheet, which is distinct from both that sheet and the sheet adjacent across the elastic cut. Take the squared modulus at each and report their product, rounded to six decimal places. In the short reasoning, state the source's sheet prescription and report the intermediate images of the left-hand-cut opening and normalization point, the mapped third pole, the intermediate and composed variables at both evaluation points and the two squared form-factor moduli. Algebraically equivalent constructions with the same sheet-specific values are acceptable. Your final answer must be a single number: the value of that product.

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

01_channel_geometry

Goal
----
Turn the two channel masses and the two reference points into the four real numbers that fix the geometry of the expansion. The two thresholds are four times the squared mass of the lighter and the heavier channel respectively, and the cut opening and the normalization point are passed through unchanged. The returned tuple carries the lower threshold, the upper threshold, the cut opening and the normalization point in that order, all in squared energy units. Invalid input raises ValueError when either mass is not a finite positive scalar, when either reference point is not a finite real scalar, when the two masses do not place the lower threshold strictly below the upper one or when a squared threshold overflows.

```python
import numpy as np


def channel_geometry(light_mass, heavy_mass, cut_opening, normalization_point):
    """Return the two thresholds, the cut opening and the normalization point.

    Parameters
    ----------
    light_mass : float
        Finite positive mass of the lighter channel constituent, in energy units.
    heavy_mass : float
        Finite positive mass of the heavier channel constituent, in energy units.
    cut_opening : float
        Finite real squared energy at which the left-hand cut opens.
    normalization_point : float
        Finite real squared energy at which the expansion variable vanishes.

    Returns
    -------
    tuple of float
        The lower threshold, the upper threshold, the cut opening and the
        normalization point, in that order, in squared energy units.

    Raises
    ------
    ValueError
        If either mass is not a finite positive scalar, if either reference
        point is not a finite real scalar, or if four times the squared light
        mass is not strictly below four times the squared heavy mass, or if
        either squared threshold overflows.
    """
    return None
```

### Step 2

02_threshold_branch_map

Goal
----
Evaluate one selected branch of the intermediate conformal map that resolves two ordered right-hand thresholds. All square roots use their principal branches. An exactly real energy, including either sign of a zero imaginary part, denotes the limit from the upper half-plane; a negative real threshold-minus-energy radicand therefore has a square root with negative imaginary part. The input energy may be real or complex, the thresholds are finite real scalars with the lower one strictly below the upper one and the two-character sheet label selects the returned branch as a Python complex number. Invalid input raises ValueError when the energy is not a finite scalar, either threshold is not a finite real scalar, the thresholds are not strictly ordered, the sheet label is not one of the four supported strings, the selected outer branch diverges at the lower threshold or a nonzero branch result is outside the finite complex output range.

```python
import numpy as np


def threshold_branch_map(s, lower_threshold, upper_threshold, sheet):
    """Return the selected branch of the two-threshold conformal map.

    Parameters
    ----------
    s : float or complex
        Finite squared energy. Exactly real inputs use the upper-half-plane
        boundary value, independent of the sign of a zero imaginary part.
    lower_threshold : float
        Finite real position of the first right-hand threshold.
    upper_threshold : float
        Finite real position of the second right-hand threshold.
    sheet : str
        Branch label, chosen from "11", "21", "22" and "12".

    Returns
    -------
    complex
        Value of the selected conformal-map branch.

    Raises
    ------
    ValueError
        If the energy is not a finite real or complex scalar, if either
        threshold is not a finite real scalar, if the lower threshold is not
        strictly below the upper threshold, if the sheet label is invalid or if
        the energy equals the lower threshold on sheet "22" or "12", or a
        nonzero result overflows or underflows the finite complex output range.
        Avoid intermediate overflow when the final result is representable.
    """
    return None
```

### Step 3

03_leftcut_map

Goal
----
Evaluate the conformal map that sends the left-hand cut of an intermediate variable to the unit circle. The energy-like argument and both reference arguments may be finite real or complex scalars, and the result is returned as a Python complex number. Invalid input raises ValueError when any argument is not a finite scalar, when the cut and normalization references coincide, when the radicands or the result overflow to a non-finite value, or when the sum of the two principal square roots in the denominator is zero.

```python
import numpy as np


def leftcut_map(x, cut_reference, normalization_reference):
    """Return the conformal variable with the left-hand cut on the unit circle.

    Parameters
    ----------
    x : float or complex
        Finite real or complex value of the intermediate map.
    cut_reference : float or complex
        Finite mapped position at which the left-hand cut opens.
    normalization_reference : float or complex
        Finite mapped point at which the returned variable must vanish.

    Returns
    -------
    complex
        Value of the left-hand-cut conformal map.

    Raises
    ------
    ValueError
        If any argument is not a finite real or complex scalar, if the two
        reference arguments coincide, if the radicands or the result overflow
        to a non-finite value, or if the sum of the principal square roots in
        the denominator is zero or a nonzero radicand underflows to zero.
    """
    return None
```

### Step 4

04_bounded_variable

Goal
----
Compose the four-sheet two-threshold map with the left-cut map to produce one bounded complex expansion variable. The inputs specify the squared energy, two ordered channel thresholds, the left-cut opening, the normalization point and the requested sheet. The cut reference is mapped on sheet 21, the normalization reference is mapped on sheet 11 and the squared energy is mapped on the requested sheet before the piecewise composition is applied. The function returns a native Python complex number and raises ValueError for non-finite or non-scalar data, non-real geometry values, unordered thresholds, an unknown sheet, coincident mapped references, a vanishing conformal denominator including the outer reciprocal, an outer branch at the lower threshold or an overflowing map.

```python
import numpy as np


def bounded_variable(s, lower_threshold, upper_threshold, cut_opening, normalization_point, sheet):
    """Return the bounded expansion variable on one of four sheets.

    Parameters
    ----------
    s : complex
        Finite scalar squared energy at which to evaluate the variable.
    lower_threshold : float
        Finite real lower channel threshold in squared energy units.
    upper_threshold : float
        Finite real upper channel threshold in squared energy units.
    cut_opening : float
        Finite real squared energy at which the left-hand cut opens.
    normalization_point : float
        Finite real squared energy at which the bounded variable is normalized.
    sheet : str
        Four-sheet label, one of "11", "21", "22" or "12".

    Returns
    -------
    complex
        The bounded expansion variable as a native Python complex number.

    Raises
    ------
    ValueError
        If the squared energy is not a finite scalar, if any geometry value is
        not a finite real scalar, if the thresholds are not strictly ordered,
        if the sheet is unknown, if the mapped references coincide, if a
        conformal denominator vanishes (including the outer reciprocal), if an
        outer branch is requested at the lower threshold, the normalization
        point is not strictly below the lower threshold, its intermediate image
        equals either unit endpoint in double precision or a constituent map
        rejects an unsupported numerical range.
    """
    return None
```

### Step 5

05_pole_factor

Goal
----
Build the reciprocal product associated with a bounded expansion variable and a non-empty sequence of mapped poles. Each input is validated as a finite scalar, every pole contributes one factor with itself and its conjugate and the completed product is returned as a native Python complex number. Invalid input raises ValueError when the evaluation point is not finite, the pole collection is not a non-empty sequence, a pole is not finite, the evaluation point coincides with a pole or its conjugate or the nonzero final reciprocal product overflows or underflows the finite complex range. Intermediate products must be scaled so representable results are not rejected merely because the poles span widely separated magnitudes.

```python
from collections.abc import Sequence
import numpy as np


def pole_factor(variable_value, mapped_poles):
    """Return the reciprocal product over mapped poles and their conjugates.

    Parameters
    ----------
    variable_value : complex
        Finite scalar value of the bounded expansion variable.
    mapped_poles : sequence of complex
        Non-empty sequence of finite mapped pole positions.

    Returns
    -------
    complex
        The reciprocal pole product as a native Python complex number.

    Raises
    ------
    ValueError
        If the variable value is not a finite scalar, if the mapped poles are
        not a non-empty sequence, if any pole is not a finite scalar or if the
        variable value coincides with a pole or its conjugate, or the nonzero
        final reciprocal product overflows or underflows the finite complex
        output range. Representable results require scaled intermediate products.
    """
    return None
```

### Step 6

06_form_factor

Goal
----
Evaluate the bounded form factor at one squared energy on a specified Riemann sheet. The two ordered thresholds, the cut opening and the normalization point define the conformal variable, while each non-empty pole entry supplies a complex energy and its own sheet and each finite real coefficient supplies one ascending polynomial power. Every pole energy is squared before it is mapped, the reciprocal pole product multiplies the polynomial and the result is returned as a native Python complex number. Invalid input raises ValueError for non-finite or non-scalar data, non-real geometry or coefficients, unordered thresholds, unknown sheets, malformed or empty sequences, coincident mapped references, singular conformal maps including the outer reciprocal or an outer branch at the lower threshold, overflowing squared pole energies or evaluations, or coincidence with a mapped pole or its conjugate.

```python
from collections.abc import Sequence
import numpy as np


def form_factor(s, sheet, lower_threshold, upper_threshold, cut_opening, normalization_point, poles, coefficients):
    """Return the pole-dressed polynomial form factor on one sheet.

    Parameters
    ----------
    s : complex
        Finite scalar squared energy at which to evaluate the form factor.
    sheet : str
        Evaluation sheet, one of "11", "21", "22" or "12".
    lower_threshold : float
        Finite real lower channel threshold in squared energy units.
    upper_threshold : float
        Finite real upper channel threshold in squared energy units.
    cut_opening : float
        Finite real squared energy at which the left-hand cut opens.
    normalization_point : float
        Finite real squared energy strictly below the lower threshold, with
        a physical intermediate image distinct from both unit endpoints in
        double precision. Other normalization geometries raise ValueError.
    poles : sequence of pairs
        Non-empty sequence of complex pole energies and their sheet labels.
    coefficients : sequence of float
        Non-empty sequence of finite real polynomial coefficients in ascending order.

    Returns
    -------
    complex
        The pole-dressed polynomial form factor as a native Python complex number.

    Raises
    ------
    ValueError
        If the squared energy is not a finite scalar, if the geometry values or
        coefficients are not finite real scalars, if the thresholds are not
        strictly ordered, if any sheet is unknown, if either sequence is empty
        or malformed, if mapped references coincide, if a conformal denominator
        vanishes (including the outer reciprocal), if an outer branch is
        requested at the lower threshold, if a squared pole energy, conformal
        map or form-factor evaluation overflows, or if the expansion variable
        coincides with a mapped pole or its conjugate. Unsupported normalization
        geometries (point not below the lower threshold or intermediate image at
        a unit endpoint) and numerical range failures in either constituent map
        or the reciprocal product also raise ValueError.
    """
    return None
```

### Step 7

07_run_pipeline

Goal
----
Run the complete two-threshold form-factor calculation and combine a non-empty sequence of sheet-specific evaluations into one scalar. The two positive masses set the ordered thresholds, the reference points fix the bounded conformal variable, every pole energy is squared and mapped on its own sheet and the ascending polynomial is multiplied by the reciprocal pole product. The function returns the product of the squared form-factor moduli as a native Python float. Invalid input raises ValueError for non-finite or non-scalar data, non-real masses, geometry or coefficients, non-positive or unordered masses, unknown sheets, malformed or empty pole, coefficient or evaluation sequences, coincident mapped references, singular conformal maps including the outer reciprocal or an outer branch at the lower threshold, overflowing thresholds, squared pole energies or evaluations, an overflowing squared-modulus product or coincidence with a mapped pole or its conjugate.

```python
from collections.abc import Sequence
import numpy as np


def run_pipeline(light_mass, heavy_mass, cut_opening, normalization_point, poles, coefficients, evaluations):
    """Return the product of squared form-factor moduli at all evaluations.

    Parameters
    ----------
    light_mass : float
        Finite positive mass of the lighter channel constituent in energy units.
    heavy_mass : float
        Finite positive mass of the heavier channel constituent in energy units.
    cut_opening : float
        Finite real squared energy at which the left-hand cut opens.
    normalization_point : float
        Finite real squared energy at which the bounded variable is normalized.
    poles : sequence of pairs
        Non-empty sequence of complex pole energies and their sheet labels.
    coefficients : sequence of float
        Non-empty sequence of finite real polynomial coefficients in ascending order.
    evaluations : sequence of pairs
        Non-empty sequence of finite squared energies and evaluation sheet labels.

    Returns
    -------
    float
        Product of the squared form-factor moduli as a native Python float.

    Raises
    ------
    ValueError
        If either mass is not a finite positive real scalar, if their thresholds
        are not strictly ordered, if a reference value or coefficient is not a
        finite real scalar, if any energy is not a finite scalar, if any sheet
        is unknown, if a sequence is empty or malformed, if mapped references
        coincide, if a conformal denominator vanishes (including the outer
        reciprocal), if an outer branch is requested at the lower threshold,
        if a threshold, squared pole energy, map, form factor or squared-modulus
        product overflows, or if an evaluation coincides with a mapped pole or
        its conjugate. This includes every individual squared modulus, unsupported
        normalization geometries (point not below the lower threshold or image
        at a unit endpoint) and numerical range refusals in earlier steps.
    """
    return None
```
