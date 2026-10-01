# Physics-Astrophysics-11

## Background

Relativistic continuum kinetic models evolve plasma distributions on phase-space meshes. Smooth mappings in velocity space can concentrate resolution across a broad range of particle energies, while discontinuous finite-element representations exchange information through numerical fluxes at cell faces. This benchmark isolates one internal velocity-space face between two polynomial states in an electromagnetic field and asks for the associated stability diagnostic under the supplied discrete data and quadrature.

## Problem

On \((x,r,b,c)\in[-1,1]^4\), consider two adjacent velocity cells with global \(s_1\) intervals \([-1,0]\) and \([0,1]\), local coordinates \(r_L=2s_1+1\) and \(r_R=2s_1-1\), and a shared face at \(s_1=0\) oriented from the left cell to the right cell in the direction of increasing physical \(u_1\).
The physical four-velocity components use right-handed axes and are mapped by \(u_1=s_1+0.25s_1^2\), \(u_2=0.8b+0.2b^2\), and \(u_3=1.1c+0.25c^2\).

The supplied left and right cell-local tensor-\(Q_2\) mapped conserved polynomials are
\[
\begin{aligned}
q_L={}&2+0.18x+0.12r-0.14b+0.10c+0.16xr+0.11rb-0.09rc\\
&+0.08xb+0.07xc+0.06bc+0.10r^2b^2+0.08x^2c^2+0.05xrc,\\
q_R={}&1.6-0.12x-0.10r+0.16b-0.08c-0.14xr+0.09rb+0.12rc\\
&-0.07xb+0.06xc-0.05bc+0.07r^2c^2+0.09x^2b^2-0.04xrb,
\end{aligned}
\]
where \(J_{\mathrm{loc}}\) denotes the determinant of the cell-local map \((r,b,c)\mapsto(u_1,u_2,u_3)\), including the conversion between \(r\) and global \(s_1\).
In each cell, \(\gamma_h\) is the unique tensor-\(Q_2\) polynomial in \((r,b,c)\) whose values at the tensor Lobatto nodes \(\{-1,0,1\}^3\) equal \(\sqrt{1+u_1^2+u_2^2+u_3^2}\) under that cell's map.
Use \(c_{\mathrm{light}}=m=1\), species charge \(+1\), \(\mathbf E(x)=(0.10+0.24x-0.12x^2,0,0)\), and \(\mathbf B(x)=(0,0.45-0.20x+0.15x^2,0.35+0.25x-0.10x^2)\).

For the requested face estimate, use the ascending three-point Gauss-Legendre nodes \((-\sqrt{3/5},0,\sqrt{3/5})\) and weights \((5/9,8/9,5/9)\) independently along \((x,b,c)\); this prescribed tensor rule defines the discrete target rather than an exact continuous face integral.
For \(Q=\tfrac12\sum_{K=L,R}\int_K J_{\mathrm{loc}}f_h^2\,dx\,dr\,db\,dc\), determine the positive, once-counted instantaneous contribution of this shared internal face to the decrease of \(Q\) using the source's upwind face flux.
Restrict the calculation to this face contribution, with no volume evolution, time step, boundary face, or other internal face included.
Report one finite decimal; answers within absolute tolerance \(10^{-10}\) are accepted.

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

mapped_velocity_geometry

Goal
----
Step 01: mapped geometry for two cells sharing a velocity-space face.

```python
import json
import numpy as np
from numpy.polynomial import polynomial as _geometry_poly

def mapped_velocity_geometry(case_json):
    """Return mapped volume, face, and tangential geometry.

    Parameters
    ----------
    case_json : str
        JSON string encoding a complete two-cell case as an object.
        A supplied string is evaluated using its own coefficient values.

        Required JSON members:
        - ``map_coefficients_u1_u2_u3``: three length-3 lists giving the
          coefficients of ``u1(s1)``, ``u2(s2)``, and ``u3(s3)`` in
          ascending powers ``[constant, linear, quadratic]``.
        - ``electric_normal_coefficients``: length-3 list of coefficients
          of ``E1(x)`` in ascending powers.
        - ``magnetic_s2_coefficients``: length-3 list of coefficients of
          ``B2(x)`` in ascending powers.
        - ``magnetic_s3_coefficients``: length-3 list of coefficients of
          ``B3(x)`` in ascending powers.
        - ``q_left_terms``: nonempty list of left-cell polynomial terms.
        - ``q_right_terms``: nonempty list of right-cell polynomial terms.

        Each term is ``[coefficient, power_x, power_r, power_s2, power_s3]``
        and represents the coefficient times the corresponding monomial.
        The polynomial is the sum of its terms. Powers are integers in
        ``{0, 1, 2}``; coefficients are finite real numbers. Booleans are
        not accepted as coefficients or powers.

        Here ``s2=b`` and ``s3=c``. Each term's ``r`` is local to its cell:
        ``r_L=2*s1+1`` for ``s1`` in ``[-1,0]`` and ``r_R=2*s1-1`` for
        ``s1`` in ``[0,1]``. Every velocity map must have a strictly positive
        derivative throughout ``[-1,1]``. All six JSON members are required
        for every supplied case. An optional ``id`` is descriptive only.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(4, 2, 3, 3, 3)`` with axes
        ``(quantity,cell,r,s2,s3)``.  Layers are full local volume Jacobian,
        induced face Jacobian, ``du2/ds2``, and ``du3/ds3``.

    Raises
    ------
    ValueError
        If the JSON or required arrays are malformed, non-finite, or any map
        is not strictly increasing on its reference interval.
    """
    return
```

### Step 2

discrete_face_hamiltonian_gradient

Goal
----
Step 02: tangential gradient of the shared discrete Hamiltonian face.

```python
import json
import numpy as np
from numpy.polynomial import polynomial as _hamiltonian_poly

def discrete_face_hamiltonian_gradient(case_json):
    """Return the two tangential derivatives of the discrete face state.

    Parameters
    ----------
    case_json : str
        JSON string encoding a complete two-cell case as an object.
        A supplied string is evaluated using its own coefficient values.

        Required JSON members:
        - ``map_coefficients_u1_u2_u3``: three length-3 lists giving the
          coefficients of ``u1(s1)``, ``u2(s2)``, and ``u3(s3)`` in
          ascending powers ``[constant, linear, quadratic]``.
        - ``electric_normal_coefficients``: length-3 list of coefficients
          of ``E1(x)`` in ascending powers.
        - ``magnetic_s2_coefficients``: length-3 list of coefficients of
          ``B2(x)`` in ascending powers.
        - ``magnetic_s3_coefficients``: length-3 list of coefficients of
          ``B3(x)`` in ascending powers.
        - ``q_left_terms``: nonempty list of left-cell polynomial terms.
        - ``q_right_terms``: nonempty list of right-cell polynomial terms.

        Each term is ``[coefficient, power_x, power_r, power_s2, power_s3]``
        and represents the coefficient times the corresponding monomial.
        The polynomial is the sum of its terms. Powers are integers in
        ``{0, 1, 2}``; coefficients are finite real numbers. Booleans are
        not accepted as coefficients or powers.

        Here ``s2=b`` and ``s3=c``. Each term's ``r`` is local to its cell:
        ``r_L=2*s1+1`` for ``s1`` in ``[-1,0]`` and ``r_R=2*s1-1`` for
        ``s1`` in ``[0,1]``. Every velocity map must have a strictly positive
        derivative throughout ``[-1,1]``. All six JSON members are required
        for every supplied case. An optional ``id`` is descriptive only.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(2,3,3)``.  Layers contain the ``s2`` and
        ``s3`` derivatives at ascending GL3 ``(s2,s3)`` nodes.

    Raises
    ------
    ValueError
        If the JSON or required arrays are malformed, non-finite, or any map
        is not strictly increasing on its reference interval.
    """
    return
```

### Step 3

physical_distribution_nodal_state

Goal
----
Step 03: physical tensor-Q2 states at native volume nodes.

```python
import json
import numpy as np

def physical_distribution_nodal_state(case_json):
    """Return the two physical-distribution states on native volume nodes.

    Parameters
    ----------
    case_json : str
        JSON string encoding a complete two-cell case as an object.
        A supplied string is evaluated using its own coefficient values.

        Required JSON members:
        - ``map_coefficients_u1_u2_u3``: three length-3 lists giving the
          coefficients of ``u1(s1)``, ``u2(s2)``, and ``u3(s3)`` in
          ascending powers ``[constant, linear, quadratic]``.
        - ``electric_normal_coefficients``: length-3 list of coefficients
          of ``E1(x)`` in ascending powers.
        - ``magnetic_s2_coefficients``: length-3 list of coefficients of
          ``B2(x)`` in ascending powers.
        - ``magnetic_s3_coefficients``: length-3 list of coefficients of
          ``B3(x)`` in ascending powers.
        - ``q_left_terms``: nonempty list of left-cell polynomial terms.
        - ``q_right_terms``: nonempty list of right-cell polynomial terms.

        Each term is ``[coefficient, power_x, power_r, power_s2, power_s3]``
        and represents the coefficient times the corresponding monomial.
        The polynomial is the sum of its terms. Powers are integers in
        ``{0, 1, 2}``; coefficients are finite real numbers. Booleans are
        not accepted as coefficients or powers.

        Here ``s2=b`` and ``s3=c``. Each term's ``r`` is local to its cell:
        ``r_L=2*s1+1`` for ``s1`` in ``[-1,0]`` and ``r_R=2*s1-1`` for
        ``s1`` in ``[0,1]``. Every velocity map must have a strictly positive
        derivative throughout ``[-1,1]``. All six JSON members are required
        for every supplied case. An optional ``id`` is descriptive only.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(2,3,3,3,3)`` with axes
        ``(cell,x,r,s2,s3)`` on ascending GL3 nodes.

    Raises
    ------
    ValueError
        If the JSON or required arrays are malformed, non-finite, any power
        lies outside tensor Q2, or the mapped geometry is inadmissible.
    """
    return
```

### Step 4

distribution_face_traces

Goal
----
Step 04: physical-distribution values on the shared cell face.

```python
import json
import numpy as np
from numpy.polynomial import polynomial as _trace_poly

def distribution_face_traces(case_json):
    """Return the left and right physical-distribution face traces.

    Parameters
    ----------
    case_json : str
        JSON string encoding a complete two-cell case as an object.
        A supplied string is evaluated using its own coefficient values.

        Required JSON members:
        - ``map_coefficients_u1_u2_u3``: three length-3 lists giving the
          coefficients of ``u1(s1)``, ``u2(s2)``, and ``u3(s3)`` in
          ascending powers ``[constant, linear, quadratic]``.
        - ``electric_normal_coefficients``: length-3 list of coefficients
          of ``E1(x)`` in ascending powers.
        - ``magnetic_s2_coefficients``: length-3 list of coefficients of
          ``B2(x)`` in ascending powers.
        - ``magnetic_s3_coefficients``: length-3 list of coefficients of
          ``B3(x)`` in ascending powers.
        - ``q_left_terms``: nonempty list of left-cell polynomial terms.
        - ``q_right_terms``: nonempty list of right-cell polynomial terms.

        Each term is ``[coefficient, power_x, power_r, power_s2, power_s3]``
        and represents the coefficient times the corresponding monomial.
        The polynomial is the sum of its terms. Powers are integers in
        ``{0, 1, 2}``; coefficients are finite real numbers. Booleans are
        not accepted as coefficients or powers.

        Here ``s2=b`` and ``s3=c``. Each term's ``r`` is local to its cell:
        ``r_L=2*s1+1`` for ``s1`` in ``[-1,0]`` and ``r_R=2*s1-1`` for
        ``s1`` in ``[0,1]``. Every velocity map must have a strictly positive
        derivative throughout ``[-1,1]``. All six JSON members are required
        for every supplied case. An optional ``id`` is descriptive only.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(2,3,3,3)``.  Row 0 is the left-cell trace at
        ``r=+1`` and row 1 is the right-cell trace at ``r=-1``.

    Raises
    ------
    ValueError
        If ``case_json`` is invalid or the requested trace result has the wrong
        shape or contains a non-finite value.
    """
    return
```

### Step 5

mapped_normal_face_transport

Goal
----
Step 05: signed mapped transport on the shared velocity-space face.

```python
import json
import numpy as np
from numpy.polynomial import polynomial as _transport_poly

def mapped_normal_face_transport(case_json):
    """Return the signed mapped normal face-transport array.

    Parameters
    ----------
    case_json : str
        JSON string encoding a complete two-cell case as an object.
        A supplied string is evaluated using its own coefficient values.

        Required JSON members:
        - ``map_coefficients_u1_u2_u3``: three length-3 lists giving the
          coefficients of ``u1(s1)``, ``u2(s2)``, and ``u3(s3)`` in
          ascending powers ``[constant, linear, quadratic]``.
        - ``electric_normal_coefficients``: length-3 list of coefficients
          of ``E1(x)`` in ascending powers.
        - ``magnetic_s2_coefficients``: length-3 list of coefficients of
          ``B2(x)`` in ascending powers.
        - ``magnetic_s3_coefficients``: length-3 list of coefficients of
          ``B3(x)`` in ascending powers.
        - ``q_left_terms``: nonempty list of left-cell polynomial terms.
        - ``q_right_terms``: nonempty list of right-cell polynomial terms.

        Each term is ``[coefficient, power_x, power_r, power_s2, power_s3]``
        and represents the coefficient times the corresponding monomial.
        The polynomial is the sum of its terms. Powers are integers in
        ``{0, 1, 2}``; coefficients are finite real numbers. Booleans are
        not accepted as coefficients or powers.

        Here ``s2=b`` and ``s3=c``. Each term's ``r`` is local to its cell:
        ``r_L=2*s1+1`` for ``s1`` in ``[-1,0]`` and ``r_R=2*s1-1`` for
        ``s1`` in ``[0,1]``. Every velocity map must have a strictly positive
        derivative throughout ``[-1,1]``. All six JSON members are required
        for every supplied case. An optional ``id`` is descriptive only.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(3,3,3)`` containing the signed mapped normal
        face-transport coefficient.

    Raises
    ------
    ValueError
        If the JSON, serialized inputs, or requested result is malformed or
        non-finite.
    """
    return
```

### Step 6

paired_upwind_face_loss

Goal
----
Step 06: paired upwind loss across one internal velocity-space face.

```python
import json
import numpy as np

def paired_upwind_face_loss(case_json):
    """Return the nonnegative paired upwind face-loss scalar.

    Parameters
    ----------
    case_json : str
        JSON string encoding a complete two-cell case as an object.
        A supplied string is evaluated using its own coefficient values.

        Required JSON members:
        - ``map_coefficients_u1_u2_u3``: three length-3 lists giving the
          coefficients of ``u1(s1)``, ``u2(s2)``, and ``u3(s3)`` in
          ascending powers ``[constant, linear, quadratic]``.
        - ``electric_normal_coefficients``: length-3 list of coefficients
          of ``E1(x)`` in ascending powers.
        - ``magnetic_s2_coefficients``: length-3 list of coefficients of
          ``B2(x)`` in ascending powers.
        - ``magnetic_s3_coefficients``: length-3 list of coefficients of
          ``B3(x)`` in ascending powers.
        - ``q_left_terms``: nonempty list of left-cell polynomial terms.
        - ``q_right_terms``: nonempty list of right-cell polynomial terms.

        Each term is ``[coefficient, power_x, power_r, power_s2, power_s3]``
        and represents the coefficient times the corresponding monomial.
        The polynomial is the sum of its terms. Powers are integers in
        ``{0, 1, 2}``; coefficients are finite real numbers. Booleans are
        not accepted as coefficients or powers.

        Here ``s2=b`` and ``s3=c``. Each term's ``r`` is local to its cell:
        ``r_L=2*s1+1`` for ``s1`` in ``[-1,0]`` and ``r_R=2*s1-1`` for
        ``s1`` in ``[0,1]``. Every velocity map must have a strictly positive
        derivative throughout ``[-1,1]``. All six JSON members are required
        for every supplied case. An optional ``id`` is descriptive only.

    Returns
    -------
    float
        The nonnegative paired-face loss evaluated with the prescribed GL3
        rule.

    Raises
    ------
    ValueError
        If the JSON is invalid, an upstream array is malformed or non-finite,
        or the resulting loss is not finite and nonnegative.
    """
    return
```

### Step 7

mapped_velocity_face_damping

Goal
----
Step 07 (final orchestrator): mapped velocity-face damping.

```python
import json
import numpy as np

def mapped_velocity_face_damping(case_json=None):
    """Run the mapped two-cell face construction and return its damping.

    Parameters
    ----------
    case_json : str or None, optional
        JSON string encoding a complete two-cell case. ``None`` selects
        the reference maps, fields, and conserved polynomials in the
        Problem statement. A supplied string is evaluated using its own
        coefficient values.

        Required JSON members:
        - ``map_coefficients_u1_u2_u3``: three length-3 lists giving the
          coefficients of ``u1(s1)``, ``u2(s2)``, and ``u3(s3)`` in
          ascending powers ``[constant, linear, quadratic]``.
        - ``electric_normal_coefficients``: length-3 list of coefficients
          of ``E1(x)`` in ascending powers.
        - ``magnetic_s2_coefficients``: length-3 list of coefficients of
          ``B2(x)`` in ascending powers.
        - ``magnetic_s3_coefficients``: length-3 list of coefficients of
          ``B3(x)`` in ascending powers.
        - ``q_left_terms``: nonempty list of left-cell polynomial terms.
        - ``q_right_terms``: nonempty list of right-cell polynomial terms.

        Each term is ``[coefficient, power_x, power_r, power_s2, power_s3]``
        and represents the coefficient times the corresponding monomial.
        The polynomial is the sum of its terms. Powers are integers in
        ``{0, 1, 2}``; coefficients are finite real numbers. Booleans are
        not accepted as coefficients or powers.

        Here ``s2=b`` and ``s3=c``. Each term's ``r`` is local to its cell:
        ``r_L=2*s1+1`` for ``s1`` in ``[-1,0]`` and ``r_R=2*s1-1`` for
        ``s1`` in ``[0,1]``. Every velocity map must have a strictly positive
        derivative throughout ``[-1,1]``. All six JSON members are required
        for every supplied case. An optional ``id`` is descriptive only.

    Returns
    -------
    float
        The mapped velocity-face damping scalar.

    Raises
    ------
    ValueError
        If the JSON is invalid or any stage returns malformed, non-finite, or
        physically inadmissible data.
    """
    return
```
