"""
**Final orchestrator.** Run the BDF$3$ convex-splitting scheme on each of five fixed configurations and return the sum of the five terminal **modified** discrete energies. Nothing about a configuration is hard-wired except the numbers in its row: in particular the stabilization parameter is derived, not given. For each configuration, in order: call the certificate of the earlier sub-problem at $q=3$, at the published $q=3$ decomposition constants $\\kappa_3$ and $\\eta_3$ - given in the formulas below - and at that configuration's $\\varepsilon$, read $S_{\\min}$ off it and set $S=s\\,S_{\\min}$ with $s$ the configuration's safety factor; build the initial field $\\phi_0$ from its seed; set the start-up layers $\\phi^0=\\phi^1=\\phi^2=\\phi_0$ and carry a fourth copy, so that even a run of zero steps holds the $q+1=4$ levels the modified energy needs; take the prescribed number of steps with the step of the earlier sub-problem; and evaluate the modified discrete energy of the last four levels with $\\mathbf G_3$, $\\mathbf J_3$ and that configuration's $S$ and $\\tau$. The five configurations, in this order, are P1: trigonometric seed, $(L_x,L_y)=(8\\pi,8\\pi)$, $(M_x,M_y)=(32,32)$, $\\varepsilon=0.25$, $s=200$, $\\tau=0.05$, $12$ steps; P2: nucleus seed, $(25,25)$, $(32,32)$, $\\varepsilon=0.50$, $s=100$, $\\tau=0.10$, $20$ steps; P3: trigonometric, $(6\\pi,10\\pi)$, $(24,40)$, $\\varepsilon=0.40$, $s=250$, $\\tau=0.04$, $15$ steps; P4: nucleus, $(32,20)$, $(48,24)$, $\\varepsilon=0.30$, $s=50$, $\\tau=0.08$, $18$ steps; P5: trigonometric, $(10\\pi,4\\pi)$, $(40,24)$, $\\varepsilon=0.20$, $s=120$, $\\tau=0.06$, $14$ steps. The keyword arguments override, for every configuration at once, the mesh, the step size, the number of steps and the stabilization parameter - an $S$ passed explicitly bypasses the certificate; `configs` restricts the sum to the named subset. With no arguments the function returns the graded answer.

The five configurations are chosen to exercise the parts of the discretisation that a partial implementation would get wrong, and to stop while the energy is still falling steeply, so that the terminal value is sensitive to the scheme rather than to the equilibrium it eventually reaches. Every run is at $q=3$; three of the five boxes are rectangular and two of those have $h_x\\ne h_y$, so no scalar $L$ or scalar $M$ will do; and the stabilization parameter is derived from the certificate at five different $\\varepsilon$ and safety factors, so its exponent and its placement both matter. The two seeds probe different regimes. The trigonometric seed is a smooth superposition of modes at the wavelength the operator $(1+\\Delta)$ selects, so it relaxes quickly and its energy falls by one to two orders of magnitude over the run; its discrete mean is exactly $0.07$, which makes the mass-conservation check sharp. The nucleus seed is a single tanh-profiled crystallite in the middle of a large box, the standard benchmark configuration for crystal growth, and it relaxes far more slowly, so its terminal energy is still close to its initial one. Because the objective is a plain sum of per-configuration quantities it is exactly additive over any partition of the five, and the zero-step case returns the sum of the five initial energies, which isolates the energy functional from the time stepping entirely.

$$\\text{P1}:\\;(8\\pi,8\\pi),\\;(32,32),\\;\\varepsilon=0.25,\\;s=200,\\;\\tau=0.05,\\;n=12;$$

$$\\text{P2}:\\;(25,25),\\;(32,32),\\;\\varepsilon=0.50,\\;s=100,\\;\\tau=0.10,\\;n=20;$$

$$\\text{P3}:\\;(6\\pi,10\\pi),\\;(24,40),\\;\\varepsilon=0.40,\\;s=250,\\;\\tau=0.04,\\;n=15;$$

$$\\text{P4}:\\;(32,20),\\;(48,24),\\;\\varepsilon=0.30,\\;s=50,\\;\\tau=0.08,\\;n=18;$$

$$\\text{P5}:\\;(10\\pi,4\\pi),\\;(40,24),\\;\\varepsilon=0.20,\\;s=120,\\;\\tau=0.06,\\;n=14.$$



Every run uses $q=3$. The grid is $x_i=iL_x/M_x$, $y_j=jL_y/M_y$. The **trigonometric** seed is $\\phi_0=0.07+0.60\\big(\\cos x\\cos y+0.40\\sin(2x)\\cos y+0.30\\cos(x-2y)\\big)$ in those physical coordinates; the **nucleus** seed is $\\phi_0=2.5\\big(1-\\tanh(\\tfrac12(r-2))\\big)$ with $r=\\sqrt{(x-L_x/2)^2+(y-L_y/2)^2}$. The stabilization parameter of configuration $c$ is $S_c=s_c\\,S_{\\min}(\\varepsilon_c)$, with $S_{\\min}$ read off the certificate of the earlier sub-problem evaluated at $q=3$, at the decomposition data below and at that configuration's $\\varepsilon$, and



$$\\text{answer}=\\sum_{c\\in\\{\\text{P1},\\dots,\\text{P5}\\}}\\hat E\\big(\\phi^{n_c-1}_c,\\phi^{n_c}_c,\\phi^{n_c+1}_c,\\phi^{n_c+2}_c\\big),$$



the modified energy of the last four levels of each run, evaluated with $\\mathbf G_3$, $\\mathbf J_3$, $S_c$ and $\\tau_c$.



The published $q=3$ decomposition data to use are $\\kappa_3=95/48$ and $\\eta_3=1/2$, and, in the slot ordering $(\\phi^{n-2},\\phi^{n-1},\\phi^{n})$ of the earlier sub-problem,



$$\\mathbf G_3=\\begin{pmatrix}0&0&0\\\\0&1/6&-7/24\\\\0&-7/24&65/96\\end{pmatrix},\\qquad\\mathbf J_3=\\begin{pmatrix}0&0&0\\\\0&1/2&-1/2\\\\0&-1/2&1\\end{pmatrix}.$$

Returns
-------
A Python `float`, finite. With no arguments it is the graded answer. The value is exactly additive over any partition of the five tags, so `f(configs=A) + f(configs=B)` reproduces `f()` whenever `A` and `B` partition the five. With `n_steps=0` it returns the sum of the five INITIAL ORIGINAL energies: no step is taken, so every first difference in the start-up history vanishes and the three correction terms of the modified energy are identically zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sqpfc_terminal_energy_sum(configs: "Sequence[str] | None" = None,
                              mesh: "int | Sequence[int] | None" = None,
                              tau: "float | None" = None,
                              n_steps: "int | None" = None,
                              S: "float | None" = None) -> float:
    """configs: list of configuration tags to include; None means all five.
    mesh, tau, n_steps, S: if given, override that setting for every
    configuration; None means use each configuration's own value.
    Return the float sum of the terminal modified discrete energies.
    Raise ValueError if configs names a tag that is not one of the five,
    if n_steps is negative, if tau is not positive, or if S is negative."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _seed_field(kind, M, L):
    a = np.atleast_1d(np.asarray(M, float))
    Mx, My = int(round(float(a.flat[0]))), int(round(float(a.flat[-1])))
    b = np.atleast_1d(np.asarray(L, float))
    Lx, Ly = float(b.flat[0]), float(b.flat[-1])
    X, Y = np.meshgrid(np.arange(Mx) * Lx / Mx, np.arange(My) * Ly / My,
                       indexing="ij")
    if kind == "trig":
        return 0.07 + 0.60 * (np.cos(X) * np.cos(Y)
                              + 0.40 * np.sin(2.0 * X) * np.cos(Y)
                              + 0.30 * np.cos(X - 2.0 * Y))
    r = np.sqrt((X - 0.5 * Lx) ** 2 + (Y - 0.5 * Ly) ** 2)
    return 2.5 * (1.0 - np.tanh(0.5 * (r - 2.0)))


def _oracle_sqpfc_terminal_energy_sum(configs: "Sequence[str] | None" = None,
                                      mesh: "int | Sequence[int] | None" = None,
                                      tau: "float | None" = None,
                                      n_steps: "int | None" = None,
                                      S: "float | None" = None) -> float:
    G3 = np.array([[0.0, 0.0, 0.0],
                   [0.0, 1.0 / 6.0, -7.0 / 24.0],
                   [0.0, -7.0 / 24.0, 65.0 / 96.0]])
    J3 = np.array([[0.0, 0.0, 0.0],
                   [0.0, 0.5, -0.5],
                   [0.0, -0.5, 1.0]])
    kappa3 = 95.0 / 48.0
    eta3 = 0.5
    qq = 3
    table = (
        ("P1", "trig",    (8.0 * np.pi, 8.0 * np.pi),  (32, 32), 0.25, 200.0, 0.05, 12),
        ("P2", "nucleus", (25.0, 25.0),                (32, 32), 0.50, 100.0, 0.10, 20),
        ("P3", "trig",    (6.0 * np.pi, 10.0 * np.pi), (24, 40), 0.40, 250.0, 0.04, 15),
        ("P4", "nucleus", (32.0, 20.0),                (48, 24), 0.30,  50.0, 0.08, 18),
        ("P5", "trig",    (10.0 * np.pi, 4.0 * np.pi), (40, 24), 0.20, 120.0, 0.06, 14),
    )
    names = [c[0] for c in table] if configs is None else list(configs)
    known = [c[0] for c in table]
    for nm in names:
        if nm not in known:
            raise ValueError("unknown configuration tag: %r" % (nm,))
    if n_steps is not None and int(n_steps) < 0:
        raise ValueError("n_steps must be non-negative")
    if tau is not None and not float(tau) > 0.0:
        raise ValueError("tau must be positive")
    if S is not None and float(S) < 0.0:
        raise ValueError("S must be non-negative")
    total = 0.0
    for nm, kind, L, Mc, eps, sfac, tc, nc in table:
        if nm not in names:
            continue
        Mv = Mc if mesh is None else mesh
        tv = tc if tau is None else float(tau)
        nv = nc if n_steps is None else int(n_steps)
        cert = _oracle_sqpfc_stabilization_certificate(qq, kappa3, eta3,
                                                       eps, 1.0)
        Sv = sfac * float(cert[2]) if S is None else float(S)
        u = _seed_field(kind, Mv, L)
        hist = [u] * (qq + 1)
        for _ in range(nv):
            hist.append(_oracle_sqpfc_convex_split_step(
                np.stack(hist[-qq:]), L, eps, Sv, tv))
        total += _oracle_sqpfc_modified_energy(
            np.stack(hist[-(qq + 1):]), L, eps, Sv, tv, G3, J3)
    return total

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # a cheap common override, so the case is not just the headline again
        {"setup": "",
         "call": "sqpfc_terminal_energy_sum(mesh=(16, 16), tau=0.02, n_steps=4)",
         "gold_call": ("_oracle_sqpfc_terminal_energy_sum(mesh=(16, 16), tau=0.02,"
                       " n_steps=4)")},
        # a single configuration, at its own settings
        {"setup": "",
         "call": "sqpfc_terminal_energy_sum(configs=['P1'])",
         "gold_call": "_oracle_sqpfc_terminal_energy_sum(configs=['P1'])"},
        {"setup": "",
         "call": "sqpfc_terminal_energy_sum(configs=['P4'])",
         "gold_call": "_oracle_sqpfc_terminal_energy_sum(configs=['P4'])"},
        # zero steps: the sum of the five initial energies
        {"setup": "",
         "call": "sqpfc_terminal_energy_sum(n_steps=0)",
         "gold_call": "_oracle_sqpfc_terminal_energy_sum(n_steps=0)"},
        # the stabilization override, on a cheap mesh
        {"setup": "",
         "call": ("sqpfc_terminal_energy_sum(mesh=(16, 16), tau=0.02,"
                  " n_steps=3, S=0.0)"),
         "gold_call": ("_oracle_sqpfc_terminal_energy_sum(mesh=(16, 16), tau=0.02,"
                       " n_steps=3, S=0.0)")},
        # additivity over a partition of the five tags
        {"setup": "",
         "call": ("sqpfc_terminal_energy_sum(configs=['P2', 'P3'])"
                  " + sqpfc_terminal_energy_sum(configs=['P1', 'P4', 'P5'])"),
         "gold_call": "_oracle_sqpfc_terminal_energy_sum()"},
        # the graded answer itself
        {"setup": "",
         "call": "sqpfc_terminal_energy_sum()",
         "gold_call": "_oracle_sqpfc_terminal_energy_sum()"},
        # contract: the five tags are the whole instance, so an unknown tag must
        # raise rather than be skipped, which would silently return a partial sum.
        {"setup": ("def trap(f):\n"
                  "    try:\n"
                  "        return float(np.asarray(f()).ravel()[0])\n"
                  "    except ValueError:\n"
                  "        return -12345.0\n"),
         "call": "trap(lambda: sqpfc_terminal_energy_sum(configs=['P9']))",
         "gold_call": ("trap(lambda: _oracle_sqpfc_terminal_energy_sum("
                       "configs=['P9']))")},
    ]
