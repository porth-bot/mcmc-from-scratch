# Derivations

Everything the code does, proved or derived. Sections map onto modules:
Sec. 2 → `mcmc/metropolis.py`, Sec. 3 → `mcmc/gibbs.py`, Sec. 4 → `mcmc/hmc.py`,
Sec. 5 → `mcmc/models.py`, Sec. 6 → `mcmc/diagnostics.py`.

## 1. The problem

We want expectations under a distribution known only up to a constant,

$$\pi(x) = \frac{\tilde\pi(x)}{Z}, \qquad Z = \int \tilde\pi(x)\,dx \text{ unknown},$$

which is exactly the situation in Bayesian inference: the posterior
$p(\theta \mid y) \propto p(y \mid \theta)\, p(\theta)$ has an intractable
normalizer $p(y)$. MCMC constructs a Markov chain $x_0, x_1, \dots$ whose
stationary distribution is $\pi$; then, under conditions in Sec. 1.2, time
averages converge:

$$\frac{1}{N}\sum_{t=1}^{N} f(x_t) \;\xrightarrow{\text{a.s.}}\; \mathbb{E}_\pi[f].$$

Everything below only ever evaluates **ratios** $\tilde\pi(x')/\tilde\pi(x)$,
so $Z$ cancels — that cancellation is the entire reason MCMC works with
unnormalized densities.

### 1.1 Invariance and detailed balance

A transition kernel $P(x \to x')$ leaves $\pi$ **invariant** if
$\int \pi(x) P(x \to x')\,dx = \pi(x')$. A stronger, easier-to-verify
condition is **detailed balance** (reversibility):

$$\pi(x)\,P(x \to x') = \pi(x')\,P(x' \to x) \quad \forall x, x'.$$

Integrating both sides over $x$ gives invariance immediately. Detailed
balance is sufficient, not necessary — but every sampler here satisfies it
(Gibbs blockwise, HMC after the momentum-flip construction).

### 1.2 From invariance to convergence

Invariance alone is not enough (a chain stuck at its start point trivially
preserves any $\pi$ supported there). The chain must also be
$\pi$-**irreducible** (every set with positive $\pi$-mass is reachable) and
**aperiodic**. For Harris-recurrent chains this yields the ergodic law of
large numbers and convergence of the marginal distribution of $x_t$ to $\pi$
in total variation (Meyn & Tweedie 2009; Roberts & Rosenthal 2004 is a
readable survey). A Gaussian random-walk proposal is positive everywhere, so
RWMH on a positive continuous target is irreducible and aperiodic; the same
holds for HMC because the momentum refresh is full-support.

## 2. Metropolis–Hastings

From $x$, propose $x' \sim q(\cdot \mid x)$ and accept with probability

$$\alpha(x \to x') = \min\!\left(1,\; \frac{\tilde\pi(x')\, q(x \mid x')}{\tilde\pi(x)\, q(x' \mid x)}\right),$$

otherwise stay at $x$.

**Claim: the resulting kernel satisfies detailed balance.**
For $x \ne x'$ the transition density is $P(x \to x') = q(x' \mid x)\,\alpha(x \to x')$. Then

$$\pi(x)\,q(x' \mid x)\,\min\!\left(1, \frac{\pi(x')q(x \mid x')}{\pi(x)q(x' \mid x)}\right)
= \min\Big(\pi(x)\,q(x' \mid x),\; \pi(x')\,q(x \mid x')\Big),$$

which is **symmetric under $x \leftrightarrow x'$**, hence equals
$\pi(x')\,P(x' \to x)$. The $x = x'$ (rejection) term balances trivially. ∎

With a symmetric proposal $q(x' \mid x) = q(x \mid x')$ (our Gaussian random
walk $x' = x + \sigma\varepsilon$), the ratio collapses to
$\min(1, \tilde\pi(x')/\tilde\pi(x))$: *always accept moves uphill, accept
downhill moves with probability equal to the density ratio*. In code this is
done in log space, $\log U < \log\tilde\pi(x') - \log\tilde\pi(x)$, to avoid
overflow.

**Scaling facts** (Roberts, Gelman & Gilks 1997): for product-form targets in
dimension $d \to \infty$, the optimal proposal scale is
$\sigma \propto d^{-1/2}$ and the optimal acceptance rate tends to
$\approx 0.234$ (in low dimension the optimum is higher). Two consequences
visible in our experiments: RWMH needs more, smaller steps as $d$ grows, and
even at the optimum it explores diffusively — distance covered grows like
$\sqrt{N}$, which is why its autocorrelation time on the correlated Gaussian
is ~37× HMC's.

## 3. Gibbs sampling

Partition $x = (x_1, \dots, x_K)$ into blocks and repeatedly resample block
$i$ from its **full conditional** $\pi(x_i \mid x_{-i})$.

**Claim: a full-conditional update is an MH move that is always accepted.**
Take the proposal $q(x' \mid x) = \pi(x_i' \mid x_{-i})$ with $x'_{-i} = x_{-i}$.
Using $\pi(x) = \pi(x_i \mid x_{-i})\,\pi(x_{-i})$:

$$\frac{\pi(x')\,q(x \mid x')}{\pi(x)\,q(x' \mid x)}
= \frac{\pi(x_i' \mid x_{-i})\,\pi(x_{-i})\;\pi(x_i \mid x_{-i})}
       {\pi(x_i \mid x_{-i})\,\pi(x_{-i})\;\pi(x_i' \mid x_{-i})} = 1. \;\blacksquare$$

Each block update leaves $\pi$ invariant; a fixed-order sweep (systematic
scan) is a composition of invariant kernels, hence invariant. The sweep is
not reversible as a whole, but invariance is all Sec. 1.2 needs.

### 3.1 Gaussian full conditionals

For $x \sim N(\mu, \Sigma)$ with precision $P = \Sigma^{-1}$, isolate the
terms in $\log\pi$ involving $x_i$:

$$\log\pi = -\tfrac12 (x-\mu)^\top P (x-\mu)
= -\tfrac12 P_{ii} x_i^2 + x_i\Big(P_{ii}\mu_i - \sum_{j\ne i} P_{ij}(x_j - \mu_j)\Big) + \text{const}.$$

A quadratic in $x_i$ is a Gaussian:

$$x_i \mid x_{-i} \sim N\!\Big(\mu_i - \tfrac{1}{P_{ii}}\sum_{j \ne i} P_{ij}(x_j - \mu_j),\; \tfrac{1}{P_{ii}}\Big).$$

**Why Gibbs can still mix slowly.** For the bivariate Gaussian with
correlation $\rho$, substituting the conditionals shows the sub-chain in
$x_1$ is an AR(1) process with coefficient $\rho^2$: each half-sweep loses
only a factor $\rho$ of memory. Lag-1 autocorrelation $\rho^2$ means
$\tau \approx (1+\rho^2)/(1-\rho^2)$, diverging as $\rho \to 1$ — the
coordinate axes are simply the wrong directions to move in. (Verified
empirically in `tests/test_gibbs.py`.)

## 4. Hamiltonian Monte Carlo

### 4.1 Augmentation

Introduce a momentum $p \in \mathbb{R}^d$ and define the joint target

$$\pi(x, p) \propto e^{-H(x,p)}, \qquad
H(x, p) = U(x) + K(p), \quad U = -\log\tilde\pi(x), \quad K = \tfrac12 \lVert p\rVert^2.$$

Because $H$ separates, $x$ and $p$ are independent under the joint; the
$x$-marginal is exactly $\pi$ and $p \sim N(0, I)$. Sampling the joint and
discarding $p$ solves the original problem. (A general mass matrix $M$ gives
$K = \tfrac12 p^\top M^{-1} p$ and $p \sim N(0, M)$; the default is $M = I$, and
Sec. 4.8 adapts a diagonal $M$ as a preconditioner.)

### 4.2 Three properties of Hamiltonian flow

Hamilton's equations
$\dot x = \partial H/\partial p = p$, $\dot p = -\partial H/\partial x = -\nabla U(x)$
generate a flow $\Phi_t$ with:

1. **Energy conservation.**
   $\frac{dH}{dt} = \nabla_x H \cdot \dot x + \nabla_p H \cdot \dot p
   = \nabla U \cdot p + p \cdot (-\nabla U) = 0.$
2. **Volume preservation (Liouville).** The phase-space velocity field
   $V = (\partial H/\partial p, -\partial H/\partial x)$ has divergence
   $\nabla\!\cdot\!V = \sum_i \big(\partial^2 H/\partial x_i \partial p_i - \partial^2 H/\partial p_i \partial x_i\big) = 0$,
   so the flow has unit Jacobian.
3. **Time reversibility.** With $F(x,p) = (x,-p)$:
   $F \circ \Phi_t \circ F = \Phi_t^{-1}$ (run it backwards by flipping momentum).

If we could apply $\Phi_t$ exactly, the proposal $(x', p') = F(\Phi_t(x, p))$
would change $H$ by zero and be accepted always (see 4.4). We can't, so we
discretize — with an integrator chosen to keep properties 2 and 3 **exactly**
and lose only property 1 to $O(\varepsilon^2)$ error.

### 4.3 Leapfrog

One step of size $\varepsilon$ (kick–drift–kick):

$$p_{1/2} = p_0 + \tfrac{\varepsilon}{2}\nabla\log\tilde\pi(x_0), \qquad
x_1 = x_0 + \varepsilon\, p_{1/2}, \qquad
p_1 = p_{1/2} + \tfrac{\varepsilon}{2}\nabla\log\tilde\pi(x_1).$$

- **Volume preservation, exactly.** Each substep is a shear: the kick
  $(x, p) \mapsto (x,\, p + c\,g(x))$ has Jacobian
  $\begin{pmatrix} I & 0 \\ c\,\partial g/\partial x & I \end{pmatrix}$ with
  determinant 1; likewise the drift. A composition of unit-Jacobian maps has
  unit Jacobian.
- **Reversibility, exactly.** The step is a symmetric (palindromic)
  composition, so $F \circ L_\varepsilon^{n} \circ F = L_\varepsilon^{-n}$:
  flip the momentum and the same code retraces its path to float roundoff
  (asserted at $10^{-10}$ in `tests/test_hmc.py`).
- **Energy error, second order.** Symmetric one-step maps have local error
  $O(\varepsilon^3)$, hence global $O(\varepsilon^2)$ over a fixed trajectory
  time $T = L\varepsilon$; moreover leapfrog is *symplectic*, so it exactly
  conserves a nearby "shadow Hamiltonian", which keeps the energy error
  bounded (oscillating, not drifting) over long trajectories. The test
  halves $\varepsilon$ at fixed $T$ and checks the peak $|\Delta H|$ drops
  ~4×.

### 4.4 The accept step: MH for a deterministic involution

Let $S = F \circ L_\varepsilon^{L}$ (leapfrog $L$ steps, then flip momentum).
By the properties above, $S$ is (i) volume-preserving, $|\det DS| = 1$, and
(ii) an **involution**: $S(S(z)) = z$. For a deterministic involutive
volume-preserving proposal, accepting $z \to S(z)$ with probability

$$\alpha(z) = \min\!\left(1, \frac{\pi(S(z))}{\pi(z)}\right) = \min\!\left(1, e^{-\Delta H}\right),
\qquad \Delta H = H(S(z)) - H(z),$$

satisfies detailed balance w.r.t. the joint: the flux from a set $A$ to
$S(A)$ is $\int_A \pi(z)\,\alpha(z)\,dz = \int_A \min(\pi(z), \pi(S(z)))\,dz$,
and the reverse flux from $S(A)$ pulls back through $S$ (unit Jacobian,
$S^{-1} = S$) to the identical integrand. ∎

The full HMC iteration alternates two $\pi(x,p)$-invariant kernels:

1. **Momentum refresh** $p \sim N(0, I)$ — an exact Gibbs draw from
   $\pi(p \mid x) = \pi(p)$ (Sec. 3's argument: always accepted).
2. **Trajectory + flip + MH accept** as above.

The final flip can be dropped in practice because $K(p) = K(-p)$ and step 1
discards $p$ anyway. Without step 1 the chain would be confined to a single
energy level set; the refresh is what moves the chain across energies.

**Why this beats a random walk.** Along a trajectory the position moves
$O(L\varepsilon)$ in a *consistent* direction guided by $\nabla\log\pi$,
instead of $O(\varepsilon\sqrt{L})$ diffusively; with $\Delta H = O(\varepsilon^2)$
the move is still accepted with high probability. Distant proposals + high
acceptance = small autocorrelation (measured: $\tau \approx 1.8$ vs $68$ for
RWMH on the $\rho = 0.9$ Gaussian).

**Divergences.** When $\varepsilon$ is too large for the local curvature
(stiffest direction roughly requires $\varepsilon < 2/\sqrt{\lambda_{\max}(\nabla^2 U)}$,
the stability bound of leapfrog on a quadratic), the discretization explodes,
$\Delta H \to \infty$, and the proposal is rejected. We flag
$\Delta H > 25$ as a divergence and report counts; clusters of divergences
mark regions the sampler cannot enter, i.e. *biased* exploration (funnel
experiment).

### 4.5 Step-size adaptation (dual averaging)

During warmup we tune $\log\varepsilon$ to drive the mean acceptance
probability $\alpha_t$ to a target $\delta$ (Hoffman & Gelman 2014, Alg. 5,
after Nesterov 2009):

$$\bar h_t = \Big(1 - \tfrac{1}{t + t_0}\Big)\bar h_{t-1} + \tfrac{\delta - \alpha_t}{t + t_0}, \qquad
\log\varepsilon_t = \mu - \tfrac{\sqrt{t}}{\gamma}\,\bar h_t, \qquad
\log\bar\varepsilon_t = t^{-\kappa}\log\varepsilon_t + (1 - t^{-\kappa})\log\bar\varepsilon_{t-1},$$

with $\mu = \log 10\varepsilon_0$, $\gamma = 0.05$, $t_0 = 10$,
$\kappa = 0.75$; after warmup $\varepsilon$ is frozen at $\bar\varepsilon$.
Adaptation during sampling would break invariance (the kernel would depend on
the chain's own history), which is why it must stop at the end of warmup.

### 4.6 Funnels and non-centering

Neal's funnel $v \sim N(0, \sigma_v^2)$, $x_i \mid v \sim N(0, e^v)$ defeats
any single step size: leapfrog's stability limit in the $x$-directions is
$\propto e^{v/2}$, varying by orders of magnitude over the range of $v$.
The cure is a change of variables, not a better tuner. Let $x_i = e^{v/2} z_i$.
The density transforms with the Jacobian $\big|\partial x/\partial z\big| = e^{(d-1)v/2}$:

$$p(v, z) = N(v; 0, \sigma_v^2)\prod_i N\!\big(e^{v/2} z_i;\, 0,\, e^v\big)\; e^{(d-1)v/2}
= N(v; 0, \sigma_v^2)\prod_i N(z_i; 0, 1),$$

because $N(e^{v/2}z; 0, e^v) = (2\pi e^v)^{-1/2} e^{-z^2/2} = e^{-v/2}\,N(z; 0, 1)$
and the $e^{\pm(d-1)v/2}$ factors cancel exactly. In $(v, z)$ the target is a
product of independent Gaussians — trivial for HMC. This is the same
transformation as the eight-schools non-centered parameterization
($\theta_j = \mu + \tau\eta_j$), where hierarchical posteriors develop the
identical funnel in $(\log\tau, \theta)$ whenever the data only weakly
identify the group-level scale.

### 4.7 The Rosenbrock/banana: closed-form ground truth from a curved target

The funnel varies *scale*; the Rosenbrock density varies *direction* — its mass
lies on a thin parabolic ridge whose local covariance rotates along the arc, so
again no single step size and unit mass are ideal. Take

$$p(x_1, x_2) \propto \exp\!\big[-(x_1 - a)^2 - b\,(x_2 - x_1^2)^2\big].$$

Unlike a generic banana this one has *fully closed-form* ground truth, because
the $b$-term is Gaussian in $x_2$. Read the density as a conditional times a
marginal. Integrating $x_2$ out,

$$\int e^{-b(x_2 - x_1^2)^2}\,dx_2 = \sqrt{\pi/b}\quad(\text{independent of }x_1),$$

so the $x_1$-marginal is exactly $x_1 \sim N(a, \tfrac12)$ and the conditional is
$x_2 \mid x_1 \sim N(x_1^2,\ \tfrac1{2b})$. The normalizer is
$\sqrt{\pi}\cdot\sqrt{\pi/b}$. Its moments follow from the normal identities
$\mathbb E[X^2] = \mu^2+\sigma^2$, $\operatorname{Var}[X^2] = 2\sigma^4 + 4\mu^2\sigma^2$,
$\operatorname{Cov}[X, X^2] = 2\mu\sigma^2$ applied to $X = x_1$ with
$\mu=a,\ \sigma^2=\tfrac12$:

$$\mathbb E[x_1]=a,\quad \operatorname{Var}[x_1]=\tfrac12,\qquad
\mathbb E[x_2]=a^2+\tfrac12,$$
$$\operatorname{Var}[x_2]=\underbrace{\tfrac1{2b}}_{\mathbb E[\operatorname{Var}(x_2\mid x_1)]}
+\underbrace{\tfrac12 + 2a^2}_{\operatorname{Var}(x_1^2)},\qquad
\operatorname{Cov}[x_1,x_2]=\operatorname{Cov}(x_1,x_1^2)=a.$$

This gives an exact answer key on a *curved* problem: a generative sampler
($x_1$ then $x_2\mid x_1$), exact moments, and the exact $x_1$-marginal all
follow with no approximation. The hand-derived gradient is
$\partial_{x_1}\log p = -2(x_1-a) + 4b\,x_1(x_2 - x_1^2)$,
$\partial_{x_2}\log p = -2b\,(x_2 - x_1^2)$; `experiments/validate_exact.py`
(Part C) checks HMC against these and shows the residual covariance error that a
single step size leaves on the high-curvature arms — the motivation for
mass-matrix adaptation and NUTS.

### 4.8 Diagonal mass-matrix adaptation

Sections 4.6–4.7 show unit-metric HMC losing to *geometry*. The cheapest partial
fix is the mass matrix $M$. Keep $K(p) = \tfrac12 p^\top M^{-1} p$; then Hamilton's
equations become $\dot x = M^{-1} p$, $\dot p = \nabla\log\tilde\pi(x)$, so the
leapfrog *drift* carries the metric and the *kicks* do not:

$$p_{1/2} = p_0 + \tfrac{\varepsilon}{2}\nabla\log\tilde\pi(x_0),\quad
x_1 = x_0 + \varepsilon\, M^{-1} p_{1/2},\quad
p_1 = p_{1/2} + \tfrac{\varepsilon}{2}\nabla\log\tilde\pi(x_1),$$

with momentum refreshed from $p \sim N(0, M)$ and acceptance from
$\Delta H = \Delta U + \tfrac12(p'^\top M^{-1} p' - p^\top M^{-1} p)$. A diagonal
rescaling is still a shear composition, so volume preservation and reversibility
(Sec. 4.3–4.4) — hence exactness — are untouched: $M$ changes *efficiency only*,
never the stationary distribution. In code this is the single line
`x += eps * inv_mass * p`, with `inv_mass` the diagonal of $M^{-1}$.

**Why it is a preconditioner.** Take a separable quadratic
$U = \sum_i x_i^2/(2\sigma_i^2)$, so the curvature along axis $i$ is $1/\sigma_i^2$.
Leapfrog on a harmonic oscillator of frequency $\omega_i = \sqrt{(M^{-1})_{ii}/\sigma_i^2}$
is stable only for $\varepsilon\,\omega_i < 2$, so a *single* $\varepsilon$ must
respect the stiffest $\omega_i$ while the softest direction is then integrated far
below its stability limit and drifts slowly. Choosing $(M^{-1})_{ii} = \sigma_i^2$
makes every $\omega_i = 1$: all directions share one natural frequency, one step
size fits all, and the trajectory traverses each coordinate at its own scale. This
is exactly whitening — HMC with metric $M = \Sigma^{-1}$ on $x$ equals unit-metric
HMC on $\Sigma^{-1/2}x$. The diagonal version whitens the *marginals*; it cannot
rotate, so it leaves correlations (and the funnel/banana curvature of 4.6–4.7)
uncorrected — that is what NUTS with a dense metric, or Riemannian HMC, is for.

**Estimating it.** We want $(M^{-1})_{ii} = \operatorname{Var}_\pi[x_i]$, which we
do not know a priori, so it is learned during warmup from the sample variances.
Two practical points, both mirroring Stan:

- *Memoryless expanding windows.* Warmup splits into an initial buffer (metric
  fixed at $I$ while the chain first reaches the typical set), a sequence of
  windows each ~2× the last, and a terminal buffer. Each window estimates a
  *fresh* diagonal from only its own draws — early, pre-convergence samples are
  discarded rather than averaged in — and the step-size dual averaging (Sec. 4.5)
  is **restarted** after each metric change, because a new metric is a new
  integrator whose optimal $\varepsilon$ differs. The terminal buffer re-tunes
  $\varepsilon$ to the frozen final metric.
- *Regularization.* A short window gives a noisy variance; we shrink it toward a
  unit metric, $\hat v \leftarrow \tfrac{n}{n+5}\hat v + \tfrac{5}{n+5}\cdot 10^{-3}$,
  so a degenerate window cannot emit a wild scale.

Adaptation stops at the end of warmup for the same reason the step size freezes:
a kernel that depends on the chain's own history is no longer $\pi$-invariant.

**Measured (`experiments/mass_matrix.py`).** On diagonal Gaussians
$N(0,\operatorname{diag}(1, r^2))$ the adapted metric recovers
$(M^{-1})_{22}\approx r^2$ and whitens every $r$ to the *same* isotropic problem —
a flat $\approx 30$ ESS per 1000 gradients independent of $r$ — while unit-metric
HMC swings erratically ($\approx 3$–$34$) as its single step size resonates with
the wide direction. On non-centered eight schools the diagonal metric drives the
wide $\mu$ and the $\eta_j$ to near-independence ($\tau\to 1$, up to $15\times$ the
unit-metric ESS/grad on $\eta_1$), but $\log\tau$ gains only $\sim 2.4\times$: the
funnel is curvature the diagonal cannot touch, the honest limit that motivates
Days 17–18 (NUTS).

### 4.9 The No-U-Turn Sampler

Dual averaging (4.5) removes the step-size knob; the mass matrix (4.8) removes
one scale mismatch. What is left in fixed-length HMC is the **trajectory length**
$L$. Too small and the proposal barely moves; too large and the trajectory
U-turns and comes back toward the start, so the extra gradients buy nothing (and
on a near-Gaussian target a fixed $L$ can *resonate* with the oscillation period,
which is why `hmc()` jitters $L$). NUTS (Hoffman & Gelman 2014; the multinomial
formulation of Betancourt 2017 used here) removes the knob by growing each
trajectory until it starts to fold back on itself.

**Recursive doubling.** From the current $(x, p)$, repeatedly *double* the
trajectory: draw a random time direction $\in\{-1,+1\}$ and integrate a new
sub-trajectory of the same length as everything built so far, extending the
appropriate end. After $j$ doublings the tree holds up to $2^j$ states built with
$2^j-1$ leapfrog steps. The two endpoints of every *balanced* sub-tree are exact
leapfrog images of each other, and the direction is chosen independently of the
current state, so the transition is reversible — this is what lets the accept
step be subsumed into the weighting rather than written out (contrast 4.4).

**The no-U-turn criterion.** A balanced sub-tree spanning
$(x_-,p_-)\ldots(x_+,p_+)$ is *turning* once advancing either end would stop
increasing the distance between them, i.e. the span vector no longer projects
positively onto the velocity $M^{-1}p$ at that end:

$$(x_+ - x_-)^\top M^{-1} p_- < 0 \quad\text{or}\quad (x_+ - x_-)^\top M^{-1} p_+ < 0.$$

(The metric enters through the velocity, so the criterion is the natural one in
the same geometry the leapfrog uses; identity metric $\Rightarrow$ just $p$.)
The instant *any* sub-tree turns, doubling stops — the length adapts to local
geometry, long in flat directions, short in tight ones, with no user input.

**Multinomial (canonical) selection.** Every visited state $z=(x,p)$ carries
canonical weight $\exp(-H(z))=\exp(\log\tilde\pi(x) - K(p))$. The next sample is
drawn from the whole trajectory with probability proportional to that weight.
Because the trajectory is a slice of the joint $\propto e^{-H}$ whose $x$-marginal
is $\pi$, the draw leaves $\pi$ invariant — the accept/reject of HMC is replaced
by *weighting the states the trajectory already visited*. We realise the
multinomial progressively while building: within a balanced doubling the newer
half is taken with probability equal to its share $W_{\text{new}}/(W_{\text{old}}+W_{\text{new}})$
of the canonical weight; at the top level the new half is taken with Stan's
*biased* probability $\min(1, W_{\text{new}}/W_{\text{old}})$, which pushes the
sample outward along the trajectory for faster mixing while remaining a valid
transition (Betancourt 2017, App.).

**Termination on pathological geometry.** Two valves keep the recursion finite
where the criterion never fires. A **maximum tree depth** caps work per
iteration (Stan's default 10 $\Rightarrow \le 1023$ steps). A **divergence**
check marks any leaf whose energy error $\Delta H$ exceeds a threshold (or is
non-finite) as invalid — weight zero, expansion halted — because a large $\Delta H$
means the symplectic integrator has left the level set the exact flow would
preserve (4.2), the signal that $\varepsilon$ is too large for the local
curvature. Divergences are recorded per iteration, so their *positions* are a
diagnostic: on the centered funnel they pile into the neck (§ measured below).

**Cost accounting.** Each leaf is one leapfrog step. A leapfrog step needs the
gradient at both endpoints, but the second endpoint of one leaf is the first
endpoint of the next, so caching the gradient makes NUTS cost exactly *one*
gradient per leaf — the honest denominator for the ESS-per-gradient comparison.
Step size is still dual-averaged (4.5), here driving the mean per-leaf
acceptance statistic $\overline{\alpha}$ to the target.

**Simplifications vs Stan (stated, not hidden).** This is a faithful but minimal
NUTS. It uses (i) a *diagonal* metric only — no dense or Riemannian metric, so
correlations and the funnel's varying curvature are untouched, exactly as in
4.8; (ii) the endpoint-momentum U-turn check on each balanced sub-tree, not the
finer generalized criterion Stan also applies to the leftmost/rightmost
sub-sub-trees; (iii) a single shared step size and depth cap across chains. The
consequence is honest and visible: on the **centered** funnel our NUTS still
diverges in the neck — but so does Stan's, because the neck is a property of the
*parameterization*, not the sampler. The fix is the non-centering of 4.6, after
which NUTS mixes cleanly. NUTS removes the length knob; it does not remove the
need to choose good coordinates.

**Measured (`experiments/nuts_benchmark.py`).** Per gradient, on the
non-centered funnel NUTS delivers $\approx 29$ ESS$(v)$ per 1000 gradients
against a hand-tuned fixed-$L$ HMC's $\approx 7$ and RWMH's $\approx 4$ — the
length automation *buys* efficiency rather than costing it, because a fixed
$L=20$ overshoots the U-turn on the easy directions. On non-centered eight
schools NUTS gives the most ESS$(\tau)$ per gradient of the three. On the
centered funnel it logs $\sim 13\%$ divergent iterations clustered in the neck
and under-covers $v$ ($\mathrm{sd}\,2.7$ vs the true $3.0$); non-centering drops
that to zero divergences and $\mathrm{sd}\,3.0$.

## 5. The models

### 5.1 Conjugate Bayesian linear regression

$y = X\beta + \epsilon$, $\epsilon \sim N(0, \sigma^2 I)$, prior
$\beta \sim N(0, \tau^2 I)$. The log posterior is quadratic in $\beta$;
completing the square:

$$\Sigma_n = \Big(\tfrac{X^\top X}{\sigma^2} + \tfrac{I}{\tau^2}\Big)^{-1}, \qquad
\mu_n = \Sigma_n \tfrac{X^\top y}{\sigma^2}, \qquad
\beta \mid y \sim N(\mu_n, \Sigma_n).$$

Samplers receive only $\log\tilde\pi(\beta) = -\tfrac{\lVert y - X\beta\rVert^2}{2\sigma^2} - \tfrac{\lVert\beta\rVert^2}{2\tau^2}$
and its gradient $\tfrac{X^\top(y - X\beta)}{\sigma^2} - \tfrac{\beta}{\tau^2}$;
the closed form is the answer key.

### 5.2 Eight schools: conjugate conditionals (centered)

Model: $y_j \mid \theta_j \sim N(\theta_j, \sigma_j^2)$ ($\sigma_j$ known),
$\theta_j \mid \mu, \tau^2 \sim N(\mu, \tau^2)$, $p(\mu) \propto 1$,
$\tau^2 \sim \text{InvGamma}(a, b)$.

- $\theta_j$: two Gaussian likelihood terms in $\theta_j$; precision-weighted
  combination gives
  $\theta_j \mid \cdot \sim N\!\Big(\tfrac{y_j/\sigma_j^2 + \mu/\tau^2}{1/\sigma_j^2 + 1/\tau^2},\; \tfrac{1}{1/\sigma_j^2 + 1/\tau^2}\Big)$ —
  the same "shrink toward $\mu$ by noise-to-signal" form as classical
  James–Stein estimators.
- $\mu$: flat prior, $J$ Gaussian terms
  $\Rightarrow \mu \mid \cdot \sim N(\bar\theta,\, \tau^2/J)$.
- $\tau^2$: with the InvGamma$(a,b)$ prior,
  $$p(\tau^2 \mid \cdot) \propto (\tau^2)^{-J/2} e^{-\sum_j(\theta_j - \mu)^2 / (2\tau^2)}
  \cdot (\tau^2)^{-(a+1)} e^{-b/\tau^2}
  = \text{InvGamma}\Big(a + \tfrac{J}{2},\; b + \tfrac12\textstyle\sum_j (\theta_j - \mu)^2\Big).$$

### 5.3 Eight schools: non-centered log posterior and gradient (for HMC)

Variables $z = (\mu, t, \eta)$ with $t = \log\tau$, $\theta_j = \mu + e^t \eta_j$.
Transforming the InvGamma prior to $t$: with $u = \tau^2 = e^{2t}$,
$|du/dt| = 2e^{2t}$, so
$\log p(t) = -2at - b e^{-2t} + \text{const}$ (the $+2t$ from the Jacobian
cancels part of the $-(a+1)\cdot 2t$). With $r_j = y_j - \mu - e^t\eta_j$:

$$\mathcal{L}(z) = -\sum_j \frac{r_j^2}{2\sigma_j^2} - \sum_j \frac{\eta_j^2}{2} - 2at - be^{-2t},$$

$$\frac{\partial\mathcal{L}}{\partial\mu} = \sum_j \frac{r_j}{\sigma_j^2}, \qquad
\frac{\partial\mathcal{L}}{\partial t} = e^t \sum_j \frac{r_j}{\sigma_j^2}\eta_j - 2a + 2be^{-2t}, \qquad
\frac{\partial\mathcal{L}}{\partial \eta_j} = \frac{r_j}{\sigma_j^2} e^t - \eta_j.$$

(Each term checked against central finite differences in
`tests/test_models.py`; the $t$-derivative is where a dropped Jacobian shows
up as a systematic bias in $\tau$, which the Gibbs/HMC agreement test would
catch.)

## 6. Diagnostics

### 6.1 Autocorrelation and effective sample size

For a stationary chain with variance $\sigma^2$ and autocorrelations $\rho_k$:

$$\operatorname{Var}(\bar x_N) = \frac{\sigma^2}{N}\Big[1 + 2\sum_{k=1}^{N-1}\big(1 - \tfrac{k}{N}\big)\rho_k\Big]
\;\xrightarrow{N\to\infty}\; \frac{\sigma^2}{N}\,\tau, \qquad
\tau = 1 + 2\sum_{k=1}^{\infty}\rho_k.$$

$\tau$ is the *integrated autocorrelation time*: $N$ correlated draws carry
the information of $N/\tau$ independent ones, so
$\text{ESS} = mN/\tau$ for $m$ chains.

Estimating $\tau$ naively by summing all empirical $\rho_k$ fails (the noise
in the tail has variance that doesn't vanish; the sum random-walks). We use
**Geyer's (1992) initial monotone positive sequence**: for a reversible
chain the pair sums $\Gamma_m = \rho_{2m} + \rho_{2m+1}$ are provably
positive and non-increasing, so we sum $\hat\Gamma_m$ only while positive,
after enforcing monotonicity — an adaptive truncation with no tuning
parameter. The estimator is itself validated against the AR(1) closed form
$\tau = (1+\rho)/(1-\rho)$ in `tests/test_diagnostics.py`.

Autocovariances are computed by FFT (Wiener–Khinchin: the autocovariance is
the inverse transform of the periodogram), zero-padded to $\ge 2n$ to undo
circular wraparound — $O(n\log n)$.

### 6.2 Split-$\hat R$

Run $m$ chains from overdispersed starts, split each in half ($2m$ chains of
length $n$), and compare two variance estimates: the within-chain mean $W$
(too *small* before convergence — no chain has covered $\pi$ yet) and

$$\widehat{\text{var}}^{+} = \frac{n-1}{n}W + \frac{B}{n}, \qquad
B = n \cdot \operatorname{Var}(\text{chain means}),$$

which overestimates under overdispersed initialization.
$\hat R = \sqrt{\widehat{\text{var}}^{+}/W} \to 1$ from above as the chains
forget their starts; we flag $\hat R > 1.01$. Splitting catches the failure
mode where every chain drifts identically (between-chain agreement but
within-chain nonstationarity). $\hat R \approx 1$ is necessary, never
sufficient — the funnel RWMH run reaches $\hat R = 1.04$ while missing the
neck entirely, which is why the exact marginal check matters.

### 6.3 Thinning wastes information

Folklore says: the draws are autocorrelated, so keep every $k$-th and discard
the rest. For *accuracy* this is always a loss. Correlated draws are worth
less than independent ones — that is exactly what $\tau$ measures — but they
are not worth *nothing*, and thinning throws away the value they had.

For AR(1) the cost is exact. With $\rho_j = \rho^j$, $\rho \in [0,1)$, Sec. 6.1
gives $\tau = (1+\rho)/(1-\rho)$ and

$$\operatorname{Var}(\bar x_N) = \frac{\sigma^2}{N}\cdot\frac{1+\rho}{1-\rho}.$$

Now thin by $k$. A Markov chain observed every $k$ steps is still a Markov
chain, and the kept draws are themselves AR(1) with lag-1 correlation
$\rho^k$ — of which there are only $N/k$. So

$$\operatorname{Var}(\bar x_{N/k}) = \frac{\sigma^2 k}{N}\cdot\frac{1+\rho^k}{1-\rho^k},
\qquad
R(\rho,k) \;\equiv\; \frac{\operatorname{Var}(\text{thinned mean})}{\operatorname{Var}(\text{full mean})}
\;=\; k\,\frac{(1+\rho^k)(1-\rho)}{(1-\rho^k)(1+\rho)}.$$

**Claim: $R \ge 1$ for every $k \ge 1$, with equality only at $k = 1$.**
Write $\rho = e^{-\lambda}$, $\lambda > 0$. Then
$\frac{1+\rho^k}{1-\rho^k} = \coth(\lambda k/2)$, so

$$R(k) = \frac{k \coth(\lambda k / 2)}{\coth(\lambda / 2)}.$$

It suffices that $u \mapsto u\coth(\lambda u/2)$ is increasing on $u>0$.
Substituting $v = \lambda u/2$, this is $v \mapsto v \coth v$ up to a positive
constant, and

$$\frac{d}{dv}\big(v \coth v\big) = \frac{\sinh v\cosh v - v}{\sinh^2 v}
= \frac{\tfrac12\sinh(2v) - v}{\sinh^2 v} > 0,$$

since $\sinh(2v) > 2v$ for all $v > 0$. Hence $R(k) \ge R(1) = 1$. $\square$

Two limits organize the picture:

| regime | $R$ | reading |
|---|---|---|
| $\rho = 0$ (independent draws) | $R = k$ exactly | pure waste: discard $k-1$ of every $k$ good samples, inflate the variance by exactly $k$ |
| $\rho \to 1$ (very sticky chain) | $R \to 1$ | nearly free — the discarded draws *were* near-duplicates. Still not an improvement. |

So the cost of thinning is **largest exactly where it is least often
proposed** (a fast-mixing chain) and smallest where the chain is so sticky the
draws really were redundant. There is no regime in which it improves accuracy.

Measured (`experiments/thinning.py`, `diagnostics.thinning_variance_ratio`).
Part 1 brute-forces $\operatorname{Var}(\bar x)$ over 4000 independent AR(1)
chains at $\rho = 0.9$ ($\tau = 19$):

| $k$ | draws kept | ESS | $R$ predicted | $R$ measured |
|---|---|---|---|---|
| 1 | 2000 | 440,142 | 1.000 | 1.000 |
| 5 | 400 | 431,103 | 1.022 | 1.025 |
| 10 | 200 | 406,599 | 1.090 | 1.101 |
| 20 | 100 | 328,111 | 1.344 | 1.386 |

Part 2 asks whether the formula survives contact with a *real* sampler. It
should not obviously: a Metropolis chain repeats its state on rejection, so its
autocorrelation is not a clean geometric $\rho^j$ and it is not AR(1). Running
RWMH on the correlated Gaussian (accept 0.485, $\tau = 47.2$, measured lag-1
$\hat\rho = 0.944$) and feeding that $\hat\rho$ to the AR(1) formula:

| $k$ | ESS | $R$ measured | $R$ predicted |
|---|---|---|---|
| 1 | 16,954 | 1.000 | 1.000 |
| 5 | 16,841 | 1.007 | 1.007 |
| 10 | 16,541 | 1.025 | 1.027 |
| 20 | 15,587 | 1.088 | 1.108 |

— within a couple of percent throughout. Note the honest magnitude: at
$\rho \approx 0.94$, thinning by 5 costs under 1%. The point is not that
thinning is catastrophic on a sticky chain; it is that the cost is *never
negative*, and it becomes large precisely when the chain mixes well.

**When thinning is legitimate**: when the binding constraint is cost rather
than accuracy — RAM or disk for a long high-dimensional run, or an expensive
per-draw post-processing step (each retained draw seeding a downstream
simulation). Then $R$ is the exchange rate, and you are knowingly buying a
cheaper pipeline with a quantified amount of precision. What is not defensible
is thinning in the belief that it makes the answer better.

### 6.4 Rank-normalized split-$\hat R$

The split-$\hat R$ of Sec. 6.2 is a ratio of variances, and a variance is only a
meaningful summary when the target *has* one. Consider a heavy-tailed posterior —
a Cauchy is the sharp case, with no finite mean or variance at all. The
within-chain estimate

$$W = \frac1{2m}\sum_{\text{halves}} \frac1{n-1}\sum_t (x_{ct}-\bar x_c)^2$$

is then dominated by whichever half happened to catch the largest excursion; it
is enormous and has huge sampling noise. A genuine between-chain location
disagreement $B$ that would set off the alarm on a light-tailed target is, on the
Cauchy, tiny next to that noise, so $\widehat{\text{var}}^+ / W \to 1$ and
$\hat R$ reads a falsely reassuring $\approx 1.00$ on chains that have not mixed.
This is not a rare corner: heavy tails are exactly where a sampler struggles, so
it is precisely when the chains have *not* converged that the variance-based
statistic is least able to say so.

**The fix (Vehtari et al. 2021): work with ranks, which are finite regardless of
the tails.** Pool all $mn$ draws, replace each by its rank $r$ (ties share their
average rank — a Metropolis chain repeats its state on every rejection, so ties
are common and must not bias the transform), and map ranks to normal scores by
the **Blom (1958) rankit transform**

$$z = \Phi^{-1}\!\Big(\frac{r - 3/8}{mn - 1/4}\Big).$$

The offsets $3/8,\,1/4$ make $\mathbb E[z_{(i)}]$ match the expected order
statistics of a standard normal to high accuracy, and — the load-bearing
property — the transform is invariant to any monotone reparameterization of the
target. The scores $z$ are approximately $N(0,1)$ for draws from *any* continuous
distribution, Cauchy included, so they have the finite moments that ordinary
split-$\hat R$ needs. **Bulk-$\hat R$** is split-$\hat R$ applied to $z$. Because
rank-normalization only relabels values by their order, a between-chain *location*
shift survives it intact — the shifted chain's draws still hold the larger ranks —
so bulk-$\hat R$ flags exactly the location non-convergence the raw statistic
missed, and does so without ever forming a variance of the raw draws.

**Folding, for scale.** A location statistic — classic $\hat R$ *and* bulk-$\hat R$
alike — is blind to chains that agree on the centre but differ in spread: their
rank distributions are symmetric about the pooled median either way. Fold the
draws about the pooled median,

$$\zeta_{ct} = |x_{ct} - \operatorname{median}(x)|,$$

and a scale disagreement becomes a *location* disagreement of the absolute
deviations (the wider chain has systematically larger $\zeta$). Rank-normalize
$\zeta$ and take split-$\hat R$ again: this **folded-$\hat R$** catches the scale
failure. The reported statistic is the worst case,

$$\hat R_{\text{rank}} = \max(\hat R_{\text{bulk}},\, \hat R_{\text{fold}}),$$

so convergence must be declared on both the centre and the spread. The
`experiments/rank_rhat.py` cases pin all three behaviours: on mixed $N(0,1)$
chains both statistics sit at $1.00$ (the transform invents nothing); on
Cauchy chains shifted by three IQRs the classic $\hat R = 1.00$ while
$\hat R_{\text{bulk}} = 1.27$; on Cauchy chains with equal medians but scales
$1$ and $6$ the location terms are both $1.00$ and only $\hat R_{\text{fold}}
= 1.18$ fires. The one implementation nicety is that $\Phi^{-1}$ has no
closed form: we use Acklam's rational approximation refined by a single Halley
step against the exact CDF (error $< 10^{-8}$), keeping the package numpy-only.

## References

- Metropolis, Rosenbluth, Rosenbluth, Teller & Teller (1953), *J. Chem. Phys.* 21.
- Hastings (1970), *Biometrika* 57.
- Geman & Geman (1984), *IEEE TPAMI* 6 (Gibbs sampling).
- Duane, Kennedy, Pendleton & Roweth (1987), *Phys. Lett. B* 195 (hybrid Monte Carlo).
- Neal (2011), "MCMC using Hamiltonian dynamics", *Handbook of MCMC*. The canonical HMC reference.
- Neal (2003), "Slice sampling", *Ann. Statist.* 31 (the funnel, Sec. 8).
- Hoffman & Gelman (2014), "The No-U-Turn Sampler", *JMLR* 15 (dual averaging, Alg. 5).
- Geyer (1992), "Practical Markov chain Monte Carlo", *Statist. Sci.* 7 (initial sequence estimators).
- Gelman & Rubin (1992), *Statist. Sci.* 7 (original $\hat R$).
- Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021), "Rank-normalization, folding, and localization: an improved $\hat R$", *Bayesian Anal.* 16 (Sec. 6.4 rank-normalized/folded $\hat R$, tail-ESS).
- Blom (1958), *Statistical Estimates and Transformed Beta-Variables* (the $3/8$ rankit offset).
- Roberts, Gelman & Gilks (1997), *Ann. Appl. Probab.* 7 (0.234 optimal scaling).
- Rubin (1981), "Estimation in parallel randomized experiments", *J. Educ. Statist.* 6 (eight schools data).
- Betancourt (2017), "A conceptual introduction to HMC", arXiv:1701.02434 (typical sets, divergences).
