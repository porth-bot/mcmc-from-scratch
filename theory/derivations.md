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
$K = \tfrac12 p^\top M^{-1} p$; we use $M = I$.)

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

## References

- Metropolis, Rosenbluth, Rosenbluth, Teller & Teller (1953), *J. Chem. Phys.* 21.
- Hastings (1970), *Biometrika* 57.
- Geman & Geman (1984), *IEEE TPAMI* 6 (Gibbs sampling).
- Duane, Kennedy, Pendleton & Roweth (1987), *Phys. Lett. B* 195 (hybrid Monte Carlo).
- Neal (2011), "MCMC using Hamiltonian dynamics", *Handbook of MCMC*. The canonical HMC reference.
- Neal (2003), "Slice sampling", *Ann. Statist.* 31 (the funnel, Sec. 8).
- Hoffman & Gelman (2014), "The No-U-Turn Sampler", *JMLR* 15 (dual averaging, Alg. 5).
- Geyer (1992), "Practical Markov chain Monte Carlo", *Statist. Sci.* 7 (initial sequence estimators).
- Gelman & Rubin (1992), *Statist. Sci.* 7; split/rank-normalized refinements in Vehtari et al. (2021), *Bayesian Anal.* 16.
- Roberts, Gelman & Gilks (1997), *Ann. Appl. Probab.* 7 (0.234 optimal scaling).
- Rubin (1981), "Estimation in parallel randomized experiments", *J. Educ. Statist.* 6 (eight schools data).
- Betancourt (2017), "A conceptual introduction to HMC", arXiv:1701.02434 (typical sets, divergences).
