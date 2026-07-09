"""MCMC samplers from scratch: Metropolis-Hastings, Gibbs, MALA, and Hamiltonian Monte Carlo.

Everything is built on NumPy only. Samplers operate on batched log-densities
(``logpdf(x)`` with ``x`` of shape ``(n_chains, dim)``) so that multiple chains
run in lockstep without Python-level per-chain loops.
"""
