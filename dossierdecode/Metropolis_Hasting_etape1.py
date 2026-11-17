import math
import numpy as np


def ising_JK(J: float = 1.0, K: float = 0.2):
    """H = -J sum_<ij> s_i s_j - K sum_plaquettes s_i s_j s_k s_l (periodic), as in the notebook.
    Returns (energy, delta_energy): total energy, and energy change when flipping s[i, j]."""
    def energy(s):
        r0, r1 = np.roll(s, -1, axis=0), np.roll(s, -1, axis=1)
        return float(-J * np.sum(s * (r0 + r1)) - K * np.sum(s * r0 * r1 * np.roll(r0, -1, axis=1)))

    def delta_energy(s, i, j):
        L = s.shape[0]; ip, jp = (i + 1) % L, (j + 1) % L
        dJ = s[i-1, j] + s[ip, j] + s[i, j-1] + s[i, jp]
        dK = (s[i-1, j-1]*s[i-1, j]*s[i, j-1] + s[i-1, j]*s[i-1, jp]*s[i, jp]
              + s[i, j-1]*s[ip, j-1]*s[ip, j] + s[i, jp]*s[ip, j]*s[ip, jp])
        return float(2 * s[i, j] * (J * dJ + K * dK))
    return energy, delta_energy


def metropolis_hastings(energy, n_configs: int, L: int, T: float, rng: np.random.Generator,
                        delta_energy=None, n_warmup: int = 1000, length_cycle: int | None = None):
    """Step (i) of SLMC: local-update Metropolis-Hastings simulation with the original Hamiltonian H.

    energy(s)            -> total energy of the L x L spin array s (entries +-1), i.e. H.
    delta_energy(s,i,j)  -> optional, energy change if s[i,j] is flipped (fast, O(1)). Without it,
                            the change comes from two full evaluations of `energy` (slow, but works for any H).
    n_warmup             -> sweeps (L*L moves) discarded to reach equilibrium.
    length_cycle         -> single-spin moves between two stored configurations (default: one sweep, L*L).
    Returns configs (n_configs, L, L) of +-1 (int8) and their energies (n_configs,).
    """
    beta = 1.0 / T
    length_cycle = L * L if length_cycle is None else length_cycle
    s = rng.choice([-1, 1], size=(L, L))
    E = energy(s)

    def run(n_moves):
        nonlocal E
        sites = rng.integers(L, size=(n_moves, 2)).tolist()   # all random numbers drawn at once
        uniforms = rng.random(n_moves).tolist()
        for (i, j), u in zip(sites, uniforms):
            if delta_energy is not None:
                dE = delta_energy(s, i, j)
            else:                                              # generic H: flip, measure, flip back
                s[i, j] *= -1
                dE = energy(s) - E
                s[i, j] *= -1
            if dE <= 0 or u < math.exp(-beta * dE):            # accept with min(1, e^{-beta dE})
                s[i, j] *= -1
                E += dE

    run(n_warmup * L * L)
    configs = np.empty((n_configs, L, L), dtype=np.int8)
    energies = np.empty(n_configs)
    for n in range(n_configs):
        run(length_cycle)
        configs[n], energies[n] = s, E
    assert abs(E - energy(s)) < 1e-6 * max(1, abs(E)), "the running energy drifted"
    return configs, energies