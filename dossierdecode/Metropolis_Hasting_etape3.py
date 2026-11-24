import math
import numpy as np
from dossierdecode.wolff_algo import wolff

def ising_JK(E0: int = 0.0, J: float = 1.0, K: float = 0.2):
    """H = -J sum_<ij> s_i s_j - K sum_plaquettes s_i s_j s_k s_l (periodic), as in the notebook.
    Returns (energy, delta_energy): total energy, and energy change when flipping s[i, j]."""
    def energy(s):
        r0, r1 = np.roll(s, -1, axis=0), np.roll(s, -1, axis=1)
        return E0 + float(-J * np.sum(s * (r0 + r1)) - K * np.sum(s * r0 * r1 * np.roll(r0, -1, axis=1)))
    
    def delta_energy_cluster(Spins, cluster):
        """Energy change when flipping all spins in `mask` (bool (L, L)), computed before the flip.
        A term of H changes iff it contains an odd number of flipped spins; it then changes sign:
        dE = 2 * sum(coupling * product of its spins), over those terms only."""
        r0, r1 = np.roll(Spins, -1, axis=0), np.roll(Spins, -1, axis=1)
        m0, m1 = np.roll(cluster, -1, axis=0), np.roll(cluster, -1, axis=1)
        dE_J = 2 * J * ((Spins * r0)[cluster ^ m0].sum() + (Spins * r1)[cluster ^ m1].sum())          # bonds on the cluster boundary
        plaquette = Spins * r0 * r1 * np.roll(r0, -1, axis=1)
        dE_K = 2 * K * plaquette[cluster ^ m0 ^ m1 ^ np.roll(m0, -1, axis=1)].sum()        # plaquettes cut an odd number of times
        return E0 + float(dE_J + dE_K)
    return energy, delta_energy_cluster

def metropolis_hastings_wolff(energy_H, delta_energy_H, delta_energy_Heff, J_eff,
                              n_configs, L, T, rng, n_warmup=1000, length_cycle=1):
    """SLMC simulation: Wolff clusters built with H_eff = -J_eff * sum_<ij> s_i s_j, accepted with Eq. 3.
    delta_energy_H(s, mask), delta_energy_Heff(s, mask) -> energy change if the cluster `mask` is flipped.
    n_warmup, length_cycle count cluster flips."""
    beta = 1.0 / T
    Spins = rng.choice([-1, 1], size=(L, L))

    def to_mask(cluster):
        c = np.asarray(cluster)
        if c.dtype == bool:
            return c
        mask = np.zeros((L, L), dtype=bool); mask[c[:, 0], c[:, 1]] = True
        return mask

    def run(n_moves):
        for _ in range(n_moves):
            mask = to_mask(wolff(Spins, J_eff, beta))
            dE = delta_energy_H(Spins, mask) - delta_energy_Heff(Spins, mask)
            if dE <= 0 or rng.random() < math.exp(-beta * dE):
                Spins[mask] *= -1                      # in place, no copy needed

    run(n_warmup)
    configs = np.empty((n_configs, L, L), dtype=np.int8)
    energies = np.empty(n_configs)
    for n in range(n_configs):
        run(length_cycle)
        configs[n], energies[n] = Spins, energy_H(Spins)
    return configs, energies