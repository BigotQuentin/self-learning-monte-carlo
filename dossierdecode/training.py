from learningH import learningH, meanDifferenceH
from dossierdecode.Metropolis_Hasting_etape1 import ising_JK, metropolis_hastings
import numpy as np


def correlations(configs: np.ndarray, n_max: int = 2) -> np.ndarray:
    """C_n^a = sum over the n-th nearest-neighbor pairs <ij>_n of S_i S_j, for each configuration a.

    configs: (n_configs, L, L) array of +-1 (or a single (L, L) configuration), periodic boundaries.
    Returns C of shape (n_configs, n_max): column n-1 holds C_n (n=1: NN, 2: next-NN diagonal, 3: distance 2, ...).
    Each pair is counted once. Needs L > 2*sqrt(d_n^2) so that no pair is counted twice by periodicity.
    """
    s = np.asarray(configs)
    single = s.ndim == 2
    s = s[None] if single else s
    s = s.astype(np.int64)
    R = n_max
    # all displacements (dx, dy), grouped by squared distance -> shells 1, 2, 3, ... of neighbors
    shells = sorted({dx*dx + dy*dy for dx in range(-R, R+1) for dy in range(-R, R+1)} - {0})[:n_max]
    C = np.empty((len(s), n_max))
    for n, d2 in enumerate(shells):
        # one of the two opposite displacements, so that every pair appears once
        disp = [(dx, dy) for dx in range(-R, R+1) for dy in range(-R, R+1)
                if dx*dx + dy*dy == d2 and (dx > 0 or (dx == 0 and dy > 0))]
        C[:, n] = sum((s * np.roll(s, (-dx, -dy), axis=(1, 2))).sum(axis=(1, 2)) for dx, dy in disp)
    return C[0] if single else C


energy, delta_energy=ising_JK(1,0.2)
rng=rng = np.random.default_rng(42)
configs, energies=metropolis_hastings(energy, 1000,8,4,rng, delta_energy,1000)
Cn=correlations(configs, n_max=4)
print(learningH(Cn,energies))
print(meanDifferenceH(Cn,energies))