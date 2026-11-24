import numpy as np

def learningH(CaN, Ea) -> "list[float]":
    """
    learning Heff based on the CaN and Ea values with least square method

    Arguments :
    ---
        - Ea -> np.array(n, 1) : Matrix of training energies for a state n
        - CaN -> np.array(n, m) : Matrix of sum of m^{th} closest neighbors for a state n
    Return :
    ---
        np.array(m) : [E0, J1, ... , Jm]
    """

    ones = np.ones((CaN.shape[0], 1))
    A = np.hstack((ones, -CaN))
    # Perform least squares fit
    H = np.linalg.lstsq(A, Ea, rcond=None)[0]
    return H.flatten().tolist()


def meanDifferenceH(CaN, Ea) -> float:
    """
    Calculate the mean difference between the predicted Ea and the actual Ea
    Expecting Ea to be a matrix of shape (n, 1) and CaN to be a matrix of shape (n, m)
    """
    H = learningH(CaN, Ea)
    ones = np.ones((CaN.shape[0], 1))
    A = np.hstack((ones, -CaN))
    Ea_predicted = A @ H
    difference = np.sum(np.abs(Ea - Ea_predicted))
    return difference/CaN.shape[0]


def correlations(configs: np.ndarray, n_max: int = 2) -> np.ndarray:
    """
    C_n^a = sum over the n-th nearest-neighbor pairs <ij>_n of S_i S_j, for each configuration a.

    configs: (n_configs, L, L) array of +-1 (or a single (L, L) configuration), periodic boundaries.
    Returns C of shape (n_configs, n_max): column n-1 holds C_n (n=1: NN, 2: next-NN diagonal, 3: distance 2, ...).
    Each pair is counted once. Needs L > 2*sqrt(d_n^2) so that no pair is counted twice by periodicity.
    """
    s = np.asarray(configs)
    R = n_max
    # all displacements (dx, dy), grouped by squared distance -> shells 1, 2, 3, ... of neighbors
    shells = sorted({dx*dx + dy*dy for dx in range(-R, R+1) for dy in range(-R, R+1)} - {0})[:n_max]
    C = np.empty((len(s), n_max))
    
    for n, d2 in enumerate(shells):
        # one of the two opposite displacements, so that every pair appears once
        disp = [(dx, dy) for dx in range(-R, R+1) for dy in range(-R, R+1)
                if dx*dx + dy*dy == d2 and (dx > 0 or (dx == 0 and dy > 0))]
        C[:, n] = sum((s * np.roll(s, (-dx, -dy), axis=(1, 2))).sum(axis=(1, 2)) for dx, dy in disp)
    
    return C