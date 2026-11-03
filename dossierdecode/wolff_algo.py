import numpy as np
rng = np.random.default_rng(42)


def neighbor(spin : tuple, L):
    i,j=spin[0],spin[1]
    return [((i+1)%L, j), ((i-1)%L, j), (i, (j+1)%L), (i, (j-1)%L)]

def wolff(Spins : np.ndarray,J : float, beta : float ):
    L=len(Spins)
    p_add=1-np.exp(-2*beta*J)
    cluster=[]
    i, j = rng.integers(0, L, size=2)
    s0=Spins[i,j]
    Spins[i, j] = -s0
    pile=[(i,j)]

    while len(pile)!=0:
        spin=pile.pop()
        cluster.append(spin)
        neighbors=neighbor(spin,L)
        for nb in neighbors:
            if s0==Spins[nb]:
                r = rng.random()
                if r<=p_add:
                    pile.append(nb)
                    Spins[nb] = -s0
    
    return cluster