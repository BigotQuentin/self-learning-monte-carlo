import dossierdecode.learningH as trainingH
import dossierdecode.Metropolis_Hasting_etape1 as etape1
import dossierdecode.Metropolis_Hasting_etape3 as etape3
import numpy as np


energy, delta_energy = etape1.ising_JK(1,0.2)
rng = np.random.default_rng(42)
configs, energies = etape1.metropolis_hastings(energy, 1000,8,4,rng, delta_energy,1000)

Cn = trainingH.correlations(configs, n_max=2)
E0, J1, J2 = trainingH.learningH(Cn, energies)

print("\n" + "=" * 55)
print("       HAMILTONIEN EFFECTIF DU MODÈLE D'ISING")
print("=" * 55)

print(f"\nÉnergie fondamentale E0 : {E0:.8f}")

print("\nConstantes de couplage :")
print(f"  J1 (1ers voisins) : {J1:+.8f}")
print(f"  J2 (2èmes voisins) : {J2:+.8f}")

print("\nHamiltonien effectif :")
print("-" * 55)
print(f"H_eff = {E0:+.6f}")
print(f"        {-J1:+.6f} Σ_<ij> s_i s_j")
print(f"        {-J2:+.6f} Σ_<<ij>> s_i s_j")
print("=" * 55)

print(trainingH.meanDifferenceH(Cn, energies))