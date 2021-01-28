import numpy as np

# Algorithme de Gram-Schmidt vidéo vecteur
v1 = np.array([1, 0, 2])
v2 = np.array([2, 1, 0])
v3 = np.array([2, 2, 1])

u1 = v1 / np.linalg.norm(v1)

u2 = v2 - np.dot(u1, v2) * u1
u2 /= np.linalg.norm(u2)

u3 = v3 - np.dot(u1, v3) * u1 - np.dot(u2, v3) * u2
u3 /= np.linalg.norm(u3)

# Verifier 2 à 2 que les vecteurs sont bien orthogonaux et de longueur 1
print("Longueur de u1:", np.linalg.norm(u1))
print("Longueur de u2:", np.linalg.norm(u2))
print("Longueur de u3:", np.linalg.norm(u3))

print("Produit scalaire entre u1 et u2:", np.dot(u1, u2))
print("Produit scalaire entre u1 et u3:", np.dot(u1, u3))
print("Produit scalaire entre u2 et u3:", np.dot(u2, u3))
