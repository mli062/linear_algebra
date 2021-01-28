import numpy as np

def gram_schmidt(A):
    """Orthogonaliser un ensemble de vecteurs stockés sous forme de colonnes dans la matrice A"""
    res = A.copy()  # Copie de la matrice A
    for i in range(res.shape[1]):  # res.shape[1] renvoie le nombre de vecteur colonne
        for j in range(i):
            # Processus de Gram-Schmidt
            res[:, i] = res[:, i] - np.dot(res[:, j], res[:, i]) * res[:, j]
        # Normalisation
        res[:, i] = res[:, i] / np.linalg.norm(res[:, i])
    return res

A = np.array([
    [1, 2, 2],
    [0, 1, 2],
    [2, 0, 1]
], dtype=float)

print("Matrice A:\n", A)
print("Dimensionde A:", A.shape)

res = gram_schmidt(A)
print("Matrice Orthonormale:", res)
print("Dimension de res:", res.shape)

# Verifier 2 à 2 que les vecteurs sont bien orthogonaux et de longueur 1
print("Longueur de u1:", np.linalg.norm(res[:, 0]))
print("Longueur de u2:", np.linalg.norm(res[:, 1]))
print("Longueur de u3:", np.linalg.norm(res[:, 2]))

print("Produit scalaire entre u1 et u2:", np.dot(res[:, 0], res[:, 1]))
print("Produit scalaire entre u1 et u3:", np.dot(res[:, 0], res[:, 2]))
print("Produit scalaire entre u2 et u3:", np.dot(res[:, 1], res[:, 2]))
