import numpy as np

# Création d'une matrice
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print('Matrice A:\n', A)
print('Dimension de la Matrice A:', A.shape)

# Accès aux elements de la matrice A
print('Premiere de la matrice A:', A[0])
print('Premiere element de la premiere ligne de A:', A[0][0])

print('Premiere colonne de la matrice A:', A[:, 0])
print('Seconde colonne de la matrice A:', A[:, 1])
print('Premiere element de la seconde colonne de A:', A[:, 1][0])

# Initialiser une matrice de 0
print('Matrice de 0 de dimension 3x3:\n', np.zeros((3, 3)))

# Initialiser une matrice de 1
print('Matrice de 1 dimension 3x3:\n', np.ones((3,3)))

# Initialiser une matrice de nombre aléatoire
print('Matrice de nombre aléatoire de dimension 3x3:\n', np.random.random((3,3)))

# Initiliaser une matrice avec un nombre précis
print('Matrice avec un nombre précis comme valeur de dimension 3x3:\n', np.full((3,3), 2))

# Addition de matrices

A = np.array([
    [1, 1, 1],
    [2, 2, 2]
])

B = np.array([
    [3, 3, 3],
    [4, 4, 4]
])

print('Addition de deux matrices:', A+B)
print('Addition de deux matrices avec numpy:\n', np.add(A, B))

# Multiplication d'une matrice par un scalaire
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

s = 2

print("Multiplication d'une matrice par un scalaire:\n", s * A)

# Transposée d'une matrice
A = np.array([
    [1,2,3],
    [4,5,6]
])

print('Transposée de A:\n', A.T)
print('Transposée de A avec numpy:\n', np.transpose(A))

# Produit Matriciel

A = np.array([
    [1,2,3],
    [4,5,6]
])

B = np.array([
    [1, 2],
    [7, 3],
])

print('Dimension de A:', A.shape)
print('Dimension de B:', B.shape)

if A.shape[0] == B.shape[1] and A.shape[1] == B.shape[0]:
    print('Produit Matriciel entre A et B:\n', np.dot(A, B))
else:
    print("Le Produit Matriciel n'est pas possible")

# Produit entre une Matrice et un Vecteur
v = np.array([1, 2, 3])

print('Produit entre Matrice et Vecteur:', np.dot(A, v))

# Initialisation Matrice Identité
I = np.eye(2)

print('Dimension de I:', I.shape)
print('Matrice identité de dimension 2x2:\n', I)

A = np.array([
    [1, 2],
    [3, 4]
])

print('Produit Matriciel de A et I:\n', np.dot(A, I))

# Verifier qu'une matrice est orthogonale
A = np.array([
    [-1, 0],
    [0, 1]
])

I = np.eye(2)

print('Produit Matriciel entre A et sa transposée:\n', np.dot(A, A.T))
print('Vérifier que A est orthogonale:', (np.dot(A, A.T) == I).all())

# Calcule du déterminant d'une matrice
A = np.array([
    [1, 2],
    [3, 4]
])

print('Déterminant de A:', np.linalg.det(A))

# Calcule de l'inverse d'une matrice
A = np.array([
    [1, 2],
    [3, 4]
])

print('Inverse de A:\n', np.linalg.inv(A))
print('Inverse de A en utilisant la méthode solve:\n', np.linalg.solve(A, np.eye(2)))
print('Produit Matriciel entre A et son inverse doit être égale la Matrice Identité:', np.dot(A, np.linalg.inv(A)))

# Valeurs propres et Vecteur propres
A = np.array([
    [3, 1],
    [1, 3]
])

valeurs_propres, vecteurs_propres = np.linalg.eig(A)
print('Valeurs propres de la matrice A:', valeurs_propres)
print('Vecteur propres de la matrice A:\n', vecteurs_propres)
print('Vecteur propres de A non-normalisés:\n', vecteurs_propres * np.sqrt(2))
