import math
# Le mot clé as permet d'associer la librairie importer
# à un nom plus court à utiliser dans le code
import numpy as np

# Initialiser un vecteur v
v = [1, 2, 3]

print('Voici le vecteur:', v)
print('Première Composante du vecteur v:', v[0])
print('Seconde Composante du vecteur v:', v[1])
print('Troisieme Composante du vecteur v:', v[2])

u = np.array([4, 5, 6])
v = np.array([1, 2, 3])

def add_vectors(u, v):
    assert len(u) == len(v), 'Les vecteurs doivent etre de meme dimension'
    res = []
    for i in range(len(u)):
        res.append(u[i]+v[i])
    return res

res = add_vectors(u, v)
print('Addition u et v:', res)
print('Addition u et v:', u + v)

# Multiplication d'un scalaire avec un vecteur
s = 2
v = np.array([1, 2, 3])

print('Multiplication par un scalaire:', s * v)

# Produit scalaire de deux vecteurs
u = np.array([3, 2, 1, 8])
v = np.array([6, 4, 7, 1])

def dot_product(u, v):
    assert u.shape == v.shape, 'Les vecteurs doivent etre de meme dimension'
    res = 0
    for i in range(u.shape[0]):
        res += u[i] * v[i]
    return res

res = dot_product(u, v)
print('Produit scalaire entre u et v:', res)
print('Produit scalaire entre u et v avec numpy:', np.dot(u, v))

# Multiplication de deux vecteurs
u = np.array([3, 2, 1, 8])
v = np.array([6, 4, 7, 1])

print('Multiplication de u par v:', np.multiply(u, v))

# Verifier que deux vecteurs soit bien orthogonaux
def is_orthogonal(u, v):
    return math.isclose(np.dot(u, v), 0, abs_tol=1e-10)

u = np.array([1, 0])
v = np.array([0, 1])

res = is_orthogonal(u, v)
print('u et v sont-ils orthogonaux?', res)

# Longueur d'un vecteur

def vector_length_loop(v):
    res = 0
    for i in range(len(v)):
        res += v[i]**2
    return np.sqrt(res)

def vector_length(v):
    return np.sqrt(np.dot(v, v))

v = np.array([1, 2, 3])
# Avec une boucle
res = vector_length_loop(v)
print('Longueur de v:', res)
# Avec numpy
print('Longueur de v avec numpy:', vector_length(v))

# Vecteur unitaire
def normalized_vector(v):
    v_len = vector_length(v)
    return v / v_len

v = np.array([1, 2, 3])
res = normalized_vector(v)
print('Vecteur unitaire de v:', normalized_vector(v))

def is_normalized(u):
    return math.isclose(vector_length(u), 1, abs_tol=1e-10)

print('Vérifier que u est bien un vecteur unitaire:', is_normalized(v))

# Vecteur orthonormales
def is_orthonormal(u1, u2):
    u1_is_normalized = is_normalized(u1)
    u2_is_normalized = is_normalized(u2)
    if u1_is_normalized and u2_is_normalized:
        return is_orthogonal(u1, u2)
    return False

v1 = np.array([1, 1, 1])
v2 = np.array([0, 0, 1])

u1 = normalized_vector(v1)
u2 = normalized_vector(v2)

print('u1 et u2 sont-ils orthornormales?', is_orthonormal(u1, u2))

# Algorithme de Gram-Schmidt Exemple de la première partie de la vidéo
v1 = np.array([1, 0, 2])
v2 = np.array([2, 1, 0])
v3 = np.array([2, 2, 1])

u1 = normalized_vector(v1)

u2 = v2 - np.dot(u1, v2) * u1
u2 /= vector_length(u2)

u3 = v3 - np.dot(u1, v3) * u1 - np.dot(u2, v3) * u2
u3 /= vector_length(u3)

# Vous pouvez verifier les résultats obtenus et les comparer à ceux 
# présenter dans la première partie de la vidéo
print("Resultat u1 après utilisation de l'algorithme de Gram-Schmidt", u1)
print("Resultat u2 après utilisation de l'algorithme de Gram-Schmidt", u2)
print("Resultat u3 après utilisation de l'algorithme de Gram-Schmidt", u3)

print('Est-ce que u1 et u2 sont orthornormales?', is_orthonormal(u1, u2))
print('Est-ce que u1 et u3 sont orthornormales?', is_orthonormal(u1, u3))
print('Est-ce que u2 et u3 sont orthornormales?', is_orthonormal(u2, u3))
