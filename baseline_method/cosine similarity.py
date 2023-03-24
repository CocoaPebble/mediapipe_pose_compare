import numpy as np

a = np.random.rand(33, 3)
b = np.random.rand(33, 3)

vector_diff = a - b

def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

a_norm = normalize_vectors(a)
b_norm = normalize_vectors(b)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    return dot_product

similarities = [cosine_similarity(a_norm[i], b_norm[i]) for i in range(a_norm.shape[0])]

mean_similarity = np.mean(similarities)
sum_similarity = np.sum(similarities)

print("Mean similarity:", mean_similarity)
print("Sum similarity:", sum_similarity)
