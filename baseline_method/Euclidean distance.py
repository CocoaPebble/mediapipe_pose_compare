import numpy as np

a = np.random.rand(33, 3)
b = np.random.rand(33, 3)

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

distances = [euclidean_distance(a[i], b[i]) for i in range(a.shape[0])]

mean_distance = np.mean(distances)
sum_distance = np.sum(distances)

print("Mean distance:", mean_distance)
print("Sum distance:", sum_distance)