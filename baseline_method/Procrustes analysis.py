import numpy as np
from scipy.spatial import procrustes

# create two random 3D arrays
a = np.random.rand(33, 3)
b = np.random.rand(33, 3)

print("Matrix 1:\n", a)
print("Matrix 2:\n", b)

mtx1, mtx2, disparity = procrustes(a, b)

# print the results in detail
# not print the scientific notation
np.set_printoptions(suppress=True)
print("Matrix 1:\n", mtx1)
print("Matrix 2:\n", mtx2)
print("Disparity:", disparity)