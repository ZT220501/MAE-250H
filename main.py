import numpy as np
import matplotlib.pyplot as plt

from utils import conjugate_gradient


A = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])

x0 = np.array([2, 1])

x = conjugate_gradient(A, b, x0)
print(x)

