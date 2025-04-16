import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils

A = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])

x0 = np.array([2, 1])

x = utils.conjugate_gradient(A, b, x0)
print(x)
