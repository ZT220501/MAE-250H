import numpy as np
from numpy import float64
import matplotlib.pyplot as plt

'''
This is the file for the utility functions.
'''



'''
Conjugate Gradient Method for solving linear systems of equations Ax = b
'''
def conjugate_gradient(A, b, x0, tol=1e-6, max_iter=100):
    '''
    Conjugate Gradient Method for solving linear systems of equations Ax = b
    '''
    x = x0
    r_old = b - A @ x
    p = r_old
    r_new = r_old

    for i in range(max_iter):
        r_old_inner_product = np.dot(r_old, r_old)

        Ap = A @ p
        alpha = r_old_inner_product / np.dot(p, Ap)
        x = x + alpha * p
        r_new = r_old - alpha * Ap

        # Exit the algorithm if the residual is small enough
        if np.linalg.norm(r_new) < tol:
            print("Converged in {} iterations".format(i+1))
            return x
        
        beta = np.dot(r_new, r_new) / r_old_inner_product
        p = r_new + beta * p
        r_old = r_new

    raise ValueError("Maximum number of iterations reached")









def visualize_divergence(divergence, pressure_mesh_grid):
    X, Y = pressure_mesh_grid
    plt.contourf(X, Y, divergence, cmap='bwr')
    plt.colorbar(label='Divergence')
    plt.title('Divergence Field')
    plt.axis('equal')
    plt.show()





