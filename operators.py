import numpy as np
from numpy import float64





'''
This file contains the operators on the staggered grid, including the vorticity, divergence, gradient, nonlinear advection, and Laplacian.
'''





'''
Discrete vorticity operator
'''
def vorticity(u, v, vorticity_mesh_grid):
    '''
    Compute the vorticity at the inner cell vertices
    '''
    X, Y = vorticity_mesh_grid
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    vorticity = (-v[:, :-1] + v[:, 1:]) / dx - (-u[:-1, :] + u[1:, :]) / dy
    return vorticity




     



'''
Discrete divergence operator
'''
def divergence(u, v, pressure_mesh_grid):
    '''
    Central difference is used for the divergence operator
    Since we're using the staggered grid, the divergence is at the cell centers
    The spatial mesh grid is used in order to compute the divergence at the cell centers
    '''

    X, Y = pressure_mesh_grid
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    Nx, Ny = X.shape
    # The divergence are calculated at the cell centers
    div = np.zeros((Nx, Ny))
    div = (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy
    return div



'''
Discrete gradient operator
TODO: Test this part
'''
def gradient(p, pressure_mesh_grid):
    '''
    Calculate the gradient of a scalar field p at the cell faces.
    The x-component of the gradient is at the x-faces, and the y-component of the gradient is at the y-faces.

    p: np.ndarray(Nx, Ny)
    pressure_mesh_grid: np.meshgrid with x and y coordinates have shapes (Nx, Ny)
    '''
    X, Y = pressure_mesh_grid
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    Nx, Ny = X.shape
    grad_x = np.zeros((Nx, Ny-1))
    grad_y = np.zeros((Nx-1, Ny))

    grad_x = (p[:, 1:] - p[:, :-1]) / dx
    grad_y = (p[1:, :] - p[:-1, :]) / dy
    return grad_x, grad_y





'''
2D Discrete Laplacian operator
'''
def laplacian(u, mesh_grid, bc="Periodic"):
    '''
    Implement the 2D discrete Laplacian, under Dirichlet, Neumann, and periodic BC
    '''
    x_grid, y_grid = mesh_grid
    # Only consider the uniform grid case
    u_laplacian = np.zeros(u.shape, dtype=float64)
    h = x_grid[0, 1] - x_grid[0, 0]

    if bc == "Periodic":
        # Periodic boundary condition
        # Edge case
        u_laplacian[0, 1:-1] = (u[1, 1:-1] - 2 * u[0, 1:-1] + u[-2, 1:-1]) / h**2 + (u[0, 2:] - 2 * u[0, 1:-1] + u[0, :-2]) / h**2
        u_laplacian[-1, 1:-1] = (u[1, 1:-1] - 2 * u[-1, 1:-1] + u[-2, 1:-1]) / h**2 + (u[-1, 2:] - 2 * u[-1, 1:-1] + u[-1, :-2]) / h**2
        u_laplacian[1:-1, 0] = (u[1:-1, 1] - 2 * u[1:-1, 0] + u[1:-1, -2]) / h**2 + (u[2:, 0] - 2 * u[1:-1, 0] + u[:-2, 0]) / h**2
        u_laplacian[1:-1, -1] = (u[1:-1, 1] - 2 * u[1:-1, -1] + u[1:-1, -2]) / h**2 + (u[2:, -1] - 2 * u[1:-1, -1] + u[:-2, -1]) / h**2

        # Corner case
        u_laplacian[0, 0] = (u[1, 0] - 2 * u[0, 0] + u[-2, 0]) / h**2 + (u[0, 1] - 2 * u[0, 0] + u[0, -2]) / h**2
        u_laplacian[-1, 0] = (u[1, 0] - 2 * u[-1, 0] + u[-2, 0]) / h**2 + (u[-1, 1] - 2 * u[-1, 0] + u[-1, -2]) / h**2
        u_laplacian[0, -1] = (u[0, 1] - 2 * u[0, -1] + u[0, -2]) / h**2 + (u[1, -1] - 2 * u[0, -1] + u[-2, -1]) / h**2
        u_laplacian[-1, -1] = (u[-2, -1] - 2 * u[-1, -1] + u[1, -1]) / h**2 + (u[-1, -2] - 2 * u[-1, -1] + u[-1, 1]) / h**2

    elif bc == "Neumann":
        # (Homogeneous) Neumann boundary condition
        # Here we use the second order Neumann boundary condition approximation
        # Assume that near the boundary, the mesh grid is uniform
         u_laplacian[0, 1:-1] = (2 * u[1, 1:-1] + u[0, :-2] + u[0, 2:] - 4 * u[0, 1:-1]) / (y_grid[1, 1:-1] - y_grid[0, 1:-1])**2
         u_laplacian[-1, 1:-1] = (2 * u[-2, 1:-1] + u[-1, :-2] + u[-1, 2:] - 4 * u[-1, 1:-1]) / (y_grid[-1, 1:-1] - y_grid[-2, 1:-1])**2
         u_laplacian[1:-1, 0] =(2 * u[1:-1, 1] + u[:-2, 0] + u[2:, 0] - 4 * u[1:-1, 0]) / (x_grid[1:-1, 1] - x_grid[1:-1, 0])**2
         u_laplacian[1:-1, -1] = (2 * u[1:-1, -2] + u[:-2, -1] + u[2:, -1] - 4 * u[1:-1, -1]) / (x_grid[1:-1, -1] - x_grid[1:-1, -2])**2
 
         # Treat corner points for the Neumann boundary condition
         # We assume that the grid are uniform and the x&y direction spatial discretization are of the same size, for all corners
         u_laplacian[0, 0] = (2 * u[0, 1] + 2 * u[1, 0] - 4 * u[0, 0]) / (y_grid[1, 0] - y_grid[0, 0])**2
         u_laplacian[0, -1] = (2 * u[0, -2] + 2 * u[1, -1] - 4 * u[0, -1]) / (y_grid[1, 0] - y_grid[0, 0])**2
         u_laplacian[-1, 0] = (2 * u[-2, 0] + 2 * u[-1, 1] - 4 * u[-1, 0]) / (y_grid[1, 0] - y_grid[0, 0])**2
         u_laplacian[-1, -1] = (2 * u[-2, -1] + 2 * u[-1, -2] - 4 * u[-1, -1]) / (y_grid[1, 0] - y_grid[0, 0])**2
    else:
        print("Wrong boundary condition; check your spelling.")

    x_central = x_grid[1:-1, 1:-1]
    x_left = x_grid[1:-1, :-2]
    x_right = x_grid[1:-1, 2:]

    y_central = y_grid[1:-1, 1:-1]
    y_left = y_grid[:-2, 1:-1]
    y_right = y_grid[2:, 1:-1]


    # 2D Laplacian for the non-uniform grid case
    u_laplacian[1:-1, 1:-1] = 2 / ((x_right - x_central) * (x_central - x_left) * (x_right - x_left)) * (u[1:-1, 2:] * (x_central - x_left) - u[1:-1, 1:-1] * (x_right - x_left) + u[1:-1, :-2] * (x_right - x_central)) \
              + 2 / ((y_right - y_central) * (y_central - y_left) * (y_right - y_left)) * (u[2:, 1:-1] * (y_central - y_left) - u[1:-1, 1:-1] * (y_right - y_left) + u[:-2, 1:-1] * (y_right - y_central))


    return u_laplacian






def nonlinear_advection(u_previous, mesh_grid, V0, vector_field):
    '''
    Solution for the advection equation in 2D, with the vector field \vec{v}=(V_0x, V_0y) or \vec{v}=(0, -V_0x)

    u_previous: the previous time step solution
    mesh_grid: the mesh grid
    dt: the time step
    V0: the vector field strength
    vector_field: the vector field
    '''
    pass