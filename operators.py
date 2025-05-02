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
def divergence(u, v, pressure_mesh_grid, v_top=0, v_bottom=0, u_left=0, u_right=0):
    '''
    Central difference is used for the divergence operator
    Since we're using the staggered grid, the divergence is at the cell centers
    The spatial mesh grid is used in order to compute the divergence at the cell centers
    '''

    X, Y = pressure_mesh_grid
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    Nx, Ny = X.shape

    if isinstance(v_top, float):
        v_top = v_top * np.ones(Nx)
    if isinstance(v_bottom, float):
        v_bottom = v_bottom * np.ones(Nx)
    if isinstance(u_left, float):
        u_left = u_left * np.ones(Ny)
    if isinstance(u_right, float):
        u_right = u_right * np.ones(Ny)


    # The divergence are calculated at the cell centers
    div = np.zeros((Nx, Ny))

    # Interior divergence
    div[1:-1, 1:-1] = (u[1:-1, 1:] - u[1:-1, :-1]) / dx + (v[1:, 1:-1] - v[:-1, 1:-1]) / dy

    # Top boundary, exclude corners
    div[0, 1:-1] = (u[0, 1:] - u[0, :-1]) / dx + (v[0, 1:-1] - v_top[1:-1]) / dy
    # Bottom boundary, exclude corners
    div[-1, 1:-1] = (u[-1, 1:] - u[-1, :-1]) / dx + (v_bottom[1:-1] - v[-1, 1:-1]) / dy
    # Left boundary, exclude corners
    div[1:-1, 0] = (u[1:-1, 0] - u_left[1:-1]) / dx + (v[1:, 0] - v[:-1, 0]) / dy
    # Right boundary, exclude corners
    div[1:-1, -1] = (u_right[1:-1] - u[1:-1, -1]) / dx + (v[1:, -1] - v[:-1, -1]) / dy

    # Top-left corner
    div[0, 0] = (u[0, 0] - u_left[0]) / dx + (v[0, 0] - v_top[0]) / dy
    # Top-right corner
    div[0, -1] = (u_right[0] - u[0, -1]) / dx + (v[0, -1] - v_top[-1]) / dy
    # Bottom-left corner
    div[-1, 0] = (u[-1, 0] - u_left[-1]) / dx + (v_bottom[0] - v[-1, 0]) / dy
    # Bottom-right corner
    div[-1, -1] = (u_right[-1] - u[-1, -1]) / dx + (v_bottom[-1] - v[-1, -1]) / dy

    # In divergence, we also pin the bottom-left corner value, as in the pressure case.
    div[0, 0] = np.nan
    
    return div



'''
Discrete divergence operator at the boundary
It has the same shape as the divergence operator, but only at the four boundaries are non-zero.
'''
def divergence_bc(pressure_mesh_grid, v_top=0, v_bottom=0, u_left=0, u_right=0):
    '''
    Calculate the divergence at the boundary
    '''
    # FIXME: Currently the implementation is as in the class, but it looks quite confusing...
    X, Y = pressure_mesh_grid
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    Nx, Ny = X.shape
    div_bc = np.zeros((Nx, Ny))

    if isinstance(v_top, float):
        v_top = v_top * np.ones(Nx)
    if isinstance(v_bottom, float):
        v_bottom = v_bottom * np.ones(Nx)
    if isinstance(u_left, float):
        u_left = u_left * np.ones(Ny)
    if isinstance(u_right, float):
        u_right = u_right * np.ones(Ny)

    # Top boundary, exclude corners
    div_bc[0, 1:-1] = -v_top[1:-1] / dy
    # Bottom boundary, exclude corners
    div_bc[-1, 1:-1] = v_bottom[1:-1] / dy
    # Left boundary, exclude corners
    div_bc[1:-1, 0] = -u_left[1:-1] / dx
    # Right boundary, exclude corners
    div_bc[1:-1, -1] = u_right[1:-1] / dx

    # Top-left corner
    div_bc[0, 0] = -u_left[0] / dx - v_top[0] / dy
    # Top-right corner
    div_bc[0, -1] = u_right[0] / dx - v_top[-1] / dy
    # Bottom-left corner
    div_bc[-1, 0] = -u_left[-1] / dx + v_bottom[0] / dy
    # Bottom-right corner
    div_bc[-1, -1] = u_right[-1] / dx + v_bottom[-1] / dy

    # In divergence, we also pin the bottom-left corner value, as in the pressure case.
    div_bc[0, 0] = np.nan

    return div_bc







'''
Discrete gradient operator
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
Discrete Laplacian operator
'''
def laplacian(u, mesh_grid, U):
    '''
    Calculate the the Laplacian at the FIXME: where is the Laplacian?
    '''
    x_grid, y_grid = mesh_grid
    # Only consider the uniform grid case
    u_laplacian = np.zeros(u.shape, dtype=float64)
    h = x_grid[0, 1] - x_grid[0, 0]


    u_laplacian[0, 1:-1] = (u[1, 1:-1] - 2 * u[0, 1:-1] + u[-2, 1:-1]) / h**2 + (u[0, 2:] - 2 * u[0, 1:-1] + u[0, :-2]) / h**2
    u_laplacian[-1, 1:-1] = (u[1, 1:-1] - 2 * u[-1, 1:-1] + u[-2, 1:-1]) / h**2 + (u[-1, 2:] - 2 * u[-1, 1:-1] + u[-1, :-2]) / h**2
    u_laplacian[1:-1, 0] = (u[1:-1, 1] - 2 * u[1:-1, 0] + u[1:-1, -2]) / h**2 + (u[2:, 0] - 2 * u[1:-1, 0] + u[:-2, 0]) / h**2
    u_laplacian[1:-1, -1] = (u[1:-1, 1] - 2 * u[1:-1, -1] + u[1:-1, -2]) / h**2 + (u[2:, -1] - 2 * u[1:-1, -1] + u[:-2, -1]) / h**2

    # Corner case
    u_laplacian[0, 0] = (u[1, 0] - 2 * u[0, 0] + u[-2, 0]) / h**2 + (u[0, 1] - 2 * u[0, 0] + u[0, -2]) / h**2
    u_laplacian[-1, 0] = (u[1, 0] - 2 * u[-1, 0] + u[-2, 0]) / h**2 + (u[-1, 1] - 2 * u[-1, 0] + u[-1, -2]) / h**2
    u_laplacian[0, -1] = (u[0, 1] - 2 * u[0, -1] + u[0, -2]) / h**2 + (u[1, -1] - 2 * u[0, -1] + u[-2, -1]) / h**2
    u_laplacian[-1, -1] = (u[-2, -1] - 2 * u[-1, -1] + u[1, -1]) / h**2 + (u[-1, -2] - 2 * u[-1, -1] + u[-1, 1]) / h**2


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






def nonlinear_advection(u, v, pressure_mesh_grid, U):
    '''
    Calculate the nonlinear advection term at the cell faces.
    Here the implementation uses the divergence form.
    Here U is the horizontal velocity at the top boundary.
    '''
    X, Y = pressure_mesh_grid

    Nx, Ny = X.shape

   # Extend the velocity field to the boundarys
    # Since we deal with the lid driven cavity problem, WLOG we can assume that the velocity field is zero on the boundary
    # and this won't hurt the velocity divergence calculation
    u_extended = np.zeros((Nx, Ny+1))
    v_extended = np.zeros((Nx+1, Ny))

    u_extended[:, 1:-1] = u
    v_extended[1:-1, :] = v



    return
