import numpy as np
from numpy import float64




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
            return x
        
        beta = np.dot(r_new, r_new) / r_old_inner_product
        p = r_new + beta * p
        r_old = r_new

    raise ValueError("Maximum number of iterations reached")



'''
Generate the staggered grid
TODO: Test this part
'''
class staggered_grid:


    def __init__(self, Lx, Ly, Nx, Ny):
        '''
        Initialize the staggered grid.
        In this implementation, it follows the convention of Python indexing, 
        so that the horizontally it mimics that x is increasing from left to right,
        and the vertically it mimics that y is increasing from top to bottom.

        To vary the x value, change the second argument in the mesh_grid.
        To vary the y value, change the first argument in the mesh_grid.
        '''
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny

        # Define the grid points
        self.x_grid = np.linspace(0, Lx, Nx+1)
        self.y_grid = np.linspace(0, Ly, Ny+1)
        self.mesh_grid = np.meshgrid(self.x_grid, self.y_grid)

        # Define the pressures at the cell centers
        self.pressure_grid = np.zeros((Nx, Ny))

        # Define the velocities at the faces
        self.u = np.zeros((Nx-1, Ny))
        self.v = np.zeros((Nx, Ny-1))

        # Define the vorticity at the cell centers
        self.vorticity_grid = np.zeros((Nx-1, Ny-1))


        # Get the grid points
        def get_grid(self):
            return self.x_grid, self.y_grid, self.mesh_grid
        # Get the pressures at the cell centers
        def get_pressure(self):
            return self.pressure_grid
        # Get the velocities at the faces
        def get_velocity(self):
            return self.u, self.v
        # Get the vorticity at the cell vertices
        def get_vorticity(self):
            return self.vorticity_grid
        



     



'''
Discrete divergence operator
TODO: Test this part
'''
def divergence(u, v, mesh_grid):
    # Central difference is used for the divergence operator
    # Since we're using the staggered grid, the divergence is at the cell centers
    x_grid, y_grid = mesh_grid
    dx = x_grid[0, 1] - x_grid[0, 0]
    dy = y_grid[1, 0] - y_grid[0, 0]

    Nx, Ny = x_grid.shape
    # The divergence are calculated at the cell centers
    div = np.zeros((Nx-1, Ny-1))
    div[1:-1, 1:-1] = (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy
    return div



'''
Discrete gradient operator
TODO: Finish this part: How to handle the boundary condition?
'''
def gradient(p, mesh_grid):
    '''
    Calculate the gradient of a scalar field p at the cell faces.

    p: np.ndarray(Nx, Ny)
    mesh_grid: np.meshgrid with x and y coordinates have shapes (Nx+1, Ny+1)
    '''
    x_grid, y_grid = mesh_grid
    dx = x_grid[0, 1] - x_grid[0, 0]
    dy = y_grid[1, 0] - y_grid[0, 0]

    Nx, Ny = x_grid.shape
    grad = np.zeros((Nx, Ny))
    grad[1:-1, 1:-1] = (u[1:-1, 1:] - u[1:-1, :-1]) / dx + (v[1:, 1:-1] - v[:-1, 1:-1]) / dy
    return grad





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






def upwind_2D(u_previous, mesh_grid, V0, vector_field):
    '''
    Solution for the advection equation in 2D, with the vector field \vec{v}=(V_0x, V_0y) or \vec{v}=(0, -V_0x)

    u_previous: the previous time step solution
    mesh_grid: the mesh grid
    dt: the time step
    V0: the vector field strength
    vector_field: the vector field


    TODO: Finish this part for general nonlinear advection instead of simply two cases
    '''
    x_grid, y_grid = mesh_grid

    # Get the grid spacing
    dx = x_grid[0, 1] - x_grid[0, 0]
    dy = y_grid[1, 0] - y_grid[0, 0]




    # Implement the Engquist-Osher flux
    # Notice that the vector field is convex in each component in both cases
    if vector_field == "expanding":
        # Expanding flow case vector field \vec{v}=(V_0x, V_0y)
        x_faces = 0.5 * (x_grid[1:-1, 1:] + x_grid[1:-1, :-1])
        F_x = enquist_osher_flux_expanding(u_previous[1:-1, :-1], u_previous[1:-1, 1:], V0, x_faces)
        y_faces = 0.5 * (y_grid[1:, 1:-1] + y_grid[:-1, 1:-1])
        F_y = enquist_osher_flux_expanding(u_previous[:-1, 1:-1], u_previous[1:, 1:-1], V0, y_faces)

        # Finite volume update
        u_convection = (F_x[:, 1:] - F_x[:, :-1]) / dx + (F_y[1:, :] - F_y[:-1, :]) / dy
    elif vector_field == "sheer":
        # Sheer flow case vector field \vec{v}=(0, -V_0x)
        # TODO: Implement the sheer flow case using the Engquist-Osher flux
        vy_positive = np.maximum(0, -V0 * x_grid[1:-1, 1:-1])
        vy_negative = np.minimum(0, -V0 * x_grid[1:-1, 1:-1])
        uy_positive = (u_previous[2:, 1:-1] - u_previous[1:-1, 1:-1]) / dy
        uy_negative = (u_previous[1:-1, 1:-1] - u_previous[:-2, 1:-1]) / dy
        u_convection = vy_positive * uy_negative + vy_negative * uy_positive
    else:
        raise ValueError("Invalid vector field.")   


    return u_convection