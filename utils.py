import numpy as np
from numpy import float64
import matplotlib.pyplot as plt

'''
This is the file for the utility functions, including the conjugate gradient method, the staggered grid, and all the operators.
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


    def __init__(self, Lx, Ly, Nx, Ny, initial_condition_velocity, initial_condition_pressure):
        '''
        Initialize the staggered grid.
        In this implementation, it follows the convention of Python indexing, 
        so that the horizontally it mimics that x is increasing from left to right,
        and the vertically it mimics that y is increasing from top to bottom.

        To vary the x value, change the second argument in the mesh_grid.
        To vary the y value, change the first argument in the mesh_grid.

        We assume the grid is uniform in this implementation
        '''
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.dx = Lx / Nx
        self.dy = Ly / Ny



        '''
        Staggered grid initialization
        '''
        # Define the mesh grid for the spatial points
        self.spatial_mesh_grid = np.meshgrid(np.linspace(0, Lx, Nx+1), np.linspace(0, Ly, Ny+1))
        # Define the mesh grid for the center points
        self.pressure_mesh_grid = np.meshgrid(np.linspace(self.dx/2, Lx-self.dx/2, Nx), np.linspace(self.dy/2, Ly-self.dy/2, Ny))
        # Define the mesh grid for the velocity points
        self.u_mesh_grid = np.meshgrid(np.linspace(0, Lx, Nx+1), np.linspace(self.dy/2, Ly-self.dy/2, Ny))
        self.v_mesh_grid = np.meshgrid(np.linspace(self.dx/2, Lx-self.dx/2, Nx), np.linspace(0, Ly, Ny+1))
        # Define the mesh grid for the vorticity points
        self.vorticity_mesh_grid = np.meshgrid(np.linspace(self.dx, Lx-self.dx, Nx-1), np.linspace(self.dy, Ly-self.dy, Ny-1))

        # Define the pressure at the cell centers using the initial condition
        self.pressure = initial_condition_pressure(self.pressure_mesh_grid)

        # Define the velocity field at the faces using the initial condition
        self.u, self.v = initial_condition_velocity(self.u_mesh_grid, self.v_mesh_grid)

        # Compute the vorticity at the cell centers based on the velocity field
        self.vorticity = vorticity(self.u, self.v, self.vorticity_mesh_grid)



    def compute_divergence(self, u, v, spatial_mesh_grid):
        '''
        Compute the divergence of the velocity field
        '''
        return divergence(u, v, spatial_mesh_grid)
    

    def compute_vorticity(self, u, v, vorticity_mesh_grid):
        '''
        Compute the vorticity of the velocity field
        '''
        return vorticity(u, v, vorticity_mesh_grid)


    # Get the grid points
    def get_grid(self):
        return self.spatial_mesh_grid, self.pressure_mesh_grid, self.u_mesh_grid, self.v_mesh_grid, self.vorticity_mesh_grid
    # Get the pressures at the cell centers
    def get_pressure(self):
        return self.pressure
    # Get the velocities at the faces
    def get_velocity(self):
        return self.u, self.v
    # Get the vorticity at the cell vertices
    def get_vorticity(self):
        return self.vorticity
    

    def visualize_velocity(self):
        '''
        Visualize the x and y components of the velocity field
        In order to do the visualization, we need to make sure that the velocity field is defined at the centers
        Thus interpolation of the velocity field is done here, so that the pressure_mesh_grid is used.
        '''
        X, Y = self.pressure_mesh_grid
        # Interpolate the x-component of the velocity field to the pressure mesh grid
        u_interpolated = (self.u[:, :-1] + self.u[:, 1:]) / 2
        v_interpolated = (self.v[:-1, :] + self.v[1:, :]) / 2


        plt.quiver(X, Y, u_interpolated, v_interpolated, color='r', scale=1, scale_units='xy')
        plt.title('Staggered Grid Velocity Field')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.grid(True)

    def visualize_vorticity(self):
        X, Y = self.vorticity_mesh_grid
        plt.contourf(X, Y, self.vorticity, cmap='bwr')
        plt.colorbar(label='Vorticity')
        plt.title('Vorticity Field')
        plt.axis('equal')
        plt.show()

    def visualize_pressure(self):
        X, Y = self.pressure_mesh_grid
        plt.contourf(X, Y, self.pressure, cmap='bwr')
        plt.colorbar(label='Pressure')
        plt.title('Pressure Field')
        plt.axis('equal')
        plt.show()





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


    vorticity = (-u[:-1, 1:-1] + u[1:, 1:-1]) / dx + (-v[1:-1, :-1] + v[1:-1, 1:]) / dy
    return vorticity




     



'''
Discrete divergence operator
TODO: Test this part
'''
def divergence(u, v, spatial_mesh_grid):
    '''
    Central difference is used for the divergence operator
    Since we're using the staggered grid, the divergence is at the cell centers
    The spatial mesh grid is used in order to compute the divergence at the cell centers
    '''

    x_grid, y_grid = spatial_mesh_grid
    dx = x_grid[0, 1] - x_grid[0, 0]
    dy = y_grid[1, 0] - y_grid[0, 0]

    Nx, Ny = x_grid.shape
    # The divergence are calculated at the cell centers
    div = np.zeros((Nx, Ny))
    div = (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy
    return div



'''
Discrete gradient operator
TODO: Test this part
'''
def gradient(p, mesh_grid):
    '''
    Calculate the gradient of a scalar field p at the cell faces.
    The x-component of the gradient is at the x-faces, and the y-component of the gradient is at the y-faces.

    p: np.ndarray(Nx, Ny)
    mesh_grid: np.meshgrid with x and y coordinates have shapes (Nx+1, Ny+1)
    '''
    x_grid, y_grid = mesh_grid
    dx = x_grid[0, 1] - x_grid[0, 0]
    dy = y_grid[1, 0] - y_grid[0, 0]

    Nx, Ny = x_grid.shape
    grad_x = np.zeros((Nx-1, Ny))
    grad_y = np.zeros((Nx, Ny-1))

    grad_x[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dx
    grad_y[1:-1, :] = (p[1:, :] - p[:-1, :]) / dy
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