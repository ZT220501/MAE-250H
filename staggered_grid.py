import numpy as np
import matplotlib.pyplot as plt
import importlib
import operators

importlib.reload(operators)
from operators import vorticity, divergence, gradient



'''
Generate the staggered grid.
The position of the variables have been verified.
TODO: Test the operator implementations.
'''
class StaggeredGrid:


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



    def compute_divergence(self):
        '''
        Compute the divergence of the velocity field
        '''
        return divergence(self.u, self.v, self.pressure_mesh_grid)
    

    def compute_gradient(self):
        '''
        Compute the gradient of the pressure field
        '''
        return gradient(self.pressure, self.spatial_mesh_grid)
    


    def compute_vorticity(self):
        '''
        Compute the vorticity of the velocity field
        '''
        return vorticity(self.u, self.v, self.vorticity_mesh_grid)


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
    

    def visualize_velocity(self, scale=0.5):
        '''
        Visualize the x and y components of the velocity field
        In order to do the visualization, we need to make sure that the velocity field is defined at the vertices of the cell.
        Thus interpolation of the velocity field is done here, so that the pressure_mesh_grid is used.
        '''
        X, Y = self.vorticity_mesh_grid
        # Interpolate the x-component of the velocity field to the pressure mesh grid
        u_interpolated = (self.u[:-1, 1:-1] + self.u[1:, 1:-1]) / 2
        v_interpolated = (self.v[1:-1, :-1] + self.v[1:-1, 1:]) / 2


        plt.quiver(X, Y, u_interpolated, v_interpolated, color='blue', scale=scale, scale_units='xy')
        plt.title('Staggered Grid Velocity Field')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks(np.round(X[0, :], 2))
        plt.yticks(np.round(Y[:, 0], 2))
        plt.axis('equal')
        plt.grid(axis='both')

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