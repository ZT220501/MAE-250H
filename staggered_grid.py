import numpy as np
import matplotlib.pyplot as plt
import importlib
import operators

importlib.reload(operators)
from operators import vorticity, divergence, gradient



'''
Generate the staggered grid.
The position of the variables have been verified.
'''
class StaggeredGrid:


    def __init__(self, Lx, Ly, Nx, Ny, initial_condition_velocity, initial_condition_pressure, u_left=0.0, u_right=0.0, v_top=0.0, v_bottom=0.0):
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

        # Create mesh grids for visualization and initialization
        self.u_mesh_grid = np.meshgrid(np.linspace(self.dx, Lx-self.dx, Nx-1), np.linspace(0.5*self.dy, Ly-0.5*self.dy, Ny))
        self.v_mesh_grid = np.meshgrid(np.linspace(0.5*self.dx, Lx-0.5*self.dx, Nx), np.linspace(self.dy, Ly-self.dy, Ny-1))
        self.pressure_mesh_grid = np.meshgrid(np.linspace(0.5*self.dx, Lx-0.5*self.dx, Nx), np.linspace(0.5*self.dy, Ly-0.5*self.dy, Ny))
        self.vorticity_mesh_grid = np.meshgrid(np.linspace(self.dx, Lx-self.dx, Nx-1), np.linspace(self.dy, Ly-self.dy, Ny-1))

        # Staggered grid initialization, 2D version
        self.u, self.v = initial_condition_velocity(self.u_mesh_grid, self.v_mesh_grid)         # Shape (Nx-1, Ny) and (Nx, Ny-1)
        self.pressure = initial_condition_pressure(self.pressure_mesh_grid)                     # Shape (Nx, Ny)
        self.vorticity = np.zeros((Ny-1, Nx-1))                                                 # Shape (Nx-1, Ny-1)

        # Staggered grid initialization, 1D version
        self.u_vector, self.v_vector = self.velocity_2D_to_1D()                               # Shape (Nx-1)*Ny, Nx*(Ny-1)
        self.pressure_vector = self.pressure_2D_to_1D()                                       # Shape Nx*Ny-1, pinned at the bottom left corner
        self.vorticity_vector = self.vorticity_2D_to_1D()                                      # Shape (Nx-1)*(Ny-1)

    ###############################
    # Compute discrete quantities #
    ###############################
    def compute_divergence(self):
        '''
        Compute the divergence of the velocity field
        '''
        return divergence(self.u, self.v, self.pressure_mesh_grid)
    
    def compute_gradient(self):
        '''
        Compute the gradient of the pressure field
        '''
        return gradient(self.pressure, self.pressure_mesh_grid)
    
    def compute_vorticity(self):
        '''
        Compute the vorticity of the velocity field
        '''
        return vorticity(self.u, self.v, self.vorticity_mesh_grid)

    #################
    # Get functions #
    #################
    def get_variables(self):
        return self.u, self.v, self.pressure, self.vorticity
    def get_mesh_grids(self):
        return self.u_mesh_grid, self.v_mesh_grid, self.pressure_mesh_grid, self.vorticity_mesh_grid

    
    #####################
    # Pointer functions #
    #####################
    def pointer_velocity(self):
        '''
        Generate a map that maps the 2D index of the 
        velocity field to the 1D index of the velocity field.
        '''
        u_pointer = np.zeros((self.Nx-1, self.Ny))
        v_pointer = np.zeros((self.Nx, self.Ny-1))

        idx = 0
        for i in range(self.Nx-1):
            for j in range(self.Ny):
                u_pointer[i, j] = idx
                idx += 1
        idx = 0
        for i in range(self.Nx):
            for j in range(self.Ny-1):
                v_pointer[i, j] = idx
                idx += 1

        return u_pointer.astype(int), v_pointer.astype(int)
    
    def pointer_pressure(self):
        '''
        Generate a map that maps the 2D index of the 
        pressure field to the 1D index of the pressure field.
        '''
        idx = 0
        pressure_pointer = np.zeros((self.Nx, self.Ny))
        for i in range(self.Nx):
            for j in range(self.Ny):
                if i == 0 and j == 0:
                    # Pinned pressure value at the bottom left corner.
                    # We set it to NaN, so that we'll not use it in the computation by mistake.
                    pressure_pointer[i, j] = np.nan
                else:
                    pressure_pointer[i, j] = idx
                    idx += 1
        return pressure_pointer.astype(int)

    def pointer_vorticity(self):
        '''
        Generate a map that maps the 2D index of the 
        vorticity field to the 1D index of the vorticity field.
        '''
        idx = 0
        vorticity_pointer = np.zeros((self.Nx-1, self.Ny-1))
        for i in range(self.Nx-1):
            for j in range(self.Ny-1):
                vorticity_pointer[i, j] = idx
                idx += 1
        return vorticity_pointer.astype(int)


    ###################################################
    # Synchronization of 2D and 1D pressure, velocity #
    ###################################################
    def pressure_2D_to_1D(self):
        '''
        After changing the 2D representation of the pressure field,
        we need to update the 1D representation of the pressure field.
        '''
        # The bottom left corner is pinned to 0, so that we 
        # don't need to solve it.
        return self.pressure.reshape(-1)[1:]
    
    def vorticity_2D_to_1D(self):
        '''
        After changing the 2D representation of the vorticity field,
        we need to update the 1D representation of the vorticity field.
        '''
        return self.vorticity.reshape(-1)
    
    def velocity_2D_to_1D(self):
        '''
        After changing the 2D representation of the pressure field,
        we need to update the 1D representation of the velocity field.
        '''
        # The boundary velocities are specified, so that we
        # don't need to solve them.
        return self.u.reshape(-1), self.v.reshape(-1)
    
    def pressure_1D_to_2D(self):
        '''
        After changing the 1D representation of the pressure field,
        we need to update the 2D representation of the pressure field.
        '''
        pressure = np.insert(self.pressure_vector, 0, 0)
        return pressure.reshape(self.Nx, self.Ny)

    def vorticity_1D_to_2D(self):
        '''
        After changing the 1D representation of the vorticity field,
        we need to update the 2D representation of the vorticity field.
        '''
        return self.vorticity_vector.reshape(self.Nx-1, self.Ny-1)
    
    def velocity_1D_to_2D(self):
        '''
        After changing the 1D representation of the velocity field,
        we need to update the 2D representation of the velocity field.
        '''
        u = self.velocity[:self.u_pointer.size].reshape(self.u.shape)
        v = self.velocity[self.u_pointer.size:].reshape(self.v.shape)
        return u, v

    ###########################
    # Visualization functions #
    ###########################
    def visualize_velocity(self, scale=0.5):
        '''
        Visualize the x and y components of the velocity field, on the inner grids.
        In order to do the visualization, we need to make sure that the velocity field is defined at the vertices of the cell.
        Thus interpolation of the velocity field is done here, so that the pressure_mesh_grid is used.
        '''
        X, Y = self.vorticity_mesh_grid
        # Interpolate the x-component of the velocity field to the pressure mesh grid
        u_interpolated = (self.u[:-1, :] + self.u[1:, :]) / 2
        v_interpolated = (self.v[:, :-1] + self.v[:, 1:]) / 2


        plt.quiver(X, Y, u_interpolated, v_interpolated, color='blue', scale=scale, scale_units='xy')
        plt.title('Staggered Grid Velocity Field', fontsize=16)
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.xticks(np.round(X[0, :], 2), fontsize=16)
        plt.yticks(np.round(Y[:, 0], 2), fontsize=16)
        plt.axis('equal')
        plt.grid(axis='both')

    def visualize_vorticity(self):
        X, Y = self.vorticity_mesh_grid
        vorticity = self.compute_vorticity()
        plt.contourf(X, Y, vorticity, cmap='bwr')
        plt.colorbar(label='Vorticity')
        plt.title('Vorticity Field', fontsize=16)
        plt.xticks(np.round(X[0, :], 2), fontsize=12)
        plt.yticks(np.round(Y[:, 0], 2), fontsize=12)
        plt.axis('equal')
        plt.show()

    def visualize_pressure(self):
        X, Y = self.pressure_mesh_grid
        plt.contourf(X, Y, self.pressure, cmap='bwr')
        plt.colorbar(label='Pressure')
        plt.title('Pressure Field', fontsize=16)
        plt.xticks(np.round(X[0, :], 2), fontsize=12)
        plt.yticks(np.round(Y[:, 0], 2), fontsize=12)
        plt.axis('equal')
        plt.show()