import os
import numpy as np
import torch
import scipy.sparse
from scipy.interpolate import griddata
from sklearn import linear_model
import warnings
from sklearn.exceptions import ConvergenceWarning

class Dares:
    def __init__(self):
        current_dir = os.path.dirname(__file__)
        # load files from this directory, even though this class may be called by a parent directory
        laplace_path = os.path.join(current_dir, "laplace.npz")
        grad_x_path = os.path.join(current_dir, "grad_x.npz")
        grad_y_path = os.path.join(current_dir, "grad_y.npz")
        verts_path = os.path.join(current_dir, "vertices.npy")

       # laplace="laplace.npz", grad_x="grad_x.npz", grad_y="grad_y.npz", verts="vertices.npy"
        # Load operators and mesh
        L = scipy.sparse.load_npz(laplace_path)
        Gx = scipy.sparse.load_npz(grad_x_path)
        Gy = scipy.sparse.load_npz(grad_y_path)
        self.verts = np.load(verts_path)
        self.dim = self.verts.shape[0]

        # Mesh boundaries
        self.x_min, self.x_max = np.min(self.verts[:, 0]), np.max(self.verts[:, 0])
        self.y_min, self.y_max = np.min(self.verts[:, 1]), np.max(self.verts[:, 1])

        # Parameters
        wind_x = 0.0
        wind_y = 0.0
        kappa = 0.2

        # Find border
        eps = 0.05
        b = 1. * np.logical_or(np.logical_or(self.verts[:, 0] < eps + self.x_min, self.verts[:, 0] > self.x_max - eps),
                            np.logical_or(self.verts[:, 1] < eps + self.y_min, self.verts[:, 1] > self.y_max - eps))

        # Set boundary condition (i.e., = 0)
        R = scipy.sparse.diags(b)

        # PDE
        O = kappa * L + wind_x * Gx + wind_y * Gy
        # Convert to LIL for efficient modification
        O = O.tolil()
        # Combine model and boundary condition into one matrix
        O[b == 1] = R.todense()[b == 1]
        # Convert back to CSR after modification
        O = O.tocsr()
        self.Oinv = np.linalg.inv(O.todense())

    def set_measurements(self, sensor_positions, measurements):
        num_msmt = len(measurements)

        # Initialize measurement matrix
        self.M = np.zeros((num_msmt, self.dim))
        self.m = np.zeros((num_msmt, 1))

        # Fill the measurement matrix
        for i, (x, y) in enumerate(sensor_positions):
            idx = np.argmin(np.linalg.norm(self.verts - np.array([x, y]), axis=1))
            self.M[i, idx] = 1
            self.m[i] = measurements[i]

    def predict(self):
        # Suppress the ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            
            # Lasso reconstruction
            clf = linear_model.Lasso(alpha=0.001, max_iter=5000)
            clf.fit(np.asarray(np.dot(self.M, self.Oinv)), self.m)
            
        q_rec = clf.coef_
        f_rec = np.array(np.dot(self.Oinv, q_rec))

        # Convert triangulated mesh to regular grid for visualization
        grid_x_regular, grid_y_regular = np.meshgrid(np.linspace(self.x_min, self.x_max, 30), np.linspace(self.y_min, self.y_max, 25))
        f_rec_grid = griddata(self.verts, f_rec.flatten(), (grid_x_regular, grid_y_regular), method='cubic')

        pred_dares = torch.rot90(torch.tensor(f_rec_grid), 1, dims=(0,1))
        pred_dares = torch.flip(pred_dares, dims=(0,))

        return pred_dares
