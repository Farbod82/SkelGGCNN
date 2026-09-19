import numpy as np
from typing import Union
from scipy import sparse
# from scipy import linalg as sla
from scipy.sparse import linalg as sla

import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import robust_laplacian






class skeleton3D():
    def __init__(self, pcd, init_contraction: float = 1.,
                 init_attraction: float = 0.5,
                 max_contraction: int = 4096,
                 max_attraction: int = 512,
                 termination_ratio: float = 0.003,
                 max_iteration_steps: int = 10,
                 filter_nb_neighbors: int = 20,
                 filter_std_ratio: float = 2.0,):
        self.pcd = pcd
        self.param_init_contraction = init_contraction
        self.param_init_attraction = init_attraction
        self.param_max_contraction = max_contraction
        self.param_max_attraction = max_attraction
        self.param_termination_ratio = termination_ratio
        self.max_iteration_steps = max_iteration_steps
        self.filter_nb_neighbors = filter_nb_neighbors
        self.filter_std_ratio = filter_std_ratio        
        self.contraction_type =self.__least_squares_sparse
        self.set_amplification()
        if filter_nb_neighbors and filter_std_ratio:
            self.pcd, _ = self.pcd.remove_statistical_outlier(nb_neighbors=filter_nb_neighbors,
                                                              std_ratio=filter_std_ratio)


        
    def set_amplification(self):
        num_pcd_points = np.asarray(self.pcd.points).shape[0]
        if num_pcd_points < 1000:
            contraction_amplification = 1
            termination_ratio = 0.01
        elif num_pcd_points < 1e4:
            contraction_amplification = 2
            termination_ratio = 0.007
        elif num_pcd_points < 1e5:
            contraction_amplification = 5
            termination_ratio = 0.005
        elif num_pcd_points < 0.5 * 1e6:
            contraction_amplification = 5
            termination_ratio = 0.004
        elif num_pcd_points < 1e6:
            contraction_amplification = 5
            termination_ratio = 0.003
        else:
            contraction_amplification = 8
            termination_ratio = 0.0005

        self.param_contraction_amplification = contraction_amplification
    
    def __least_squares_sparse(self, pcd_points, L, laplacian_weighting, positional_weighting):
        """
        Perform least squares sparse solving for the Laplacian-based contraction.

        Args:
            pcd_points: The input point cloud points.
            L: The Laplacian matrix.
            laplacian_weighting: The Laplacian weighting matrix.
            positional_weighting: The positional weighting matrix.

        Returns:
            The contracted point cloud.
        """
        # Define Weights
        WL = sparse.diags(laplacian_weighting)  # I * laplacian_weighting
        WH = sparse.diags(positional_weighting)

        A = sparse.vstack([L.dot(WL), WH]).tocsc()
        b = np.vstack([np.zeros((pcd_points.shape[0], 3)), WH.dot(pcd_points)])

        A_new = A.T @ A

        x = sla.spsolve(A_new, A.T @ b[:, 0], permc_spec='COLAMD')
        y = sla.spsolve(A_new, A.T @ b[:, 1], permc_spec='COLAMD')
        z = sla.spsolve(A_new, A.T @ b[:, 2], permc_spec='COLAMD')

        ret = np.vstack([x, y, z]).T

        if (np.isnan(ret)).all():
            ret = pcd_points

        return ret
    
    def extract_skeleton(self):
        pcd_points = np.asarray(self.pcd.points)
        L, M = robust_laplacian.point_cloud_laplacian(pcd_points, mollify_factor=1e-5, n_neighbors=30)
        M_list = [M.diagonal()]

        positional_weights = self.param_init_attraction * np.ones(M.shape[0])
        laplacian_weights  = (
            self.param_init_contraction * 1e3 * np.sqrt(np.mean(M.diagonal())) * np.ones(M.shape[0])
        )

        iteration     = 0
        pcd_points_current = pcd_points

        while (np.mean(M_list[-1]) / np.mean(M_list[0])) > self.param_termination_ratio:

            pcd_points_new = self.contraction_type(
                pcd_points=pcd_points_current,
                L=L,
                laplacian_weighting=laplacian_weights,
                positional_weighting=positional_weights,
            )

            if (pcd_points_new == pcd_points_current).all():
                break
            else:
                pcd_points_current = pcd_points_new

            # Update weights
            laplacian_weights  *= self.param_contraction_amplification
            positional_weights  = positional_weights * np.sqrt((M_list[0] / M.diagonal()))

            # Clip
            laplacian_weights  = np.clip(laplacian_weights, 0.1, self.param_max_contraction)
            positional_weights = np.clip(positional_weights, 0.1, self.param_max_attraction)

            # Save mass matrix diagonal
            M_list.append(M.diagonal())

            iteration += 1
            try:
                L, M = robust_laplacian.point_cloud_laplacian(
                    pcd_points_current, mollify_factor=1e-5, n_neighbors=30
                )
            except RuntimeError as er:
                print(er)
                break

            if iteration >= self.max_iteration_steps:
                break

        return pcd_points_current



