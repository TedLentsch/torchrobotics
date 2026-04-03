import torch
from typing import Dict, Optional


PARAMS = {
    'xyradius_threshold': float('inf'),   # Unit: meters.
    'z_min_threshold': -float('inf'),   # Unit: meters.
    'z_max_threshold': float('inf'),   # Unit: meters.
    'num_trials': 100,   # Unit: 1.
    'inlier_threshold': 0.30,   # Unit: meters.
    'ground_threshold': 0.30,   # Unit: meters.
}


class RANSACFlatPlane:
    """
    RANSAC-based plane fitting for ground removal.
    """
    METHOD_NAME = 'RANSACFlatPlane'

    def __init__(
            self, 
            params: Optional[Dict] = None,
            device: Optional[torch.device] = None,
        ):
        """
        Args:
            params (dict, optional) : Hyperparameters for ground removal.
            device (torch.device, optional) : If None, use the input tensor's device at call time.
        """
        self.params = dict(PARAMS) if params is None else dict(params)
        self.fitted_plane: Optional[torch.Tensor] = None
        self.device = device

    @torch.no_grad()
    def fit(
            self,
            pc: torch.Tensor,
            generator: Optional[torch.Generator] = None,
        ) -> torch.Tensor:
        """
        Fit a plane using RANSAC and return a ground mask.

        Example usage:
            torch_generator = torch.Generator(device=pc.device).manual_seed(42)
            bool_ground = ransac.fit(pc=pc, generator=torch_generator)
            ground_pc = pc[bool_ground]
            where pc has shape (N,3) <x,y,z>, bool_ground has shape (N,),
            ground_pc has shape (M,3), where M <= N.

        Args:
            pc (torch.Tensor): Shape (N,3) <x,y,z>. Unit: meters.
            generator (torch.Generator, optional): Random number generator (RNG) for reproducibility.

        Returns:
            bool_ground (torch.Tensor): Boolean mask (N,), where True = ground. Unit: 1.
        """
        assert pc.ndim == 2 and pc.shape[1] == 3, (
            f'Point cloud must be (N, 3), got {pc.shape}'
        )

        dev = self.device or pc.device
        pc = pc.to(dev)
        
        # Homogeneous coords for fast plane-point distance calculation.
        pc_hom = torch.cat([pc, torch.ones((pc.shape[0], 1), device=dev)], dim=1)

        # Pre-filter points to a region of interest.
        R = self.params['xyradius_threshold']
        zmin, zmax = self.params['z_min_threshold'], self.params['z_max_threshold']
        
        mask_roi = ((pc[:,0]**2 + pc[:,1]**2 <= R**2)
                    & (pc[:,2] >= zmin) & (pc[:,2] <= zmax))
        pc_roi = pc_hom[mask_roi]

        assert pc_roi.shape[0] >= 3, (
            'Insufficient points in ROI to fit plane!'
        )

        # Vectorized RANSAC Sampling.
        T, M = self.params['num_trials'], pc_roi.shape[0]
        
        # Sample triplets.
        ids = torch.randint(
            low=0, high=M, size=(T, 3), device=dev, generator=generator
        )
        
        # Re-sample duplicates (rare but necessary for safety).
        is_dup = (ids[:,0] == ids[:,1]) | (ids[:,0] == ids[:,2]) | (ids[:,1] == ids[:,2])
        while is_dup.any():
            ids[is_dup] = torch.randint(
                low=0, high=M, size=(int(is_dup.sum()), 3), device=dev, generator=generator
            )
            is_dup = (ids[:,0] == ids[:,1]) | (ids[:,0] == ids[:,2]) | (ids[:,1] == ids[:,2])

        # Calculate plane parameters (a, b, c, d).
        triplets = pc_roi[ids,:3]
        v1 = triplets[:,1] - triplets[:,0]
        v2 = triplets[:,2] - triplets[:,0]
        normals = torch.linalg.cross(v1, v2)
        
        norms = normals.norm(dim=1, keepdim=True)
        valid = (norms > 1e-6).flatten()
        
        if not valid.any():
            raise RuntimeError('RANSAC failed: No valid non-collinear triplets found!')

        # Normalize normals to unit vectors.
        unit_normals = torch.zeros_like(normals)
        unit_normals[valid] = normals[valid] / norms[valid]
        
        # Calculate d: d = -(ax + by + cz).
        ds = torch.zeros((T, 1), device=dev)
        ds[valid] = -(unit_normals[valid] * triplets[valid,0]).sum(dim=1, keepdim=True)
        
        planes = torch.cat([unit_normals, ds], dim=1)   # Shape (T, 4).

        # Inlier counting.
        dist_roi = planes[valid] @ pc_roi.T
        inlier_counts = (dist_roi.abs() <= self.params['inlier_threshold']).sum(dim=1)
        
        # Select Best Plane.
        best_idx = inlier_counts.argmax()
        self.fitted_plane = planes[valid][best_idx]
        
        # Ensure normal points UP.
        if self.fitted_plane[2] < 0:
            self.fitted_plane *= -1.0

        # Final Classification for ALL points.
        dist_all = self.fitted_plane @ pc_hom.T
        bool_ground = dist_all <= self.params['ground_threshold']
        return bool_ground
    
    @torch.no_grad()
    def get_plane_params(
            self,
        ) -> Optional[torch.Tensor]:
        """
        Get the parameters of the last fitted plane.

        Returns:
            plane_params (torch.Tensor) : Plane parameters (a,b,c,d)
                for ax + by + cz + d = 0, or None if no plane fitted yet.
        """
        if self.fitted_plane is None:
            raise RuntimeError('Call fit() before accessing plane parameters!')
        plane_params = self.fitted_plane
        return plane_params
