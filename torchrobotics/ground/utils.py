import torch


def filter_by_radius_origin(
        pc: torch.Tensor,
        radius: float = 1.0,
        mode: str = 'keep_inner',
    ) -> torch.Tensor:
    """
    Keep points inside or outside a specified radius from the origin.

    Example usage:
        bool_valid = filter_by_radius_origin(pc, radius=1.0, mode='keep_inner')
        pc_valid = pc[bool_valid]
        where pc has shape (N,M) with M>=3, and bool_valid has shape (N,).
    
    Args:
        pc (torch.Tensor) : Point cloud with shape (N,M) with M>=3, 
            e.g. <x,y,z,...>. Unit: meters.
        radius (float) : Threshold distance. Unit: meters.
        mode (str) : 'keep_inner' to return a mask of points inside the radius, 
            'keep_outer' to return a mask of points outside the radius.
    
    Returns:
        bool_valid (torch.Tensor) : Boolean tensor indicating the valid points. 
            Unit: 1.
    """
    assert pc.shape[1] >= 3, 'Input point cloud must have shape (N,3+)!'
    assert radius > 0.0, 'Radius must be positive!'
    assert mode in ['keep_inner', 'keep_outer'], (
        "Mode must be 'keep_inner' or 'keep_outer'!"
    )

    dist2 = pc[:,0]**2 + pc[:,1]**2 + pc[:,2]**2
    
    if mode == 'keep_inner':
        bool_valid = dist2 <= (radius ** 2)
    else:   # Mode 'keep_outer'.
        bool_valid = dist2 > (radius ** 2)
    return bool_valid


def get_T_plane_reference(
        plane_parameters: torch.Tensor,
    ) -> torch.Tensor:
    """
    Get homogeneous transformation matrix T_plane_reference.

    Example usage:
        pc_plane = (T_plane_reference @ pc_reference.T).T
        where pc_plane has shape (N,4) in homogeneous coordinates,
        T_plane_reference has shape (4,4), and pc_reference has 
        shape (N,4) in homogeneous coordinates.
    
    Args:
        plane_parameters (torch.Tensor) : Plane parameters for plane equation 
            a*x + b*y + c*z + d = 0. Shape (4,). Unit: 1.
        
    Returns:
        T_plane_reference (torch.Tensor) : Homogeneous transformation matrix 
            mapping 3D points from reference frame to plane frame. 
            Shape (4,4).
    """
    assert plane_parameters.shape == (4,), (
         'Plane parameters must be a 1D tensor of length 4!'
    )
    
    device = plane_parameters.device
    dtype = plane_parameters.dtype

    # Extract and normalize plane normal vector.
    n = plane_parameters[:3]
    norm_n = torch.linalg.norm(n)
    assert norm_n > 1e-6, 'Plane normal vector cannot be zero!'
    
    # Normalize a, b, c, and d.
    z_vec = n / norm_n
    d = plane_parameters[3] / norm_n

    # Origin of plane frame (closest point on plane to reference origin).
    p_org = -d * z_vec

    # Find orthogonal X-axis safely.
    # Pick an arbitrary axis that is NOT parallel to z_vec to cross product with.
    # If z_vec is pointing mostly straight up, use the X-axis. 
    # Otherwise, use the Z-axis.
    if torch.abs(z_vec[2]) < 0.99:
        arbitrary_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    else:
        arbitrary_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        
    x_vec = torch.linalg.cross(arbitrary_axis, z_vec)
    x_vec = x_vec / torch.linalg.norm(x_vec)

    # Y-axis is naturally orthogonal to both.
    y_vec = torch.linalg.cross(z_vec, x_vec)

    # Assemble the T_reference_plane matrix.
    T_reference_plane = torch.eye(4, device=device, dtype=dtype)
    T_reference_plane[:3, 0] = x_vec
    T_reference_plane[:3, 1] = y_vec
    T_reference_plane[:3, 2] = z_vec
    T_reference_plane[:3, 3] = p_org

    # Invert to get T_plane_reference.
    T_plane_reference = torch.linalg.inv(T_reference_plane)
    return T_plane_reference
