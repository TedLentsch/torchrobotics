from typing import Tuple

import torch


@torch.no_grad()
def voxel_downsample(
    pc: torch.Tensor,
    voxel: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pick one representative point per occupied voxel.

    Reduces local point redundancy before clustering. Run a clusterer on
    pc[rep_idx], then use expand_labels() with `inverse` to propagate the
    per-voxel labels back to every original point. The clustering estimators
    in this package call this internally when their "voxel" param is set.

    Example usage:
        rep_idx, inverse = voxel_downsample(pc=pc, voxel=0.10)
        labels_v = dbscan.fit(pc[rep_idx])
        labels_pc = expand_labels(labels_v, inverse)
        where pc has shape (N,3) <x,y,z>, rep_idx has shape (V,),
        inverse has shape (N,), and labels_pc has shape (N,).

    Args:
        pc (torch.Tensor): Shape (N,3) <x,y,z>. Points to downsample. Unit: meters.
        voxel (float): Voxel edge length. Larger = fewer points = faster but
            coarser. A good start is eps/4 to eps/2. Unit: meters.

    Returns:
        rep_idx (torch.Tensor): Shape (V,). Index into pc of the representative
            (lowest original index) of each occupied voxel. Unit: 1.
        inverse (torch.Tensor): Shape (N,). For each original point, the voxel
            id (0..V-1) it belongs to. Unit: 1.
    """
    assert pc.ndim == 2 and pc.shape[1] == 3, f"Point cloud must be (N, 3), got {pc.shape}"
    assert voxel > 0.0, f"voxel must be positive, got {voxel}"

    dev = pc.device
    n = pc.shape[0]

    # Integer voxel coordinates (floor handles negatives consistently).
    keys = torch.floor(pc / voxel).to(torch.int64)
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    n_vox = int(inverse.max().item()) + 1 if n > 0 else 0

    # Representative = smallest original index in each voxel.
    rep_idx = torch.full((n_vox,), n, dtype=torch.long, device=dev)
    order = torch.arange(n, dtype=torch.long, device=dev)
    rep_idx.scatter_reduce_(0, inverse, order, reduce="amin", include_self=True)
    return rep_idx, inverse


@torch.no_grad()
def expand_labels(
    labels_v: torch.Tensor,
    inverse: torch.Tensor,
) -> torch.Tensor:
    """
    Propagate per-voxel labels back to every original point.

    Example usage:
        rep_idx, inverse = voxel_downsample(pc=pc, voxel=0.10)
        labels_v = dbscan.fit(pc[rep_idx])
        labels_pc = expand_labels(labels_v, inverse)

    Args:
        labels_v (torch.Tensor): Shape (V,). Label per voxel representative,
            as returned by clustering pc[rep_idx]. Unit: 1.
        inverse (torch.Tensor): Shape (N,). Voxel id per original point, from
            voxel_downsample(). Unit: 1.

    Returns:
        labels (torch.Tensor): Shape (N,). Label for every original point;
            noise points keep label -1. Unit: 1.
    """
    assert labels_v.ndim == 1, f"Variable labels_v must be (V,), got {tuple(labels_v.shape)}"
    assert inverse.ndim == 1, f"Variable inverse must be (N,), got {tuple(inverse.shape)}"
    assert int(inverse.max().item()) < labels_v.shape[0], (
        "Variable inverse references missing voxels!"
    )
    return labels_v[inverse.to(labels_v.device)]
