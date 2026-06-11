from typing import Dict, Optional, Tuple

import torch

from .utils import expand_labels, voxel_downsample

PARAMS = {
    "eps": 1.0,  # Unit: meters.
    "min_samples": 10,  # Unit: 1.
    "tile_size": 4096,  # Unit: 1.
    "voxel": 0.10,  # Unit: meters. None or <= 0 disables voxel downsampling.
}


class DBSCAN:
    """
    DBSCAN for spatial clustering.
    """

    METHOD_NAME = "DBSCAN"

    def __init__(
        self,
        params: Optional[Dict] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            params (dict, optional): Hyperparameters. Keys: "eps" (meters),
                "min_samples" (1), "tile_size" (rows per distance tile, 1),
                "voxel" (meters; None or <= 0 disables downsampling).
            device (torch.device, optional): If None, use the input tensor's
                device at call time.
        """
        self.params = dict(PARAMS) if params is None else dict(params)
        assert self.params["eps"] > 0.0, "Key eps must be positive!"
        assert self.params["min_samples"] >= 1, "Key min_samples must be >= 1!"
        assert self.params["tile_size"] >= 1, "Key tile_size must be >= 1!"
        self.labels_: Optional[torch.Tensor] = None
        self.core_sample_indices_: Optional[torch.Tensor] = None
        self.device = device

    def _neighbor_counts(
        self,
        pc: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """
        Count eps-neighbors per point (including self), tiled, no edges stored.

        Args:
            pc (torch.Tensor): Shape (N,3) <x,y,z> on the compute device.
            eps (float): Neighbor radius. Unit: meters.

        Returns:
            counts (torch.Tensor): Shape (N,), int64. Neighbors per point. Unit: 1.
        """
        n = pc.shape[0]
        tile = self.params["tile_size"]
        counts = torch.empty(n, dtype=torch.long, device=pc.device)
        for a in range(0, n, tile):
            b = min(a + tile, n)
            d = torch.cdist(pc[a:b], pc)  # (B,N) euclidean.
            counts[a:b] = (d <= eps).sum(dim=1)
        return counts

    def _core_edges(
        self,
        pc: torch.Tensor,
        eps: float,
        is_core: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Materialize directed edges (row -> col) where col is a core point.

        Serves both connected components (restrict to core rows) and border
        assignment (all rows). Edges are symmetric across core points.

        Args:
            pc (torch.Tensor): Shape (N,3) <x,y,z>.
            eps (float): Neighbor radius. Unit: meters.
            is_core (torch.Tensor): Shape (N,), bool. Core-point mask.

        Returns:
            row (torch.Tensor): Shape (E,), int64. Source point per edge. Unit: 1.
            col (torch.Tensor): Shape (E,), int64. Core target per edge. Unit: 1.
        """
        n = pc.shape[0]
        tile = self.params["tile_size"]
        rows, cols = [], []
        for a in range(0, n, tile):
            b = min(a + tile, n)
            d = torch.cdist(pc[a:b], pc)
            mask = (d <= eps) & is_core[None, :]  # Keep only core columns.
            local = torch.nonzero(mask, as_tuple=False)  # (k,2): [local_row, col].
            rows.append(local[:, 0] + a)
            cols.append(local[:, 1])
        return torch.cat(rows), torch.cat(cols)

    @staticmethod
    def _connected_components(
        row: torch.Tensor,
        col: torch.Tensor,
        n: int,
        dev: torch.device,
    ) -> torch.Tensor:
        """
        Connected components by min-label propagation and pointer jumping.

        Args:
            row (torch.Tensor): Shape (E,), int64. Symmetric edge sources.
            col (torch.Tensor): Shape (E,), int64. Symmetric edge targets.
            n (int): Number of nodes. Unit: 1.
            dev (torch.device): Compute device.

        Returns:
            comp (torch.Tensor): Shape (N,), int64. Component root (the minimum
                node index reachable) per node. Unit: 1.
        """
        labels = torch.arange(n, dtype=torch.long, device=dev)
        while True:
            m = labels.clone()
            m.scatter_reduce_(0, row, labels[col], reduce="amin", include_self=True)
            m = m[m]  # Pointer jump toward the component minimum.
            if torch.equal(m, labels):
                return labels
            labels = m

    def _fit_points(
        self,
        pc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run DBSCAN on the given points (no downsampling).

        Args:
            pc (torch.Tensor): Shape (N,3) <x,y,z> on the compute device.

        Returns:
            labels (torch.Tensor): Shape (N,). Cluster id per point; -1 = noise. Unit: 1.
            core_idx (torch.Tensor): Shape (n_core,). Core indices, ascending. Unit: 1.
        """
        dev = pc.device
        eps = self.params["eps"]
        min_samples = self.params["min_samples"]
        n = pc.shape[0]

        # Core points (tiled counts, no edges materialized).
        counts = self._neighbor_counts(pc, eps)
        is_core = counts >= min_samples
        core_idx = torch.nonzero(is_core, as_tuple=False).flatten()

        # Edges with a core target (col is core).
        row, col = self._core_edges(pc, eps, is_core)

        # Connected components on the core<->core sub-graph.
        cc_keep = is_core[row]  # col already core => core<->core.
        comp = self._connected_components(row[cc_keep], col[cc_keep], n, dev)

        # Number clusters by the smallest core index they contain (sklearn order).
        rep = torch.full((n,), n, dtype=torch.long, device=dev)
        rep.scatter_reduce_(0, comp[core_idx], core_idx, reduce="amin", include_self=True)
        core_roots = torch.unique(comp[core_idx])
        order = core_roots[torch.argsort(rep[core_roots])]
        remap = torch.full((n,), -1, dtype=torch.long, device=dev)
        remap[order] = torch.arange(order.numel(), dtype=torch.long, device=dev)
        node_cluster = remap[comp]

        # Each point = min cluster id among its core neighbors (border rule).
        big = order.numel() if order.numel() > 0 else 1
        contrib = torch.where(is_core[col], node_cluster[col], torch.full_like(col, big))
        labels = torch.full((n,), big, dtype=torch.long, device=dev)
        labels.scatter_reduce_(0, row, contrib, reduce="amin", include_self=True)
        labels[labels == big] = -1  # No core neighbor -> noise.
        return labels, core_idx

    @torch.no_grad()
    def fit(
        self,
        pc: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Cluster a point cloud with DBSCAN and return the labels.

        If the "voxel" param is set, the cloud is downsampled to one point per
        voxel, clustered, and the labels are expanded back to all input points.

        Example usage:
            dbscan = DBSCAN()
            cluster_labels = dbscan.fit(pc=pc, generator=torch_generator)
            where pc has shape (N,3) <x,y,z>, cluster_labels has shape (N,),
            and noise points have label -1.

        Args:
            pc (torch.Tensor): Shape (N,3) <x,y,z>. Unit: meters.
            generator (torch.Generator, optional): Random number generator (RNG) for reproducibility (not used).

        Returns:
            labels (torch.Tensor): Shape (N,). Cluster id per point; noise
                points have label -1. Unit: 1.
        """
        assert pc.ndim == 2 and pc.shape[1] == 3, f"Point cloud must be (N, 3), got {pc.shape}"
        assert pc.shape[0] > 0, "Point cloud is empty!"

        dev = self.device or pc.device
        pc = pc.to(dev).contiguous()
        voxel = self.params.get("voxel", None)

        if voxel is not None and voxel > 0.0:
            rep_idx, inverse = voxel_downsample(pc, voxel)
            labels_v, core_v = self._fit_points(pc[rep_idx])
            labels = expand_labels(labels_v, inverse)
            core_idx = torch.sort(rep_idx[core_v]).values
        else:
            labels, core_idx = self._fit_points(pc)

        self.labels_ = labels
        self.core_sample_indices_ = core_idx
        return labels

    @torch.no_grad()
    def get_labels(
        self,
    ) -> torch.Tensor:
        """
        Get the labels from the last fit() call.

        Returns:
            labels (torch.Tensor): Shape (N,). Noise points have label -1. Unit: 1.
        """
        if self.labels_ is None:
            raise RuntimeError("Call fit() before accessing labels!")
        return self.labels_

    @torch.no_grad()
    def get_core_sample_indices(
        self,
    ) -> torch.Tensor:
        """
        Get the indices of the core samples from the last fit() call.

        When voxel downsampling is used, these are the original indices of the
        core voxel representatives.

        Returns:
            core_sample_indices (torch.Tensor): Shape (n_core,), ascending. Unit: 1.
        """
        if self.core_sample_indices_ is None:
            raise RuntimeError("Call fit() before accessing core sample indices!")
        return self.core_sample_indices_
