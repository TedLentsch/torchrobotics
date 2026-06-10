from typing import Dict, List, Optional, Tuple

import torch

from .utils import expand_labels, voxel_downsample

PARAMS = {
    "min_cluster_size": 10,  # Unit: 1.
    "min_samples": None,  # Unit: 1. None -> equals min_cluster_size.
    "graph_neighbors": 32,  # Unit: 1. k for the approximate k-NN-graph MST.
    "tile_size": 4096,  # Unit: 1.
    "voxel": 0.10,  # Unit: meters. None or <= 0 disables voxel downsampling.
}


class HDBSCAN:
    """
    HDBSCAN for spatial clustering.
    """

    METHOD_NAME = "HDBSCAN"

    def __init__(
        self,
        params: Optional[Dict] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            params (dict, optional): Hyperparameters. Keys: "min_cluster_size"
                (1), "min_samples" (1, or None to mirror min_cluster_size),
                "graph_neighbors" (k for the k-NN-graph MST, 1), "tile_size"
                (rows per distance tile, 1), "voxel" (meters; None or <= 0
                disables downsampling).
            device (torch.device, optional): If None, use the input tensor's
                device at call time.
        """
        self.params = dict(PARAMS) if params is None else dict(params)
        assert self.params["min_cluster_size"] >= 2, "min_cluster_size must be >= 2!"
        assert self.params["graph_neighbors"] >= 1, "graph_neighbors must be >= 1!"
        assert self.params["tile_size"] >= 1, "tile_size must be >= 1!"
        self.labels_: Optional[torch.Tensor] = None
        self.device = device

    def _knn_graph(
        self,
        pc: torch.Tensor,
        k: int,
        ms: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the k-nearest-neighbour mutual-reachability graph (tiled).

        Computes, in one tiled pass, the core distance per point and the
        mutual-reachability weight of each k-NN edge. Edges are deduplicated to
        undirected (i < j). The full N x N distance matrix is never stored;
        memory is O(tile_size x N).

        Args:
            pc (torch.Tensor): Shape (N,3) <x,y,z> on the compute device.
            k (int): Neighbours per point in the graph. Unit: 1.
            ms (int): min_samples; core distance is the ms-th nearest. Unit: 1.

        Returns:
            core (torch.Tensor): Shape (N,). Core distance per point. Unit: meters.
            ew (torch.Tensor): Shape (E,). Mutual-reachability weight per edge. Unit: meters.
            ea (torch.Tensor): Shape (E,), int64. Edge endpoint i (i < j). Unit: 1.
            eb (torch.Tensor): Shape (E,), int64. Edge endpoint j (i < j). Unit: 1.
        """
        n = pc.shape[0]
        dev = pc.device
        tile = self.params["tile_size"]
        k_eff = min(k, n - 1)  # at most N-1 real neighbours
        kk = min(max(k_eff + 1, ms), n)  # include self (col 0); enough for core

        core = torch.empty(n, dtype=pc.dtype, device=dev)
        rows, cols, dists = [], [], []
        for a in range(0, n, tile):
            b = min(a + tile, n)
            d = torch.cdist(pc[a:b], pc)  # (B,N) euclidean
            vals, idx = torch.topk(d, kk, dim=1, largest=False, sorted=True)
            core[a:b] = vals[:, ms - 1]  # ms-th nearest (self is col 0)
            nbr = idx[:, 1 : k_eff + 1]  # k nearest, excluding self
            nbr_d = vals[:, 1 : k_eff + 1]
            r = torch.arange(a, b, device=dev)[:, None].expand(-1, k_eff).reshape(-1)
            rows.append(r)
            cols.append(nbr.reshape(-1))
            dists.append(nbr_d.reshape(-1))

        row = torch.cat(rows)
        col = torch.cat(cols)
        dist = torch.cat(dists)

        # Mutual reachability: max(core_i, core_j, d_ij). Symmetric in (i, j).
        mr = torch.maximum(torch.maximum(core[row], core[col]), dist)

        # Deduplicate to undirected edges (keep one of each (min, max) pair).
        lo = torch.minimum(row, col)
        hi = torch.maximum(row, col)
        key = lo.to(torch.int64) * n + hi.to(torch.int64)
        key_sorted, order = torch.sort(key)
        keep = torch.ones_like(key_sorted, dtype=torch.bool)
        keep[1:] = key_sorted[1:] != key_sorted[:-1]
        sel = order[keep]
        return core, mr[sel], lo[sel], hi[sel]

    def _kruskal_mst(
        self,
        core: torch.Tensor,
        ew: torch.Tensor,
        ea: torch.Tensor,
        eb: torch.Tensor,
        pc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Minimum spanning tree over the k-NN graph (Kruskal), bridged to a tree.

        Sorts the k-NN edges once and runs union-find. If the k-NN graph is
        disconnected (sparse far-range returns), the resulting forest is bridged
        into a single spanning tree by a small MST over one representative point
        per component, using real mutual-reachability weights so the pieces join
        high in the hierarchy (and separate back out during condensation).

        Args:
            core (torch.Tensor): Shape (N,). Core distance per point. Unit: meters.
            ew (torch.Tensor): Shape (E,). k-NN edge weights. Unit: meters.
            ea, eb (torch.Tensor): Shape (E,), int64. k-NN edge endpoints. Unit: 1.
            pc (torch.Tensor): Shape (N,3) <x,y,z>. Used for bridging. Unit: meters.

        Returns:
            weight (torch.Tensor): Shape (N-1,). MST edge weights, ascending. Unit: meters.
            ma (torch.Tensor): Shape (N-1,), int64. MST endpoint per edge. Unit: 1.
            mb (torch.Tensor): Shape (N-1,), int64. MST endpoint per edge. Unit: 1.
        """
        n = pc.shape[0]
        dev = pc.device

        order = torch.argsort(ew)
        wl = ew[order].tolist()
        al = ea[order].tolist()
        bl = eb[order].tolist()

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        mw: List[float] = []
        ma: List[int] = []
        mb: List[int] = []
        for w, a, b in zip(wl, al, bl):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
                mw.append(w)
                ma.append(a)
                mb.append(b)
                if len(mw) == n - 1:
                    break

        # Bridge a disconnected k-NN graph into a single spanning tree.
        if len(mw) < n - 1:
            seen = {}
            reps: List[int] = []
            for i in range(n):
                r = find(i)
                if r not in seen:
                    seen[r] = i
                    reps.append(i)
            reps_t = torch.tensor(reps, dtype=torch.long, device=dev)
            dr = torch.cdist(pc[reps_t], pc[reps_t])
            cr = core[reps_t]
            mrr = torch.maximum(torch.maximum(cr[:, None], cr[None, :]), dr).tolist()
            m = len(reps)
            intree = [False] * m
            best = [float("inf")] * m
            par = [-1] * m
            best[0] = 0.0
            for _ in range(m):
                u = min((v for v in range(m) if not intree[v]), key=lambda v: best[v])
                intree[u] = True
                if par[u] >= 0:
                    mw.append(best[u])
                    ma.append(reps[par[u]])
                    mb.append(reps[u])
                for v in range(m):
                    if not intree[v] and mrr[u][v] < best[v]:
                        best[v] = mrr[u][v]
                        par[v] = u

        weight = torch.tensor(mw, dtype=pc.dtype, device=dev)
        a_t = torch.tensor(ma, dtype=torch.long, device=dev)
        b_t = torch.tensor(mb, dtype=torch.long, device=dev)
        o = torch.argsort(weight)
        return weight[o], a_t[o], b_t[o]

    def _single_linkage(
        self,
        weight: torch.Tensor,
        ea: torch.Tensor,
        eb: torch.Tensor,
        n: int,
    ) -> List[Tuple[int, int, float, int]]:
        """
        Build a single-linkage hierarchy from sorted MST edges (union-find).

        Args:
            weight (torch.Tensor): Shape (N-1,). Edge weights, ascending.
            ea, eb (torch.Tensor): Shape (N-1,). Edge endpoints.
            n (int): Number of points. Unit: 1.

        Returns:
            hierarchy (list): N-1 merges as (left, right, distance, size).
        """
        parent = list(range(2 * n - 1))
        size = [1] * (2 * n - 1)
        rep = list(range(2 * n - 1))  # component root -> current node id

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        hierarchy: List[Tuple[int, int, float, int]] = []
        nxt = n
        for w, a, b in zip(weight.tolist(), ea.tolist(), eb.tolist()):
            ra, rb = find(int(a)), find(int(b))
            if ra == rb:
                continue
            ca, cb = rep[ra], rep[rb]
            hierarchy.append((ca, cb, float(w), size[ca] + size[cb]))
            parent[ra] = nxt
            parent[rb] = nxt
            parent[nxt] = nxt
            size[nxt] = size[ca] + size[cb]
            rep[find(nxt)] = nxt
            nxt += 1
        return hierarchy

    def _bfs(
        self,
        hierarchy: List,
        root: int,
        n: int,
    ) -> List[int]:
        """
        Return all nodes in the subtree of `root` (root first).

        Args:
            hierarchy (list): N-1 merges as (left, right, distance, size).
            root (int): Node id to start from. Unit: 1.
            n (int): Number of points. Unit: 1.

        Returns:
            out (list): Node ids in the subtree of root, starting with root. Unit: 1.
        """
        out, stack = [], [root]
        while stack:
            node = stack.pop()
            out.append(node)
            if node >= n:
                left, right, _, _ = hierarchy[node - n]
                stack.append(int(left))
                stack.append(int(right))
        return out

    def _condense(
        self,
        hierarchy: List,
        n: int,
        mcs: int,
    ) -> List[Tuple[int, int, float, int]]:
        """
        Condense the single-linkage tree using min_cluster_size.

        Returns:
            condensed (list): rows (parent, child, lambda, child_size), where
                child < n is a point and child >= n is a sub-cluster.
        """
        if not hierarchy:
            return []
        root = 2 * len(hierarchy)
        nxt = n + 1
        relabel = {root: n}
        ignore = set()
        out: List[Tuple[int, int, float, int]] = []

        def count(node: int) -> int:
            return 1 if node < n else hierarchy[node - n][3]

        for node in self._bfs(hierarchy, root, n):
            if node in ignore or node < n:
                continue
            left, right, dist, _ = hierarchy[node - n]
            left, right = int(left), int(right)
            lam = 1.0 / dist if dist > 0 else float("inf")
            lc, rc = count(left), count(right)
            if lc >= mcs and rc >= mcs:
                relabel[left] = nxt
                out.append((relabel[node], nxt, lam, lc))
                nxt += 1
                relabel[right] = nxt
                out.append((relabel[node], nxt, lam, rc))
                nxt += 1
            elif lc < mcs and rc < mcs:
                for sub in self._bfs(hierarchy, node, n):
                    if sub != node and sub < n:
                        out.append((relabel[node], sub, lam, 1))
                    ignore.add(sub)
            else:
                keep, drop = (left, right) if lc >= mcs else (right, left)
                relabel[keep] = relabel[node]
                for sub in self._bfs(hierarchy, drop, n):
                    if sub < n:
                        out.append((relabel[node], sub, lam, 1))
                    ignore.add(sub)
        return out

    def _extract_eom(
        self,
        condensed: List,
        n: int,
    ) -> dict:
        """
        Excess-of-Mass cluster selection.

        Args:
            condensed (list): rows (parent, child, lambda, child_size) from condense().
            n (int): Number of points. Unit: 1.

        Returns:
            selected (dict): cluster_id -> True if selected, else False. Unit: 1.
        """
        if not condensed:
            return {}
        births = {}
        children_of: dict = {}
        stab: dict = {}
        for parent, child, lam, size in condensed:
            children_of.setdefault(parent, [])
            if child >= n:
                births[child] = lam
                children_of[parent].append(child)
        root = min(p for p, _, _, _ in condensed)
        births[root] = 0.0
        for parent, child, lam, size in condensed:
            stab[parent] = stab.get(parent, 0.0) + (lam - births.get(parent, 0.0)) * size

        selected = {c: True for c in stab}
        work = dict(stab)
        for c in sorted(stab.keys(), reverse=True):  # leaves -> root
            kids = children_of.get(c, [])
            kid_sum = sum(work[k] for k in kids if k in work)
            if kids and work[c] < kid_sum:
                selected[c] = False
                work[c] = kid_sum
            else:
                stack = list(kids)
                while stack:
                    d = stack.pop()
                    selected[d] = False
                    stack.extend(children_of.get(d, []))
        selected[root] = False  # allow_single_cluster=False
        return selected

    def _labelling(
        self,
        condensed: List,
        selected: dict,
        n: int,
    ) -> List[int]:
        """
        Assign each point the selected cluster it belongs to, else -1.

        Args:
            condensed (list): rows (parent, child, lambda, child_size) from condense().
            selected (dict): cluster_id -> True if selected, else False from _extract_eom().
            n (int): Number of points. Unit: 1.

        Returns:
            labels (list): index -> cluster_id if selected, else -1. Unit: 1.
        """
        labels = [-1] * n
        if not condensed:
            return labels
        parent_of = {}
        for parent, child, lam, size in condensed:
            if child >= n:
                parent_of[child] = parent
        sel = sorted([c for c, keep in selected.items() if keep])
        lab_of = {c: i for i, c in enumerate(sel)}
        sel_set = set(sel)
        for parent, child, lam, size in condensed:
            if child < n:
                c = parent
                while c not in sel_set and c in parent_of:
                    c = parent_of[c]
                if c in sel_set:
                    labels[child] = lab_of[c]
        return labels

    def _fit_points(
        self,
        pc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run approximate HDBSCAN* on the given points (no downsampling).

        Args:
            pc (torch.Tensor): Shape (N,3) <x,y,z> on the compute device.

        Returns:
            labels (torch.Tensor): Shape (N,). Cluster id per point; -1 = noise. Unit: 1.
        """
        dev = pc.device
        mcs = self.params["min_cluster_size"]
        ms = self.params["min_samples"] or mcs
        k = self.params["graph_neighbors"]
        n = pc.shape[0]
        assert n >= mcs, "Fewer points than min_cluster_size!"

        # Core distances + k-NN mutual-reachability graph (one tiled pass).
        core, ew, ea, eb = self._knn_graph(pc, k, ms)

        # MST over the k-NN graph (Kruskal), bridged into a single tree.
        weight, ma, mb = self._kruskal_mst(core, ew, ea, eb, pc)

        # Single-linkage hierarchy -> condense -> Excess of Mass -> labels.
        hierarchy = self._single_linkage(weight, ma, mb, n)
        condensed = self._condense(hierarchy, n, mcs)
        selected = self._extract_eom(condensed, n)
        labels = self._labelling(condensed, selected, n)
        return torch.tensor(labels, dtype=torch.long, device=dev)

    @torch.no_grad()
    def fit(
        self,
        pc: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Cluster a point cloud with HDBSCAN* and return the labels.

        If the "voxel" param is set, the cloud is downsampled to one point per
        voxel, clustered, and the labels are expanded back to all input points.

        Example usage:
            hdbscan = HDBSCAN()
            cluster_labels = hdbscan.fit(pc=pc, generator=torch_generator)
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
            labels_v = self._fit_points(pc[rep_idx])
            labels = expand_labels(labels_v, inverse)
        else:
            labels = self._fit_points(pc)

        self.labels_ = labels
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
