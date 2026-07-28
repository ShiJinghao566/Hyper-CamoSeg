import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2tuple(value, name):
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"{name} must be an int or a pair of ints, got {value!r}")


class TargetBackgroundHypergraphModule(nn.Module):
    def __init__(
        self,
        vis_dim=1536,
        text_dim=768,
        hidden_dim=768,
        anchor_pool_kernel=2,
        anchor_pool_stride=None,
        anchor_feature_hw=(12, 12),
        topk=32,
        visual_topk=None,
        hypergraph_k=18,
        e2v_ratio=2.0,
        laplacian_alpha=0.1,
        use_structure_induction=True,
        hyperedge_dropout=False,
        e_drop_rate=0.2,
        num_heads=8,
        sim_temperature=0.07,
        fg_temperature=0.1,
        propagation_eta=0.5,
        eps=1e-6,
    ):
        super().__init__()
        self.anchor_pool_kernel = _to_2tuple(anchor_pool_kernel, "anchor_pool_kernel")
        self.anchor_pool_stride = (
            self.anchor_pool_kernel
            if anchor_pool_stride is None
            else _to_2tuple(anchor_pool_stride, "anchor_pool_stride")
        )
        _to_2tuple(anchor_feature_hw, "anchor_feature_hw")
        self.visual_topk = topk if visual_topk is None else visual_topk
        self.hypergraph_k = hypergraph_k
        self.e2v_ratio = e2v_ratio
        self.laplacian_alpha = laplacian_alpha
        self.use_structure_induction = use_structure_induction
        self.hyperedge_dropout = hyperedge_dropout
        self.e_drop_rate = float(e_drop_rate)
        self.sim_temperature = sim_temperature
        self.fg_temperature = fg_temperature
        self.propagation_eta = propagation_eta
        self.eps = eps

        self.vis_proj = nn.Linear(vis_dim, hidden_dim)
        self.node_to_edge = nn.Linear(hidden_dim, hidden_dim)
        self.node_query = nn.Linear(hidden_dim, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, hidden_dim)
        self.affinity_proj = nn.Linear(hidden_dim, hidden_dim)

        self.obj_proj = nn.Linear(text_dim, hidden_dim)
        self.bg_proj = nn.Linear(text_dim, hidden_dim)
        self.anchor_text_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.anchor_out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.high_pass = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.low_node_proj = nn.Linear(hidden_dim, hidden_dim)
        self.low_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.obj_text_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.bg_text_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.text_fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim),
        )
        self.text_out_norm = nn.LayerNorm(text_dim)

        self.vis_out_norm = nn.LayerNorm(hidden_dim)
        self.vis_out = nn.Linear(hidden_dim, vis_dim)

    def _build_spatial_anchors(self, node_map, obj_text):
        pooled = F.avg_pool2d(
            node_map,
            kernel_size=self.anchor_pool_kernel,
            stride=self.anchor_pool_stride,
        )
        pooled_anchors = pooled.flatten(2).transpose(1, 2)
        if obj_text.dim() == 2:
            text_prototype = obj_text.max(dim=0).values.unsqueeze(0)
        else:
            text_prototype = obj_text.max(dim=1).values
        text_prototype = self.anchor_text_mlp(text_prototype)
        return self.anchor_out_proj(pooled_anchors + text_prototype.unsqueeze(1))

    def _build_incidence(self, nodes, anchors):
        sim = torch.einsum(
            "bpd,bmd->bpm",
            F.normalize(nodes, dim=-1),
            F.normalize(anchors, dim=-1),
        )
        return F.softmax(sim / self.sim_temperature, dim=1)

    def _node_anchor_similarity(self, nodes, anchors):
        return torch.einsum(
            "bpd,bmd->bpm",
            F.normalize(nodes, dim=-1),
            F.normalize(anchors, dim=-1),
        ) / self.sim_temperature

    @staticmethod
    def _make_keep_mask(shape, drop_rate, device):
        if drop_rate <= 0.0:
            return torch.ones(shape, device=device, dtype=torch.bool)

        mask = torch.rand(shape, device=device) > drop_rate
        all_dropped = ~mask.any(dim=-1)
        if all_dropped.any():
            K = shape[-1]
            rand_keep = torch.randint(0, K, all_dropped.shape, device=device)
            mask2 = mask.reshape(-1, K)
            all2 = all_dropped.reshape(-1)
            rand2 = rand_keep.reshape(-1)
            rows = torch.nonzero(all2, as_tuple=False).squeeze(1)
            if rows.numel() > 0:
                mask2[rows, rand2[rows]] = True
        return mask

    def _structure_induction(self, nodes):
        if not self.use_structure_induction or self.hypergraph_k <= 1:
            return nodes

        B, P, D = nodes.shape
        if P == 1:
            return nodes

        k = min(self.hypergraph_k, P - 1)
        sim = torch.bmm(F.normalize(nodes, dim=-1), F.normalize(nodes, dim=-1).transpose(1, 2))
        eye = torch.eye(P, device=nodes.device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(eye, -torch.finfo(sim.dtype).max)
        _, indices = sim.topk(k, dim=-1, sorted=False)
        expanded_nodes = nodes.unsqueeze(1).expand(B, P, P, D)
        neighbors = torch.gather(
            expanded_nodes,
            dim=2,
            index=indices.unsqueeze(-1).expand(B, P, k, D),
        )
        return (1.0 - self.laplacian_alpha) * nodes + self.laplacian_alpha * neighbors.mean(dim=2)

    def _vic_e2v_mapping(self, structure_nodes, anchors):
        sim = self._node_anchor_similarity(structure_nodes, anchors)
        k = min(self.hypergraph_k, structure_nodes.shape[1])
        topk_values, topk_indices = torch.topk(sim.transpose(1, 2), k=k, dim=2, sorted=False)
        return topk_indices, topk_values

    def _vic_v2e_aggregate(self, nodes, topk_indices, topk_values):
        B, _, D = nodes.shape
        M, k = topk_indices.shape[1], topk_indices.shape[2]

        if self.hyperedge_dropout and self.training and (0.0 < self.e_drop_rate < 1.0):
            mask = self._make_keep_mask((B, M, k), self.e_drop_rate, topk_values.device)
            topk_values = topk_values.masked_fill(~mask, float("-inf"))
        topk_weights = F.softmax(topk_values, dim=2)

        Mk = topk_indices.numel() // B
        idx = topk_indices.reshape(B, Mk).unsqueeze(-1).expand(B, Mk, D)
        selected_nodes = nodes.gather(dim=1, index=idx).reshape(B, M, k, D)
        return torch.sum(selected_nodes * topk_weights.unsqueeze(-1), dim=2)

    def _vic_v2e_mapping(self, structure_nodes, hyperedge_features):
        sim = self._node_anchor_similarity(structure_nodes, hyperedge_features)
        k_v2e = max(1, int(self.hypergraph_k / self.e2v_ratio))
        k_v2e = min(k_v2e, hyperedge_features.shape[1])
        _, v2e_indices = torch.topk(sim, k=k_v2e, dim=2, sorted=False)
        return v2e_indices

    def _vic_v2e_diff(self, nodes, hyperedge_features, v2e_indices):
        B, P, D = nodes.shape
        k_v2e = v2e_indices.shape[2]
        Nk = v2e_indices.numel() // B
        idx = v2e_indices.reshape(B, Nk).unsqueeze(-1).expand(B, Nk, D)
        connected_edges = hyperedge_features.gather(dim=1, index=idx).reshape(B, P, k_v2e, D)
        rel = connected_edges - nodes.unsqueeze(2)

        if self.hyperedge_dropout and self.training and (0.0 < self.e_drop_rate < 1.0):
            mask = self._make_keep_mask((B, P, k_v2e), self.e_drop_rate, rel.device)
            rel = rel.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        return rel.max(dim=2).values

    def _visual_reference_delta(self, nodes):
        B, P, D = nodes.shape
        if P == 1:
            return nodes.new_zeros(B, P, D)

        topk = min(self.visual_topk, P - 1)
        query_nodes = F.normalize(self.node_query(nodes), dim=-1)
        sim = torch.bmm(query_nodes, query_nodes.transpose(1, 2))
        eye = torch.eye(P, device=nodes.device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(eye, -torch.finfo(sim.dtype).max)

        _, indices = sim.topk(topk, dim=-1)
        expanded_nodes = nodes.unsqueeze(1).expand(B, P, P, D)
        neighbors = torch.gather(
            expanded_nodes,
            dim=2,
            index=indices.unsqueeze(-1).expand(B, P, topk, D),
        )
        return (neighbors - nodes.unsqueeze(2)).max(dim=2).values

    def _edge_reference_delta(self, nodes, edges, incidence):
        B, P, D = nodes.shape
        diff = edges.unsqueeze(1) - nodes.unsqueeze(2)
        connected = incidence.gt(0).unsqueeze(-1)
        diff = diff.masked_fill(~connected, -torch.finfo(diff.dtype).max)
        delta = diff.max(dim=2).values
        return torch.where(torch.isfinite(delta), delta, nodes.new_zeros(B, P, D))

    def _cut_incidence(self, nodes, obj_text, bg_text, incidence):
        B, P, _ = nodes.shape
        degree_e = incidence.sum(dim=1).clamp_min(self.eps)
        degree_v = incidence.sum(dim=2).clamp_min(self.eps)

        score_nodes = F.normalize(self.score_proj(nodes), dim=-1)
        if obj_text.dim() == 2:
            obj_global = F.normalize(obj_text.mean(dim=1), dim=-1)
            bg_global = F.normalize(bg_text.mean(dim=1), dim=-1)
            obj_score = torch.einsum("bpd,d->bp", score_nodes, obj_global)
            bg_score = torch.einsum("bpd,d->bp", score_nodes, bg_global)
        else:
            obj_global = F.normalize(obj_text.mean(dim=1), dim=-1)
            bg_global = F.normalize(bg_text.mean(dim=1), dim=-1)
            obj_score = torch.einsum("bpd,bd->bp", score_nodes, obj_global)
            bg_score = torch.einsum("bpd,bd->bp", score_nodes, bg_global)
        fg_prob_0 = torch.sigmoid((obj_score - bg_score) / self.fg_temperature)

        incidence_de = incidence / degree_e.unsqueeze(1)
        theta = torch.bmm(incidence_de, incidence.transpose(1, 2))
        dv_sqrt_inv = degree_v.rsqrt()
        theta = theta * dv_sqrt_inv.unsqueeze(2) * dv_sqrt_inv.unsqueeze(1)

        affinity_nodes = F.normalize(self.affinity_proj(nodes), dim=-1)
        visual_affinity = (torch.bmm(affinity_nodes, affinity_nodes.transpose(1, 2)) + 1) * 0.5
        affinity = theta * visual_affinity
        eye = torch.eye(P, device=nodes.device, dtype=nodes.dtype).unsqueeze(0)
        affinity = affinity + eye
        propagation = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        fg_prob = (1 - self.propagation_eta) * fg_prob_0 + self.propagation_eta * torch.bmm(
            propagation, fg_prob_0.unsqueeze(-1)
        ).squeeze(-1)

        fg_mask = (fg_prob >= 0.5).to(dtype=incidence.dtype)
        fg_incidence = incidence * fg_mask.unsqueeze(-1)
        bg_incidence = incidence * (1 - fg_mask).unsqueeze(-1)
        return torch.cat([fg_incidence, bg_incidence], dim=2)

    def forward(self, image_feat, obj_text, bg_text, return_intermediates=False):
        B, C, H, W = image_feat.shape
        nodes = image_feat.flatten(2).transpose(1, 2)
        nodes = self.vis_proj(nodes)
        node_map = nodes.transpose(1, 2).reshape(B, -1, H, W)

        obj_text = self.obj_proj(obj_text)
        bg_text = self.bg_proj(bg_text)
        anchors = self._build_spatial_anchors(node_map, obj_text)

        incidence = self._build_incidence(nodes, anchors)
        structure_nodes = self._structure_induction(nodes)
        topk_indices, topk_values = self._vic_e2v_mapping(structure_nodes, anchors)
        edges = self._vic_v2e_aggregate(nodes, topk_indices, topk_values)
        v2e_indices = self._vic_v2e_mapping(structure_nodes, edges)
        delta = self._vic_v2e_diff(nodes, edges, v2e_indices)
        nodes = nodes + self.high_pass(torch.cat([nodes, delta], dim=-1))
        high_pass_nodes = nodes

        cut_incidence = self._cut_incidence(nodes, obj_text, bg_text, incidence)
        cut_degree_e = cut_incidence.sum(dim=1).clamp_min(self.eps)
        cut_degree_v = cut_incidence.sum(dim=2).clamp_min(self.eps)

        low_nodes = self.low_node_proj(nodes)
        cut_degree_v_inv_sqrt = cut_degree_v.rsqrt()
        low_nodes = low_nodes * cut_degree_v_inv_sqrt.unsqueeze(-1)
        cut_edges = torch.bmm(cut_incidence.transpose(1, 2), low_nodes) / cut_degree_e.unsqueeze(-1)
        low_message = torch.bmm(cut_incidence, cut_edges) * cut_degree_v_inv_sqrt.unsqueeze(-1)
        nodes = nodes + self.low_out(low_message)
        low_pass_nodes = nodes

        edge_context = torch.bmm(cut_incidence.transpose(1, 2), nodes) / cut_degree_e.unsqueeze(-1)
        obj_query = obj_text if obj_text.dim() == 3 else obj_text.unsqueeze(0).expand(B, -1, -1)
        bg_query = bg_text if bg_text.dim() == 3 else bg_text.unsqueeze(0).expand(B, -1, -1)

        obj_updated = obj_query + self.obj_text_attn(
            query=obj_query,
            key=edge_context,
            value=edge_context,
            need_weights=False,
        )[0]
        bg_updated = bg_query + self.bg_text_attn(
            query=bg_query,
            key=edge_context,
            value=edge_context,
            need_weights=False,
        )[0]
        fused_text = self.text_out_norm(self.text_fuse(obj_updated - bg_updated))

        vis_update = self.vis_out(self.vis_out_norm(nodes))
        vis_update = vis_update.transpose(1, 2).reshape(B, C, H, W)
        output_feat = image_feat + vis_update
        if return_intermediates:
            return output_feat, fused_text, {
                "high_pass": high_pass_nodes.transpose(1, 2).reshape(B, -1, H, W),
                "low_pass": low_pass_nodes.transpose(1, 2).reshape(B, -1, H, W),
                "cut_incidence": cut_incidence,
            }
        return output_feat, fused_text
