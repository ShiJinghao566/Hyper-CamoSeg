import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalCrossAttention(nn.Module):
    """
    双向跨模态注意力：
    输入：
        image_embedding_res2/3/4/5: 4个尺度视觉特征, 形状分别为
            (B, 192, 96,96), (B,384,48,48), (B,768,24,24), (B,1536,12,12)
        normed_class_embs: 文本特征, (Nc, 768)
    输出：
        enhanced_vis: list 与原视觉特征同形状
        enhanced_txt: (Nc, 768)
    """
    def __init__(self, vis_dims=[192,384,768,1536], txt_dim=768, hidden_dim=768, num_heads=8):
        super().__init__()
        self.vis_dims = vis_dims
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        # 视觉投影
        self.vis_proj_q = nn.ModuleList([nn.Linear(d, hidden_dim) for d in vis_dims])
        self.vis_proj_kv = nn.ModuleList([nn.Linear(d, hidden_dim*2) for d in vis_dims])

        # 文本投影
        self.txt_proj_q = nn.Linear(txt_dim, hidden_dim)
        self.txt_proj_kv = nn.Linear(txt_dim, hidden_dim*2)

        # 输出映射
        self.vis_out = nn.ModuleList([nn.Linear(hidden_dim, d) for d in vis_dims])
        self.txt_out = nn.Linear(hidden_dim, txt_dim)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, vis_features, txt_features):
        # vis_features: list of (B,C,H,W)
        # txt_features: (Nc,768)
        B = vis_features[0].shape[0]
        Nc = txt_features.shape[0]

        # ---------------- 1. prepare visual tokens ----------------
        vis_tokens, shapes = [], []                         # 每个元素 (B, L, C)
        q_vis, k_vis, v_vis = [], [], []
        for feat, proj_q, proj_kv in zip(vis_features, self.vis_proj_q, self.vis_proj_kv):
            B, C, H, W = feat.shape
            shapes.append((B, C, H, W))
            feat = feat.flatten(2).permute(0, 2, 1)        # (B, H*W, C)
            vis_tokens.append(feat)

            q_vis.append(proj_q(feat))                     # (B, L, hidden)
            kv = proj_kv(feat).chunk(2, dim=-1)
            k_vis.append(kv[0])
            v_vis.append(kv[1])

        # 拼接所有尺度
        q_vis_all = torch.cat(q_vis, dim=1)                # (B, L_total, hidden)
        k_vis_all = torch.cat(k_vis, dim=1)
        v_vis_all = torch.cat(v_vis, dim=1)
        L_total = q_vis_all.shape[1]

        # ---------------- 2. prepare text tokens ----------------
        q_txt = self.txt_proj_q(txt_features).unsqueeze(0).expand(B, -1, -1)  # (B, Nc, hidden)
        kv_txt = self.txt_proj_kv(txt_features).unsqueeze(0).expand(B, -1, -1)
        k_txt, v_txt = kv_txt.chunk(2, dim=-1)             # (B, Nc, hidden)

        # ---------------- 3. reshape for multi-head ----------------
        def reshape(x):     # (B, L, hidden) -> (B, num_heads, L, head_dim)
            B, L, D = x.shape
            return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Visual->Text attention
        qv, kv, vv = reshape(q_vis_all), reshape(k_txt), reshape(v_txt)
        scores = torch.matmul(qv, kv.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)                   # (B,heads,L_total,Nc)
        vis_enh = torch.matmul(attn, vv)                   # (B,heads,L_total,head)
        vis_enh = vis_enh.transpose(1, 2).contiguous().view(B, L_total, self.hidden_dim)
        vis_enh = torch.split(vis_enh, [s[2]*s[3] for s in shapes], dim=1)
        enhanced_vis = []
        for enh, (B, C, H, W) in zip(vis_enh, shapes):
            enh = self.vis_out[len(enhanced_vis)](enh)     # (B, L, C_orig)
            enh = enh.transpose(1, 2).view(B, C, H, W)
            enhanced_vis.append(enh)

        # Text->Visual attention
        qt, kt, vt = reshape(q_txt), reshape(k_vis_all), reshape(v_vis_all)
        scores = torch.matmul(qt, kt.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)                   # (B,heads,Nc,L_total)
        txt_enh = torch.matmul(attn, vt)                   # (B,heads,Nc,head)
        txt_enh = txt_enh.transpose(1, 2).contiguous().view(B, Nc, self.hidden_dim)
        enhanced_txt = self.txt_out(txt_enh.mean(0))       # (Nc,768)

        return enhanced_vis, enhanced_txt