import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttnEnhancer(nn.Module):
    def __init__(self, vis_ch=1536, txt_ch=768, hidden=768, n_heads=8):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = hidden // n_heads
        self.scale    = self.head_dim ** -0.5

        # 视觉 QKV
        self.vis_q = nn.Conv2d(vis_ch, hidden, 1)
        self.vis_kv = nn.Conv2d(vis_ch, hidden*2, 1)

        # 文本 QKV
        self.txt_q = nn.Linear(txt_ch, hidden)
        self.txt_kv = nn.Linear(txt_ch, hidden*2)

        # 输出映射（保持通道数不变）
        self.vis_out = nn.Conv2d(hidden, vis_ch, 1)
        self.txt_out = nn.Linear(hidden, txt_ch)

    def _attn(self, q, k, v):
        # q,k,v : (B,heads,L,head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        out  = (attn @ v).transpose(1, 2).contiguous()   # (B,L,heads*head_dim)
        return out

    def forward(self, vis, txt):
        B, C, H, W = vis.shape
        Nc = txt.shape[0]

        # ---------- 1. 视觉→文本 ----------
        q_vis = self.vis_q(vis).flatten(2).permute(0,2,1)          # (B,L,C)
        kv_vis = self.vis_kv(vis).flatten(2).permute(0,2,1).chunk(2, dim=-1)
        k_vis, v_vis = map(lambda x: x.transpose(0,1), kv_vis)      # (L,B,C)

        q_txt = self.txt_q(txt)                                    # (Nc,C)
        kv_txt = self.txt_kv(txt).chunk(2, dim=-1)
        k_txt, v_txt = kv_txt                                       # (Nc,C)

        # 统一 reshape 成多头
        def _mh(x):  # (B_or_Nc, seq, C) -> (B, heads, seq, head_dim)
            if x.dim()==2: x = x.unsqueeze(0).expand(B,-1,-1)      # 文本先扩 B
            return x.view(x.size(0), x.size(1), self.n_heads, self.head_dim).transpose(1,2)

        # 视觉→文本
        out_vis = self._attn(_mh(q_vis), _mh(k_txt), _mh(v_txt))   # (B,L,C)
        out_vis = out_vis.to_dense()  # 先变回稠密
        out_vis = out_vis.transpose(1, 2)  # 等价 permute(0,2,1)
        out_vis = out_vis.reshape(B, -1, H, W)
        out_vis = self.vis_out(out_vis)

        # ---------- 2. 文本→视觉 ----------
        q_txt = q_txt.unsqueeze(0).expand(B,-1,-1)                 # (B,Nc,C)
        out_txt = self._attn(_mh(q_txt), _mh(k_vis.transpose(0,1)),
                                         _mh(v_vis.transpose(0,1))) # (B,Nc,C)
        out_txt = out_txt.transpose(1, 2).contiguous()  # (B, Nc, heads, head_dim)
        out_txt = out_txt.view(B, Nc, -1)  # (B, Nc, 768)
        out_txt = self.txt_out(out_txt.mean(0))  # (Nc, 768)

        return out_vis, out_txt
