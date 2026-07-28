import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttnEnhancer(nn.Module):
    def __init__(self, vis_ch=1536, txt_ch=768, hidden=768, tri_hidden=512, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.scale = self.head_dim ** -0.5
        self.hidden = hidden
        self.tri_hidden = tri_hidden

        # Same foreground-visual cross attention structure as the original crossattforlast.py.
        self.vis_q = nn.Conv2d(vis_ch, hidden, 1)
        self.vis_kv = nn.Conv2d(vis_ch, hidden * 2, 1)
        self.txt_q = nn.Linear(txt_ch, hidden)
        self.txt_kv = nn.Linear(txt_ch, hidden * 2)
        self.vis_out = nn.Conv2d(hidden, vis_ch, 1)
        self.txt_out = nn.Linear(hidden, txt_ch)

        self.tri_vis_q = nn.Conv2d(vis_ch, tri_hidden, 1)
        self.tri_fg_kv = nn.Linear(txt_ch, tri_hidden * 2)
        self.tri_bg_kv = nn.Linear(txt_ch, tri_hidden * 2)
        self.tri_vis_fg_attn = nn.MultiheadAttention(tri_hidden, n_heads, batch_first=True)
        self.tri_vis_bg_attn = nn.MultiheadAttention(tri_hidden, n_heads, batch_first=True)
        self.tri_vis_fg_out = nn.Conv2d(tri_hidden, vis_ch, 1)
        self.tri_vis_bg_out = nn.Conv2d(tri_hidden, vis_ch, 1)

        self.gate_mlp = nn.Sequential(
            nn.Conv2d(vis_ch * 4, tri_hidden, 1),
            nn.GELU(),
            nn.Conv2d(tri_hidden, 2, 1),
        )

        self.tri_fg_q = nn.Linear(txt_ch, tri_hidden)
        self.tri_bg_q = nn.Linear(txt_ch, tri_hidden)
        self.tri_vis_kv = nn.Conv2d(vis_ch, tri_hidden * 2, 1)
        self.tri_fg_vis_attn = nn.MultiheadAttention(tri_hidden, n_heads, batch_first=True)
        self.tri_bg_vis_attn = nn.MultiheadAttention(tri_hidden, n_heads, batch_first=True)
        self.tri_fg_out = nn.Linear(tri_hidden, txt_ch)
        self.tri_bg_out = nn.Linear(tri_hidden, txt_ch)

    def _expand_text(self, text, batch_size):
        if text.dim() == 2:
            return text.unsqueeze(0).expand(batch_size, -1, -1)
        if text.dim() == 3:
            if text.shape[0] == batch_size:
                return text
            if text.shape[0] == 1:
                return text.expand(batch_size, -1, -1)
        raise ValueError(f"Expected text with shape (N, C) or (B, N, C), got {tuple(text.shape)}")

    def _attn(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        return (attn @ v).transpose(1, 2).contiguous()

    def _visual_text_attn(self, vis_tokens, text, text_kv_proj, attn, out_proj, height, width):
        B = vis_tokens.shape[0]
        k_text, v_text = text_kv_proj(text).chunk(2, dim=-1)
        k_text = self._expand_text(k_text, B)
        v_text = self._expand_text(v_text, B)
        out_tokens, _ = attn(vis_tokens, k_text, v_text, need_weights=False)
        out = out_tokens.transpose(1, 2).reshape(B, self.tri_hidden, height, width)
        return out_proj(out)

    def _text_visual_attn(self, text, text_q_proj, x_tri_k, x_tri_v, attn, out_proj, batch_size):
        q_text = self._expand_text(text_q_proj(text), batch_size)
        out_text, _ = attn(q_text, x_tri_k, x_tri_v, need_weights=False)
        return out_proj(out_text)

    def _foreground_cross_attn(self, vis, fg_txt):
        B, _, H, W = vis.shape
        q_vis = self.vis_q(vis).flatten(2).permute(0, 2, 1)
        kv_vis = self.vis_kv(vis).flatten(2).permute(0, 2, 1).chunk(2, dim=-1)
        k_vis, v_vis = map(lambda x: x.transpose(0, 1), kv_vis)

        q_txt = self._expand_text(self.txt_q(fg_txt), B)
        kv_txt = self.txt_kv(fg_txt).chunk(2, dim=-1)
        k_txt, v_txt = (self._expand_text(x, B) for x in kv_txt)

        def _mh(x):
            if x.dim() == 2:
                x = x.unsqueeze(0).expand(B, -1, -1)
            return x.view(x.size(0), x.size(1), self.n_heads, self.head_dim).transpose(1, 2)

        out_vis = self._attn(_mh(q_vis), _mh(k_txt), _mh(v_txt))
        out_vis = out_vis.to_dense()
        out_vis = out_vis.transpose(1, 2)
        out_vis = out_vis.reshape(B, -1, H, W)
        out_vis = self.vis_out(out_vis)

        out_txt = self._attn(_mh(q_txt), _mh(k_vis.transpose(0, 1)), _mh(v_vis.transpose(0, 1)))
        out_txt = out_txt.transpose(1, 2).contiguous()
        out_txt = out_txt.view(B, q_txt.shape[1], -1)
        out_txt = self.txt_out(out_txt)

        return out_vis, out_txt

    def forward(self, vis, fg_txt, bg_txt, return_intermediates=False):
        B, C, H, W = vis.shape
        vis_cross, fg_cross = self._foreground_cross_attn(vis, fg_txt)
        tri_vis_tokens = self.tri_vis_q(vis_cross).flatten(2).transpose(1, 2)

        x_obj = self._visual_text_attn(
            tri_vis_tokens,
            fg_cross,
            self.tri_fg_kv,
            self.tri_vis_fg_attn,
            self.tri_vis_fg_out,
            H,
            W,
        )
        x_bg = self._visual_text_attn(
            tri_vis_tokens,
            bg_txt,
            self.tri_bg_kv,
            self.tri_vis_bg_attn,
            self.tri_vis_bg_out,
            H,
            W,
        )

        gate = torch.sigmoid(self.gate_mlp(torch.cat([vis_cross, x_obj, x_bg, x_obj - x_bg], dim=1)))
        obj_gate, bg_gate = gate.chunk(2, dim=1)
        x_tri = vis_cross + obj_gate * x_obj - bg_gate * x_bg

        x_tri_k, x_tri_v = self.tri_vis_kv(x_tri).flatten(2).transpose(1, 2).chunk(2, dim=-1)
        out_fg_txt = fg_cross + self._text_visual_attn(
            fg_cross,
            self.tri_fg_q,
            x_tri_k,
            x_tri_v,
            self.tri_fg_vis_attn,
            self.tri_fg_out,
            B,
        )
        out_bg_txt = bg_txt + self._text_visual_attn(
            bg_txt,
            self.tri_bg_q,
            x_tri_k,
            x_tri_v,
            self.tri_bg_vis_attn,
            self.tri_bg_out,
            B,
        )

        if return_intermediates:
            return x_tri, out_fg_txt, out_bg_txt, {
                "cross_attention": vis_cross,
                "tri_attention": x_tri,
            }
        return x_tri, out_fg_txt, out_bg_txt
