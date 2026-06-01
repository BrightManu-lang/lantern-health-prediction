import torch
import torch.nn as nn

from typing import List, Dict, Optional

class Time2Vec(nn.Module):
    def __init__(self, t2v_dim: int):
        super().__init__()
        assert t2v_dim >= 2
        self.lin = nn.Linear(1, 1)
        self.freq = nn.Linear(1, t2v_dim - 1)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        dt = dt.clamp(min=0.0)
        dt = torch.log1p(dt)
        linear = self.lin(dt)
        sinus = torch.sin(self.freq(dt))
        return torch.cat([linear, sinus], dim=-1)

class LANTERN(nn.Module):
    """
    Hierarchical:
      - Head 1: Death vs Not-Death (sigmoid)
      - Head 2: Alive state (Healthy/Mild/Severe) conditional softmax
      - Final probs: [p_H, p_M, p_S, p_D]
    """
    def __init__(self, in_feats: int, hidden_dim: int, attr_cols: List[str], attr_vocabs: Dict[str, Dict[str, int]], t2v_dim: int = 8,
                 attn_heads: int = 2, dropout: float = 0.1, ablate_time2vec: bool = False,  ablate_attr_attention: bool = False):
        super().__init__()
        assert hidden_dim % attn_heads == 0

        self.t2v_dim = t2v_dim
        self.ablate_time2vec = ablate_time2vec
        self.ablate_attr_attention = ablate_attr_attention

        self.d_x = in_feats
        self.d_h = hidden_dim
        self.attr_cols = list(attr_cols)

        # time embedding module kept, but will be bypassed during ablation
        self.t2v = Time2Vec(t2v_dim)

        # attribute embeddings
        self.attr_emb = nn.ModuleDict()
        self.attr_proj = nn.ModuleDict()
        for col in self.attr_cols:
            n = len(attr_vocabs[col])
            self.attr_emb[col] = nn.Embedding(n, hidden_dim)
            self.attr_proj[col] = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))

        # attention machinery (used only if not ablated)
        self.q_proj_attr = nn.Sequential(nn.Linear(hidden_dim + t2v_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.mha_attr = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attn_heads, dropout=dropout, batch_first=True)

        # keep GRU input dims fixed across ablations
        self.gru = nn.GRUCell(in_feats + hidden_dim + t2v_dim, hidden_dim)

        d_mid = max(8, hidden_dim // 2)
        self.trunk = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, d_mid), nn.ReLU(), nn.Dropout(dropout))
        self.death_head = nn.Linear(d_mid, 1)
        self.alive_head = nn.Linear(d_mid, 3)

        self.last_attn_attr: Optional[torch.Tensor] = None

    def encode_attrs(self, attr_ids: torch.Tensor) -> torch.Tensor:
        outs = []
        for j, col in enumerate(self.attr_cols):
            e = self.attr_emb[col](attr_ids[:, j])
            e = self.attr_proj[col](e)
            outs.append(e.unsqueeze(1))
        return torch.cat(outs, dim=1)  # [B, A, d_h]

    def forward_one_batch(self, m_prev, X_num2, dt_self, attr_ids):
        # Time embedding
        if self.ablate_time2vec:
            t_self = torch.zeros((dt_self.size(0), self.t2v_dim), device=dt_self.device)
        else:
            t_self = self.t2v(dt_self)

        # Attributes
        KVs_attr = self.encode_attrs(attr_ids)  # [B, A, d_h]
        B, A, _ = KVs_attr.shape

        if self.ablate_attr_attention:
            # Attention OFF, but still use attrs as covariates
            msg_attr = KVs_attr.mean(dim=1)

            # "attention weights" are all zeros (off)
            self.last_attn_attr = torch.zeros((B, A), device=KVs_attr.device).detach()
        else:
            q_attr = self.q_proj_attr(torch.cat([m_prev, t_self], dim=-1)).unsqueeze(1)
            msg_attr, attn_attr = self.mha_attr(
                q_attr, KVs_attr, KVs_attr,
                need_weights=True, average_attn_weights=True
            )
            msg_attr = msg_attr.squeeze(1)
            self.last_attn_attr = attn_attr.squeeze(1).detach()

        # GRU update
        updater_in = torch.cat([X_num2, msg_attr, t_self], dim=-1)
        m_new = self.gru(updater_in, m_prev)

        h = self.trunk(m_new)
        z_death = self.death_head(h).squeeze(1)
        logits_alive = self.alive_head(h)

        pD = torch.sigmoid(z_death)
        p_alive = torch.softmax(logits_alive, dim=1)
        probs3 = (1.0 - pD).unsqueeze(1) * p_alive
        probs4 = torch.cat([probs3, pD.unsqueeze(1)], dim=1)

        return m_new, probs4, z_death, logits_alive