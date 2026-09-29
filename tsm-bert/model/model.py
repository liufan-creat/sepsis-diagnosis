"""
model.py - THE single source of truth for the BERT-TSM architecture.

All training / eval / inference scripts in this repo import BertTSM and
RoPEMHA from here, so there is exactly one implementation to keep
consistent.  This file was extracted programmatically (span splice,
byte-verified load parity) from the verified eval script of the original
project - do NOT hand-edit the class bodies.

Architecture summary
--------------------
Encoder-only, pre-LN Transformer:
  d_model=64, n_heads=4, n_layers=6, ffn=256, dropout=0.3, CLS readout
  (pool="cls"), RoPE on the day-major token index (CLS at position -1),
  ~0.3M parameters, binary logit head (BCEWithLogitsLoss + pos_weight
  auto during training; sigmoid at inference - softmax only inside
  attention).

Token vocabulary (per window w, sequence length L = w * C, day-major
index = day*C + channel):
  tok 0 = PAD   post-right-censorship or out-of-source-range; excluded
                from attention via the key mask, zero value injection
  tok 1 = OBS   observed, value inside the source (train-fitted) range;
                token = W_v[ch] * val + E_day[day] + E_ch[ch],
                val = robust-standardized value (median/IQR, source
                train only)
  tok 2 = MISS  missing at/before the per-admission right-censorship
                boundary (last day with ANY observation); slot becomes
                e_miss[day] + e_ch[ch] and STAYS in attention -
                missingness is structural input ("censoring as signal"),
                not imputed
"""
import torch
import torch.nn as nn

class RoPEMHA(nn.Module):
    """Drop-in nn.MultiheadAttention with Rotary Position Embedding.

    Position of a token = its day-major flat index (t*C + c), so the
    relative rotation encodes |day delta|*C + |channel delta| (same
    channel consecutive days = exactly C).  CLS row sits at position
    -1.  No learned parameters; identical interface to
    nn.MultiheadAttention (2-tuple return) so
    nn.TransformerEncoderLayer works unchanged.
    """

    def __init__(self, hidden, heads):
        super().__init__()
        assert hidden % heads == 0 and hidden // heads % 2 == 0
        self.heads = heads
        self.hd = hidden // heads
        self.batch_first = False  # force encoder slow path
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.o = nn.Linear(hidden, hidden)
        inv = torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd
        self.register_buffer("inv_freq", 1.0 / (10000.0 ** inv))

    def _rope(self, x, pos):
        # x: (B, H, L, hd); pos: (L,) long
        h = self.hd // 2
        ang = pos.float().view(1, 1, -1, 1) * self.inv_freq.view(1, 1, 1, -1)
        cos, sin = ang.cos(), ang.sin()
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    def forward(self, query, key, value, attn_mask=None,
                key_padding_mask=None, need_weights=None, **kw):
        # self_attn gets (B, L, E) when batch_first=True else (L, B, E);
        # key_padding_mask is always (B, L_key)
        bf = self.batch_first
        B, Lq = ((query.shape[0], query.shape[1]) if bf
                 else (query.shape[1], query.shape[0]))
        Lk = key.shape[1] if bf else key.shape[0]
        ph = lambda z, L: (z.view(B, L, self.heads, self.hd).transpose(1, 2)
                           if bf else
                           z.view(L, B, self.heads, self.hd).permute(1, 2, 0, 3))
        hd = self.hd
        q = ph(self.q(query), Lq)
        k = ph(self.k(key), Lk)
        v = ph(self.v(value), Lk)
        Ls = q.shape[2]
        pos = torch.arange(-1, Ls - 1, device=q.device)  # CLS=-1, token j -> j (day-major index)
        q, k = self._rope(q, pos), self._rope(k, pos)
        s = torch.matmul(q, k.transpose(-2, -1)) * (hd ** -0.5)
        if key_padding_mask is not None:
            # bool padding mask (0/1 kept, True=mask) or float additive;
            # CLS key never masked -> no all-masked row, softmax safe
            kpm = key_padding_mask[:, None, None, :]
            if kpm.is_floating_point():
                s = s + kpm
            else:
                s = s.masked_fill(kpm, float("-inf"))
        a = s.softmax(dim=-1)
        o = torch.matmul(a, v)                       # (B, H, Lq, hd)
        o = (o.transpose(1, 2).reshape(B, Lq, self.heads * hd)
             if bf else
             o.permute(2, 0, 1, 3).reshape(Lq, B, self.heads * hd))
        return self.o(o), None


class BertTSM(nn.Module):
    """BERT-style encoder over linear-value tokens (see README)."""

    CLAMP_LIM = 20.0
    SQUASH_B = 3.0

    def __init__(self, hidden, w_max, c_n, heads=4,
                 layers=6, ffn=256, drop=0.3,
                 pool="cls", value_mlp=False, value_mode="linear",
                 sc_init=None, miss_mode="pad", imp_mode=False):
        super().__init__()
        self.pool = pool
        self.value_mlp = value_mlp
        self.value_mode = value_mode
        self.miss_mode = miss_mode
        self.imp_mode = imp_mode
        self.w_v = nn.Parameter(torch.zeros(c_n, hidden))
        nn.init.normal_(self.w_v, std=0.02)
        # squash mode bound (class attr SQUASH_B, like CLAMP_LIM above)
        self.squash_B = self.SQUASH_B
        if value_mode == "raw":
            init = sc_init if sc_init is not None else torch.zeros(c_n)
            self.scale = nn.Parameter(init.clone().float())
        if value_mlp:
            self.val_mlp = nn.Sequential(
                nn.Linear(1, 32), nn.GELU(), nn.Linear(32, hidden))
        self.e_day = nn.Embedding(w_max + 1, hidden)
        self.e_ch = nn.Embedding(c_n, hidden)
        for p in (self.e_day, self.e_ch):
            nn.init.normal_(p.weight, std=0.02)
        # miss_mode=embed: day-indexed [MISS] embedding for pre-censorship
        # missing cells (tok 2) - the missing slot itself becomes an
        # attended structural token (e_miss[daypos] + e_ch[chpos]).
        # trend mode keeps e_miss AND adds the e_trend direction embedding.
        if miss_mode in ("embed", "trend", "last"):
            self.e_miss = nn.Embedding(w_max + 1, hidden)
            nn.init.normal_(self.e_miss.weight, std=0.02)
        # miss_mode=trend: 4-class embedding of the direction of the LAST
        # VISIBLE segment of the channel before a MISS gap
        # (0=NA no usable pair / 1=down / 2=flat / 3=up; labels from the
        # tokens npz's bert_trend_<w>, reach/eps fixed in prepare)
        if miss_mode == "trend":
            self.e_trend = nn.Embedding(4, hidden)
            nn.init.normal_(self.e_trend.weight, std=0.02)
        # miss_mode=last: the missing slot carries the channel's LAST KNOWN
        # READING (a real past measurement, never an imputed present value)
        # via a per-channel linear map, plus the age bucket of that reading:
        #   tok2 input = e_miss[day] + e_ch + W_last[ch]*v1 + e_age[age]
        # v1/bert_age sidecars live in the tokens npz (miss cells only);
        # age 0 = NA (channel never observed in-window before the gap)
        if miss_mode == "last":
            self.w_last = nn.Parameter(torch.zeros(c_n, hidden))
            nn.init.normal_(self.w_last, std=0.02)
            self.e_age = nn.Embedding(5, hidden)
            nn.init.normal_(self.e_age.weight, std=0.02)
        # imp_mode (embed + gen_imputed.py sidecar): a gated-passed MISS cell
        # is upgraded from the generic [MISS] token to a DIRECTION token:
        #   tok2 input = e_miss[day] + e_ch + e_imp_trend[label]
        # label 0=down/1=flat/2=up, read from imputed_trend.npz; its
        # imputed_valid mask keeps gate-BLOCKED MISS cells on the plain
        # [MISS] path. The embedding is NEW (absent from the pretrained
        # backbone keys) -> randomly initialized, like e_trend in the old
        # trend mode.
        if imp_mode:
            self.e_imp_trend = nn.Embedding(3, hidden)
            nn.init.normal_(self.e_imp_trend.weight, std=0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.head_drop = nn.Dropout(drop)
        layer = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=ffn, dropout=drop,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        for l_ in self.encoder.layers:
            l_.self_attn = RoPEMHA(hidden, heads)
        try:
            self.encoder.enable_nested_tensor = False
        except AttributeError:
            pass
        if pool == "dual":
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(128, 1))
        else:
            self.head = nn.Linear(hidden, 1)

    def forward(self, tok, val, daypos, chpos, key_mask, trend=None,
                v1=None, age=None, imp_label=None, imp_valid=None):
        # tok/val/daypos/chpos: (B, L); tok: 0=PAD 1=OBS 2=MISS(embed);
        # key_mask: (B, L+1) bool, True=PAD (only tok 0; MISS is attended)
        # trend: (B, L) long, 0=NA/1=down/2=flat/3=up; used ONLY on tok==2
        #        cells when miss_mode == "trend"; None in all other modes
        # v1/age: (B, L) float/long; miss_mode == "last" only - last known
        #        reading + age bucket of tok==2 cells; None in other modes
        # imp_label/imp_valid: (B, L) long/bool; imp_mode only. imp_label is
        #        the per-MISS-cell direction (0=down/1=flat/2=up) from the
        #        gen_imputed.py sidecar; imp_valid marks the gated-passed
        #        cells (only those get e_imp_trend added; blocked MISS keep
        #        the plain [MISS] slot). val is 0 at PAD/MISS cells.
        B, L = tok.shape
        obs = (tok == 1).float()
        if self.value_mode == "raw":
            scaled = torch.clamp(val * self.scale[chpos],
                                 -self.CLAMP_LIM, self.CLAMP_LIM)
            xv = self.w_v[chpos] * scaled.unsqueeze(-1)
        elif self.value_mode == "squash":
            # TabPFN SquashingScaler on the standardized val (linear npz):
            # soft-clip z = val to z/sqrt(1+(z/B)^2), bounded at +/-B, B=3.
            # NB: use Bsq, NOT B (B is the batch size from `B, L = tok.shape`)
            Bsq = self.squash_B
            soft = val / torch.sqrt(1.0 + (val / Bsq) ** 2)
            xv = self.w_v[chpos] * soft.unsqueeze(-1)
        else:
            xv = self.w_v[chpos] * val.unsqueeze(-1)
        if self.value_mlp:
            xv = xv + self.val_mlp(val.unsqueeze(-1)) * obs.unsqueeze(-1)
        if self.miss_mode in ("embed", "trend", "last"):
            # MISS cells (tok 2): the slot becomes e_miss[daypos] +
            # e_ch[chpos] and stays IN attention; OBS cells keep the value
            # path; PAD cells keep the zero-injected row (masked anyway).
            miss = (tok == 2)
            em = self.e_miss(daypos) + self.e_ch(chpos)  # (B, L, D)
            if self.miss_mode == "trend":
                # + direction embedding of the last visible segment of the
                # channel before the gap (0=NA/1=down/2=flat/3=up)
                em = em + self.e_trend(trend)
            if self.miss_mode == "last":
                # + last known reading of the channel (a REAL past value,
                # 0 at NA cells) via its per-channel map + age bucket
                # (0=NA/1=1-3d/2=4-7d/3=8-14d/4=>14d)
                em = (em + self.w_last[chpos] * v1.unsqueeze(-1)
                      + self.e_age(age))
            if self.imp_mode and imp_label is not None:
                # gated-passed MISS cells: [MISS] upgraded to a DIRECTION
                # token. imp_valid restricts the addition to passed cells;
                # blocked MISS (imp_valid=0) keep the plain em.
                imp = miss & imp_valid
                em = torch.where(
                    imp.unsqueeze(-1),
                    em + self.e_imp_trend(imp_label), em)
            xv = torch.where(miss.unsqueeze(-1), em, xv)
        x = xv + self.e_day(daypos) + self.e_ch(chpos)
        cls = self.cls.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.head_drop(x)
        x = x.transpose(0, 1)
        x = self.encoder(x, src_key_padding_mask=key_mask)
        cls_out = x[0]
        if self.pool == "dual":
            keep = (~key_mask[:, 1:]).permute(1, 0).unsqueeze(-1)
            mm = (x[1:] * keep).sum(0) / keep.sum(0).clamp(min=1)
            return self.head(torch.cat([cls_out, mm], dim=1)).squeeze(-1)
        return self.head(cls_out).squeeze(-1)

CLAMP_LIM = 20.0
SQUASH_B = 3.0


def build_model(cfg, state_dict=None, device="cpu"):
    """Instantiate BertTSM from a checkpoint 'config' dict and, when
    `state_dict` is given (the 'state_dict' of a bert_trm_w*.pt ckpt),
    load it STRICTLY (full key parity, fails loudly on mismatch).
    Returns the model in .eval() mode, ready for inference on `device`.

    Calling build_model WITHOUT state_dict returns a randomly
    initialized model (intended for training entry points only)."""
    mm = cfg.get("miss_mode", "pad")
    model = BertTSM(cfg.get("d_model", 64), cfg["w_max"], cfg["c_n"],
                    heads=cfg.get("n_heads", 4),
                    layers=cfg.get("n_layers", 6),
                    ffn=cfg.get("ffn", 256),
                    drop=cfg.get("dropout", 0.3),
                    pool=cfg.get("pool", "cls"),
                    value_mlp=cfg.get("value_mlp", False),
                    value_mode=cfg.get("value_mode", "linear"),
                    miss_mode=mm,
                    imp_mode=cfg.get("imp_mode", False)).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

