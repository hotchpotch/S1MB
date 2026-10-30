"""Jalebi-E: a 66.6M-parameter decision encoder trained from scratch (Airavat AI). One file: model, tokenizer, input encoding, scoring.

    from jalebi_e import JalebiE
    model = JalebiE.from_pretrained("Airavat-ai/Jalebi-E")           # or a local folder holding jalebi-decision.pt + tok24k.json
    probs = model.score("Customer writes: my card was declined abroad", "Where should this request go?",
                        ["billing team", "fraud team", "general support"])   # -> [p1, p2, p3], sums to 1

Input layout: [CLS] question [SEP] state [SEP] [OPTION] option 1 [OPTION] option 2 ... [SEP]. Every option is scored in one forward
pass by a shared scorer head and the scores are softmaxed over the options (choice, yes/no and score questions all use this).

Input length: the window is 2,048 tokens. By default an input that does not fit is REJECTED (ValueError), never truncated.
Pass trim=True to opt in to the shortening used for our published S1MB run: the question is cut at 1,024 tokens, each option at
256 tokens and the END of the state is dropped so that the whole input fits, keeping at least 64 state tokens. Options are never
split over several windows; if they cannot fit in one window the input is rejected in both modes.
"""
import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID, OPTION_ID = 0, 1, 2, 3, 4, 5
WINDOW = 2048
QUESTION_CAP, OPTION_CAP, MIN_STATE = 1024, 256, 64          # used only with trim=True
WEIGHTS, TOKENIZER = "jalebi-decision.pt", "tok24k.json"


@dataclass
class Config:
    vocab_size: int = 24576
    d_model: int = 576
    n_layers: int = 13
    n_heads: int = 9
    d_ff: int = 1536
    max_len: int = 2048
    rope_base: float = 10000.0
    dropout: float = 0.0
    norm_eps: float = 1e-6
    init_std: float = 0.02


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class Rotary(nn.Module):
    """Interleaved rotary position embedding."""

    def __init__(self, head_dim, max_len, base):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(max_len).float(), inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x):                       # (batch, heads, length, head_dim)
        n = x.size(-2)
        cos, sin = self.cos[:n].to(x.dtype), self.sin[:n].to(x.dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.head_dim = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, rope, mask):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q), rope(k), v, attn_mask=mask)
        return self.out(a.transpose(1, 2).reshape(b, n, d))


class GeGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_in = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)
        self.w_out = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        a, g = self.w_in(x).chunk(2, -1)
        return self.w_out(F.gelu(a, approximate="tanh") * g)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps), RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn, self.ffn = Attention(cfg), GeGLU(cfg)

    def forward(self, x, rope, mask):
        x = x + self.attn(self.norm1(x), rope, mask)
        return x + self.ffn(self.norm2(x))


class JalebiE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = Rotary(cfg.d_model // cfg.n_heads, cfg.max_len, cfg.rope_base)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlm_norm = RMSNorm(cfg.d_model, cfg.norm_eps)             # pretraining head, unused at inference (kept so the
        self.mlm_bias = nn.Parameter(torch.zeros(cfg.vocab_size))       # checkpoint loads strictly and parameter counts match)
        self.scorer = nn.Sequential(nn.Linear(2 * cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 1))
        self.tokenizer = None

    # ---- loading ------------------------------------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, path_or_repo, revision=None, device=None):
        import os
        from tokenizers import Tokenizer
        if os.path.isdir(path_or_repo):
            weights, tokenizer = os.path.join(path_or_repo, WEIGHTS), os.path.join(path_or_repo, TOKENIZER)
        else:
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download(path_or_repo, WEIGHTS, revision=revision)
            tokenizer = hf_hub_download(path_or_repo, TOKENIZER, revision=revision)
        payload = torch.load(weights, map_location="cpu", weights_only=True)
        known = {f.name for f in fields(Config)}
        model = cls(Config(**{k: v for k, v in payload["cfg"].items() if k in known}))
        model.load_state_dict({k: v for k, v in payload["model"].items() if not k.startswith("rope.")}, strict=True)
        model.tokenizer = Tokenizer.from_file(tokenizer)
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return model.to(device).eval()

    @property
    def device(self):
        return self.emb.weight.device

    # ---- forward ------------------------------------------------------------------------------------------------
    def forward(self, input_ids, attention_mask, option_positions):
        """Scores, one per option position (padded positions, -1, score -1e4)."""
        x = self.emb(input_ids)
        mask = attention_mask.bool()[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, self.rope, mask)
        hidden = self.norm(x)
        valid = option_positions >= 0
        pos = option_positions.clamp_min(0)
        d = hidden.size(-1)
        opt_vec = hidden.gather(1, pos[..., None].expand(-1, -1, d)).float()
        length = attention_mask.sum(1).long()
        nxt = torch.cat([option_positions[:, 1:], option_positions.new_full((option_positions.size(0), 1), -1)], 1)
        ends = torch.where(nxt >= 0, nxt, (length - 1)[:, None]).clamp_min(0)       # next [OPTION], or the final [SEP]
        cs = F.pad(hidden.float().cumsum(1), (0, 0, 1, 0))
        take = lambda idx: cs.gather(1, idx[..., None].expand(-1, -1, d))
        count = (ends - pos - 1).clamp_min(1)[..., None].float()
        mean = (take(ends) - take(pos + 1)) / count                                 # mean of the option's own tokens
        with torch.autocast(device_type=hidden.device.type, enabled=False):          # fp32 head: low precision creates false ties
            scores = self.scorer(torch.cat([opt_vec, mean], -1)).squeeze(-1)
        return scores.masked_fill(~valid, -1e4)

    # ---- encoding -----------------------------------------------------------------------------------------------
    def encode(self, state, question, options, trim=False, window=WINDOW):
        """(input_ids, option_positions) for one decision. Raises ValueError when it does not fit (see the module docstring)."""
        if len(options) < 2:
            raise ValueError("a decision needs at least two options")
        enc = lambda text: self.tokenizer.encode(text, add_special_tokens=False).ids
        q, s = enc(question), enc(state)
        opts = [enc(o) or enc("none")[:1] for o in options]                         # an empty option becomes the token for "none"
        if trim:
            q, opts = q[:QUESTION_CAP], [o[:OPTION_CAP] for o in opts]
        fixed = len(q) + 4                                                           # CLS q SEP state SEP ... SEP
        option_tokens = sum(1 + len(o) for o in opts)
        keep = len(s)
        if trim:
            keep = min(len(s), max(0, window - fixed - option_tokens))
            if keep < min(len(s), MIN_STATE) and len(s) > keep:
                raise ValueError("options and question leave fewer than %d state tokens; refusing" % MIN_STATE)
        need = fixed + keep + option_tokens
        if need > window:
            raise ValueError("input needs %d tokens, exceeding the %d-token window; refusing truncation" % (need, window))
        ids = [CLS_ID] + q + [SEP_ID] + s[:keep] + [SEP_ID]
        positions = []
        for o in opts:
            positions.append(len(ids))
            ids += [OPTION_ID] + o
        return ids + [SEP_ID], positions, len(s) - keep

    # ---- scoring ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def probabilities(self, requests, trim=False, batch_size=32, window=WINDOW, bf16="auto"):
        """requests: list of (state, question, options). Returns a list of probability lists plus the number of state tokens
        dropped per request (always 0 unless trim=True). A request that cannot be encoded raises ValueError for the whole call;
        use encode() first if you need per-request handling."""
        encoded = [self.encode(s, q, o, trim, window) for s, q, o in requests]
        if bf16 == "auto":
            bf16 = self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] >= 8
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
        scores = [None] * len(encoded)
        for start in range(0, len(order), batch_size):
            part = order[start:start + batch_size]
            longest = max(len(encoded[i][0]) for i in part)
            most = max(len(encoded[i][1]) for i in part)
            ids = torch.full((len(part), longest), PAD_ID, dtype=torch.long)
            att = torch.zeros((len(part), longest), dtype=torch.long)
            pos = torch.full((len(part), most), -1, dtype=torch.long)
            for r, i in enumerate(part):
                row, positions, _ = encoded[i]
                ids[r, :len(row)] = torch.tensor(row)
                att[r, :len(row)] = 1
                pos[r, :len(positions)] = torch.tensor(positions)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=bool(bf16) and self.device.type == "cuda"):
                out = self(ids.to(self.device), att.to(self.device), pos.to(self.device)).float().cpu()
            for r, i in enumerate(part):
                scores[i] = out[r, :len(encoded[i][1])].tolist()
        probs = []
        for s in scores:
            top = max(s)
            e = [math.exp(v - top) for v in s]
            total = sum(e)
            probs.append([v / total for v in e])
        return probs, [e[2] for e in encoded]

    def score(self, state, question, options, trim=False):
        """Probabilities of one decision, one per option (they sum to 1)."""
        return self.probabilities([(state, question, options)], trim=trim)[0][0]
