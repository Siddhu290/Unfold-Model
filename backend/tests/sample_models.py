"""Small real models used as test fixtures and built-in demos."""

import torch
from torch import nn


class MNISTClassifierMLP(nn.Module):
    """Plain MLP for 28x28 grayscale digits."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


class MNISTClassifierCNN(nn.Module):
    """Small conv net with a residual-style skip to exercise non-sequential
    execution order."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = x + self.relu(self.bn2(self.conv2(x)))  # skip connection
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        return self.head(x)


class CausalSelfAttention(nn.Module):
    """Explicit multi-head attention with named Q/K/V projections so every
    step of the mechanism is individually hookable. The attention
    probabilities pass through an nn.Identity tap (`attn_probs`) purely so
    the standard forward-hook pipeline captures them."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_probs = nn.Identity()

    def forward(self, x):
        B, T, C = x.shape
        hd = C // self.n_heads
        q = self.q_proj(x).view(B, T, self.n_heads, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf"))
        att = self.attn_probs(torch.softmax(att, dim=-1))  # (B, H, T, T)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # pre-norm residual
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    """Character-level causal transformer, small enough to inspect fully.
    4 identical blocks so repeat-group detection has something to find."""

    VOCAB = 96  # printable ASCII
    D_MODEL = 64
    N_LAYERS = 4
    N_HEADS = 4
    MAX_LEN = 128

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(self.VOCAB, self.D_MODEL)
        self.pos_emb = nn.Embedding(self.MAX_LEN, self.D_MODEL)
        self.blocks = nn.ModuleList(
            TransformerBlock(self.D_MODEL, self.N_HEADS)
            for _ in range(self.N_LAYERS)
        )
        self.ln_f = nn.LayerNorm(self.D_MODEL)
        self.lm_head = nn.Linear(self.D_MODEL, self.VOCAB, bias=False)

    def forward(self, idx):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    # simple char codec so the demo can accept text prompts
    def encode(self, text: str) -> torch.Tensor:
        ids = [max(0, min(self.VOCAB - 1, ord(c) - 32)) for c in text][: self.MAX_LEN]
        return torch.tensor([ids or [0]], dtype=torch.long)

    def decode(self, ids) -> str:
        return "".join(chr(int(i) + 32) for i in ids)


def build_demo(name: str):
    """Returns (model, example_input, meta)."""
    torch.manual_seed(0)
    if name == "mlp":
        return MNISTClassifierMLP().eval(), torch.randn(1, 1, 28, 28), {
            "input_kind": "tensor", "input_shape": [1, 1, 28, 28],
            "task": "classification", "num_classes": 10,
        }
    if name == "cnn":
        return MNISTClassifierCNN().eval(), torch.randn(1, 1, 28, 28), {
            "input_kind": "tensor", "input_shape": [1, 1, 28, 28],
            "task": "classification", "num_classes": 10,
        }
    if name == "tiny_transformer":
        m = TinyTransformerLM().eval()
        return m, m.encode("hello world"), {
            "input_kind": "text", "task": "char_lm", "vocab": m.VOCAB,
        }
    raise ValueError(f"unknown demo model: {name}")


DEMO_MODELS = {
    "mlp": "MNIST-style MLP (784→128→64→10)",
    "cnn": "Small CNN with skip connection + BatchNorm",
    "tiny_transformer": "4-layer character-level transformer LM",
}
