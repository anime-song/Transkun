import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import einops
from typing import Tuple


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class RotaryEmbeddings(torch.nn.Module):
    """
    RoPE 用の sin・cos テーブルをキャッシュし、必要に応じて伸張／切り詰めるクラス。

    Args:
        head_dim: 1 ヘッドあたりの埋め込み次元 (必ず偶数にする)
        max_seq_len: 事前に準備しておく最大シーケンス長
        base_theta: 周波数スケーリング係数 (多くの論文では 10000.0)
        learned: True にすると sin, cos をパラメータとして学習させられる
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base_theta: float = 10000.0,
        learned: bool = False,
        device: torch.device | None = None,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("head_dim (＝1 ヘッドの次元数) は偶数にしてください。")

        self.head_dim = head_dim
        self.base_theta = base_theta
        self.max_seq_len = max_seq_len

        # 角周波数: θ_k = (θ_base)^(2k / d)
        freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base_theta ** (freqs / head_dim))  # (head_dim/2,)

        # time 方向へアウター積 → (max_seq_len, head_dim/2)
        t = torch.arange(max_seq_len, device=device).float()
        sinusoid_inp = torch.einsum("i,j->ij", t, inv_freq)

        sin, cos = (
            sinusoid_inp.sin(),
            sinusoid_inp.cos(),
        )  # 各が (max_seq_len, head_dim/2)

        if learned:
            self.register_parameter("sin_cached", torch.nn.Parameter(sin))
            self.register_parameter("cos_cached", torch.nn.Parameter(cos))
        else:
            self.register_buffer("sin_cached", sin, persistent=False)
            self.register_buffer("cos_cached", cos, persistent=False)

    def forward(
        self,
        seq_len: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        指定長の sin, cos を返す。

        Returns:
            cos: (seq_len, head_dim/2)
            sin: (seq_len, head_dim/2)
        """
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"要求シーケンス長 {seq_len} は max_seq_len={self.max_seq_len} を超えています。"
            )
        cos = self.cos_cached[:seq_len].to(dtype=dtype, device=device)
        sin = self.sin_cached[:seq_len].to(dtype=dtype, device=device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    偶数次元のテンソルを (…, 2i, 2i+1) → (…, -2i+1, 2i) のように 90° 回転させる。
    具体的には (x_even, x_odd) → (-x_odd, x_even)。
    """
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rotary_embedding(
    query: torch.Tensor,  # (batch, num_heads, seq_len, head_dim)
    key: torch.Tensor,  # (batch, num_heads, seq_len, head_dim)
    cos: torch.Tensor,  # (seq_len, head_dim/2)
    sin: torch.Tensor,  # (seq_len, head_dim/2)
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)

    q_rot = query * cos + rotate_half(query) * sin
    k_rot = key * cos + rotate_half(key) * sin
    return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # make sure hiddenSize to be divisible by num_heads
        self.head_dim = hidden_size // num_heads

        self.to_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rope = RotaryEmbeddings(
            head_dim=self.head_dim,
            learned=False,
        )

    def forward(self, x):
        q, k, v = einops.rearrange(
            self.to_qkv(x), "b t (qkv h d) -> qkv b h t d", qkv=3, h=self.num_heads
        )

        cos, sin = self.rope(q.shape[-2], dtype=q.dtype, device=q.device)
        q, k = apply_rotary_embedding(q, k, cos, sin)

        q, k, v = (q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16))
        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            fetched = F.scaled_dot_product_attention(q, k, v)

        fetched = fetched.float()
        fetched = einops.rearrange(fetched, "b h t d -> b t (h d)")
        return self.out_proj(fetched)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size_factor: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads=num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(input_size, hidden_size * ffn_hidden_size_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * ffn_hidden_size_factor, input_size),
            nn.Dropout(dropout),
        )

        self.norm_before_attn = RMSNorm(input_size)
        self.norm_before_ffn = RMSNorm(input_size)

        self.out_norm = RMSNorm(input_size)

    def forward(self, x):
        # x: [B, T, F]
        residual = x
        x = self.norm_before_attn(x)
        x = self.attention(x)
        x = x + residual

        residual = x
        x = self.norm_before_ffn(x)
        x = self.ffn(x)
        x = x + residual
        x = self.out_norm(x)
        return x
