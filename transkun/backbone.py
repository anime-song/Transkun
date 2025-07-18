import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np
import librosa
import einops
from typing import Tuple
from .utils import checkpointByPass
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing import List, Tuple


class ScaledInnerProductIntervalScorer(nn.Module):
    def __init__(
        self,
        size,
        expansionFactor=1,
        dropoutProb=0.0,
        withScoreEps=False,
        lengthScaling="linear",
    ):
        super().__init__()

        self.size = size

        if not withScoreEps:
            self.map = nn.Sequential(
                nn.Linear(
                    size, 2 * size * expansionFactor + 1
                ),  # only inner product plus diagonal
            )
        else:
            self.map = nn.Sequential(
                nn.Linear(size, 2 * size * expansionFactor + 1 + 1),
            )

        self.dropout = nn.Dropout(dropoutProb)

        self.expansionFactor = expansionFactor

        self.lengthScaling = lengthScaling

    def forward(self, ctx):
        q, k, diag = (self.map(ctx)).split(
            [self.size * self.expansionFactor, self.size * self.expansionFactor, 1],
            dim=-1,
        )

        # print(q.std(), k.std(), b.std())
        q = q / math.sqrt(q.shape[-1])

        # part1 innerproduct
        S = torch.einsum("iped, ipbd-> ipeb", q, k)
        # diagS = S.diagonal(dim1= -2, dim2=-1)

        tmpIdx_e = torch.arange(S.shape[-2], device=S.device)
        tmpIdx_b = torch.arange(S.shape[-1], device=S.device)
        len_eb = (tmpIdx_e.unsqueeze(-1) - tmpIdx_b.unsqueeze(0)).abs()

        if self.lengthScaling == "linear":
            S = S * (len_eb)
        elif self.lengthScaling == "sqrt":
            S = S * (len_eb).float().sqrt()
        elif self.lengthScaling == "none":
            pass
        else:
            raise Exception("Unrecognized lengthScaling")

        diagM = torch.diag_embed(diag.squeeze(-1))

        S = S + diagM

        # dummy eps score for testing
        b = diag * 0.0
        b = b[..., 1:, 0]

        S = S.permute(2, 3, 0, 1).contiguous()
        b = b.permute(2, 0, 1).contiguous()
        return S, b


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


def build_mel_band_indices(
    sample_rate: int,
    n_fft: int,
    n_bands: int,
) -> list[np.ndarray[int]]:
    """
    各 Mel バンドがカバーする「周波数ビン番号だけ」を返す。
    返り値: list[np.ndarray]  (len = n_bands)
    """
    # --- Mel フィルタ → 0/1 マスク --------------------------------
    mel_fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_bands)
    mel_fb[0, 0] = 1.0
    mel_fb[-1, -1] = 1.0
    mask = mel_fb > 0  # (B, F) bool

    return [np.where(mask[b])[0].astype(np.int32) for b in range(n_bands)]


def build_band_indices(
    sample_rate: int = 44_100, n_fft: int = 2_048
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    BSRoformer 用バンド分割 (各バンドの bin 数が必ず閾値以上)。

    Returns
    -------
    freqs : np.ndarray
        rfft の周波数 (長さ n_fft//2 + 1) [Hz]
    bands : List[np.ndarray]
        各要素が「そのバンドに含まれるビン番号」の 1‑D 配列
    """
    freqs = np.fft.rfftfreq(n_fft, d=1 / sample_rate)

    # バンドが始まるビンの周波数から必要 bin 数を決定
    def bins_per_band(freq_hz: float) -> int:
        if freq_hz < 1_000:  # 0‑1 kHz
            return 2
        elif freq_hz < 2_000:  # 1‑2 kHz
            return 4
        elif freq_hz < 4_000:  # 2‑4 kHz
            return 12
        elif freq_hz < 8_000:  # 4‑8 kHz
            return 24
        elif freq_hz < 16_000:  # 8‑16 kHz
            return 48
        else:  # ≥16 kHz → 後で 2 分割
            return -1

    freq_indices: List[np.ndarray] = []
    idx = 0
    n_bins = len(freqs)

    while idx < n_bins:
        # 16 kHz 以上は残りを 2 分割して終了
        if freqs[idx] >= 16_000:
            remaining = np.arange(idx, n_bins)
            half = len(remaining) // 2
            freq_indices.append(remaining[:half])
            freq_indices.append(remaining[half:])
            break

        size = bins_per_band(freqs[idx])
        band = np.arange(idx, min(idx + size, n_bins))
        freq_indices.append(band)
        idx += size  # bin数 単位で次へ

    return freqs, freq_indices


class BandSplit(torch.nn.Module):
    def __init__(self, hidden_size: int, band_indices, num_channels: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.band_indices = band_indices

        for i, idx in enumerate(band_indices):
            self.register_buffer(f"band_idx_{i}", torch.tensor(idx, dtype=torch.long))

        self.num_bands = len(band_indices)
        self.projections = torch.nn.ModuleList([])
        for i in range(self.num_bands):
            sub_band_freqs = len(band_indices[i]) * 2 * num_channels
            self.projections.append(
                torch.nn.Sequential(
                    RMSNorm(sub_band_freqs),
                    torch.nn.Linear(sub_band_freqs, hidden_size, bias=False),
                )
            )

    def forward(self, x):
        # x: (B, T, C, F)

        sub_band_list = []
        for i, proj_layer in enumerate(self.projections):
            band_indices = getattr(self, f"band_idx_{i}")
            # サブバンドインデックスで周波数軸から抜きだす
            sub_band = x[..., band_indices]  # (B, T, C, sub_band_freqs)
            sub_band = einops.rearrange(sub_band, "b t c f -> b t (c f)")
            sub_band = proj_layer(sub_band)  # (B, T, hidden_size)
            sub_band_list.append(sub_band)

        return torch.stack(sub_band_list, dim=-2)  # (B, T, num_bands, hidden_size)


class BandMaskMLP(nn.Module):
    def __init__(self, input_dim, band_dim, hidden_dim, depth):
        """
        input_dim : Backbone から来る特徴次元 (= dim)
        band_dim  : このバンドで予測したい出力次元 (= dim_in)
        hidden_dim: 中間層のユニット数
        depth     : Tanh 層を挟む数
        """
        super().__init__()
        layers = []
        # 入力→隠れ→…→隠れ→(band_dim*2) の構造
        dims = [input_dim] + [hidden_dim] * depth + [band_dim * 2]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # 最終層の後に活性化は入れない
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        # GLU でチャネルを半分に
        self.glu = nn.GLU(dim=-1)

    def forward(self, x):
        # x: (..., input_dim)
        x = self.net(x)  # → (..., band_dim*2)
        x = self.glu(x)  # → (..., band_dim)
        return x


class MaskEstimator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        band_indices,
        num_channels: int,
        mlp_expansion_factor: int = 4,
        depth: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.band_indices = band_indices

        self.num_bands = len(band_indices)
        self.projections = torch.nn.ModuleList([])
        for i in range(self.num_bands):
            sub_band_freqs = len(band_indices[i]) * 2 * num_channels
            self.projections.append(
                BandMaskMLP(
                    input_dim=input_dim,
                    band_dim=sub_band_freqs,
                    hidden_dim=sub_band_freqs * mlp_expansion_factor,
                    depth=depth,
                )
            )

    def forward(self, x):
        # x: (B, T, K, D)

        out = []
        sub_band_list = torch.unbind(x, dim=2)  # list([B, T, D])
        for i, proj_layer in enumerate(self.projections):
            sub_band = proj_layer(sub_band_list[i])  # (B, T, sub_band_freqs)
            sub_band = einops.rearrange(
                sub_band,
                "b t (f c s) -> b t (c s) f",
                c=self.num_channels,
                s=2,
                f=len(self.band_indices[i]),
            )
            out.append(sub_band)

        return torch.concat(out, dim=-1)  # (B, T, C, F)


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
        self.to_gates = nn.Linear(hidden_size, self.num_heads)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rope = RotaryEmbeddings(
            head_dim=self.head_dim,
            learned=False,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qkv.weight)
        if self.to_qkv.bias is not None:
            nn.init.zeros_(self.to_qkv.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        q, k, v = einops.rearrange(
            self.to_qkv(x), "b t (qkv h d) -> qkv b h t d", qkv=3, h=self.num_heads
        )

        cos, sin = self.rope(q.shape[-2], dtype=q.dtype, device=q.device)
        q, k = apply_rotary_embedding(q, k, cos, sin)

        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            with torch.autocast(device_type=q.device.type, dtype=torch.bfloat16):
                fetched = F.scaled_dot_product_attention(q, k, v)

        gates = self.to_gates(x)
        gates = einops.rearrange(gates, "b t h -> b h t 1").sigmoid()

        fetched = fetched.float() * gates
        fetched = einops.rearrange(fetched, "b h t d -> b t (h d)")
        return self.out_proj(fetched)


class RoformerLayer(nn.Module):
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


class BandSplitRoformerLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size_factor: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_roformer = RoformerLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            ffn_hidden_size_factor=ffn_hidden_size_factor,
            dropout=dropout,
        )
        self.band_roformer = RoformerLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            ffn_hidden_size_factor=ffn_hidden_size_factor,
            dropout=dropout,
        )

    def forward(self, x):
        # x: [B, T, K, F]
        B, T, K, F = x.shape

        # 時間軸Transformer
        x = x.reshape(B * K, T, F).contiguous()  # [B*K, T, F]
        x = self.time_roformer(x)
        x = x.reshape(B, K, T, F).transpose(1, 2).contiguous()  # [B, T, K, F]

        # バンド軸Transformer
        x = x.reshape(B * T, K, F)  # [B*T, K, F]
        x = self.band_roformer(x)
        x = x.reshape(B, T, K, F)

        return x


class Backbone(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        num_channels: int,
        n_fft: int,
        n_bands: int,
        hidden_size: int,
        num_heads: int,
        target_midi_pitches: list[int],
        scoring_expansion_factor: int = 1,
        ffn_hidden_size_factor: int = 2,
        dropout: float = 0.0,
        num_layers: int = 4,
        use_gradient_checkpoint=True,
        num_stems: int = 2,
        band_split_type: str = "bs",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_pitches = len(target_midi_pitches)
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.num_channels = num_channels
        self.n_fft = n_fft

        if band_split_type == "bs":
            _, self.band_indices = build_band_indices(
                sample_rate=sampling_rate, n_fft=n_fft
            )
        elif band_split_type == "mel":
            self.band_indices = build_mel_band_indices(
                sr=sampling_rate, n_fft=n_fft, n_bands=n_bands
            )
        else:
            raise NotImplementedError("サポートしていない band_split_type です")

        self.num_bands = len(self.band_indices)
        self.band_split = BandSplit(
            hidden_size=hidden_size,
            band_indices=self.band_indices,
            num_channels=self.num_channels,
        )

        encoder_layers = [
            BandSplitRoformerLayer(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_hidden_size_factor=ffn_hidden_size_factor,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]
        self.encoder_layers = nn.ModuleList(encoder_layers)

        self.last_proj = nn.Linear(
            self.hidden_size, self.hidden_size * scoring_expansion_factor
        )

        self.register_buffer(
            "freq_indices",
            torch.tensor(np.concatenate(self.band_indices), dtype=torch.long),
            persistent=False,
        )
        # そのビンが何本のバンドに含まれるかを数える
        counts = np.zeros(n_fft // 2 + 1, dtype=np.float32)
        for idx in self.band_indices:
            counts[idx] += 1
        self.register_buffer(
            "num_bands_per_freq",
            torch.tensor(counts, dtype=torch.float32),
            persistent=False,
        )  # shape (F_total,)
        self.mask_estimators = nn.ModuleList(
            [
                MaskEstimator(
                    input_dim=hidden_size,
                    band_indices=self.band_indices,
                    num_channels=num_channels,
                    mlp_expansion_factor=1,
                )
                for _ in range(num_stems)
            ]
        )

        self.pitch_id_embed = nn.Embedding(self.num_pitches, hidden_size)
        self.register_buffer(
            "pitch_ids",
            torch.arange(self.num_pitches, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x):
        # x: [B, C*S, T, F]
        if self.use_gradient_checkpoint or self.training:
            checkpoint = torch.utils.checkpoint.checkpoint
        else:
            checkpoint = checkpointByPass

        x = einops.rearrange(x, "b c t f -> b t c f")

        # 勾配が流れない問題に対処
        x = checkpoint(
            self.band_split, x, use_reentrant=False
        )  # (B, T, K, hidden_size)

        # バンド軸にピッチクエリを追加
        B, T, K, D = x.shape
        pitch_query = self.pitch_id_embed(self.pitch_ids)  # [1, 1, E, D]
        pitch_query = pitch_query.expand(B, T, -1, -1)  # [B, T, E, D]
        x = torch.cat([x, pitch_query], dim=2)  # [B, T, K+E, D]

        x = F.pad(x, (0, 0, 1, 0, 1, 0))

        for layer in self.encoder_layers:
            x = checkpoint(layer, x, use_reentrant=False)
        # x: [B, T, K+E, D]

        x = x[:, 1:, 1:, :]

        # 音源分離マスク
        mask_list = []
        for mask_estimator in self.mask_estimators:
            pred_mask = checkpoint(
                mask_estimator, x[:, :, : self.num_bands], use_reentrant=False
            )  # [B, T, C*S, F]
            pred_mask = einops.rearrange(
                pred_mask, "b t (c s) f -> b t c f s", s=2
            ).contiguous()
            pred_mask = torch.view_as_complex(pred_mask)  # [B, T, C, F]
            mask_list.append(pred_mask)

        mask_all = torch.stack(mask_list, dim=-2)  # [B, T, C, N, F]
        # 周波数ビンで重なる部分を加算する
        B, T, C, N, F_concat = mask_all.shape
        F_total = self.n_fft // 2 + 1
        mask_sum = torch.zeros(
            (B, T, C, N, F_total), dtype=mask_all.dtype, device=mask_all.device
        )
        freq_idx = self.freq_indices.view(1, 1, 1, 1, -1).expand(
            B, T, C, N, -1
        )  # [B, T, C, N, F_concat]
        mask_sum.scatter_add_(
            dim=-1,
            index=freq_idx,
            src=mask_all,
        )

        # 重なった本数で割って平均
        denom = self.num_bands_per_freq.clamp(min=1e-8)  # [F_total]
        denom = denom.view(1, 1, 1, 1, -1)  # broadcast
        mask_avg = mask_sum / denom  # [B, T, C, N, F_total]
        mask_avg = einops.rearrange(mask_avg, "b t c n f -> b c n f t")

        # イベント列のみ取り出す
        event_x = x[:, :, -self.num_pitches :, :]  # [B, T, E, D]
        event_x = event_x.permute(0, 2, 1, 3).contiguous()  # [B, E, T, D]
        event_x = self.last_proj(event_x)
        return event_x, mask_avg
