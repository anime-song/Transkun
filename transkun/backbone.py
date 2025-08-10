import math

import torch
import torch.nn as nn
import torch.utils.checkpoint
import numpy as np
import librosa
import einops
from .utils import checkpointByPass
from .transformer import TransformerLayer, RMSNorm

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
                nn.Linear(size, 2 * size * expansionFactor + 1),  # only inner product plus diagonal
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


def build_band_indices(sample_rate: int = 44_100, n_fft: int = 2_048) -> Tuple[np.ndarray, List[np.ndarray]]:
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
    def __init__(self, hidden_size: int, band_indices, num_channels: int, extra_windows: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.band_indices = band_indices

        for i, idx in enumerate(band_indices):
            self.register_buffer(f"band_idx_{i}", torch.tensor(idx, dtype=torch.long))

        self.num_bands = len(band_indices)
        self.projections = torch.nn.ModuleList([])
        for i in range(self.num_bands):
            sub_band_freqs = len(band_indices[i]) * num_channels * 2 * (extra_windows + 1)
            self.projections.append(
                torch.nn.Sequential(
                    RMSNorm(sub_band_freqs),
                    torch.nn.Linear(sub_band_freqs, hidden_size),
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
            sub_band_freqs = len(band_indices[i]) * num_channels * 2
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
        assert len(sub_band_list) == len(self.projections)
        for i, proj_layer in enumerate(self.projections):
            sub_band = proj_layer(sub_band_list[i])  # (B, T, sub_band_freqs)
            sub_band = einops.rearrange(
                sub_band,
                "b t (c f) -> b t c f",
                c=self.num_channels * 2,
                f=len(self.band_indices[i]),
            )
            out.append(sub_band)

        return torch.concat(out, dim=-1)  # (B, T, C, F)


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
        self.time_roformer = TransformerLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            ffn_hidden_size_factor=ffn_hidden_size_factor,
            dropout=dropout,
        )
        self.band_roformer = TransformerLayer(
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
        x = einops.rearrange(x, "b t k f -> (b k) t f")  # [B*K, T, F]
        x = self.time_roformer(x)
        x = einops.rearrange(x, "(b k) t f -> b t k f", k=K)  # [B, T, K, F]

        # バンド軸Transformer
        x = x.reshape(B * T, K, F)  # [B*T, K, F]
        x = self.band_roformer(x)
        x = x.reshape(B, T, K, F)
        return x


class GaussianWindows(nn.Module):
    """
    Learnable set of 1-D Gaussian windows.

    Args:
        num_windows:  追加ガウス窓の本数
        window_size:  FFT / STFT ウィンドウ長
    """

    def __init__(self, num_windows: int, window_size: int) -> None:
        super().__init__()
        self.num_windows = num_windows
        self.window_size = window_size

        self.logit_sigma = nn.Parameter(torch.full((num_windows,), -1.0))
        mu_init = torch.arange(1, num_windows + 1) / (num_windows + 1)
        self.logit_mu = nn.Parameter(torch.logit(mu_init))

    def forward(self) -> List[torch.Tensor]:
        """各ガウス窓 (length = window_size) をリストで返す。"""
        sigmas = torch.sigmoid(self.logit_sigma)  # (num_windows,)
        centers = torch.sigmoid(self.logit_mu)  # (num_windows,)

        x = torch.arange(self.window_size, device=self.logit_sigma.device).float()
        windows = []
        for sigma, mu in zip(sigmas, centers):
            exponent = -0.5 * ((x - self.window_size * mu) / (sigma * self.window_size / 2)) ** 2
            windows.append(torch.exp(exponent))  # shape: (window_size,)
        return windows


class Backbone(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        num_channels: int,
        n_fft: int,
        hop_size: int,
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
        self.window_size = n_fft
        self.band_split_type = band_split_type
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.num_stems = num_stems

        self.gaussian_windows = GaussianWindows(num_windows=5, window_size=self.window_size)
        self.register_buffer("hann_window", torch.hann_window(self.window_size))

        if band_split_type == "bs":
            _, self.band_indices = build_band_indices(sample_rate=sampling_rate, n_fft=n_fft)
        elif band_split_type == "mel":
            self.band_indices = build_mel_band_indices(sampling_rate=sampling_rate, n_fft=n_fft, n_bands=n_bands)
        else:
            raise NotImplementedError("サポートしていない band_split_type です")

        self.num_bands = len(self.band_indices)
        self.band_split = BandSplit(
            hidden_size=hidden_size,
            band_indices=self.band_indices,
            num_channels=self.num_channels,
            extra_windows=self.gaussian_windows.num_windows,
        )

        self.down_conv = nn.Sequential(
            nn.ConstantPad2d((0, 0, 4, 3), value=0.0),
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, padding=1, stride=(2, 1)),
            nn.GroupNorm(4, hidden_size * 2),
            nn.GELU(),
            nn.Conv2d(
                hidden_size * 2,
                hidden_size * 4,
                kernel_size=3,
                padding=1,
                stride=(2, 1),
            ),
            nn.GroupNorm(4, hidden_size * 4),
            nn.GELU(),
            nn.Conv2d(
                hidden_size * 4,
                hidden_size * 4,
                kernel_size=3,
                padding=1,
                stride=(2, 1),
            ),
            nn.GroupNorm(4, hidden_size * 4),
            nn.GELU(),
            nn.Conv2d(hidden_size * 4, hidden_size * 4, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_size * 4),
        )
        self.up_conv = nn.ConvTranspose1d(hidden_size * 4, hidden_size, kernel_size=8, stride=8)

        self.pitch_id_embed = nn.Embedding(self.num_pitches, hidden_size * 4)
        self.register_buffer(
            "pitch_ids",
            torch.arange(self.num_pitches, dtype=torch.long),
            persistent=False,
        )
        self.encoder_layers = nn.ModuleList(
            [
                BandSplitRoformerLayer(
                    input_size=hidden_size * 4,
                    hidden_size=hidden_size * 4,
                    num_heads=num_heads,
                    ffn_hidden_size_factor=ffn_hidden_size_factor,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
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

        self.last_proj = nn.Linear(self.hidden_size, self.hidden_size * scoring_expansion_factor)

    def to_spectrogram(self, inputs: torch.Tensor) -> torch.Tensor:
        # (B, C, T)
        inputs = einops.rearrange(inputs, "b c t -> (b c) t")

        windows: list[torch.Tensor] = [self.hann_window]
        windows.extend(self.gaussian_windows())

        spectrograms = []
        for win in windows:
            spectrogram = torch.stft(
                input=inputs,
                n_fft=self.n_fft,
                hop_length=self.hop_size,
                win_length=self.window_size,
                window=win.to(inputs.device),
                center=True,
                return_complex=True,
            )
            spectrogram = einops.rearrange(spectrogram, "(b c) f t -> b c f t", c=self.num_channels)
            spectrograms.append(spectrogram)  # list of [B, C, F, T]

        spectrogram = torch.concat(spectrograms, dim=1)
        spectrogram = torch.view_as_real(spectrogram)
        spectrogram = einops.rearrange(spectrogram, "b c f t s -> b (c s) t f", s=2)
        return spectrogram, spectrograms[0]

    def mixture_consistency_projection(
        self,
        source_estimates: torch.Tensor,  # [B, C, N, F, T] complex
        mixture: torch.Tensor,  # [B, C, F, T] complex
        weights: torch.Tensor | None = None,  # [B, C, N, F, T] real, 任意
    ) -> torch.Tensor:
        residual = mixture[:, :, None] - source_estimates.sum(dim=2, keepdim=True)
        if weights is None:
            correction = residual / source_estimates.shape[2]
        else:
            weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)
            correction = weights * residual
        return source_estimates + correction

    def to_recon_audio(
        self,
        mask: torch.Tensor,
        original_spec_complex: torch.Tensor,
        original_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        # mask: [B, C, N, F, T]
        source_estimates = original_spec_complex[:, :, None] * mask

        # mixture consistency
        magnitude = source_estimates.abs().clamp_min(1e-8)
        weights = magnitude**2
        separated_spectrogram = self.mixture_consistency_projection(
            source_estimates=source_estimates, mixture=original_spec_complex, weights=weights
        )

        separated_spectrogram = einops.rearrange(separated_spectrogram, "b c n f t -> (b c n) f t")
        recon_audio = torch.istft(
            separated_spectrogram,
            n_fft=self.window_size,
            hop_length=self.hop_size,
            win_length=self.window_size,
            window=self.hann_window.to(device),
            center=True,
            return_complex=False,
            length=original_length,
        )  # [B*C, T]
        recon_audio = einops.rearrange(recon_audio, "(b c n) t -> b c n t", c=self.num_channels, n=self.num_stems)
        return recon_audio

    def forward(self, x):
        # x: (B, C, T)
        if self.use_gradient_checkpoint or self.training:
            checkpoint = torch.utils.checkpoint.checkpoint
        else:
            checkpoint = checkpointByPass

        istft_length = x.shape[-1]
        x, original_spec_complex = self.to_spectrogram(x)  # x: [B, C*S, T, F]

        x = einops.rearrange(x, "b c t f -> b t c f")
        original_time_steps = x.shape[1]

        # 勾配が流れない問題に対処
        x = checkpoint(self.band_split, x, use_reentrant=False)  # (B, T, K, hidden_size)

        # ダウンサンプリング
        x = x.permute(0, 3, 1, 2)  # (B, hidden_size, T, K)
        x = self.down_conv(x)
        x = x.permute(0, 2, 3, 1)  # (B, downT, K, hidden_size*4)

        # バンド軸にピッチクエリを追加
        B, T, _, _ = x.shape
        pitch_query = self.pitch_id_embed(self.pitch_ids)  # [1, 1, E, D]
        pitch_query = pitch_query.expand(B, T, -1, -1)  # [B, T, E, D]
        x = torch.cat([x, pitch_query], dim=2)  # [B, T, K+E, D]
        for layer in self.encoder_layers:
            x = checkpoint(layer, x, use_reentrant=False)
        # x: [B, T, K+E, D]

        # アップサンプリング
        x = einops.rearrange(x, "b t ke d -> (b ke) d t")
        x = self.up_conv(x)
        x = einops.rearrange(x, "(b ke) d t -> b t ke d", ke=self.num_bands + self.num_pitches)
        x = x[:, :original_time_steps]

        # 音源分離マスク
        mask_list = []
        for mask_estimator in self.mask_estimators:
            pred_mask = checkpoint(mask_estimator, x[:, :, : self.num_bands], use_reentrant=False)  # [B, T, C*S, F]
            pred_mask = einops.rearrange(pred_mask, "b t (c s) f -> b t c f s", s=2).contiguous()
            pred_mask = torch.view_as_complex(pred_mask)  # [B, T, C, F]
            mask_list.append(pred_mask)

        mask_all = torch.stack(mask_list, dim=-2)  # [B, T, C, N, F]
        if self.band_split_type == "mel":
            # 周波数ビンで重なる部分を加算する
            B, T, C, N, F_concat = mask_all.shape
            F_total = self.n_fft // 2 + 1
            mask_sum = torch.zeros((B, T, C, N, F_total), dtype=mask_all.dtype, device=mask_all.device)
            freq_idx = self.freq_indices.view(1, 1, 1, 1, -1).expand(B, T, C, N, -1)  # [B, T, C, N, F_concat]
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
        else:
            mask_avg = einops.rearrange(mask_all, "b t c n f -> b c n f t")

        recon_audio = self.to_recon_audio(
            mask=mask_avg,
            original_spec_complex=original_spec_complex,
            original_length=istft_length,
            dtype=x.dtype,
            device=x.device,
        )

        # イベント列のみ取り出す
        event_x = x[:, :, -self.num_pitches :, :]  # [B, T, E, D]
        event_x = event_x.permute(0, 2, 1, 3).contiguous()  # [B, E, T, D]
        event_x = self.last_proj(event_x)
        return event_x, recon_audio
