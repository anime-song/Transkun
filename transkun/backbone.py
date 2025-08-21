import math

import torch
import torch.nn as nn
import torch.utils.checkpoint
import librosa
import einops
from .transformer import Transformer, RMSNorm
from .band_split_roformer import BSRoformer
from typing import List


def checkpoint_bypass(f, *args):
    return f(*args)


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


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        num_extra_windows: int = 5,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.num_extra_windows = num_extra_windows
        self.log_eps = 1e-5

        self.register_buffer("hann_window", torch.hann_window(self.n_fft))
        self.extra_windows = GaussianWindows(num_windows=num_extra_windows, window_size=n_fft)
        self.n_channel = 1 + num_extra_windows

        mel_filter = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=0.0,
            fmax=None,
            htk=False,
            norm="slaney",
        )  # shape: (n_mels, n_fft//2 + 1)
        self.register_buffer("mel_filter", torch.tensor(mel_filter, dtype=torch.float32))

    def _stft(self, waveform: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        """複素 STFT → パワースペクトル（|X|²）を返す。"""
        spec = torch.stft(
            input=waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            return_complex=True,
            normalized=False,
        )
        power_spec = spec.abs().pow(2)  # (B, F, T)
        return power_spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)  … 16-bit PCM を −1〜1 正規化した波形を想定
        Returns:
            mel: (B, n_channel, n_mels, time)
        """
        batch, channels, time = x.shape
        waveform = einops.rearrange(x, "b c t -> (b c) t")  # → (B*C, T)

        windows: list[torch.Tensor] = [self.hann_window]

        if self.extra_windows is not None:
            windows.extend(self.extra_windows())  # append individual tensors

        mel_specs = []
        for win in windows:
            power_spec = self._stft(waveform, win.to(x.device))  # (B*C, F, T)
            mel = torch.matmul(self.mel_filter, power_spec)  # (n_mels, T)
            mel = torch.log(mel + self.log_eps)  # log-mel
            mel_specs.append(mel)  # list of (B*C, n_mels, T)

        mel_stack = torch.stack(mel_specs, dim=1)  # (B*C, n_channel, n_mels, T)
        mel_stack = einops.rearrange(mel_stack, "(b c) ch f t -> b (c ch) f t", b=batch, c=channels)
        return mel_stack


class Backbone(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        num_channels: int,
        n_fft: int,
        hop_size: int,
        n_mels: int,
        hidden_size: int,
        num_heads: int,
        target_midi_pitches: list[int],
        scoring_expansion_factor: int = 1,
        ffn_hidden_size_factor: int = 2,
        dropout: float = 0.0,
        num_layers: int = 4,
        use_gradient_checkpoint=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_pitches = len(target_midi_pitches)
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.num_channels = num_channels
        self.n_fft = n_fft
        self.window_size = n_fft
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size

        num_stems = 6
        self.mss_model = BSRoformer(
            dim=256,
            num_layers=12,
            sample_rate=44100,
            num_channels=2,
            head_dim=64,
            num_heads=8,
            n_fft=2048,
            hop_length=512,
            num_stems=num_stems,
        )
        self.mss_model.eval()
        for param in self.mss_model.parameters():
            param.requires_grad = False

        self.mel_spectrogram = MelSpectrogram(
            sampling_rate=sampling_rate, n_fft=n_fft, hop_length=hop_size, n_mels=n_mels
        )

        self.conv1 = nn.Conv2d(self.num_channels * 6 * num_stems, hidden_size, kernel_size=3, padding=1)

        self.down_conv = nn.Sequential(
            nn.ConstantPad2d((2, 1, 4, 3), value=0.0),
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, padding=1, stride=(2, 1)),
            nn.GroupNorm(4, hidden_size * 2),
            nn.GELU(),
            nn.Conv2d(
                hidden_size * 2,
                hidden_size * 4,
                kernel_size=3,
                padding=1,
                stride=(2, 2),
            ),
            nn.GroupNorm(4, hidden_size * 4),
            nn.GELU(),
            nn.Conv2d(
                hidden_size * 4,
                hidden_size * 4,
                kernel_size=3,
                padding=1,
                stride=(2, 2),
            ),
            nn.GroupNorm(4, hidden_size * 4),
            nn.GELU(),
            nn.Conv2d(hidden_size * 4, hidden_size * 4, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_size * 4),
        )

        self.pitch_id_embed = nn.Embedding(self.num_pitches, hidden_size * 4)
        self.register_buffer(
            "pitch_ids",
            torch.arange(self.num_pitches, dtype=torch.long),
            persistent=False,
        )
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            time_roformer = Transformer(
                input_dim=hidden_size * 4,
                head_dim=hidden_size * 4 // num_heads,
                num_layers=1,
                num_heads=num_heads,
                ffn_hidden_size_factor=ffn_hidden_size_factor,
                dropout=dropout,
            )
            band_roformer = Transformer(
                input_dim=hidden_size * 4,
                head_dim=hidden_size * 4 // num_heads,
                num_layers=1,
                num_heads=num_heads,
                ffn_hidden_size_factor=ffn_hidden_size_factor,
                dropout=dropout,
            )
            self.layers.append(nn.ModuleList([time_roformer, band_roformer]))
        self.final_norm = RMSNorm(hidden_size * 4)

        self.up_conv = nn.ConvTranspose1d(
            hidden_size * 4,
            hidden_size * scoring_expansion_factor,
            kernel_size=8,
            stride=8,
        )

    def forward(self, x):
        # x: (B, C, T)
        if self.use_gradient_checkpoint or self.training:
            checkpoint = torch.utils.checkpoint.checkpoint
        else:
            checkpoint = checkpoint_bypass

        # 音源分離
        with torch.no_grad():
            stems = self.mss_model(x)  # (B, N, C, T)
        x = einops.rearrange(stems, "b n c t -> b (n c) t")

        x = self.mel_spectrogram(x)  # (B, C, n_mels, T)
        x = x.to(memory_format=torch.channels_last)
        original_time_steps = x.shape[-1]

        # 正規化
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-8
        x = (x - mean) / std

        x = self.conv1(x)
        # ダウンサンプリング
        x = einops.rearrange(x, "b c f t -> b c t f")
        x = self.down_conv(x)
        x = einops.rearrange(x, "b c t f -> b t f c")  # (B, downT, F, C)

        # バンド軸にピッチクエリを追加
        B, T, _, _ = x.shape
        pitch_query = self.pitch_id_embed(self.pitch_ids)  # [1, 1, E, D]
        pitch_query = pitch_query.expand(B, T, -1, -1)  # [B, T, E, D]
        x = torch.cat([x, pitch_query], dim=2)  # [B, T, K+E, D]
        for time_roformer, band_roformer in self.layers:
            B, T, K, F = x.shape
            # 時間軸Transformer
            x = einops.rearrange(x, "b t k f -> (b k) t f")  # [B*K, T, F]
            x = checkpoint(time_roformer, x, use_reentrant=False)
            x = einops.rearrange(x, "(b k) t f -> b t k f", k=K)  # [B, T, K, F]

            # バンド軸Transformer
            x = x.reshape(B * T, K, F)  # [B*T, K, F]
            x = checkpoint(band_roformer, x, use_reentrant=False)
            x = x.reshape(B, T, K, F)
        # x: [B, T, K+E, D]
        x = self.final_norm(x)

        # アップサンプリング
        x = x[:, :, -self.num_pitches :, :]

        x = einops.rearrange(x, "b t e d -> (b e) d t")
        x = self.up_conv(x)
        x = einops.rearrange(x, "(b e) d t -> b e t d", e=self.num_pitches)
        x = x[:, :, :original_time_steps]
        return x
