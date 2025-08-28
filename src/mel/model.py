import torch
import torch.nn as nn
from torch import Tensor
from torchaudio.transforms import MelScale
from dataclasses import dataclass


# @dataclass(frozen=True)
# class MelChordModelConfig:
#     # Audio processing parameters
#     sr: int  # Sample rate of the audio input
#     n_fft: int  # Number of FFT components for STFT
#     hop_length: int  # Number of audio samples between adjacent STFT columns
#     win_length: int  # Size of window function for STFT
#     n_mels: int  # Number of mel filter banks
#     f_min: float  # Lowest frequency (Hz) for mel scale
#     f_max: float  # Highest frequency (Hz) for mel scale

#     # Model architecture parameters
#     latent_dim: int  # Dimensionality of the model's hidden representations
#     layer_dropout: float = 0.2  # Dropout rate applied between transformer layers
#     attention_dropout: float = 0.2  # Dropout rate within attention mechanisms
#     relu_dropout: float = 0.2  # Dropout rate in feed-forward ReLU layers
#     num_layers: int = 8  # Number of transformer/attention layers in the model
#     num_heads: int = 4  # Number of attention heads in multi-head attention
#     max_seq_len: int = 1024  # Maximum sequence length for positional encoding


class LinearSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        center=False,
        mode="pow2_sqrt",
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.mode = mode

        self.window: Tensor
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, y: Tensor) -> Tensor:
        if y.ndim == 3:
            y = y.squeeze(1)

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                (self.win_length - self.hop_length) // 2,
                (self.win_length - self.hop_length + 1) // 2,
            ),
            mode="reflect",
        ).squeeze(1)
        dtype = y.dtype
        spec = torch.stft(
            y.float(),
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)

        if self.mode == "pow2_sqrt":
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        spec = spec.to(dtype)
        return spec


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        center=False,
        f_min=0.0,
        f_max=None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2

        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length, center)
        self.mel_scale = MelScale(
            self.n_mels,
            self.sample_rate,
            self.f_min,
            self.f_max,
            self.n_fft // 2 + 1,
            "slaney",
            "slaney",
        )

    def compress(self, x: Tensor) -> Tensor:
        return torch.log(torch.clamp(x, min=1e-5))

    def decompress(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    def forward_linear(self, x: Tensor) -> tuple[Tensor, Tensor]:
        linear = self.spectrogram(x)
        x = self.mel_scale(linear)
        x = self.compress(x)
        return x, self.compress(linear)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_linear(x)[0]


class SelfAttentionBlock(nn.Module):
    def __init__(self, latent_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len).float()
        sinusoid_inp = torch.einsum("i , j -> i j", position, inv_freq)
        pos_emb = torch.zeros((max_seq_len, dim))
        pos_emb[:, 0::2] = torch.sin(sinusoid_inp)
        pos_emb[:, 1::2] = torch.cos(sinusoid_inp)

        self.pos_emb: Tensor
        self.register_buffer("pos_emb", pos_emb)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        return self.pos_emb[:seq_len, :].unsqueeze(0)  # Shape: (1, seq_len, dim)


class MelChordModel(nn.Module):
    def __init__(
        self,
        n_mels: int,
        hidden_size: int,
        layer_dropout: float,
        attention_dropout: float,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, hidden_size)
        self.positional_encoding = RotaryPositionalEncoding(hidden_size, max_seq_len)

        self.layers = nn.ModuleList([
            SelfAttentionBlock(
                hidden_size,
                num_heads,
                attention_dropout
            ) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(layer_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        x = x + self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)
        return x
