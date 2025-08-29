import wandb
import os
import math
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, asdict
from typing import Any
import random
import warnings
from tqdm import tqdm
import gc

from src.mel.model import LogMelSpectrogram, MelChordModel
from src.btc.model import LargeBTCExtractor
from src.dataset.src import YouTubeURL, AUDIO_DIR, load_audio, iterate_urls
from src.dataset.src.util import get_filepath


@dataclass(frozen=True)
class TrainMelConfig:
    # Audio processing parameters
    sr: int = 44100  # Sample rate of the audio input
    n_fft: int = 2048  # Number of FFT components for STFT
    hop_length: int = 512  # Number of audio samples between adjacent STFT columns
    win_length: int = 2048  # Size of window function for STFT
    n_mels: int = 128  # Number of mel filter banks
    f_min: float = 0.0  # Lowest frequency (Hz) for mel scale
    f_max: float | None = None  # Highest frequency (Hz) for mel scale
    latent_dim: int = 128  # Dimensionality of the feature

    # Model architecture parameters
    layer_dropout: float = 0.2  # Dropout rate applied between transformer layers
    attention_dropout: float = 0.2  # Dropout rate within attention mechanisms
    relu_dropout: float = 0.2  # Dropout rate in feed-forward ReLU layers
    num_layers: int = 8  # Number of transformer/attention layers in the model
    num_heads: int = 4  # Number of attention heads in multi-head attention
    max_seq_len: int = 1024  # Maximum sequence length for positional encoding
    hidden_size: int = 256  # Dimensionality of the latent dimension

    # Training parameters
    batch_size: int = 16
    max_step: int = 1_000_000
    learning_rate: float = 1e-4
    segment_duration: float = 10.0
    resume_from: str | None = None  # Path to checkpoint to resume training from
    save_interval: int = 10_000  # Steps between saving model checkpoints
    val_interval: int = 1_000  # Steps between validation runs
    clear_cuda_cache_interval: int = 100  # Steps between clearing CUDA cache
    num_workers: int = 4  # Number of DataLoader workers
    save_dir: str = "./checkpoints"  # Directory to save model checkpoints
    use_wandb: bool = True  # Whether to use Weights & Biases for logging

    @property
    def multiplier(self) -> int:
        # Ratio between audio time resolution and BTC features time resolution
        return 8


class WrappedMelChordModel(nn.Module):
    def __init__(self, config: TrainMelConfig):
        super().__init__()
        self.mel_chord_model = MelChordModel(
            n_mels=config.n_mels,
            hidden_size=config.hidden_size,
            layer_dropout=config.layer_dropout,
            attention_dropout=config.attention_dropout,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_len
        )
        self.proj = nn.Linear(config.hidden_size, config.latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, n_mels)
        hidden = self.mel_chord_model(x)  # (B, T, hidden_size)
        out = self.proj(hidden)  # (B, T, latent_dim)
        return out


class AudioChordLatentDataset(Dataset):
    def __init__(self, config: TrainMelConfig, urls: list[YouTubeURL]):
        self.urls = urls
        self.btc_extractor = LargeBTCExtractor()
        self.mel_spectrogram = LogMelSpectrogram(
            sample_rate=config.sr,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
        )
        self.config = config
        self.segment_length = int(self.sr * self.segment_duration)

    @property
    def sr(self) -> int:
        return self.config.sr

    @property
    def segment_duration(self) -> float:
        return self.config.segment_duration

    def __len__(self) -> int:
        return len(self.urls)

    def get_mel_spec_latents(self, idx: int) -> tuple[Tensor, Tensor]:
        url = self.urls[idx]
        audio_path = get_filepath(AUDIO_DIR, url.video_id)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                audio, _ = load_audio(audio_path, sr=self.sr)
            except Exception as e:
                raise ValueError(f"Failed to load audio for {audio_path}") from e
        if audio is None:
            raise ValueError(f"Failed to load audio for URL: {url}")
        starting = random.randint(0, max(0, audio.shape[-1] - self.segment_length))
        audio = audio[:, starting:starting + self.segment_length]
        if audio.shape[-1] < self.segment_length:
            raise ValueError(f"Audio segment is shorter than expected: {audio.shape[-1]} < {self.segment_length}")

        melspec = self.mel_spectrogram(audio)  # (C, n_mels, T1)
        latents = self.btc_extractor.extract_latents(audio, self.sr)  # (T2, latent_dim)

        # Dumb way to align the time resolution of mel spectrogram and latents, oh well
        tiling = torch.ones((latents.shape[0],), dtype=torch.int32) * self.config.multiplier
        tiling[-1] = 0
        last_frame = melspec.shape[-1] - self.config.multiplier * latents.shape[0] + self.config.multiplier
        if last_frame < 0:
            raise ValueError(f"Mel spectrogram has more frames than expected: {melspec.shape[-1]} > {torch.sum(tiling).item()}")
        elif last_frame > self.config.multiplier:
            raise ValueError(f"Last frame tiling is too large: {last_frame} > {self.config.multiplier} (mel frames: {melspec.shape[1]}, latent frames: {latents.shape[0]})")
        tiling[-1] += last_frame
        latents = torch.repeat_interleave(latents, tiling, dim=0)  # (T1, latent_dim)

        melspec = melspec.mean(dim=0) if melspec.ndim == 3 else melspec  # (n_mels, T1)
        melspec = melspec.T  # (T1, n_mels)

        return melspec, latents

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        while True:
            try:
                return self.get_mel_spec_latents(idx)
            except Exception as e:
                idx = random.randint(0, len(self.urls) - 1)
                print(f"Warning: {e}. Retrying with segment {idx}.")


def parse_args() -> TrainMelConfig:
    parser = argparse.ArgumentParser(description="Train Mel to Chord Latent Model")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--max_step', type=int, default=1_000_000, help='Maximum number of training steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--segment_duration', type=float, default=10.0, help='Duration of audio segments in seconds')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--save_interval', type=int, default=10_000, help='Steps between saving model checkpoints')
    parser.add_argument('--val_interval', type=int, default=1_000, help='Steps between validation runs')
    parser.add_argument('--clear_cuda_cache_interval', type=int, default=100, help='Steps between clearing CUDA cache')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use Weights & Biases for logging')
    args = parser.parse_args()

    return TrainMelConfig(
        batch_size=args.batch_size,
        max_step=args.max_step,
        learning_rate=args.learning_rate,
        segment_duration=args.segment_duration,
        resume_from=args.resume_from,
        save_interval=args.save_interval,
        val_interval=args.val_interval,
        clear_cuda_cache_interval=args.clear_cuda_cache_interval,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb
    )


def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    melspecs, latents = zip(*batch)
    melspecs = nn.utils.rnn.pad_sequence(melspecs, batch_first=True)  # (B, T1, n_mels) # type: ignore
    latents = nn.utils.rnn.pad_sequence(latents, batch_first=True)  # (B, T1, latent_dim) # type: ignore
    return melspecs, latents


def validate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for melspecs, latents in dataloader:
            melspecs = melspecs.to(device)
            latents = latents.to(device)
            outputs = model(melspecs)
            loss = criterion(outputs, latents)
            total_loss += loss.item() * melspecs.size(0)
    return total_loss / len(dataloader)


def get_dataloaders(
    config: TrainMelConfig,
) -> tuple[DataLoader, DataLoader]:
    urls = list(iterate_urls())
    train_urls = [url for url in urls if not url.video_id.endswith("_")]
    val_urls = [url for url in urls if url.video_id.endswith("_")]
    train_dataset = AudioChordLatentDataset(config, train_urls)
    val_dataset = AudioChordLatentDataset(config, val_urls)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True
    )
    return train_loader, val_loader


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    directory: str,
    global_step: int,
    config: TrainMelConfig
):
    path = os.path.join(directory, f"ckpt_step_{global_step}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'config': asdict(config),
    }, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str
):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_step = checkpoint['global_step']
    print(f"Loaded checkpoint from {path} at step {global_step}")
    return global_step


def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainMelConfig,
    global_step: int,
    device: torch.device
):
    pbar = tqdm(total=config.max_step, desc="Training Progress", unit="step")
    pbar.update(global_step)
    stop_training = False
    while True:
        for melspecs, latents in train_loader:
            model.train()
            melspecs = melspecs.to(device)
            latents = latents.to(device)

            optimizer.zero_grad()
            outputs = model(melspecs)
            loss = criterion(outputs, latents)
            loss.backward()
            optimizer.step()

            global_step += 1
            if config.use_wandb:
                wandb.log({"train/loss": loss.item(), "step": global_step})

            if global_step % config.val_interval == 0 and global_step > 0:
                val_loss = validate(model, criterion, val_loader, device)
                if config.use_wandb:
                    wandb.log({"val/loss": val_loss, "step": global_step})
                print(f"Step {global_step}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

            if global_step % config.save_interval == 0 and global_step > 0:
                os.makedirs(config.save_dir, exist_ok=True)
                save_checkpoint(model, optimizer, config.save_dir, global_step, config)
                print(f"Saved checkpoint at step {global_step}")

            if global_step % config.clear_cuda_cache_interval == 0:
                torch.cuda.empty_cache()
                gc.collect()

            if global_step >= config.max_step:
                stop_training = True
                break

            pbar.update(1)

        if stop_training:
            break
    pbar.close()


def main():
    config = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(config)

    model = WrappedMelChordModel(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4, betas=(0.9, 0.99), eps=1e-8)

    global_step = 0
    if config.resume_from:
        global_step = load_checkpoint(model, optimizer, config.resume_from)

    if config.use_wandb:
        wandb.init(project="mel-to-chord-latent", config=asdict(config))

    try:
        train(model, criterion, optimizer, train_loader, val_loader, config, global_step, device)
    finally:
        if config.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
