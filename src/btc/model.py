# Contains the definition of the Chord BTC model
# Code originally taken from https://github.com/jayg996/BTC-ISMIR19/tree/master

from .utils import get_small_voca, get_voca
import os
import torch
import librosa
from dataclasses import dataclass
from typing import Any
from .chord_modules import *
import warnings
import numpy as np
import torchaudio.functional as F
from ..base import ChordExtractor
import math


@dataclass
class Hyperparameters:
    mp3: dict[str, float]
    feature: dict[str, Any]
    model: dict[str, Any]


def get_default_config() -> Hyperparameters:
    return Hyperparameters(
        mp3={
            'song_hz': 22050,
            'inst_len': 10.0,
            'skip_interval': 5.0
        },
        feature={
            'n_bins': 144,
            'bins_per_octave': 24,
            'hop_length': 2048,
            'large_voca': False
        },
        model={
            'feature_size': 144,
            'timestep': 108,
            'num_chords': 25,
            'input_dropout': 0.2,
            'layer_dropout': 0.2,
            'attention_dropout': 0.2,
            'relu_dropout': 0.2,
            'num_layers': 8,
            'num_heads': 4,
            'hidden_size': 128,
            'total_key_depth': 128,
            'total_value_depth': 128,
            'filter_size': 128,
            'loss': 'ce',
            'probs_out': False
        },
    )


def get_model(model_path: str, device: torch.device, use_voca: bool) -> tuple[BTCModel, Hyperparameters, Any, Any]:
    # Init config
    config = get_default_config()

    if use_voca:
        config.feature['large_voca'] = True
        config.model['num_chords'] = 170
        if "large_voca" not in model_path:
            model_path = model_path.replace("model.pt", "model_large_voca.pt")

    # Load the model
    model = BTCModel(config=config.model).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    mean = checkpoint['mean']
    std = checkpoint['std']
    try:
        model.load_state_dict(checkpoint['model'])
    except Exception as e:
        raise ValueError("Model cannot load checkpoint. Perhaps you provided the large voca model for small voca or vice versa.") from e
    del model.output_layer.lstm
    return model, config, mean, std


@dataclass
class ChordModelOutput:
    logits: torch.Tensor
    features: torch.Tensor
    duration: float

    def save(self, path: str):
        np.savez_compressed(
            path,
            logits=self.logits.numpy(),
            features=self.features.numpy(),
        )

    @staticmethod
    def load(path: str) -> 'ChordModelOutput':
        file = np.load(path)
        return ChordModelOutput(
            duration=float(file['duration']),
            logits=torch.tensor(file['logits']),
            features=torch.tensor(file['features']),
        )


class SmallBTCExtractor(ChordExtractor):
    def __init__(self, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        model_path = os.path.join(os.path.dirname(__file__), 'pretrained', 'btc_model.pt')
        self.model, self.config, self.mean, self.std = get_model(model_path, self.device, use_voca=False)
        fps = self.config.model['timestep'] / self.config.mp3['inst_len']
        assert math.isclose(fps, self.get_feature_hz()), f"Feature hz ({self.get_feature_hz()}) does not match the model config ({fps})."

    @staticmethod
    def get_latent_dimension() -> int:
        return 128

    @staticmethod
    def get_feature_hz() -> float:
        return 10.8

    @staticmethod
    def get_mapping() -> list[str]:
        return get_small_voca()

    def extract_logits(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        return self.extract(audio, sr).logits

    def extract_latents(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        return self.extract(audio, sr).features

    def extract(self, waveform: torch.Tensor, sr: int) -> ChordModelOutput:
        """ Extracts logits and features from the given waveform.

        Args:
            waveform (torch.Tensor): The input audio tensor in shape (n_channels, n_samples)
            sr (int): The sample rate of the audio.
        Returns:
            ChordModelOutput: The extracted logits and features.
        """
        waveform = waveform.mean(dim=0) if waveform.ndim == 2 else waveform  # Convert to mono if stereo
        results = inference(waveform, sr, self.config, self.mean, self.std, self.model, self.device)
        latents = results.features
        expected_length = math.ceil(self.get_feature_hz() * waveform.shape[-1] / sr)
        if latents.shape[0] != expected_length:
            raise ValueError(f"Latents length {latents.shape[0]} does not match expected length {expected_length} (input shape: {waveform.shape}).")
        if latents.shape[1] != self.get_latent_dimension():
            raise ValueError(f"Latents dimension {latents.shape[1]} does not match expected dimension {self.get_latent_dimension()}.")
        logits = results.logits
        if logits.shape[0] != expected_length:
            raise ValueError(f"Logits length {logits.shape[0]} does not match expected length {expected_length}.")
        if logits.shape[1] != len(self.get_mapping()):
            raise ValueError(f"Logits dimension {logits.shape[1]} does not match expected dimension {len(self.get_mapping())}.")
        return results


class LargeBTCExtractor(SmallBTCExtractor):
    def __init__(self, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        model_path = os.path.join(os.path.dirname(__file__), 'pretrained', 'btc_model_large_voca.pt')
        self.model, self.config, self.mean, self.std = get_model(model_path, self.device, use_voca=True)
        fps = self.config.model['timestep'] / self.config.mp3['inst_len']
        assert math.isclose(fps, self.get_feature_hz()), f"Feature hz ({self.get_feature_hz()}) does not match the model config ({fps})."

    @staticmethod
    def get_mapping() -> list[str]:
        return get_voca()


def inference(waveform: torch.Tensor, sr: int, config: Hyperparameters, mean, std, model, device) -> ChordModelOutput:
    # Handle audio and resample to the requied sr
    audio_duration = waveform.shape[-1] / sr
    original_wav: np.ndarray = F.resample(waveform, sr, 22050).detach().cpu().numpy()
    sr = 22050

    # Compute audio features
    currunt_sec_hz = 0
    feature = np.array([])
    while len(original_wav) > currunt_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len']:
        start_idx = int(currunt_sec_hz)
        end_idx = int(currunt_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len'])
        tmp = librosa.cqt(
            original_wav[start_idx:end_idx],
            sr=sr,
            n_bins=config.feature['n_bins'],
            bins_per_octave=config.feature['bins_per_octave'],
            hop_length=config.feature['hop_length']
        )
        if start_idx == 0:
            feature = tmp
        else:
            feature = np.concatenate((feature, tmp), axis=1)
        currunt_sec_hz = end_idx

    # Concatenate the last part of the audio onto the feature
    try:
        tmp = librosa.cqt(
            original_wav[currunt_sec_hz:],
            sr=sr,
            n_bins=config.feature['n_bins'],
            bins_per_octave=config.feature['bins_per_octave'],
            hop_length=config.feature['hop_length']
        )
    except Exception:
        # Last part is too short, pad one frame of silence
        tmp = np.zeros((config.feature['n_bins'], 1), dtype=np.complex64)

    if currunt_sec_hz == 0:
        feature = tmp
    else:
        feature = np.concatenate((feature, tmp), axis=1)

    feature = np.log(np.abs(feature) + 1e-6)

    num_features = feature.shape[1]

    # Process features
    feature = feature.T
    feature = (feature - mean) / std
    n_timestep = config.model['timestep']
    num_pad = n_timestep - (feature.shape[0] % n_timestep)
    feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
    num_instance = feature.shape[0] // n_timestep

    # Inference
    features = []
    logits = []
    with torch.no_grad():
        model.eval()
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        for t in range(num_instance):
            self_attn_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
            features.append(self_attn_output)
            prediction, logit = model.output_layer(self_attn_output)
            prediction = prediction.squeeze()
            logits.append(logit)
    features = torch.cat(features, dim=1)[0].cpu().numpy()
    logits = torch.cat(logits, dim=1)[0].cpu().numpy()
    features = features[:num_features, :]
    logits = logits[:num_features, :]
    return ChordModelOutput(
        duration=audio_duration,
        logits=torch.tensor(logits),
        features=torch.tensor(features),
    )
