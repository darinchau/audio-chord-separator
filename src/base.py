import os
import torch
import librosa
from dataclasses import dataclass
from abc import ABC, abstractmethod


class ChordExtractor(ABC):
    @abstractmethod
    @staticmethod
    def get_latent_dimension() -> int:
        """
        Returns the dimension of the latent space used by the chord extractor. (=latent_dimension)
        """
        pass

    @abstractmethod
    @staticmethod
    def get_feature_hz() -> float:
        """
        Returns the feature extraction rate in Hertz (Hz).
        It should be the case int(n_samples / sr * feature_hz) = n_frames.
        """
        pass

    @abstractmethod
    @staticmethod
    def get_mapping() -> list[str]:
        """
        Returns a mapping from class indices to chord labels.
        """
        pass

    @abstractmethod
    def extract_logits(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Extracts logits from the given audio tensor.

        Args:
            audio (torch.Tensor): The input audio tensor in shape (n_channels, n_samples).
            sr (int): The sample rate of the audio.

        Returns:
            torch.Tensor: A tensor containing the extracted chords in shape (n_frames, n_classes).
        """
        pass

    def extract_chords(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Extracts chords from the given audio tensor.

        Args:
            audio (torch.Tensor): The input audio tensor in shape (n_channels, n_samples).
            sr (int): The sample rate of the audio.

        Returns:
            torch.Tensor: A int ensor containing the extracted chords in shape (n_frames,).
        """
        return torch.argmax(self.extract_logits(audio, sr), dim=-1)

    @abstractmethod
    def extract_latents(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Extracts latents from the given audio tensor.

        Args:
            audio (torch.Tensor): The input audio tensor in shape (n_channels, n_samples).
            sr (int): The sample rate of the audio.

        Returns:
            torch.Tensor: A float tensor containing the extracted latents in shape (n_frames, latent_dimension).
        """
        pass
