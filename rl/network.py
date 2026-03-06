"""Custom CNN feature extractor for the 2048 RL agent."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNN2048FeaturesExtractor(BaseFeaturesExtractor):
    """Two-layer CNN followed by a fully-connected projection layer.

    Architecture (input shape: ``(16, 4, 4)`` one-hot tensor)::

        Conv2d(16 → 128, kernel=2, pad=1) → ReLU   # out: (128, 5, 5)
        Conv2d(128 → 128, kernel=2)       → ReLU   # out: (128, 4, 4)
        Flatten                                      # out: 2048
        Linear(2048 → features_dim)       → ReLU   # out: features_dim

    The spatial padding in the first conv layer preserves edge information
    that is important for boundary-aware strategies in 2048.

    Args:
        observation_space: The (16, 4, 4) observation space.
        features_dim:       Dimensionality of the output feature vector.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 16

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 128, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Infer the flattened CNN output size dynamically
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
