from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 128
    d_model: int = 384


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        # Token ID -> embedding vector
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        # Token'ın sequence içerisindeki pozisyonu -> embedding vector
        self.position_embedding = nn.Embedding(
            config.context_length,
            config.d_model
        )

    def forward(self, idx):

        B, T = idx.shape

        if T > self.config.context_length:
            raise ValueError(
                f"Sequence length {T}, "
                f"maximum context length "
                f"{self.config.context_length}."
            )

        # [B, T] -> [B, T, C]
        token_embeddings = self.token_embedding(idx)

        positions = torch.arange(
            0,
            T,
            device=idx.device
        )

        # [T] -> [T, C]
        position_embeddings = self.position_embedding(
            positions
        )

        # [B, T, C] + [T, C]
        x = token_embeddings + position_embeddings

        return x