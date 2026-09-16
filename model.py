import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 128
    d_model: int = 384

class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.d_model = config.d_model

        self.query = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False
        )

        self.key = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False
        )

        self.value = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False
        )

    def forward(self, x):

        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_scores = q @ k.transpose(-2, -1)

        attention_scores = attention_scores / math.sqrt(C)

        # 1) Kim kime bakabilir?
        mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        )

        # 2) Geleceğe bakılan yerleri -∞ yap
        attention_scores = attention_scores.masked_fill(
            ~mask,
            float("-inf")
        )

        # 3) Skorları attention ağırlıklarına çevir
        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )

        # 4) Bu ağırlıklara göre Value'lardan bilgi topla
        output = attention_weights @ v

        return output



    
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