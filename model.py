import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 128
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6  



class TransformerBlock(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        # Attention tarafı
        # BURADA self.ln1 OLUŞUYOR
        self.ln1 = nn.LayerNorm(
            config.d_model
        )

        # BURADA self.attention OLUŞUYOR
        # Yukarıdaki CausalSelfAttention class'ını kullanıyoruz
        self.attention = CausalSelfAttention(
            config
        )

        # MLP tarafı
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):

        # x:
        # [B,T,C]

        # =========================
        # 1. Attention
        # =========================

        # 1) Pre-Norm
        normalized_x = self.ln1(x)

        # 2) Multi-Head Causal Self-Attention
        attention_output = self.attention(
            normalized_x
        )

        # İlk residual connection
        x = x + attention_output

        # =========================
        # 2. MLP
        # =========================

        normalized_x = self.ln2(x)

        mlp_output = self.mlp(
            normalized_x
        )

        # İkinci residual connection
        x = x + mlp_output

        return x


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.d_model % config.n_heads == 0


        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

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

          # [B,T,C]
        # ->
        # [B,T,H,D]
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        # [B,T,H,D]
        # ->
        # [B,H,T,D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attention_scores = q @ k.transpose(-2, -1)

        attention_scores = (
            attention_scores
            / math.sqrt(self.head_dim)
        )

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

        # [B,H,T,T] @ [B,H,T,D]
        # ->
        # [B,H,T,D]
        output = attention_weights @ v

        # [B,H,T,D]
        # ->
        # [B,T,H,D]
        output = output.transpose(1, 2)

        # [B,T,H,D]
        # ->
        # [B,T,C]
        output = output.contiguous().view(
            B,
            T,
            C
        )

        return output


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        hidden_dim = 4 * config.d_model

        self.fc1 = nn.Linear(
            config.d_model,
            hidden_dim
        )

        self.activation = nn.GELU()

        self.fc2 = nn.Linear(
            hidden_dim,
            config.d_model
        )

    def forward(self, x):

        # [B,T,384] -> [B,T,1536]
        x = self.fc1(x)

        # Shape değişmez
        # [B,T,1536]
        x = self.activation(x)

        # [B,T,1536] -> [B,T,384]
        x = self.fc2(x)

        return x


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

        self.blocks = nn.ModuleList(
            [
            TransformerBlock(config)
            for _ in range(config.n_layers)
            ]
        )

        self.final_ln = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
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

        for block in self.blocks:
            x = block(x)


        x = self.final_ln(x)  

        logits = self.lm_head(x)

        return logits