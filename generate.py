import torch
from pathlib import Path
from transformers import AutoTokenizer

from model import GPT, GPTConfig


# ==========================================
# 1. DEVICE
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# ==========================================
# 2. TOKENIZER
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    "openai-community/gpt2"
)


# ==========================================
# 3. MODEL
# ==========================================

config = GPTConfig()

model = GPT(config)
model = model.to(device)


# ==========================================
# 4. CHECKPOINT LOAD
# ==========================================

checkpoint_path = Path("checkpoints/latest.pt")

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ==========================================
# 5. PROMPT
# ==========================================

prompt = "Yapay zeka"

token_ids = tokenizer.encode(
    prompt,
    add_special_tokens=False
)

idx = torch.tensor(
    token_ids,
    dtype=torch.long,
    device=device
).unsqueeze(0)


# ==========================================
# 6. GENERATION
# ==========================================

max_new_tokens = 30
temperature = 0.8
top_k = 50

with torch.no_grad():

    for _ in range(max_new_tokens):

        # Context window'dan uzun olursa
        # sadece son tokenları kullan
        idx_context = idx[
            :,
            -config.context_length:
        ]

        # Model prediction
        logits, _ = model(idx_context)

        # Sadece SON pozisyonun logits'i
        next_token_logits = logits[:, -1, :]

        next_token_logits = (
            next_token_logits / temperature
        )

        top_k_values, _ = torch.topk(
            next_token_logits,
            k=top_k
        )

        threshold = top_k_values[:, -1].unsqueeze(-1)

        next_token_logits = next_token_logits.masked_fill(
            next_token_logits < threshold,
            float("-inf")
        )

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # Yeni tokenı dizinin sonuna ekle
        idx = torch.cat(
            [idx, next_token],
            dim=1
        )


# ==========================================
# 7. DECODE
# ==========================================

generated_text = tokenizer.decode(
    idx[0].tolist()
)

print("\nGenerated text:")
print(generated_text)
