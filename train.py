import torch
from torch.utils.data import DataLoader

from dataset import LMDataset
from model import GPT, GPTConfig


# ==========================================
# 1. DEVICE
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# ==========================================
# 2. MODEL CONFIG
# ==========================================

config = GPTConfig()


# ==========================================
# 3. DATASET
# ==========================================

dataset = LMDataset(
    "data/cleaned/data.jsonl",
    context_length=config.context_length
)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


# ==========================================
# 4. MODEL
# ==========================================

model = GPT(config)

model = model.to(device)


# ==========================================
# 5. OPTIMIZER
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)


# ==========================================
# 6. TRAINING MODE
# ==========================================

model.train()


# ==========================================
# 7. TRAINING LOOP
# ==========================================

for x, y in dataloader:

    # CPU -> GPU
    x = x.to(device)
    y = y.to(device)

    # --------------------------
    # FORWARD
    # --------------------------

    logits, loss = model(x, y)

    # --------------------------
    # GRADIENTLERİ TEMİZLE
    # --------------------------

    optimizer.zero_grad()

    # --------------------------
    # BACKPROPAGATION
    # --------------------------

    loss.backward()

    # --------------------------
    # WEIGHT UPDATE
    # --------------------------

    optimizer.step()

    # --------------------------
    # LOSS'U GÖSTER
    # --------------------------

    print("Loss:", loss.item())
