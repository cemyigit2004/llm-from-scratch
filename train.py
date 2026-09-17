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
# 6. TRAINING
# ==========================================

num_epochs = 5

model.train()

for epoch in range(num_epochs):

    total_loss = 0.0

    for x, y in dataloader:

        x = x.to(device)
        y = y.to(device)

        # 1. Forward
        logits, loss = model(x, y)

        # 2. Eski gradientleri temizle
        optimizer.zero_grad()

        # 3. Backpropagation
        loss.backward()

        # 4. Weightleri güncelle
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    print(
        f"Epoch {epoch + 1}/{num_epochs} "
        f"- Loss: {average_loss:.4f}"
    )
