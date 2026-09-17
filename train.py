import torch
from torch.utils.data import DataLoader, random_split

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

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False
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


num_epochs = 5

for epoch in range(num_epochs):

    # ======================================
    # TRAINING
    # ======================================

    model.train()

    total_train_loss = 0.0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        # Forward
        logits, loss = model(x, y)

        # Eski gradientleri temizle
        optimizer.zero_grad()

        # Gradientleri hesapla
        loss.backward()

        # Weightleri güncelle
        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = (
        total_train_loss / len(train_loader)
    )


    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            logits, loss = model(x, y)

            total_val_loss += loss.item()

    average_val_loss = (
        total_val_loss / len(val_loader)
    )


    # ======================================
    # RESULTS
    # ======================================

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Val Loss: {average_val_loss:.4f}"
    )
