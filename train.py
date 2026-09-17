from pathlib import Path

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

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
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

num_epochs = 5

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs
)


# ==========================================
# 6. CHECKPOINT LOAD (if exists)
# ==========================================

start_epoch = 0

checkpoint_path = Path("checkpoints/latest.pt")

if checkpoint_path.exists():

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    print(
        f"Checkpoint loaded. "
        f"Continuing from epoch {start_epoch + 1}."
    )


for epoch in range(start_epoch, num_epochs):

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

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

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

    perplexity = torch.exp(torch.tensor(average_val_loss))

    scheduler.step()


    # ======================================
    # RESULTS
    # ======================================

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Val Loss: {average_val_loss:.4f} | "
        f"Perplexity: {perplexity:.2f} | "
        f"LR: {current_lr:.6f}"
    )


    # ======================================
    # CHECKPOINT SAVE
    # ======================================

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": average_train_loss,
        "val_loss": average_val_loss,
    }

    torch.save(
        checkpoint,
        "checkpoints/latest.pt"
    )
