from dataset import LMDataset

from torch.utils.data import DataLoader


dataset = LMDataset(
    "data/cleaned/data.jsonl",
    context_length=16
)

print("\nDataset uzunluğu:")
print(len(dataset))


if len(dataset) > 0:

    x, y = dataset[0]

    print("\nX:")
    print(x)

    print("\nY:")
    print(y)

    print("\nX shape:")
    print(x.shape)

    print("\nY shape:")
    print(y.shape)

    print("\nX decoded:")
    print(
        dataset.tokenizer.decode(
            x.tolist()
        )
    )

    print("\nY decoded:")
    print(
        dataset.tokenizer.decode(
            y.tolist()
        )
    )


loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)



for x_batch, y_batch in loader:

    print("\nBatch X shape:")
    print(x_batch.shape)

    print("Batch Y shape:")
    print(y_batch.shape)

    break