import torch

from model import GPT, GPTConfig


config = GPTConfig(
    vocab_size=50257,
    context_length=128,
    d_model=384
)

model = GPT(config)


print("Token embedding matrix:")
print(model.token_embedding.weight.shape)

print("\nPosition embedding matrix:")
print(model.position_embedding.weight.shape)


# 2 sequence
# Her sequence 16 token
fake_input = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(2, 16)
)

print("\nInput shape:")
print(fake_input.shape)


output = model(fake_input)


print("\nOutput shape:")
print(output.shape)


print("\nİlk token'ın embedding'i:")
print(output[0, 0])

print("\nİlk token embedding shape:")
print(output[0, 0].shape)