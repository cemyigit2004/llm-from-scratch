import torch

from model import GPT, GPTConfig, CausalSelfAttention


config = GPTConfig(
    vocab_size=50257,
    context_length=128,
    d_model=384,
    n_heads=6
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



#q, k, v = attention(output)

"""print("\nQ shape:")
print(q.shape)

print("\nK shape:")
print(k.shape)

print("\nV shape:")
print(v.shape)"""


attention = CausalSelfAttention(config)

attention_output = attention(output)

print("\nAttention input:")
print(output.shape)

print("\nAttention output:")
print(attention_output.shape)

print("\nHead sayısı:")
print(attention.n_heads)

print("\nHead dimension:")
print(attention.head_dim)