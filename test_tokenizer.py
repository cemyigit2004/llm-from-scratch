from transformers import AutoTokenizer


TOKENIZER_NAME = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME
)

text = "Transformer modelleri nasıl çalışır?"

encoded = tokenizer(
    text,
    add_special_tokens=False
)

print("Orijinal text:")
print(text)

print("\nToken ID'leri:")
print(encoded["input_ids"])

print("\nTokenlar:")
print(
    tokenizer.convert_ids_to_tokens(
        encoded["input_ids"]
    )
)

print("\nDecode edilmiş hali:")
print(
    tokenizer.decode(
        encoded["input_ids"]
    )
)

print("\nVocabulary size:")
print(len(tokenizer))

print("\nEOS token:")
print(tokenizer.eos_token)

print("\nEOS token ID:")
print(tokenizer.eos_token_id)