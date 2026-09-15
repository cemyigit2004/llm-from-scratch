import json

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class LMDataset(Dataset):

    def __init__(
        self,
        file_path,
        context_length=128
    ):
        self.context_length = context_length

        # Hazır GPT-2 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "openai-community/gpt2"
        )

        all_tokens = []

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                document = json.loads(line)

                text = document["text"]

                # Text -> token IDs
                token_ids = self.tokenizer.encode(
                    text,
                    add_special_tokens=False
                )

                all_tokens.extend(token_ids)

                # Document sonu
                all_tokens.append(
                    self.tokenizer.eos_token_id
                )

        self.tokens = torch.tensor(
            all_tokens,
            dtype=torch.long
        )

        print(
            f"Toplam token: {len(self.tokens):,}"
        )

    def __len__(self):

        return (
            len(self.tokens) - 1
        ) // self.context_length

    def __getitem__(self, index):

        start = index * self.context_length

        end = (
            start
            + self.context_length
            + 1
        )

        chunk = self.tokens[
            start:end
        ]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y