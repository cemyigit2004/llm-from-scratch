import json
import re
from pathlib import Path


INPUT_FILE = Path("data/raw/data.jsonl")
OUTPUT_FILE = Path("data/cleaned/data.jsonl")


def clean_text(text: str) -> str:
    # Remove null characters
    text = text.replace("\x00", "")

    # Windows newline -> standard newline
    text = text.replace("\r\n", "\n")

    # Collapse multiple spaces into one
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ newlines into two
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    seen = set()

    total = 0
    kept = 0
    duplicates = 0

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as fin, OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as fout:

        for line in fin:

            total += 1

            obj = json.loads(line)

            text = clean_text(obj["text"])

            # Too short / empty document
            if len(text) < 50:
                continue

            # Exact duplicate
            if text in seen:
                duplicates += 1
                continue

            seen.add(text)

            fout.write(
                json.dumps(
                    {"text": text},
                    ensure_ascii=False
                )
                + "\n"
            )

            kept += 1

    print("Data preparation complete.")
    print("Total documents:", total)
    print("Kept documents:", kept)
    print("Duplicate:", duplicates)
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()