#!/usr/bin/env python3

import json

# Load the vocabulary
with open("tests/fixtures/gpt2_vocab.json", "r") as f:
    vocab = json.load(f)

print(f"Vocab type: {type(vocab)}")
print(f"Vocab length: {len(vocab)}")

# Check the first few items
print("\nFirst few items:")
items = list(vocab.items())[:5]
for k, v in items:
    print(f"  {k}: {v} (type: {type(v)})")

# Check if it's a string-to-int mapping or int-to-string
print(f"\nVocab keys type: {type(list(vocab.keys())[0])}")
print(f"Vocab values type: {type(list(vocab.values())[0])}")

# Check if the structure matches what the test helper expects
print(f"\nExpected structure: string -> int")
print(f"Actual structure: {type(list(vocab.keys())[0])} -> {type(list(vocab.values())[0])}")

# Let's also check what the test helper is actually doing
print(f"\nTest helper processing:")
gpt2_vocab = vocab
print(f"  gpt2_vocab type: {type(gpt2_vocab)}")
print(f"  First item: {list(gpt2_vocab.items())[0]}")

# The test helper expects gpt2_vocab_item to be a string (the token)
# and gpt2_vocab_index to be an int (the token ID)
# But it seems like the structure might be reversed
