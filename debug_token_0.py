#!/usr/bin/env python3

import json
from tests.common import gpt2_bytes_to_unicode

# Load the vocabulary
with open("tests/fixtures/gpt2_vocab.json") as vocab_f:
    gpt2_vocab = json.load(vocab_f)

print("Checking token 0:")
print(f"  In original vocab: '{gpt2_vocab['!']}'")

# Check what the first few tokens should be
print("\nFirst 10 tokens in original vocab:")
for i, (key, value) in enumerate(list(gpt2_vocab.items())[:10]):
    print(f"  {i}: '{key}' -> {value}")

# Check if there are any special tokens
print("\nLooking for special tokens:")
for key, value in gpt2_vocab.items():
    if key.startswith('<|') and key.endswith('|>'):
        print(f"  '{key}' -> {value}")

# Check if there's a token with value 0
print("\nLooking for token with value 0:")
for key, value in gpt2_vocab.items():
    if value == 0:
        print(f"  Token with value 0: '{key}'")

# Check if there's a token with value 1
print("\nLooking for token with value 1:")
for key, value in gpt2_vocab.items():
    if value == 1:
        print(f"  Token with value 1: '{key}'")
