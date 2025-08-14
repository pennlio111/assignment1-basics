#!/usr/bin/env python3

import json
from tests.common import gpt2_bytes_to_unicode

# Load the vocabulary and merges
with open("tests/fixtures/gpt2_vocab.json", "r") as f:
    gpt2_vocab = json.load(f)

with open("tests/fixtures/gpt2_merges.txt", "r") as f:
    gpt2_bpe_merges = []
    for line in f:
        cleaned_line = line.rstrip()
        if cleaned_line and len(cleaned_line.split(" ")) == 2:
            gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))

# Create the byte decoder
gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

print(f"Loaded {len(gpt2_vocab)} vocabulary entries")
print(f"Loaded {len(gpt2_bpe_merges)} merges")

# Create the vocabulary mapping
vocab = {}
for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items():
    try:
        byte_sequence = bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        vocab[gpt2_vocab_index] = byte_sequence
    except KeyError as e:
        print(f"Error processing '{gpt2_vocab_item}': character {e} not found in byte decoder")
        break

print(f"Created vocabulary with {len(vocab)} entries")

# Check if specific bytes exist in the vocabulary
test_bytes = [b'\xf0', b'\x9f', b'\x99', b'\x83']
print(f"\nChecking if test bytes exist in vocabulary:")
for test_byte in test_bytes:
    found = False
    for token_id, byte_seq in vocab.items():
        if byte_seq == test_byte:
            print(f"  {test_byte} found with token ID {token_id}")
            found = True
            break
    if not found:
        print(f"  {test_byte} NOT found in vocabulary")

# Check what's in the vocabulary around the expected range
print(f"\nChecking vocabulary around expected token IDs:")
for token_id in range(170, 175):  # Around 172
    if token_id in vocab:
        byte_seq = vocab[token_id]
        print(f"  Token {token_id}: {byte_seq} (hex: {byte_seq.hex()})")

for token_id in range(250, 255):  # Around 253
    if token_id in vocab:
        byte_seq = vocab[token_id]
        print(f"  Token {token_id}: {byte_seq} (hex: {byte_seq.hex()})")

for token_id in range(245, 250):  # Around 247
    if token_id in vocab:
        byte_seq = vocab[token_id]
        print(f"  Token {token_id}: {byte_seq} (hex: {byte_seq.hex()})")

for token_id in range(223, 228):  # Around 225
    if token_id in vocab:
        byte_seq = vocab[token_id]
        print(f"  Token {token_id}: {byte_seq} (hex: {byte_seq.hex()})")

# Let's also check what the unicode character "🙃" encodes to
test_string = "🙃"
test_bytes_actual = test_string.encode('utf-8')
print(f"\nUnicode character '{test_string}' encodes to: {test_bytes_actual}")
print(f"Individual bytes: {[b for b in test_bytes_actual]}")

# Check if these individual bytes exist in the vocabulary
for i, byte_val in enumerate(test_bytes_actual):
    byte_token = bytes([byte_val])
    found = False
    for token_id, byte_seq in vocab.items():
        if byte_seq == byte_token:
            print(f"  Byte {i} ({byte_token}) found with token ID {token_id}")
            found = True
            break
    if not found:
        print(f"  Byte {i} ({byte_token}) NOT found in vocabulary")
