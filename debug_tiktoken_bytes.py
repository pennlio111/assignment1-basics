#!/usr/bin/env python3

import tiktoken

# Get tiktoken's encoding
enc = tiktoken.get_encoding("gpt2")

test_string = "🙃"
print(f"Testing string: {test_string}")
print(f"UTF-8 bytes: {test_string.encode('utf-8')}")

# Check what tiktoken produces
reference_ids = enc.encode(test_string)
print(f"\nTiktoken result: {reference_ids}")

# Let's try to understand how tiktoken breaks this down
print(f"\nAnalyzing tiktoken's behavior:")

# Try encoding individual bytes
print(f"Encoding individual bytes:")
for i, byte in enumerate(test_string.encode('utf-8')):
    byte_char = chr(byte)
    try:
        ids = enc.encode(byte_char)
        print(f"  Byte {i}: {byte} (0x{byte:02x}) -> {ids}")
    except Exception as e:
        print(f"  Byte {i}: {byte} (0x{byte:02x}) -> Error: {e}")

# Try encoding 2-byte sequences
print(f"\nEncoding 2-byte sequences:")
emoji_bytes = test_string.encode('utf-8')
for i in range(len(emoji_bytes) - 1):
    two_bytes = emoji_bytes[i:i+2]
    try:
        ids = enc.encode(two_bytes.decode('utf-8', errors='replace'))
        print(f"  Bytes {i}-{i+1}: {two_bytes} -> {ids}")
    except Exception as e:
        print(f"  Bytes {i}-{i+1}: {two_bytes} -> Error: {e}")

# Try encoding 3-byte sequences
print(f"\nEncoding 3-byte sequences:")
for i in range(len(emoji_bytes) - 2):
    three_bytes = emoji_bytes[i:i+3]
    try:
        ids = enc.encode(three_bytes.decode('utf-8', errors='replace'))
        print(f"  Bytes {i}-{i+2}: {three_bytes} -> {ids}")
    except Exception as e:
        print(f"  Bytes {i}-{i+2}: {three_bytes} -> Error: {e}")

# Let's also check what happens if we try to encode the emoji character by character
print(f"\nEncoding emoji character by character:")
for char in test_string:
    try:
        ids = enc.encode(char)
        print(f"  Character '{char}' -> {ids}")
    except Exception as e:
        print(f"  Character '{char}' -> Error: {e}")

# The key insight: tiktoken must be doing byte-level pre-tokenization
# Let's check if we can simulate this
print(f"\nSimulating byte-level pre-tokenization:")
byte_tokens = [bytes([b]) for b in test_string.encode('utf-8')]
print(f"  Byte tokens: {byte_tokens}")

# Now let's see what merges would apply
print(f"\nChecking what merges would apply to byte tokens:")
from tests.common import gpt2_bytes_to_unicode
import json

gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

with open("tests/fixtures/gpt2_vocab.json") as vocab_f:
    gpt2_vocab = json.load(vocab_f)

vocab = {}
for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items():
    byte_sequence = bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
    vocab[gpt2_vocab_index] = byte_sequence

# Load merges
with open("tests/fixtures/gpt2_merges.txt") as f:
    gpt2_bpe_merges = []
    for line in f:
        cleaned_line = line.rstrip()
        if cleaned_line and len(cleaned_line.split(" ")) == 2:
            gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))

merges = []
for merge_token_1, merge_token_2 in gpt2_bpe_merges:
    byte_merge_1 = bytes([gpt2_byte_decoder[token] for token in merge_token_1])
    byte_merge_2 = bytes([gpt2_byte_decoder[token] for token in merge_token_2])
    merges.append((byte_merge_1, byte_merge_2))

print(f"  Loaded {len(merges)} merges")

# Check if the first merge applies
first_pair = (byte_tokens[0], byte_tokens[1])
print(f"  First pair: {first_pair}")
if first_pair in merges:
    idx = merges.index(first_pair)
    print(f"    Found merge at index {idx}")
    merged = first_pair[0] + first_pair[1]
    print(f"    Would create: {merged}")
else:
    print(f"    No merge found")

# Check if the second merge applies
second_pair = (byte_tokens[1], byte_tokens[2])
print(f"  Second pair: {second_pair}")
if second_pair in merges:
    idx = merges.index(second_pair)
    print(f"    Found merge at index {idx}")
    merged = second_pair[0] + second_pair[1]
    print(f"    Would create: {merged}")
else:
    print(f"    No merge found")
