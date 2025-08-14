#!/usr/bin/env python3

from tests.common import gpt2_bytes_to_unicode

# Get the byte mapping
byte_to_unicode = gpt2_bytes_to_unicode()
unicode_to_byte = {v: k for k, v in byte_to_unicode.items()}

print("Byte mapping analysis:")
print(f"Total mappings: {len(byte_to_unicode)}")

# Check the emoji bytes
emoji_bytes = "🙃".encode('utf-8')
print(f"\nEmoji bytes: {emoji_bytes}")

for i, byte in enumerate(emoji_bytes):
    if byte in byte_to_unicode:
        unicode_char = byte_to_unicode[byte]
        print(f"  Byte {i}: {byte} (0x{byte:02x}) -> unicode: '{unicode_char}' (ord: {ord(unicode_char)})")
    else:
        print(f"  Byte {i}: {byte} (0x{byte:02x}) -> NOT in mapping")

# Check if the unicode characters are in the reverse mapping
print(f"\nReverse mapping check:")
for i, byte in enumerate(emoji_bytes):
    if byte in byte_to_unicode:
        unicode_char = byte_to_unicode[byte]
        if unicode_char in unicode_to_byte:
            mapped_byte = unicode_to_byte[unicode_char]
            print(f"  Byte {i}: {byte} -> unicode '{unicode_char}' -> mapped back to {mapped_byte}")
            print(f"    Match: {byte == mapped_byte}")
        else:
            print(f"  Byte {i}: {byte} -> unicode '{unicode_char}' -> NOT in reverse mapping")
    else:
        print(f"  Byte {i}: {byte} -> NOT in forward mapping")

# Let's check what the actual vocabulary contains for these bytes
print(f"\nChecking vocabulary for emoji bytes:")
import json
from tests.common import gpt2_bytes_to_unicode

gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

with open("tests/fixtures/gpt2_vocab.json") as vocab_f:
    gpt2_vocab = json.load(vocab_f)

# Create the vocab
vocab = {}
for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items():
    byte_sequence = bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
    vocab[gpt2_vocab_index] = byte_sequence

# Check if individual bytes exist
for i, byte in enumerate(emoji_bytes):
    byte_token = bytes([byte])
    if byte_token in vocab.values():
        # Find the token ID
        token_id = None
        for tid, bytes_val in vocab.items():
            if bytes_val == byte_token:
                token_id = tid
                break
        print(f"  Byte {i}: {byte} (0x{byte:02x}) -> found as token {token_id}")
    else:
        print(f"  Byte {i}: {byte} (0x{byte:02x}) -> NOT found in vocab")

# Check if the 3-byte sequence exists
first_three = emoji_bytes[:3]
if first_three in vocab.values():
    token_id = None
    for tid, bytes_val in vocab.items():
        if bytes_val == first_three:
            token_id = tid
            break
    print(f"  First 3 bytes {first_three} -> found as token {token_id}")
else:
    print(f"  First 3 bytes {first_three} -> NOT found in vocab")

# Let's check what's in the vocabulary around the expected token IDs
print(f"\nChecking vocabulary around expected token IDs:")
expected_tokens = [8582, 247, 225]
for token_id in expected_tokens:
    if token_id in vocab:
        print(f"  Token {token_id}: {vocab[token_id]} (length: {len(vocab[token_id])})")
    else:
        print(f"  Token {token_id}: NOT found")

# Let's search for similar byte sequences
print(f"\nSearching for byte sequences similar to emoji bytes:")
for token_id, bytes_val in vocab.items():
    if len(bytes_val) >= 3 and bytes_val.startswith(emoji_bytes[:2]):
        print(f"  Token {token_id}: {bytes_val} (starts with first 2 bytes)")
        break
