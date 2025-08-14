#!/usr/bin/env python3

import json
from tests.common import gpt2_bytes_to_unicode

# Load the vocabulary and merges
with open("tests/fixtures/gpt2_vocab.json", "r") as f:
    gpt2_vocab = json.load(f)

# Create the byte decoder
gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

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

# Check for duplicate byte sequences
byte_to_token_ids = {}
for token_id, byte_seq in vocab.items():
    if byte_seq in byte_to_token_ids:
        byte_to_token_ids[byte_seq].append(token_id)
    else:
        byte_to_token_ids[byte_seq] = [token_id]

# Find duplicates
duplicates = {byte_seq: token_ids for byte_seq, token_ids in byte_to_token_ids.items() if len(token_ids) > 1}

print(f"\nFound {len(duplicates)} duplicate byte sequences")

if duplicates:
    print("\nFirst 10 duplicates:")
    for i, (byte_seq, token_ids) in enumerate(list(duplicates.items())[:10]):
        print(f"  {byte_seq} (hex: {byte_seq.hex()}) -> Token IDs: {token_ids}")

# Check the specific bytes we need
test_bytes = [b'\xf0', b'\x9f', b'\x99', b'\x83']
print(f"\nChecking our test bytes:")
for test_byte in test_bytes:
    if test_byte in byte_to_token_ids:
        token_ids = byte_to_token_ids[test_byte]
        print(f"  {test_byte} -> Token IDs: {token_ids}")
        if len(token_ids) > 1:
            print(f"    WARNING: Multiple token IDs for this byte!")
    else:
        print(f"  {test_byte} -> NOT FOUND")

# Now let's simulate what happens when creating reverse_vocab
print(f"\nSimulating reverse_vocab creation:")
reverse_vocab = {}
for token_id, byte_seq in vocab.items():
    reverse_vocab[byte_seq] = token_id

print(f"Reverse vocab size: {len(reverse_vocab)}")
print(f"Original vocab size: {len(vocab)}")

# Check if our test bytes are in the reverse vocab
for test_byte in test_bytes:
    if test_byte in reverse_vocab:
        token_id = reverse_vocab[test_byte]
        print(f"  {test_byte} -> Token ID {token_id}")
    else:
        print(f"  {test_byte} -> NOT FOUND in reverse vocab")

# The issue might be that the last token ID overwrites earlier ones
# Let's check what the last token ID is for each test byte
for test_byte in test_bytes:
    if test_byte in byte_to_token_ids:
        token_ids = byte_to_token_ids[test_byte]
        last_token_id = max(token_ids)
        print(f"  {test_byte} -> Last token ID: {last_token_id}")
        
        # Check if this matches what's in reverse_vocab
        if test_byte in reverse_vocab:
            reverse_token_id = reverse_vocab[test_byte]
            print(f"    Reverse vocab has: {reverse_token_id}")
            print(f"    Match: {last_token_id == reverse_token_id}")
