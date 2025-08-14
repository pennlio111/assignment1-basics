#!/usr/bin/env python3

import json
from tests.common import gpt2_bytes_to_unicode

# Load the merges
with open("tests/fixtures/gpt2_merges.txt", "r") as f:
    gpt2_bpe_merges = []
    for line in f:
        cleaned_line = line.rstrip()
        if cleaned_line and len(cleaned_line.split(" ")) == 2:
            gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))

# Create the byte decoder
gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

# Create the merges
merges = [
    (
        bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
        bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
    )
    for merge_token_1, merge_token_2 in gpt2_bpe_merges
]

# Test the unicode character bytes
test_bytes = [b'\xf0', b'\x9f', b'\x99', b'\x83']
print(f"Initial bytes: {test_bytes}")

# Check for merges that would create the full sequence
target_sequence = b'\xf0\x9f\x99\x83'
print(f"\nLooking for merges that create: {target_sequence}")

# Check if there's a merge that creates \xf0\x9f\x99\x83
for i, merge in enumerate(merges):
    if merge[0] + merge[1] == target_sequence:
        print(f"Found merge at index {i}: {merge} -> {merge[0] + merge[1]}")
        break
else:
    print("No merge creates the full sequence directly")
    
    # Check if we can build it up step by step
    print("\nChecking step-by-step merges:")
    
    # Step 1: Merge \xf0 and \x9f
    step1 = b'\xf0' + b'\x9f'
    print(f"Step 1: {b'\\xf0'} + {b'\\x9f'} = {step1}")
    
    # Step 2: Check if we can merge step1 with \x99
    step2 = step1 + b'\x99'
    print(f"Step 2: {step1} + {b'\\x99'} = {step2}")
    
    # Check if this merge exists
    for i, merge in enumerate(merges):
        if merge[0] + merge[1] == step2:
            print(f"  Found merge at index {i}: {merge} -> {merge[0] + merge[1]}")
            break
    else:
        print("  No merge creates step2")
    
    # Step 3: Check if we can merge step2 with \x83
    step3 = step2 + b'\x83'
    print(f"Step 3: {step2} + {b'\\x83'} = {step3}")
    
    # Check if this merge exists
    for i, merge in enumerate(merges):
        if merge[0] + merge[1] == step3:
            print(f"  Found merge at index {i}: {merge} -> {merge[0] + merge[1]}")
            break
    else:
        print("  No merge creates step3")

# Let's also check what tiktoken actually produces
import tiktoken
reference_tokenizer = tiktoken.get_encoding("gpt2")
reference_ids = reference_tokenizer.encode("🙃")
print(f"\nTiktoken produces: {reference_ids}")

# Decode each token individually
for token_id in reference_ids:
    decoded = reference_tokenizer.decode([token_id])
    print(f"Token {token_id}: {repr(decoded)} (bytes: {decoded.encode('utf-8')})")

# Now let's check if our vocabulary has the token ID 8582
print(f"\nChecking if our vocabulary has token 8582...")
# We need to load the actual vocabulary from the pickle file
import pickle
with open("tests/fixtures/gpt2_vocab.json", "r") as f:
    gpt2_vocab = json.load(f)

# Create the vocabulary mapping
vocab = {
    gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
    for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
}

if 8582 in vocab:
    print(f"Token 8582 exists in our vocab: {vocab[8582]}")
    print(f"  Decoded: {vocab[8582].decode('utf-8', errors='replace')}")
else:
    print("Token 8582 NOT found in our vocabulary!")
    
    # Check what tokens we do have around that range
    nearby_tokens = [(k, v) for k, v in vocab.items() if 8580 <= k <= 8585]
    print(f"Nearby tokens: {nearby_tokens}")
