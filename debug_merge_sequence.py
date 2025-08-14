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
merges = []
for merge_token_1, merge_token_2 in gpt2_bpe_merges:
    try:
        byte_merge_1 = bytes([gpt2_byte_decoder[token] for token in merge_token_1])
        byte_merge_2 = bytes([gpt2_byte_decoder[token] for token in merge_token_2])
        merges.append((byte_merge_1, byte_merge_2))
    except KeyError as e:
        print(f"Error processing merge '{merge_token_1}' + '{merge_token_2}': character {e} not found in byte decoder")
        break

print(f"Created {len(merges)} merges")

# The bytes we need to merge
test_bytes = [b'\xf0', b'\x9f', b'\x99', b'\x83']
print(f"\nTest bytes: {test_bytes}")

# Check if there's a merge that creates \xf0\x9f\x99
target_sequence = b'\xf0\x9f\x99'
print(f"\nLooking for merge that creates: {target_sequence}")

found = False
for i, merge in enumerate(merges):
    if merge[0] + merge[1] == target_sequence:
        print(f"Found merge at index {i}: {merge} -> {merge[0] + merge[1]}")
        found = True
        break

if not found:
    print("No merge creates this sequence directly")
    
    # Check if we can build it up step by step
    print("\nChecking step-by-step merges:")
    
    # Step 1: Merge \xf0 and \x9f
    step1 = b'\xf0' + b'\x9f'
    print(f"Step 1: {b'\\xf0'} + {b'\\x9f'} = {step1}")
    
    # Check if this merge exists
    step1_merge = None
    for i, merge in enumerate(merges):
        if merge == (b'\xf0', b'\x9f'):
            step1_merge = merge
            print(f"  Found merge at index {i}: {merge}")
            break
    
    if step1_merge:
        # Step 2: Check if we can merge step1 with \x99
        step2 = step1 + b'\x99'
        print(f"Step 2: {step1} + {b'\\x99'} = {step2}")
        
        # Check if this merge exists
        for i, merge in enumerate(merges):
            if merge == (step1, b'\x99'):
                print(f"  Found merge at index {i}: {merge} -> {merge[0] + merge[1]}")
                found = True
                break
        else:
            print("  No merge creates step2")
    else:
        print("  No merge creates step1")

# Let's also check what tiktoken actually produces
import tiktoken
reference_tokenizer = tiktoken.get_encoding("gpt2")
reference_ids = reference_tokenizer.encode("🙃")
print(f"\nTiktoken produces: {reference_ids}")

# Decode each token individually to see what they represent
for token_id in reference_ids:
    decoded = reference_tokenizer.decode([token_id])
    print(f"Token {token_id}: {repr(decoded)} (bytes: {decoded.encode('utf-8')})")

# Now let's check if our vocabulary has token 8582
print(f"\nChecking if our vocabulary has token 8582...")
with open("tests/fixtures/gpt2_vocab.json", "r") as f:
    gpt2_vocab = json.load(f)

# Create the vocabulary mapping
vocab = {}
for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items():
    try:
        byte_sequence = bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        vocab[gpt2_vocab_index] = byte_sequence
    except KeyError as e:
        print(f"Error processing '{gpt2_vocab_item}': character {e} not found in byte decoder")
        break

if 8582 in vocab:
    print(f"Token 8582 exists in our vocab: {vocab[8582]}")
    print(f"  Decoded: {vocab[8582].decode('utf-8', errors='replace')}")
else:
    print("Token 8582 NOT found in our vocabulary!")
    
    # Check what tokens we do have around that range
    nearby_tokens = [(k, v) for k, v in vocab.items() if 8580 <= k <= 8585]
    print(f"Nearby tokens: {nearby_tokens}")
