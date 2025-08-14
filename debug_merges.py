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

# The bytes we need to merge
byte1 = b'\xf0'
byte2 = b'\x9f'
target_pair = (byte1, byte2)

print(f"Looking for merge: {target_pair}")
print(f"Byte1: {byte1} (int: {byte1[0]})")
print(f"Byte2: {byte2} (int: {byte2[0]})")

# Check if this merge exists
found = False
for i, merge in enumerate(merges):
    if merge == target_pair:
        print(f"Found merge at index {i}: {merge}")
        found = True
        break

if not found:
    print("Merge not found!")
    
    # Let's check what merges we do have that start with \xf0
    xf0_merges = [merge for merge in merges if merge[0] == byte1]
    print(f"\nMerges starting with \\xf0: {len(xf0_merges)}")
    for merge in xf0_merges[:5]:  # Show first 5
        print(f"  {merge}")
    
    # Let's also check what merges we have that start with \x9f
    x9f_merges = [merge for merge in merges if merge[0] == byte2]
    print(f"\nMerges starting with \\x9f: {len(x9f_merges)}")
    for merge in x9f_merges[:5]:  # Show first 5
        print(f"  {merge}")

# Let's also check what the first few merges look like
print(f"\nFirst 10 merges:")
for i, merge in enumerate(merges[:10]):
    print(f"  {i}: {merge} -> {merge[0] + merge[1]}")

# Check if there's a merge that would create \xf0\x9f
target_merged = byte1 + byte2
print(f"\nLooking for any merge that creates: {target_merged}")
for i, merge in enumerate(merges):
    if merge[0] + merge[1] == target_merged:
        print(f"Found merge at index {i}: {merge} -> {merge[0] + merge[1]}")
        break
else:
    print("No merge creates this byte sequence")
