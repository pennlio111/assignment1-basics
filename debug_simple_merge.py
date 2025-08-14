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

print(f"Loaded {len(merges)} merges")

# Test with a simple case: "he"
test_string = "he"
test_bytes = [bytes([b]) for b in test_string.encode('utf-8')]
print(f"\nTest string: '{test_string}'")
print(f"Test bytes: {test_bytes}")

# Check if the merge ('h', 'e') exists
h_e_merge = (b'h', b'e')
print(f"\nLooking for merge: {h_e_merge}")

found = False
for i, merge in enumerate(merges):
    if merge == h_e_merge:
        print(f"Found merge at index {i}: {merge}")
        found = True
        break

if not found:
    print("Merge not found!")
    
    # Check what merges we do have that start with 'h'
    h_merges = [merge for merge in merges if merge[0] == b'h']
    print(f"\nMerges starting with 'h': {len(h_merges)}")
    for merge in h_merges[:5]:  # Show first 5
        print(f"  {merge}")

# Test the merge algorithm
def test_merge_algorithm(byte_tokens, merges):
    token_list = byte_tokens.copy()
    print(f"\nStarting with: {token_list}")
    
    scan_token_list = True
    while scan_token_list:
        scan_token_list = False
        i = 0
        while i < len(token_list) - 1:
            pair = (token_list[i], token_list[i + 1])
            print(f"Checking pair: {pair}")
            
            for merge in merges:
                if pair == merge:
                    new_token = merge[0] + merge[1]
                    token_list[i] = new_token
                    del token_list[i + 1]
                    scan_token_list = True
                    print(f"  Applied merge {merge} -> {new_token}, result: {token_list}")
                    break
            else:
                print(f"  No merge found for {pair}")
                i += 1
    return token_list

result = test_merge_algorithm(test_bytes, merges)
print(f"\nFinal result: {result}")

# Let's also check what the first few merges look like
print(f"\nFirst 10 merges:")
for i, merge in enumerate(merges[:10]):
    print(f"  {i}: {merge} -> {merge[0] + merge[1]}")
