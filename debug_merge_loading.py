#!/usr/bin/env python3

import json
from tests.common import gpt2_bytes_to_unicode

print("=== Debug script merge loading ===")
# Load the merges (same as debug script)
with open("tests/fixtures/gpt2_merges.txt", "r") as f:
    gpt2_bpe_merges = []
    for line in f:
        cleaned_line = line.rstrip()
        if cleaned_line and len(cleaned_line.split(" ")) == 2:
            gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))

print(f"Loaded {len(gpt2_bpe_merges)} merges")
print(f"First 5 merges: {gpt2_bpe_merges[:5]}")

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

print(f"Converted to {len(merges)} byte merges")
print(f"First 5 byte merges: {merges[:5]}")

print("\n=== Test helper merge loading ===")
# Simulate the test helper function
def get_tokenizer_from_vocab_merges_path(vocab_path, merges_path, special_tokens=None):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    
    print(f"Test helper loaded {len(gpt2_bpe_merges)} merges")
    print(f"First 5 merges: {gpt2_bpe_merges[:5]}")
    
    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    
    print(f"Test helper converted to {len(merges)} byte merges")
    print(f"First 5 byte merges: {merges[:5]}")
    
    return merges

# Test the helper function
test_merges = get_tokenizer_from_vocab_merges_path(
    "tests/fixtures/gpt2_vocab.json",
    "tests/fixtures/gpt2_merges.txt"
)

print(f"\n=== Comparison ===")
print(f"Debug script merges length: {len(merges)}")
print(f"Test helper merges length: {len(test_merges)}")
print(f"Equal: {merges == test_merges}")

# Check if the specific merge we need exists in both
xf0_x9f_merge = (b'\xf0', b'\x9f')
print(f"\nLooking for merge {xf0_x9f_merge}:")
print(f"  In debug script: {xf0_x9f_merge in merges}")
print(f"  In test helper: {xf0_x9f_merge in test_merges}")

if xf0_x9f_merge in merges:
    idx = merges.index(xf0_x9f_merge)
    print(f"  Debug script index: {idx}")
if xf0_x9f_merge in test_merges:
    idx = test_merges.index(xf0_x9f_merge)
    print(f"  Test helper index: {idx}")
