#!/usr/bin/env python3

import json
from tests.common import gpt2_bytes_to_unicode
from tests.tokenizer import Tokenizer

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

# Create the vocabulary mapping
vocab = {
    gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
    for gpt2_vocab_index, gpt2_vocab_item in gpt2_vocab.items()
}

# Create the merges
merges = [
    (
        bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
        bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
    )
    for merge_token_1, merge_token_2 in gpt2_bpe_merges
]

print(f"Created vocabulary with {len(vocab)} tokens")
print(f"Created {len(merges)} merges")

# Create the tokenizer
tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=None)

print(f"\nTokenizer merges length: {len(tokenizer.merges)}")
print(f"First 5 merges: {tokenizer.merges[:5]}")

# Check if the specific merge exists in the tokenizer
xf0_x9f_merge = (b'\xf0', b'\x9f')
print(f"\nMerge {xf0_x9f_merge} in tokenizer: {xf0_x9f_merge in tokenizer.merges}")

if xf0_x9f_merge in tokenizer.merges:
    idx = tokenizer.merges.index(xf0_x9f_merge)
    print(f"  Found at index: {idx}")

# Test encoding
test_string = "🙃"
print(f"\nTesting encoding of: {test_string}")

try:
    result = tokenizer.encode(test_string)
    print(f"Encoding result: {result}")
    
    # Decode to verify
    decoded = tokenizer.decode(result)
    print(f"Decoded result: {repr(decoded)}")
    print(f"Round-trip successful: {decoded == test_string}")
    
except Exception as e:
    print(f"Error during encoding: {e}")
    import traceback
    traceback.print_exc()

# Test with a simple ASCII string
test_string2 = "he"
print(f"\nTesting encoding of: {test_string2}")

try:
    result2 = tokenizer.encode(test_string2)
    print(f"Encoding result: {result2}")
    
    # Decode to verify
    decoded2 = tokenizer.decode(result2)
    print(f"Decoded result: {repr(decoded2)}")
    print(f"Round-trip successful: {decoded2 == test_string2}")
    
except Exception as e:
    print(f"Error during encoding: {e}")
    import traceback
    traceback.print_exc()
