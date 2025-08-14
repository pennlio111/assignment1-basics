#!/usr/bin/env python3

import json
from tests.common import gpt2_bytes_to_unicode
from tests.tokenizer import Tokenizer
from tests.adapters import get_tokenizer

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

print("=== Direct approach ===")
# Direct approach
vocab_direct = {}
for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items():
    try:
        byte_sequence = bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        vocab_direct[gpt2_vocab_index] = byte_sequence
    except KeyError as e:
        print(f"Error processing '{gpt2_vocab_item}': character {e} not found in byte decoder")
        break

merges_direct = []
for merge_token_1, merge_token_2 in gpt2_bpe_merges:
    try:
        byte_merge_1 = bytes([gpt2_byte_decoder[token] for token in merge_token_1])
        byte_merge_2 = bytes([gpt2_byte_decoder[token] for token in merge_token_2])
        merges_direct.append((byte_merge_1, byte_merge_2))
    except KeyError as e:
        print(f"Error processing merge '{merge_token_1}' + '{merge_token_2}': character {e} not found in byte decoder")
        break

print(f"Direct vocab size: {len(vocab_direct)}")
print(f"Direct merges size: {len(merges_direct)}")

# Create tokenizer directly
tokenizer_direct = Tokenizer(vocab=vocab_direct, merges=merges_direct, special_tokens=None)

print(f"\n=== Test helper approach ===")
# Test helper approach
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
    
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_index, gpt2_vocab_item in gpt2_vocab.items()
    }
    
    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    
    return get_tokenizer(vocab, merges, special_tokens)

# Create tokenizer via test helper
tokenizer_helper = get_tokenizer_from_vocab_merges_path(
    "tests/fixtures/gpt2_vocab.json",
    "tests/fixtures/gpt2_merges.txt"
)

print(f"Helper vocab size: {len(tokenizer_helper.vocab)}")
print(f"Helper merges size: {len(tokenizer_helper.merges)}")

# Compare the vocabularies
print(f"\n=== Comparison ===")
print(f"Vocab sizes match: {len(tokenizer_direct.vocab) == len(tokenizer_helper.vocab)}")
print(f"Merges sizes match: {len(tokenizer_direct.merges) == len(tokenizer_helper.merges)}")

# Check if the vocabularies are identical
vocab_match = True
for token_id in tokenizer_direct.vocab:
    if token_id not in tokenizer_helper.vocab:
        vocab_match = False
        print(f"Token ID {token_id} missing in helper vocab")
        break
    if tokenizer_direct.vocab[token_id] != tokenizer_helper.vocab[token_id]:
        vocab_match = False
        print(f"Token ID {token_id} mismatch: {tokenizer_direct.vocab[token_id]} vs {tokenizer_helper.vocab[token_id]}")
        break

print(f"Vocabularies identical: {vocab_match}")

# Check if the merges are identical
merges_match = True
for i, merge in enumerate(tokenizer_direct.merges):
    if i >= len(tokenizer_helper.merges):
        merges_match = False
        print(f"Merge {i} missing in helper merges")
        break
    if merge != tokenizer_helper.merges[i]:
        merges_match = False
        print(f"Merge {i} mismatch: {merge} vs {tokenizer_helper.merges[i]}")
        break

print(f"Merges identical: {merges_match}")

# Test encoding with both tokenizers
test_string = "🙃"
print(f"\n=== Testing encoding ===")
print(f"Test string: {test_string}")

try:
    result_direct = tokenizer_direct.encode(test_string)
    print(f"Direct result: {result_direct}")
except Exception as e:
    print(f"Direct error: {e}")

try:
    result_helper = tokenizer_helper.encode(test_string)
    print(f"Helper result: {result_helper}")
except Exception as e:
    print(f"Helper error: {e}")

# Check if results match
if 'result_direct' in locals() and 'result_helper' in locals():
    print(f"Results match: {result_direct == result_helper}")
