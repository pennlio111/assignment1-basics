#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'tests'))

from tests.tokenizer import Tokenizer
from tests.common import gpt2_bytes_to_unicode
import json

def get_tokenizer_from_vocab_merges_path(
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    
    vocab = {}
    for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items():
        byte_sequence = bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        vocab[gpt2_vocab_index] = byte_sequence

    merges = []
    for merge_token_1, merge_token_2 in gpt2_bpe_merges:
        byte_merge_1 = bytes([gpt2_byte_decoder[token] for token in merge_token_1])
        byte_merge_2 = bytes([gpt2_byte_decoder[token] for token in merge_token_2])
        merges.append((byte_merge_1, byte_merge_2))
    
    return Tokenizer(vocab, merges, special_tokens)

# Load tokenizer
tokenizer = get_tokenizer_from_vocab_merges_path(
    vocab_path="tests/fixtures/gpt2_vocab.json",
    merges_path="tests/fixtures/gpt2_merges.txt",
)

test_string = "🙃"
print(f"Testing string: {test_string}")

# Current behavior (character-level pre-tokenization)
from tests.constants import PAT
import regex as re
pre_tokens = re.findall(PAT, test_string)
print(f"\nCurrent pre-tokenization (PAT):")
print(f"  Pre-tokens: {pre_tokens}")
print(f"  Pre-tokens as bytes: {[pre_token.encode('utf-8') for pre_token in pre_tokens]}")

# Simulate byte-level pre-tokenization (what tiktoken does)
byte_tokens = [bytes([b]) for b in test_string.encode('utf-8')]
print(f"\nByte-level pre-tokenization (tiktoken style):")
print(f"  Byte tokens: {byte_tokens}")

# Apply BPE merges to byte tokens
print(f"\nApplying BPE merges to byte tokens:")
merged_tokens = tokenizer._merge_tokens(byte_tokens)
print(f"  After merging: {merged_tokens}")

# Convert to token IDs
print(f"\nConverting to token IDs:")
token_ids = []
for token in merged_tokens:
    if token in tokenizer.reverse_vocab:
        token_id = tokenizer.reverse_vocab[token]
        token_ids.append(token_id)
        print(f"  {token} -> token {token_id}")
    else:
        print(f"  {token} -> NOT in vocabulary")

print(f"\nFinal result: {token_ids}")

# Compare with tiktoken
print(f"\nTiktoken comparison:")
try:
    import tiktoken
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    reference_ids = reference_tokenizer.encode(test_string)
    print(f"  Tiktoken result: {reference_ids}")
    print(f"  Match: {token_ids == reference_ids}")
except Exception as e:
    print(f"  Error with tiktoken: {e}")
