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
        # Convert the GPT-2 token string to bytes using the byte decoder
        byte_sequence = bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        vocab[gpt2_vocab_index] = byte_sequence

    merges = []
    for merge_token_1, merge_token_2 in gpt2_bpe_merges:
        # Convert the GPT-2 merge tokens to bytes using the byte decoder
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
print(f"UTF-8 bytes: {test_string.encode('utf-8')}")

# Step 1: Check pre-tokenization
from tests.constants import PAT
import regex as re
pre_tokens = re.findall(PAT, test_string)
print(f"\nStep 1 - Pre-tokenization with PAT:")
print(f"  PAT: {PAT}")
print(f"  Pre-tokens: {pre_tokens}")

# Step 2: Convert pre-tokens to bytes
pre_tokens_bytes = [pre_token.encode('utf-8') for pre_token in pre_tokens]
print(f"\nStep 2 - Pre-tokens as bytes:")
for i, token in enumerate(pre_tokens_bytes):
    print(f"  {i}: {token} (length: {len(token)})")

# Step 3: Apply BPE merges
print(f"\nStep 3 - Applying BPE merges:")
merged_tokens = tokenizer._merge_tokens(pre_tokens_bytes)
print(f"  After merging: {merged_tokens}")
for i, token in enumerate(merged_tokens):
    print(f"  {i}: {token} (length: {len(token)})")

# Step 4: Join and decode
joined_bytes = b"".join(merged_tokens)
print(f"\nStep 4 - Joined bytes:")
print(f"  Joined: {joined_bytes}")
print(f"  Decoded: {joined_bytes.decode('utf-8', errors='replace')}")

# Step 5: Basic encode
print(f"\nStep 5 - Basic encoding:")
try:
    basic_ids = tokenizer._basic_encode(joined_bytes.decode('utf-8', errors='replace'))
    print(f"  Basic encode result: {basic_ids}")
except Exception as e:
    print(f"  Error in basic encode: {e}")

# Step 6: Full encode
print(f"\nStep 6 - Full encoding:")
try:
    full_ids = tokenizer.encode(test_string)
    print(f"  Full encode result: {full_ids}")
except Exception as e:
    print(f"  Error in full encode: {e}")
    import traceback
    traceback.print_exc()

# Step 7: Check what tiktoken produces
print(f"\nStep 7 - Tiktoken comparison:")
try:
    import tiktoken
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    reference_ids = reference_tokenizer.encode(test_string)
    print(f"  Tiktoken result: {reference_ids}")
    print(f"  Tiktoken decoded: {reference_tokenizer.decode(reference_ids)}")
except Exception as e:
    print(f"  Error with tiktoken: {e}")
