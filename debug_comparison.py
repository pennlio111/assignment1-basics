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
    special_tokens=["<|endoftext|>"]
)

test_string = "Hello, how are you?"
print(f"Testing string: '{test_string}'")

# Test our tokenizer
ids = tokenizer.encode(test_string)
print(f"\nOur tokenizer result: {ids}")

# Decode each token individually
tokenized_string = [tokenizer.decode([x]) for x in ids]
print(f"Individual tokens: {tokenized_string}")

# Test tiktoken
try:
    import tiktoken
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    reference_ids = reference_tokenizer.encode(test_string)
    print(f"\nTiktoken result: {reference_ids}")
    
    # Decode each token individually
    reference_tokenized_string = [reference_tokenizer.decode([x]) for x in reference_ids]
    print(f"Tiktoken individual tokens: {reference_tokenized_string}")
    
    print(f"\nMatch: {ids == reference_ids}")
    
except Exception as e:
    print(f"Error with tiktoken: {e}")

# Let's also check what the regex pattern produces
from tests.constants import PAT
import regex as re
pre_tokens = re.findall(PAT, test_string)
print(f"\nRegex pre-tokens: {pre_tokens}")

# Check if these tokens exist in vocabulary
print(f"\nChecking if pre-tokens exist in vocabulary:")
for pre_token in pre_tokens:
    encoded = pre_token.encode('utf-8')
    if encoded in tokenizer.reverse_vocab:
        token_id = tokenizer.reverse_vocab[encoded]
        print(f"  '{pre_token}' -> {encoded} -> token {token_id}")
    else:
        print(f"  '{pre_token}' -> {encoded} -> NOT in vocabulary")
