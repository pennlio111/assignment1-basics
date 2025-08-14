#!/usr/bin/env python3

import json
import pickle
from tests.common import gpt2_bytes_to_unicode
from tests.constants import PAT
import regex as re

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
    for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
}

# Create the merges
merges = [
    (
        bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
        bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
    )
    for merge_token_1, merge_token_2 in gpt2_bpe_merges
]

# Test the unicode character
test_string = "🙃"
print(f"Test string: {test_string}")
print(f"UTF-8 bytes: {test_string.encode('utf-8')}")

# Step 1: Pre-tokenization
chunks = [test_string]  # No special tokens
print(f"\nStep 1 - Chunks: {chunks}")

for chunk in chunks:
    if chunk in []:  # No special tokens
        print(f"Chunk '{chunk}' is a special token")
    else:
        # Pre-tokenize the chunk
        pre_tokens = [pre_token.encode('utf-8') for pre_token in re.findall(PAT, chunk)]
        print(f"Pre-tokens: {pre_tokens}")
        
        # Step 2: Merge tokens
        token_list_after_merge = pre_tokens.copy()
        print(f"Step 2 - After merge: {token_list_after_merge}")
        
        # Step 3: Basic encode
        combined_bytes = b"".join(token_list_after_merge)
        print(f"Combined bytes: {combined_bytes}")
        
        # Convert to string for _basic_encode
        text_for_basic_encode = combined_bytes.decode('utf-8', errors='replace')
        print(f"Text for basic encode: {text_for_basic_encode}")
        
        # Now do the basic encoding
        byte_tokens = [bytes([b]) for b in text_for_basic_encode.encode('utf-8')]
        print(f"Byte tokens: {byte_tokens}")
        
        # Create reverse vocab for lookup
        reverse_vocab = {v: k for k, v in vocab.items()}
        
        token_ids = []
        for byte_token in byte_tokens:
            if byte_token in reverse_vocab:
                token_ids.append(reverse_vocab[byte_token])
                print(f"Byte token {byte_token} -> ID {reverse_vocab[byte_token]}")
            else:
                print(f"Byte token {byte_token} NOT FOUND in vocabulary!")
                raise ValueError(f"Token {byte_token} not found in vocabulary.")
        
        print(f"Final token IDs: {token_ids}")

# Now let's check what tiktoken would do
import tiktoken
reference_tokenizer = tiktoken.get_encoding("gpt2")
reference_ids = reference_tokenizer.encode(test_string)
print(f"\nTiktoken reference IDs: {reference_ids}")
print(f"Tiktoken decoded: {reference_tokenizer.decode(reference_ids)}")
