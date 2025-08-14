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
vocab = {}
for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items():
    try:
        byte_sequence = bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        vocab[gpt2_vocab_index] = byte_sequence
    except KeyError as e:
        print(f"Error processing '{gpt2_vocab_item}': character {e} not found in byte decoder")
        break

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

print(f"Created vocabulary with {len(vocab)} entries")
print(f"Created {len(merges)} merges")

# Create the tokenizer
tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=None)

print(f"\nTokenizer created successfully")
print(f"Tokenizer vocab size: {len(tokenizer.vocab)}")
print(f"Tokenizer merges size: {len(tokenizer.merges)}")
print(f"Tokenizer reverse_vocab size: {len(tokenizer.reverse_vocab)}")

# Check if our test bytes exist in the reverse vocab
test_bytes = [b'\xf0', b'\x9f', b'\x99', b'\x83']
print(f"\nChecking test bytes in tokenizer reverse vocab:")
for test_byte in test_bytes:
    if test_byte in tokenizer.reverse_vocab:
        token_id = tokenizer.reverse_vocab[test_byte]
        print(f"  {test_byte} -> Token ID {token_id}")
    else:
        print(f"  {test_byte} -> NOT FOUND")

# Test the _basic_encode method directly
print(f"\nTesting _basic_encode method:")
test_string = "🙃"
test_bytes_actual = test_string.encode('utf-8')
print(f"String '{test_string}' encodes to: {test_bytes_actual}")

try:
    result = tokenizer._basic_encode(test_string)
    print(f"_basic_encode result: {result}")
except Exception as e:
    print(f"Error in _basic_encode: {e}")
    import traceback
    traceback.print_exc()

# Test the _merge_tokens method
print(f"\nTesting _merge_tokens method:")
test_byte_tokens = [bytes([b]) for b in test_bytes_actual]
print(f"Input byte tokens: {test_byte_tokens}")

try:
    merged_result = tokenizer._merge_tokens(test_byte_tokens)
    print(f"_merge_tokens result: {merged_result}")
except Exception as e:
    print(f"Error in _merge_tokens: {e}")
    import traceback
    traceback.print_exc()

# Test the full encode method
print(f"\nTesting full encode method:")
try:
    full_result = tokenizer.encode(test_string)
    print(f"Full encode result: {full_result}")
except Exception as e:
    print(f"Error in full encode: {e}")
    import traceback
    traceback.print_exc()
