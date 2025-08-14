#!/usr/bin/env python3

import json
from tests.common import gpt2_bytes_to_unicode

# Load the vocabulary
with open("tests/fixtures/gpt2_vocab.json") as vocab_f:
    gpt2_vocab = json.load(vocab_f)

print("Vocabulary structure:")
print(f"Type: {type(gpt2_vocab)}")
print(f"Length: {len(gpt2_vocab)}")

# Check the first few items
print("\nFirst 10 items:")
for i, (key, value) in enumerate(list(gpt2_vocab.items())[:10]):
    print(f"  {i}: Key: '{key}' (type: {type(key)}), Value: {value} (type: {type(value)})")

# Now let's test the vocabulary creation logic
print(f"\nTesting vocabulary creation logic:")
gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

# Test with a simple case first
test_key = "!"
test_value = gpt2_vocab[test_key]
print(f"Test case: key='{test_key}', value={test_value}")

# Convert the GPT-2 token string to bytes using the byte decoder
try:
    byte_sequence = bytes([gpt2_byte_decoder[token] for token in test_key])
    print(f"  Converted to bytes: {byte_sequence}")
    print(f"  This should go into vocab[{test_value}] = {byte_sequence}")
except Exception as e:
    print(f"  Error: {e}")

# Test with a more complex case
test_key2 = "Ġt"  # This should be a common pattern
if test_key2 in gpt2_vocab:
    test_value2 = gpt2_vocab[test_key2]
    print(f"\nTest case 2: key='{test_key2}', value={test_value2}")
    
    try:
        byte_sequence2 = bytes([gpt2_byte_decoder[token] for token in test_key2])
        print(f"  Converted to bytes: {byte_sequence2}")
        print(f"  This should go into vocab[{test_value2}] = {byte_sequence2}")
    except Exception as e:
        print(f"  Error: {e}")

# Now let's create the actual vocab
print(f"\nCreating actual vocabulary:")
vocab = {}
for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items():
    # Convert the GPT-2 token string to bytes using the byte decoder
    byte_sequence = bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
    vocab[gpt2_vocab_index] = byte_sequence

print(f"Created vocab with {len(vocab)} items")

# Check if our test cases are in the created vocab
if test_value in vocab:
    print(f"  vocab[{test_value}] = {vocab[test_value]} (should be {byte_sequence})")
    print(f"  Match: {vocab[test_value] == byte_sequence}")

if 'test_value2' in locals() and test_value2 in vocab:
    print(f"  vocab[{test_value2}] = {vocab[test_value2]} (should be {byte_sequence2})")
    print(f"  Match: {vocab[test_value2] == byte_sequence2}")

# Check if we have the right tokens for the emoji
emoji_bytes = "🙃".encode('utf-8')
print(f"\nEmoji analysis:")
print(f"  Emoji bytes: {emoji_bytes}")

# Check if we have individual byte tokens
for i, byte in enumerate(emoji_bytes):
    if byte in vocab.values():
        # Find the token ID
        token_id = None
        for tid, bytes_val in vocab.items():
            if bytes_val == bytes([byte]):
                token_id = tid
                break
        print(f"  Byte {i}: {byte} (0x{byte:02x}) -> found as token {token_id}")
    else:
        print(f"  Byte {i}: {byte} (0x{byte:02x}) -> NOT found in vocab")

# Check if we have the 3-byte sequence
first_three = emoji_bytes[:3]
if first_three in vocab.values():
    token_id = None
    for tid, bytes_val in vocab.items():
        if bytes_val == first_three:
            token_id = tid
            break
    print(f"  First 3 bytes {first_three} -> found as token {token_id}")
else:
    print(f"  First 3 bytes {first_three} -> NOT found in vocab")
