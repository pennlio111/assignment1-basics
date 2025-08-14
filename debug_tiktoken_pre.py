#!/usr/bin/env python3

import tiktoken
import regex as re

# Get tiktoken's encoding
enc = tiktoken.get_encoding("gpt2")

test_string = "🙃"
print(f"Testing string: {test_string}")
print(f"UTF-8 bytes: {test_string.encode('utf-8')}")

# Check what tiktoken produces
reference_ids = enc.encode(test_string)
print(f"\nTiktoken result: {reference_ids}")
print(f"Tiktoken decoded: {enc.decode(reference_ids)}")

# Let's try to understand tiktoken's internal behavior
# First, let's see if we can access the raw bytes before tokenization
print(f"\nAnalyzing tiktoken's behavior:")

# Try to understand the byte-level tokenization
for token_id in reference_ids:
    try:
        # This might not work, but let's try
        decoded = enc.decode([token_id])
        print(f"  Token {token_id}: '{decoded}' (bytes: {decoded.encode('utf-8')})")
    except Exception as e:
        print(f"  Token {token_id}: Error decoding: {e}")

# Let's also check what happens if we encode individual bytes
print(f"\nByte-by-byte encoding:")
for i, byte in enumerate(test_string.encode('utf-8')):
    byte_str = chr(byte)
    try:
        ids = enc.encode(byte_str)
        print(f"  Byte {i}: {byte} (0x{byte:02x}) -> {ids}")
    except Exception as e:
        print(f"  Byte {i}: {byte} (0x{byte:02x}) -> Error: {e}")

# Let's try to understand the pattern
print(f"\nTrying to understand the pattern:")
print("If tiktoken produces [8582, 247, 225], and we know:")
print("  - 247 and 225 are individual bytes")
print("  - 8582 must represent the first 3 bytes")

# Let's check if our vocabulary has the right tokens
print(f"\nChecking our vocabulary for key tokens:")
try:
    from tests.common import gpt2_bytes_to_unicode
    import json
    
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    
    with open("tests/fixtures/gpt2_vocab.json") as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    
    # Check token 8582
    if "8582" in gpt2_vocab:
        token_str = gpt2_vocab["8582"]
        print(f"  Token 8582 in vocab: '{token_str}'")
        # Convert to bytes
        byte_sequence = bytes([gpt2_byte_decoder[token] for token in token_str])
        print(f"  Token 8582 as bytes: {byte_sequence}")
        print(f"  Length: {len(byte_sequence)}")
        
        # Check if this matches the first 3 bytes of our emoji
        emoji_bytes = test_string.encode('utf-8')
        first_three = emoji_bytes[:3]
        print(f"  First 3 bytes of emoji: {first_three}")
        print(f"  Match: {byte_sequence == first_three}")
    else:
        print("  Token 8582 not found in vocab")
        
    # Check tokens 247 and 225
    for token_id in [247, 225]:
        if str(token_id) in gpt2_vocab:
            token_str = gpt2_vocab[str(token_id)]
            print(f"  Token {token_id} in vocab: '{token_str}'")
            byte_sequence = bytes([gpt2_byte_decoder[token] for token in token_str])
            print(f"  Token {token_id} as bytes: {byte_sequence}")
        else:
            print(f"  Token {token_id} not found in vocab")
            
except Exception as e:
    print(f"  Error checking vocabulary: {e}")
    import traceback
    traceback.print_exc()
