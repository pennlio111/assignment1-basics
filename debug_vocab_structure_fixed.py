#!/usr/bin/env python3

import json
from tests.common import gpt2_bytes_to_unicode

# Load the vocabulary
with open("tests/fixtures/gpt2_vocab.json") as vocab_f:
    gpt2_vocab = json.load(vocab_f)

print("Vocabulary structure analysis:")
print(f"Type: {type(gpt2_vocab)}")
print(f"Length: {len(gpt2_vocab)}")

# Check the first few items
print("\nFirst 10 items:")
for i, (key, value) in enumerate(list(gpt2_vocab.items())[:10]):
    print(f"  {i}: Key: '{key}' (type: {type(key)}), Value: {value} (type: {type(value)})")

# Check if token 8582 exists
print(f"\nToken 8582 exists: {'8582' in gpt2_vocab}")
if '8582' in gpt2_vocab:
    print(f"  Value: {gpt2_vocab['8582']}")

# Check tokens 247 and 225
for token_id in [247, 225]:
    exists = str(token_id) in gpt2_vocab
    print(f"Token {token_id} exists: {exists}")
    if exists:
        print(f"  Value: {gpt2_vocab[str(token_id)]}")

# Let's look at the byte mapping
print(f"\nByte mapping analysis:")
gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

# Check what the emoji bytes map to
emoji_bytes = "🙃".encode('utf-8')
print(f"Emoji bytes: {emoji_bytes}")

for i, byte in enumerate(emoji_bytes):
    if byte in gpt2_byte_decoder:
        unicode_char = gpt2_byte_decoder[byte]
        print(f"  Byte {i}: {byte} (0x{byte:02x}) -> unicode: '{unicode_char}'")
    else:
        print(f"  Byte {i}: {byte} (0x{byte:02x}) -> not in mapping")

# Let's check if there are any 3-byte sequences in the vocabulary
print(f"\nLooking for 3-byte sequences that might match the first 3 bytes:")
first_three = emoji_bytes[:3]
print(f"First 3 bytes: {first_three}")

# Convert to the GPT-2 unicode representation
first_three_unicode = ''.join([gpt2_byte_decoder.get(b, chr(b)) for b in first_three])
print(f"First 3 bytes as GPT-2 unicode: '{first_three_unicode}'")

# Check if this sequence exists in the vocabulary
if first_three_unicode in gpt2_vocab:
    print(f"  Found in vocab with ID: {gpt2_vocab[first_three_unicode]}")
else:
    print(f"  Not found in vocab")
    
    # Let's search for similar patterns
    print(f"  Searching for similar patterns...")
    for key, value in gpt2_vocab.items():
        if len(key) >= 3 and key.startswith(first_three_unicode[:2]):
            print(f"    Similar: '{key}' -> {value}")
            break
