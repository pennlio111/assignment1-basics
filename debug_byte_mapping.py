#!/usr/bin/env python3

from tests.common import gpt2_bytes_to_unicode
import json

# Get the byte-to-unicode mapping
byte_to_unicode = gpt2_bytes_to_unicode()
unicode_to_byte = {v: k for k, v in byte_to_unicode.items()}

print(f"Byte-to-unicode mapping has {len(byte_to_unicode)} entries")

# Check some key mappings
print("\nSome key mappings:")
for byte_val in [32, 33, 34, 65, 97]:  # space, !, ", A, a
    if byte_val in byte_to_unicode:
        unicode_char = byte_to_unicode[byte_val]
        print(f"  Byte {byte_val} (0x{byte_val:02x}) -> '{unicode_char}' (U+{ord(unicode_char):04X})")

# Check the reverse mapping
print("\nReverse mapping examples:")
for unicode_char in ['Ġ', '!', 'A', 'a']:
    if unicode_char in unicode_to_byte:
        byte_val = unicode_to_byte[unicode_char]
        print(f"  '{unicode_char}' (U+{ord(unicode_char):04X}) -> Byte {byte_val} (0x{byte_val:02x})")

# Load the vocabulary and check some entries
with open("tests/fixtures/gpt2_vocab.json", "r") as f:
    gpt2_vocab = json.load(f)

print(f"\nVocabulary has {len(gpt2_vocab)} entries")

# Check some vocabulary entries
print("\nSome vocabulary entries:")
for token, token_id in list(gpt2_vocab.items())[:10]:
    print(f"  '{token}' -> {token_id}")

# Now let's check what happens when we process a vocabulary entry
print("\nProcessing vocabulary entries:")
for token, token_id in list(gpt2_vocab.items())[:5]:
    print(f"\nToken: '{token}' (ID: {token_id})")
    
    # Convert each character in the token to bytes using the mapping
    try:
        byte_sequence = bytes([unicode_to_byte[char] for char in token])
        print(f"  Converted to bytes: {byte_sequence}")
        print(f"  As hex: {byte_sequence.hex()}")
        
        # Check if this byte sequence is what we expect
        if token == "!":
            print(f"  Expected: b'!' (0x21)")
            print(f"  Got: {byte_sequence}")
            print(f"  Match: {byte_sequence == b'!'}")
            
    except KeyError as e:
        print(f"  Error: Character {e} not found in unicode_to_byte mapping")

# Check if the unicode character 'Ġ' exists in the vocabulary
print(f"\nChecking for 'Ġ' in vocabulary: {'Ġ' in gpt2_vocab}")
if 'Ġ' in gpt2_vocab:
    print(f"  'Ġ' -> {gpt2_vocab['Ġ']}")

# Check what byte 'Ġ' maps to
if 'Ġ' in unicode_to_byte:
    byte_val = unicode_to_byte['Ġ']
    print(f"  'Ġ' maps to byte {byte_val} (0x{byte_val:02x})")
    print(f"  Expected space byte: 32 (0x20)")
    print(f"  Match: {byte_val == 32}")
