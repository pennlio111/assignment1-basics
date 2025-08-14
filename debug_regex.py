#!/usr/bin/env python3

import regex as re
from tests.constants import PAT

test_string = "Hello, how are you?"
print(f"Testing string: '{test_string}'")
print(f"Regex pattern: {PAT}")

# Test the regex
pre_tokens = re.findall(PAT, test_string)
print(f"\nPre-tokens: {pre_tokens}")

# Check each token
for i, token in enumerate(pre_tokens):
    print(f"  {i}: '{token}' (length: {len(token)}, starts with space: {token.startswith(' ')})")

# Test with a simpler string
test_string2 = "Hello"
print(f"\n\nTesting string: '{test_string2}'")
pre_tokens2 = re.findall(PAT, test_string2)
print(f"Pre-tokens: {pre_tokens2}")

# Test with just a space
test_string3 = " how"
print(f"\n\nTesting string: '{test_string3}'")
pre_tokens3 = re.findall(PAT, test_string3)
print(f"Pre-tokens: {pre_tokens3}")

# Test with the emoji
test_string4 = "🙃"
print(f"\n\nTesting string: '{test_string4}'")
pre_tokens4 = re.findall(PAT, test_string4)
print(f"Pre-tokens: {pre_tokens4}")
print(f"  Is ASCII: {test_string4.isascii()}")
print(f"  Length: {len(test_string4)}")
