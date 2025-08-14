#!/usr/bin/env python3

import regex as re
from tests.constants import PAT

test_string = "Héllò hôw are ü? 🙃"
print(f"Testing string: '{test_string}'")
print(f"Regex pattern: {PAT}")

# Test the regex
pre_tokens = re.findall(PAT, test_string)
print(f"\nPre-tokens: {pre_tokens}")

# Check each token
for i, token in enumerate(pre_tokens):
    print(f"  {i}: '{token}' (length: {len(token)}, is ASCII: {token.isascii()})")

# Test with just the first part
test_string2 = "Héllò hôw are ü?"
print(f"\n\nTesting string: '{test_string2}'")
pre_tokens2 = re.findall(PAT, test_string2)
print(f"Pre-tokens: {pre_tokens2}")

# Check each token
for i, token in enumerate(pre_tokens2):
    print(f"  {i}: '{token}' (length: {len(token)}, is ASCII: {token.isascii()})")
