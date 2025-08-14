#!/usr/bin/env python3

from tests.common import gpt2_bytes_to_unicode
import json

# Load merges
with open("tests/fixtures/gpt2_merges.txt") as f:
    gpt2_bpe_merges = []
    for line in f:
        cleaned_line = line.rstrip()
        if cleaned_line and len(cleaned_line.split(" ")) == 2:
            gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))

print(f"Loaded {len(gpt2_bpe_merges)} merges")

# Convert to bytes
gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
merges = []
for merge_token_1, merge_token_2 in gpt2_bpe_merges:
    byte_merge_1 = bytes([gpt2_byte_decoder[token] for token in merge_token_1])
    byte_merge_2 = bytes([gpt2_byte_decoder[token] for token in merge_token_2])
    merges.append((byte_merge_1, byte_merge_2))

print(f"Converted to {len(merges)} byte merges")

# Check for the specific merge we need
target_merge = (b'\xf0', b'\x9f')
print(f"\nLooking for merge: {target_merge}")

if target_merge in merges:
    idx = merges.index(target_merge)
    print(f"  Found at index {idx}")
else:
    print(f"  NOT found in merges")

# Check what would be created by this merge
if target_merge in merges:
    merged_result = target_merge[0] + target_merge[1]
    print(f"  This merge would create: {merged_result}")
    
    # Check if this merged result exists in vocabulary
    from tests.common import gpt2_bytes_to_unicode
    import json
    
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    
    with open("tests/fixtures/gpt2_vocab.json") as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    
    vocab = {}
    for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items():
        byte_sequence = bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        vocab[gpt2_vocab_index] = byte_sequence
    
    if merged_result in vocab.values():
        token_id = None
        for tid, bytes_val in vocab.items():
            if bytes_val == merged_result:
                token_id = tid
                break
        print(f"  Merged result {merged_result} exists in vocab as token {token_id}")
    else:
        print(f"  Merged result {merged_result} NOT in vocab")

# Let's also check what merges we DO have that start with b'\xf0'
print(f"\nMerges that start with b'\\xf0':")
xf0_merges = [merge for merge in merges if merge[0] == b'\xf0']
for i, merge in enumerate(xf0_merges[:5]):  # Show first 5
    print(f"  {i}: {merge} -> {merge[0] + merge[1]}")

# And merges that start with b'\xf0\x9f'
print(f"\nMerges that start with b'\\xf0\\x9f':")
xf0_x9f_merges = [merge for merge in merges if merge[0] == b'\xf0\x9f']
for i, merge in enumerate(xf0_x9f_merges[:5]):  # Show first 5
    print(f"  {i}: {merge} -> {merge[0] + merge[1]}")
