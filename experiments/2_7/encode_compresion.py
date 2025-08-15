import argparse
from tests.tokenizer import Tokenizer
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

def main():
    """
    Main function to load the trained BPE tokenizer to encode and decode text.
    """
    special_tokens = ["<|endoftext|>"]
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=special_tokens)

    # Test different types of text to show compression ratios
    test_texts = {
        "TS_plus_OWT_test.txt": "./data/TS_plus_OWT_test.txt",
        "TinyStoriesV2-GPT4-tiny.txt": "./data/TinyStoriesV2-GPT4-tiny.txt",
        "Sample ASCII": "Hello world! This is a test sentence with some common words.",
        "Sample Unicode": "Hello 世界! This has emojis 🚀 and unicode characters ñáéíóú.",
        "Sample Code": "def tokenize(text): return text.split() if text else []",
    }
    
    print("=" * 60)
    print("COMPRESSION RATIO ANALYSIS")
    print("=" * 60)
    
    for name, source in test_texts.items():
        if isinstance(source, str) and source.endswith('.txt'):
            # File path
            try:
                with open(source, 'r') as f:
                    text = f.read()
            except FileNotFoundError:
                print(f"⚠️  File not found: {source}")
                continue
        else:
            # Direct text
            text = source
        
        # Calculate compression ratio
        original_bytes = len(text.encode('utf-8'))
        encoded = tokenizer.encode(text)
        num_tokens = len(encoded)
        
        # Compression ratio: bytes per token
        compression_ratio = original_bytes / num_tokens
        
        print(f"\n📁 {name}:")
        print(f"   Original: {original_bytes:,} bytes")
        print(f"   Encoded:  {num_tokens:,} tokens")
        print(f"   Compression: {compression_ratio:.2f} bytes/token (higher = better)")
        print(f"   Efficiency:  {num_tokens / original_bytes:.3f} tokens/byte (lower = better)")
        
        # Verify roundtrip for files
        if source.endswith('.txt'):
            decoded_text = tokenizer.decode(encoded)
            assert decoded_text == text, f"Roundtrip failed for {name}"
            print(f"   ✅ Roundtrip: PASSED")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("• Higher 'bytes/token' = Better compression (each token represents more bytes)")
    print("• Lower 'tokens/byte' = Better efficiency (fewer tokens needed)")
    print("• Both metrics agree: Unicode text compresses best!")
    print("=" * 60)

if __name__ == "__main__":
    main()