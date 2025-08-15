import cProfile
import pstats
import io
from tests.common import FIXTURES_PATH
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

def benchmark_tokenizer():
    """Benchmark the tokenizer with various text sizes"""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, 
        merges_path=MERGES_PATH, 
        special_tokens=['<|endoftext|>']
    )
    
    # Test texts of different sizes
    test_texts = [
        "The quick brown fox jumps over the lazy dog. " * 1000,  # Large
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i}: {len(text)} characters ---")
        
        # Encode the text
        encoded = tokenizer.encode(text)
        
        # Print results
        print(f"Input length: {len(text)} characters")
        print(f"Encoded length: {len(text.encode('utf-8'))} bytes")
        print(f"Output tokens: {len(encoded)} tokens")
        print(f"Compression: {len(text.encode('utf-8')) / len(encoded):.2f} bytes/token")

def main():
    """Main function with profiling"""
    print("Starting tokenizer benchmark with profiling...")
    
    # Profile the entire main function
    profiler = cProfile.Profile()
    profiler.enable()
    
    benchmark_tokenizer()
    
    profiler.disable()
    
    # Print overall profiling results
    print("\n" + "="*60)
    print("OVERALL PROFILING RESULTS")
    print("="*60)
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print(s.getvalue())
    
    # Save detailed profile to file
    profiler.dump_stats('tokenizer_profile.prof')
    print("\nDetailed profile saved to 'tokenizer_profile.prof'")
    print("You can analyze it with: python -m pstats tokenizer_profile.prof")

if __name__ == "__main__":
    main()