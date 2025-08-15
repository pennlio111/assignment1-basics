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
    # Example input path and parameters
    parser = argparse.ArgumentParser(description="Load a trained BPE tokenizer and encode/decode.")
    parser.add_argument("--vocab_file_path", type=str, default="./data/my_bpe_vocab.pkl",
                        help="Path to the trained vocab file.")
    parser.add_argument("--merge_file_path", type=str, default="./data/my_bpe_merges.pkl", help="Path to the trained merges file.")
    parser.add_argument("--special_tokens", nargs="+", default=["<|endoftext|>"],
                        help="List of special tokens to be included in the vocabulary.")
    args = parser.parse_args()
    # Load the tokenizer from files
    # vocab_filepath = args.vocab_file_path
    # merges_filepath = args.merge_file_path
    special_tokens = args.special_tokens

    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=special_tokens)

    # Example usage of the tokenizer
    # sample_text = "🙃"
    sample_text = "Hello, world! 🙃<|endoftext|>\n\n"

    encoded = tokenizer.encode(sample_text)
    print("encoded:", encoded)
    decoded_text = tokenizer.decode(encoded)
    print("decoded_text:", decoded_text)
    assert decoded_text == sample_text, "Decoded text does not match the original."

if __name__ == "__main__":
    main()