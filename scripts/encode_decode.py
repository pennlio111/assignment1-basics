import argparse
from tests.tokenizer import Tokenizer

def main():
    """
    Main function to load the trained BPE tokenizer to encode and decode text.
    """
    # Example input path and parameters
    parser = argparse.ArgumentParser(description="Load a trained BPE tokenizer and encode/decode.")
    parser.add_argument("--vocab_file_path", type=str, default="./tests/fixtures/gpt2_vocab.json",
                        help="Path to the trained vocab file.")
    parser.add_argument("--merge_file_path", type=str, default="./tests/fixtures/gpt2_merges.txt", help="Path to the trained merges file.")
    parser.add_argument("--special_tokens", nargs="+", default=["<|endoftext|>"],
                        help="List of special tokens to be included in the vocabulary.")
    args = parser.parse_args()
    # Load the tokenizer from files
    vocab_filepath = args.vocab_file_path
    merges_filepath = args.merge_file_path
    special_tokens = args.special_tokens

    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_filepath, merges_filepath=merges_filepath, special_tokens=special_tokens)
    return tokenizer


if __name__ == "__main__":
    main()