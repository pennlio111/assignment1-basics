import argparse
from tests.tokenizer import Tokenizer

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
    vocab_filepath = args.vocab_file_path
    merges_filepath = args.merge_file_path
    special_tokens = args.special_tokens

    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_filepath, merges_filepath=merges_filepath, special_tokens=special_tokens)
    
    # vocab
    # print("Vocabulary size:", len(tokenizer.vocab))
    # assert b" " in tokenizer.vocab.values(), "Space token not found in vocabulary."
    # merges
    # print("Number of merges:", len(tokenizer.merges))
    # print("First 10 merges:", tokenizer.merges[:10])

    # Example usage of the tokenizer
    sample_text = "<|endoftext|>\n\n"
    encoded = tokenizer.encode(sample_text)
    print("encoded:", encoded)
    decoded_text = tokenizer.decode(encoded)
    # print("decoded_text:", decoded_text)
    assert decoded_text == sample_text, "Decoded text does not match the original."

if __name__ == "__main__":
    main()