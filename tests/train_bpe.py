import argparse
from adapters import run_train_bpe
import pickle

def main():
    """
    Main function to run the BPE training adapter.
    It uses a small dataset and a specified vocabulary size.
    """
    # Example input path and parameters
    parser = argparse.ArgumentParser(description="Run BPE training on a dataset.")
    parser.add_argument("--input_path", type=str, default="./data/TinyStoriesV2-GPT4-train.txt",
                        help="Path to the input text file for BPE training.")
    parser.add_argument("--vocab_size", type=int, default=500, help="Size of the vocabulary to be created.")
    parser.add_argument("--special_tokens", nargs="+", default=["<|endoftext|>"],
                        help="List of special tokens to be included in the vocabulary.")
    parser.add_argument("--output_path", type=str, default="./data/my_bpe_data", 
                        help="Path to save the BPE vocabulary and merges.")
    args = parser.parse_args()

    vocab, merges = run_train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )

    print("Vocabulary size:", len(vocab))
    print("Sample vocabulary:", list(vocab.items())[:10])

    print("Number of merges:", len(merges))
    print("Sample merges:", merges[:10])

    
    bpe_data = {
        "vocab": vocab,
        "merges": merges,
    }

    # Save the vocabulary and merges to files
    with open(args.output_path, "wb") as f:
        pickle.dump(bpe_data, f)

if __name__ == "__main__":
    main()