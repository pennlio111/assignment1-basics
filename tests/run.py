from adapters import run_train_bpe

vocab, _ = run_train_bpe(
    input_path="../data/TinyStoriesV2-GPT4-valid.txt",
    vocab_size=256,
    special_tokens=["<|endoftext|>"])

# print("Vocabulary size:", len(vocab))
# print(vocab)

# print("Total unique tokens:", len(set(vocab.values())))