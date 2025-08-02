import pickle

with open("./data/my_bpe_data", "rb") as f:
    bpe_data = pickle.load(f)

vocab = bpe_data["vocab"]
merges = bpe_data["merges"]

print(vocab[3522])

longest_token = b""
for index, token in vocab.items():
    if len(token) > len(longest_token):
        longest_token = token
print(f"Longest token: {longest_token.decode('utf-8')} with length {len(longest_token)}") 