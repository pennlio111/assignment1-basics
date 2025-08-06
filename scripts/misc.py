import pickle
import json

with open("./data/my_bpe_data", "rb") as f:
    bpe_data = pickle.load(f)

vocab = bpe_data["vocab"]
merges = bpe_data["merges"]

# save the vocabulary as json file 
with open("./data/my_bpe_vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

with open("./data/my_bpe_merges.pkl", "wb") as f:
    pickle.dump(merges, f)

space_index = [k for k, v in vocab.items() if v == b' ']
print("Space token index:", space_index)  # should be [len(special_tokens) + 32]

# longest_token = b""
# for index, token in vocab.items():
#     if len(token) > len(longest_token):
#         longest_token = token
# print(f"Longest token: {longest_token.decode('utf-8')} with length {len(longest_token)}") 