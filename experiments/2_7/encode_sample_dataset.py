from tests.tokenizer import Tokenizer
import numpy as np
DATA_PATH = './data/'

def main():
    tokenizer = Tokenizer.from_files(
        vocab_filepath=DATA_PATH + 'my_ts_vocab.pkl',
        merges_filepath=DATA_PATH + 'my_ts_merges.pkl',
        special_tokens=['<|endoftext|>']
    )
    # encode the tiny stories dataset
    def encode_dataset(dataset_path):
        # serialize the token_ids as numpy array of data type int16
        with open(dataset_path, 'r') as f:
            dataset = f.read()
        token_ids = tokenizer.encode(dataset)
        token_ids = np.array(token_ids, dtype=np.uint16)
        print(token_ids.shape)
        np.save(DATA_PATH + dataset_path.split('/')[-1] + '_token_ids.npy', token_ids)
    
    encode_dataset(DATA_PATH + 'TinyStoriesV2-GPT4-valid.txt')
    print("TinyStoriesV2-GPT4-valid.txt encoded")
    encode_dataset(DATA_PATH + 'TinyStoriesV2-GPT4-train.txt')
    print("TinyStoriesV2-GPT4-train.txt encoded")

if __name__ == "__main__":
    main()