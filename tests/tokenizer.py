from .constants import PAT
from typing import Iterable, Iterator
import pickle
import regex as re

class Tokenizer(object):
    """
    A class to handle tokenization of text data.
    """
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Initialize the Tokenizer with vocabulary and merges.
        Add special tokens to the vocabulary if provided and not present.
        
        Args:
            vocab (dict[int, bytes]): A dictionary mapping token IDs to byte strings.
            merges (list[tuple]): A list of tuples representing merges.
            special_tokens (list, optional): List of special tokens to include.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = list(set(special_tokens)) if special_tokens else []  # Ensure special tokens are unique
        self.reverse_vocab = {v: k for k, v in vocab.items()}  # Reverse mapping for decoding
        
        # Add special tokens to the vocabulary if they are not already present
        for special_token in self.special_tokens:
            if special_token.encode('utf-8') not in self.vocab.values():
                self.vocab[len(self.vocab)] = special_token.encode('utf-8')
                print(f"Added special token '{special_token}' to vocabulary with ID {len(self.vocab) - 1}.")
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Load a tokenizer from vocabulary and merges files.
        
        Args:
            vocab_filepath (str): Path to the vocabulary file.
            merges_filepath (str): Path to the merges file.
            special_tokens (list, optional): List of special tokens to include.
        
        Returns:
            Tokenizer: An instance of the Tokenizer class.
        """
        # assume vocab is in JSON format dict[int, bytes]
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        # merges is list of tuple[bytes, bytes]
        
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def _split_and_preserve_special_tokens(self, text, special_tokens):
            """
            LEARNING:
            1. With (), the regex patter is constructed to capture the special tokens after the split. 
            2. For example, if special_tokens = ["<|endoftext|>", "<|startoftext|>"], the pattern becomes 
                "(<\|endoftext\|>|<\|startoftext\|>)". This means that when we split the text, 
                the special tokens themselves will be included in the resulting list of parts.
            """
            # Create a regex pattern that captures the special tokens
            escaped_tokens = list(map(re.escape, special_tokens))
            pattern = f"({'|'.join(map(re.escape, special_tokens))})"
            # Split while preserving the tokens as separate elements
            parts = re.split(pattern, text)
            return [part for part in parts if part]  # Remove empty strings    
    
    
    def _basic_encode(self, text: str) -> list[int]:
        """
        A basic encoding method that splits text into bytes and maps them to token IDs.
        
        Args:
            text (str): The input text to encode.
        
        Returns:
            list: A list of token IDs.
        """
        byte_tokens = [bytes([b]) for b in text.encode('utf-8')]
        token_ids = []
        for byte_token in byte_tokens:
            if byte_token in self.reverse_vocab:
                token_ids.append(self.reverse_vocab[byte_token])
            else:
                raise ValueError(f"Token {byte_token} not found in vocabulary.")
        return token_ids
    
    def _merge_tokens(self, byte_tokens: list[bytes]) -> list[bytes]:
        """
        Merge byte tokens based on the BPE merges.
        
        Args:
            byte_tokens (list): A list of byte tokens to merge.
        
        Returns:
            list: A list of merged byte tokens.
        """
        token_list = byte_tokens.copy()
        scan_token_list = True # indicate if need to scan the list again
        while scan_token_list:
            scan_token_list = False
            i = 0
            while i < len(token_list) - 1:
                pair = (token_list[i], token_list[i + 1])
                for merge in self.merges:
                    if pair == merge:
                        new_token = merge[0] + merge[1]
                        token_list[i] = new_token # replace the first token with the merged one
                        del token_list[i + 1] #
                        scan_token_list = True
                        break
                else:
                    i += 1
        return token_list


    def encode(self, text: str) -> list[int]:
        """
        Encode a given text into tokens.
        
        Args:
            text (str): The input text to tokenize.
        
        Returns:
            list: A list of token IDs.
        """    

        chunks = self.split_and_preserve_special_tokens(text, self.special_tokens) if self.special_tokens else [text]
        print(f"Chunks after split: {chunks}")
        # pre-tokenize each chunk
        full_text_in_pre_token = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                full_text_in_pre_token.append(chunk.encode('utf-8'))
            else:
                pre_tokens = re.findall(PAT, chunk)
                full_text_in_pre_token.extend(pre_token.encode('utf-8') for pre_token in pre_tokens)
        print(f"Full text in pre-token: {full_text_in_pre_token}")
        # todo: how to handle special tokens in the pre-tokenization, tokenize and merge steps, to ensure they are not split?

        token_list_after_merge = self._merge_tokens(full_text_in_pre_token)
        
        # convert byte tokens to IDs
        token_ids = []
        # print(f"Byte pre-token list after merge: {token_list_after_merge}")
        for byte_pre_token_list in token_list_after_merge:
            for byte_token in byte_pre_token_list:
                if byte_token in self.reverse_vocab:
                    token_ids.append(self.reverse_vocab[byte_token])
                else:
                    raise ValueError(f"Token {byte_token} not found in vocabulary.")
        return token_ids
                

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[str]:
        """
        Encode an iterable of text into tokens.
        
        Args:
            iterable (iterable): An iterable containing text strings.
        
        Returns:
            list: A list of token IDs for each text in the iterable.
        """
        raise NotImplementedError("Tokenization of iterable not implemented")
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back into text.
        
        Args:
            ids (list): A list of token IDs to decode.
        
        Returns:
            str: The decoded text.
        """
        byte_tokens = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_tokens.append(self.vocab[token_id])
            else:
                raise ValueError(f"Token ID {token_id} not found in vocabulary.")
        decoded_text = b"".join(byte_tokens).decode('utf-8', errors='replace')
        return decoded_text
