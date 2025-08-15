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
        
        # Add special tokens to the vocabulary if they are not already present
        for special_token in self.special_tokens:
            if special_token.encode('utf-8') not in self.vocab.values():
                self.vocab[len(self.vocab)] = special_token.encode('utf-8')
                print(f"Added special token '{special_token}' to vocabulary with ID {len(self.vocab) - 1}.")
        
        self.reverse_vocab = {v: k for k, v in vocab.items()}  # Reverse mapping for decoding
    
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
                the special tokens themselves will be included in the resulting list of parts, as a separate item in the list.
            """
            # Sort special tokens by length in descending order to match longer tokens first
            sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
            # Create a regex pattern that captures the special tokens
            pattern = f"({'|'.join(map(re.escape, sorted_special_tokens))})"
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
    
    def _merge_pre_token(self, byte_tokens: list[bytes]) -> list[bytes]:
        """
        Merge byte tokens based on the BPE merges for a single pre-token.
        
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
                        # print(f"Merging {pair}")
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

        chunks = self._split_and_preserve_special_tokens(text, self.special_tokens) if self.special_tokens else [text]
        # pre-tokenize each chunk
        full_text_after_encoding = []
        for chunk in chunks:
            if chunk in self.special_tokens: # the entire chunk is a special token
                full_text_after_encoding.append(self.reverse_vocab[chunk.encode('utf-8')])
            else:
                # pre-tokenize the chunk
                pre_tokens_raw = re.findall(PAT, chunk)
                pre_tokens = [pre_token.encode('utf-8') for pre_token in pre_tokens_raw]
                # Process all pre-tokens using BPE merging (inspired by reference implementation)
                for pre_token in pre_tokens:
                    # Convert pre-token to individual byte tokens first
                    byte_tokens = [bytes([b]) for b in pre_token]
                    # Convert bytes to token IDs
                    token_ids = [self.reverse_vocab[bt] for bt in byte_tokens if bt in self.reverse_vocab]
                    if len(token_ids) == 1:
                        full_text_after_encoding.append(token_ids[0])
                        continue
                    
                    """
                    LEARNING:
                    The encoding process is done in three steps:
                    1. First, the text is split into chunks, and each chunk is pre-tokenized.
                    2. Then, convert the pre-token to list of individual tokens
                    3. Apply BPE merging to the list of individual tokens, prioritize the smallest rank merges in the vocabulary (that generates the most common pairs)
                    """
                    # Apply BPE merging using "greedy smallest-rank merge"
                    while True:
                        best_rank = float('inf')
                        best_pos = -1
                        
                        # Find the pair with lowest rank (highest priority)
                        for i in range(len(token_ids) - 1):
                            # Get the byte sequences for the current pair
                            byte1 = self.vocab[token_ids[i]]
                            byte2 = self.vocab[token_ids[i + 1]]
                            pair = (byte1, byte2)
                            
                            # Find the rank of this pair in merges
                            rank = -1
                            for j, merge in enumerate(self.merges):
                                if merge == pair:
                                    rank = j
                                    break
                            
                            if rank != -1 and rank < best_rank:
                                best_rank = rank
                                best_pos = i
                        
                        if best_pos == -1:
                            break  # No more merges possible
                        
                        # Merge the best pair
                        byte1 = self.vocab[token_ids[best_pos]]
                        byte2 = self.vocab[token_ids[best_pos + 1]]
                        merged_bytes = byte1 + byte2
                        
                        # Find the token ID for the merged bytes
                        if merged_bytes in self.reverse_vocab:
                            new_id = self.reverse_vocab[merged_bytes]
                            # Replace the pair with the merged token
                            token_ids[best_pos:best_pos + 2] = [new_id]
                        else:
                            # If merged token not in vocab, stop merging
                            break
                    
                    full_text_after_encoding.extend(token_ids)
        return full_text_after_encoding
                

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
