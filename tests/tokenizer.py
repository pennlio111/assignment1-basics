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
    
    
    def _merge_bpe(self, token_ids: list[int]) -> list[int]:
        """
        Merge BPE tokens to the smallest rank merges in the vocabulary (that generates the most common pairs)
        Args:
            token_ids (list[int]): A list of token IDs to merge.
        
        Returns:
            list[int]: A list of merged token IDs.
        """
        # Pre-compute merge lookup table for O(1) access
        if not hasattr(self, '_merge_ranks'):
            self._merge_ranks = {merge: rank for rank, merge in enumerate(self.merges)}
        
        # Pre-compute byte sequences to avoid repeated lookups
        byte_sequences = [self.vocab[tid] for tid in token_ids]
        
        while True:
            best_rank = float('inf')
            best_pos = -1
            
            # Find the pair with lowest rank (highest priority) in the merges
            for i in range(len(byte_sequences) - 1):
                pair = (byte_sequences[i], byte_sequences[i + 1])
                
                # O(1) lookup instead of O(n) search
                rank = self._merge_ranks.get(pair, -1)
                
                if rank != -1 and rank < best_rank:
                    best_rank = rank
                    best_pos = i
            
            if best_pos == -1:
                break  # No more merges possible
            
            # Merge the best pair
            merged_bytes = byte_sequences[best_pos] + byte_sequences[best_pos + 1]
            
            # Find the token ID for the merged bytes
            if merged_bytes in self.reverse_vocab:
                new_id = self.reverse_vocab[merged_bytes]
                new_byte_sequence = self.vocab[new_id]
                
                # Update byte sequences efficiently
                byte_sequences[best_pos] = new_byte_sequence
                byte_sequences.pop(best_pos + 1)
                
                # Update token IDs efficiently
                token_ids[best_pos] = new_id
                token_ids.pop(best_pos + 1)
            else:
                # If merged token not in vocab, mark this pair as unmergeable
                # by setting a very high rank to avoid checking it again
                self._merge_ranks[(byte_sequences[best_pos], byte_sequences[best_pos + 1])] = float('inf')
        
        return token_ids


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
                pre_tokens_strs = re.findall(PAT, chunk)
                pre_tokens_bytes = [pre_token.encode('utf-8') for pre_token in pre_tokens_strs]
                # Encode each pre-token to token IDs
                for pre_token in pre_tokens_bytes:
                    # Convert pre-token to individual byte tokens first
                    byte_tokens = [bytes([b]) for b in pre_token]
                    # Encode bytes to token IDs
                    token_ids = [self.reverse_vocab[bt] for bt in byte_tokens if bt in self.reverse_vocab]                    
                    """
                    LEARNING:
                    The encoding process is done in three steps:
                    1. First, the text is split into chunks, and each chunk is pre-tokenized.
                    2. Then, convert the pre-token to list of individual tokens
                    3. Apply BPE merging to the list of individual tokens, prioritize the smallest rank merges in the vocabulary (that generates the most common pairs)
                    """
                    token_ids = self._merge_bpe(token_ids)
                    full_text_after_encoding.extend(token_ids)
        return full_text_after_encoding
                

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable of text into tokens.
        
        Args:
            iterable (iterable): An iterable containing text strings.
        
        Returns:
            list: A list of token IDs for each text in the iterable.
        """
        for text in iterable:
            # Get the token IDs for this text
            token_ids = self.encode(text)
            # Yield each individual token ID
            """
            LEARNING:
            1. the generator approach returns a generator object, which is an iterator being able to yield values one at a time when called
            2. when it is called, it yield the next value in the sequence, and pause the execution until the next call
            """
            for token_id in token_ids:
                yield token_id
    
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
