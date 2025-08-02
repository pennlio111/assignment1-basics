from typing import Iterable, Iterator
import json

class Tokenizer(object):
    """
    A class to handle tokenization of text data.
    """
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Initialize the Tokenizer with vocabulary and merges.
        
        Args:
            vocab (dict): A dictionary mapping token IDs to byte strings.
            merges (list): A list of tuples representing merges.
            special_tokens (list, optional): List of special tokens to include.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
    
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
        
        with open(vocab_filepath, "r") as f:
            reverse_vocab = json.load(f)
        vocab = {v:k.encode('utf-8') for k, v in reverse_vocab.items()}
        
        with open(merges_filepath, "r") as f:
            merges = [tuple(line.strip().split()) for line in f.readlines()]
            merges = [(bytes(merge[0], 'utf-8'), bytes(merge[1], 'utf-8')) for merge in merges]
        
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        Encode a given text into tokens.
        
        Args:
            text (str): The input text to tokenize.
        
        Returns:
            list: A list of token IDs.
        """
        raise NotImplementedError("Tokenization not implemented")
    

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
        raise NotImplementedError("Decoding not implemented")
