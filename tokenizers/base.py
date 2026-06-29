from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    
    @abstractmethod
    def build_vocab(self, text: str, **kwargs) -> None:
        """Build vocabulary and mappings from the input text."""
        pass
    
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert text into a list of token ids."""
        pass
    
    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """Convert token ids back into text."""
        pass
    
    @abstractmethod
    def save_vocab(self, path: str) -> None:
        """Save the vocabulary to a file."""
        pass
    
    @abstractmethod
    def load_vocab(self, path: str) -> None:
        """Load the vocabulary from a file."""
        pass