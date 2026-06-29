import json

from .base import BaseTokenizer

class CharacterTokenizer(BaseTokenizer):
    def __init__(self):
        self.char_to_index = {}
        self.index_to_char = {}

    def build_vocab(self, text: str) -> None:
        """Build vocabulary and mappings from the input text."""
        vocab = sorted(set(text))
        self.char_to_index = {char: idx for idx, char in enumerate(vocab)}
        self.index_to_char = {idx: char for idx, char in enumerate(vocab)}

    def encode(self, text: str) -> list[int]:
        """Convert text into a list of token ids."""
        if not self.char_to_index:
            raise ValueError("Vocabulary has not been built. Call build_vocab() first.")

        return [self.char_to_index[char] for char in text]

    def decode(self, tokens: list[int]) -> str:
        """Convert token ids back into text."""
        if not self.index_to_char:
            raise ValueError("Vocabulary has not been built. Call build_vocab() first.")

        return "".join(self.index_to_char[token] for token in tokens)
    
    def save_vocab(self, path: str) -> None:
        """Save the vocabulary to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char_to_index, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str) -> None:
        """Load the vocabulary from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            self.char_to_index = json.load(f)

        # Rebuild the index_to_char mapping after loading the vocabulary 
        self.index_to_char = {
            idx: char for char, idx in self.char_to_index.items()
        }

if __name__ == "__main__":
    # Build the vocabulary
    with open("data/wikitext-103/wiki.valid.tokens", "r") as file:
        text = file.read()

    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(text)
    tokenizer.save_vocab("vocabs/char_vocab.json")
    # tokenizer.load_vocab("vocabs/char_vocab.json")
    print(f"Vocabulary size: {len(tokenizer.char_to_index)}")
    # print(f"Vocabulary: {tokenizer.char_to_index}")

    # Test the tokenizer with a sample text
    text = "Hello, my dog is cute"
    tokens = tokenizer.encode(text)
    decoded_text = tokenizer.decode(tokens)
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded_text}")