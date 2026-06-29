from collections import defaultdict
import json

from base import BaseTokenizer

class BPETokenizer(BaseTokenizer):
    
    def __init__(self):
        super().__init__()
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
    
    def build_vocab(
        self,
        text: str,
        vocab_size: int,
    ) -> None:
        """Build vocabulary and mappings from the input text."""
        assert vocab_size >= 256, "vocab_size must be at least 256"

        # Start from raw UTF-8 bytes
        tokens = list(text.encode("utf-8"))
        num_merges = vocab_size - 256

        # Initialise base vocabulary (single bytes)
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

        ids = list(tokens)  # working copy

        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break  # nothing left to merge

            # Pick the most frequent pair (ties broken by pair value for determinism)
            best_pair = max(stats, key=lambda p: (stats[p], p))
            new_id = 256 + i

            ids = self._merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

    def encode(self, text: str) -> list[int]:
        """Convert text into a list of token ids."""
        ids = list(text.encode("utf-8"))

        # Apply merges in the order they were learned
        for pair, new_id in self.merges.items():
            ids = self._merge(ids, pair, new_id)

        return ids

    def decode(self, tokens: list[int]) -> str:
        """Convert token ids back into text."""
        byte_string = b"".join(self.vocab[t] for t in tokens)
        return byte_string.decode("utf-8", errors="replace")

    def save_vocab(self, path: str) -> None:
        """Save the vocabulary to a file."""
        data = {
            "merges": [[p0, p1, new_id] for (p0, p1), new_id in self.merges.items()],
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str) -> None:
        """Load the vocabulary from a file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.merges = {
            (p0, p1): new_id for p0, p1, new_id in data["merges"]
        }
        self.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        
    # -------- Helpers --------
    def _get_stats(self, ids: list[int]) -> dict[tuple[int, int], int]:
        """Count consecutive pair frequencies."""
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts
    
    def _merge(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace every occurrence of *pair* in ids with new_id."""
        result = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

if __name__ == "__main__":
    # Build the vocabulary
    with open("data/wikitext-103/wiki.valid.tokens", "r") as file:
        text = file.read()

    tokenizer = BPETokenizer()
    # tokenizer.build_vocab(text, vocab_size=1000)
    # tokenizer.save_vocab("vocabs/bpe_vocab.json")
    tokenizer.load_vocab("vocabs/bpe_vocab.json")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")

    # Test the tokenizer with a sample text
    text = "Hello, my dog is cute"
    tokens = tokenizer.encode(text)
    decoded_text = tokenizer.decode(tokens)
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded_text}")