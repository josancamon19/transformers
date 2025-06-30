from collections.abc import Iterable, Iterator
import regex as re
import json


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # print(
        #     f"[Tokenizer.__init__] vocab_size: {len(vocab)}, merges_size: {len(merges)}, special_tokens: {special_tokens}",
        # )
        special_tokens = special_tokens or []
        special_tokens = sorted(special_tokens, key=len, reverse=True)

        self.vocab_reversed = {v: k for k, v in vocab.items()}

        for st in special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes in self.vocab_reversed:
                continue

            vocab[len(vocab)] = st_bytes
            self.vocab_reversed[st_bytes] = len(vocab)

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        # --
        self.split_special_tokens = "|".join(re.escape(token) for token in self.special_tokens)
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def encode(self, input_text: str) -> list[int]:
        # print("[Tokenizer.encode] input_text:", input_text)
        if self.special_tokens:
            strings = re.split(self.split_special_tokens, input_text)
            special_tokens_sep = [m.group().encode("utf-8") for m in re.finditer(self.split_special_tokens, input_text)]
            # print("[Tokenizer.encode] special_tokens_sep:", special_tokens_sep)
        else:
            strings, special_tokens_sep = [input_text], []

        pretokens = set()
        pretokenized_strings = []
        for si, string in enumerate(strings):
            pretokenized = []
            for match in self.PAT.finditer(string):
                pretoken_bytes = match.group().encode("utf-8")
                pretokens.add(pretoken_bytes)
                pretokenized.append(pretoken_bytes)
            pretokenized_strings.append(pretokenized)

        pretokens_map = {}
        for pretoken in pretokens:
            pretoken_bytes = [bytes([b]) for b in pretoken]
            for merge in self.merges:
                i = 0
                while i < len(pretoken_bytes) - 1:
                    if pretoken_bytes[i] == merge[0] and pretoken_bytes[i + 1] == merge[1]:
                        pretoken_bytes = pretoken_bytes[:i] + [merge[0] + merge[1]] + pretoken_bytes[i + 2 :]
                    else:
                        i += 1

            pretokens_map[pretoken] = [self.vocab_reversed[pb] for pb in pretoken_bytes]

        # print(pretokens_map)
        tokenized = []
        for si, string in enumerate(pretokenized_strings):
            for pretoken_bytes in string:
                tokenized.extend(pretokens_map[pretoken_bytes])

            if si < len(strings) - 1:
                tokenized.append(self.vocab_reversed[special_tokens_sep[si]])

        # print("[Tokenizer.encode] tokenized:", tokenized)
        # print("[Tokenizer.encode] tokenized.pre:", [self.vocab[i] for i in tokenized])
        return tokenized

    def encode_(self, input_text: str) -> list[int]:
        if self.special_tokens:
            strings = re.split(self.split_special_tokens, input_text)
            special_tokens_sep = [m.group().encode("utf-8") for m in re.finditer(self.split_special_tokens, input_text)]
        else:
            strings, special_tokens_sep = [input_text], []

        tokenized = []
        for si, string in enumerate(strings):
            for match in self.PAT.finditer(string):
                pretoken_bytes = match.group().encode("utf-8")

                tokens = [bytes([b]) for b in pretoken_bytes]
                for merge in self.merges:
                    i = 0
                    while i < len(tokens) - 1:
                        if tokens[i] == merge[0] and tokens[i + 1] == merge[1]:
                            tokens = tokens[:i] + [merge[0] + merge[1]] + tokens[i + 2 :]
                        else:
                            i += 1

                for token in tokens:
                    tokenized.append(self.vocab_reversed[token])

            if si < len(strings) - 1:
                tokenized.append(self.vocab_reversed[special_tokens_sep[si]])

        return tokenized

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            token_ids = self.encode(chunk)
            yield from token_ids

    def decode(self, ids: list[int]):
        # print("[Tokenizer.decode] ids:", ids)
        decoded_bytes = b"".join([self.vocab[_id] for _id in ids])
        decoded = decoded_bytes.decode("utf-8")
        # print("[Tokenizer.decode] decoded:", decoded)
        return decoded

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)

        merges = []

        with open(merges_filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and " " in line:
                    parts = line.split(" ")
                    assert len(parts) == 2
                    # Convert merge pairs to bytes
                    merge1 = parts[0].encode("utf-8")
                    merge2 = parts[1].encode("utf-8")
                    merges.append((merge1, merge2))
        return cls(vocab, merges, special_tokens)


# TODO: parallelize encode
# TODO: check difference with repeated characters on test_train_bpe
# TODO: train on tinystories dataset, vocabsize 10k (store to disk)
# TODO: profile the code
# TODO: parallelize training

# TODO: train on openwebtext dataset.
# TODO: 2.7 experiments.
