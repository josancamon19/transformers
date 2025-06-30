# ====================================================
# INITIAL ALGORITHM, doesn't satisfy speed needs, very
# @DEPRECATED
# suboptimal, 5.2 seconds compared to test 1 in test_train_bpe.py, benchmark 1.5 secs
# train_tokenizer.py, does it in 0.18 seconds
# but this one apparently works lol, at least on test 3, there's 1 difference 'na' 'nn' in the full diff
# when using the first algorith, no idea why.
# ====================================================


from collections import defaultdict
from typing import Iterable, Iterator  # noqa: UP035
import regex as re
from src.shared import timeit, init_vocabulary, find_chunk_boundaries


# @timeit
def merge_step(
    vocab: set,
    vocab_dict: dict,
    pretokenized: list[list[list[bytes]]],
) -> tuple[bytes, bytes]:
    """
    return merge step made,
    pretokenized: list of sentences, each one split with the pretokenized
    """

    # @timeit
    def count_bigrams():  # 99% of the function time
        common = defaultdict(int)
        for sample in pretokenized:
            for word in sample:
                for i in range(len(word) - 1):
                    common[(word[i], word[i + 1])] += 1
        return common

    common = count_bigrams()

    if not common:
        return None

    # choose by value, then by lex
    merge = max(common.items(), key=lambda x: (x[1], x[0]))
    print("merge_step:", merge)
    merge = merge[0]
    vocab_dict[len(vocab)] = merge[0] + merge[1]
    vocab.add(merge[0] + merge[1])
    return merge


# @timeit
def update_pretokenized_with_merge(
    _merge: tuple[bytes, bytes],
    pretokenized: list[list[list[bytes]]],
):
    merge_first, merge_second = _merge
    merge = merge_first + merge_second

    updated_pretokenized = []
    for sentence in pretokenized:
        new_sentence = []
        for word in sentence:
            if len(word) == 1:
                # this stupid thing, reduced 1 whole second!!
                new_sentence.append(word)
                continue

            new_word = []
            i = 0
            while i < len(word):
                # 0.3 seconds reduced, instead of (word[i] + word[j]) == merge, creating a new byte
                if i + 1 < len(word) and (word[i] == merge_first and word[i + 1] == merge_second):
                    new_word.append(merge)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_sentence.append(new_word)
            # print(merged_word)
        updated_pretokenized.append(new_sentence)

    return updated_pretokenized


# @timeit
def pretokenize_all(
    chunk: str,
    special_tokens: list[str],
    encoding: bool = False,
) -> list[list[list[bytes]]]:
    if special_tokens:
        split_special_tokens = "|".join(re.escape(token) for token in special_tokens)
        chunk_sentences = re.split(split_special_tokens, chunk)

        # Keep track of the order of special tokens used for splitting, only when encoding
        special_token_order = (
            [[m.group().encode("utf-8")] for m in re.finditer(split_special_tokens, chunk)] if encoding else []
        )

    else:
        chunk_sentences = [chunk]
        special_token_order = []

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pretokenized = []
    for i, sentence in enumerate(chunk_sentences):
        sentence_pretokenized = []
        for match in re.finditer(PAT, sentence):
            if encoding:  # no char by char
                sentence_pretokenized.append(match.group().encode("utf-8"))
            else:
                word = [char.encode("utf-8") for char in match.group()]
                sentence_pretokenized.append(word)

        pretokenized.append(sentence_pretokenized)

        if encoding and i + 1 < len(chunk_sentences):
            pretokenized.append(special_token_order[i])

    return pretokenized


# @timeit
def train_tokenizer(
    input_text_file: str = "data/TinyStoriesV2-GPT4-valid.txt",
    target_vocab_size: int = 350,
    special_tokens: list[str] = ["<|endoftext|>"],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """returns: vocab, merges"""

    assert "<|endoftext|>" in special_tokens

    vocab_dict, vocab = init_vocabulary()
    for st in special_tokens:
        vocab_dict[len(vocab)] = st.encode("utf-8")
        vocab.add(st.encode("utf-8"))

    chunks = []
    with open(input_text_file, "rb") as f:
        boundaries = find_chunk_boundaries(f, 1, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
        # Run pre-tokenization on your chunk and store the counts for each pre-token

    chunk = chunks[0]
    pretokenized = pretokenize_all(chunk, special_tokens)
    merges = []
    while len(vocab) < target_vocab_size:
        merge = merge_step(vocab, vocab_dict, pretokenized)
        if not merge:
            break
        merges.append(merge)
        pretokenized = update_pretokenized_with_merge(merge, pretokenized)

    return vocab_dict, merges
