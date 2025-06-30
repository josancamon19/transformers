from collections import defaultdict
import json
import os
from line_profiler import profile
from src.shared import timeit
import regex as re
import heapq


@timeit
@profile
def initialize(text: str, special_tokens: list[str]):
    special_tokens = sorted(special_tokens, key=len, reverse=True)  # overlapping issue
    split_special_tokens = "|".join(re.escape(token) for token in special_tokens)
    strings = re.split(split_special_tokens, text)  # [:1]
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    pairs_count = defaultdict(int)
    pretokens_to_split = {}  # b"hi": [b"h", b"i"]
    pretokens_counts = defaultdict(int)
    pairs_to_pretokens = defaultdict(set)

    for string in strings:
        for match in PAT.finditer(string):
            pretoken_bytes = match.group().encode("utf-8")
            if len(pretoken_bytes) == 1:
                continue

            pretoken_split = []
            for j in range(len(pretoken_bytes)):
                pretoken_split.append(pretoken_bytes[j : j + 1])

                if j + 1 < len(pretoken_bytes):
                    pair = (pretoken_bytes[j : j + 1], pretoken_bytes[j + 1 : j + 2])
                    pairs_count[pair] += 1
                    pairs_to_pretokens[pair].add(pretoken_bytes)

            pretokens_to_split[pretoken_bytes] = pretoken_split
            pretokens_counts[pretoken_bytes] += 1

    return pairs_count, pretokens_to_split, pretokens_counts, pairs_to_pretokens


@timeit
@profile
def get_max_priority_queue(priority_queue: list):
    # Pop all items with minimum count
    min_items = []
    if not priority_queue:
        return None, None

    # Get the minimum count
    min_count = priority_queue[0][0]

    # Pop all items with the same minimum count
    while priority_queue and priority_queue[0][0] == min_count:
        min_items.append(heapq.heappop(priority_queue))

    max_item = max(min_items, key=lambda x: x[1])

    for item in min_items:
        if item != max_item:
            heapq.heappush(priority_queue, item)

    count = -max_item[0]
    return count, max_item[1]


@timeit
@profile
def update_pairs_count_after_merge(priority_queue, new_created_pairs_count, affected_pairs_count):
    na_pair = (b"n", b"a")
    # if na_pair in affected_pairs_count:
    #     print("affected_pairs_count:", affected_pairs_count[na_pair])

    # if na_pair in new_created_pairs_count:
    #     print("new_created_pairs_count:", new_created_pairs_count[na_pair])

    for pair, count in new_created_pairs_count.items():
        priority_queue.append((-count, pair))

    for i, item in enumerate(priority_queue):
        count, pair = -item[0], item[1]
        if pair in affected_pairs_count:
            new_count = -(count - affected_pairs_count[pair])
            if pair == na_pair:
                print("affected_pairs_count: b'na'", count, -new_count)
            priority_queue[i] = (new_count, pair)
    heapq.heapify(priority_queue)


@timeit
@profile
def train_tokenizer(
    input_text_file: str = "data/TinyStoriesV2-GPT4-valid.txt",
    target_vocab_size: int = 300,
    special_tokens: list[str] = ["<|endoftext|>"],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_text_file, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")

    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    vocab_set = set(vocab.values())

    pairs_count, pretokens_to_split, pretokens_counts, pairs_to_pretokens = initialize(text, special_tokens)
    # pairs_count[(b"n", b"a")] = 0
    priority_queue = [(-count, pair) for pair, count in pairs_count.items()]

    heapq.heapify(priority_queue)
    merges = []

    # print("priority_queue:", priority_queue)

    while len(vocab) < target_vocab_size:
        pcount, pair = get_max_priority_queue(priority_queue)
        pair_bytes = pair[0] + pair[1]
        # print(f"merge {len(merges) + 1}:", pcount, pair)
        merges.append(pair)
        vocab[len(vocab)] = pair_bytes
        vocab_set.add(pair_bytes)

        new_created_pairs_count = defaultdict(int)
        affected_pairs_count = defaultdict(int)

        #  ==== MERGE ====
        matching_pretokens = pairs_to_pretokens[pair]

        for pretoken in matching_pretokens:
            split = pretokens_to_split[pretoken]  # [b"h", b"i"]
            count = pretokens_counts[pretoken]

            i = 0
            updated_split = []
            while i < len(split):
                if i + 1 < len(split) and split[i] == pair[0] and split[i + 1] == pair[1]:
                    updated_split.append(pair_bytes)
                    affected_pairs_count[pair] += count
                    i += 2
                else:
                    updated_split.append(split[i])
                    i += 1

            pretokens_to_split[pretoken] = updated_split

            for i in range(len(updated_split)):
                if updated_split[i] == pair_bytes:
                    if i > 0:
                        old_pair = (updated_split[i - 1], pair[0])
                        affected_pairs_count[old_pair] += count

                        new_pair = (updated_split[i - 1], pair_bytes)
                        new_created_pairs_count[new_pair] += count

                        pairs_to_pretokens[new_pair].add(pretoken)

                    if i + 1 < len(updated_split):
                        old_pair = (pair[1], updated_split[i + 1])
                        affected_pairs_count[old_pair] += count

                        new_pair = (pair_bytes, updated_split[i + 1])
                        new_created_pairs_count[new_pair] += count

                        pairs_to_pretokens[new_pair].add(pretoken)

        if (b"n", b"a") in affected_pairs_count:
            print(f"merge {len(merges) + 1}:", pcount, pair)
            
        update_pairs_count_after_merge(
            priority_queue,
            new_created_pairs_count,
            affected_pairs_count,
        )
    json_path = "result.json"
    with open(json_path, "w") as f:
        json.dump(merges, f, indent=2, default=str)
    return vocab, merges


# data = "cs336_basics/sample_file.txt"
# size = 270
# vocab, merges1 = train_tokenizer(data, size)
# print([p1 + p2 for p1, p2 in merges1], len(vocab), len(merges1))


from tokenizers.trainers import BpeTrainer  # noqa: E402
from tokenizers.pre_tokenizers import ByteLevel  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import BPE  # noqa: E402


def use_hf_tokenizer(
    input_text_file: str = "data/TinyStoriesV2-GPT4-valid.txt",
    target_vocab_size: int = 300,
    special_tokens: list[str] = ["<|endoftext|>"],
):
    input_text_file = "tests/fixtures/corpus.en"
    print("use_hf_tokenizer: ", input_text_file)

    trainer = BpeTrainer(
        special_tokens=special_tokens,
        vocab_size=target_vocab_size,
        # initial_alphabet=[chr(i) for i in range(256)],
        show_progress=False,
        min_frequency=2,
    )
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.train([input_text_file], trainer)
    tokenizer.save("hf.json")

    with open("hf.json", "rb") as f:
        data = json.loads(f.read())["model"]
        vocab = data["vocab"]
        vocab = {v: k.replace("Ġ", " ").encode("utf-8") for k, v in vocab.items()}
        merges = data["merges"]
        merges = [(p1.replace("Ġ", " ").encode("utf-8"), p2.replace("Ġ", " ").encode("utf-8")) for p1, p2 in merges]
        # print(vocab)
        print("merges:", len(merges))
        print("vocab:", len(vocab))
        return vocab, merges


# use_hf_tokenizer()
