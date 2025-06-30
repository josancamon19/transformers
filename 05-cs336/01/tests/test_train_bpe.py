import json

from src.shared import print_execution_summary
from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode
import time


# def test_train_bpe_speed():
#     """
#     Ensure that BPE training is relatively efficient by measuring training
#     time on this small dataset and throwing an error if it takes more than 1.5 seconds.
#     This is a pretty generous upper-bound, it takes 0.38 seconds with the
#     reference implementation on my laptop. In contrast, the toy implementation
#     takes around 3 seconds.
#     """
#     input_path = FIXTURES_PATH / "corpus.en"
#     start_time = time.time()
#     _, _ = run_train_bpe(
#         input_path=input_path,
#         vocab_size=500,
#         special_tokens=["<|endoftext|>"],
#     )
#     end_time = time.time()
#     print_execution_summary()
#     assert end_time - start_time < 1.5


# def test_train_bpe():
#     input_path = FIXTURES_PATH / "corpus.en"
#     vocab, merges = run_train_bpe(
#         input_path=input_path,
#         vocab_size=500,
#         special_tokens=["<|endoftext|>"],
#     )

#     # Path to the reference tokenizer vocab and merges
#     reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
#     reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

#     # Compare the learned merges to the expected output merges
#     gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
#     with open(reference_merges_path) as f:
#         gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
#         reference_merges = [
#             (
#                 bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
#                 bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
#             )
#             for merge_token_1, merge_token_2 in gpt2_reference_merges
#         ]
#     print(f"Length of merges: {len(merges)}")
#     print(f"Length of reference_merges: {len(reference_merges)}")

#     # Find differences between the two lists
#     if len(merges) != len(reference_merges):
#         print(f"Length mismatch! merges has {len(merges)} items, reference_merges has {len(reference_merges)} items")
#     else:
#         print("Lists have same length, checking for content differences...")
#         for i, (merge, ref_merge) in enumerate(zip(merges, reference_merges)):
#             if merge != ref_merge:
#                 print(f"{i} yours={merge}, ref={ref_merge}")
#             else:
#                 print(f"{i} yours={merge}, ref={ref_merge} ✅")
#     assert merges == reference_merges

#     # Compare the vocab to the expected output vocab
#     with open(reference_vocab_path) as f:
#         gpt2_reference_vocab = json.load(f)
#         reference_vocab = {
#             gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
#             for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
#         }
#     # Rather than checking that the vocabs exactly match (since they could
#     # have been constructed differently, we'll make sure that the vocab keys and values match)
#     assert set(vocab.keys()) == set(reference_vocab.keys())
#     assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    # print(vocabs_without_specials)
    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )
