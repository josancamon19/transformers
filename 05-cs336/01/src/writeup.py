print(f'chr(0): "{chr(0)}"')
print(f"chr(0): {chr(0).__repr__()}")

# ----

print("this is a test" + chr(0) + "string")

# unicode standard 150k vocab
# unicode encoding, takes a character into bytes, utf-8 (dominant on internet)

print("--------")
test_string = "hello! こんにちは!"
utf8_encoded = test_string.encode("utf-8")
print(utf8_encoded)
print(type(utf8_encoded))
# Get the byte values for the encoded string (integers from 0 to 255).
list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
# One byte does not necessarily correspond to one Unicode character!
print(len(test_string))
print(len(utf8_encoded))
print(utf8_encoded.decode("utf-8"))

# vocabulary size of 256, bytes can have 256 possible values
# 1 byte = 8 bits [0 or 1 at each state] 2^8, 256

# why utf-8 vs 16/32
# -- utf instead of unicodes, limited vocab size, and composability, utf-16/32 have 2/4x the size, most internet


def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])


# print('"にち".encode("utf-8"):', "にち".encode("utf-8"))
# print("bytes([b]): ", bytes(["にち".encode("utf-8")[0]]))
# output = decode_utf8_bytes_to_str_wrong("にち".encode("utf-8"))  # noqa: UP012
# print("decode_utf8_bytes_to_str_wrong:", output)

# when a character requires more than 1 byte it fails.

# C, 2 byte sequence that doesn't decode to any unicode char ??
# -- any 2 bytes that go independently?

# ----

print("-------")
# BPE, as a midpoint between byte level tokenizer and word level tokenizer.

# import regex as re  # noqa: E402
# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# re.findall(PAT, "some text that i'll pre-tokenize")


def bpe_example(merges: int = 5):
    from collections import Counter, defaultdict  # noqa: E402

    corpus = """
    low low low low low
    lower lower widest widest widest
    newest newest newest newest newest newest
    """.replace("\n", " ")
    pretokenization = [w.strip() for w in corpus.split(" ") if w.strip()]
    count = Counter(pretokenization)
    print("pretokenized counter:", count)
    vocabulary = set([bytes([b]) for w in set(pretokenization) for b in w.encode("utf-8")])
    print("vocabulary:", vocabulary)
    print()

    def merge():
        pretokenization_bytes = []
        for w in pretokenization:
            w = [bytes([b]) for b in w.encode("utf-8")]
            word = []
            i, j = 0, 1
            while j <= len(w):
                subw = b"".join(w[i:j])
                if subw not in vocabulary:
                    word.append(b"".join(w[i : j - 1]))
                    i = j - 1
                elif j == len(w):
                    word.append(b"".join(w[i:j]))
                    j += 1
                else:
                    j += 1
            pretokenization_bytes.append(word)
        print("pretokenization_bytes:", pretokenization_bytes, "...")
        print("pretokenization_bytes[0][0]:", pretokenization_bytes[0][0], type(pretokenization_bytes[0][0]))

        common_pairs = defaultdict(int)
        for w in pretokenization_bytes:
            for i in range(len(w)):
                if i == len(w) - 1:
                    break
                j = i + 1
                if w[i] + w[j] not in vocabulary:
                    common_pairs[w[i] + w[j]] += 1

        if not common_pairs:
            return True
        print("common_pairs:", common_pairs)
        max_count = max(common_pairs.values())
        max_pairs = [pair for pair, count in common_pairs.items() if count == max_count]
        print(f"Most common pairs (count={max_count}): {max_pairs}")
        print(f"New Vocab Word: {max(max_pairs)}")
        vocabulary.add(max(max_pairs))
        print("updated vocabulary:", vocabulary)
        print("-")

    for i in range(merges):
        if merge():
            print(f"nothing else to merge at {i} iter")
            break


bpe_example(6)
