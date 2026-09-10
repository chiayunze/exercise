from src.generated.common.corpus import full_text


def count_freq(input: list[int]) -> list[tuple[tuple[int, int], int]]:
    counts = {}
    for a, b in zip(input, input[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1

    ordered_counts = sorted(
        counts.items(),
        key=lambda x: (x[1], x[0]),  # sort on counts then on the pair for tie-breaking
        reverse=True,  # greatest first
    )
    return ordered_counts


def replace_pairs(
    input: list[int], pair: tuple[int, int], replacement: int
) -> list[int]:
    result = []
    idx = 0
    while idx < len(input):
        if idx + 1 < len(input) and (input[idx], input[idx + 1]) == pair:
            result.append(replacement)  # append new token index
            idx += 2
        else:
            result.append(input[idx])  # append existing 0-255 token
            idx += 1
    return result


def to_bytes(input: str) -> list[int]:
    return list(input.encode("utf-8"))


def from_bytes(input: list[int]) -> str:
    text_bytes = b"".join(bytes([i]) for i in input)
    return text_bytes.decode("utf-8", errors="replace")


class BPETokenizer:
    def __init__(self):
        self.merges = []

    def train(self, input: list[int], num_merges: int):
        bytelist = input
        for merge_idx in range(num_merges):
            frequencies = count_freq(bytelist)
            pair, _ = frequencies[0]
            new_token = 256 + merge_idx
            bytelist = replace_pairs(bytelist, pair, new_token)
            self.merges.append((pair, new_token))

    def encode(self, input: list[int]) -> list[int]:
        """
        Compresses inputs into a shorter token representation.
        """
        result = input
        for pair, token in self.merges:
            result = replace_pairs(result, pair, token)
        return result

    def decode(self, input: list[int]) -> list[int]:
        """
        Reverses the encode operation.
        """
        result = input
        for pair, token in reversed(self.merges):
            result2 = []
            for i in result:
                if i == token:
                    result2.append(pair[0])
                    result2.append(pair[1])
                else:
                    result2.append(i)
            result = result2
        return result


if __name__ == "__main__":
    tokenizer = BPETokenizer()
    input_text = full_text()
    input_bytes = to_bytes(full_text())
    tokenizer.train(input_bytes, 20)
    encoded = tokenizer.encode(input_bytes)
    decoded = tokenizer.decode(encoded)
    output_text = from_bytes(decoded)
    print(f"input {len(input_bytes)} encoded {len(encoded)} decoded {len(decoded)}")
    print(f"bytes eq {input_bytes == decoded} text eq {input_text == output_text}")
