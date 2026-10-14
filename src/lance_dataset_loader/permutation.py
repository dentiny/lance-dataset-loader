"""A keyed Feistel permutation with cycle walking; constant space per lookup."""

import hashlib


def permute(index: int, size: int, key: bytes) -> int:
    if size <= 0 or not 0 <= index < size:
        raise ValueError("index must be within the permutation")
    if size == 1:
        return 0
    half_bits = ((size - 1).bit_length() + 1) // 2
    mask = (1 << half_bits) - 1
    width = (half_bits + 7) // 8
    value = index
    while True:
        left, right = value >> half_bits, value & mask
        for round_number in range(8):
            digest = hashlib.blake2b(
                bytes([round_number]) + right.to_bytes(width, "little"),
                key=key,
                digest_size=8,
            ).digest()
            left, right = right, left ^ (int.from_bytes(digest, "little") & mask)
        value = (left << half_bits) | right
        if value < size:
            return value
