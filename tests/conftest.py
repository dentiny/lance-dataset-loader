import threading

import pytest

from lance_dataset_loader import SourceSnapshot


class MemorySource:
    def __init__(self, name="data", size=101, version=1, tokens=None, fail=False):
        self.snapshot = SourceSnapshot(
            f"memory://{name}", version, size, tokens_per_sequence=tokens
        )
        self.rows = [{"id": i, "source": name} for i in range(size)]
        self.fail = fail
        self.read_count = 0
        self.lock = threading.Lock()
        self.prefetched = threading.Event()

    def take(self, indices):
        if self.fail:
            raise OSError("storage unavailable")
        with self.lock:
            self.read_count += len(indices)
            if self.read_count >= 24:
                self.prefetched.set()
        return [self.rows[i] for i in indices]


@pytest.fixture
def sources():
    return {"coding": MemorySource("coding", 17), "math": MemorySource("math", 23)}
