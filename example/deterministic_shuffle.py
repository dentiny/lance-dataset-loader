"""The same seed and snapshots replay the same source/row order."""

import hashlib
import json

from _common import DATA_DIR

from lance_dataset_loader import LanceSource, MixtureDataset


def sample_order(seed):
    sources = {
        name: LanceSource(str(DATA_DIR / f"{name}.lance"), version=1) for name in ("coding", "math")
    }
    dataset = MixtureDataset(sources, {"coding": 0.5, "math": 0.5}, global_batch_size=8, seed=seed)
    order = []
    with dataset.iter_batches() as batches:
        for _ in range(3):
            batch = next(batches)
            order.extend((sample.source_id, sample.row_index) for sample in batch.samples)
            dataset.commit(batch)
    return order


def main():
    for seed in (42, 42, 99):
        order = sample_order(seed)
        fingerprint = hashlib.sha256(json.dumps(order).encode()).hexdigest()[:16]
        print(f"seed={seed}: fingerprint={fingerprint}, first eight={order[:8]}")
    print("The two seed=42 runs match; seed=99 changes the order.")


if __name__ == "__main__":
    main()
