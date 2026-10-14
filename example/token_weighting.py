"""Compare 50/50 sequence mixing with 50/50 fixed-length token mixing."""

from collections import Counter

from _common import DATA_DIR

from lance_dataset_loader import LanceSource, MixtureDataset


def main():
    lengths = {"coding": 8, "math": 4}
    sources = {
        name: LanceSource(
            str(DATA_DIR / f"{name}.lance"),
            version=1,
            token_column="input_ids",
            tokens_per_sequence=length,
        )
        for name, length in lengths.items()
    }
    for mode in ("sequences", "tokens"):
        dataset = MixtureDataset(
            sources, {"coding": 0.5, "math": 0.5}, global_batch_size=8, weight_by=mode
        )
        counts = Counter()
        with dataset.iter_batches() as batches:
            for _ in range(100):
                batch = next(batches)
                counts.update(sample.source_id for sample in batch.samples)
                dataset.commit(batch)
        tokens = {name: counts[name] * length for name, length in lengths.items()}
        proportions = {name: count / sum(tokens.values()) for name, count in tokens.items()}
        print(f"{mode}: sequence counts={dict(counts)}, token proportions={proportions}")
    print("Fixed total lengths only: padding/masked loss tokens are not subtracted.")


if __name__ == "__main__":
    main()
