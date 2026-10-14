"""Mix three source streams by probability, without rewriting their files."""

from collections import Counter

from _common import DATA_DIR, show_batch

from lance_dataset_loader import LanceSource, MixtureDataset


def main():
    sources = {
        name: LanceSource(str(DATA_DIR / f"{name}.lance"), version=1)
        for name in ("coding", "math", "general")
    }
    weights = {"coding": 0.2, "math": 0.3, "general": 0.5}
    dataset = MixtureDataset(sources, weights, global_batch_size=8, seed=42)
    counts = Counter()
    with dataset.iter_batches() as batches:
        for step in range(100):
            batch = next(batches)
            if step == 0:
                show_batch(batch)
            counts.update(sample.source_id for sample in batch.samples)
            dataset.commit(batch)
    print("Target sequence proportions:", weights)
    print("Observed over 800 occurrences:", {name: counts[name] / 800 for name in weights})
    print("These are sampling probabilities, not exact quotas; sources repeat by default.")


if __name__ == "__main__":
    main()
