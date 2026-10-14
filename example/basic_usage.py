"""Read one Lance dataset in training batches. Run: uv run python example/basic_usage.py"""

from _common import DATA_DIR, show_batch

from lance_dataset_loader import LanceSource, MixtureDataset


def main():
    source = LanceSource(str(DATA_DIR / "coding.lance"), version=1)
    dataset = MixtureDataset({"coding": source}, {"coding": 1.0}, global_batch_size=4)
    print("Source snapshot:", source.snapshot)
    with dataset.iter_batches() as batches:
        for _ in range(2):
            batch = next(batches)
            show_batch(batch)
            # In a real trainer: forward/backward + optimizer.step() happen here.
            dataset.commit(batch)
    print("Next training step:", dataset.next_step)


if __name__ == "__main__":
    main()
